# NL Database Search

Ask a PostgreSQL database a question in plain text. Get the rows back.

```bash
curl -s localhost:8000/api/v1/search \
  -H 'Content-Type: application/json' \
  -d '{"query":"how many cases are still under investigation"}' | jq
```

## Pipeline

```
question
  ↓  link             embed the question, rank catalog elements by cosine distance
  ↓  collect_values   sample small-vocabulary columns, guided by pg_stats
  ↓  plan             the model emits a QueryPlan, it is checked against the
                      catalog, and a rejection is fed back for repair
  ↓  render           plan → SQL using catalog facts only
  ↓  guard            sqlglot AST guardrail
  ↓  run_sql          read-only transaction, server timeout, app timeout, row cap
  ↓  vector_fallback  if empty, nearest rows by embedding over pgvector
  ↓  answer           the template the model wrote, or a narration over the rows
```

The pipeline is a plain async function in `app/graph.py` with exactly one retry
point: `plan` re-enters itself on a rejected plan, up to `max_plan_attempts`
times. Everything else is linear.

Sampled column values are memoised per table and dropped whenever the catalog is
re-read, so they are fetched once per schema change rather than once per request.

## Configuration

`.env` is the only source of configuration. Every field on `Settings` is
required and has no Python default, so a missing or misspelled key fails at
startup rather than silently falling back to a built-in value.

```bash
cp .env.example .env    # Windows: copy .env.example .env
```

`tests/test_config_logging.py` asserts that `.env.example` contains exactly the
keys `Settings` declares — no missing keys, no stray ones.

### Logging

`LOG_LEVEL` is `DEBUG`, `INFO`, `WARNING`, `ERROR` or `CRITICAL`. It defaults to
`DEBUG`, and output is colourised: dim timestamp, coloured level, dim logger name.
`LOG_SQL=true` prints every statement the engine executes.

`LOG_NAMESPACE` is the prefix for this service's loggers; module loggers are
built from it rather than repeating the string. `LOG_DEPENDENCY_LEVEL` caps
third-party libraries — at `DEBUG`, `httpcore` and `openai` otherwise emit
thousands of lines that bury the service's own output.

### Security posture worth changing before production

`CORS_ALLOW_ORIGINS=*` is the development default. Set it to the real origins
before exposing the service, and set `CORS_ALLOW_CREDENTIALS=true` only alongside
a specific origin.

## Rules this codebase holds to

**No keyword lists, no word lists, no stopwords, no stemming, no regex matching.**

Schema understanding comes from `pg_catalog`, not from guessing at names:

| Question | Answered by |
|---|---|
| Is this column numeric? | `pg_type.typcategory` |
| Does it hold a small vocabulary? | `pg_stats.n_distinct` and observed value length |
| Which tables are relevant? | `BAAI/bge-small-en-v1.5` embeddings, cosine distance |
| Which columns hold vectors? | `pg_depend` on extension-owned types |
| How do these tables join? | `pg_constraint` foreign keys |

`tests/test_no_hardcoding.py` enforces the bans by walking the AST of every file
in `app/`:

| Test | Catches |
|---|---|
| `test_no_hardcoded_domain_vocabulary_in_app` | literal table, column or value names, including inside SQL strings |
| `test_no_word_lists_in_app` | collections of bare lowercase words — the shape of a stopword list |
| `test_no_regex_or_lexical_matching_in_app` | `re` imports and regex calls |
| `test_no_magic_numbers_in_app` | any tunable number written inline instead of read from `.env` |
| `test_no_scenario_branching_on_the_question` | the runtime inspecting what the user asked — `question == "…"`, `"x" in question`, `question.startswith(…)` |
| `test_every_setting_is_declared_without_a_default` | a `Settings` field carrying a Python default |
| `test_runtime_constants_are_referenced_not_repeated` | the SQL operator set declared in more than one module |

The magic-number allowlist holds only positions and unit conversions — `0`, `1`,
`-1`, `2`, `1000` — and HTTP status codes. Everything else is policy and belongs
in `.env`.

Each guard self-tests, so none of them can silently rot: there are cases proving
`"x" in question` and `LIMIT = 25` are rejected, and that `len(query) <
settings.min_query_length` and `embedder.aencode([question])` are not.

### What runtime sampling replaces

Nothing in the code knows that a column called `status` holds the values
`Under Investigation` or `Closed`. Those are read from the database at request
time and shown to the planner, so the mapping cannot drift from the data.

## What the model is given, and what it returns

**Given** a pruned schema — types, keys, foreign keys, operator `COMMENT`s — plus
the observed values of small-vocabulary columns.

**Returns** a `QueryPlan`: intent, table, related tables, measure column and
function, filter column/value/operator, projection, ordering, grouping, and
optional full-text columns. The reply is parsed as JSON structurally rather than
through provider-specific structured output, so it works on any OpenAI-compatible
endpoint — including ones where tool calling is unreliable.

**Never** emits SQL. The engine renders it, which is why a hallucinated table
cannot become a query.

## Safety

Layered, because any single layer can be wrong:

1. **Plan validation** — every table and column checked against the catalog.
2. **AST guardrail** — the root must be `Select`. `COPY`, `DO`, `CALL` and `SET`
   parse to other roots and are refused; `SELECT … INTO OUTFILE` fails to parse.
   Unknown functions are checked on both `node.name` and `node.sql_name()`,
   because sqlglot reports them as `Anonymous` and reading one field lets every
   unknown function through.
3. **Identifiers** — accepted only if they are plain SQL identifiers, checked by
   character set rather than pattern. Strings and `LIKE` values escaped.
4. **Read-only transaction** — `readonly=True`, `statement_timeout`,
   `idle_in_transaction_session_timeout`, fixed `search_path`.
5. **Application timeout** and **row cap** enforced after the fetch.
6. **Error redaction** — catalog names and driver internals never reach the caller.
7. **Catch-all handler** — an unexpected exception returns the same envelope, never
   a stack trace.

## Setup

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # Windows
source .venv/bin/activate             # macOS / Linux

pip install -r requirements.txt
```

Create the database first:

```bash
psql -f artifacts/init_schema.sql
psql -f artifacts/setup_roles.sql
psql -f artifacts/seed_data.sql
```

`seed_data.sql` ends with `ANALYZE`. Without it `reltuples` and `n_distinct` are
empty, no column values are discovered at runtime, and prompts silently lose the
data they need.

Then start the server:

```bash
.\.venv\Scripts\python.exe -m uvicorn app.main:app --port 8000
```

Docker is also supported:

```bash
docker compose up -d --build
```

The image ships without `.env`, so pass one at run time:

```bash
docker run --env-file .env -p 8000:8000 nl-db-search
```

`LLM_BASE_URL` is the **base**, not the full endpoint. LangChain appends
`/chat/completions` itself, so use `https://openrouter.ai/api/v1`, not
`https://openrouter.ai/api/v1/chat/completions`.

### Enabling the vector fallback

The fallback needs populated vector columns:

```bash
.\.venv\Scripts\python.exe scripts/backfill_embeddings.py
```

Columns are discovered from the catalog, so nothing is named in the script.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Status, table count, model, embedding backend |
| `GET` | `/api/v1/schema` | Live catalog. `?refresh=true` re-introspects |
| `POST` | `/api/v1/search` | `{"query": "..."}` |

## Embeddings

`BAAI/bge-small-en-v1.5`, 384 dimensions, downloaded on first use. With
`EMBEDDING_ENABLED=false` the service still works: schema linking falls back to
sending the largest tables and the vector fallback is skipped rather than faked.

## Tests and linting

```bash
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m ruff format --check .
```

Behavioural tests run against three unrelated synthetic schemas to show nothing is
schema-specific.
