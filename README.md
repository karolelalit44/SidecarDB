# askdb

**Ask a PostgreSQL database a question in plain English and get the rows back — safely.**

`askdb` is a plug-and-play natural-language search engine for PostgreSQL. It introspects your
schema at runtime, lets a language model *plan* a query as a constrained JSON object (never raw
SQL), validates that plan against an AST guard and your live catalog, executes it inside a
read-only sandbox with bound parameters, and returns the rows with a plain-English answer.

It ships as both a **library** you embed and an **HTTP service** you deploy. The whole
configuration is two values.

```
question → schema retrieval → LLM planner (JSON plan) → AST guard → sandboxed execution → answer
```

## Install

```bash
pip install askdb
# optional: local embedding models (pulls torch, several GB)
pip install 'askdb[embeddings]'
```

## Use it as a library

```python
from core import AskDB

async with AskDB(
    database_url="postgresql://reader@localhost:5432/sales",
    llm_api_key="sk-...",
) as db:
    answer = await db.search("which region had the highest revenue last quarter?")
    print(answer.answer)      # plain English
    print(answer.records)     # the rows
```

Two required values; everything else has production-sane defaults. You can also pass
environment variables instead of keywords — the instance starts from the environment and an
optional `.env` file, exactly like pydantic-settings.

## Run it as a service

```bash
export DATABASE_URL=postgresql://reader@localhost:5432/sales
export LLM_API_KEY=sk-...
askdb serve --port 8000
# or: uvicorn 'core.server:create_app' --factory
```

| Method | Path                       | Purpose                                   |
| ------ | -------------------------- | ----------------------------------------- |
| GET    | `/health`                  | Liveness: pings the database, reports degraded as `503` |
| GET    | `/api/v1/schema`           | The mapped catalog (`?refresh=true` re-introspects) |
| POST   | `/api/v1/search`           | `{"query": "..."}` → answer + rows        |
| GET    | `/docs`                    | OpenAPI UI                                |

Every setting is documented in [`.env.example`](.env.example); the defaults shown there apply
when a key is absent.

## Docker

```bash
cp .env.example .env   # fill in DATABASE_URL, LLM_API_KEY, and passwords
docker compose up --build
```

The compose stack starts a pgvector PostgreSQL seeded with a demo schema **and creates a
dedicated read-only role** — no credential is committed to this repository.

## Security model

Untrusted questions never become SQL. Six independent layers stand between a question and your
data:

1. **Input validation** rejects comment markers, statement separators, and control characters.
2. **The LLM plans, it does not execute.** It returns a JSON `QueryPlan` (table, intent,
   filter, projection…), never SQL. The plan is checked against the *live catalog*; unknown
   tables and columns are rejected and fed back for repair.
3. **AST guard** (sqlglot): only `SELECT` is accepted; mutations, DDL, `COPY … TO PROGRAM`,
   `DO $$`, transactions, and a blocklist of dangerous functions (`pg_read_file`, `dblink`,
   `pg_terminate_backend`, …) are refused. Table references must live in the configured
   schema — any database-level qualifier is rejected, and a schema qualifier is only accepted
   when it names the configured schema. A `LIMIT` is always enforced.
4. **Bound parameters**: filter and search values travel as `$1` placeholders, never
   interpolated into SQL text.
5. **Sandboxed execution**: read-only session + read-only transaction + `statement_timeout` +
   `idle_in_transaction_session_timeout` + row cap.
6. **Database role**: run as a role with `SELECT` only and
   `default_transaction_read_only = on` (see `artifacts/setup_roles.sh`).

Operational hardening: optional API-key auth (`API_KEYS`), per-client rate limiting, request
body caps, request-id sanitisation, catalog names redacted from errors, and generated SQL
withheld from responses unless you opt in (`EXPOSE_SQL=true`).

> Use a dedicated read-only database role. The sandbox is defense in depth, not a substitute
> for database-level privileges.

## Production checklist

- Set `API_KEYS` to one or more secrets (`python -c "import secrets; print(secrets.token_urlsafe(32))"`).
- Connect through a **read-only role**; grant `SELECT` only.
- Set `DB_SSL=require` (or stronger) for any remote database.
- Keep `EXPOSE_SQL=false` and `REDACT_CATALOG_IN_ERRORS=true` unless callers are trusted.
- Terminate TLS and enforce global rate limits at your edge; the built-in limiter is
  per-process.
- Point `LLM_BASE_URL`/`LLM_MODEL` at a provider you trust; questions and schema summaries are
  sent to it.
- Run the container as the built-in non-root user (default) and keep `docker compose` passwords
  out of version control.

## How it works

1. **Introspect** — tables, columns, types, comments and row estimates are read from
   `pg_catalog` for one schema (`DB_SCHEMA`).
2. **Retrieve** — table/column texts are embedded (API or local model) and ranked against the
   question; on low confidence the engine falls back to catalog order and says so in the
   response (`retrieval_mode`).
3. **Plan** — one LLM call produces a JSON plan; rejected plans are retried with the rejection
   reason at a higher sampling temperature.
4. **Render & guard** — the plan becomes a parameterized `SELECT`, validated by the AST guard.
5. **Execute & answer** — rows run through the sandbox, then a model-written sentence template
   is bound to the actual values. Execution failures surface as errors (`502`), never as "no
   records found".

## Development

```bash
python -m venv .venv && .venv\Scripts\activate    # or source .venv/bin/activate
pip install -e '.[dev]'
pytest                    # unit tests, no database or model needed
ruff check src tests && ruff format src tests
mypy
```

Integration tests live behind the `integration` marker and need a real PostgreSQL; CI runs
lint, typecheck, tests, package build and a Docker build on every push.

## License

MIT — see [LICENSE](LICENSE).
