<div align="center">

# sidecarDB

**Natural-language search for PostgreSQL.**

Ask a question in plain English. Get a written answer and the exact rows behind it —
without the model ever writing SQL.

[![PyPI](https://img.shields.io/pypi/v/sidecardb?color=2563eb)](https://pypi.org/project/sidecardb/)
[![Python](https://img.shields.io/pypi/pyversions/sidecardb)](https://pypi.org/project/sidecardb/)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-ready-336791?logo=postgresql&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-16a34a)

[Quickstart](#quickstart) · [Library](#use-as-a-library) · [Service](#run-as-a-service) · [How it works](#how-it-works)

</div>

---

```text
$ curl -s localhost:8000/api/v1/search \
    -d '{"question": "Which region had the highest revenue last quarter?"}'

{
  "answer": "EMEA led last quarter with $4.2M in revenue.",
  "records": [{ "region": "EMEA", "revenue": 4200000 }],
  "retrieved_tables": ["sales"],
  "elapsed_ms": 142.5
}
```

sidecarDB sits beside your database: no schema changes, no migrations, no data copies.
It introspects the schema, lets the model plan the search, validates the plan, and executes
it in a read-only sandbox.

## Quickstart

**1 — Install** · Python 3.12+ and PostgreSQL

```bash
pip install sidecardb
```

**2 — Configure** · a `.env` file or exported variables

```dotenv
DATABASE_URL=postgresql://nl_reader:secret@localhost:5432/sales
LLM_BASE_URL=https://api.openai.com/v1   # any OpenAI-compatible endpoint
LLM_MODEL=gpt-4.1-mini
LLM_API_KEY=sk-...
```

**3 — Ask**

```python
import asyncio
from core import SidecarDB

async def main():
    async with SidecarDB() as db:
        result = await db.search("which region had the highest revenue last quarter?")
        print(result.answer)    # plain English
        print(result.records)   # the rows

asyncio.run(main())
```

That's the whole setup. The schema is mapped automatically at startup.

## Why sidecarDB

| | |
| --- | --- |
| **No SQL from the model** | The LLM returns a constrained JSON plan. sidecarDB compiles and validates it — raw SQL never reaches your database. |
| **Schema-aware** | Tables, columns, types, comments and row estimates are read live from `pg_catalog`, then embedded for retrieval. |
| **Answers plus data** | Every call returns a human-readable sentence *and* the exact rows behind it. |
| **Library or service** | Embed it in Python in five lines, or expose it as an HTTP API or container. |
| **Read-only by construction** | SELECT-only AST guard, bound parameters, and a sandboxed read-only session. |

## Use as a library

```python
from core import SidecarDB

async with SidecarDB(
    database_url="postgresql://nl_reader:secret@localhost:5432/sales",
    llm_base_url="https://api.openai.com/v1",
    llm_model="gpt-4.1-mini",
    llm_api_key="sk-...",
) as db:
    result = await db.search("how many orders were refunded this month?")
    print(result.answer)
    for row in result.records:
        print(row)
```

Prefer environment variables? Call `SidecarDB()` and it reads `.env` automatically.
Need the mapped schema? `await db.catalog()`.

## Run as a service

```bash
sidecardb serve --port 8000
```

Or with Docker:

```bash
docker build -t sidecardb .
docker run -p 8000:8000 --env-file .env sidecardb
```

### HTTP API

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/api/v1/search` | `{"question": "..."}` → answer + rows |
| `GET` | `/api/v1/schema` | Live catalog (`?refresh=true` to re-introspect) |
| `GET` | `/health` | Liveness + database status |
| `GET` | `/docs` | Interactive OpenAPI UI |

```bash
curl -X POST http://localhost:8000/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{"question": "How many cases are under investigation?"}'
```

```json
{
  "answer": "Found 4 case(s) currently under investigation.",
  "row_count": 1,
  "records": [{ "record_count": 4 }],
  "retrieved_tables": ["cases"],
  "retrieval_mode": "embedding",
  "elapsed_ms": 142.5
}
```

## How it works

```mermaid
flowchart LR
    A[Question] --> B[Schema retrieval]
    B --> C["LLM planner<br/>JSON plan, never SQL"]
    C --> D["AST guard<br/>SELECT only"]
    D --> E[Read-only sandbox]
    E --> F[Answer + rows]
```

1. **Introspect** — map tables, columns, types and comments from `pg_catalog`.
2. **Retrieve** — rank the most relevant tables for the question, with a catalog fallback.
3. **Plan** — one LLM call produces a JSON plan; rejected plans are repaired and retried.
4. **Guard** — compile to parameterized SQL, then validate with `sqlglot` before execution.
5. **Answer** — run in a read-only sandbox and bind the result to a plain-English sentence.

## Configuration

Only the connection and model are required; everything else has a sane default.

| Variable | Required | Description |
| --- | --- | --- |
| `DATABASE_URL` | Yes | PostgreSQL connection string |
| `LLM_BASE_URL` | Yes | Any OpenAI-compatible API base URL |
| `LLM_MODEL` | Yes | Model name |
| `LLM_API_KEY` | Yes | Provider key |
| `DB_SCHEMA` | No | Schema to expose (default: `public`) |
| `EMBEDDING_PROVIDER` | No | `none` (default), `api`, or `local` |
| `API_KEYS` | No | Comma-separated keys to protect the HTTP API |

Behavioural settings live in [`src/core/constants.py`](src/core/constants.py), so results stay
consistent across databases and providers.

## Safe by default

- Questions are **never** turned into SQL. The model plans; sidecarDB compiles.
- The AST guard allows `SELECT` only, confines access to one schema, and always applies a `LIMIT`.
- Filters travel as bound parameters and run in a read-only, time-limited session.

For production, connect with a dedicated read-only role:

```bash
sidecardb setup-roles \
  --admin-dsn postgresql://postgres:pw@localhost:5432/sales \
  --role nl_reader --schema public
```

## Development

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e '.[dev]'
ruff check src && ruff format src
mypy
pytest
```

## License

MIT — see [LICENSE](LICENSE).
