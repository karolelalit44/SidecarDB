# NL Database Search

## Setup

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # Windows
source .venv/bin/activate             # macOS / Linux
pip install -r requirements.txt

#kill existing uvicorn process if running any
Stop-Process -Name uvicorn -ErrorAction SilentlyContinue

```

## cURL request 
```bash
curl -s localhost:8000/api/v1/search \
  -H 'Content-Type: application/json' \
  -d '{"query":"how many cases are still under investigation"}' | jq
```



Docker: `docker compose up -d --build`. The image ships without `.env`, so pass
one at run time with `--env-file .env`.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Status, table count, model, embedding backend |
| `GET` | `/api/v1/schema` | Live catalog. `?refresh=true` re-introspects |
| `POST` | `/api/v1/search` | `{"query": "..."}` |


