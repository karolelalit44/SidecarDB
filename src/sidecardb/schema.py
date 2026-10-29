from pydantic import BaseModel, Field

from .types import RetrievalMode


class SearchRequest(BaseModel):
    query: str


class SearchResponse(BaseModel):
    query: str
    answer: str
    # Present only when settings.expose_sql is on; off by default because it
    # discloses table names and query shape to the caller.
    sql: str | None = None
    row_count: int = 0
    elapsed_ms: float = 0.0
    records: list[dict] = Field(default_factory=list)
    retrieved_tables: list[str] = Field(default_factory=list)
    retrieval_mode: RetrievalMode = RetrievalMode.catalog
    request_id: str | None = None


class HealthResponse(BaseModel):
    status: str
    app: str
    version: str = ""
    database: str
    tables: int = 0
    model: str = ""
    embedding: str = ""
