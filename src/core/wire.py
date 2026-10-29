from pydantic import BaseModel, Field

from core.retrieval import RetrievalMode

AUTHORIZATION_HEADER = "Authorization"
BEARER_PREFIX = "bearer "

STATUS_OK = "ok"
STATUS_DEGRADED = "degraded"
DATABASE_UP = "connected"
DATABASE_DOWN = "unavailable"
EMBEDDING_OFF = "disabled"


class SearchRequest(BaseModel):
    query: str


class SearchResponse(BaseModel):
    query: str
    answer: str
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


__all__ = [
    "AUTHORIZATION_HEADER",
    "BEARER_PREFIX",
    "DATABASE_DOWN",
    "DATABASE_UP",
    "EMBEDDING_OFF",
    "STATUS_DEGRADED",
    "STATUS_OK",
    "HealthResponse",
    "SearchRequest",
    "SearchResponse",
]
