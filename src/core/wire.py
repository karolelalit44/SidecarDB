from pydantic import BaseModel, Field

from core.retrieval import RetrievalMode


class SearchRequest(BaseModel):
    question: str


class SearchResponse(BaseModel):
    question: str
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
    "HealthResponse",
    "SearchRequest",
    "SearchResponse",
]
