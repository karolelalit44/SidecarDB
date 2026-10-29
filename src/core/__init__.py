from typing import TYPE_CHECKING, Any

from core.catalog import SchemaGraph
from core.client import SidecarDB, _detect_version
from core.config import Settings, get_settings
from core.exceptions import (
    ConfigurationError,
    EmbeddingError,
    ExecutionError,
    PlanningError,
    ProviderError,
    QueryError,
    SchemaError,
    SearchError,
    SecurityError,
)
from core.plan import MeasureFunction, QueryIntent, QueryPlan
from core.retrieval import RetrievalMode
from core.wire import HealthResponse, SearchRequest, SearchResponse

__version__ = _detect_version()

if TYPE_CHECKING:
    from fastapi import FastAPI


def create_app(sidecar_db: SidecarDB | None = None, **overrides: Any) -> "FastAPI":
    from core.server import create_app as _create_app

    if sidecar_db is None and overrides:
        sidecar_db = SidecarDB(**overrides)
    return _create_app(sidecar_db)


__all__ = [
    "ConfigurationError",
    "EmbeddingError",
    "ExecutionError",
    "HealthResponse",
    "MeasureFunction",
    "PlanningError",
    "ProviderError",
    "QueryError",
    "QueryIntent",
    "QueryPlan",
    "RetrievalMode",
    "SchemaError",
    "SchemaGraph",
    "SearchError",
    "SearchRequest",
    "SearchResponse",
    "SecurityError",
    "Settings",
    "SidecarDB",
    "__version__",
    "create_app",
    "get_settings",
]
