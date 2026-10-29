"""sidecardb: natural-language search over a PostgreSQL schema.

Library::

    from sidecardb import SidecarDB

    async with SidecarDB(database_url="postgresql://...", llm_api_key="sk-...") as db:
        answer = await db.search("how many open cases were filed this week?")

Service::

    sidecardb serve --host 0.0.0.0 --port 8000
"""

from typing import TYPE_CHECKING, Any

from .client import SidecarDB, _detect_version
from .config import Settings, get_settings
from .exceptions import (
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
from .schema import HealthResponse, SearchRequest, SearchResponse
from .types import MeasureFunction, QueryIntent, QueryPlan, RetrievalMode, SchemaGraph

__version__ = _detect_version()

if TYPE_CHECKING:
    from fastapi import FastAPI


def create_app(sidecar: SidecarDB | None = None, **overrides: Any) -> "FastAPI":
    """Build the FastAPI service; with no instance, configuration comes from
    the environment. Also usable as a uvicorn factory."""
    from .server import create_app as _create_app

    if sidecar is None and overrides:
        sidecar = SidecarDB(**overrides)
    return _create_app(sidecar)


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
