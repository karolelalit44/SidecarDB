"""askdb: natural-language search over a PostgreSQL schema.

Library::

    from core import AskDB

    async with AskDB(database_url="postgresql://...", llm_api_key="sk-...") as db:
        answer = await db.search("how many open cases were filed this week?")

Service::

    askdb serve --host 0.0.0.0 --port 8000
"""

from typing import TYPE_CHECKING, Any

from core.client import AskDB, _detect_version
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
from core.schema import HealthResponse, SearchRequest, SearchResponse
from core.types import MeasureFunction, QueryIntent, QueryPlan, RetrievalMode, SchemaGraph

__version__ = _detect_version()

if TYPE_CHECKING:
    from fastapi import FastAPI


def create_app(askdb: AskDB | None = None, **overrides: Any) -> "FastAPI":
    """Build the FastAPI service; with no instance, configuration comes from
    the environment. Also usable as a uvicorn factory."""
    from core.server import create_app as _create_app

    if askdb is None and overrides:
        askdb = AskDB(**overrides)
    return _create_app(askdb)


__all__ = [
    "AskDB",
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
    "__version__",
    "create_app",
    "get_settings",
]
