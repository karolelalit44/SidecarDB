from core.exceptions import (
    CatalogError,
    ConfigurationError,
    EmbeddingError,
    ExecutionError,
    PlanningError,
    ProviderError,
    QuestionError,
    SearchError,
    SecurityError,
)
from core.logging import configure_logging, get_logger
from core.protocols import DatabaseGateway, Embedder, Planner

__all__ = [
    "CatalogError",
    "ConfigurationError",
    "DatabaseGateway",
    "Embedder",
    "EmbeddingError",
    "ExecutionError",
    "Planner",
    "PlanningError",
    "ProviderError",
    "QuestionError",
    "SearchError",
    "SecurityError",
    "SidecarDB",
    "_detect_version",
    "configure_logging",
    "get_logger",
    "run_search",
]


def __getattr__(name: str):
    if name in ("SidecarDB", "_detect_version"):
        from core.client import SidecarDB, _detect_version

        return SidecarDB if name == "SidecarDB" else _detect_version
    if name == "run_search":
        from core.pipeline import run_search

        return run_search
    raise AttributeError(f"module 'core' has no attribute {name!r}")
