from core.catalog import SchemaGraph
from core.client import SidecarDB, _detect_version
from core.config import Settings, get_settings
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
from core.plan import JoinSpec, JoinType, MeasureFunction, Plan, QueryIntent
from core.retrieval import RetrievalMode
from core.wire import HealthResponse, SearchRequest, SearchResponse

__version__ = _detect_version()

__all__ = [
    "CatalogError",
    "ConfigurationError",
    "EmbeddingError",
    "ExecutionError",
    "HealthResponse",
    "JoinSpec",
    "JoinType",
    "MeasureFunction",
    "Plan",
    "PlanningError",
    "ProviderError",
    "QueryIntent",
    "QuestionError",
    "RetrievalMode",
    "SchemaGraph",
    "SearchError",
    "SearchRequest",
    "SearchResponse",
    "SecurityError",
    "Settings",
    "SidecarDB",
    "__version__",
    "get_settings",
]
