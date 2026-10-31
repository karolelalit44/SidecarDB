from pathlib import Path

# Support package-style submodule lookup (e.g., from sidecardb.catalog.models import ...)
__path__ = [str(Path(__file__).parent)]

from catalog.models import (
    ColumnInfo,
    ForeignKeyInfo,
    SchemaGraph,
    TableInfo,
)
from config.settings import Settings, get_settings
from core.client import SidecarDB, _detect_version
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
from planner.models import (
    JoinSpec,
    JoinType,
    MeasureFunction,
    Plan,
    QueryIntent,
    Statement,
)
from retrieval.memory import RetrievalMode
from server.schemas import HealthResponse, SearchRequest, SearchResponse
from synthesis.formatter import bind_template

__version__ = _detect_version()

__all__ = [
    "CatalogError",
    "ColumnInfo",
    "ConfigurationError",
    "EmbeddingError",
    "ExecutionError",
    "ForeignKeyInfo",
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
    "Statement",
    "TableInfo",
    "__version__",
    "bind_template",
    "get_settings",
]
