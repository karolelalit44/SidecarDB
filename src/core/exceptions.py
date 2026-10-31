class SearchError(Exception):
    status_code = 400


class ConfigurationError(SearchError):
    status_code = 500


class QuestionError(SearchError):
    pass


class SecurityError(SearchError):
    status_code = 403


class CatalogError(SearchError):
    status_code = 500


class ExecutionError(SearchError):
    status_code = 502


class PlanningError(SearchError):
    status_code = 502


class ProviderError(SearchError):
    status_code = 503


class EmbeddingError(SearchError):
    status_code = 503


__all__ = [
    "CatalogError",
    "ConfigurationError",
    "EmbeddingError",
    "ExecutionError",
    "PlanningError",
    "ProviderError",
    "QuestionError",
    "SearchError",
    "SecurityError",
]
