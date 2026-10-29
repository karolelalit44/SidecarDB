class SearchError(Exception):
    """Base class for every failure the engine reports to a caller."""

    status_code = 400


class ConfigurationError(SearchError):
    status_code = 500


class QueryError(SearchError):
    pass


class SecurityError(SearchError):
    status_code = 403


class SchemaError(SearchError):
    status_code = 500


class ExecutionError(SearchError):
    status_code = 502


class PlanningError(SearchError):
    status_code = 502


class ProviderError(SearchError):
    """Provider outage error."""

    status_code = 503


class EmbeddingError(SearchError):
    status_code = 503
