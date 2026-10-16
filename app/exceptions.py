class SearchError(Exception):
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
    """The model endpoint could not be reached or returned a transport-level failure.

    Distinct from PlanningError so that an outage is never mistaken for a bad plan:
    retrying a planner call cannot fix a provider that is down.
    """

    status_code = 503


class EmbeddingError(SearchError):
    status_code = 503
