from typing import Literal

from sqlglot import exp

APP_NAME = "sidecardb"

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8000
DEFAULT_SCHEMA = "public"

LOG_LEVEL = "INFO"
LOG_DEPENDENCY_LEVEL = "WARNING"
LOG_SQL = False
LOG_LLM_PAYLOADS = False
LOG_FORMAT = "text"
LOG_COLOR = "auto"

RATE_LIMIT_ENABLED = True
RATE_LIMIT_REQUESTS = 60
RATE_LIMIT_WINDOW_SECONDS = 60.0
MAX_REQUEST_BODY_BYTES = 65536
CORS_ALLOW_ORIGINS: tuple[str, ...] = ()
CORS_ALLOW_METHODS: tuple[str, ...] = ("GET", "POST", "OPTIONS")
CORS_ALLOW_HEADERS: tuple[str, ...] = (
    "Authorization",
    "Content-Type",
    "X-API-Key",
    "X-Request-ID",
)
CORS_ALLOW_CREDENTIALS = False

POOL_MIN_SIZE = 1
POOL_MAX_SIZE = 8
DB_CONNECT_RETRIES = 3
DB_CONNECT_BACKOFF_SECONDS = 0.5
DB_MAX_INACTIVE_CONNECTION_LIFETIME = 300.0
INTROSPECTION_TIMEOUT_SECONDS = 15.0
DB_BACKOFF_FACTOR = 2

SANDBOX_TRANSACTION_READ_ONLY = True
SANDBOX_STATEMENT_TIMEOUT_SECONDS = 5.0
SANDBOX_APP_TIMEOUT_SECONDS = 15.0
SANDBOX_IDLE_TIMEOUT_MS = 10000
SANDBOX_MAX_ROWS = 5000

MIN_QUESTION_LENGTH = 2
MAX_QUESTION_LENGTH = 250
DEFAULT_LIMIT = 25
MAX_LIMIT = 200

REPAIR_CHAR_BUDGET = 800
MAX_PLAN_ATTEMPTS = 3
MAX_SCHEMA_TABLES = 30
MAX_SCHEMA_COLUMNS = 120

LLM_TEMPERATURE = 0.0
LLM_REPAIR_TEMPERATURE = 0.6
LLM_MAX_TOKENS = 2048
LLM_TIMEOUT_SECONDS = 120.0
LLM_RETRIES = 2
LLM_CONTEXT_TOKENS = 8192
LLM_CHARS_PER_TOKEN = 4

EMBEDDING_BATCH_SIZE = 32

RETRIEVAL_TOP_K = 12
RETRIEVAL_MIN_SCORE = 0.15
RETRIEVAL_SCORE_PRECISION = 6

ERROR_MESSAGE_MAX_CHARS = 200
REDACT_CATALOG_IN_ERRORS = True
EXPOSE_SQL = False
MS_PER_SECOND = 1000
ELAPSED_PRECISION = 2

NO_RECORDS_MESSAGE = "No matching records were found."
NO_ANSWER_MESSAGE = "Records were returned but no answer could be produced."
PLANNING_FAILED_MESSAGE = "No valid plan was produced for this question."

HEALTH_PING_CACHE_SECONDS = 2.0
RECOVERY_INTERVAL_SECONDS = 15.0

ROLE_STATEMENT_TIMEOUT_MS = 3000
ROLE_IDLE_IN_TRANSACTION_TIMEOUT_MS = 5000

AUTHORIZATION_HEADER = "Authorization"
BEARER_PREFIX = "bearer "

STATUS_OK = "ok"
STATUS_DEGRADED = "degraded"
DATABASE_UP = "connected"
DATABASE_DOWN = "unavailable"
EMBEDDING_OFF = "disabled"

PROTECTED_PREFIX = "/api/v1"
REQUEST_ID_HEADER = "X-Request-ID"
API_KEY_HEADER = "X-API-Key"
RETRY_AFTER_HEADER = "Retry-After"
REQUEST_ID_MAX_LENGTH = 64

DISALLOWED_SUBSTRINGS = ("--", "/*", "*/", ";", "\0")
ALLOWED_CONTROL = ("\t", "\n", "\r")
BLOCKED_CATEGORIES = ("Cc", "Cf", "Cs")

TYPE_CATEGORY_NUMERIC = "N"
TYPE_CATEGORY_UNKNOWN = "X"

COMPARISON_OPERATORS = ("=", "!=", "<>", "<", "<=", ">", ">=")
PATTERN_OPERATORS = ("LIKE", "ILIKE")
NULL_TESTS = ("IS NULL", "IS NOT NULL")
OPERATORS = (*COMPARISON_OPERATORS, *PATTERN_OPERATORS, *NULL_TESTS)
DEFAULT_COMPARISON = COMPARISON_OPERATORS[0]
COUNT_ALIAS = "record_count"
WILDCARD = "*"

INTENT_COUNT = "count"
INTENT_AGGREGATE = "aggregate"
INTENT_RETRIEVE = "retrieve"
INTENT_DISTINCT = "distinct"

SCALAR_INTENTS = frozenset(("count", "aggregate"))
MEASURED_INTENTS = frozenset(("aggregate",))

COMPARISON_TEXT = ", ".join(repr(op) for op in COMPARISON_OPERATORS)
PATTERN_TEXT = ", ".join(repr(op) for op in PATTERN_OPERATORS)
NULL_TEST_TEXT = ", ".join(repr(op) for op in NULL_TESTS)
OPERATOR_TEXT = (
    f"a comparison ({COMPARISON_TEXT}), a pattern match ({PATTERN_TEXT}), "
    f"or a null test ({NULL_TEST_TEXT})"
)
TRUNCATION_MARK = " [...]"

ANONYMOUS = "anonymous"
DIALECT = "postgres"
ROOT_TYPE = exp.Select
ALLOWED_FUNCTIONS = frozenset(("count", "sum", "avg", "min", "max"))
BLOCKED_NODES = (
    exp.Insert,
    exp.Update,
    exp.Delete,
    exp.Drop,
    exp.Alter,
    exp.Create,
    exp.TruncateTable,
    exp.Command,
    exp.Grant,
    exp.Revoke,
    exp.Transaction,
    exp.Commit,
    exp.Rollback,
    exp.Set,
    exp.Use,
    exp.Merge,
    exp.Copy,
    exp.Pragma,
    exp.Into,
)

PG_IDENTIFIER_MAX_LENGTH = 63
IDENTIFIER_EXTRA_CHARACTERS = frozenset("_$")

REDACTED_BY_SQLSTATE = {
    "42501": "the database denied the statement access to an object",
    "57014": "the statement exceeded the database's execution budget",
}
REDACTED_BY_SQLSTATE_CLASS = {
    "42": "the statement named an object the database does not provide",
}

PRELOAD_COMMAND = (
    'python -c "from sentence_transformers import SentenceTransformer; '
    "SentenceTransformer('{model}')\""
)

JSON_INSTRUCTION = (
    "Reply with one JSON object and nothing else. No prose, no explanation, "
    "no markdown code fences. Start with { and end with }."
)
METADATA_SOURCES = ("response_metadata", "additional_kwargs")
FINISH_REASON_FIELD = "finish_reason"
FINISH_REASON_LENGTH = "length"
REASONING_FIELD = "reasoning_content"
USAGE_FIELD = "usage_metadata"
SERVED_MODEL_FIELD = "model_name"
COMPLETION_TOKENS_FIELD = "output_tokens"
PROMPT_TOKENS_FIELD = "input_tokens"
REASONING_BUDGET_WARNING = (
    "The model spent its entire completion budget reasoning and returned no answer. "
    "Raise the completion budget or turn the model's thinking mode off."
)
REASONING_EMPTY_RESPONSE = (
    "The model used its entire completion budget without returning an answer. "
    "Raise the completion budget."
)

PROVIDER_API: Literal["api"] = "api"
PROVIDER_LOCAL: Literal["local"] = "local"
PROVIDER_NONE: Literal["none"] = "none"

COLUMNS_SQL = """SELECT c.relname AS table_name,
       a.attname AS name,
       format_type(a.atttypid, a.atttypmod) AS data_type,
       base.typcategory AS type_category,
       NOT a.attnotnull AS nullable,
       col_description(c.oid, a.attnum) AS description
FROM pg_attribute a
JOIN pg_class c ON c.oid = a.attrelid
JOIN pg_namespace n ON n.oid = c.relnamespace
JOIN LATERAL (
    SELECT bt.typcategory
    FROM pg_type rt
    JOIN pg_type bt ON bt.oid = COALESCE(NULLIF(rt.typbasetype, 0), rt.oid)
    WHERE rt.oid = a.atttypid
) base ON true
WHERE n.nspname = $1
  AND c.relkind IN ('r', 'p', 'v', 'm', 'f')
  AND a.attnum > 0
  AND NOT a.attisdropped
ORDER BY c.relname, a.attnum"""

TABLES_SQL = """SELECT c.relname AS name,
       obj_description(c.oid, 'pg_class') AS description,
       c.reltuples AS row_estimate
FROM pg_class c
JOIN pg_namespace n ON n.oid = c.relnamespace
WHERE n.nspname = $1 AND c.relkind IN ('r', 'p', 'v', 'm', 'f')
ORDER BY c.relname"""
