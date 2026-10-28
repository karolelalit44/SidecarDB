from typing import Literal

from sqlglot import exp

FORBIDDEN_FUNCTIONS = frozenset(
    {
        "pg_sleep",
        "pg_read_file",
        "pg_read_binary_file",
        "pg_ls_dir",
        "pg_stat_file",
        "pg_file_read",
        "pg_logdir_ls",
        "dblink",
        "dblink_exec",
        "dblink_connect",
        "query_to_xml",
        "query_to_xml_and_xmlschema",
        "cursor_to_xml",
        "pg_terminate_backend",
        "pg_cancel_backend",
        "pg_reload_conf",
        "lo_import",
        "lo_export",
        "set_config",
        "pg_advisory_lock",
        "pg_advisory_lock_shared",
        "current_setting",
    }
)

COMPARISON_OPERATORS = ("=", "!=", "<>", "<", "<=", ">", ">=")
PATTERN_OPERATORS = ("LIKE", "ILIKE")
NULL_TESTS = ("IS NULL", "IS NOT NULL")
OPERATORS = (*COMPARISON_OPERATORS, *PATTERN_OPERATORS, *NULL_TESTS)

DEFAULT_COMPARISON = COMPARISON_OPERATORS[0]

EQUALITY_ALIASES = ("==",)

TYPE_CATEGORY_NUMERIC = "N"
TYPE_CATEGORY_STRING = "S"
TYPE_CATEGORY_UNKNOWN = "X"

VECTOR_CAST = "::vector"
VECTOR_TYPE = "vector"
SIMILARITY_ALIAS = "similarity"

COUNT_ALIAS = "record_count"


STATUS_OK = "ok"
STATUS_DEGRADED = "degraded"
DATABASE_UP = "connected"
DATABASE_DOWN = "unavailable"
EMBEDDING_OFF = "disabled"

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


FENCE = "```"
JSON_TAG = "json"

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
    "Raise LLM_MAX_TOKENS or turn the model's thinking mode off."
)
REASONING_EMPTY_RESPONSE = (
    "The model used its entire completion budget without returning an answer. Raise LLM_MAX_TOKENS."
)

JSON_INSTRUCTION = (
    "Reply with one JSON object and nothing else. No prose, no explanation, "
    "no markdown code fences. Start with { and end with }."
)


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

IDENTIFIER_EXTRA_CHARACTERS = frozenset("_$")
ANONYMOUS = "anonymous"
DIALECT = "postgres"
ROOT_TYPE = exp.Select


RESET = "\033[0m"
DIM = "\033[2m"

COLOURS = {
    "DEBUG": "\033[36m",
    "INFO": "\033[32m",
    "WARNING": "\033[33m",
    "ERROR": "\033[31m",
    "CRITICAL": "\033[1;31m",
}


WILDCARD = "*"

TRUNCATION_MARK = " [...]"
