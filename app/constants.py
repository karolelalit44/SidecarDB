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

TYPE_CATEGORY_NUMERIC = "N"
TYPE_CATEGORY_STRING = "S"
TYPE_CATEGORY_UNKNOWN = "X"

VECTOR_CAST = "::vector"
SIMILARITY_ALIAS = "similarity"


STATUS_OK = "ok"
STATUS_DEGRADED = "degraded"
DATABASE_UP = "connected"
DATABASE_DOWN = "unavailable"
EMBEDDING_OFF = "disabled"

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


FENCE = "```"
JSON_TAG = "json"

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
