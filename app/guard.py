import sqlglot
from sqlglot import exp

from .config import settings
from .constants import (
    ANONYMOUS,
    BLOCKED_NODES,
    DIALECT,
    FORBIDDEN_FUNCTIONS,
    IDENTIFIER_EXTRA_CHARACTERS,
    ROOT_TYPE,
)
from .exceptions import SecurityError
from .types import SchemaGraph


def is_safe_identifier(name: str) -> bool:
    if not name or len(name) > settings.db_identifier_max_length:
        return False
    if not (name[0].isalpha() or name[0] == "_"):
        return False
    return all(
        character.isalnum() or character in IDENTIFIER_EXTRA_CHARACTERS for character in name
    )


def _function_names(node) -> set[str]:
    names = set()
    for value in (node.name, node.sql_name()):
        if isinstance(value, str):
            lowered = value.lower()
            if lowered and lowered != ANONYMOUS:
                names.add(lowered)
    return names


def _is_aggregate(tree: exp.Expression) -> bool:
    """Check if expression contains GROUP BY or aggregate functions."""
    return tree.find(exp.Group) is not None or any(True for _ in tree.find_all(exp.AggFunc))


def _rewrites_meaning(tree: exp.Expression, statement: str) -> bool:
    """Return True if sqlglot rewrite alters semantic meaning of the SQL."""
    produced = any(True for _ in tree.find_all(exp.NullSafeEQ))
    if not produced:
        return False
    return "IS NOT DISTINCT FROM" not in statement.upper()


def validate(sql: str, graph: SchemaGraph | None = None) -> str:
    statement = (sql or "").strip()
    if not statement:
        raise SecurityError("Empty statement.")

    try:
        tree = sqlglot.parse_one(statement, read=DIALECT)
    except Exception as exc:
        raise SecurityError("Statement could not be parsed.") from exc

    if not isinstance(tree, ROOT_TYPE):
        raise SecurityError("Only SELECT is permitted.")

    for node in BLOCKED_NODES:
        if tree.find(node):
            raise SecurityError(f"Blocked operation: {node.__name__}")

    for node in list(tree.find_all(exp.Func)) + list(tree.find_all(exp.Anonymous)):
        for name in _function_names(node):
            if name in FORBIDDEN_FUNCTIONS:
                raise SecurityError(f"Blocked function: {name}")

    if graph and graph.tables:
        known = {name.lower() for name in graph.tables}
        local = {c.alias_or_name.lower() for c in tree.find_all(exp.CTE) if c.alias_or_name}
        for table in tree.find_all(exp.Table):
            name = table.name.lower()
            if name and name not in known and name not in local:
                raise SecurityError("Statement references a table outside the catalog.")

    limit = tree.find(exp.Limit)
    rewritten = False
    if limit is None:
        if not _is_aggregate(tree):
            tree = tree.limit(settings.default_limit)
            rewritten = True
    elif isinstance(limit.expression, exp.Literal) and limit.expression.is_int:
        try:
            value = int(limit.expression.this)
        except (TypeError, ValueError):
            value = settings.default_limit
        if value > settings.max_limit:
            limit.expression.set("this", str(settings.max_limit))
            rewritten = True

    if not rewritten:
        return statement

    if _rewrites_meaning(tree, statement):
        raise SecurityError("Statement uses an operator this validator cannot render faithfully.")
    return tree.sql(dialect=DIALECT)


def quote(name: str) -> str:
    if not is_safe_identifier(name):
        raise ValueError("Unsafe identifier")
    return f'"{name}"'


def literal(value: str) -> str:
    return "'" + value.replace("\\", "\\\\").replace("'", "''") + "'"


def like_literal(value: str) -> str:
    escaped = literal(value)
    return escaped[:-1].replace("%", r"\%").replace("_", r"\_") + "'"


def vector_literal(vector) -> str:
    precision = settings.vector_literal_precision
    body = ",".join(f"{value:.{precision}f}" for value in vector)
    return f"[{body}]"
