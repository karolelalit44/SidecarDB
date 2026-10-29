import sqlglot
from sqlglot import exp

from .config import Settings
from .constants import (
    ANONYMOUS,
    BLOCKED_NODES,
    DIALECT,
    FORBIDDEN_FUNCTIONS,
    IDENTIFIER_EXTRA_CHARACTERS,
    PG_IDENTIFIER_MAX_LENGTH,
    ROOT_TYPE,
)
from .exceptions import SecurityError
from .logging_config import get_logger
from .types import SchemaGraph

logger = get_logger("guard")


def is_safe_identifier(name: str) -> bool:
    if not name or len(name) > PG_IDENTIFIER_MAX_LENGTH:
        return False
    if not (name[0].isalpha() or name[0] == "_"):
        return False
    return all(
        character.isalnum() or character in IDENTIFIER_EXTRA_CHARACTERS for character in name
    )


def quote(name: str) -> str:
    if not is_safe_identifier(name):
        raise ValueError("Unsafe identifier")
    return f'"{name}"'


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


def validate(sql: str, graph: SchemaGraph | None, settings: Settings) -> str:
    """Accept only a SELECT against catalog tables in the configured schema.

    The returned statement is safe to hand to the sandbox: read-only shape,
    known tables, no forbidden functions, and a LIMIT that bounds the result.
    """
    statement = (sql or "").strip()
    if not statement:
        raise SecurityError("Empty statement.")

    logger.debug("guard: checking a statement of %d chars", len(statement))

    try:
        tree = sqlglot.parse_one(statement, read=DIALECT)
    except Exception as exc:
        logger.debug("guard: parse failed: %s: %s", type(exc).__name__, exc)
        raise SecurityError("Statement could not be parsed.") from exc

    if not isinstance(tree, ROOT_TYPE):
        logger.debug("guard: rejected; the root node is %s", type(tree).__name__)
        raise SecurityError("Only SELECT is permitted.")

    logger.debug("guard: parsed root node %s", type(tree).__name__)

    for node in BLOCKED_NODES:
        if tree.find(node):
            logger.debug("guard: rejected; it contains a %s node", node.__name__)
            raise SecurityError(f"Blocked operation: {node.__name__}")

    for func_node in list(tree.find_all(exp.Func)) + list(tree.find_all(exp.Anonymous)):
        for name in _function_names(func_node):
            if name in FORBIDDEN_FUNCTIONS:
                logger.debug("guard: rejected; it calls the %s function", name)
                raise SecurityError(f"Blocked function: {name}")

    if graph and graph.tables:
        known = {name.lower() for name in graph.tables}
        local = {c.alias_or_name.lower() for c in tree.find_all(exp.CTE) if c.alias_or_name}
        schema = settings.db_schema.lower()
        logger.debug("guard: checking every table reference against %d catalog tables", len(known))
        for table in tree.find_all(exp.Table):
            # An unqualified name is resolved through search_path, which the
            # sandbox pins to the configured schema. A schema qualifier is
            # allowed only when it names the configured schema; a database
            # qualifier names something outside it entirely and is refused.
            if table.catalog:
                logger.debug("guard: rejected; %s names another database", table.sql())
                raise SecurityError("Statement references a table outside the configured schema.")
            if table.db and table.db.lower() != schema:
                logger.debug("guard: rejected; %s names a schema outside the catalog", table.sql())
                raise SecurityError("Statement references a table outside the configured schema.")
            name = table.name.lower()
            if name and name not in known and name not in local:
                logger.debug("guard: rejected; it reads the unknown table %s", name)
                raise SecurityError("Statement references a table outside the catalog.")

    limit = tree.find(exp.Limit)
    rewritten = False
    if limit is None:
        if not _is_aggregate(tree):
            tree = tree.limit(settings.default_limit)
            rewritten = True
            logger.debug("guard: the statement carried no limit; adding %d", settings.default_limit)
    elif isinstance(limit.expression, exp.Literal) and limit.expression.is_int:
        try:
            value = int(limit.expression.this)
        except (TypeError, ValueError):
            value = settings.default_limit
        if value > settings.max_limit:
            limit.expression.set("this", str(settings.max_limit))
            rewritten = True
            logger.debug(
                "guard: the statement asked for %d rows; clamping to %d",
                value,
                settings.max_limit,
            )

    if not rewritten:
        logger.debug("guard: statement accepted unchanged")
        return statement

    if _rewrites_meaning(tree, statement):
        logger.debug("guard: rejected; the rewrite would change the meaning of the statement")
        raise SecurityError("Statement uses an operator this validator cannot render faithfully.")

    logger.debug("guard: statement accepted; a limit was added or clamped")
    return tree.sql(dialect=DIALECT)
