import sqlglot
from sqlglot import exp

from catalog.models import SchemaGraph
from config.constants import (
    ALLOWED_FUNCTIONS,
    ANONYMOUS,
    BLOCKED_NODES,
    DEFAULT_LIMIT,
    DIALECT,
    MAX_LIMIT,
    ROOT_TYPE,
)
from core.exceptions import SecurityError
from core.logging import get_logger
from planner.models import Statement

logger = get_logger("guard.sql")


def _unallowed_function(node: exp.Func) -> str | None:
    if isinstance(node, exp.Cast):
        return None
    lowered = (node.name if isinstance(node, exp.Anonymous) else node.sql_name() or "").lower()
    if lowered and lowered != ANONYMOUS and lowered not in ALLOWED_FUNCTIONS:
        return lowered
    return None


def validate(sql: str, graph: SchemaGraph | None, schema: str) -> Statement:
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

    for func_node in tree.find_all(exp.Func):
        function = _unallowed_function(func_node)
        if function:
            logger.debug("guard: rejected; it calls the %s function", function)
            raise SecurityError(
                f"Statement calls a function outside the plan vocabulary: {function}."
            )

    known = {name.lower() for name in graph.tables} if graph and graph.tables else None
    local = {c.alias_or_name.lower() for c in tree.find_all(exp.CTE) if c.alias_or_name}
    schema_name = schema.lower() if schema else ""
    for table in tree.find_all(exp.Table):
        if table.catalog:
            logger.debug("guard: rejected; %s names another database", table.sql())
            raise SecurityError("Statement references a table outside the configured schema.")
        if schema_name and table.db and table.db.lower() != schema_name:
            logger.debug("guard: rejected; %s names a schema outside the catalog", table.sql())
            raise SecurityError("Statement references a table outside the configured schema.")
        name = table.name.lower()
        if known is not None and name and name not in known and name not in local:
            logger.debug("guard: rejected; it reads the unknown table %s", name)
            raise SecurityError("Statement references a table outside the catalog.")

    limit = tree.find(exp.Limit)
    rewritten = False
    if limit is None:
        tree = tree.limit(DEFAULT_LIMIT)
        rewritten = True
        logger.debug("guard: the statement carried no limit; adding %d", DEFAULT_LIMIT)
    elif isinstance(limit.expression, exp.Literal) and limit.expression.is_int:
        try:
            value = int(limit.expression.this)
        except (TypeError, ValueError):
            value = DEFAULT_LIMIT
        if value > MAX_LIMIT:
            limit.expression.set("this", str(MAX_LIMIT))
            rewritten = True
            logger.debug(
                "guard: the statement asked for %d rows; clamping to %d",
                value,
                MAX_LIMIT,
            )
    else:
        limit.set("expression", exp.Literal.number(DEFAULT_LIMIT))
        rewritten = True
        logger.debug("guard: the limit was not an integer literal; setting %d", DEFAULT_LIMIT)

    if not rewritten:
        logger.debug("guard: statement accepted unchanged")
        return Statement(statement)

    if tree.find(exp.NullSafeEQ):
        logger.debug("guard: rejected; the rewrite would change the meaning of the statement")
        raise SecurityError("Statement uses an operator the Guard cannot express faithfully.")

    logger.debug("guard: statement accepted; a limit was added or clamped")
    return Statement(tree.sql(dialect=DIALECT))


__all__ = ["validate"]
