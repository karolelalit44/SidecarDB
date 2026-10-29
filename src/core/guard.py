from typing import TYPE_CHECKING

import sqlglot
from sqlglot import exp

from core.constants import (
    ALLOWED_FUNCTIONS,
    ANONYMOUS,
    BLOCKED_NODES,
    COUNT_ALIAS,
    DEFAULT_LIMIT,
    DIALECT,
    MAX_LIMIT,
    MEASURED_INTENTS,
    NULL_TESTS,
    OPERATORS,
    PATTERN_OPERATORS,
    ROOT_TYPE,
    WILDCARD,
)
from core.exceptions import SecurityError
from core.plan import (
    MeasureFunction,
    QueryIntent,
    Statement,
)
from utils.logging_config import get_logger
from utils.sql import is_safe_identifier, quote_identifier

if TYPE_CHECKING:
    from core.catalog import SchemaGraph
    from core.plan import Plan

logger = get_logger("guard")


def quote(name: str) -> str:
    if not is_safe_identifier(name):
        logger.debug("guard: refusing to compose the identifier %r", name)
        raise SecurityError("The Guard cannot write this identifier safely.")
    return quote_identifier(name)


def _unallowed_function(node: exp.Func) -> str | None:
    if isinstance(node, exp.Cast):
        return None
    lowered = (node.name if isinstance(node, exp.Anonymous) else node.sql_name() or "").lower()
    if lowered and lowered != ANONYMOUS and lowered not in ALLOWED_FUNCTIONS:
        return lowered
    return None


def like_pattern(value: str) -> str:
    return value.replace("\\", "\\\\").replace("%", r"\%").replace("_", r"\_")


def statement_for(plan: "Plan", graph: "SchemaGraph", schema: str) -> Statement:
    sql, params = _compose(plan, graph)
    validated = validate(sql, graph, schema)
    return Statement(sql=validated.sql, params=params)


def _compose(plan: "Plan", graph: "SchemaGraph") -> tuple[str, tuple[str, ...]]:
    binder = _Binder()
    group = f" GROUP BY {', '.join(quote(c) for c in plan.group_by)}" if plan.group_by else ""
    order = ""
    if plan.order_by and plan.intent is not QueryIntent.count:
        direction = "DESC" if plan.descending else "ASC"
        order = f" ORDER BY {quote(plan.order_by)} {direction}"
    sql = f"SELECT {_projection(plan, graph)} FROM {quote(plan.table)}"
    sql += _where(plan, binder, graph) + group + order
    if plan.intent is not QueryIntent.count and not (
        plan.intent in MEASURED_INTENTS and not plan.group_by
    ):
        sql += f" LIMIT {DEFAULT_LIMIT}"
    return sql, tuple(binder.params)


def _projection(plan: "Plan", graph: "SchemaGraph") -> str:
    if plan.intent is QueryIntent.count:
        return f"COUNT(*) AS {COUNT_ALIAS}"

    if plan.intent in MEASURED_INTENTS:
        assert plan.measure_function is not None
        measure_sql = f"COUNT(*) AS {COUNT_ALIAS}"
        if plan.measure_function is not MeasureFunction.count:
            assert plan.measure_column is not None
            alias = f"{plan.measure_function.value}_of_{plan.measure_column}"
            function = plan.measure_function.value.upper()
            measure_sql = f"{function}({quote(plan.measure_column)}) AS {quote(alias)}"
        if plan.group_by:
            group_cols = ", ".join(quote(column) for column in plan.group_by)
            return f"{group_cols}, {measure_sql}"
        return measure_sql

    if plan.projection:
        prefix = "DISTINCT " if plan.intent is QueryIntent.distinct else ""
        return prefix + ", ".join(quote(column) for column in plan.projection)

    table = graph.tables.get(plan.table)
    if table is None or not table.columns:
        return WILDCARD
    return ", ".join(quote(column.name) for column in table.columns)


def _where(plan: "Plan", binder: "_Binder", graph: "SchemaGraph | None" = None) -> str:
    clauses: list[str] = []

    if plan.filter_column:
        if plan.filter_operator is None:
            raise SecurityError("A filter column requires a filter operator.")
        operator = plan.filter_operator.upper()
        if operator not in OPERATORS:
            raise SecurityError("Unsupported filter operator.")
        column = quote(plan.filter_column)
        if operator in NULL_TESTS:
            clauses.append(f"{column} {operator}")
        elif plan.filter_value is None:
            raise SecurityError(f"Operator {operator} requires a comparison value.")
        elif operator in PATTERN_OPERATORS:
            clauses.append(f"{column} {operator} {binder.bind(like_pattern(plan.filter_value))}")
        else:
            cast_type = ""
            if graph and plan.table in graph.tables:
                col = next(
                    (c for c in graph.tables[plan.table].columns if c.name == plan.filter_column),
                    None,
                )
                if col and col.data_type and col.data_type.lower() != "text":
                    cast_type = f"::{col.data_type}"
            clauses.append(f"{column} {operator} {binder.bind(plan.filter_value)}{cast_type}")

    if plan.text_search_query and plan.text_search_columns:
        pattern = binder.bind(like_pattern(plan.text_search_query))
        parts = [f"{quote(column)} ILIKE {pattern}" for column in plan.text_search_columns]
        clauses.append("(" + " OR ".join(parts) + ")")

    if not clauses:
        return ""
    return " WHERE " + " AND ".join(clauses)


class _Binder:
    def __init__(self) -> None:
        self.params: list[str] = []

    def bind(self, value: str) -> str:
        self.params.append(value)
        return f"${len(self.params)}"


def validate(sql: str, graph: "SchemaGraph | None", schema: str) -> Statement:
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


__all__ = ["is_safe_identifier", "like_pattern", "quote", "statement_for", "validate"]
