from catalog.models import SchemaGraph
from config.constants import (
    COUNT_ALIAS,
    DEFAULT_LIMIT,
    MEASURED_INTENTS,
    NULL_TESTS,
    OPERATORS,
    PATTERN_OPERATORS,
    WILDCARD,
)
from core.exceptions import SecurityError
from core.logging import get_logger
from guard.quoting import like_pattern, quote, quote_column_ref
from guard.sql import validate
from planner.models import MeasureFunction, Plan, QueryIntent, Statement

logger = get_logger("planner.compiler")


def statement_for(plan: Plan, graph: SchemaGraph, schema: str) -> Statement:
    logger.debug(
        "guard: statement_for initiated for table=%s intent=%s schema=%s",
        plan.table,
        plan.intent.value,
        schema,
    )
    sql, params = _compose(plan, graph)
    logger.debug("guard: composed SQL: %s with %d bound parameter(s)", sql, len(params))
    validated = validate(sql, graph, schema)
    logger.debug("guard: statement validated successfully")
    return Statement(sql=validated.sql, params=params)


def _compose(plan: Plan, graph: SchemaGraph) -> tuple[str, tuple[str, ...]]:
    logger.debug(
        "guard _compose: table=%s intent=%s group_by=%s order_by=%s descending=%s",
        plan.table,
        plan.intent.value,
        plan.group_by,
        plan.order_by,
        plan.descending,
    )
    binder = _Binder()
    group = (
        f" GROUP BY {', '.join(quote_column_ref(c) for c in plan.group_by)}"
        if plan.group_by
        else ""
    )
    order = ""
    if plan.order_by and plan.intent is not QueryIntent.count:
        direction = "DESC" if plan.descending else "ASC"
        order = f" ORDER BY {quote_column_ref(plan.order_by)} {direction}"
    projection_clause = _projection(plan, graph)
    where_clause = _where(plan, binder, graph)
    joins_clause = ""
    if plan.joins:
        parts = [
            f"{'LEFT JOIN' if j.join_type.value == 'left' else 'INNER JOIN'} "
            f"{quote(j.target_table)} ON "
            f"{quote(j.table)}.{quote(j.on_column)} = "
            f"{quote(j.target_table)}.{quote(j.target_column)}"
            for j in plan.joins
        ]
        joins_clause = " " + " ".join(parts)
    sql = f"SELECT {projection_clause} FROM {quote(plan.table)}{joins_clause}"
    sql += where_clause + group + order
    if plan.intent is not QueryIntent.count and not (
        plan.intent in MEASURED_INTENTS and not plan.group_by
    ):
        sql += f" LIMIT {DEFAULT_LIMIT}"
    logger.debug("guard _compose complete: sql=%r, bound_params=%s", sql, binder.params)
    return sql, tuple(binder.params)


def _projection(plan: Plan, graph: SchemaGraph) -> str:
    logger.debug("guard _projection: building projection for intent=%s", plan.intent.value)
    if plan.intent is QueryIntent.count:
        return f"COUNT(*) AS {COUNT_ALIAS}"

    if plan.intent in MEASURED_INTENTS:
        assert plan.measure_function is not None
        measure_sql = f"COUNT(*) AS {COUNT_ALIAS}"
        if plan.measure_function is not MeasureFunction.count:
            assert plan.measure_column is not None
            clean_alias = plan.measure_column.replace(".", "_")
            alias = f"{plan.measure_function.value}_of_{clean_alias}"
            function = plan.measure_function.value.upper()
            measure_sql = f"{function}({quote_column_ref(plan.measure_column)}) AS {quote(alias)}"
        if plan.group_by:
            group_cols = ", ".join(quote_column_ref(column) for column in plan.group_by)
            return f"{group_cols}, {measure_sql}"
        return measure_sql

    if plan.projection:
        prefix = "DISTINCT " if plan.intent is QueryIntent.distinct else ""
        return prefix + ", ".join(quote_column_ref(column) for column in plan.projection)

    table = graph.tables.get(plan.table)
    if table is None or not table.columns:
        return WILDCARD
    return ", ".join(quote(column.name) for column in table.columns)


def _where(plan: Plan, binder: "_Binder", graph: SchemaGraph | None = None) -> str:
    logger.debug(
        "guard _where: building clauses (filter_col=%s, filter_op=%s, text_search=%s)",
        plan.filter_column,
        plan.filter_operator,
        bool(plan.text_search_query),
    )
    clauses: list[str] = []

    if plan.filter_column:
        if plan.filter_operator is None:
            raise SecurityError("A filter column requires a filter operator.")
        operator = plan.filter_operator.upper()
        if operator not in OPERATORS:
            raise SecurityError("Unsupported filter operator.")
        column = quote_column_ref(plan.filter_column)
        if operator in NULL_TESTS:
            clauses.append(f"{column} {operator}")
        elif plan.filter_value is None:
            raise SecurityError(f"Operator {operator} requires a comparison value.")
        elif operator in PATTERN_OPERATORS:
            clauses.append(f"{column} {operator} {binder.bind(like_pattern(plan.filter_value))}")
        else:
            bound_param = binder.bind(plan.filter_value)
            val_expr = bound_param
            if graph:
                target_tbl = plan.table
                target_col = plan.filter_column
                if "." in plan.filter_column:
                    target_tbl, target_col = plan.filter_column.split(".", 1)
                if target_tbl in graph.tables:
                    col = next(
                        (c for c in graph.tables[target_tbl].columns if c.name == target_col),
                        None,
                    )
                    if col and col.data_type and col.data_type.lower() != "text":
                        val_expr = f"CAST({bound_param} AS {col.data_type})"
            clauses.append(f"{column} {operator} {val_expr}")

    if plan.text_search_query and plan.text_search_columns:
        pattern = binder.bind(like_pattern(plan.text_search_query))
        parts = [
            f"LOWER({quote_column_ref(column)}) LIKE LOWER({pattern})"
            for column in plan.text_search_columns
        ]
        clauses.append("(" + " OR ".join(parts) + ")")

    if not clauses:
        logger.debug("guard _where: no WHERE clauses generated")
        return ""
    result = " WHERE " + " AND ".join(clauses)
    logger.debug("guard _where: generated clause: %s", result)
    return result


class _Binder:
    def __init__(self, param_style: str = "numeric") -> None:
        self.params: list[str] = []
        self.param_style = param_style

    def bind(self, value: str) -> str:
        self.params.append(value)
        param_placeholder = "?" if self.param_style == "qmark" else f"${len(self.params)}"
        logger.debug("binder: bound parameter %s = %r", param_placeholder, value)
        return param_placeholder


__all__ = ["statement_for"]
