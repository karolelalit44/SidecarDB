from typing import NamedTuple

from .constants import COUNT_ALIAS, NULL_TESTS, OPERATORS, PATTERN_OPERATORS, WILDCARD
from .exceptions import PlanningError, SecurityError
from .guard import quote
from .types import MeasureFunction, QueryIntent, QueryPlan, SchemaGraph


class Rendered(NamedTuple):
    """A statement with $n placeholders and the values bound to them.

    Filter and search values never ride inside the SQL text; they travel as
    parameters the database binds, so no escaping discipline stands between a
    user-supplied value and the parser.
    """

    sql: str
    params: tuple[str, ...]


class _Binder:
    """Hands out one placeholder per value."""

    def __init__(self) -> None:
        self.params: list[str] = []

    def bind(self, value: str) -> str:
        self.params.append(value)
        return f"${len(self.params)}"


def like_pattern(value: str) -> str:
    """Escape the LIKE metacharacters so the value matches literally."""
    return value.replace("\\", "\\\\").replace("%", r"\%").replace("_", r"\_")


def projection(plan: QueryPlan, graph: SchemaGraph) -> str:
    if plan.intent is QueryIntent.count:
        return f"COUNT(*) AS {COUNT_ALIAS}"

    if plan.intent is QueryIntent.aggregate:
        # check_plan rejects an incomplete aggregate before render ever runs;
        # these raises are the guard for render.build() called directly.
        if plan.measure_function is None:
            raise PlanningError("an aggregate plan requires measure_function")
        if plan.measure_function is MeasureFunction.count:
            measure_sql = f"COUNT(*) AS {COUNT_ALIAS}"
        else:
            if plan.measure_column is None:
                raise PlanningError("a non-count aggregate requires measure_column")
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


def where(plan: QueryPlan, binder: _Binder) -> str:
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
            clauses.append(f"{column} {operator} {binder.bind(plan.filter_value)}")

    if plan.text_search_query and plan.text_search_columns:
        pattern = binder.bind(like_pattern(plan.text_search_query))
        parts = [f"{quote(column)} ILIKE {pattern}" for column in plan.text_search_columns]
        clauses.append("(" + " OR ".join(parts) + ")")

    if not clauses:
        return ""
    return " WHERE " + " AND ".join(clauses)


def build(plan: QueryPlan, graph: SchemaGraph) -> Rendered:
    binder = _Binder()
    group = f" GROUP BY {', '.join(quote(c) for c in plan.group_by)}" if plan.group_by else ""
    order = ""
    if plan.order_by and plan.intent is not QueryIntent.count:
        direction = "DESC" if plan.descending else "ASC"
        order = f" ORDER BY {quote(plan.order_by)} {direction}"
    sql = f"SELECT {projection(plan, graph)} FROM {quote(plan.table)}"
    sql += where(plan, binder) + group + order
    return Rendered(sql=sql, params=tuple(binder.params))
