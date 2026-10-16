from .constants import NULL_TESTS, OPERATORS, PATTERN_OPERATORS, WILDCARD
from .exceptions import SecurityError
from .guard import like_literal, literal, quote
from .types import ForeignKeyInfo, QueryIntent, QueryPlan, SchemaGraph


def find_relation(left: str, right: str, graph: SchemaGraph) -> ForeignKeyInfo | None:
    for fk in graph.foreign_keys:
        if fk.from_table == left and fk.to_table == right:
            return fk
        if fk.from_table == right and fk.to_table == left:
            return ForeignKeyInfo(
                from_table=right,
                from_column=fk.to_column,
                to_table=left,
                to_column=fk.from_column,
            )
    return None


def owner_of(column: str, plan: QueryPlan, graph: SchemaGraph) -> str | None:
    for name in plan.tables:
        info = graph.tables.get(name)
        if info is not None and column in info.column_names:
            return name
    return None


def qualify(column: str, plan: QueryPlan, graph: SchemaGraph) -> str:
    if len(plan.tables) < 2:
        return quote(column)
    owner = owner_of(column, plan, graph)
    return f"{quote(owner)}.{quote(column)}" if owner else quote(column)


def projection(plan: QueryPlan, graph: SchemaGraph) -> str:
    if plan.intent is QueryIntent.count:
        return "COUNT(*) AS record_count"

    if plan.intent is QueryIntent.aggregate:
        column = qualify(plan.measure_column, plan, graph)
        alias = f"{plan.measure_function.value}_of_{plan.measure_column}"
        return f"{quote(plan.measure_function.value.upper())}({column}) AS {quote(alias)}"

    if plan.intent is QueryIntent.distinct and plan.projection:
        return "DISTINCT " + ", ".join(qualify(c, plan, graph) for c in plan.projection)

    if plan.projection:
        return ", ".join(qualify(c, plan, graph) for c in plan.projection)

    return WILDCARD


def joins(plan: QueryPlan, graph: SchemaGraph) -> str:
    parts = []
    for related in plan.related_tables:
        relation = find_relation(plan.table, related, graph)
        if relation is None:
            parts.append(f", {quote(related)}")
            continue
        parts.append(
            f" JOIN {quote(related)}"
            f" ON {quote(plan.table)}.{quote(relation.from_column)}"
            f" = {quote(related)}.{quote(relation.to_column)}"
        )
    return "".join(parts)


def where(plan: QueryPlan, graph: SchemaGraph) -> str:
    clauses: list[str] = []

    if plan.filter_column:
        if plan.filter_operator is None:
            raise SecurityError("A filter column requires a filter operator.")
        operator = plan.filter_operator.upper()
        if operator not in OPERATORS:
            raise SecurityError("Unsupported filter operator.")
        column = qualify(plan.filter_column, plan, graph)
        if operator in NULL_TESTS:
            clauses.append(f"{column} {operator}")
        elif plan.filter_value is None:
            raise SecurityError(f"Operator {operator} requires a comparison value.")
        elif operator in PATTERN_OPERATORS:
            clauses.append(f"{column} {operator} {like_literal(plan.filter_value)}")
        else:
            clauses.append(f"{column} {operator} {literal(plan.filter_value)}")

    if plan.text_search_query and plan.text_search_columns:
        parts = [
            f"{qualify(column, plan, graph)} ILIKE {like_literal(plan.text_search_query)}"
            for column in plan.text_search_columns
        ]
        clauses.append("(" + " OR ".join(parts) + ")")

    if not clauses:
        return ""
    return " WHERE " + " AND ".join(clauses)


def build(plan: QueryPlan, graph: SchemaGraph) -> str:
    group = (
        f" GROUP BY {', '.join(qualify(c, plan, graph) for c in plan.group_by)}"
        if plan.group_by
        else ""
    )
    order = ""
    if plan.order_by:
        direction = "DESC" if plan.descending else "ASC"
        order = f" ORDER BY {qualify(plan.order_by, plan, graph)} {direction}"
    return (
        f"SELECT {projection(plan, graph)} FROM {quote(plan.table)}"
        f"{joins(plan, graph)}{where(plan, graph)}{group}{order}"
    )
