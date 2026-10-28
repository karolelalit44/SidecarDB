from .constants import COUNT_ALIAS, NULL_TESTS, OPERATORS, PATTERN_OPERATORS, WILDCARD
from .exceptions import SecurityError
from .guard import like_literal, literal, quote
from .types import ForeignKeyInfo, MeasureFunction, QueryIntent, QueryPlan, SchemaGraph


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


def content_columns(plan: QueryPlan, graph: SchemaGraph) -> list[str]:
    """The plan's columns, minus vector columns."""
    table = graph.tables.get(plan.table)
    if table is None:
        return list(plan.projection)
    vectors = set(table.embedding_columns)
    return [c for c in plan.projection if c not in vectors]


def projection(plan: QueryPlan, graph: SchemaGraph) -> str:
    if plan.intent is QueryIntent.count:
        return f"COUNT(*) AS {COUNT_ALIAS}"

    if plan.intent is QueryIntent.aggregate:
        if plan.measure_function is MeasureFunction.count:
            measure_sql = f"COUNT(*) AS {COUNT_ALIAS}"
        else:
            column = qualify(plan.measure_column, plan, graph)
            alias = f"{plan.measure_function.value}_of_{plan.measure_column}"
            function = plan.measure_function.value.upper()
            measure_sql = f"{function}({column}) AS {quote(alias)}"

        if plan.group_by:
            group_cols = ", ".join(qualify(c, plan, graph) for c in plan.group_by)
            return f"{group_cols}, {measure_sql}"
        return measure_sql

    columns = content_columns(plan, graph)

    if plan.intent is QueryIntent.distinct and columns:
        return "DISTINCT " + ", ".join(qualify(c, plan, graph) for c in columns)

    if columns:
        return ", ".join(qualify(c, plan, graph) for c in columns)

    table = graph.tables.get(plan.table)
    plain = (
        [c.name for c in table.columns if c.name not in set(table.embedding_columns)]
        if table
        else []
    )
    if not plain:
        return WILDCARD
    return ", ".join(qualify(c, plan, graph) for c in plain)


def joins(plan: QueryPlan, graph: SchemaGraph) -> str:
    parts = []
    for related in plan.related_tables:
        relation = find_relation(plan.table, related, graph)
        if relation is None:
            raise SecurityError(f"No foreign key joins {plan.table} to {related}.")
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
    if plan.order_by and plan.intent is not QueryIntent.count:
        direction = "DESC" if plan.descending else "ASC"
        order = f" ORDER BY {qualify(plan.order_by, plan, graph)} {direction}"
    return (
        f"SELECT {projection(plan, graph)} FROM {quote(plan.table)}"
        f"{joins(plan, graph)}{where(plan, graph)}{group}{order}"
    )
