from typing import TYPE_CHECKING
from unicodedata import category

import sqlglot
from sqlglot import exp

from core.constants import (
    ALLOWED_CONTROL,
    ALLOWED_FUNCTIONS,
    ANONYMOUS,
    BLOCKED_CATEGORIES,
    BLOCKED_NODES,
    COUNT_ALIAS,
    DEFAULT_LIMIT,
    DIALECT,
    DISALLOWED_SUBSTRINGS,
    IDENTIFIER_EXTRA_CHARACTERS,
    MAX_LIMIT,
    MAX_QUESTION_LENGTH,
    MEASURED_INTENTS,
    MIN_QUESTION_LENGTH,
    NULL_TESTS,
    OPERATORS,
    PATTERN_OPERATORS,
    PG_IDENTIFIER_MAX_LENGTH,
    ROOT_TYPE,
    WILDCARD,
)
from core.exceptions import QuestionError, SecurityError
from core.plan import (
    MeasureFunction,
    QueryIntent,
    Statement,
)
from utils.logging_config import get_logger

if TYPE_CHECKING:
    from core.catalog import SchemaGraph
    from core.plan import Plan

logger = get_logger("guard")


def is_safe_identifier(name: str) -> bool:
    if not name or len(name) > PG_IDENTIFIER_MAX_LENGTH:
        return False
    if not (name[0].isalpha() or name[0] == "_"):
        return False
    return all(
        character.isalnum() or character in IDENTIFIER_EXTRA_CHARACTERS for character in name
    )


def quote_identifier(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def quote_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def quote(name: str) -> str:
    logger.debug("guard quote: identifier=%r", name)
    if not is_safe_identifier(name):
        logger.debug("guard: refusing to compose the identifier %r", name)
        raise SecurityError("The Guard cannot write this identifier safely.")
    return quote_identifier(name)


def quote_column_ref(name: str) -> str:
    if "." in name:
        table_part, col_part = name.split(".", 1)
        return f"{quote(table_part)}.{quote(col_part)}"
    return quote(name)


def validate_question(raw_question: str) -> str:
    question = (raw_question or "").strip()
    if not question or len(question) < MIN_QUESTION_LENGTH:
        logger.debug(
            "question rejected: length %d < MIN_QUESTION_LENGTH (%d)",
            len(question),
            MIN_QUESTION_LENGTH,
        )
        raise QuestionError("The question is too short.")
    if len(question) > MAX_QUESTION_LENGTH:
        logger.debug(
            "question rejected: length %d > MAX_QUESTION_LENGTH (%d)",
            len(question),
            MAX_QUESTION_LENGTH,
        )
        raise QuestionError(f"The question is too long (maximum {MAX_QUESTION_LENGTH} characters).")
    for marker in DISALLOWED_SUBSTRINGS:
        if marker in question:
            logger.warning(
                "Rejected a %d-character question: disallowed symbols (%r)",
                len(question),
                marker,
            )
            raise QuestionError("The question contains disallowed symbols or comment characters.")
    for ch in question:
        if ch in ALLOWED_CONTROL:
            continue
        if category(ch) in BLOCKED_CATEGORIES:
            logger.warning(
                "Rejected question: unsupported control character %r (category %s)",
                ch,
                category(ch),
            )
            raise QuestionError(
                "The question contains unsupported control or formatting characters."
            )
    if not any(ch.isalnum() for ch in question):
        logger.warning("Rejected question: no alphanumeric character")
        raise QuestionError("The question must contain at least one alphanumeric character.")
    logger.debug("question validated successfully: %r (%d chars)", question, len(question))
    return question


def _unallowed_function(node: exp.Func) -> str | None:
    if isinstance(node, exp.Cast):
        return None
    lowered = (node.name if isinstance(node, exp.Anonymous) else node.sql_name() or "").lower()
    if lowered and lowered != ANONYMOUS and lowered not in ALLOWED_FUNCTIONS:
        return lowered
    return None


def like_pattern(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace("%", r"\%").replace("_", r"\_")
    logger.debug("guard like_pattern: escaped %r -> %r", value, escaped)
    return escaped


def statement_for(plan: "Plan", graph: "SchemaGraph", schema: str) -> Statement:
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


def _compose(plan: "Plan", graph: "SchemaGraph") -> tuple[str, tuple[str, ...]]:
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


def _projection(plan: "Plan", graph: "SchemaGraph") -> str:
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
            measure_sql = (
                f"{function}({quote_column_ref(plan.measure_column)}) AS {quote(alias)}"
            )
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


def _where(plan: "Plan", binder: "_Binder", graph: "SchemaGraph | None" = None) -> str:
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


__all__ = [
    "is_safe_identifier",
    "like_pattern",
    "quote",
    "quote_column_ref",
    "quote_identifier",
    "quote_literal",
    "statement_for",
    "validate",
    "validate_question",
]
