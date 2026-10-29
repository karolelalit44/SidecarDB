from typing import TYPE_CHECKING

import sqlglot
from sqlglot import exp

from core.constants import DEFAULT_LIMIT, MAX_LIMIT
from core.exceptions import SecurityError
from core.plan import (
    COUNT_ALIAS,
    NULL_TESTS,
    OPERATORS,
    PATTERN_OPERATORS,
    WILDCARD,
    MeasureFunction,
    QueryIntent,
    Statement,
)
from utils.logging_config import get_logger
from utils.sql import is_safe_identifier, quote_identifier

if TYPE_CHECKING:
    from core.catalog import SchemaGraph
    from core.plan import QueryPlan

logger = get_logger("guard")


ANONYMOUS = "anonymous"
DIALECT = "postgres"
ROOT_TYPE = exp.Select

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


def quote(name: str) -> str:
    if not is_safe_identifier(name):
        raise ValueError("Unsafe identifier")
    return quote_identifier(name)


def _function_names(node) -> set[str]:
    names = set()
    for value in (node.name, node.sql_name()):
        if isinstance(value, str):
            lowered = value.lower()
            if lowered and lowered != ANONYMOUS:
                names.add(lowered)
    return names


def _is_aggregate(tree: exp.Expression) -> bool:
    return tree.find(exp.Group) is not None or any(True for _ in tree.find_all(exp.AggFunc))


def _rewrites_meaning(tree: exp.Expression, statement: str) -> bool:
    produced = any(True for _ in tree.find_all(exp.NullSafeEQ))
    if not produced:
        return False
    return "IS NOT DISTINCT FROM" not in statement.upper()


def like_pattern(value: str) -> str:
    return value.replace("\\", "\\\\").replace("%", r"\%").replace("_", r"\_")


def statement_for(plan: "QueryPlan", graph: "SchemaGraph", schema: str) -> Statement:
    sql, params = _compose(plan, graph)
    validated = validate(sql, graph, schema)
    return Statement(sql=validated.sql, params=params)


def _compose(plan: "QueryPlan", graph: "SchemaGraph") -> tuple[str, tuple[str, ...]]:
    binder = _Binder()
    group = f" GROUP BY {', '.join(quote(c) for c in plan.group_by)}" if plan.group_by else ""
    order = ""
    if plan.order_by and plan.intent is not QueryIntent.count:
        direction = "DESC" if plan.descending else "ASC"
        order = f" ORDER BY {quote(plan.order_by)} {direction}"
    sql = f"SELECT {_projection(plan, graph)} FROM {quote(plan.table)}"
    sql += _where(plan, binder) + group + order
    return sql, tuple(binder.params)


def _projection(plan: "QueryPlan", graph: "SchemaGraph") -> str:
    if plan.intent is QueryIntent.count:
        return f"COUNT(*) AS {COUNT_ALIAS}"

    if plan.intent is QueryIntent.aggregate:
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


def _where(plan: "QueryPlan", binder: "_Binder") -> str:
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

    for func_node in list(tree.find_all(exp.Func)) + list(tree.find_all(exp.Anonymous)):
        for name in _function_names(func_node):
            if name in FORBIDDEN_FUNCTIONS:
                logger.debug("guard: rejected; it calls the %s function", name)
                raise SecurityError(f"Blocked function: {name}")

    if graph and graph.tables:
        known = {name.lower() for name in graph.tables}
        local = {c.alias_or_name.lower() for c in tree.find_all(exp.CTE) if c.alias_or_name}
        schema_name = schema.lower()
        logger.debug("guard: checking every table reference against %d catalog tables", len(known))
        for table in tree.find_all(exp.Table):
            if table.catalog:
                logger.debug("guard: rejected; %s names another database", table.sql())
                raise SecurityError("Statement references a table outside the configured schema.")
            if table.db and table.db.lower() != schema_name:
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

    if not rewritten:
        logger.debug("guard: statement accepted unchanged")
        return Statement(statement)

    if _rewrites_meaning(tree, statement):
        logger.debug("guard: rejected; the rewrite would change the meaning of the statement")
        raise SecurityError("Statement uses an operator this validator cannot render faithfully.")

    logger.debug("guard: statement accepted; a limit was added or clamped")
    return Statement(tree.sql(dialect=DIALECT))


__all__ = ["is_safe_identifier", "like_pattern", "quote", "statement_for", "validate"]
