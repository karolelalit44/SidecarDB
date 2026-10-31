from catalog.models import SchemaGraph
from config.constants import (
    DEFAULT_COMPARISON,
    MEASURED_INTENTS,
    NULL_TESTS,
    OPERATORS,
)
from core.logging import get_logger
from planner.models import (
    MeasureFunction,
    Plan,
    PlanPolicy,
    QueryIntent,
)
from planner.prompts import PlanRequest

logger = get_logger("planner.validation")


def column_scope(plan: Plan, graph: SchemaGraph) -> set[str]:
    tables_to_check = [plan.table]
    for j in plan.joins:
        if j.target_table not in tables_to_check:
            tables_to_check.append(j.target_table)
        if j.table not in tables_to_check:
            tables_to_check.append(j.table)
    cols: set[str] = set()
    for t_name in tables_to_check:
        info = graph.tables.get(t_name)
        if info is not None:
            for c in info.column_names:
                cols.add(c)
                cols.add(f"{t_name}.{c}")
    logger.debug(
        "column_scope: table=%s resolved %d column(s) across %d table(s)",
        plan.table,
        len(cols),
        len(tables_to_check),
    )
    return cols


def normalise_plan(plan: Plan, request: PlanRequest) -> Plan:
    logger.debug("normalise_plan: table=%s intent=%s", plan.table, plan.intent.value)
    available = column_scope(plan, request.graph)

    def fix(column: str | None) -> str | None:
        if column is None or column in available:
            return column
        _, dot, tail = column.rpartition(".")
        if dot and tail in available:
            return tail
        return column

    def fix_all(columns: list[str]) -> list[str]:
        fixed = [fix(column) for column in columns]
        return [value for value in fixed if value is not None]

    def fix_template(tpl: str | None) -> str | None:
        if not tpl:
            return None
        cleaned = " ".join(tpl.strip().split())
        return cleaned[: request.policy.error_message_max_chars]

    order_by = plan.order_by.strip() if plan.order_by else None
    descending = plan.descending
    if order_by is None or order_by not in available:
        order_by = None
        descending = False
    if plan.intent is QueryIntent.count:
        order_by = None
        descending = False

    filter_operator = plan.filter_operator
    if filter_operator is None and plan.filter_column and plan.filter_value is not None:
        filter_operator = DEFAULT_COMPARISON

    updates = {
        "filter_column": fix(plan.filter_column),
        "filter_operator": filter_operator,
        "order_by": order_by,
        "descending": descending,
        "projection": fix_all(plan.projection),
        "group_by": fix_all(plan.group_by) if plan.intent in MEASURED_INTENTS else [],
        "text_search_columns": fix_all(plan.text_search_columns),
        "template": fix_template(plan.template),
    }

    if plan.intent in MEASURED_INTENTS:
        if plan.measure_function is MeasureFunction.count:
            updates["measure_column"] = None
        else:
            updates["measure_column"] = fix(plan.measure_column)
    else:
        updates["measure_column"] = None
        updates["measure_function"] = None

    logger.debug("normalise_plan: normalized updates: %s", updates)
    return plan.model_copy(update=updates)


def check_plan(plan: Plan, request: PlanRequest) -> str | None:
    logger.debug(
        "check_plan: validating plan for table=%s intent=%s", plan.table, plan.intent.value
    )
    if plan.table not in request.graph.tables:
        logger.debug("check_plan: table %r does not exist in catalog", plan.table)
        return f"table '{plan.table}' does not exist in the catalog"

    for join in plan.joins:
        if join.table not in request.graph.tables:
            logger.debug("check_plan: joined table %r not in catalog", join.table)
            return f"joined table '{join.table}' does not exist in the catalog"
        if join.target_table not in request.graph.tables:
            logger.debug("check_plan: joined target table %r not in catalog", join.target_table)
            return f"joined target table '{join.target_table}' does not exist in the catalog"
        src_table = request.graph.tables[join.table]
        tgt_table = request.graph.tables[join.target_table]
        if join.on_column not in src_table.column_names:
            logger.debug("check_plan: join column %r not in %s", join.on_column, join.table)
            return f"join column '{join.on_column}' does not exist in table '{join.table}'"
        if join.target_column not in tgt_table.column_names:
            logger.debug(
                "check_plan: target join column %r not in %s",
                join.target_column,
                join.target_table,
            )
            return (
                f"join column '{join.target_column}' does not exist in table '{join.target_table}'"
            )

    available = column_scope(plan, request.graph)
    policy = request.policy

    for value in (plan.measure_column, plan.filter_column, plan.order_by):
        if value and value not in available:
            logger.debug("check_plan: unknown column %r", value)
            return _unknown_column(value, available, policy)

    for value in (*plan.projection, *plan.group_by, *plan.text_search_columns):
        if value not in available:
            logger.debug("check_plan: unknown column %r", value)
            return _unknown_column(value, available, policy)

    if plan.filter_column and not plan.filter_operator:
        logger.debug("check_plan: filter_column %r has no filter_operator", plan.filter_column)
        return "a filter column requires filter_operator"

    if plan.filter_operator and plan.filter_operator not in OPERATORS:
        logger.debug("check_plan: unsupported filter_operator %r", plan.filter_operator)
        return "unsupported filter operator; use one of " + ", ".join(OPERATORS)

    if plan.filter_column and plan.filter_value is None and plan.filter_operator not in NULL_TESTS:
        logger.debug("check_plan: filter_column without filter_value")
        return (
            "a filter column without filter_value is not a predicate; "
            "supply filter_value or leave the filter fields unset"
        )

    if plan.intent in MEASURED_INTENTS:
        if not plan.measure_function:
            logger.debug("check_plan: aggregate plan missing measure_function")
            return "an aggregate plan requires measure_function"
        if plan.measure_function.needs_column:
            if not plan.measure_column:
                logger.debug("check_plan: aggregate plan missing measure_column")
                return "an aggregate plan requires measure_column"
            if "." in plan.measure_column:
                meas_tbl_name, meas_col_name = plan.measure_column.split(".", 1)
                meas_tbl = request.graph.tables.get(meas_tbl_name)
            else:
                meas_tbl = request.graph.tables.get(plan.table)
                meas_col_name = plan.measure_column
            if meas_tbl and meas_col_name not in meas_tbl.numeric_columns:
                numeric = ", ".join(meas_tbl.numeric_columns) or "none"
                logger.debug("check_plan: column %r is not numeric", plan.measure_column)
                return (
                    f"column '{plan.measure_column}' is not numeric; numeric columns are: {numeric}"
                )

    logger.debug("check_plan: plan passed all checks")
    return None


def _unknown_column(value: str, available: set[str], policy: PlanPolicy) -> str:
    listing = ", ".join(sorted(available)[: policy.max_schema_columns])
    msg = f"column '{value}' does not exist; available columns are: {listing}"
    logger.debug("_unknown_column: %s", msg)
    return msg


__all__ = ["check_plan", "column_scope", "normalise_plan"]
