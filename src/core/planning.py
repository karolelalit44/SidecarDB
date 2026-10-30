from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from pydantic import ValidationError

from core.catalog import SchemaGraph
from core.constants import (
    DEFAULT_COMPARISON,
    MEASURED_INTENTS,
    NULL_TESTS,
    OPERATOR_TEXT,
    OPERATORS,
    PLANNING_FAILED_MESSAGE,
    TRUNCATION_MARK,
)
from core.exceptions import ConfigurationError, PlanningError, ProviderError
from core.plan import (
    MeasureFunction,
    Plan,
    PlanPolicy,
    QueryIntent,
)
from utils.logging_config import get_logger

if TYPE_CHECKING:
    from core.capabilities import Planner

logger = get_logger("planning")


@dataclass(frozen=True, slots=True)
class PlanRequest:
    question: str
    graph: SchemaGraph
    policy: PlanPolicy
    tables: list[str] = field(default_factory=list)


SYSTEM = f"""\
You produce a Plan JSON object for the schema supplied below.
The JSON object must include "table", "intent" and "template", plus any needed
optional fields. Never return data results directly; return the plan
specification only. There is exactly one model call per question, so the
template must be complete and answerable on its own.

Inputs you are given:
- SCHEMA: tables, columns, data types, operator-written comments.

Field contracts:
- table: required. The primary table that answers the question.
- joins: optional list of join objects when data or filter conditions come from
  related tables connected by foreign keys:
  [{{"table": "...", "on_column": "...", "target_table": "...",
    "target_column": "...", "join_type": "inner"|"left"}}]
- intent: one of {", ".join(repr(i.value) for i in QueryIntent)}.
  "count" yields one row holding a row count. "aggregate" yields one row per group
  (or one row overall) holding measure_function applied to measure_column.
  "distinct" yields unique combinations of projection columns. "retrieve" yields
  one row per matching record.
- measure_column / measure_function: required together by "aggregate".
  measure_function is one of {", ".join(repr(m.value) for m in MeasureFunction)}.
  measure_function "count" counts rows and needs no measure_column; every other
  measure_function requires measure_column, which must be of a numeric type.
- filter_column / filter_value / filter_operator: a predicate on one column.
  filter_operator is {OPERATOR_TEXT}. Use '=' (never '==') for equality.
  Pattern and comparison operators need filter_value; null tests must leave it unset.
- text_search_query / text_search_columns: use these instead of filter_column when
  the question supplies wording to be found inside the content of a text column.
- projection: the columns to return. Prefer human-readable descriptor columns over
  raw ID or UUID columns. Leave empty to return every column.
- order_by / descending: optional ordering. order_by must be a single column name only,
  taken from SCHEMA; do not include ASC or DESC in order_by; set descending to
  true or false.
- group_by: the non-measured columns of an aggregate that must be listed per group.
  Leave empty when not aggregating.
- template: required. The English sentence that answers the question once the
  rows are in. One or two sentences, max 200 chars, no markdown, no mention of
  SQL, tables or columns. Use placeholders in curly braces and only these:
  {{row_count}} and {{count}} for the number of rows returned, the number in a
  count or aggregate row, plus the name of each projected column.

Constraints:
- Every table and column must appear in SCHEMA. The plan is checked against the
  catalog and a rejected plan is returned to you with the reason.
- Never guess column names not in SCHEMA. For recency or date ordering, select an
  existing date or timestamp column from SCHEMA.
- Prefer a value taken from the question over an invented one.
- Leave any field you do not need unset.
"""

USER = """\
SCHEMA
{schema}

QUESTION
{question}
"""

REPAIR = """\
The previous plan was rejected: {error}

Return a corrected plan. Verify each table and column against SCHEMA before
answering. If a column was rejected, pick only from the available columns.
Read the field contracts above.
"""


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


def prompt_chars(reserve: int, policy: PlanPolicy) -> int:
    chars = max(policy.llm_context_tokens - reserve, 0) * policy.llm_chars_per_token
    logger.debug("prompt_chars: reserve=%d tokens -> budget=%d chars", reserve, chars)
    return chars


def trim(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    logger.debug("trim: trimming text from %d to %d chars", len(text), limit)
    if limit <= len(TRUNCATION_MARK):
        return text[:limit]
    return text[: limit - len(TRUNCATION_MARK)] + TRUNCATION_MARK


def build_prompt(request: PlanRequest) -> str:
    policy = request.policy
    reserve = policy.llm_max_tokens + len(SYSTEM) // policy.llm_chars_per_token
    budget = prompt_chars(reserve, policy)
    schema_overhead = len(USER) - len("{schema}") - len("{question}")

    schema = trim(
        request.graph.describe(request.tables, policy.max_schema_columns),
        budget - schema_overhead,
    )
    logger.debug(
        "build_prompt: tables=%d budget=%d chars schema_overhead=%d chars schema_len=%d chars",
        len(request.tables),
        budget,
        schema_overhead,
        len(schema),
    )

    return USER.format(schema=schema, question=request.question)


async def make_plan(
    request: PlanRequest,
    llm: "Planner",
    error: str | None = None,
    attempt: int = 0,
) -> tuple[Plan | None, str | None]:
    policy = request.policy
    prompt = build_prompt(request)
    if error:
        prompt = f"{REPAIR.format(error=error[: policy.repair_char_budget])}\n\n{prompt}"

    temperature = policy.llm_temperature if attempt == 0 else policy.llm_repair_temperature

    logger.debug(
        "Planning attempt %d: tables=%d prompt_chars=%d temperature=%s repairing=%s",
        attempt + 1,
        len(request.tables),
        len(prompt),
        temperature,
        error,
    )

    try:
        payload = await llm.complete_json(SYSTEM, prompt, temperature)
        logger.debug("Plan payload: %s", payload)
        result = normalise_plan(Plan.model_validate(payload), request)
    except (ConfigurationError, ProviderError):
        raise
    except (PlanningError, ValidationError) as exc:
        logger.debug("Plan malformed: %s: %s", type(exc).__name__, exc)
        return None, " ".join(str(exc).split())[: policy.error_message_max_chars]
    except Exception as exc:
        logger.warning("Planner call failed: %s: %s", type(exc).__name__, exc)
        return None, "a usable plan could not be produced"

    reason = check_plan(result, request)
    if reason:
        logger.debug(
            "Plan rejected: %s (intent=%s table=%s)",
            reason,
            result.intent.value,
            result.table,
        )
        return None, reason

    logger.debug(
        "Plan accepted: intent=%s table=%s measure=%s/%s filter=%s %s %r projection=%d order_by=%s",
        result.intent.value,
        result.table,
        result.measure_function.value if result.measure_function else None,
        result.measure_column,
        result.filter_column,
        result.filter_operator,
        result.filter_value,
        len(result.projection),
        result.order_by,
    )
    return result, None


async def generate_plan(
    question: str,
    graph: SchemaGraph,
    tables: list[str],
    planner: "Planner",
    policy: PlanPolicy | None = None,
) -> Plan:
    active_policy = policy or PlanPolicy()
    request = PlanRequest(
        question=question,
        graph=graph,
        policy=active_policy,
        tables=tables,
    )
    logger.debug(
        "planning pipeline: starting planning loop (max_attempts=%d)",
        active_policy.max_plan_attempts,
    )
    error: str | None = None
    for attempt in range(active_policy.max_plan_attempts):
        plan_result, error = await make_plan(request, planner, error, attempt)
        if plan_result is not None:
            logger.debug(
                "planning pipeline: plan produced on attempt %d of %d",
                attempt + 1,
                active_policy.max_plan_attempts,
            )
            return plan_result
        logger.debug(
            "plan: attempt %d of %d failed: %s",
            attempt + 1,
            active_policy.max_plan_attempts,
            error,
        )

    logger.error("No plan survived %d attempts: %s", active_policy.max_plan_attempts, error)
    raise PlanningError(PLANNING_FAILED_MESSAGE)


__all__ = [
    "SYSTEM",
    "PlanRequest",
    "build_prompt",
    "check_plan",
    "column_scope",
    "generate_plan",
    "make_plan",
    "normalise_plan",
    "prompt_chars",
    "trim",
]
