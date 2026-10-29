from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from pydantic import ValidationError

from core.catalog import SchemaGraph
from core.exceptions import ConfigurationError, PlanningError, ProviderError
from core.plan import (
    COMPARISON_OPERATORS,
    DEFAULT_COMPARISON,
    MEASURED_INTENTS,
    NULL_TESTS,
    OPERATORS,
    PATTERN_OPERATORS,
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


COMPARISON_TEXT = ", ".join(repr(op) for op in COMPARISON_OPERATORS)
PATTERN_TEXT = ", ".join(repr(op) for op in PATTERN_OPERATORS)
NULL_TEST_TEXT = ", ".join(repr(op) for op in NULL_TESTS)
OPERATOR_TEXT = (
    f"a comparison ({COMPARISON_TEXT}), a pattern match ({PATTERN_TEXT}), "
    f"or a null test ({NULL_TEST_TEXT})"
)

SYSTEM = f"""\
You produce a Plan JSON object for the schema supplied below.
The JSON object must include "table", "intent" and "template", plus any needed
optional fields. Never return data results directly; return the plan
specification only. There is exactly one model call per question, so the
template must be complete and answerable on its own.

Inputs you are given:
- SCHEMA: tables, columns, data types, operator-written comments.

Field contracts:
- table: required. The one table that answers the question.
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

TRUNCATION_MARK = " [...]"


def column_scope(plan: Plan, graph: SchemaGraph) -> set[str]:
    info = graph.tables.get(plan.table)
    return set(info.column_names) if info is not None else set()


def normalise_plan(plan: Plan, request: PlanRequest) -> Plan:
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

    return plan.model_copy(update=updates)


def check_plan(plan: Plan, request: PlanRequest) -> str | None:
    if plan.table not in request.graph.tables:
        return f"table '{plan.table}' does not exist in the catalog"

    available = column_scope(plan, request.graph)
    policy = request.policy

    for value in (plan.measure_column, plan.filter_column, plan.order_by):
        if value and value not in available:
            return _unknown_column(value, available, policy)

    for value in (*plan.projection, *plan.group_by, *plan.text_search_columns):
        if value not in available:
            return _unknown_column(value, available, policy)

    if plan.filter_column and not plan.filter_operator:
        return "a filter column requires filter_operator"

    if plan.filter_operator and plan.filter_operator not in OPERATORS:
        return "unsupported filter operator; use one of " + ", ".join(OPERATORS)

    if plan.filter_column and plan.filter_value is None and plan.filter_operator not in NULL_TESTS:
        return (
            "a filter column without filter_value is not a predicate; "
            "supply filter_value or leave the filter fields unset"
        )

    if plan.intent in MEASURED_INTENTS:
        if not plan.measure_function:
            return "an aggregate plan requires measure_function"
        if plan.measure_function.needs_column:
            if not plan.measure_column:
                return "an aggregate plan requires measure_column"
            if plan.measure_column not in request.graph.tables[plan.table].numeric_columns:
                numeric = ", ".join(request.graph.tables[plan.table].numeric_columns) or "none"
                return (
                    f"column '{plan.measure_column}' is not numeric; numeric columns are: {numeric}"
                )

    return None


def _unknown_column(value: str, available: set[str], policy: PlanPolicy) -> str:
    listing = ", ".join(sorted(available)[: policy.max_schema_columns])
    return f"column '{value}' does not exist; available columns are: {listing}"


def prompt_chars(reserve: int, policy: PlanPolicy) -> int:
    return max(policy.llm_context_tokens - reserve, 0) * policy.llm_chars_per_token


def trim(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
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


__all__ = [
    "SYSTEM",
    "PlanRequest",
    "build_prompt",
    "check_plan",
    "column_scope",
    "make_plan",
    "normalise_plan",
    "prompt_chars",
    "trim",
]
