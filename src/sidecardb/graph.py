import time
from typing import TYPE_CHECKING, NamedTuple

from pydantic import ValidationError

from . import render
from .config import Settings
from .constants import (
    COMPARISON_OPERATORS,
    COUNT_ALIAS,
    DEFAULT_COMPARISON,
    EQUALITY_ALIASES,
    NULL_TESTS,
    OPERATORS,
    PATTERN_OPERATORS,
)
from .exceptions import ConfigurationError, ExecutionError, PlanningError, ProviderError
from .llm import prompt_chars, trim
from .logging_config import get_logger
from .types import MeasureFunction, QueryIntent, QueryPlan, RetrievalMode, SchemaGraph

if TYPE_CHECKING:
    from .execute import Executor
    from .llm import LLMClient
    from .retrieval import SchemaIndex

logger = get_logger("graph")

OPERATOR_TEXT = (
    f"a comparison ({', '.join(repr(op) for op in COMPARISON_OPERATORS)}), "
    f"a pattern match ({', '.join(repr(op) for op in PATTERN_OPERATORS)}), "
    f"or a null test ({', '.join(repr(op) for op in NULL_TESTS)})"
)

SYSTEM = f"""\
You produce a QueryPlan JSON object for the schema supplied below.
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
  Pattern operators need filter_value; null tests must leave it unset.
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


class Outcome(NamedTuple):
    """What one completed search produced."""

    plan: QueryPlan
    sql: str
    params: tuple[str, ...]
    rows: list[dict]
    retrieved_tables: list[str]
    retrieval_mode: RetrievalMode


def column_scope(plan: QueryPlan, graph: SchemaGraph) -> set[str]:
    info = graph.tables.get(plan.table)
    return set(info.column_names) if info is not None else set()


def normalise_plan(plan: QueryPlan, graph: SchemaGraph, settings: Settings) -> QueryPlan:
    available = column_scope(plan, graph)

    def fix(column: str | None) -> str | None:
        if column is None or column in available:
            return column
        stripped = column.strip('"` ')
        if stripped in available:
            return stripped
        _, dot, tail = column.rpartition(".")
        tail_stripped = tail.strip('"` ')
        if dot and tail_stripped in available:
            return tail_stripped
        lowered = column.lower()
        if "creat" in lowered or "time" in lowered or "date" in lowered:
            table_info = graph.tables.get(plan.table)
            if table_info:
                for c in table_info.columns:
                    if c.name in available and (
                        c.type_category == "D"
                        or "time" in c.data_type.lower()
                        or "date" in c.data_type.lower()
                    ):
                        return c.name
        return column

    def fix_all(columns: list[str]) -> list[str]:
        return [
            fixed
            for c in columns
            if (fixed := fix(c)) is not None
            and not (plan.intent is QueryIntent.aggregate and (c == "count" or c == COUNT_ALIAS))
        ]

    def fix_template(tpl: str | None) -> str | None:
        if not tpl:
            return None
        cleaned = " ".join(tpl.strip().split())
        return cleaned[: settings.error_message_max_chars]

    order_by = plan.order_by
    descending = plan.descending
    if order_by is not None:
        order_cleaned = order_by.strip()
        order_upper = order_cleaned.upper()
        desc_suffix = " DESC"
        asc_suffix = " ASC"
        if order_upper.endswith(desc_suffix):
            order_by = order_cleaned[: -len(desc_suffix)].rstrip()
            descending = True
        elif order_upper.endswith(asc_suffix):
            order_by = order_cleaned[: -len(asc_suffix)].rstrip()
            descending = False
        else:
            order_by = order_cleaned
        order_by = fix(order_by)
        if order_by not in available:
            order_by = None

    if plan.intent is QueryIntent.count:
        order_by = None
        descending = False

    filter_operator = plan.filter_operator
    if filter_operator in EQUALITY_ALIASES or (
        filter_operator is None and plan.filter_column and plan.filter_value is not None
    ):
        filter_operator = DEFAULT_COMPARISON

    filter_column = fix(plan.filter_column)
    filter_value = plan.filter_value
    group_by = fix_all(plan.group_by) if plan.intent is QueryIntent.aggregate else []

    if (
        plan.intent is QueryIntent.aggregate
        and not group_by
        and filter_column in available
        and filter_value is None
    ):
        group_by = [filter_column]
        filter_column = None
        filter_operator = None
    elif filter_column and filter_value is None and filter_operator not in NULL_TESTS:
        filter_column = None
        filter_operator = None

    updates = {
        "filter_column": filter_column,
        "filter_operator": filter_operator,
        "order_by": order_by,
        "descending": descending,
        "projection": fix_all(plan.projection),
        "group_by": group_by,
        "text_search_columns": fix_all(plan.text_search_columns),
        "template": fix_template(plan.template),
    }

    if plan.intent is QueryIntent.aggregate:
        if plan.measure_function is MeasureFunction.count:
            updates["measure_column"] = None
        else:
            updates["measure_column"] = fix(plan.measure_column)
    else:
        updates["measure_column"] = None
        updates["measure_function"] = None

    return plan.model_copy(update=updates)


def check_plan(plan: QueryPlan, graph: SchemaGraph, settings: Settings) -> str | None:
    if plan.table not in graph.tables:
        return f"table '{plan.table}' does not exist in the schema"

    available = column_scope(plan, graph)

    for value in (plan.measure_column, plan.filter_column, plan.order_by):
        if value and value not in available:
            return _unknown_column(value, available, settings)

    for value in (*plan.projection, *plan.group_by, *plan.text_search_columns):
        if value not in available:
            return _unknown_column(value, available, settings)

    if plan.filter_column and not plan.filter_operator:
        return "a filter column requires filter_operator"

    if plan.filter_operator and plan.filter_operator not in OPERATORS:
        return "unsupported filter operator; use one of " + ", ".join(OPERATORS)

    if plan.intent is QueryIntent.aggregate:
        if not plan.measure_function:
            return "an aggregate plan requires measure_function"
        if plan.measure_function.needs_column:
            if not plan.measure_column:
                return "an aggregate plan requires measure_column"
            if plan.measure_column not in graph.tables[plan.table].numeric_columns:
                numeric = ", ".join(graph.tables[plan.table].numeric_columns) or "none"
                return (
                    f"column '{plan.measure_column}' is not numeric; numeric columns are: {numeric}"
                )

    return None


def _unknown_column(value: str, available: set[str], settings: Settings) -> str:
    """Return error message listing available columns for repair."""
    listing = ", ".join(sorted(available)[: settings.max_schema_columns])
    return f"column '{value}' does not exist; available columns are: {listing}"


def redact(reason: str, settings: Settings) -> str:
    if not settings.redact_catalog_in_errors:
        return reason
    return reason.split(";")[0].strip()


def _elapsed_ms(started: float) -> int:
    """Whole milliseconds since a perf_counter reading."""
    return int((time.perf_counter() - started) * 1000)


async def link(
    question: str, graph: SchemaGraph, index: "SchemaIndex | None", settings: Settings
) -> tuple[list[str], RetrievalMode, list[str]]:
    if index is not None:
        tables = await index.tables_for(
            question, settings.max_schema_tables, settings.retrieval_min_score
        )
        if tables:
            return tables, RetrievalMode.embedding, tables

    logger.debug(
        "link: no embedding match at or above %s; falling back to catalog order",
        settings.retrieval_min_score,
    )
    ordered = sorted(graph.tables.values(), key=lambda t: t.row_estimate, reverse=True)
    names = [t.name for t in ordered[: settings.max_schema_tables]]
    return names, RetrievalMode.catalog, []


async def plan(
    question: str,
    graph: SchemaGraph,
    tables: list[str],
    settings: Settings,
    llm: "LLMClient",
    error: str | None = None,
    attempt: int = 0,
) -> tuple[QueryPlan | None, str | None]:
    """Execute one query planning attempt, with repair prompt if retrying."""
    prompt = build_prompt(question, graph, tables, settings)
    if error:
        prompt = f"{REPAIR.format(error=error[: settings.repair_char_budget])}\n\n{prompt}"

    temperature = settings.llm_temperature if attempt == 0 else settings.llm_repair_temperature

    logger.debug(
        "Planning attempt %d: tables=%d prompt_chars=%d temperature=%s repairing=%s",
        attempt + 1,
        len(tables),
        len(prompt),
        temperature,
        error,
    )

    try:
        payload = await llm.complete_json(SYSTEM, prompt, temperature)
        logger.debug("Plan payload: %s", payload)
        result = normalise_plan(QueryPlan.model_validate(payload), graph, settings)
    except (ConfigurationError, ProviderError):
        # A misconfigured or unreachable provider is not a bad plan; retrying
        # it as one would burn every attempt on the same outage.
        raise
    except (PlanningError, ValidationError) as exc:
        logger.debug("Plan malformed: %s: %s", type(exc).__name__, exc)
        return None, " ".join(str(exc).split())[: settings.error_message_max_chars]
    except Exception as exc:
        logger.warning("Planner call failed: %s: %s", type(exc).__name__, exc)
        return None, "a usable plan could not be produced"

    reason = check_plan(result, graph, settings)
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


def build_prompt(question: str, graph: SchemaGraph, tables: list[str], settings: Settings) -> str:
    """Assemble the planner prompt bounded by LLM context budget."""
    reserve = settings.llm_max_tokens + len(SYSTEM) // settings.llm_chars_per_token
    budget = prompt_chars(reserve, settings)
    schema_overhead = len(USER) - len("{schema}") - len("{question}")

    schema = trim(graph.describe(tables, settings.max_schema_columns), budget - schema_overhead)

    return USER.format(schema=schema, question=question)


async def run(
    question: str,
    graph: SchemaGraph,
    index: "SchemaIndex | None",
    settings: Settings,
    llm: "LLMClient",
    executor: "Executor",
) -> Outcome:
    """Execute query generation pipeline: link -> plan -> render -> run."""
    logger.debug("run: planning for %r", question)
    stage = time.perf_counter()
    tables, retrieval_mode, retrieved_tables = await link(question, graph, index, settings)
    linked_ms = _elapsed_ms(stage)
    logger.debug(
        "link: mode=%s tables=%s in %d ms",
        retrieval_mode.value,
        tables,
        linked_ms,
    )

    plan_result = None
    error = None
    for attempt in range(settings.max_plan_attempts):
        plan_result, error = await plan(question, graph, tables, settings, llm, error, attempt)
        if plan_result is not None:
            break
        logger.debug(
            "plan: attempt %d of %d failed: %s",
            attempt + 1,
            settings.max_plan_attempts,
            error,
        )

    if plan_result is None:
        logger.error("No plan survived %d attempts: %s", settings.max_plan_attempts, error)
        raise PlanningError(redact(error or "No valid plan was produced.", settings))

    try:
        rendered = render.build(plan_result, graph)
        sql = executor.prepare(rendered, graph)
    except Exception as exc:
        logger.warning("Render failed: %s: %s", type(exc).__name__, exc)
        raise PlanningError(redact(str(exc), settings)) from exc

    logger.debug("render: %s (%d bound parameter(s))", sql, len(rendered.params))

    # An execution failure surfaces as itself, never as an empty result set:
    # "the query failed" and "no rows matched" are different answers.
    try:
        rows = await executor.run(sql, *rendered.params)
    except ExecutionError as exc:
        logger.warning("Execution failed: %s", exc)
        raise

    logger.debug(
        "run: finished mode=%s tables=%d rows=%d in %d ms",
        retrieval_mode.value,
        len(tables),
        len(rows),
        _elapsed_ms(stage),
    )

    return Outcome(
        plan=plan_result,
        sql=sql,
        params=rendered.params,
        rows=rows,
        retrieved_tables=retrieved_tables,
        retrieval_mode=retrieval_mode,
    )
