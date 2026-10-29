from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING
from unicodedata import category

from pydantic import BaseModel, Field, ValidationError, field_validator

from core.catalog import SchemaGraph
from core.constants import (
    ERROR_MESSAGE_MAX_CHARS,
    LLM_CHARS_PER_TOKEN,
    LLM_CONTEXT_TOKENS,
    LLM_MAX_TOKENS,
    LLM_REPAIR_TEMPERATURE,
    LLM_TEMPERATURE,
    MAX_PLAN_ATTEMPTS,
    MAX_SCHEMA_COLUMNS,
    MAX_SCHEMA_TABLES,
    REDACT_CATALOG_IN_ERRORS,
    REPAIR_CHAR_BUDGET,
    RETRIEVAL_MIN_SCORE,
    RETRIEVAL_SCORE_PRECISION,
    RETRIEVAL_TOP_K,
)
from core.exceptions import ConfigurationError, PlanningError, ProviderError
from core.retrieval import RetrievalMode
from utils.logging_config import get_logger

if TYPE_CHECKING:
    from core.capabilities import Planner

logger = get_logger("plan")


class QueryIntent(StrEnum):
    count = "count"  # type: ignore[assignment]
    aggregate = "aggregate"
    retrieve = "retrieve"
    distinct = "distinct"


class MeasureFunction(StrEnum):
    sum = "sum"
    avg = "avg"
    min = "min"
    max = "max"
    count = "count"  # type: ignore[assignment]

    @property
    def needs_column(self) -> bool:
        return self is not MeasureFunction.count


class QueryPlan(BaseModel):
    intent: QueryIntent
    table: str
    measure_column: str | None = None
    measure_function: MeasureFunction | None = None
    filter_column: str | None = None
    filter_value: str | None = None
    filter_operator: str | None = None
    projection: list[str] = Field(default_factory=list)
    order_by: str | None = None
    descending: bool = False
    group_by: list[str] = Field(default_factory=list)
    text_search_columns: list[str] = Field(default_factory=list)
    text_search_query: str | None = None
    template: str | None = Field(default=None, max_length=200)

    @field_validator("projection", "group_by", "text_search_columns", mode="before")
    @classmethod
    def _coerce_list(cls, value: object) -> object:
        if value is None:
            return []
        if isinstance(value, str):
            if not value.strip():
                return []
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    @field_validator("order_by", mode="before")
    @classmethod
    def _coerce_order_by(cls, value: object) -> object:
        if isinstance(value, list):
            return str(value[0]) if value else None
        return value


@dataclass(frozen=True, slots=True)
class PlanPolicy:
    llm_temperature: float = LLM_TEMPERATURE
    llm_repair_temperature: float = LLM_REPAIR_TEMPERATURE
    repair_char_budget: int = REPAIR_CHAR_BUDGET
    max_plan_attempts: int = MAX_PLAN_ATTEMPTS
    error_message_max_chars: int = ERROR_MESSAGE_MAX_CHARS
    llm_max_tokens: int = LLM_MAX_TOKENS
    llm_chars_per_token: int = LLM_CHARS_PER_TOKEN
    llm_context_tokens: int = LLM_CONTEXT_TOKENS
    max_schema_tables: int = MAX_SCHEMA_TABLES
    max_schema_columns: int = MAX_SCHEMA_COLUMNS
    retrieval_top_k: int = RETRIEVAL_TOP_K
    retrieval_min_score: float = RETRIEVAL_MIN_SCORE
    retrieval_score_precision: int = RETRIEVAL_SCORE_PRECISION


@dataclass(frozen=True, slots=True)
class Statement:
    sql: str
    params: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class Outcome:
    plan: QueryPlan
    statement: Statement
    rows: list[dict]
    retrieved_tables: list[str]
    retrieval_mode: RetrievalMode


COMPARISON_OPERATORS = ("=", "!=", "<>", "<", "<=", ">", ">=")
PATTERN_OPERATORS = ("LIKE", "ILIKE")
NULL_TESTS = ("IS NULL", "IS NOT NULL")
OPERATORS = (*COMPARISON_OPERATORS, *PATTERN_OPERATORS, *NULL_TESTS)

DEFAULT_COMPARISON = COMPARISON_OPERATORS[0]

EQUALITY_ALIASES = ("==",)

COUNT_ALIAS = "record_count"

WILDCARD = "*"

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


def column_scope(plan: QueryPlan, graph: SchemaGraph) -> set[str]:
    info = graph.tables.get(plan.table)
    return set(info.column_names) if info is not None else set()


def normalise_plan(plan: QueryPlan, graph: SchemaGraph, policy: PlanPolicy) -> QueryPlan:
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
        return cleaned[: policy.error_message_max_chars]

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
            descending = False

    if plan.intent is QueryIntent.count:
        order_by = None
        descending = False

    filter_operator = plan.filter_operator
    if filter_operator in EQUALITY_ALIASES or (
        filter_operator is None and plan.filter_column and plan.filter_value is not None
    ):
        filter_operator = DEFAULT_COMPARISON

    filter_column = fix(plan.filter_column)
    group_by = fix_all(plan.group_by) if plan.intent is QueryIntent.aggregate else []

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


def check_plan(plan: QueryPlan, graph: SchemaGraph, policy: PlanPolicy) -> str | None:
    if plan.table not in graph.tables:
        return f"table '{plan.table}' does not exist in the schema"

    available = column_scope(plan, graph)

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


def _unknown_column(value: str, available: set[str], policy: PlanPolicy) -> str:
    listing = ", ".join(sorted(available)[: policy.max_schema_columns])
    return f"column '{value}' does not exist; available columns are: {listing}"


def redact(reason: str) -> str:
    if not REDACT_CATALOG_IN_ERRORS:
        return reason
    return reason.split(";")[0].strip()


def prompt_chars(reserve: int, policy: PlanPolicy) -> int:
    return max(policy.llm_context_tokens - reserve, 0) * policy.llm_chars_per_token


def trim(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    if limit <= len(TRUNCATION_MARK):
        return text[:limit]
    return text[: limit - len(TRUNCATION_MARK)] + TRUNCATION_MARK


def build_prompt(question: str, graph: SchemaGraph, tables: list[str], policy: PlanPolicy) -> str:
    reserve = policy.llm_max_tokens + len(SYSTEM) // policy.llm_chars_per_token
    budget = prompt_chars(reserve, policy)
    schema_overhead = len(USER) - len("{schema}") - len("{question}")

    schema = trim(graph.describe(tables, policy.max_schema_columns), budget - schema_overhead)

    return USER.format(schema=schema, question=question)


async def plan(
    question: str,
    graph: SchemaGraph,
    tables: list[str],
    policy: PlanPolicy,
    llm: "Planner",
    error: str | None = None,
    attempt: int = 0,
) -> tuple[QueryPlan | None, str | None]:
    prompt = build_prompt(question, graph, tables, policy)
    if error:
        prompt = f"{REPAIR.format(error=error[: policy.repair_char_budget])}\n\n{prompt}"

    temperature = policy.llm_temperature if attempt == 0 else policy.llm_repair_temperature

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
        result = normalise_plan(QueryPlan.model_validate(payload), graph, policy)
    except (ConfigurationError, ProviderError):
        raise
    except (PlanningError, ValidationError) as exc:
        logger.debug("Plan malformed: %s: %s", type(exc).__name__, exc)
        return None, " ".join(str(exc).split())[: policy.error_message_max_chars]
    except Exception as exc:
        logger.warning("Planner call failed: %s: %s", type(exc).__name__, exc)
        return None, "a usable plan could not be produced"

    reason = check_plan(result, graph, policy)
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


_DISALLOWED_SUBSTRINGS = ("--", "/*", "*/", ";", "\0")
_ALLOWED_CONTROL = ("\t", "\n", "\r")
_BLOCKED_CATEGORIES = ("Cc", "Cf", "Cs")


def check_question(text: str) -> str | None:
    if not text:
        return "The question is empty."

    for marker in _DISALLOWED_SUBSTRINGS:
        if marker in text:
            return "The question contains disallowed symbols or comment characters."

    for ch in text:
        if ch in _ALLOWED_CONTROL:
            continue
        if category(ch) in _BLOCKED_CATEGORIES:
            return "The question contains unsupported control or formatting characters."

    if not any(ch.isalnum() for ch in text):
        return "The question must contain at least one alphanumeric character."

    return None
