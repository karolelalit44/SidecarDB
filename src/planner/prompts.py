from dataclasses import dataclass, field

from catalog.models import SchemaGraph
from config.constants import OPERATOR_TEXT, TRUNCATION_MARK
from core.logging import get_logger
from planner.models import MeasureFunction, PlanPolicy, QueryIntent

logger = get_logger("planner.prompts")


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


__all__ = [
    "REPAIR",
    "SYSTEM",
    "USER",
    "PlanRequest",
    "build_prompt",
    "prompt_chars",
    "trim",
]
