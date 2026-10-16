from typing import NamedTuple

from pydantic import ValidationError

from . import execute, fallback, render
from .config import settings
from .constants import COMPARISON_OPERATORS, NULL_TESTS, PATTERN_OPERATORS
from .db import join_path
from .exceptions import ConfigurationError, ExecutionError, PlanningError, ProviderError
from .llm import complete_json
from .logging_config import get_logger
from .retrieval import SchemaIndex
from .types import MeasureFunction, QueryIntent, QueryPlan, RetrievalMode, SchemaGraph

logger = get_logger("graph")

JOIN_AWARE_INTENTS = (QueryIntent.retrieve, QueryIntent.lookup)

OPERATOR_TEXT = (
    f"a comparison ({', '.join(repr(op) for op in COMPARISON_OPERATORS)}), "
    f"a pattern match ({', '.join(repr(op) for op in PATTERN_OPERATORS)}), "
    f"or a null test ({', '.join(repr(op) for op in NULL_TESTS)})"
)

SYSTEM = f"""\
You produce a QueryPlan for the schema supplied below.

Inputs you are given:
- SCHEMA: tables, columns, data types, keys, foreign keys, operator-written comments.
- KNOWN VALUES: values observed in low-cardinality columns.

Field contracts:
- table: required. The one table that answers the question on its own.
- related_tables: only tables the answer also needs. Every one must have a
  foreign-key path to table within the allowed join depth.
- intent: one of {", ".join(repr(i.value) for i in QueryIntent)}.
  "count" yields one row holding a row count. "aggregate" yields one row per group
  (or one row overall) holding measure_function applied to measure_column.
  "distinct" yields unique combinations of projection columns. "retrieve" yields
  one row per matching record. "lookup" is "retrieve" narrowed by a key or an
  exact value.
- measure_column / measure_function: required together by "aggregate", and
  measure_column must be of a numeric type. measure_function is one of
  {", ".join(repr(m.value) for m in MeasureFunction)}.
- filter_column / filter_value / filter_operator: a predicate on one column.
  filter_operator is {OPERATOR_TEXT}.
  Pattern operators need filter_value; null tests must leave it unset.
- text_search_query / text_search_columns: use these instead of filter_column when
  the question supplies wording to be found inside the content of a text column.
- projection: the columns to return. Leave empty to return every column.
- order_by / descending: optional ordering.
- group_by: the non-measured columns of an aggregate that must be listed per group.
- template: a concise single-sentence English response template (max 200 chars)
  with placeholders in curly braces. Do not include markdown.
  For "count" and "aggregate" the computed number is bound to {{row_count}},
  {{count}}, {{value}}, {{result}} and {{total}}.
  For every intent, {{row_count}} and {{count}} are the number of rows returned,
  and each returned column name is also bound.

Constraints:
- Every table and column must appear in SCHEMA. The plan is checked against the
  catalog and a rejected plan is returned to you with the reason.
- Prefer a value taken from KNOWN VALUES or from the question over an invented one.
- Leave any field you do not need unset.
"""

USER = """\
SCHEMA
{schema}

KNOWN VALUES
{values}

QUESTION
{question}
"""

REPAIR = """\
The previous plan was rejected: {error}

Return a corrected plan. Verify each table and column against SCHEMA before
answering, and read the field contracts above.
"""


class Outcome(NamedTuple):
    """What one completed search produced."""

    plan: QueryPlan
    sql: str
    rows: list[dict]
    retrieved_tables: list[str]
    retrieval_mode: RetrievalMode


def column_scope(plan: QueryPlan, graph: SchemaGraph) -> set[str]:
    """Every column name reachable from the plan's tables."""
    available: set[str] = set()
    for name in plan.tables:
        info = graph.tables.get(name)
        if info is not None:
            available.update(info.column_names)
    return available


def normalise_plan(plan: QueryPlan, graph: SchemaGraph) -> QueryPlan:
    available = column_scope(plan, graph)

    def fix(column: str | None) -> str | None:
        if column is None or column in available:
            return column
        _, dot, tail = column.rpartition(".")
        if dot and tail in available:
            return tail
        return column

    def fix_all(columns: list[str]) -> list[str]:
        return [fix(column) for column in columns]

    def fix_template(tpl: str | None) -> str | None:
        if not tpl:
            return None
        cleaned = " ".join(tpl.strip().split())
        return cleaned[: settings.error_message_max_chars]

    return plan.model_copy(
        update={
            "measure_column": fix(plan.measure_column),
            "filter_column": fix(plan.filter_column),
            "order_by": fix(plan.order_by),
            "projection": fix_all(plan.projection),
            "group_by": fix_all(plan.group_by),
            "text_search_columns": fix_all(plan.text_search_columns),
            "template": fix_template(plan.template),
        }
    )


def with_join_keys(plan: QueryPlan, graph: SchemaGraph) -> QueryPlan:
    if plan.intent not in JOIN_AWARE_INTENTS:
        return plan
    if not plan.related_tables or not plan.projection:
        return plan

    projection = list(plan.projection)
    added = False

    for related in plan.related_tables:
        for fk in graph.foreign_keys:
            if {fk.from_table, fk.to_table} != {plan.table, related}:
                continue
            for column in (fk.from_column, fk.to_column):
                if column not in projection:
                    projection.append(column)
                    added = True

    if not added:
        return plan

    return plan.model_copy(update={"projection": sorted(set(projection), key=projection.index)})


def check_plan(plan: QueryPlan, graph: SchemaGraph) -> str | None:
    if plan.table not in graph.tables:
        return f"table '{plan.table}' does not exist in the schema"

    for related in plan.related_tables:
        if related not in graph.tables:
            return f"related table '{related}' does not exist in the schema"

    available = column_scope(plan, graph)

    for value in (plan.measure_column, plan.filter_column, plan.order_by):
        if value and value not in available:
            return f"column '{value}' does not exist"

    for value in (*plan.projection, *plan.group_by, *plan.text_search_columns):
        if value not in available:
            return f"column '{value}' does not exist"

    if plan.filter_column and not plan.filter_operator:
        return "a filter column requires filter_operator"

    if plan.intent is QueryIntent.aggregate:
        if not plan.measure_column or not plan.measure_function:
            return "an aggregate plan requires measure_column and measure_function"
        numeric_anywhere = any(
            plan.measure_column in graph.tables[name].numeric_columns
            for name in plan.tables
            if name in graph.tables
        )
        if not numeric_anywhere:
            numeric = ", ".join(graph.tables[plan.table].numeric_columns) or "none"
            return f"column '{plan.measure_column}' is not numeric; numeric columns are: {numeric}"

    for related in plan.related_tables:
        if join_path(plan.table, related, graph, settings.max_join_depth) is None:
            return f"no foreign key path connects {plan.table} and {related}"

    return None


def redact(reason: str) -> str:
    if not settings.redact_catalog_in_errors:
        return reason
    return reason.split(";")[0].strip()


async def link(
    question: str, graph: SchemaGraph, index: SchemaIndex | None
) -> tuple[list[str], RetrievalMode, list[str]]:
    """Chooses the tables worth showing the planner, and how they were chosen."""
    if index is not None:
        tables = await index.tables_for(
            question, settings.max_schema_tables, settings.retrieval_min_score
        )
        if tables:
            return tables, RetrievalMode.embedding, tables

    ordered = sorted(graph.tables.values(), key=lambda t: t.row_estimate, reverse=True)
    names = [t.name for t in ordered[: settings.max_schema_tables]]
    return names, RetrievalMode.catalog, []


async def collect_values(graph: SchemaGraph, tables: list[str], cache: dict | None = None) -> dict:
    """Observed values of small-vocabulary columns, memoised per table in cache.

    The samples change only when the catalog is re-read, so recomputing them on
    every request issued one query per eligible column for nothing.
    """
    if not settings.sample_values_enabled:
        return {}

    samples: dict[str, dict[str, list[str]]] = {}
    for table_name in tables:
        if graph.tables.get(table_name) is None:
            continue
        if cache is not None and table_name in cache:
            if cache[table_name]:
                samples[table_name] = cache[table_name]
            continue

        sampled, complete = await _sample_table(graph, table_name)
        # An incomplete sweep may be a transient fault, so it is left uncached
        # rather than remembered as "this table has nothing to show".
        if cache is not None and complete:
            cache[table_name] = sampled
        if sampled:
            samples[table_name] = sampled
    return samples


async def _sample_table(graph: SchemaGraph, table_name: str) -> tuple[dict[str, list[str]], bool]:
    table = graph.tables[table_name]
    ceiling = settings.value_sample_max_length
    samples: dict[str, list[str]] = {}
    complete = True

    for column in table.columns:
        if not column.is_string or column.primary_key:
            continue
        if not 1 < column.distinct_estimate <= settings.value_sample_max:
            continue

        try:
            prepared = execute.prepare(
                f'SELECT DISTINCT "{column.name}" AS value FROM "{table_name}" '
                f'WHERE "{column.name}" IS NOT NULL '
                f"LIMIT {settings.value_sample_limit}",
                graph,
            )
            rows = await execute.run(prepared)
        except Exception as exc:
            logger.debug("Sampling failed for %s.%s: %s", table_name, column.name, exc)
            complete = False
            continue

        values = [
            str(row["value"])
            for row in rows
            if row.get("value") is not None and len(str(row["value"])) <= ceiling
        ]
        if 1 < len(values) <= settings.value_sample_max:
            samples[column.name] = values
    return samples, complete


async def plan(
    question: str,
    graph: SchemaGraph,
    tables: list[str],
    values: dict,
    error: str | None = None,
) -> tuple[QueryPlan | None, str | None]:
    """One planning attempt. Returns (plan, None) or (None, rejection-reason).

    A rejected plan is worth retrying. A provider outage is not, so ConfigurationError
    and ProviderError propagate instead of being reported as a plan defect.
    """
    prompt = USER.format(
        schema=graph.describe(tables, settings.max_schema_columns),
        values=render_values(values),
        question=question,
    )
    if error:
        prompt = f"{REPAIR.format(error=error)}\n\n{prompt}"

    try:
        payload = await complete_json(SYSTEM, prompt)
        result = with_join_keys(normalise_plan(QueryPlan.model_validate(payload), graph), graph)
    except (ConfigurationError, ProviderError):
        raise
    except (PlanningError, ValidationError) as exc:
        return None, " ".join(str(exc).split())[: settings.error_message_max_chars]
    except Exception as exc:
        logger.warning("Planner call failed: %s: %s", type(exc).__name__, exc)
        return None, "a usable plan could not be produced"

    reason = check_plan(result, graph)
    return (None, reason) if reason else (result, None)


async def run_sql(sql: str) -> list[dict]:
    if not sql:
        return []
    try:
        return await execute.run(sql)
    except ExecutionError as exc:
        logger.warning("Execution failed: %s", exc)
        return []


async def vector_fallback(graph: SchemaGraph, tables: list[str], question: str) -> list[dict]:
    return await fallback.search(graph, tables, question)


def render_values(values: dict) -> str:
    if not values:
        return "No low-cardinality columns were available to sample."
    lines = [
        f"{table}.{column}: {', '.join(items[: settings.value_sample_limit])}"
        for table, columns in values.items()
        for column, items in columns.items()
    ]
    return "\n".join(lines)


async def run(
    question: str,
    graph: SchemaGraph,
    index: SchemaIndex | None,
    samples: dict | None = None,
) -> Outcome:
    """link -> collect_values -> plan (-(repair)-> plan) -> render -> run_sql -> fallback.

    Every step is a plain call: the pipeline has one retry point and one optional
    tail, which a workflow framework would not have made any shorter.
    """
    tables, retrieval_mode, retrieved_tables = await link(question, graph, index)
    values = await collect_values(graph, tables, samples)

    plan_result = None
    error = None
    for _ in range(settings.max_plan_attempts):
        plan_result, error = await plan(question, graph, tables, values, error)
        if plan_result is not None:
            break

    if plan_result is None:
        raise PlanningError(redact(error or "No valid plan was produced."))

    try:
        sql = execute.prepare(render.build(plan_result, graph), graph)
    except Exception as exc:
        logger.warning("Render failed: %s", exc)
        raise PlanningError(redact(str(exc))) from exc

    rows = await run_sql(sql)
    if not rows:
        rows = await vector_fallback(graph, tables, question)

    return Outcome(
        plan=plan_result,
        sql=sql,
        rows=rows,
        retrieved_tables=retrieved_tables,
        retrieval_mode=retrieval_mode,
    )
