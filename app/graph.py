import time
from typing import NamedTuple

from pydantic import ValidationError

from . import execute, fallback, render
from .config import settings
from .constants import (
    COMPARISON_OPERATORS,
    COUNT_ALIAS,
    DEFAULT_COMPARISON,
    EQUALITY_ALIASES,
    NULL_TESTS,
    OPERATORS,
    PATTERN_OPERATORS,
)
from .db import join_path
from .exceptions import ConfigurationError, ExecutionError, PlanningError, ProviderError
from .llm import complete_json, prompt_chars, trim
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
You produce a QueryPlan JSON object for the schema supplied below.
The JSON object must include "table" and "intent", plus any needed optional fields.
Never return data results directly; return the plan specification only.

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
- template: a concise single-sentence English response template (max 200 chars)
  with placeholders in curly braces. Do not include markdown.
  For "count" and "aggregate" the computed number is bound to {{row_count}},
  {{count}}, {{value}}, {{result}} and {{total}}.
  For every intent, {{row_count}} and {{count}} are the number of rows returned,
  and each returned column name is also bound.

Constraints:
- Every table and column must appear in SCHEMA. The plan is checked against the
  catalog and a rejected plan is returned to you with the reason.
- Never guess column names not in SCHEMA. For recency or date ordering, select an
  existing date or timestamp column from SCHEMA.
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
answering. If a column was rejected, pick only from the available columns.
Read the field contracts above.
"""


class Outcome(NamedTuple):
    """What one completed search produced."""

    plan: QueryPlan
    sql: str
    rows: list[dict]
    retrieved_tables: list[str]
    retrieval_mode: RetrievalMode


def column_scope(plan: QueryPlan, graph: SchemaGraph) -> set[str]:
    available: set[str] = set()
    for name in plan.tables:
        info = graph.tables.get(name)
        if info is not None:
            available.update(info.column_names)
    return available


def normalise_plan(plan: QueryPlan, graph: SchemaGraph) -> QueryPlan:
    related_tables = list(plan.related_tables)
    available = column_scope(plan, graph)

    if plan.filter_column and plan.filter_column not in available:
        for t_name, t_info in graph.tables.items():
            if (
                t_name != plan.table
                and plan.filter_column in t_info.column_names
                and render.find_relation(plan.table, t_name, graph) is not None
                and t_name not in related_tables
            ):
                related_tables.append(t_name)
                available.update(t_info.column_names)
                break

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
            fix(c)
            for c in columns
            if not (plan.intent is QueryIntent.aggregate and (c == "count" or c == COUNT_ALIAS))
        ]

    def fix_related(names: list[str]) -> list[str]:
        """Keep only tables that can be joined to the base table."""
        kept: list[str] = []
        for name in names:
            candidate = name.split(".", 1)[0]
            if candidate == plan.table or candidate not in graph.tables:
                continue
            if candidate in kept:
                continue
            if render.find_relation(plan.table, candidate, graph) is None:
                continue
            kept.append(candidate)
        return kept

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
        "related_tables": fix_related(related_tables),
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
            return _unknown_column(value, available)

    for value in (*plan.projection, *plan.group_by, *plan.text_search_columns):
        if value not in available:
            return _unknown_column(value, available)

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
            numeric_anywhere = any(
                plan.measure_column in graph.tables[name].numeric_columns
                for name in plan.tables
                if name in graph.tables
            )
            if not numeric_anywhere:
                numeric = ", ".join(graph.tables[plan.table].numeric_columns) or "none"
                return (
                    f"column '{plan.measure_column}' is not numeric; numeric columns are: {numeric}"
                )

    for related in plan.related_tables:
        if join_path(plan.table, related, graph, settings.max_join_depth) is None:
            return f"no foreign key path connects {plan.table} and {related}"

    return None


def _unknown_column(value: str, available: set[str]) -> str:
    """Return error message listing available columns for repair."""
    listing = ", ".join(sorted(available)[: settings.max_schema_columns])
    return f"column '{value}' does not exist; available columns are: {listing}"


def redact(reason: str) -> str:
    if not settings.redact_catalog_in_errors:
        return reason
    return reason.split(";")[0].strip()


def _elapsed_ms(started: float) -> int:
    """Whole milliseconds since a perf_counter reading."""
    return int((time.perf_counter() - started) * 1000)


async def link(
    question: str, graph: SchemaGraph, index: SchemaIndex | None
) -> tuple[list[str], RetrievalMode, list[str], dict[str, float]]:
    if index is not None:
        columns = await index.columns_for(question, settings.retrieval_min_score)
        tables = await index.tables_for(
            question, settings.max_schema_tables, settings.retrieval_min_score
        )
        if tables:
            return tables, RetrievalMode.embedding, tables, columns

    ordered = sorted(graph.tables.values(), key=lambda t: t.row_estimate, reverse=True)
    names = [t.name for t in ordered[: settings.max_schema_tables]]
    return names, RetrievalMode.catalog, [], {}


async def collect_values(
    graph: SchemaGraph,
    tables: list[str],
    cache: dict | None = None,
    relevance: dict | None = None,
) -> dict:
    """Sampled values per table, memoised until the catalog is rebuilt."""
    if not settings.sample_values_enabled:
        return {}

    wanted: set[tuple[str, str]] | None = None
    if relevance:
        ranked = sorted(relevance.items(), key=lambda pair: pair[1], reverse=True)
        wanted = set()
        for qualified, _score in ranked[: settings.retrieval_top_k]:
            table_name, _, column_name = qualified.rpartition(".")
            wanted.add((table_name, column_name))
        if not wanted:
            wanted = None

    samples: dict[str, dict[str, list[str]]] = {}
    for table_name in tables:
        if graph.tables.get(table_name) is None:
            continue
        if cache is not None and table_name in cache:
            if cache[table_name]:
                samples[table_name] = cache[table_name]
            continue

        sampled, complete = await _sample_table(graph, table_name, wanted)
        if cache is not None and complete:
            cache[table_name] = sampled
        if sampled:
            samples[table_name] = sampled
    return samples


def _samplable(_table_name: str, column, row_estimate: int) -> bool:
    """Check if column values are suitable and safe to sample."""
    if column.primary_key or not column.is_string:
        return False

    lowered = column.name.lower()
    if any(lowered == entry.lower() for entry in settings.value_sample_denylist):
        return False
    marker = settings.value_sample_sensitive_marker
    if marker and column.description and marker.lower() in column.description.lower():
        return False

    if row_estimate > 0 and column.distinct_estimate >= row_estimate:
        return False

    return 1 < column.distinct_estimate <= settings.value_sample_max


async def _sample_table(
    graph: SchemaGraph, table_name: str, wanted: set[tuple[str, str]] | None
) -> tuple[dict[str, list[str]], bool]:
    table = graph.tables[table_name]
    ceiling = settings.value_sample_max_length
    samples: dict[str, list[str]] = {}
    complete = True

    for column in table.columns:
        if wanted is not None and (table_name, column.name) not in wanted:
            continue
        if not _samplable(table_name, column, table.row_estimate):
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
            logger.debug("Sampling skipped for %s.%s: %s", table_name, column.name, exc)
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
    attempt: int = 0,
) -> tuple[QueryPlan | None, str | None]:
    """Execute one query planning attempt, with repair prompt if retrying."""
    prompt = build_prompt(question, graph, tables, values)
    if error:
        prompt = f"{REPAIR.format(error=error[: settings.repair_char_budget])}\n\n{prompt}"

    temperature = settings.llm_temperature if attempt == 0 else settings.llm_repair_temperature

    logger.debug(
        "Planning attempt %d: tables=%d prompt_chars=%d sampled_value_tables=%d "
        "temperature=%s repairing=%s",
        attempt + 1,
        len(tables),
        len(prompt),
        len(values),
        temperature,
        error,
    )

    try:
        payload = await complete_json(SYSTEM, prompt, temperature)
        result = with_join_keys(normalise_plan(QueryPlan.model_validate(payload), graph), graph)
    except (ConfigurationError, ProviderError):
        raise
    except (PlanningError, ValidationError) as exc:
        logger.debug("Plan malformed: %s: %s", type(exc).__name__, exc)
        return None, " ".join(str(exc).split())[: settings.error_message_max_chars]
    except Exception as exc:
        logger.warning("Planner call failed: %s: %s", type(exc).__name__, exc)
        return None, "a usable plan could not be produced"

    reason = check_plan(result, graph)
    if reason:
        logger.debug(
            "Plan rejected: %s (intent=%s table=%s)",
            reason,
            result.intent.value,
            result.table,
        )
        return None, reason

    logger.debug(
        "Plan accepted: intent=%s table=%s related=%s measure=%s/%s filter=%s %s %r "
        "projection=%d order_by=%s",
        result.intent.value,
        result.table,
        result.related_tables,
        result.measure_function.value if result.measure_function else None,
        result.measure_column,
        result.filter_column,
        result.filter_operator,
        result.filter_value,
        len(result.projection),
        result.order_by,
    )
    return result, None


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


def build_prompt(question: str, graph: SchemaGraph, tables: list[str], values: dict) -> str:
    """Assemble the planner prompt bounded by LLM context budget."""
    reserve = settings.llm_max_tokens + len(SYSTEM) // settings.llm_chars_per_token
    budget = prompt_chars(reserve)
    overhead = len(USER) + len(question)

    schema = trim(graph.describe(tables, settings.max_schema_columns), budget // 2)
    known = trim(render_values(values), max(budget - len(schema) - overhead, 0))

    return USER.format(schema=schema, values=known, question=question)


async def run(
    question: str,
    graph: SchemaGraph,
    index: SchemaIndex | None,
    samples: dict | None = None,
) -> Outcome:
    """Execute query generation pipeline: link -> collect -> plan -> render -> run."""
    stage = time.perf_counter()
    tables, retrieval_mode, retrieved_tables, relevance = await link(question, graph, index)
    linked_ms = _elapsed_ms(stage)
    logger.debug(
        "link: mode=%s tables=%s scored_columns=%d in %d ms",
        retrieval_mode.value,
        tables,
        len(relevance),
        linked_ms,
    )

    stage = time.perf_counter()
    values = await collect_values(graph, tables, samples, relevance)
    logger.debug(
        "collect_values: %d tables sampled, %d columns known, %d ms",
        len(values),
        sum(len(columns) for columns in values.values()),
        _elapsed_ms(stage),
    )

    plan_result = None
    error = None
    for attempt in range(settings.max_plan_attempts):
        plan_result, error = await plan(question, graph, tables, values, error, attempt)
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
        raise PlanningError(redact(error or "No valid plan was produced."))

    try:
        sql = execute.prepare(render.build(plan_result, graph), graph)
    except Exception as exc:
        logger.warning("Render failed: %s: %s", type(exc).__name__, exc)
        raise PlanningError(redact(str(exc))) from exc

    logger.debug("render: %s", sql)

    rows = await run_sql(sql)
    if rows:
        logger.debug("run_sql: %d rows returned", len(rows))
    else:
        logger.info("run_sql returned no rows; falling back to vector similarity")
        rows = await vector_fallback(graph, tables, question)
        logger.debug("vector_fallback: %d rows returned", len(rows))

    return Outcome(
        plan=plan_result,
        sql=sql,
        rows=rows,
        retrieved_tables=retrieved_tables,
        retrieval_mode=retrieval_mode,
    )
