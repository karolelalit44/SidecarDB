"""The full pipeline, wired with fakes for the model, database and embedder."""

import pytest

from core.exceptions import (
    ConfigurationError,
    ExecutionError,
    PlanningError,
    ProviderError,
    QueryError,
)
from core.graph import link, plan
from core.retrieval import SchemaIndex
from core.service import SearchService
from core.types import MeasureFunction, QueryIntent, QueryPlan, RetrievalMode
from tests.conftest import FakeDatabase, FakeEmbedder


class FakeLLM:
    """Stands in for the planner client; returns canned plan payloads."""

    def __init__(self) -> None:
        self.plan_calls: list[str] = []
        self.temperatures: list[float | None] = []
        self.responder = None

    async def complete_json(self, system_prompt, user_prompt, temperature=None):
        self.plan_calls.append(user_prompt)
        self.temperatures.append(temperature)
        assert self.responder is not None, "call wired['configure'] before searching"
        return self.responder(user_prompt, temperature)


class FakeExecutor:
    """Records what would have run, and returns canned rows."""

    def __init__(self) -> None:
        self.rows: list[dict] = []
        self.executed: list[tuple[str, tuple]] = []

    def prepare(self, rendered, graph):
        return rendered.sql

    async def run(self, sql, *args):
        self.executed.append((sql, args))
        return self.rows


@pytest.fixture
async def wired(settings, ecommerce, fake_embedder):
    llm = FakeLLM()
    executor = FakeExecutor()
    service = SearchService(
        settings,
        FakeDatabase(),
        fake_embedder,
        llm=llm,
        executor=executor,
    )
    service._graph = ecommerce
    # The fake embedder scores every element identically, so retrieval_top_k
    # must span all of them; otherwise the prompt would only ever surface the
    # first couple of tables.
    settings.retrieval_top_k = len(ecommerce.tables) + sum(
        len(t.columns) for t in ecommerce.tables.values()
    )
    service._index = await SchemaIndex(fake_embedder, settings).build(ecommerce)

    def configure(plan_payload, rows):
        llm.responder = lambda *_args, **_kwargs: plan_payload.model_dump(
            mode="json", exclude_none=True
        )
        executor.rows = rows

    return {"service": service, "configure": configure, "llm": llm, "executor": executor}


async def test_full_pipeline(wired):
    plan_payload = QueryPlan(
        intent=QueryIntent.retrieve,
        table="orders",
        projection=["order_reference", "grand_total"],
        filter_column="order_state",
        filter_operator="=",
        filter_value="shipped",
        template="Shipped order {order_reference} is worth {grand_total}.",
    )
    rows = [{"order_reference": "R-1", "grand_total": 10.0}]
    wired["configure"](plan_payload, rows)

    result = await wired["service"].search("show shipped orders")

    assert result.row_count == 1
    assert result.records == rows
    assert result.answer == "Shipped order R-1 is worth 10.0."
    # The value is bound, never interpolated into the statement text.
    assert '"order_state" = $1' in result.sql
    sql, params = wired["executor"].executed[0]
    assert params == ("shipped",)
    assert "shipped" not in sql


async def test_an_execution_failure_is_reported_as_a_failure(wired):
    """A broken query must never masquerade as an empty result set."""

    class ExplodingExecutor(FakeExecutor):
        async def run(self, sql, *args):
            raise ExecutionError("Query rejected: syntax error")

    wired["service"]._executor = ExplodingExecutor()
    wired["configure"](QueryPlan(intent=QueryIntent.count, table="orders"), [])

    with pytest.raises(ExecutionError):
        await wired["service"].search("how many orders")


async def test_a_plan_without_a_template_falls_back_to_the_configured_message(wired, settings):
    wired["configure"](
        QueryPlan(intent=QueryIntent.retrieve, table="orders", projection=["order_reference"]),
        [{"order_reference": "R-1"}],
    )

    result = await wired["service"].search("show orders")

    assert result.answer == settings.no_summary_message
    assert len(wired["llm"].plan_calls) == 1


async def test_count_plan(wired):
    wired["configure"](QueryPlan(intent=QueryIntent.count, table="orders"), [{"record_count": 2}])

    result = await wired["service"].search("how many orders")

    assert "COUNT(*)" in result.sql
    assert result.records == [{"record_count": 2}]


async def test_aggregate_plan(wired):
    plan_payload = QueryPlan(
        intent=QueryIntent.aggregate,
        table="orders",
        measure_column="grand_total",
        measure_function=MeasureFunction.sum,
    )
    wired["configure"](plan_payload, [{"sum_of_grand_total": 99.0}])

    result = await wired["service"].search("total spend")

    assert 'SUM("grand_total")' in result.sql


async def test_text_search_plan(wired):
    plan_payload = QueryPlan(
        intent=QueryIntent.retrieve,
        table="orders",
        text_search_columns=["order_reference"],
        text_search_query="R-1",
    )
    wired["configure"](plan_payload, [{"order_reference": "R-1"}])

    result = await wired["service"].search("find R-1")

    assert "ILIKE" in result.sql
    _sql, params = wired["executor"].executed[0]
    assert params == ("R-1",)


async def test_empty_result_reports_plainly(wired):
    wired["configure"](QueryPlan(intent=QueryIntent.retrieve, table="orders"), [])

    result = await wired["service"].search("anything")

    assert result.row_count == 0
    assert result.answer == "No matching records were found."


async def test_schema_reaches_the_planner(wired):
    wired["configure"](QueryPlan(intent=QueryIntent.count, table="orders"), [])
    await wired["service"].search("how many orders")

    prompt = wired["llm"].plan_calls[0]
    assert "TABLE orders" in prompt
    assert "order_reference" in prompt
    assert "how many orders" in prompt


async def test_invalid_plan_is_retried_then_fails(wired):
    wired["configure"](QueryPlan(intent=QueryIntent.count, table="ghost_table"), [])

    with pytest.raises(PlanningError):
        await wired["service"].search("how many")


async def test_short_question_is_rejected(wired):
    wired["configure"](QueryPlan(intent=QueryIntent.count, table="orders"), [])
    with pytest.raises(QueryError):
        await wired["service"].search("x")


async def test_long_question_is_rejected(wired, settings):
    wired["configure"](QueryPlan(intent=QueryIntent.count, table="orders"), [])
    with pytest.raises(QueryError):
        await wired["service"].search("a" * (settings.max_query_length + 1))


async def test_risky_question_is_rejected(wired):
    wired["configure"](QueryPlan(intent=QueryIntent.count, table="orders"), [])
    with pytest.raises(QueryError, match="disallowed symbols"):
        await wired["service"].search("how many orders; drop table orders")


async def test_model_decided_template_eliminates_second_api_call(wired):
    plan_payload = QueryPlan(
        intent=QueryIntent.count,
        table="orders",
        template="Found {record_count} orders in the database.",
    )
    wired["configure"](plan_payload, [{"record_count": 5}])

    result = await wired["service"].search("how many orders")

    assert result.answer == "Found 5 orders in the database."
    assert len(wired["llm"].plan_calls) == 1


async def test_count_template_reports_the_count_not_the_row_number(wired):
    """A COUNT returns exactly one row, so {row_count} must be the aggregate itself."""
    plan_payload = QueryPlan(
        intent=QueryIntent.count,
        table="orders",
        template="There are {row_count} matching orders.",
    )
    wired["configure"](plan_payload, [{"record_count": 3}])

    result = await wired["service"].search("how many orders")

    assert result.row_count == 1
    assert result.records == [{"record_count": 3}]
    assert result.answer == "There are 3 matching orders."


async def test_aggregate_template_reports_the_measure(wired):
    plan_payload = QueryPlan(
        intent=QueryIntent.aggregate,
        table="orders",
        measure_column="grand_total",
        measure_function=MeasureFunction.sum,
        template="Total spend was {row_count}.",
    )
    wired["configure"](plan_payload, [{"sum_of_grand_total": 99.0}])

    result = await wired["service"].search("total spend")

    assert result.answer == "Total spend was 99.0."


async def test_retrieve_template_still_counts_rows(wired):
    """For a row-returning intent {row_count} stays the number of rows."""
    plan_payload = QueryPlan(
        intent=QueryIntent.retrieve,
        table="orders",
        projection=["order_reference"],
        template="Returned {row_count} orders.",
    )
    wired["configure"](plan_payload, [{"order_reference": "R-1"}, {"order_reference": "R-2"}])

    result = await wired["service"].search("show orders")

    assert result.answer == "Returned 2 orders."


async def test_response_reports_the_retrieval_mode(wired):
    """Embedding fallback to catalog order must be visible, not silent."""
    wired["configure"](QueryPlan(intent=QueryIntent.count, table="orders"), [])

    result = await wired["service"].search("how many orders")

    assert result.retrieval_mode == RetrievalMode.embedding
    assert result.retrieved_tables


async def test_link_tolerates_none_index(ecommerce, settings):
    """When index is None, link should fall back to catalog mode without error."""
    tables, mode, retrieved = await link("show orders", ecommerce, None, settings)
    assert mode == RetrievalMode.catalog
    assert len(tables) > 0
    assert retrieved == []


async def test_link_uses_the_index_when_available(ecommerce, settings, fake_embedder):
    index = await SchemaIndex(fake_embedder, settings).build(ecommerce)
    tables, mode, retrieved = await link("show orders", ecommerce, index, settings)
    assert mode == RetrievalMode.embedding
    assert tables
    assert retrieved == tables


async def test_plan_reraises_configuration_error(ecommerce, settings):
    """ConfigurationError should fail fast and not be caught as a bad plan."""

    class MisconfiguredLLM:
        async def complete_json(self, *_args, **_kwargs):
            raise ConfigurationError("LLM API key not configured")

    with pytest.raises(ConfigurationError):
        await plan("how many orders", ecommerce, ["orders"], settings, MisconfiguredLLM())


async def test_plan_reraises_provider_error(ecommerce, settings):
    """A provider outage must not be retried as if it were a bad plan."""
    calls = []

    class OutageLLM:
        async def complete_json(self, *_args, **_kwargs):
            calls.append(1)
            raise ProviderError("The model provider could not be reached.")

    with pytest.raises(ProviderError):
        await plan("how many orders", ecommerce, ["orders"], settings, OutageLLM())
    assert len(calls) == 1


async def test_rejected_plan_is_retried(wired):
    """A plan naming a table that does not exist comes back for repair."""
    calls = wired["llm"]

    class GhostLLM:
        async def complete_json(self, system_prompt, user_prompt, temperature=None):
            calls.plan_calls.append(user_prompt)
            return {"intent": "count", "table": "ghost_table"}

    wired["service"]._llm = GhostLLM()

    with pytest.raises(PlanningError):
        await wired["service"].search("how many")

    assert len(calls.plan_calls) > 1, "the repair loop must issue more than one planner call"
    assert "rejected" in calls.plan_calls[-1], "the later attempt must carry the rejection reason"


async def test_repair_attempts_sample_at_a_different_temperature(ecommerce, settings):
    """A repair that cannot differ from the reply it repairs is not a repair.

    At the configured sampling temperature the model reproduces a rejected plan
    verbatim, so retrying costs the full model latency to reach the same
    failure. The repair attempts must therefore sample.
    """

    class GhostLLM:
        def __init__(self) -> None:
            self.temperatures: list[float | None] = []

        async def complete_json(self, _system_prompt, _user_prompt, temperature=None):
            self.temperatures.append(temperature)
            return {"intent": "count", "table": "ghost_table"}

    llm = GhostLLM()
    service = SearchService(settings, FakeDatabase(), FakeEmbedder(), llm=llm)
    service._graph = ecommerce

    with pytest.raises(PlanningError):
        await service.search("how many")

    assert llm.temperatures, "the planner must have been called"
    first = llm.temperatures[0]
    assert first == settings.llm_temperature, "the first attempt is not a repair"
    assert settings.llm_repair_temperature != first, "the repair must sample differently"
    for later in llm.temperatures[1:]:
        assert later == settings.llm_repair_temperature


async def test_first_plan_attempt_is_not_a_repair(ecommerce, settings):
    """The initial attempt runs at the configured temperature."""
    seen = []

    class RecordingLLM:
        async def complete_json(self, _system_prompt, _user_prompt, temperature=None):
            seen.append(temperature)
            return {"intent": "count", "table": "orders"}

    result, error = await plan("how many orders", ecommerce, ["orders"], settings, RecordingLLM())

    assert error is None
    assert result is not None
    assert seen == [settings.llm_temperature]
