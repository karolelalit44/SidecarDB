import pytest

import app.answer as answer_module
import app.execute as execute_module
import app.graph as graph_module
from app.config import settings
from app.exceptions import PlanningError, QueryError
from app.service import SearchService
from app.types import MeasureFunction, QueryIntent, QueryPlan


class FakeResult:
    def __init__(self, text):
        self.content = text


class FakeChat:
    """Stands in for the chat model. The planner now asks for raw JSON and parses
    it itself, so the fake returns a JSON string rather than a tool call."""

    def __init__(self, plan, text="Two orders were found."):
        self.plan = plan
        self._text = text
        self.plan_calls = []
        self.answer_calls = []

    async def ainvoke(self, messages, **_kwargs):
        joined = " ".join(str(m) for m in messages)
        if "JSON object" in joined and "rows" not in joined.lower():
            self.plan_calls.append(messages)
            return FakeResult(self.plan.model_dump_json(exclude_none=True))
        self.answer_calls.append(messages)
        return FakeResult(self._text)


def _json_chat(chat):
    async def fake_complete_json(_system_prompt, user_prompt, _temperature=None):
        chat.plan_calls.append(user_prompt)
        return chat.plan.model_dump(mode="json", exclude_none=True)

    return fake_complete_json


@pytest.fixture
async def wired(monkeypatch, ecommerce):
    state = {"rows": [], "executed": []}

    async def fake_run(sql):
        state["executed"].append(sql)
        return state["rows"]

    monkeypatch.setattr(execute_module, "run", fake_run)

    svc = SearchService()
    svc._graph = ecommerce

    from app.retrieval import SchemaIndex

    svc._index = await SchemaIndex().build(ecommerce)

    def configure(plan, rows, text="Two orders were found."):
        chat = FakeChat(plan, text)
        monkeypatch.setattr(graph_module, "complete_json", _json_chat(chat))
        monkeypatch.setattr(answer_module, "chat_model", lambda: chat)
        state["rows"] = rows
        state["chat"] = chat

    return {"service": svc, "configure": configure, "state": state}


async def test_full_pipeline(wired):
    plan = QueryPlan(
        intent=QueryIntent.retrieve,
        table="orders",
        projection=["order_reference", "grand_total"],
        filter_column="order_state",
        filter_operator="=",
        filter_value="shipped",
    )
    rows = [{"order_reference": "R-1", "grand_total": 10.0}]
    wired["configure"](plan, rows)

    result = await wired["service"].search("show shipped orders")

    assert result.row_count == 1
    assert result.records == rows
    assert result.answer == "Two orders were found."
    assert "\"order_state\" = 'shipped'" in result.sql


async def test_count_plan(wired):
    wired["configure"](QueryPlan(intent=QueryIntent.count, table="orders"), [{"record_count": 2}])

    result = await wired["service"].search("how many orders")

    assert "COUNT(*)" in result.sql
    assert result.records == [{"record_count": 2}]


async def test_aggregate_plan(wired):
    plan = QueryPlan(
        intent=QueryIntent.aggregate,
        table="orders",
        measure_column="grand_total",
        measure_function=MeasureFunction.sum,
    )
    wired["configure"](plan, [{"sum_of_grand_total": 99.0}])

    result = await wired["service"].search("total spend")

    assert 'SUM("grand_total")' in result.sql


async def test_related_tables_are_joined(wired):
    plan = QueryPlan(
        intent=QueryIntent.retrieve,
        table="orders",
        related_tables=["customers"],
        projection=["order_reference"],
    )
    wired["configure"](plan, [{"order_reference": "R-1"}])

    result = await wired["service"].search("orders with customers")

    assert 'JOIN "customers"' in result.sql


async def test_text_search_plan(wired):
    plan = QueryPlan(
        intent=QueryIntent.retrieve,
        table="orders",
        text_search_columns=["order_reference"],
        text_search_query="R-1",
    )
    wired["configure"](plan, [{"order_reference": "R-1"}])

    result = await wired["service"].search("find R-1")

    assert "ILIKE" in result.sql


async def test_empty_result_reports_plainly(wired):
    wired["configure"](QueryPlan(intent=QueryIntent.retrieve, table="orders"), [])

    result = await wired["service"].search("anything")

    assert result.row_count == 0
    assert result.answer == "No matching records were found."


async def test_schema_reaches_the_planner(wired):
    wired["configure"](QueryPlan(intent=QueryIntent.count, table="orders"), [])
    await wired["service"].search("how many orders")

    prompt = str(wired["state"]["chat"].plan_calls[0])
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


async def test_long_question_is_rejected(wired):
    from app.config import settings

    wired["configure"](QueryPlan(intent=QueryIntent.count, table="orders"), [])
    with pytest.raises(QueryError):
        await wired["service"].search("a" * (settings.max_query_length + 1))


async def test_risky_question_is_rejected(wired):
    wired["configure"](QueryPlan(intent=QueryIntent.count, table="orders"), [])
    with pytest.raises(QueryError, match="disallowed symbols"):
        await wired["service"].search("how many orders; drop table orders")


async def test_model_decided_template_eliminates_second_api_call(wired):
    plan = QueryPlan(
        intent=QueryIntent.count,
        table="orders",
        template="Found {record_count} orders in the database.",
    )
    wired["configure"](plan, [{"record_count": 5}])

    result = await wired["service"].search("how many orders")

    assert result.answer == "Found 5 orders in the database."
    assert len(wired["state"]["chat"].answer_calls) == 0


async def test_count_template_reports_the_count_not_the_row_number(wired):
    """A COUNT returns exactly one row, so {row_count} must be the aggregate itself."""
    plan = QueryPlan(
        intent=QueryIntent.count,
        table="orders",
        template="There are {row_count} matching orders.",
    )
    wired["configure"](plan, [{"record_count": 3}])

    result = await wired["service"].search("how many orders")

    assert result.row_count == 1
    assert result.records == [{"record_count": 3}]
    assert result.answer == "There are 3 matching orders."


async def test_aggregate_template_reports_the_measure(wired):
    plan = QueryPlan(
        intent=QueryIntent.aggregate,
        table="orders",
        measure_column="grand_total",
        measure_function=MeasureFunction.sum,
        template="Total spend was {row_count}.",
    )
    wired["configure"](plan, [{"sum_of_grand_total": 99.0}])

    result = await wired["service"].search("total spend")

    assert result.answer == "Total spend was 99.0."


async def test_retrieve_template_still_counts_rows(wired):
    """For a row-returning intent {row_count} stays the number of rows."""
    plan = QueryPlan(
        intent=QueryIntent.retrieve,
        table="orders",
        projection=["order_reference"],
        template="Returned {row_count} orders.",
    )
    wired["configure"](plan, [{"order_reference": "R-1"}, {"order_reference": "R-2"}])

    result = await wired["service"].search("show orders")

    assert result.answer == "Returned 2 orders."


async def test_link_tolerates_none_index(ecommerce):
    """When index is None, link should fall back to catalog mode without error."""
    from app.graph import link
    from app.types import RetrievalMode

    tables, mode, retrieved, relevance = await link("show orders", ecommerce, None)
    assert mode == RetrievalMode.catalog
    assert len(tables) > 0
    assert retrieved == []
    assert relevance == {}


async def test_collect_values_tolerates_exceptions(ecommerce, monkeypatch):
    """Errors during value sampling should be swallowed without crashing the pipeline."""
    from app.graph import collect_values

    async def failing_run(_sql):
        raise RuntimeError("Database connection timeout")

    monkeypatch.setattr(execute_module, "run", failing_run)
    samples = await collect_values(ecommerce, ["orders"])
    assert isinstance(samples, dict)


async def test_collect_values_respect_column_relevance(ecommerce, monkeypatch):
    """Only the columns the question is about are worth a query each."""
    from app.graph import collect_values

    queries = []

    async def counting_run(sql):
        queries.append(sql)
        return [{"value": "a"}, {"value": "b"}]

    monkeypatch.setattr(execute_module, "run", counting_run)

    samples = await collect_values(
        ecommerce,
        ["orders"],
        None,
        {"orders.order_state": 0.9, "customers.city": 0.8},
    )

    assert set(samples.get("orders", {})) == {"order_state"}
    assert len(queries) == 1
    assert "order_state" in queries[0]


async def test_collect_values_fall_back_when_nothing_is_relevant(ecommerce, monkeypatch):
    """A weak embedding match must not strip the planner of all vocabulary."""
    from app.graph import collect_values

    async def fake_run(_sql):
        return [{"value": "a"}, {"value": "b"}]

    monkeypatch.setattr(execute_module, "run", fake_run)
    samples = await collect_values(ecommerce, ["orders"], None, {})
    assert samples


async def test_collect_values_are_cached_per_table(ecommerce, monkeypatch):
    from app.graph import collect_values

    queries = []

    async def counting_run(sql):
        queries.append(sql)
        return [{"value": "a"}, {"value": "b"}]

    monkeypatch.setattr(execute_module, "run", counting_run)

    cache = {}
    first = await collect_values(ecommerce, ["orders"], cache)
    assert queries, "first sweep must query"
    assert cache

    queries.clear()
    second = await collect_values(ecommerce, ["orders"], cache)
    assert queries == [], "second sweep must issue no queries"
    assert second == first


async def test_incomplete_sampling_is_not_cached(ecommerce, monkeypatch):
    """A failed column must not be remembered as 'this table has nothing to show'."""
    from app.graph import collect_values

    calls = []

    async def flaky_run(_sql):
        calls.append(1)
        raise RuntimeError("transient")

    monkeypatch.setattr(execute_module, "run", flaky_run)

    cache = {}
    await collect_values(ecommerce, ["orders"], cache)
    first = len(calls)
    await collect_values(ecommerce, ["orders"], cache)
    assert len(calls) > first, "a failed sweep must be retried, not cached"


async def test_plan_reraises_configuration_error(ecommerce, monkeypatch):
    """ConfigurationError should fail fast and not be caught by generic error handler."""
    from app.exceptions import ConfigurationError
    from app.graph import plan

    async def misconfigured_complete(*_args, **_kwargs):
        raise ConfigurationError("LLM API key not configured")

    monkeypatch.setattr(graph_module, "complete_json", misconfigured_complete)

    with pytest.raises(ConfigurationError):
        await plan("how many orders", ecommerce, ["orders"], {})


async def test_plan_reraises_provider_error(ecommerce, monkeypatch):
    """A provider outage must not be retried as if it were a bad plan."""
    from app.exceptions import ProviderError
    from app.graph import plan

    calls = []

    async def failing_complete(*_args, **_kwargs):
        calls.append(1)
        raise ProviderError("The model provider could not be reached.")

    monkeypatch.setattr(graph_module, "complete_json", failing_complete)

    with pytest.raises(ProviderError):
        await plan("how many orders", ecommerce, ["orders"], {})
    assert len(calls) == 1


async def test_rejected_plan_is_retried(wired, monkeypatch):
    """A plan naming a table that does not exist comes back for repair."""
    calls = []

    async def always_ghost(_system_prompt, user_prompt, _temperature=None):
        calls.append(user_prompt)
        return {"intent": "count", "table": "ghost_table"}

    monkeypatch.setattr(graph_module, "complete_json", always_ghost)

    with pytest.raises(PlanningError):
        await wired["service"].search("how many")

    assert len(calls) > 1, "the repair loop must issue more than one planner call"
    assert "rejected" in calls[-1], "the later attempt must carry the rejection reason"


async def test_repair_attempts_sample_at_a_different_temperature(wired, monkeypatch):
    """A repair that cannot differ from the reply it repairs is not a repair.

    At the configured sampling temperature the model reproduces a rejected plan
    verbatim, so retrying costs the full model latency to reach the same
    failure. The repair attempts must therefore sample.
    """
    temperatures = []

    async def always_ghost(_system_prompt, user_prompt, temperature=None):
        temperatures.append(temperature)
        return {"intent": "count", "table": "ghost_table"}

    monkeypatch.setattr(graph_module, "complete_json", always_ghost)

    with pytest.raises(PlanningError):
        await wired["service"].search("how many")

    assert temperatures, "the planner must have been called"
    first = temperatures[0]
    assert first == settings.llm_temperature, "the first attempt is not a repair"
    assert settings.llm_repair_temperature != first, "the repair must sample differently"
    for later in temperatures[1:]:
        assert later == settings.llm_repair_temperature


async def test_first_plan_attempt_is_not_a_repair(ecommerce, monkeypatch):
    """The initial attempt runs at the configured temperature."""
    from app.graph import plan

    seen = []

    async def record_temperature(_system_prompt, _user_prompt, temperature=None):
        seen.append(temperature)
        return {"intent": "count", "table": "orders"}

    monkeypatch.setattr(graph_module, "complete_json", record_temperature)

    result, error = await plan("how many orders", ecommerce, ["orders"], {})

    assert error is None
    assert result is not None
    assert seen == [settings.llm_temperature]
