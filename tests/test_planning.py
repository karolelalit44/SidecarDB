import pytest

from catalog.models import ColumnInfo, SchemaGraph, TableInfo
from core.exceptions import PlanningError
from planner.generator import generate_plan
from planner.models import PlanPolicy, QueryIntent


class ScriptedPlanner:
    def __init__(self, responses: list[dict]) -> None:
        self.responses = list(responses)
        self.call_count = 0
        self.recorded_temperatures: list[float | None] = []

    async def complete_json(
        self, system_prompt: str, user_prompt: str, temperature: float | None = None
    ) -> dict:
        self.recorded_temperatures.append(temperature)
        if self.call_count < len(self.responses):
            resp = self.responses[self.call_count]
            self.call_count += 1
            return resp
        raise RuntimeError("No more scripted responses.")


@pytest.fixture
def sample_graph() -> SchemaGraph:
    return SchemaGraph(
        tables={
            "orders": TableInfo(
                name="orders",
                columns=[
                    ColumnInfo(name="id", data_type="integer", type_category="numeric"),
                    ColumnInfo(name="amount", data_type="numeric", type_category="numeric"),
                    ColumnInfo(name="status", data_type="text", type_category="string"),
                ],
            )
        }
    )


@pytest.mark.asyncio
async def test_generate_plan_succeeds_first_attempt(sample_graph):
    planner = ScriptedPlanner(
        responses=[
            {
                "intent": "retrieve",
                "table": "orders",
                "projection": ["id", "amount"],
                "template": "Found {row_count} orders.",
            }
        ]
    )

    plan = await generate_plan(
        question="Show me all orders",
        graph=sample_graph,
        tables=["orders"],
        planner=planner,
    )

    assert plan.table == "orders"
    assert plan.intent == QueryIntent.retrieve
    assert plan.projection == ["id", "amount"]
    assert planner.call_count == 1
    assert planner.recorded_temperatures == [PlanPolicy().llm_temperature]


@pytest.mark.asyncio
async def test_generate_plan_repairs_invalid_plan(sample_graph):
    planner = ScriptedPlanner(
        responses=[
            {
                "intent": "retrieve",
                "table": "nonexistent_table",
                "template": "Invalid.",
            },
            {
                "intent": "retrieve",
                "table": "orders",
                "projection": ["id"],
                "template": "Found {row_count} orders.",
            },
        ]
    )

    policy = PlanPolicy(max_plan_attempts=2)
    plan = await generate_plan(
        question="Show me all orders",
        graph=sample_graph,
        tables=["orders"],
        planner=planner,
        policy=policy,
    )

    assert plan.table == "orders"
    assert planner.call_count == 2
    assert planner.recorded_temperatures[0] == policy.llm_temperature
    assert planner.recorded_temperatures[1] == policy.llm_repair_temperature


@pytest.mark.asyncio
async def test_generate_plan_fails_after_max_attempts(sample_graph):
    planner = ScriptedPlanner(
        responses=[
            {"intent": "retrieve", "table": "bad1", "template": "fail"},
            {"intent": "retrieve", "table": "bad2", "template": "fail"},
        ]
    )

    policy = PlanPolicy(max_plan_attempts=2)
    with pytest.raises(PlanningError):
        await generate_plan(
            question="Show orders",
            graph=sample_graph,
            tables=["orders"],
            planner=planner,
            policy=policy,
        )

    assert planner.call_count == 2


@pytest.mark.asyncio
async def test_generate_plan_with_joins():
    graph = SchemaGraph(
        tables={
            "orders": TableInfo(
                name="orders",
                columns=[
                    ColumnInfo(name="id", data_type="integer"),
                    ColumnInfo(name="user_id", data_type="integer"),
                ],
            ),
            "users": TableInfo(
                name="users",
                columns=[
                    ColumnInfo(name="id", data_type="integer"),
                    ColumnInfo(name="name", data_type="text"),
                ],
            ),
        }
    )
    planner = ScriptedPlanner(
        responses=[
            {
                "intent": "retrieve",
                "table": "orders",
                "joins": [
                    {
                        "table": "orders",
                        "on_column": "user_id",
                        "target_table": "users",
                        "target_column": "id",
                        "join_type": "inner",
                    }
                ],
                "projection": ["orders.id", "users.name"],
                "filter_column": "users.name",
                "filter_operator": "=",
                "filter_value": "Alice",
                "template": "Found {row_count} orders.",
            }
        ]
    )

    plan = await generate_plan(
        question="Show orders for Alice",
        graph=graph,
        tables=["orders", "users"],
        planner=planner,
    )
    assert plan.table == "orders"
    assert len(plan.joins) == 1
    assert plan.joins[0].target_table == "users"
    assert plan.filter_column == "users.name"
