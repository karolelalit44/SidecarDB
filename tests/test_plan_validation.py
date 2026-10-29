from core.graph import check_plan, normalise_plan, redact
from core.types import MeasureFunction, QueryIntent, QueryPlan, SchemaGraph


def test_valid_plan_is_accepted(ecommerce, settings):
    plan = QueryPlan(intent=QueryIntent.count, table="orders")
    assert check_plan(plan, ecommerce, settings) is None


def test_unknown_table_is_rejected(ecommerce, settings):
    assert check_plan(QueryPlan(intent=QueryIntent.count, table="ghost"), ecommerce, settings)


def test_case_mismatch_is_rejected(ecommerce, settings):
    assert check_plan(QueryPlan(intent=QueryIntent.count, table="Orders"), ecommerce, settings)


def test_unknown_filter_column_is_rejected(ecommerce, settings):
    plan = QueryPlan(
        intent=QueryIntent.retrieve, table="orders", filter_column="ghost", filter_value="x"
    )
    assert check_plan(plan, ecommerce, settings)


def test_unknown_projection_column_is_rejected(ecommerce, settings):
    plan = QueryPlan(intent=QueryIntent.retrieve, table="orders", projection=["ghost"])
    assert check_plan(plan, ecommerce, settings)


def test_unknown_order_column_is_rejected(ecommerce, settings):
    plan = QueryPlan(intent=QueryIntent.retrieve, table="orders", order_by="ghost")
    assert check_plan(plan, ecommerce, settings)


def test_unknown_group_column_is_rejected(ecommerce, settings):
    plan = QueryPlan(intent=QueryIntent.retrieve, table="orders", group_by=["ghost"])
    assert check_plan(plan, ecommerce, settings)


def test_unknown_text_search_column_is_rejected(ecommerce, settings):
    plan = QueryPlan(
        intent=QueryIntent.retrieve,
        table="orders",
        text_search_columns=["ghost"],
        text_search_query="x",
    )
    assert check_plan(plan, ecommerce, settings)


def test_aggregate_requires_measure(ecommerce, settings):
    assert check_plan(QueryPlan(intent=QueryIntent.aggregate, table="orders"), ecommerce, settings)


def test_aggregate_rejects_non_numeric(ecommerce, settings):
    plan = QueryPlan(
        intent=QueryIntent.aggregate,
        table="orders",
        measure_column="order_reference",
        measure_function=MeasureFunction.sum,
    )
    reason = check_plan(plan, ecommerce, settings)
    assert reason and "not numeric" in reason


def test_aggregate_on_numeric_is_accepted(ecommerce, settings):
    plan = QueryPlan(
        intent=QueryIntent.aggregate,
        table="orders",
        measure_column="grand_total",
        measure_function=MeasureFunction.sum,
    )
    assert check_plan(plan, ecommerce, settings) is None


def test_every_table_in_every_fixture_validates(ecommerce, healthcare, library, settings):
    for graph in (ecommerce, healthcare, library):
        for name in graph.names:
            assert (
                check_plan(QueryPlan(intent=QueryIntent.count, table=name), graph, settings) is None
            )


def test_qualified_columns_are_normalised(ecommerce, settings):
    """Models habitually write table.column; the prefix is stripped, not rejected."""
    plan = QueryPlan(
        intent=QueryIntent.retrieve,
        table="orders",
        projection=["orders.order_reference", "orders.grand_total"],
        filter_column="orders.order_state",
        filter_value="shipped",
        order_by="orders.placed_at",
    )
    fixed = normalise_plan(plan, ecommerce, settings)

    assert fixed.projection == ["order_reference", "grand_total"]
    assert fixed.filter_column == "order_state"
    assert fixed.order_by == "placed_at"


def test_wrong_table_prefix_is_not_stripped(ecommerce, settings):
    plan = QueryPlan(
        intent=QueryIntent.retrieve,
        table="orders",
        projection=["customers.display_name"],
    )
    fixed = normalise_plan(plan, ecommerce, settings)
    assert fixed.projection == ["customers.display_name"]
    assert check_plan(fixed, ecommerce, settings)


def test_a_measure_outside_an_aggregate_is_cleared(ecommerce, settings):
    plan = QueryPlan(
        intent=QueryIntent.retrieve,
        table="orders",
        measure_column="grand_total",
        measure_function=MeasureFunction.sum,
    )
    fixed = normalise_plan(plan, ecommerce, settings)
    assert fixed.measure_column is None
    assert fixed.measure_function is None


def test_a_measure_on_an_aggregate_is_kept(ecommerce, settings):
    plan = QueryPlan(
        intent=QueryIntent.aggregate,
        table="orders",
        measure_column="grand_total",
        measure_function=MeasureFunction.sum,
    )
    assert normalise_plan(plan, ecommerce, settings).measure_column == "grand_total"


def test_a_filter_column_without_an_operator_is_rejected(ecommerce, settings):
    plan = QueryPlan(
        intent=QueryIntent.retrieve,
        table="orders",
        filter_column="order_state",
        filter_value="shipped",
    )
    assert check_plan(plan, ecommerce, settings)


def test_a_filter_column_with_an_operator_is_accepted(ecommerce, settings):
    plan = QueryPlan(
        intent=QueryIntent.retrieve,
        table="orders",
        filter_column="order_state",
        filter_value="shipped",
        filter_operator="=",
    )
    assert check_plan(plan, ecommerce, settings) is None


def test_prompt_operator_list_is_generated_from_the_engine():
    """The model must be told the operators the engine will accept.

    Written by hand the two lists drift, and the engine then rejects a plan it
    was told to produce.
    """
    from core.constants import OPERATORS
    from core.graph import SYSTEM

    for operator in OPERATORS:
        assert repr(operator) in SYSTEM, operator


def test_prompt_intent_list_is_generated_from_the_enum():
    from core.graph import SYSTEM
    from core.types import QueryIntent

    for intent in QueryIntent:
        assert repr(intent.value) in SYSTEM, intent


def test_prompt_measure_list_is_generated_from_the_enum():
    from core.graph import SYSTEM
    from core.types import MeasureFunction

    for function in MeasureFunction:
        assert repr(function.value) in SYSTEM, function


def test_redaction_strips_catalog_enumeration(settings):
    reason = "column 'x' does not exist; numeric columns are: a, b, c"
    assert "numeric columns" not in redact(reason, settings)


def test_redaction_can_be_turned_off(settings):
    settings.redact_catalog_in_errors = False
    reason = "column 'x' does not exist; numeric columns are: a, b, c"
    assert redact(reason, settings) == reason


def test_template_within_limit_is_accepted(ecommerce, settings):
    plan = QueryPlan(
        intent=QueryIntent.count,
        table="orders",
        template="Found {row_count} matching orders.",
    )
    assert plan.template == "Found {row_count} matching orders."
    assert check_plan(plan, ecommerce, settings) is None


def test_template_exceeding_limit_is_rejected():
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        QueryPlan(
            intent=QueryIntent.count,
            table="orders",
            template="x" * 201,
        )


def test_schema_graph_type_is_exported():
    assert SchemaGraph is not None
