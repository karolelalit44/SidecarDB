from app.graph import check_plan, redact
from app.types import MeasureFunction, QueryIntent, QueryPlan, SchemaGraph


def test_valid_plan_is_accepted(ecommerce):
    assert check_plan(QueryPlan(intent=QueryIntent.count, table="orders"), ecommerce) is None


def test_unknown_table_is_rejected(ecommerce):
    assert check_plan(QueryPlan(intent=QueryIntent.count, table="ghost"), ecommerce) is not None


def test_case_mismatch_is_rejected(ecommerce):
    assert check_plan(QueryPlan(intent=QueryIntent.count, table="Orders"), ecommerce) is not None


def test_unknown_filter_column_is_rejected(ecommerce):
    plan = QueryPlan(
        intent=QueryIntent.lookup, table="orders", filter_column="ghost", filter_value="x"
    )
    assert check_plan(plan, ecommerce) is not None


def test_unknown_projection_column_is_rejected(ecommerce):
    plan = QueryPlan(intent=QueryIntent.retrieve, table="orders", projection=["ghost"])
    assert check_plan(plan, ecommerce) is not None


def test_unknown_order_column_is_rejected(ecommerce):
    plan = QueryPlan(intent=QueryIntent.retrieve, table="orders", order_by="ghost")
    assert check_plan(plan, ecommerce) is not None


def test_unknown_group_column_is_rejected(ecommerce):
    plan = QueryPlan(intent=QueryIntent.retrieve, table="orders", group_by=["ghost"])
    assert check_plan(plan, ecommerce) is not None


def test_unknown_text_search_column_is_rejected(ecommerce):
    plan = QueryPlan(
        intent=QueryIntent.retrieve,
        table="orders",
        text_search_columns=["ghost"],
        text_search_query="x",
    )
    assert check_plan(plan, ecommerce) is not None


def test_unknown_related_table_is_rejected(ecommerce):
    plan = QueryPlan(intent=QueryIntent.retrieve, table="orders", related_tables=["ghost"])
    assert check_plan(plan, ecommerce) is not None


def test_aggregate_requires_measure(ecommerce):
    assert check_plan(QueryPlan(intent=QueryIntent.aggregate, table="orders"), ecommerce)


def test_aggregate_rejects_non_numeric(ecommerce):
    plan = QueryPlan(
        intent=QueryIntent.aggregate,
        table="orders",
        measure_column="order_reference",
        measure_function=MeasureFunction.sum,
    )
    reason = check_plan(plan, ecommerce)
    assert reason and "not numeric" in reason


def test_aggregate_on_numeric_is_accepted(ecommerce):
    plan = QueryPlan(
        intent=QueryIntent.aggregate,
        table="orders",
        measure_column="grand_total",
        measure_function=MeasureFunction.sum,
    )
    assert check_plan(plan, ecommerce) is None


def test_disconnected_related_table_is_rejected(ecommerce, library):
    merged = SchemaGraph(tables={**ecommerce.tables, **library.tables})
    plan = QueryPlan(intent=QueryIntent.retrieve, table="orders", related_tables=["authors"])
    reason = check_plan(plan, merged)
    assert reason and "foreign key path" in reason


def test_every_table_in_every_fixture_validates(ecommerce, healthcare, library):
    for graph in (ecommerce, healthcare, library):
        for name in graph.names:
            assert check_plan(QueryPlan(intent=QueryIntent.count, table=name), graph) is None


def test_qualified_columns_are_normalised(ecommerce):
    """Models habitually write table.column; the prefix is stripped, not rejected."""
    from app.graph import normalise_plan

    plan = QueryPlan(
        intent=QueryIntent.retrieve,
        table="orders",
        projection=["orders.order_reference", "orders.grand_total"],
        filter_column="orders.order_state",
        filter_value="shipped",
        order_by="orders.placed_at",
    )
    fixed = normalise_plan(plan, ecommerce)

    assert fixed.projection == ["order_reference", "grand_total"]
    assert fixed.filter_column == "order_state"
    assert fixed.order_by == "placed_at"


def test_wrong_table_prefix_is_not_stripped(ecommerce):
    from app.graph import normalise_plan

    plan = QueryPlan(
        intent=QueryIntent.retrieve,
        table="orders",
        projection=["customers.display_name"],
    )
    fixed = normalise_plan(plan, ecommerce)
    assert fixed.projection == ["customers.display_name"]
    assert check_plan(fixed, ecommerce) is not None


def test_join_keys_are_added_when_missing(ecommerce):
    """Without the FK columns a joined result cannot be read back to the question."""
    from app.graph import with_join_keys

    plan = QueryPlan(
        intent=QueryIntent.retrieve,
        table="orders",
        related_tables=["customers"],
        projection=["display_name"],
    )
    fixed = with_join_keys(plan, ecommerce)

    assert "customer_id" in fixed.projection
    assert "id" in fixed.projection
    assert "display_name" in fixed.projection


def test_join_keys_are_not_duplicated(ecommerce):
    from app.graph import with_join_keys

    plan = QueryPlan(
        intent=QueryIntent.retrieve,
        table="orders",
        related_tables=["customers"],
        projection=["customer_id", "display_name"],
    )
    fixed = with_join_keys(plan, ecommerce)
    assert fixed.projection.count("customer_id") == 1


def test_join_keys_are_not_added_without_joins(ecommerce):
    from app.graph import with_join_keys

    plan = QueryPlan(intent=QueryIntent.retrieve, table="orders", projection=["grand_total"])
    assert with_join_keys(plan, ecommerce).projection == ["grand_total"]


def test_join_keys_are_not_added_to_a_distinct_projection(ecommerce):
    """Adding keys to DISTINCT would make the uniqueness collapse."""
    from app.graph import with_join_keys

    plan = QueryPlan(
        intent=QueryIntent.distinct,
        table="orders",
        related_tables=["customers"],
        projection=["city"],
    )
    assert with_join_keys(plan, ecommerce).projection == ["city"]


def test_join_keys_are_not_added_to_an_aggregate(ecommerce):
    """The aggregate projection is not rendered, so keys would change nothing useful."""
    from app.graph import with_join_keys

    plan = QueryPlan(
        intent=QueryIntent.aggregate,
        table="orders",
        related_tables=["customers"],
        measure_column="grand_total",
        measure_function=MeasureFunction.sum,
        projection=["city"],
    )
    assert with_join_keys(plan, ecommerce).projection == ["city"]


def test_a_filter_column_without_an_operator_is_rejected(ecommerce):
    plan = QueryPlan(
        intent=QueryIntent.retrieve,
        table="orders",
        filter_column="order_state",
        filter_value="shipped",
    )
    assert check_plan(plan, ecommerce) is not None


def test_a_filter_column_with_an_operator_is_accepted(ecommerce):
    plan = QueryPlan(
        intent=QueryIntent.retrieve,
        table="orders",
        filter_column="order_state",
        filter_value="shipped",
        filter_operator="=",
    )
    assert check_plan(plan, ecommerce) is None


def test_prompt_operator_list_is_generated_from_the_engine(ecommerce):
    """The model must be told the operators the engine will accept.

    Written by hand the two lists drift, and the engine then rejects a plan it
    was told to produce.
    """
    from app.constants import OPERATORS
    from app.graph import SYSTEM

    for operator in OPERATORS:
        assert repr(operator) in SYSTEM, operator


def test_prompt_intent_list_is_generated_from_the_enum():
    from app.graph import SYSTEM
    from app.types import QueryIntent

    for intent in QueryIntent:
        assert repr(intent.value) in SYSTEM, intent


def test_prompt_measure_list_is_generated_from_the_enum():
    from app.graph import SYSTEM
    from app.types import MeasureFunction

    for function in MeasureFunction:
        assert repr(function.value) in SYSTEM, function


def test_redaction_strips_catalog_enumeration():
    reason = "column 'x' does not exist; numeric columns are: a, b, c"
    assert "numeric columns" not in redact(reason)


def test_template_within_limit_is_accepted(ecommerce):
    plan = QueryPlan(
        intent=QueryIntent.count,
        table="orders",
        template="Found {row_count} matching orders.",
    )
    assert plan.template == "Found {row_count} matching orders."
    assert check_plan(plan, ecommerce) is None


def test_template_exceeding_limit_is_rejected():
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        QueryPlan(
            intent=QueryIntent.count,
            table="orders",
            template="x" * 201,
        )
