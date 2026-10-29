import pytest

from sidecardb import render
from sidecardb.exceptions import SecurityError
from sidecardb.render import like_pattern
from sidecardb.types import MeasureFunction, QueryIntent, QueryPlan, SchemaGraph


def test_count_plan(any_graph: SchemaGraph):
    name = next(iter(any_graph.tables))
    rendered = render.build(QueryPlan(intent=QueryIntent.count, table=name), any_graph)
    assert rendered.sql == f'SELECT COUNT(*) AS record_count FROM "{name}"'
    assert rendered.params == ()


def test_aggregate_plan(any_graph: SchemaGraph):
    name = next(n for n, t in any_graph.tables.items() if t.numeric_columns)
    column = any_graph.tables[name].numeric_columns[0]
    rendered = render.build(
        QueryPlan(
            intent=QueryIntent.aggregate,
            table=name,
            measure_column=column,
            measure_function=MeasureFunction.sum,
        ),
        any_graph,
    )
    # The function name must stay unquoted: PostgreSQL would look for "SUM".
    assert f'SUM("{column}")' in rendered.sql
    assert f'AS "sum_of_{column}"' in rendered.sql


def test_retrieve_defaults_to_every_column(ecommerce: SchemaGraph):
    """The wildcard is expanded so the projection is always explicit."""
    rendered = render.build(QueryPlan(intent=QueryIntent.retrieve, table="orders"), ecommerce)
    for column in ecommerce.tables["orders"].column_names:
        assert f'"{column}"' in rendered.sql
    assert "SELECT *" not in rendered.sql


def test_projection_is_quoted(ecommerce: SchemaGraph):
    rendered = render.build(
        QueryPlan(
            intent=QueryIntent.retrieve,
            table="orders",
            projection=["order_reference", "grand_total"],
        ),
        ecommerce,
    )
    assert rendered.sql.startswith('SELECT "order_reference", "grand_total"')


def test_equality_filter_is_bound_not_interpolated(ecommerce: SchemaGraph):
    rendered = render.build(
        QueryPlan(
            intent=QueryIntent.retrieve,
            table="customers",
            filter_column="city",
            filter_operator="=",
            filter_value="Bergen",
        ),
        ecommerce,
    )
    assert '"city" = $1' in rendered.sql
    assert "Bergen" not in rendered.sql
    assert rendered.params == ("Bergen",)


def test_a_quote_in_a_value_needs_no_escaping_when_bound(ecommerce: SchemaGraph):
    rendered = render.build(
        QueryPlan(
            intent=QueryIntent.retrieve,
            table="customers",
            filter_column="display_name",
            filter_operator="=",
            filter_value="o'brien",
        ),
        ecommerce,
    )
    assert rendered.params == ("o'brien",)
    assert "''" not in rendered.sql


def test_like_filter_escapes_wildcards_in_the_parameter(ecommerce: SchemaGraph):
    rendered = render.build(
        QueryPlan(
            intent=QueryIntent.retrieve,
            table="customers",
            filter_column="display_name",
            filter_operator="ILIKE",
            filter_value="100%",
        ),
        ecommerce,
    )
    assert rendered.params == (r"100\%",)
    assert '"display_name" ILIKE $1' in rendered.sql


def test_like_pattern_escapes_the_metacharacters():
    assert like_pattern("100%") == r"100\%"
    assert like_pattern("a_b") == r"a\_b"
    assert like_pattern("back\\slash") == "back\\\\slash"
    assert like_pattern("plain") == "plain"


def test_null_filters_carry_no_parameter(ecommerce: SchemaGraph):
    rendered = render.build(
        QueryPlan(
            intent=QueryIntent.retrieve,
            table="customers",
            filter_column="city",
            filter_operator="IS NULL",
        ),
        ecommerce,
    )
    assert rendered.sql.endswith('WHERE "city" IS NULL')
    assert rendered.params == ()


def test_unknown_operator_is_refused(ecommerce: SchemaGraph):
    """An unrecognised operator is an error, never a silent substitution."""
    with pytest.raises(SecurityError):
        render.build(
            QueryPlan(
                intent=QueryIntent.retrieve,
                table="customers",
                filter_column="city",
                filter_operator="REGEXP",
                filter_value="B.*",
            ),
            ecommerce,
        )


def test_operator_without_value_is_refused(ecommerce: SchemaGraph):
    with pytest.raises(SecurityError):
        render.build(
            QueryPlan(
                intent=QueryIntent.retrieve,
                table="customers",
                filter_column="city",
                filter_operator="=",
            ),
            ecommerce,
        )


def test_group_and_order(ecommerce: SchemaGraph):
    rendered = render.build(
        QueryPlan(
            intent=QueryIntent.aggregate,
            table="orders",
            measure_column="grand_total",
            measure_function=MeasureFunction.avg,
            group_by=["order_state"],
            order_by="order_state",
            descending=True,
        ),
        ecommerce,
    )
    assert 'GROUP BY "order_state"' in rendered.sql
    assert 'ORDER BY "order_state" DESC' in rendered.sql


def test_distinct(healthcare: SchemaGraph):
    rendered = render.build(
        QueryPlan(intent=QueryIntent.distinct, table="encounters", projection=["visit_status"]),
        healthcare,
    )
    assert rendered.sql.startswith('SELECT DISTINCT "visit_status"')


def test_projection_stays_unqualified(ecommerce: SchemaGraph):
    rendered = render.build(
        QueryPlan(intent=QueryIntent.retrieve, table="orders", projection=["order_reference"]),
        ecommerce,
    )
    assert '"order_reference"' in rendered.sql
    assert "orders.order_reference" not in rendered.sql


def test_three_part_column_is_normalised(ecommerce: SchemaGraph, settings):
    """Model habit prefixes names; the trailing column wins when the table has it."""
    from sidecardb.graph import normalise_plan

    plan = QueryPlan(
        intent=QueryIntent.retrieve,
        table="orders",
        projection=["orders.customers.order_reference"],
    )
    fixed = normalise_plan(plan, ecommerce, settings)
    assert fixed.projection == ["order_reference"]


def test_injection_through_a_column_name_is_impossible():
    with pytest.raises(ValueError):
        render.quote('x" ; DROP TABLE y; --')
