import pytest

from app import render
from app.exceptions import SecurityError
from app.types import MeasureFunction, QueryIntent, QueryPlan, SchemaGraph


def test_count_plan(any_graph: SchemaGraph):
    name = next(iter(any_graph.tables))
    sql = render.build(QueryPlan(intent=QueryIntent.count, table=name), any_graph)
    assert sql == f'SELECT COUNT(*) AS record_count FROM "{name}"'


def test_aggregate_plan(any_graph: SchemaGraph):
    name = next(n for n, t in any_graph.tables.items() if t.numeric_columns)
    column = any_graph.tables[name].numeric_columns[0]
    sql = render.build(
        QueryPlan(
            intent=QueryIntent.aggregate,
            table=name,
            measure_column=column,
            measure_function=MeasureFunction.sum,
        ),
        any_graph,
    )
    # The function name must stay unquoted: PostgreSQL would look for "SUM".
    assert f'SUM("{column}")' in sql
    assert f'AS "sum_of_{column}"' in sql


def test_retrieve_defaults_to_every_column(ecommerce: SchemaGraph):
    """The wildcard is expanded so a vector column can never be selected by `*`."""
    sql = render.build(QueryPlan(intent=QueryIntent.retrieve, table="orders"), ecommerce)
    for column in ecommerce.tables["orders"].column_names:
        assert f'"{column}"' in sql
    assert "SELECT *" not in sql


def test_a_vector_column_is_never_projected(ecommerce: SchemaGraph):
    """A vector column is an index, not content: it would flood the prompt."""
    plan = QueryPlan(intent=QueryIntent.retrieve, table="products")
    assert "semantic_vector" not in render.build(plan, ecommerce)


def test_an_explicitly_named_vector_column_is_dropped(ecommerce: SchemaGraph):
    plan = QueryPlan(
        intent=QueryIntent.retrieve,
        table="products",
        projection=["title", "semantic_vector"],
    )
    sql = render.build(plan, ecommerce)
    assert '"title"' in sql
    assert "semantic_vector" not in sql


def test_the_wildcard_survives_when_every_column_is_a_vector():
    from app.types import ColumnInfo, SchemaGraph, TableInfo

    graph = SchemaGraph(
        tables={
            "blobs": TableInfo(
                name="blobs",
                columns=[ColumnInfo(name="embedding", data_type="vector")],
                embedding_columns=["embedding"],
            )
        }
    )
    assert render.build(QueryPlan(intent=QueryIntent.retrieve, table="blobs"), graph) == (
        'SELECT * FROM "blobs"'
    )


def test_projection_is_quoted(ecommerce: SchemaGraph):
    sql = render.build(
        QueryPlan(
            intent=QueryIntent.retrieve,
            table="orders",
            projection=["order_reference", "grand_total"],
        ),
        ecommerce,
    )
    assert sql.startswith('SELECT "order_reference", "grand_total"')


def test_equality_filter(ecommerce: SchemaGraph):
    sql = render.build(
        QueryPlan(
            intent=QueryIntent.lookup,
            table="customers",
            filter_column="city",
            filter_operator="=",
            filter_value="Bergen",
        ),
        ecommerce,
    )
    assert "\"city\" = 'Bergen'" in sql


def test_like_filter_escapes_wildcards(ecommerce: SchemaGraph):
    sql = render.build(
        QueryPlan(
            intent=QueryIntent.lookup,
            table="customers",
            filter_column="display_name",
            filter_operator="ILIKE",
            filter_value="100%",
        ),
        ecommerce,
    )
    assert r"100\%" in sql


def test_like_filter_escapes_quotes(ecommerce: SchemaGraph):
    sql = render.build(
        QueryPlan(
            intent=QueryIntent.lookup,
            table="customers",
            filter_column="display_name",
            filter_operator="LIKE",
            filter_value="o'brien",
        ),
        ecommerce,
    )
    assert "o''brien" in sql


def test_null_filters(ecommerce: SchemaGraph):
    sql = render.build(
        QueryPlan(
            intent=QueryIntent.retrieve,
            table="customers",
            filter_column="city",
            filter_operator="IS NULL",
        ),
        ecommerce,
    )
    assert sql.endswith('WHERE "city" IS NULL')


def test_unknown_operator_is_refused(ecommerce: SchemaGraph):
    """An unrecognised operator is an error, never a silent substitution."""
    with pytest.raises(SecurityError):
        render.build(
            QueryPlan(
                intent=QueryIntent.lookup,
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
                intent=QueryIntent.lookup,
                table="customers",
                filter_column="city",
                filter_operator="=",
            ),
            ecommerce,
        )


def test_join_uses_foreign_key(ecommerce: SchemaGraph):
    sql = render.build(
        QueryPlan(intent=QueryIntent.retrieve, table="orders", related_tables=["customers"]),
        ecommerce,
    )
    assert '"orders"."customer_id" = "customers"."id"' in sql


def test_join_swaps_columns_when_reversed(ecommerce: SchemaGraph):
    sql = render.build(
        QueryPlan(intent=QueryIntent.retrieve, table="customers", related_tables=["orders"]),
        ecommerce,
    )
    assert '"customers"."id" = "orders"."customer_id"' in sql


def test_an_unjoinable_related_table_is_refused_not_cross_joined(
    ecommerce: SchemaGraph, library: SchemaGraph
):
    """A cross join returns a cartesian product dressed up as an answer."""
    merged = SchemaGraph(tables={**ecommerce.tables, **library.tables})
    with pytest.raises(SecurityError):
        render.build(
            QueryPlan(intent=QueryIntent.retrieve, table="orders", related_tables=["authors"]),
            merged,
        )


def test_a_table_reachable_but_not_directly_joined_is_refused(
    ecommerce: SchemaGraph, healthcare: SchemaGraph
):
    """customers -> products is two hops; the renderer can only express one."""
    merged = SchemaGraph(tables={**ecommerce.tables, **healthcare.tables})
    plan = QueryPlan(intent=QueryIntent.retrieve, table="customers", related_tables=["products"])
    with pytest.raises(SecurityError):
        render.build(plan, merged)


def test_group_and_order(ecommerce: SchemaGraph):
    sql = render.build(
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
    assert 'GROUP BY "order_state"' in sql
    assert 'ORDER BY "order_state" DESC' in sql


def test_distinct(healthcare: SchemaGraph):
    sql = render.build(
        QueryPlan(intent=QueryIntent.distinct, table="encounters", projection=["visit_status"]),
        healthcare,
    )
    assert sql.startswith('SELECT DISTINCT "visit_status"')


def test_joined_projection_is_qualified(ecommerce: SchemaGraph):
    """Columns from either side of a join are qualified to avoid ambiguity."""
    sql = render.build(
        QueryPlan(
            intent=QueryIntent.retrieve,
            table="orders",
            related_tables=["customers"],
            projection=["order_reference", "display_name", "city"],
        ),
        ecommerce,
    )
    assert '"orders"."order_reference"' in sql
    assert '"customers"."display_name"' in sql
    assert '"customers"."city"' in sql


def test_single_table_projection_stays_unqualified(ecommerce: SchemaGraph):
    sql = render.build(
        QueryPlan(intent=QueryIntent.retrieve, table="orders", projection=["order_reference"]),
        ecommerce,
    )
    assert '"order_reference"' in sql
    assert "orders.order_reference" not in sql


def test_joined_filter_is_qualified(ecommerce: SchemaGraph):
    sql = render.build(
        QueryPlan(
            intent=QueryIntent.lookup,
            table="orders",
            related_tables=["customers"],
            filter_column="city",
            filter_operator="=",
            filter_value="Bergen",
        ),
        ecommerce,
    )
    assert '"customers"."city" = \'Bergen\'' in sql


def test_three_part_column_is_normalised(ecommerce: SchemaGraph):
    """Model habit produces stations.officers.name; the trailing column wins."""
    from app.graph import normalise_plan

    plan = QueryPlan(
        intent=QueryIntent.retrieve,
        table="orders",
        related_tables=["customers"],
        projection=["orders.customers.display_name"],
    )
    fixed = normalise_plan(plan, ecommerce)
    assert fixed.projection == ["display_name"]


def test_injection_through_a_column_name_is_impossible():
    with pytest.raises(ValueError):
        render.quote('x" ; DROP TABLE y; --')
