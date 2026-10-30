from core.catalog import (
    ColumnInfo,
    ForeignKeyInfo,
    SchemaGraph,
    TableInfo,
    _allocate_column_budgets,
)


def test_schema_graph_relational_annotations():
    fk = ForeignKeyInfo(
        source_table="orders",
        source_column="customer_id",
        target_table="customers",
        target_column="id",
    )
    customers = TableInfo(
        name="customers",
        primary_keys=["id"],
        columns=[
            ColumnInfo(name="id", data_type="integer", is_primary_key=True),
            ColumnInfo(name="email", data_type="text", is_unique=True),
        ],
    )
    orders = TableInfo(
        name="orders",
        primary_keys=["id"],
        foreign_keys=[fk],
        columns=[
            ColumnInfo(name="id", data_type="integer", is_primary_key=True),
            ColumnInfo(name="customer_id", data_type="integer"),
            ColumnInfo(
                name="status",
                data_type="order_status",
                enum_values=["pending", "shipped", "delivered"],
            ),
        ],
    )
    graph = SchemaGraph(tables={"customers": customers, "orders": orders}, foreign_keys=[fk])
    desc = graph.describe()

    assert "id (integer) [PK]" in desc
    assert "email (text) [UQ]" in desc

    assert "customer_id (integer) [FK -> customers.id]" in desc
    assert "status (enum: 'pending' | 'shipped' | 'delivered')" in desc


def test_schema_graph_neighbors_of():
    fk1 = ForeignKeyInfo(
        source_table="orders",
        source_column="customer_id",
        target_table="customers",
        target_column="id",
    )
    fk2 = ForeignKeyInfo(
        source_table="order_items",
        source_column="order_id",
        target_table="orders",
        target_column="id",
    )
    fk3 = ForeignKeyInfo(
        source_table="order_items",
        source_column="product_id",
        target_table="products",
        target_column="id",
    )

    tables = {
        "customers": TableInfo(name="customers"),
        "orders": TableInfo(name="orders"),
        "order_items": TableInfo(name="order_items"),
        "products": TableInfo(name="products"),
        "audit_logs": TableInfo(name="audit_logs"),
    }
    graph = SchemaGraph(tables=tables, foreign_keys=[fk1, fk2, fk3])

    neighbors = graph.neighbors_of(["orders"])
    assert set(neighbors) == {"customers", "order_items"}

    assert graph.neighbors_of(["customers"]) == ["orders"]

    combined = graph.neighbors_of(["orders", "order_items"])
    assert set(combined) == {"customers", "products"}


def test_allocate_column_budgets_fair_water_filling():
    tables = [
        TableInfo(
            name="small_a", columns=[ColumnInfo(name=f"c{i}", data_type="int") for i in range(10)]
        ),
        TableInfo(
            name="small_b", columns=[ColumnInfo(name=f"c{i}", data_type="int") for i in range(10)]
        ),
        TableInfo(
            name="large_c", columns=[ColumnInfo(name=f"c{i}", data_type="int") for i in range(60)]
        ),
        TableInfo(
            name="large_d", columns=[ColumnInfo(name=f"c{i}", data_type="int") for i in range(60)]
        ),
    ]
    alloc = _allocate_column_budgets(tables, 120)
    assert alloc["small_a"] == 10
    assert alloc["small_b"] == 10
    assert alloc["large_c"] == 50
    assert alloc["large_d"] == 50
    assert sum(alloc.values()) == 120


def test_allocate_column_budgets_empty_and_zero():
    assert _allocate_column_budgets([], 100) == {}
    table = TableInfo(name="t", columns=[ColumnInfo(name="c", data_type="int")])
    assert _allocate_column_budgets([table], 0) == {}


def test_schema_graph_describe_no_starvation():
    t1 = TableInfo(
        name="table_1",
        description="First wide table",
        columns=[ColumnInfo(name=f"col_1_{i}", data_type="text") for i in range(80)],
    )
    t2 = TableInfo(
        name="table_2",
        description="Second wide table",
        columns=[ColumnInfo(name=f"col_2_{i}", data_type="text") for i in range(50)],
    )
    t3 = TableInfo(
        name="table_3",
        description="Third table",
        columns=[ColumnInfo(name=f"col_3_{i}", data_type="text") for i in range(20)],
    )

    graph = SchemaGraph(tables={"table_1": t1, "table_2": t2, "table_3": t3})
    output = graph.describe(max_columns=60)

    assert "TABLE table_1" in output
    assert "TABLE table_2" in output
    assert "TABLE table_3" in output

    assert "col_1_0" in output
    assert "col_2_0" in output
    assert "col_3_0" in output

    assert "more columns" in output


def test_schema_graph_describe_unconstrained():
    t1 = TableInfo(
        name="t1",
        columns=[
            ColumnInfo(name="id", data_type="int"),
            ColumnInfo(name="name", data_type="text", description="Name"),
        ],
    )
    graph = SchemaGraph(tables={"t1": t1})
    desc = graph.describe(max_columns=0)
    assert "TABLE t1" in desc
    assert "id (int)" in desc
    assert "name (text) -- Name" in desc
    assert "more columns" not in desc


def test_column_info_generic_numeric_detection():
    numeric_samples = [
        "int",
        "integer",
        "BIGINT",
        "smallint",
        "decimal(10,2)",
        "numeric(12,4)",
        "float",
        "double precision",
        "real",
    ]
    for dtype in numeric_samples:
        col = ColumnInfo(name="amount", data_type=dtype)
        assert col.is_numeric is True

    non_numeric_samples = ["text", "varchar(255)", "boolean", "uuid", "timestamp with time zone"]
    for dtype in non_numeric_samples:
        col = ColumnInfo(name="field", data_type=dtype)
        assert col.is_numeric is False
