from unittest.mock import AsyncMock, MagicMock

import pytest

from config.settings import Settings
from storage.connection import Database


@pytest.mark.asyncio
async def test_read_catalog_maps_constraints_fks_and_enums():
    settings = Settings(
        database_url="postgresql://test:test@localhost:5432/test",
        db_schema="public",
    )
    db = Database(settings)

    mock_tables = [
        {"name": "customers", "description": "Customer profiles"},
        {"name": "orders", "description": "Customer orders"},
    ]

    mock_columns = [
        {
            "table_name": "customers",
            "name": "id",
            "data_type": "integer",
            "type_category": "N",
            "nullable": False,
            "description": None,
        },
        {
            "table_name": "customers",
            "name": "email",
            "data_type": "text",
            "type_category": "S",
            "nullable": False,
            "description": "Email address",
        },
        {
            "table_name": "orders",
            "name": "id",
            "data_type": "integer",
            "type_category": "N",
            "nullable": False,
            "description": None,
        },
        {
            "table_name": "orders",
            "name": "customer_id",
            "data_type": "integer",
            "type_category": "N",
            "nullable": False,
            "description": None,
        },
        {
            "table_name": "orders",
            "name": "status",
            "data_type": "order_status",
            "type_category": "E",
            "nullable": False,
            "description": "Order status",
        },
    ]

    mock_constraints = [
        {"table_name": "customers", "column_name": "id", "constraint_type": "p"},
        {"table_name": "customers", "column_name": "email", "constraint_type": "u"},
        {"table_name": "orders", "column_name": "id", "constraint_type": "p"},
    ]

    mock_foreign_keys = [
        {
            "source_table": "orders",
            "source_column": "customer_id",
            "target_table": "customers",
            "target_column": "id",
        }
    ]

    mock_enums = [
        {"type_name": "order_status", "enum_label": "pending"},
        {"type_name": "order_status", "enum_label": "shipped"},
        {"type_name": "order_status", "enum_label": "delivered"},
    ]

    mock_conn = AsyncMock()

    async def mock_fetch(query, *args):
        if "pg_class" in query and "obj_description" in query:
            return mock_tables
        if "pg_attribute" in query and "format_type" in query:
            return mock_columns
        if "pg_constraint" in query and "contype IN ('p', 'u')" in query:
            return mock_constraints
        if "pg_constraint" in query and "contype = 'f'" in query:
            return mock_foreign_keys
        if "pg_enum" in query:
            return mock_enums
        return []

    mock_conn.fetch = mock_fetch

    mock_pool = MagicMock()
    mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
    mock_pool.acquire.return_value.__aexit__.return_value = None
    mock_pool.is_closing.return_value = False

    db._pool = mock_pool

    graph = await db.introspect()

    assert len(graph.tables) == 2
    assert "customers" in graph.tables
    assert "orders" in graph.tables

    customers = graph.tables["customers"]
    assert customers.primary_keys == ["id"]
    col_id = next(c for c in customers.columns if c.name == "id")
    col_email = next(c for c in customers.columns if c.name == "email")
    assert col_id.is_primary_key
    assert not col_id.is_unique
    assert not col_email.is_primary_key
    assert col_email.is_unique

    orders = graph.tables["orders"]
    assert len(orders.foreign_keys) == 1
    fk = orders.foreign_keys[0]
    assert fk.source_table == "orders"
    assert fk.source_column == "customer_id"
    assert fk.target_table == "customers"
    assert fk.target_column == "id"

    col_status = next(c for c in orders.columns if c.name == "status")
    assert col_status.enum_values == ["pending", "shipped", "delivered"]

    assert len(graph.foreign_keys) == 1
    assert graph.foreign_keys[0] == fk
