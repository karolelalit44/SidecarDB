import pytest

from app import retrieval
from app.config import settings
from app.db import join_path
from app.types import SchemaGraph


def test_join_path_direct(ecommerce: SchemaGraph):
    assert join_path("orders", "customers", ecommerce, 4) == ["customers"]


def test_join_path_multi_hop(ecommerce: SchemaGraph):
    assert join_path("customers", "products", ecommerce, 4) == [
        "orders",
        "order_lines",
        "products",
    ]


def test_join_path_none_when_disconnected(ecommerce: SchemaGraph, library: SchemaGraph):
    merged = SchemaGraph(tables={**ecommerce.tables, **library.tables})
    assert join_path("orders", "authors", merged, 4) is None


def test_join_path_respects_depth_limit(ecommerce: SchemaGraph):
    assert join_path("customers", "products", ecommerce, 1) is None


def test_join_path_self(ecommerce: SchemaGraph):
    assert join_path("orders", "orders", ecommerce, 4) == []


def test_describe_contains_columns_and_keys(ecommerce: SchemaGraph):
    text = ecommerce.describe(["orders"])
    assert "TABLE orders" in text
    assert "order_reference" in text
    assert "FOREIGN KEY customer_id -> customers.id" in text


def test_describe_limits_tables(ecommerce: SchemaGraph):
    text = ecommerce.describe(["orders"])
    assert "TABLE products" not in text


def test_describe_respects_column_budget(ecommerce: SchemaGraph):
    text = ecommerce.describe(None, max_columns=3)
    assert "more columns" in text


def test_describe_is_empty_for_empty_graph():
    assert SchemaGraph().describe() == ""


async def test_index_reports_unavailable_without_embeddings(any_graph: SchemaGraph, monkeypatch):
    """With embeddings off the index cannot rank, and says so rather than guessing."""
    monkeypatch.setattr(settings, "embedding_enabled", False)
    index = await retrieval.SchemaIndex().build(any_graph)
    assert index.ready is False


async def test_index_search_returns_nothing_when_disabled(any_graph: SchemaGraph, monkeypatch):
    monkeypatch.setattr(settings, "embedding_enabled", False)
    index = await retrieval.SchemaIndex().build(any_graph)
    assert await index.search("anything", 5) == []
    assert await index.tables_for("anything", 5, 0.0) == []


async def test_index_builds_documents_for_every_column(ecommerce: SchemaGraph):
    index = await retrieval.SchemaIndex().build(ecommerce)
    total = len(ecommerce.tables) + sum(len(t.columns) for t in ecommerce.tables.values())
    assert len(index._keys) == total


def test_embedding_column_lookup(any_graph: SchemaGraph):
    """Eligibility for vector search is read from the catalog, not configured."""
    for table in any_graph.tables.values():
        expected = table.embedding_columns[0] if table.embedding_columns else None
        assert any_graph.embedding_column_for(table.name) == expected


def test_type_categories_drive_column_classification():
    from app.types import ColumnInfo

    numeric = ColumnInfo(name="x", data_type="integer", type_category="N")
    string = ColumnInfo(name="y", data_type="varchar", type_category="S")
    other = ColumnInfo(name="z", data_type="date", type_category="D")

    assert numeric.is_numeric and not numeric.is_string
    assert string.is_string and not string.is_numeric
    assert not other.is_numeric and not other.is_string


def test_table_numeric_columns_come_from_category(any_graph: SchemaGraph):
    for table in any_graph.tables.values():
        expected = [c.name for c in table.columns if c.is_numeric]
        assert table.numeric_columns == expected


async def test_index_build_is_idempotent(ecommerce: SchemaGraph):
    """Rebuilding replaces the element set rather than accumulating onto the old one."""
    index = await retrieval.SchemaIndex().build(ecommerce)
    first = list(index._keys)
    await index.build(ecommerce)
    assert index._keys == first


async def test_scoring_uses_the_dot_product_of_normalised_vectors(ecommerce: SchemaGraph):
    """Both operands are unit vectors, so the score is the plain dot product."""
    from app.embeddings import dot

    a = [1.0, 0.0, 0.0]
    b = [0.6, 0.8, 0.0]
    assert dot(a, b) == pytest.approx(0.6)
    assert dot(b, b) == pytest.approx(1.0)
