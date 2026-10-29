import pytest

from sidecardb import retrieval
from sidecardb.types import SchemaGraph

from .conftest import FakeEmbedder


def test_describe_contains_columns(ecommerce: SchemaGraph):
    text = ecommerce.describe(["orders"])
    assert "TABLE orders" in text
    assert "order_reference" in text


def test_describe_limits_tables(ecommerce: SchemaGraph):
    text = ecommerce.describe(["orders"])
    assert "TABLE products" not in text


def test_describe_respects_column_budget(ecommerce: SchemaGraph):
    text = ecommerce.describe(None, max_columns=3)
    assert "more columns" in text


def test_describe_is_empty_for_empty_graph():
    assert SchemaGraph().describe() == ""


async def test_index_reports_unavailable_without_embeddings(any_graph: SchemaGraph, settings):
    """With embeddings off the index cannot rank, and says so rather than guessing."""
    index = await retrieval.SchemaIndex(FakeEmbedder(enabled=False), settings).build(any_graph)
    assert index.ready is False


async def test_index_search_returns_nothing_when_disabled(any_graph: SchemaGraph, settings):
    index = await retrieval.SchemaIndex(FakeEmbedder(enabled=False), settings).build(any_graph)
    assert await index.search("anything", 5) == []
    assert await index.tables_for("anything", 5, 0.0) == []


async def test_index_builds_documents_for_every_column(ecommerce: SchemaGraph, settings):
    index = await retrieval.SchemaIndex(FakeEmbedder(), settings).build(ecommerce)
    total = len(ecommerce.tables) + sum(len(t.columns) for t in ecommerce.tables.values())
    assert len(index._keys) == total


def test_type_categories_drive_column_classification():
    from sidecardb.types import ColumnInfo

    numeric = ColumnInfo(name="x", data_type="integer", type_category="N")
    other = ColumnInfo(name="z", data_type="date", type_category="D")

    assert numeric.is_numeric
    assert not other.is_numeric


def test_table_numeric_columns_come_from_category(any_graph: SchemaGraph):
    for table in any_graph.tables.values():
        expected = [c.name for c in table.columns if c.is_numeric]
        assert table.numeric_columns == expected


async def test_index_build_is_idempotent(ecommerce: SchemaGraph, settings):
    """Rebuilding replaces the element set rather than accumulating onto the old one."""
    index = await retrieval.SchemaIndex(FakeEmbedder(), settings).build(ecommerce)
    first = list(index._keys)
    await index.build(ecommerce)
    assert index._keys == first


async def test_scoring_uses_the_dot_product_of_normalised_vectors(ecommerce, settings):
    """Both operands are unit vectors, so the score is the plain dot product."""
    from sidecardb.retrieval import dot

    a = [1.0, 0.0, 0.0]
    b = [0.6, 0.8, 0.0]
    assert dot(a, b) == pytest.approx(0.6)
    assert dot(b, b) == pytest.approx(1.0)
