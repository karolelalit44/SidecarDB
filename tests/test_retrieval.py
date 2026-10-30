import pytest

from core.catalog import ColumnInfo, SchemaGraph, TableInfo
from core.plan import PlanPolicy
from core.retrieval import SchemaIndex, dot, link, rank_fallback_tables


class FakeEmbedder:
    def __init__(self, vector_map: dict[str, list[float]] | None = None) -> None:
        self.vector_map = vector_map or {}
        self.dim = 4

    @property
    def enabled(self) -> bool:
        return True

    async def aencode(self, texts: list[str]) -> list[list[float]]:
        result = []
        for text in texts:
            if text in self.vector_map:
                result.append(self.vector_map[text])
            else:
                result.append([1.0, 0.0, 0.0, 0.0])
        return result

    async def aclose(self) -> None:
        pass


def test_dot_product():
    assert dot([1.0, 0.0], [1.0, 0.0]) == 1.0
    assert dot([1.0, 0.0], [0.0, 1.0]) == 0.0
    assert dot([0.5, 0.5], [0.5, 0.5]) == 0.5


@pytest.mark.asyncio
async def test_tables_for_expands_candidates_beyond_default_top_k():
    cols_a = [ColumnInfo(name=f"col_a_{i}", data_type="text") for i in range(15)]
    table_a = TableInfo(name="table_a", columns=cols_a)

    table_b = TableInfo(name="table_b", columns=[ColumnInfo(name="col_b", data_type="text")])
    table_c = TableInfo(name="table_c", columns=[ColumnInfo(name="col_c", data_type="text")])
    table_d = TableInfo(name="table_d", columns=[ColumnInfo(name="col_d", data_type="text")])

    graph = SchemaGraph(
        tables={"table_a": table_a, "table_b": table_b, "table_c": table_c, "table_d": table_d}
    )

    embedder = FakeEmbedder()
    policy = PlanPolicy(retrieval_top_k=12, retrieval_min_score=0.1)
    index = SchemaIndex(embedder, policy)
    await index.build(graph)

    chosen = await index.tables_for(question="query", limit=4, min_score=0.1)
    assert len(chosen) >= 3
    assert "table_a" in chosen
    assert "table_b" in chosen
    assert "table_c" in chosen


def test_rank_fallback_tables_prioritizes_documented_entities():
    tables = {
        "alembic_version": TableInfo(
            name="alembic_version", columns=[ColumnInfo(name="version", data_type="text")]
        ),
        "audit_logs": TableInfo(
            name="audit_logs", row_estimate=10000, columns=[ColumnInfo(name="id", data_type="int")]
        ),
        "cases": TableInfo(
            name="cases",
            description="Crime incident records",
            row_estimate=5000,
            columns=[
                ColumnInfo(name="id", data_type="int"),
                ColumnInfo(name="title", data_type="text"),
            ],
        ),
        "officers": TableInfo(
            name="officers",
            description="Active duty police officers",
            row_estimate=200,
            columns=[ColumnInfo(name="id", data_type="int")],
        ),
    }
    graph = SchemaGraph(tables=tables)
    ranked = rank_fallback_tables(graph, max_tables=4)
    assert ranked[0] in ("cases", "officers")
    assert ranked[1] in ("cases", "officers")
    assert ranked[2] in ("audit_logs", "alembic_version")
    assert ranked[3] in ("audit_logs", "alembic_version")


@pytest.mark.asyncio
async def test_link_1hop_relational_expansion():
    from core.catalog import ForeignKeyInfo

    fk = ForeignKeyInfo(
        source_table="orders",
        source_column="customer_id",
        target_table="customers",
        target_column="id",
    )
    tables = {
        "orders": TableInfo(name="orders", columns=[ColumnInfo(name="id", data_type="int")]),
        "customers": TableInfo(name="customers", columns=[ColumnInfo(name="id", data_type="int")]),
        "audit": TableInfo(name="audit", columns=[ColumnInfo(name="id", data_type="int")]),
    }
    graph = SchemaGraph(tables=tables, foreign_keys=[fk])

    class MockIndex:
        async def tables_for(self, question, limit, min_score):
            return ["orders"]

    active_policy = PlanPolicy(max_schema_tables=5)
    names, mode, retrieved = await link("who made orders?", graph, MockIndex(), active_policy)

    assert mode.value == "embedding"
    assert retrieved == ["orders"]
    assert "orders" in names
    assert "customers" in names
    assert "audit" not in names


@pytest.mark.asyncio
async def test_hybrid_lexical_token_boost():
    table_items = TableInfo(
        name="items",
        columns=[
            ColumnInfo(name="sku", data_type="text"),
            ColumnInfo(name="notes", data_type="text"),
        ],
    )
    graph = SchemaGraph(tables={"items": table_items})

    embedder = FakeEmbedder(
        vector_map={
            "items": [0.5, 0.5, 0.0, 0.0],
            "items.sku": [0.5, 0.5, 0.0, 0.0],
            "items.notes": [0.5, 0.5, 0.0, 0.0],
            "what is the sku?": [0.5, 0.5, 0.0, 0.0],
        }
    )
    index = SchemaIndex(embedder, PlanPolicy())
    await index.build(graph)

    hits = await index.search("what is the sku?", top_k=3)
    sku_hit = next(h for h in hits if h.subject == "items.sku")
    notes_hit = next(h for h in hits if h.subject == "items.notes")
    assert sku_hit.score > notes_hit.score


@pytest.mark.asyncio
async def test_link_falls_back_to_ranked_catalog_when_no_match():
    tables = {
        "audit_log": TableInfo(name="audit_log", columns=[ColumnInfo(name="id", data_type="int")]),
        "users": TableInfo(
            name="users",
            description="User accounts",
            row_estimate=100,
            columns=[ColumnInfo(name="id", data_type="int")],
        ),
    }
    graph = SchemaGraph(tables=tables)
    names, mode, retrieved = await link(question="unrelated query", graph=graph, index=None)
    assert mode.value == "catalog"
    assert names[0] == "users"
    assert retrieved == []


@pytest.mark.asyncio
async def test_schema_vector_disk_caching(tmp_path):
    table = TableInfo(name="cases", columns=[ColumnInfo(name="id", data_type="int")])
    graph = SchemaGraph(tables={"cases": table})

    call_count = 0

    class CountingEmbedder:
        @property
        def enabled(self):
            return True

        async def aencode(self, texts):
            nonlocal call_count
            call_count += 1
            return [[1.0, 0.0] for _ in texts]

        async def aclose(self):
            pass

    embedder = CountingEmbedder()
    policy = PlanPolicy()

    idx1 = SchemaIndex(embedder, policy)
    await idx1.build(graph, cache_dir=tmp_path)
    assert call_count == 1
    assert idx1.ready

    idx2 = SchemaIndex(embedder, policy)
    await idx2.build(graph, cache_dir=tmp_path)
    assert call_count == 1
    assert idx2.ready

    table2 = TableInfo(
        name="cases",
        columns=[
            ColumnInfo(name="id", data_type="int"),
            ColumnInfo(name="new_col", data_type="text"),
        ],
    )
    graph2 = SchemaGraph(tables={"cases": table2})
    idx3 = SchemaIndex(embedder, policy)
    await idx3.build(graph2, cache_dir=tmp_path)
    assert call_count == 2
    assert idx3.ready
