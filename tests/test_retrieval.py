import pytest

from catalog.models import ColumnInfo, ForeignKeyInfo, SchemaGraph, TableInfo
from planner.models import PlanPolicy
from retrieval.linker import dot, link
from retrieval.memory import SchemaMemory, _describe_table


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
async def test_tables_for_retrieves_candidates():
    table_a = TableInfo(
        name="table_a",
        columns=[ColumnInfo(name="id", data_type="int")],
    )
    table_b = TableInfo(name="table_b", columns=[ColumnInfo(name="col_b", data_type="text")])
    table_c = TableInfo(name="table_c", columns=[ColumnInfo(name="col_c", data_type="text")])
    table_d = TableInfo(name="table_d", columns=[ColumnInfo(name="col_d", data_type="text")])

    graph = SchemaGraph(
        tables={"table_a": table_a, "table_b": table_b, "table_c": table_c, "table_d": table_d}
    )

    embedder = FakeEmbedder()
    policy = PlanPolicy(retrieval_top_k=12, retrieval_min_score=0.1)
    memory = SchemaMemory(embedder, policy)
    await memory.build(graph)

    chosen = await memory.tables_for(question="query", limit=4, min_score=0.1)
    assert len(chosen) >= 3
    assert "table_a" in chosen
    assert "table_b" in chosen
    assert "table_c" in chosen


@pytest.mark.asyncio
async def test_link_1hop_relational_expansion():
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

    class MockMemory:
        ready = True

        async def tables_for(self, question, limit, min_score):
            return ["orders"]

    active_policy = PlanPolicy(max_schema_tables=5)
    names, mode, retrieved = await link("who made orders?", graph, MockMemory(), active_policy)

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
    table_users = TableInfo(
        name="users",
        columns=[ColumnInfo(name="email", data_type="text")],
    )
    graph = SchemaGraph(tables={"items": table_items, "users": table_users})

    desc_items = _describe_table(table_items)
    desc_users = _describe_table(table_users)

    embedder = FakeEmbedder(
        vector_map={
            desc_items: [0.5, 0.5, 0.0, 0.0],
            desc_users: [0.5, 0.5, 0.0, 0.0],
            "what is the sku?": [0.5, 0.5, 0.0, 0.0],
        }
    )
    memory = SchemaMemory(embedder, PlanPolicy())
    await memory.build(graph)

    hits = await memory.search("what is the sku?", top_k=2)
    assert hits[0].table == "items"
    assert hits[0].score > hits[1].score


@pytest.mark.asyncio
async def test_link_returns_empty_when_no_match():
    tables = {
        "audit_log": TableInfo(name="audit_log", columns=[ColumnInfo(name="id", data_type="int")]),
        "users": TableInfo(
            name="users",
            description="User accounts",
            columns=[ColumnInfo(name="id", data_type="int")],
        ),
    }
    graph = SchemaGraph(tables=tables)
    names, mode, retrieved = await link(question="unrelated query", graph=graph, index=None)
    assert mode.value == "embedding"
    assert names == []
    assert retrieved == []


@pytest.mark.asyncio
async def test_schema_persistence_and_restore(tmp_path):
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

    mem1 = SchemaMemory(embedder, policy)
    await mem1.build(graph, storage_dir=tmp_path)
    assert call_count == 1
    assert mem1.ready
    assert (tmp_path / "schema.json").is_file()
    assert (tmp_path / "schema_vectors.npy").is_file()

    mem2 = SchemaMemory(embedder, policy)
    await mem2.build(graph, storage_dir=tmp_path)
    assert call_count == 1
    assert mem2.ready

    restored = SchemaMemory.load_persisted(tmp_path, embedder, policy)
    assert restored is not None
    restored_graph, restored_memory = restored
    assert "cases" in restored_graph.tables
    assert restored_memory.ready

    table2 = TableInfo(
        name="cases",
        columns=[
            ColumnInfo(name="id", data_type="int"),
            ColumnInfo(name="new_col", data_type="text"),
        ],
    )
    graph2 = SchemaGraph(tables={"cases": table2})
    mem3 = SchemaMemory(embedder, policy)
    await mem3.build(graph2, storage_dir=tmp_path)
    assert call_count == 2
    assert mem3.ready
