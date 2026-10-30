import pytest

from core.capabilities import DatabaseGateway
from core.catalog import ColumnInfo, SchemaGraph, TableInfo
from core.client import SidecarDB
from core.plan import Statement
from infra.db import Database


class FakeDatabaseAdapter:
    def __init__(self, tables: dict[str, list[str]] | None = None) -> None:
        table_map: dict[str, TableInfo] = {}
        if tables:
            for table_name, cols in tables.items():
                table_map[table_name] = TableInfo(
                    name=table_name,
                    columns=[ColumnInfo(name=c, data_type="text") for c in cols],
                )
        self._graph = SchemaGraph(tables=table_map)
        self.executed_statements: list[Statement] = []
        self.closed = False
        self.is_healthy = True
        self.rows_to_return: list[dict] = [{"id": 1, "name": "test_record"}]

    async def introspect(self) -> SchemaGraph:
        return self._graph

    async def execute(self, statement: Statement) -> list[dict]:
        self.executed_statements.append(statement)
        return self.rows_to_return

    async def ping(self) -> bool:
        return self.is_healthy

    async def close(self) -> None:
        self.closed = True


class FakePlanner:
    async def complete_json(
        self, system_prompt: str, user_prompt: str, temperature: float | None = None
    ) -> dict:
        return {}


def test_database_implements_gateway_protocol():
    assert issubclass(Database, DatabaseGateway)


def test_fake_database_implements_gateway_protocol():
    adapter = FakeDatabaseAdapter()
    assert isinstance(adapter, DatabaseGateway)


@pytest.mark.asyncio
async def test_sidecardb_uses_unified_database_gateway():
    fake_db = FakeDatabaseAdapter(tables={"products": ["id", "title", "price"]})
    fake_planner = FakePlanner()
    db = SidecarDB(database=fake_db, planner=fake_planner)

    await db.start(require_database=False)
    health = await db.health()
    assert health.database == "connected"
    assert health.tables == 1

    graph = await db.catalog()
    assert "products" in graph.tables
    assert graph.tables["products"].column_names == ["id", "title", "price"]

    assert db._owns_database is False
    await db.stop()

    statement = Statement(sql="SELECT * FROM products LIMIT 50", params=())
    rows = await db.database.execute(statement)
    assert rows == [{"id": 1, "name": "test_record"}]
    assert len(fake_db.executed_statements) == 1
    assert fake_db.executed_statements[0].sql == "SELECT * FROM products LIMIT 50"


@pytest.mark.asyncio
async def test_database_sqlite_end_to_end(tmp_path):
    import sqlite3

    from core.config import Settings

    db_file = tmp_path / "test.db"
    conn = sqlite3.connect(db_file)
    conn.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)")
    conn.execute(
        "CREATE TABLE orders ("
        "id INTEGER PRIMARY KEY, "
        "user_id INTEGER, "
        "amount REAL, "
        "FOREIGN KEY (user_id) REFERENCES users(id))"
    )
    conn.execute("INSERT INTO users VALUES (1, 'Alice')")
    conn.execute("INSERT INTO orders VALUES (101, 1, 49.99)")
    conn.commit()
    conn.close()

    settings = Settings(database_url=f"sqlite:///{db_file}")
    db = Database(settings)
    await db.connect()
    assert db.connected is True
    assert await db.ping() is True

    graph = await db.introspect()
    assert "users" in graph.tables
    assert "orders" in graph.tables
    assert len(graph.foreign_keys) == 1
    assert graph.foreign_keys[0].source_table == "orders"
    assert graph.foreign_keys[0].target_table == "users"

    stmt = Statement(
        sql='SELECT "orders"."id", "users"."name", "orders"."amount" '
        'FROM "orders" JOIN "users" ON "orders"."user_id" = "users"."id" '
        'WHERE "users"."name" = $1',
        params=("Alice",),
    )
    rows = await db.execute(stmt)
    assert len(rows) == 1
    assert rows[0]["name"] == "Alice"
    assert rows[0]["id"] == 101

    await db.close()
    assert db.connected is False
