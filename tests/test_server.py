import pytest
from fastapi.testclient import TestClient

from catalog.models import ColumnInfo, SchemaGraph, TableInfo
from core.client import SidecarDB
from planner.models import Statement
from server.app import create_app


class FakeDatabaseAdapter:
    def __init__(self) -> None:
        self._graph = SchemaGraph(
            tables={
                "users": TableInfo(
                    name="users",
                    columns=[
                        ColumnInfo(name="id", data_type="integer"),
                        ColumnInfo(name="name", data_type="text"),
                    ],
                )
            }
        )
        self.closed = False

    async def introspect(self) -> SchemaGraph:
        return self._graph

    async def execute(self, statement: Statement) -> list[dict]:
        return [{"id": 1, "name": "Alice"}]

    async def ping(self) -> bool:
        return True

    async def close(self) -> None:
        self.closed = True


class FakeEmbedder:
    @property
    def enabled(self) -> bool:
        return True

    async def aencode(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]

    async def aclose(self) -> None:
        pass


class FakePlanner:
    async def complete_json(
        self, system_prompt: str, user_prompt: str, temperature: float | None = None
    ) -> dict:
        return {
            "intent": "retrieve",
            "table": "users",
            "projection": ["id", "name"],
            "template": "Found {row_count} user(s).",
        }


@pytest.fixture
def test_app(tmp_path):
    fake_db = FakeDatabaseAdapter()
    fake_planner = FakePlanner()
    fake_embedder = FakeEmbedder()
    db = SidecarDB(
        database=fake_db,
        planner=fake_planner,
        embedder=fake_embedder,
        storage_dir=str(tmp_path),
    )
    return create_app(db)


def test_health_endpoint(test_app):
    with TestClient(test_app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["database"] == "connected"
        assert data["tables"] == 1


def test_schema_endpoint(test_app):
    with TestClient(test_app) as client:
        response = client.get("/api/v1/schema")
        assert response.status_code == 200
        data = response.json()
        assert "users" in data["tables"]


def test_search_endpoint(test_app):
    with TestClient(test_app) as client:
        response = client.post("/api/v1/search", json={"question": "Show all users"})
        assert response.status_code == 200
        data = response.json()
        assert data["question"] == "Show all users"
        assert data["row_count"] == 1
        assert data["records"] == [{"id": 1, "name": "Alice"}]
        assert data["answer"] == "Found 1 user(s)."
