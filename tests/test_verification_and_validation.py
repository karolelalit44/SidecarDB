from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.exceptions import ConfigurationError
from app.main import app
from app.service import validate_query
from app.verification import StartupVerificationService


def test_validate_query_permits_natural_language():
    safe_queries = [
        "how many orders were placed today?",
        "show orders where grand_total > 100",
        "find customer with email user@example.com",
        "orders (shipped or pending)",
        "items with price: $50 to $100",
        "growth rate + 10%",
        "what's the average order value?",
    ]
    for q in safe_queries:
        is_safe, reason = validate_query(q)
        assert is_safe is True, f"Expected {q} to be safe, but failed with: {reason}"
        assert reason is None


def test_validate_query_rejects_empty():
    is_safe, reason = validate_query("")
    assert is_safe is False
    assert "empty" in reason.lower()


def test_validate_query_rejects_risky_inputs():
    risky_queries = [
        "show orders -- drop everything",
        "show orders /* block comment */",
        "orders; DROP TABLE users",
        "orders\x00with null byte",
        "orders\x07with bell char",
        "orders\u202ewith bidi override",
    ]
    for q in risky_queries:
        is_safe, reason = validate_query(q)
        assert is_safe is False, f"Expected {q} to be flagged as risky"
        assert reason is not None


@pytest.mark.asyncio
async def test_verification_service_fails_when_db_ping_fails():
    verifier = StartupVerificationService()
    with (
        patch("app.db.database.connect", new_callable=AsyncMock),
        patch("app.db.database.ping", new_callable=AsyncMock, return_value=False),
        pytest.raises(ConfigurationError, match="Database ping failed"),
    ):
        await verifier.verify_database()


@pytest.mark.asyncio
async def test_verification_service_fails_when_llm_key_missing(monkeypatch):
    verifier = StartupVerificationService()
    monkeypatch.setattr(settings, "llm_api_key", "")
    with pytest.raises(ConfigurationError, match="LLM_API_KEY"):
        await verifier.verify_llm()


@pytest.mark.asyncio
async def test_verification_service_embeddings_optional_does_not_raise(monkeypatch):
    verifier = StartupVerificationService()
    monkeypatch.setattr(settings, "embedding_enabled", False)
    # Should complete without raising
    await verifier.verify_embeddings()


def test_search_endpoint_handles_query_error():
    client = TestClient(app, raise_server_exceptions=False)
    # Question too short -> QueryError -> 400 status code
    response = client.post("/api/v1/search", json={"query": "x"})
    assert response.status_code == 400
    data = response.json()
    assert data["error"] == "QueryError"
    assert "request_id" in data


def test_search_endpoint_handles_risky_query():
    client = TestClient(app, raise_server_exceptions=False)
    response = client.post("/api/v1/search", json={"query": "orders; drop table users"})
    assert response.status_code == 400
    data = response.json()
    assert data["error"] == "QueryError"
    assert "disallowed symbols" in data["message"]


def test_search_endpoint_handles_query_too_long():
    client = TestClient(app, raise_server_exceptions=False)
    response = client.post("/api/v1/search", json={"query": "a" * (settings.max_query_length + 1)})
    assert response.status_code == 400
    data = response.json()
    assert data["error"] == "QueryError"
    assert "too long" in data["message"].lower()
    assert str(settings.max_query_length) in data["message"]
