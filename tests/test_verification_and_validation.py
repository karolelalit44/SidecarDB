from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from core import AskDB
from core.config import Settings
from core.exceptions import ConfigurationError
from core.server import create_app
from core.service import validate_query
from core.verification import StartupVerificationService


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


def offline_settings(**overrides) -> Settings:
    """Settings that cannot reach a database, and do not retry slowly."""
    base = {
        "_env_file": None,
        "database_url": "postgresql://app:app@127.0.0.1:1/app",
        "llm_api_key": "test-key",
        "embedding_provider": "none",
        "db_connect_retries": 0,
        "recovery_interval_seconds": 600.0,
    }
    base.update(overrides)
    return Settings(**base)


def test_verification_service_fails_when_the_url_is_missing():
    verifier = StartupVerificationService(offline_settings(database_url=""))
    with pytest.raises(ConfigurationError, match="DATABASE_URL"):
        verifier.verify_database()


def test_verification_service_never_touches_the_database():
    """Verification reads configuration; the connection belongs elsewhere."""
    verifier = StartupVerificationService(offline_settings())
    with patch("infra.db.Database.connect", new_callable=AsyncMock) as connect:
        verifier.verify()
    connect.assert_not_called()


def test_verification_service_fails_when_llm_key_missing():
    verifier = StartupVerificationService(offline_settings(llm_api_key=""))
    with pytest.raises(ConfigurationError, match="LLM_API_KEY"):
        verifier.verify_llm()


def test_verification_service_embeddings_off_does_not_raise():
    verifier = StartupVerificationService(offline_settings(embedding_provider="none"))
    verifier.verify_embeddings()


def client_for(settings: Settings) -> TestClient:
    return TestClient(create_app(AskDB(settings=settings)), raise_server_exceptions=False)


def test_search_endpoint_handles_query_error():
    client = client_for(offline_settings())
    # Question too short -> QueryError -> 400 status code, before any DB touch.
    response = client.post("/api/v1/search", json={"query": "x"})
    assert response.status_code == 400
    data = response.json()
    assert data["error"] == "QueryError"
    assert "request_id" in data


def test_search_endpoint_handles_risky_query():
    client = client_for(offline_settings())
    response = client.post("/api/v1/search", json={"query": "orders; drop table users"})
    assert response.status_code == 400
    data = response.json()
    assert data["error"] == "QueryError"
    assert "disallowed symbols" in data["message"]


def test_search_endpoint_handles_query_too_long():
    settings = offline_settings(max_query_length=10)
    client = client_for(settings)
    response = client.post("/api/v1/search", json={"query": "a" * 11})
    assert response.status_code == 400
    data = response.json()
    assert data["error"] == "QueryError"
    assert "too long" in data["message"].lower()
    assert str(settings.max_query_length) in data["message"]


def test_search_route_is_exposed():
    client = client_for(offline_settings())
    paths = [route.path for route in client.app.routes]
    assert "/api/v1/search" in paths
    assert "/api/v1/schema" in paths
    assert "/health" in paths
