"""The HTTP edge: authentication, rate limiting, payload caps, request ids."""

from fastapi.testclient import TestClient

from sidecardb import SidecarDB
from sidecardb.config import Settings
from sidecardb.server import create_app

SECRET_KEY = "sekret-key"


def offline_settings(**overrides) -> Settings:
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


def client_for(settings: Settings) -> TestClient:
    return TestClient(create_app(SidecarDB(settings=settings)), raise_server_exceptions=False)


# --- authentication ---------------------------------------------------------


def test_missing_key_is_refused():
    client = client_for(offline_settings(api_keys=[SECRET_KEY]))
    response = client.post("/api/v1/search", json={"query": "how many orders"})
    assert response.status_code == 401
    assert response.json()["error"] == "Unauthorized"


def test_wrong_key_is_refused():
    client = client_for(offline_settings(api_keys=[SECRET_KEY]))
    response = client.post(
        "/api/v1/search",
        json={"query": "how many orders"},
        headers={"Authorization": "Bearer not-the-key"},
    )
    assert response.status_code == 401


def test_bearer_key_passes_authentication():
    """Passing auth means reaching the handler: a short query then 400s."""
    client = client_for(offline_settings(api_keys=[SECRET_KEY]))
    response = client.post(
        "/api/v1/search",
        json={"query": "x"},
        headers={"Authorization": f"Bearer {SECRET_KEY}"},
    )
    assert response.status_code == 400
    assert response.json()["error"] == "QueryError"


def test_api_key_header_passes_authentication():
    client = client_for(offline_settings(api_keys=[SECRET_KEY]))
    response = client.post(
        "/api/v1/search",
        json={"query": "x"},
        headers={"X-API-Key": SECRET_KEY},
    )
    assert response.status_code == 400


def test_schema_endpoint_is_protected_too():
    client = client_for(offline_settings(api_keys=[SECRET_KEY]))
    assert client.get("/api/v1/schema").status_code == 401


def test_health_stays_open_for_probes():
    """Orchestrators cannot hold a key; health must answer without one."""
    client = client_for(offline_settings(api_keys=[SECRET_KEY]))
    response = client.get("/health")
    # No database in the test environment, so the honest answer is degraded.
    assert response.status_code == 503
    assert response.json()["status"] == "degraded"


def test_no_keys_configured_means_open_but_degraded_service_runs():
    client = client_for(offline_settings(api_keys=[]))
    response = client.post("/api/v1/search", json={"query": "x"})
    assert response.status_code == 400


# --- rate limiting ----------------------------------------------------------


def test_rate_limit_returns_429_with_retry_after():
    # A short query fails validation deterministically, so the rate limiting
    # is exercised without a database in the loop.
    client = client_for(offline_settings(rate_limit_requests=2))
    payloads = [{"query": "x"}] * 3
    statuses = [client.post("/api/v1/search", json=p).status_code for p in payloads]
    assert statuses[:2] == [400, 400]
    assert statuses[2] == 429
    limited = client.post("/api/v1/search", json=payloads[0])
    assert "Retry-After" in limited.headers


def test_rate_limit_can_be_disabled():
    client = client_for(offline_settings(rate_limit_enabled=False, rate_limit_requests=1))
    statuses = [client.post("/api/v1/search", json={"query": "x"}).status_code for _ in range(3)]
    assert statuses == [400, 400, 400]


# --- payload caps -----------------------------------------------------------


def test_oversized_body_is_refused_with_413():
    client = client_for(offline_settings(max_request_body_bytes=64))
    response = client.post("/api/v1/search", json={"query": "a" * 500})
    assert response.status_code == 413
    assert response.json()["error"] == "PayloadTooLarge"


# --- request ids ------------------------------------------------------------


def test_a_clean_client_request_id_is_echoed():
    client = client_for(offline_settings())
    response = client.post(
        "/api/v1/search", json={"query": "x"}, headers={"X-Request-ID": "trace-123"}
    )
    assert response.headers["X-Request-ID"] == "trace-123"
    assert response.json()["request_id"] == "trace-123"


def test_a_hostile_request_id_is_replaced_not_echoed():
    """A crafted header must not be written into the logs verbatim."""
    client = client_for(offline_settings())
    response = client.post(
        "/api/v1/search",
        json={"query": "x"},
        headers={"X-Request-ID": "fake id with spaces and $stuff"},
    )
    assert response.headers["X-Request-ID"] != "fake id with spaces and $stuff"


def test_an_oversized_request_id_is_replaced():
    client = client_for(offline_settings())
    response = client.post(
        "/api/v1/search", json={"query": "x"}, headers={"X-Request-ID": "a" * 500}
    )
    assert response.headers["X-Request-ID"] != "a" * 500
