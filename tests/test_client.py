"""The plug-and-play client: construction, lifecycle, honest health."""

import types

import pytest
from pydantic import ValidationError

from sidecardb import Settings, SidecarDB
from sidecardb.constants import DATABASE_DOWN, STATUS_DEGRADED
from sidecardb.db import Database
from sidecardb.exceptions import ConfigurationError
from sidecardb.schema import SearchResponse


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


def test_configuration_is_two_keyword_arguments(monkeypatch):
    """The headline promise: two values in, working engine out."""
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    with pytest.raises(ValidationError):
        SidecarDB(_env_file=None)

    sidecar = SidecarDB(_env_file=None, database_url="postgresql://x/x", llm_api_key="sk-test")
    assert sidecar.settings.db_schema == "public"
    assert sidecar.settings.expose_sql is False


def test_environment_variables_alone_are_enough(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://env/env")
    monkeypatch.setenv("LLM_API_KEY", "sk-env")
    sidecar = SidecarDB(_env_file=None)
    assert sidecar.settings.database_url == "postgresql://env/env"
    assert sidecar.settings.llm_api_key == "sk-env"


def test_a_settings_instance_and_overrides_are_mutually_exclusive():
    settings = Settings(_env_file=None, database_url="postgresql://x/x", llm_api_key="k")
    with pytest.raises(ValueError):
        SidecarDB(settings, database_url="postgresql://y/y")


async def test_start_without_a_database_degrades_instead_of_crashing():
    """The service must come up even when the database is down at boot."""
    sidecar = SidecarDB(offline_settings())
    await sidecar.start(require_database=False)

    health = await sidecar.health()
    assert health.status == STATUS_DEGRADED
    assert health.database == DATABASE_DOWN
    assert health.version

    await sidecar.stop()


async def test_start_with_a_required_database_raises_when_unreachable():
    sidecar = SidecarDB(offline_settings())
    with pytest.raises(ConfigurationError):
        await sidecar.start(require_database=True)
    await sidecar.stop()


async def test_search_before_start_bootstraps_the_engine(monkeypatch):
    """A caller may simply ask; the engine starts itself.

    The bootstrap boot is the degraded kind: an unreachable database must not
    abort a question that would have failed validation anyway.
    """
    sidecar = SidecarDB(offline_settings())
    started = []

    async def fake_start(*, require_database=True):
        started.append(require_database)

    async def fake_search(question, request_id=None):
        return SearchResponse(query=question, answer="ok")

    monkeypatch.setattr(sidecar, "start", fake_start)
    monkeypatch.setattr(sidecar.service, "search", fake_search)

    result = await sidecar.search("how many orders")
    assert started == [False]
    assert result.answer == "ok"


async def test_stop_is_idempotent():
    sidecar = SidecarDB(offline_settings())
    await sidecar.start(require_database=False)
    await sidecar.stop()
    await sidecar.stop()


def test_database_connected_follows_pool_state():
    """connected() must reflect the real asyncpg pool API (is_closing)."""
    database = Database(offline_settings())
    assert database.connected is False

    stub = types.SimpleNamespace(is_closing=lambda: False)
    database._pool = stub
    assert database.connected is True

    stub.is_closing = lambda: True
    assert database.connected is False
