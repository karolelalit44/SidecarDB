"""The plug-and-play client: construction, lifecycle, honest health."""

import types

import pytest
from pydantic import ValidationError

from core import AskDB, Settings
from core.constants import DATABASE_DOWN, STATUS_DEGRADED
from core.exceptions import ConfigurationError
from core.schema import SearchResponse
from infra.db import Database


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
        AskDB(_env_file=None)

    askdb = AskDB(_env_file=None, database_url="postgresql://x/x", llm_api_key="sk-test")
    assert askdb.settings.db_schema == "public"
    assert askdb.settings.expose_sql is False


def test_environment_variables_alone_are_enough(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://env/env")
    monkeypatch.setenv("LLM_API_KEY", "sk-env")
    askdb = AskDB(_env_file=None)
    assert askdb.settings.database_url == "postgresql://env/env"
    assert askdb.settings.llm_api_key == "sk-env"


def test_a_settings_instance_and_overrides_are_mutually_exclusive():
    settings = Settings(_env_file=None, database_url="postgresql://x/x", llm_api_key="k")
    with pytest.raises(ValueError):
        AskDB(settings, database_url="postgresql://y/y")


async def test_start_without_a_database_degrades_instead_of_crashing():
    """The service must come up even when the database is down at boot."""
    askdb = AskDB(offline_settings())
    await askdb.start(require_database=False)

    health = await askdb.health()
    assert health.status == STATUS_DEGRADED
    assert health.database == DATABASE_DOWN
    assert health.version

    await askdb.stop()


async def test_start_with_a_required_database_raises_when_unreachable():
    askdb = AskDB(offline_settings())
    with pytest.raises(ConfigurationError):
        await askdb.start(require_database=True)
    await askdb.stop()


async def test_search_before_start_bootstraps_the_engine(monkeypatch):
    """A caller may simply ask; the engine starts itself.

    The bootstrap boot is the degraded kind: an unreachable database must not
    abort a question that would have failed validation anyway.
    """
    askdb = AskDB(offline_settings())
    started = []

    async def fake_start(*, require_database=True):
        started.append(require_database)

    async def fake_search(question, request_id=None):
        return SearchResponse(query=question, answer="ok")

    monkeypatch.setattr(askdb, "start", fake_start)
    monkeypatch.setattr(askdb.service, "search", fake_search)

    result = await askdb.search("how many orders")
    assert started == [False]
    assert result.answer == "ok"


async def test_stop_is_idempotent():
    askdb = AskDB(offline_settings())
    await askdb.start(require_database=False)
    await askdb.stop()
    await askdb.stop()


def test_database_connected_follows_pool_state():
    """connected() must reflect the real asyncpg pool API (is_closing)."""
    database = Database(offline_settings())
    assert database.connected is False

    stub = types.SimpleNamespace(is_closing=lambda: False)
    database._pool = stub
    assert database.connected is True

    stub.is_closing = lambda: True
    assert database.connected is False
