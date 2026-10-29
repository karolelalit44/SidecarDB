import asyncio
import contextlib
import time
from typing import TYPE_CHECKING, Any

from core.config import Settings
from core.constants import (
    BACKOFF_FACTOR,
    DATABASE_DOWN,
    DATABASE_UP,
    EMBEDDING_OFF,
    PROVIDER_NONE,
    STATUS_DEGRADED,
    STATUS_OK,
)
from core.logging_config import get_logger
from core.schema import HealthResponse, SearchResponse
from core.service import SearchService
from core.types import SchemaGraph
from core.verification import StartupVerificationService
from infra.db import Database
from infra.embeddings import Embedder

if TYPE_CHECKING:
    from fastapi import FastAPI

logger = get_logger("client")


def _detect_version() -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("askdb")
    except PackageNotFoundError:
        return "0+local"


class AskDB:
    """A natural-language search engine over one PostgreSQL schema.

    The whole configuration is two required values; construct, start, ask::

        async with AskDB(database_url="postgresql://...", llm_api_key="sk-...") as db:
            answer = await db.search("how many open cases were filed this week?")

    Nothing is imported from the environment implicitly unless you want it to
    be: with no arguments, values resolve from environment variables and an
    optional ``.env`` file, exactly like pydantic-settings.
    """

    def __init__(self, settings: Settings | None = None, **overrides: Any) -> None:
        if settings is not None and overrides:
            raise ValueError("Pass either a Settings instance or keyword overrides, not both.")
        self.settings = settings if settings is not None else Settings(**overrides)
        self.version = _detect_version()
        self.database = Database(self.settings)
        self.embedder = Embedder(self.settings)
        self.service = SearchService(self.settings, self.database, self.embedder)
        self._started = False
        self._recovery_task: asyncio.Task | None = None
        self._stop_event = asyncio.Event()
        self._last_ping: tuple[float, bool] | None = None

    # --- lifecycle ---------------------------------------------------------

    async def start(self, *, require_database: bool = True) -> None:
        """Verify configuration, connect, introspect, build the index.

        With require_database=False a database outage does not raise; the
        instance goes degraded and a background task keeps retrying. That is
        what the HTTP service wants, so a restart storm never hides a
        recoverable failure.
        """
        if self._started:
            return
        StartupVerificationService(self.settings).verify()
        if not self.settings.api_keys:
            logger.warning(
                "No API keys are configured (API_KEYS); the HTTP service will accept "
                "unauthenticated requests. Set at least one key before exposing it."
            )
        try:
            await self.database.connect()
            await self.service.refresh()
        except Exception:
            if require_database:
                raise
            logger.error(
                "Startup continuing without a database; retrying every %.0fs.",
                self.settings.recovery_interval_seconds,
            )
            self._recovery_task = asyncio.create_task(self._recover())
        self._started = True

    async def _recover(self) -> None:
        delay = self.settings.db_connect_backoff_seconds
        while not self._stop_event.is_set():
            try:
                await self.database.connect()
                await self.service.refresh()
                logger.info("Database recovered; the catalog is mapped again.")
                return
            except Exception as exc:
                logger.debug(
                    "recovery attempt failed: %s: %s; next try in %.1fs",
                    type(exc).__name__,
                    exc,
                    delay,
                )
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self._stop_event.wait(), timeout=delay)
            delay *= BACKOFF_FACTOR

    async def stop(self) -> None:
        self._stop_event.set()
        if self._recovery_task is not None:
            self._recovery_task.cancel()
            self._recovery_task = None
        if self._started:
            await self.service.shutdown()
            self._started = False
        logger.debug("client: stopped")

    async def __aenter__(self) -> "AskDB":
        await self.start()
        return self

    async def __aexit__(self, *_exc_info: object) -> None:
        await self.stop()

    # --- query surface -----------------------------------------------------

    async def search(self, question: str, request_id: str | None = None) -> SearchResponse:
        """Answer one question in plain English.

        Starts the engine on first use, so callers that prefer a single call
        site need not manage the lifecycle explicitly. The bootstrap tolerates
        an unreachable database: validation failures reach the caller as their
        own 400s, and a question that genuinely needs the database surfaces
        the connection error when introspection runs.
        """
        if not self._started:
            await self.start(require_database=False)
        return await self.service.search(question, request_id)

    async def refresh(self) -> SchemaGraph:
        """Re-introspect the catalog and rebuild the schema index."""
        return await self.service.refresh()

    async def schema(self) -> SchemaGraph:
        """The currently mapped catalog."""
        return await self.service.schema()

    # --- health ------------------------------------------------------------

    async def health(self) -> HealthResponse:
        """A real liveness answer: the database is pinged, not assumed."""
        connected = await self._fresh_ping()
        healthy = self._started and connected
        provider = self.settings.embedding_provider
        return HealthResponse(
            status=STATUS_OK if healthy else STATUS_DEGRADED,
            app=self.settings.app_name,
            version=self.version,
            database=DATABASE_UP if connected else DATABASE_DOWN,
            tables=len(self.service.catalog_names()),
            model=self.settings.llm_model,
            embedding=EMBEDDING_OFF if provider == PROVIDER_NONE else self.settings.embedding_model,
        )

    async def _fresh_ping(self) -> bool:
        ttl = self.settings.health_ping_cache_seconds
        now = time.monotonic()
        if self._last_ping is not None:
            cached_at, value = self._last_ping
            if now - cached_at < ttl:
                return value
        value = await self.database.ping()
        self._last_ping = (time.monotonic(), value)
        return value

    # --- server ------------------------------------------------------------

    def create_app(self) -> "FastAPI":
        """Build the FastAPI service around this instance."""
        from core.server import create_app

        return create_app(self)
