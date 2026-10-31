import asyncio
import contextlib
import time
from typing import Any

from catalog.manager import CatalogManager
from catalog.models import SchemaGraph
from config.constants import (
    APP_NAME,
    DATABASE_DOWN,
    DATABASE_UP,
    ELAPSED_PRECISION,
    ERROR_MESSAGE_MAX_CHARS,
    EXPOSE_SQL,
    HEALTH_PING_CACHE_SECONDS,
    LOG_SQL,
    NO_ANSWER_MESSAGE,
    NO_RECORDS_MESSAGE,
    RECOVERY_INTERVAL_SECONDS,
    STATUS_DEGRADED,
    STATUS_OK,
)
from config.settings import Settings, get_settings
from core.exceptions import ConfigurationError
from core.logging import get_logger
from core.pipeline import run_search
from core.protocols import DatabaseGateway, Embedder, Planner
from guard.question import validate_question
from planner.models import Outcome, PlanPolicy
from providers.embeddings import Embedder as DefaultEmbedder
from providers.llm import LLMClient
from retrieval.memory import SchemaMemory
from server.schemas import HealthResponse, SearchResponse
from storage.connection import Database
from synthesis.formatter import bind_template

logger = get_logger("client")


def _detect_version() -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("sidecardb")
    except PackageNotFoundError:
        return "0+local"


class SidecarDB:
    def __init__(
        self,
        settings: Settings | None = None,
        *,
        planner: Planner | None = None,
        embedder: Embedder | None = None,
        database: DatabaseGateway | None = None,
        catalog: DatabaseGateway | None = None,
        **overrides: Any,
    ) -> None:
        env_file = overrides.pop("_env_file", None)
        if settings is not None and overrides:
            raise ValueError("Pass either a Settings instance or keyword overrides, not both.")
        self.settings = (
            settings if settings is not None else get_settings(_env_file=env_file, **overrides)
        )
        self.version = _detect_version()

        effective_db = database if database is not None else catalog
        self._owns_planner = planner is None
        self._owns_embedder = embedder is None
        self._owns_database = effective_db is None
        self.planner = planner if planner is not None else LLMClient(self.settings)
        self.embedder = embedder if embedder is not None else DefaultEmbedder(self.settings)
        self.database = effective_db if effective_db is not None else Database(self.settings)

        logger.debug(
            "SidecarDB initialized (version=%s schema=%s embedding=%s model=%s database=%s)",
            self.version,
            self.settings.db_schema,
            self.settings.embedding_model,
            self.settings.llm_model or "unset",
            type(self.database).__name__,
        )

        self.plan_policy = PlanPolicy()
        self._catalog_manager = CatalogManager(
            database=self.database,
            embedder=self.embedder,
            policy=self.plan_policy,
            storage_dir=self.settings.storage_dir,
        )

        self._started = False
        self._recovery_task: asyncio.Task | None = None
        self._stop_event = asyncio.Event()
        self._last_ping: tuple[float, bool] | None = None
        self._graph: SchemaGraph | None = None
        self._index: SchemaMemory | None = None

    def _verify_configuration(self) -> None:
        settings = self.settings
        logger.debug("verification: checking required database, LLM, and embedding settings")
        if self._owns_database and not settings.database_url:
            raise ConfigurationError(
                "Mandatory dependency missing: DATABASE_URL is not configured."
            )
        logger.debug("verification: DATABASE_URL is configured")
        if self._owns_planner:
            if not (settings.llm_api_key and settings.llm_api_key.strip()):
                raise ConfigurationError(
                    "Mandatory dependency missing: LLM_API_KEY is not configured in environment."
                )
            if not (settings.llm_base_url and settings.llm_base_url.strip()):
                raise ConfigurationError(
                    "Mandatory dependency missing: LLM_BASE_URL is not configured in environment."
                )
            if not (settings.llm_model and settings.llm_model.strip()):
                raise ConfigurationError(
                    "Mandatory dependency missing: LLM_MODEL is not configured in environment."
                )
            logger.debug(
                "verification: LLM provider configured with model=%s, base_url=%s",
                settings.llm_model,
                settings.llm_base_url,
            )
        if self._owns_embedder:
            logger.debug(
                "verification: Embeddings configured (model=%s, dim=%d, path=%s)",
                settings.embedding_model,
                settings.embedding_dimensions,
                settings.embedding_local_path,
            )

    async def start(self, *, require_database: bool = True) -> "SidecarDB":
        logger.debug(
            "start requested (require_database=%s, started=%s)",
            require_database,
            self._started,
        )
        if self._started:
            logger.debug("engine: already started, ignoring redundant start request")
            return self

        self._verify_configuration()
        self._stop_event.clear()

        restored = self._catalog_manager.restore()
        if restored is not None:
            self._graph, self._index = restored
            logger.info(
                "Restored schema from %s; ready to serve queries immediately.",
                self.settings.storage_dir,
            )

        try:
            logger.debug("engine: connecting to database...")
            if hasattr(self.database, "connect"):
                await self.database.connect()
            logger.debug("engine: database connected successfully")
            if self._graph is None:
                await self._refresh_catalog()
        except Exception as exc:
            if require_database:
                logger.error(
                    "engine: mandatory database connection failed: %s: %s", type(exc).__name__, exc
                )
                await self.stop()
                raise
            logger.warning(
                "Database could not be reached at startup (%s). Starting in degraded mode; "
                "a background worker will retry every %ds.",
                exc,
                RECOVERY_INTERVAL_SECONDS,
            )
            self._recovery_task = asyncio.create_task(
                self._recovery_loop(), name="sidecardb-recovery"
            )

        self._started = True
        logger.debug("engine: start complete (healthy=%s)", self._recovery_task is None)
        return self

    async def stop(self) -> None:
        logger.debug("stop requested (started=%s)", self._started)
        self._stop_event.set()
        if self._recovery_task is not None:
            self._recovery_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._recovery_task
            self._recovery_task = None

        if self._owns_database:
            logger.debug("engine: closing database connection...")
            await self.database.close()
        if self._owns_embedder:
            logger.debug("engine: closing embedder...")
            await self.embedder.aclose()

        self._started = False
        self._graph = None
        self._index = None
        self._last_ping = None
        logger.debug("engine: stop complete")

    async def __aenter__(self) -> "SidecarDB":
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.stop()

    async def _recovery_loop(self) -> None:
        logger.debug("recovery_loop: started (interval=%ds)", RECOVERY_INTERVAL_SECONDS)
        while not self._stop_event.is_set():
            try:
                await asyncio.sleep(RECOVERY_INTERVAL_SECONDS)
            except asyncio.CancelledError:
                break

            if self._stop_event.is_set():
                break

            try:
                logger.debug("recovery_loop: attempting database reconnect...")
                if hasattr(self.database, "connect"):
                    await self.database.connect()
                logger.debug("recovery_loop: database reconnected; refreshing catalog...")
                await self._refresh_catalog()
                logger.info("Database recovered; sidecarDB is fully available.")
                self._recovery_task = None
                return
            except Exception as exc:
                logger.debug("Recovery attempt failed: %s", exc)

    async def _refresh_catalog(self) -> SchemaGraph:
        logger.debug("catalog refresh requested via CatalogManager")
        self._graph, self._index = await self._catalog_manager.refresh()
        return self._graph

    async def search(self, question: str, request_id: str | None = None) -> SearchResponse:
        started = time.perf_counter()
        logger.debug("[%s] search initiated: raw question=%r", request_id, question)
        cleaned_question = validate_question(question)
        logger.debug("[%s] question validated successfully: %r", request_id, cleaned_question)

        if not self._started:
            logger.debug(
                "[%s] engine not started; calling start(require_database=True)", request_id
            )
            await self.start()

        graph = await self.catalog()
        outcome = await self._run_pipeline(cleaned_question, graph)
        return self._respond(outcome, cleaned_question, request_id, started)

    async def _run_pipeline(self, question: str, graph: SchemaGraph) -> Outcome:
        return await run_search(
            question=question,
            graph=graph,
            index=self._index,
            planner=self.planner,
            database=self.database,
            policy=self.plan_policy,
            schema=self.settings.db_schema,
        )

    def _respond(
        self, outcome: Outcome, question: str, request_id: str | None, started: float
    ) -> SearchResponse:
        rows = outcome.rows

        if not rows:
            answer = NO_RECORDS_MESSAGE
            logger.debug(
                "[%s] no rows returned; using configured no-records message: %r",
                request_id,
                answer,
            )
        else:
            logger.debug(
                "[%s] binding template %r with %d row(s)",
                request_id,
                outcome.plan.template,
                len(rows),
            )
            answer = bind_template(
                outcome.plan.template or "",
                rows,
                outcome.plan.intent,
                max_chars=ERROR_MESSAGE_MAX_CHARS,
            )
            if not answer:
                answer = NO_ANSWER_MESSAGE
                logger.debug(
                    "[%s] template produced empty answer; using no-answer message: %r",
                    request_id,
                    answer,
                )
            else:
                logger.debug("[%s] template successfully bound answer: %r", request_id, answer)

        elapsed = round((time.perf_counter() - started) * 1000, ELAPSED_PRECISION)
        logger.debug(
            "[%s] response ready: rows=%d answer_chars=%d expose_sql=%s",
            request_id,
            len(rows),
            len(answer),
            EXPOSE_SQL,
        )
        if LOG_SQL:
            logger.info(
                "[%s] intent=%s mode=%s rows=%d tables=%s sql=%s bound_params=%d",
                request_id,
                outcome.plan.intent.value,
                outcome.retrieval_mode.value,
                len(rows),
                outcome.retrieved_tables,
                outcome.statement.sql,
                len(outcome.statement.params),
            )
        else:
            logger.info(
                "[%s] intent=%s mode=%s rows=%d tables=%s bound_params=%d "
                "(LOG_SQL is off; SQL withheld)",
                request_id,
                outcome.plan.intent.value,
                outcome.retrieval_mode.value,
                len(rows),
                outcome.retrieved_tables,
                len(outcome.statement.params),
            )

        return SearchResponse(
            question=question,
            answer=answer,
            sql=outcome.statement.sql if EXPOSE_SQL else None,
            row_count=len(rows),
            elapsed_ms=elapsed,
            records=rows,
            retrieved_tables=outcome.retrieved_tables,
            retrieval_mode=outcome.retrieval_mode,
            request_id=request_id,
        )

    async def catalog(self, refresh: bool = False) -> SchemaGraph:
        logger.debug("catalog requested (refresh=%s, cached=%s)", refresh, self._graph is not None)
        if refresh or self._graph is None:
            return await self._refresh_catalog()
        return self._graph

    async def health(self) -> HealthResponse:
        logger.debug("health check initiated")
        connected = await self._ping()
        healthy = self._started and connected
        response = HealthResponse(
            status=STATUS_OK if healthy else STATUS_DEGRADED,
            app=APP_NAME,
            version=self.version,
            database=DATABASE_UP if connected else DATABASE_DOWN,
            tables=len(self._graph.names) if self._graph is not None else 0,
            model=self.settings.llm_model,
            embedding=self.settings.embedding_model,
        )
        logger.debug(
            "health check resolved: status=%s database=%s tables=%d",
            response.status,
            response.database,
            response.tables,
        )
        return response

    async def _ping(self) -> bool:
        ttl = HEALTH_PING_CACHE_SECONDS
        now = time.monotonic()
        if self._last_ping is not None:
            cached_at, value = self._last_ping
            if now - cached_at < ttl:
                logger.debug(
                    "db ping check: using cached result %s (age=%.2fs)", value, now - cached_at
                )
                return value
        logger.debug("db ping check: executing live ping...")
        value = await self.database.ping()
        self._last_ping = (time.monotonic(), value)
        logger.debug("db ping check: result=%s", value)
        return value


__all__ = ["SidecarDB", "_detect_version"]
