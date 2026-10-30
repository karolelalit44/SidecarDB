import asyncio
import contextlib
import time
from pathlib import Path
from typing import Any

from core.answer import bind_template
from core.capabilities import DatabaseGateway, Embedder, Planner
from core.catalog import SchemaGraph
from core.config import Settings, get_settings
from core.constants import (
    APP_NAME,
    DATABASE_DOWN,
    DATABASE_UP,
    ELAPSED_PRECISION,
    EMBEDDING_OFF,
    ERROR_MESSAGE_MAX_CHARS,
    EXPOSE_SQL,
    HEALTH_PING_CACHE_SECONDS,
    LOG_LLM_PAYLOADS,
    LOG_SQL,
    NO_ANSWER_MESSAGE,
    NO_RECORDS_MESSAGE,
    PROVIDER_API,
    PROVIDER_LOCAL,
    PROVIDER_NONE,
    RECOVERY_INTERVAL_SECONDS,
    STATUS_DEGRADED,
    STATUS_OK,
)
from core.exceptions import ConfigurationError
from core.guard import statement_for, validate_question
from core.plan import Outcome, PlanPolicy
from core.planning import generate_plan
from core.retrieval import SchemaIndex, link
from core.wire import (
    HealthResponse,
    SearchResponse,
)
from infra import embeddings
from infra.db import Database
from infra.llm import LLMClient
from utils.logging_config import get_logger

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
        self.embedder = embedder if embedder is not None else embeddings.Embedder(self.settings)
        self.database = effective_db if effective_db is not None else Database(self.settings)

        logger.debug(
            "SidecarDB initialized (version=%s schema=%s provider=%s model=%s database=%s)",
            self.version,
            self.settings.db_schema,
            self.settings.embedding_provider,
            self.settings.llm_model or "unset",
            type(self.database).__name__,
        )

        self.plan_policy = PlanPolicy()

        self._started = False
        self._recovery_task: asyncio.Task | None = None
        self._stop_event = asyncio.Event()
        self._last_ping: tuple[float, bool] | None = None
        self._graph: SchemaGraph | None = None
        self._index: SchemaIndex | None = None

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
                "verification: LLM settings valid (model=%s, base_url=%s)",
                settings.llm_model,
                settings.llm_base_url,
            )
        logger.info("LLM configuration verification successful (model: %s).", settings.llm_model)
        if self._owns_embedder:
            self._verify_embeddings(settings)
        logger.debug("verification: %d api_keys configured", len(settings.api_keys))
        logger.info("Startup prerequisite verification complete.")

    @staticmethod
    def _verify_embeddings(settings: Settings) -> None:
        logger.debug(
            "verification: embeddings provider=%s model=%s local=%s device=%s",
            settings.embedding_provider,
            settings.embedding_model,
            settings.embedding_local_path or "unset",
            settings.embedding_device,
        )
        if settings.embedding_provider == PROVIDER_NONE:
            logger.info("Embeddings are off; retrieval will use catalog order.")
            return
        if not settings.embedding_model:
            raise ConfigurationError(
                "Mandatory dependency missing: EMBEDDING_MODEL is not configured in environment."
            )
        if settings.embedding_provider == PROVIDER_API and not settings.embedding_api_base_url:
            raise ConfigurationError(
                "Mandatory dependency missing: "
                "EMBEDDING_API_BASE_URL is not configured in environment."
            )
        if (
            settings.embedding_provider == PROVIDER_LOCAL
            and settings.embedding_local_path
            and not Path(settings.embedding_local_path).is_dir()
        ):
            logger.warning(
                "EMBEDDING_LOCAL_PATH %s does not exist; the model will be fetched from the Hub.",
                settings.embedding_local_path,
            )

    async def start(self, *, require_database: bool = True) -> None:
        if self._started:
            logger.debug("SidecarDB.start: already started, skipping initialization")
            return
        logger.debug(
            "SidecarDB.start: starting engine (require_database=%s, schema=%s)",
            require_database,
            self.settings.db_schema,
        )
        self._verify_configuration()
        if not self.settings.api_keys:
            logger.warning(
                "No API keys are configured (API_KEYS); the HTTP service will accept "
                "unauthenticated requests. Set at least one key before exposing it."
            )
        try:
            await self._refresh_catalog()
        except Exception:
            if require_database:
                raise
            logger.error(
                "Startup continuing without a catalog; retrying every %.0fs.",
                RECOVERY_INTERVAL_SECONDS,
            )
            self._recovery_task = asyncio.create_task(self._recover())
        self._started = True
        logger.debug("SidecarDB.start: engine successfully started")

    async def _refresh_catalog(self) -> SchemaGraph:
        started = time.perf_counter()
        logger.debug("refresh: introspecting the catalog of schema %s", self.settings.db_schema)
        graph = await self.database.introspect()
        logger.debug(
            "refresh: %d tables introspected in %d ms; rebuilding the schema index",
            len(graph.tables),
            round((time.perf_counter() - started) * 1000, ELAPSED_PRECISION),
        )
        logger.debug("refresh: table names discovered: %s", ", ".join(graph.tables.keys()))
        self._graph = graph
        self._index = await SchemaIndex(self.embedder, self.plan_policy).build(
            graph, cache_dir=self.settings.cache_dir
        )
        logger.debug(
            "refresh: schema index %s",
            "ready" if self._index is not None and self._index.ready else "unavailable",
        )
        logger.info("Catalog mapped: %d tables", len(graph.tables))
        return graph

    async def _recover(self) -> None:
        interval = RECOVERY_INTERVAL_SECONDS
        logger.debug("SidecarDB._recover: recovery task initiated (interval=%.1fs)", interval)
        while not self._stop_event.is_set():
            try:
                await self._refresh_catalog()
                logger.info("Database recovered; the catalog is mapped again.")
                return
            except Exception as exc:
                logger.debug(
                    "recovery attempt failed: %s: %s; next try in %.1fs",
                    type(exc).__name__,
                    exc,
                    interval,
                )
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self._stop_event.wait(), timeout=interval)

    async def stop(self) -> None:
        logger.debug("SidecarDB.stop: stopping engine (started=%s)", self._started)
        self._stop_event.set()
        if self._recovery_task is not None:
            self._recovery_task.cancel()
            self._recovery_task = None
        if not self._started:
            logger.debug("SidecarDB.stop: not started, returning")
            return
        self._started = False
        if self._owns_database:
            await self.database.close()
        if self._owns_embedder:
            await self.embedder.aclose()
        self._graph = None
        self._index = None
        logger.debug("client: stopped; cleanup complete")

    async def __aenter__(self) -> "SidecarDB":
        await self.start()
        return self

    async def __aexit__(self, *_exc_info: object) -> None:
        await self.stop()

    async def search(self, question: str, request_id: str | None = None) -> SearchResponse:
        logger.debug(
            "[%s] search pipeline initiated: question=%r",
            request_id,
            question,
        )
        clean_question = validate_question(question)

        if not self._started:
            logger.debug("[%s] engine not started yet; starting now", request_id)
            await self.start(require_database=False)
        graph = self._graph
        if graph is None:
            logger.debug("[%s] catalog not present in memory; refreshing now", request_id)
            graph = await self._refresh_catalog()

        started = time.perf_counter()
        try:
            outcome = await self._run_pipeline(clean_question, graph)
        except Exception as exc:
            elapsed = round((time.perf_counter() - started) * 1000, ELAPSED_PRECISION)
            logger.error(
                "[%s] search failed after %s ms: %s: %s",
                request_id,
                elapsed,
                type(exc).__name__,
                exc,
                exc_info=LOG_LLM_PAYLOADS,
            )
            raise
        return self._respond(outcome, clean_question, request_id, started)

    async def _run_pipeline(self, question: str, graph: SchemaGraph) -> Outcome:
        logger.debug("pipeline: beginning execution for question: %r", question)
        logger.debug("pipeline: step 1 - schema linking (retrieval)")
        tables, retrieval_mode, retrieved = await link(
            question, graph, self._index, self.plan_policy
        )
        logger.debug(
            "pipeline: step 1 complete - mode=%s retrieved_tables=%s final_tables=%s",
            retrieval_mode.value,
            retrieved,
            tables,
        )

        logger.debug(
            "pipeline: step 2 - generating query plan with %d table candidate(s)", len(tables)
        )
        plan = await generate_plan(
            question=question,
            graph=graph,
            tables=tables,
            planner=self.planner,
            policy=self.plan_policy,
        )
        logger.debug(
            "pipeline: step 2 complete - plan accepted: intent=%s table=%s projection=%d "
            "filter=%s %s",
            plan.intent.value,
            plan.table,
            len(plan.projection),
            plan.filter_column,
            plan.filter_operator,
        )

        logger.debug("pipeline: step 3 - compiling plan to guarded SQL statement")
        statement = statement_for(plan, graph, self.settings.db_schema)
        logger.debug(
            "pipeline: step 3 complete - statement composed: sql=%s, params_count=%d",
            statement.sql,
            len(statement.params),
        )

        logger.debug("pipeline: step 4 - executing statement via database gateway")
        rows = await self.database.execute(statement)
        logger.debug("pipeline: step 4 complete - fetched %d row(s)", len(rows))

        return Outcome(
            plan=plan,
            statement=statement,
            rows=rows,
            retrieved_tables=retrieved,
            retrieval_mode=retrieval_mode,
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
        provider = self.settings.embedding_provider
        response = HealthResponse(
            status=STATUS_OK if healthy else STATUS_DEGRADED,
            app=APP_NAME,
            version=self.version,
            database=DATABASE_UP if connected else DATABASE_DOWN,
            tables=len(self._graph.names) if self._graph is not None else 0,
            model=self.settings.llm_model,
            embedding=EMBEDDING_OFF if provider == PROVIDER_NONE else self.settings.embedding_model,
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
