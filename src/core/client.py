import asyncio
import contextlib
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from core.answer import render_template
from core.capabilities import Catalog, Embedder, Planner, Runner
from core.catalog import SchemaGraph
from core.config import PROVIDER_LOCAL, PROVIDER_NONE, Settings
from core.constants import (
    APP_NAME,
    ELAPSED_PRECISION,
    ERROR_MESSAGE_MAX_CHARS,
    EXPOSE_SQL,
    HEALTH_PING_CACHE_SECONDS,
    LOG_LLM_PAYLOADS,
    MAX_QUERY_LENGTH,
    MIN_QUERY_LENGTH,
    NO_RECORDS_MESSAGE,
    NO_SUMMARY_MESSAGE,
    RECOVERY_INTERVAL_SECONDS,
)
from core.exceptions import ConfigurationError, PlanningError, QueryError
from core.guard import statement_for
from core.plan import (
    Outcome,
    PlanPolicy,
    check_question,
    redact,
)
from core.plan import plan as plan_a_question
from core.retrieval import SchemaIndex, link
from core.wire import (
    DATABASE_DOWN,
    DATABASE_UP,
    EMBEDDING_OFF,
    STATUS_DEGRADED,
    STATUS_OK,
    HealthResponse,
    SearchResponse,
)
from infra import embeddings
from infra.db import Database
from infra.execute import Sandbox
from infra.llm import LLMClient
from utils.logging_config import get_logger

if TYPE_CHECKING:
    from fastapi import FastAPI

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
        runner: Runner | None = None,
        catalog: Catalog | None = None,
        **overrides: Any,
    ) -> None:
        if settings is not None and overrides:
            raise ValueError("Pass either a Settings instance or keyword overrides, not both.")
        self.settings = settings if settings is not None else Settings(**overrides)
        self.version = _detect_version()

        self._owns_planner = planner is None
        self._owns_embedder = embedder is None
        self._owns_catalog = catalog is None
        self.planner = planner if planner is not None else LLMClient(self.settings)
        self.embedder = embedder if embedder is not None else embeddings.Embedder(self.settings)
        self._catalog = catalog if catalog is not None else Database(self.settings)
        self.runner = runner if runner is not None else self._default_runner()

        self.plan_policy = PlanPolicy()

        self._started = False
        self._recovery_task: asyncio.Task | None = None
        self._stop_event = asyncio.Event()
        self._last_ping: tuple[float, bool] | None = None
        self._graph: SchemaGraph | None = None
        self._index: SchemaIndex | None = None

    def _default_runner(self) -> Runner:
        if isinstance(self._catalog, Database):
            return Sandbox(self._catalog)
        raise ConfigurationError(
            "No runner: pass runner= (a Statement executor) or leave catalog= unset "
            "so the built-in Database can host the sandbox."
        )

    def _verify_configuration(self) -> None:
        settings = self.settings
        if self._owns_catalog and not settings.database_url:
            raise ConfigurationError(
                "Mandatory dependency missing: DATABASE_URL is not configured."
            )
        logger.debug("verification: DATABASE_URL is configured")
        if self._owns_planner and not (settings.llm_api_key and settings.llm_api_key.strip()):
            raise ConfigurationError(
                "Mandatory dependency missing: LLM_API_KEY is not configured in environment."
            )
        logger.info("LLM configuration verification successful (model: %s).", settings.llm_model)
        if self._owns_embedder:
            self._verify_embeddings(settings)
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
            return
        self._verify_configuration()
        if not self.settings.api_keys:
            logger.warning(
                "No API keys are configured (API_KEYS); the HTTP service will accept "
                "unauthenticated requests. Set at least one key before exposing it."
            )
        try:
            await self._map()
        except Exception:
            if require_database:
                raise
            logger.error(
                "Startup continuing without a catalog; retrying every %.0fs.",
                RECOVERY_INTERVAL_SECONDS,
            )
            self._recovery_task = asyncio.create_task(self._recover())
        self._started = True

    async def _map(self) -> SchemaGraph:
        started = time.perf_counter()
        logger.debug("refresh: introspecting the catalog of schema %s", self.settings.db_schema)
        graph = await self._catalog.introspect()
        logger.debug(
            "refresh: %d tables introspected in %d ms; rebuilding the schema index",
            len(graph.tables),
            round((time.perf_counter() - started) * 1000, ELAPSED_PRECISION),
        )
        self._graph = graph
        self._index = await SchemaIndex(self.embedder, self.plan_policy).build(graph)
        logger.debug(
            "refresh: schema index %s",
            "ready" if self._index is not None and self._index.ready else "unavailable",
        )
        logger.info("Catalog mapped: %d tables", len(graph.tables))
        return graph

    async def _recover(self) -> None:
        interval = RECOVERY_INTERVAL_SECONDS
        while not self._stop_event.is_set():
            try:
                await self._map()
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
        self._stop_event.set()
        if self._recovery_task is not None:
            self._recovery_task.cancel()
            self._recovery_task = None
        if not self._started:
            return
        self._started = False
        if self._owns_catalog:
            catalog = self._catalog
            if isinstance(catalog, Database):
                await catalog.close()
        if self._owns_embedder:
            await self.embedder.aclose()
        self._graph = None
        self._index = None
        logger.debug("client: stopped")

    async def __aenter__(self) -> "SidecarDB":
        await self.start()
        return self

    async def __aexit__(self, *_exc_info: object) -> None:
        await self.stop()

    async def search(self, question: str, request_id: str | None = None) -> SearchResponse:
        query = (question or "").strip()
        logger.debug("[%s] search received %d chars", request_id, len(query))

        reason = self._reject_query(query)
        if reason:
            raise QueryError(reason)

        if not self._started:
            await self.start(require_database=False)
        graph = self._graph
        if graph is None:
            graph = await self._map()

        started = time.perf_counter()
        try:
            outcome = await self._run_pipeline(query, graph)
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
        return self._respond(outcome, query, request_id, started)

    def _reject_query(self, question: str) -> str | None:
        if len(question) < MIN_QUERY_LENGTH:
            return "The question is too short."
        if len(question) > MAX_QUERY_LENGTH:
            return f"The question is too long (maximum {MAX_QUERY_LENGTH} characters)."
        problem = check_question(question)
        if problem:
            logger.warning("Risky question rejected: %s (%s)", question, problem)
            return "Invalid query: " + problem
        return None

    async def _run_pipeline(self, query: str, graph: SchemaGraph) -> Outcome:
        policy = self.plan_policy
        tables, retrieval_mode, retrieved = await link(query, graph, self._index, policy)
        logger.debug("link: mode=%s tables=%s", retrieval_mode.value, tables)

        plan = await self._plan_pipeline(query, graph, tables)
        statement = statement_for(plan, graph, self.settings.db_schema)
        logger.debug("guard: statement composed (%d bound parameter(s))", len(statement.params))

        rows = await self.runner.run(statement)

        return Outcome(
            plan=plan,
            statement=statement,
            rows=rows,
            retrieved_tables=retrieved,
            retrieval_mode=retrieval_mode,
        )

    async def _plan_pipeline(self, question: str, graph: SchemaGraph, tables: list[str]):
        policy = self.plan_policy
        plan_result = None
        error = None
        for attempt in range(policy.max_plan_attempts):
            plan_result, error = await plan_a_question(
                question, graph, tables, policy, self.planner, error, attempt
            )
            if plan_result is not None:
                break
            logger.debug(
                "plan: attempt %d of %d failed: %s",
                attempt + 1,
                policy.max_plan_attempts,
                error,
            )

        if plan_result is None:
            logger.error("No plan survived %d attempts: %s", policy.max_plan_attempts, error)
            raise PlanningError(redact(error or "No valid plan was produced."))
        return plan_result

    def _respond(
        self, outcome: Outcome, query: str, request_id: str | None, started: float
    ) -> SearchResponse:
        rows = outcome.rows

        if not rows:
            answer = NO_RECORDS_MESSAGE
            logger.debug(
                "[%s] no rows returned; using the configured no-records message",
                request_id,
            )
        else:
            answer = render_template(
                outcome.plan.template or "",
                rows,
                outcome.plan.intent,
                max_chars=ERROR_MESSAGE_MAX_CHARS,
            )
            if not answer:
                answer = NO_SUMMARY_MESSAGE
                logger.debug(
                    "[%s] the plan carried no usable response template; using the no-summary "
                    "message",
                    request_id,
                )

        elapsed = round((time.perf_counter() - started) * 1000, ELAPSED_PRECISION)
        logger.debug(
            "[%s] response ready: rows=%d answer_chars=%d expose_sql=%s",
            request_id,
            len(rows),
            len(answer),
            EXPOSE_SQL,
        )
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

        return SearchResponse(
            query=query,
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
        if refresh or self._graph is None:
            return await self._map()
        return self._graph

    async def health(self) -> HealthResponse:
        connected = await self._ping()
        healthy = self._started and connected
        provider = self.settings.embedding_provider
        return HealthResponse(
            status=STATUS_OK if healthy else STATUS_DEGRADED,
            app=APP_NAME,
            version=self.version,
            database=DATABASE_UP if connected else DATABASE_DOWN,
            tables=len(self._graph.names) if self._graph is not None else 0,
            model=self.settings.llm_model,
            embedding=EMBEDDING_OFF if provider == PROVIDER_NONE else self.settings.embedding_model,
        )

    async def _ping(self) -> bool:
        ttl = HEALTH_PING_CACHE_SECONDS
        now = time.monotonic()
        if self._last_ping is not None:
            cached_at, value = self._last_ping
            if now - cached_at < ttl:
                return value
        value = await self._catalog.ping()
        self._last_ping = (time.monotonic(), value)
        return value

    def create_app(self) -> "FastAPI":
        from core.server import create_app

        return create_app(self)
