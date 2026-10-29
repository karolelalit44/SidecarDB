import asyncio
import contextlib
import time
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any
from unicodedata import category

from core.answer import bind_template
from core.capabilities import Catalog, Embedder, Planner, Runner
from core.catalog import SchemaGraph
from core.config import Settings
from core.constants import (
    ALLOWED_CONTROL,
    APP_NAME,
    BLOCKED_CATEGORIES,
    DATABASE_DOWN,
    DATABASE_UP,
    DISALLOWED_SUBSTRINGS,
    ELAPSED_PRECISION,
    EMBEDDING_OFF,
    ERROR_MESSAGE_MAX_CHARS,
    EXPOSE_SQL,
    HEALTH_PING_CACHE_SECONDS,
    LOG_LLM_PAYLOADS,
    LOG_SQL,
    MAX_QUESTION_LENGTH,
    MIN_QUESTION_LENGTH,
    NO_ANSWER_MESSAGE,
    NO_RECORDS_MESSAGE,
    PLANNING_FAILED_MESSAGE,
    PROVIDER_API,
    PROVIDER_LOCAL,
    PROVIDER_NONE,
    RECOVERY_INTERVAL_SECONDS,
    STATUS_DEGRADED,
    STATUS_OK,
)
from core.exceptions import ConfigurationError, PlanningError, QuestionError
from core.guard import statement_for
from core.plan import Outcome, Plan, PlanPolicy
from core.planning import PlanRequest
from core.planning import make_plan as plan_a_question
from core.retrieval import SchemaIndex, link
from core.wire import (
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
            return
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

    async def _refresh_catalog(self) -> SchemaGraph:
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
        self._stop_event.set()
        if self._recovery_task is not None:
            self._recovery_task.cancel()
            self._recovery_task = None
        if not self._started:
            return
        self._started = False
        if self._owns_catalog:
            await self._catalog.close()
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
        question = (question or "").strip()
        logger.debug("[%s] search received %d chars", request_id, len(question))

        reason = self._reject_question(question)
        if reason:
            raise QuestionError(reason)

        if not self._started:
            await self.start(require_database=False)
        graph = self._graph
        if graph is None:
            graph = await self._refresh_catalog()

        started = time.perf_counter()
        try:
            outcome = await self._run_pipeline(question, graph)
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
        return self._respond(outcome, question, request_id, started)

    def _reject_question(self, question: str) -> str | None:
        if not question or len(question) < MIN_QUESTION_LENGTH:
            return "The question is too short."
        if len(question) > MAX_QUESTION_LENGTH:
            return f"The question is too long (maximum {MAX_QUESTION_LENGTH} characters)."
        for marker in DISALLOWED_SUBSTRINGS:
            if marker in question:
                logger.warning(
                    "Rejected a %d-character question: disallowed symbols", len(question)
                )
                return "The question contains disallowed symbols or comment characters."
        for ch in question:
            if ch in ALLOWED_CONTROL:
                continue
            if category(ch) in BLOCKED_CATEGORIES:
                logger.warning("Rejected question: unsupported control characters")
                return "The question contains unsupported control or formatting characters."
        if not any(ch.isalnum() for ch in question):
            logger.warning("Rejected question: no alphanumeric character")
            return "The question must contain at least one alphanumeric character."
        return None

    async def _run_pipeline(self, question: str, graph: SchemaGraph) -> Outcome:
        request = PlanRequest(question=question, graph=graph, policy=self.plan_policy)
        tables, retrieval_mode, retrieved = await link(request, self._index)
        logger.debug("link: mode=%s tables=%s", retrieval_mode.value, tables)

        request = replace(request, tables=tables)
        plan = await self._plan_pipeline(request)
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

    async def _plan_pipeline(self, request: PlanRequest) -> Plan:
        policy = request.policy
        plan_result = None
        error = None
        for attempt in range(policy.max_plan_attempts):
            plan_result, error = await plan_a_question(request, self.planner, error, attempt)
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
            raise PlanningError(PLANNING_FAILED_MESSAGE)
        return plan_result

    def _respond(
        self, outcome: Outcome, question: str, request_id: str | None, started: float
    ) -> SearchResponse:
        rows = outcome.rows

        if not rows:
            answer = NO_RECORDS_MESSAGE
            logger.debug(
                "[%s] no rows returned; using the configured no-records message",
                request_id,
            )
        else:
            answer = bind_template(
                outcome.plan.template or "",
                rows,
                outcome.plan.intent,
                max_chars=ERROR_MESSAGE_MAX_CHARS,
            )
            if not answer:
                answer = NO_ANSWER_MESSAGE
                logger.debug(
                    "[%s] the plan carried no usable template; using the no-answer message",
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
        if refresh or self._graph is None:
            return await self._refresh_catalog()
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
