import time
import unicodedata
from typing import TYPE_CHECKING

from core.answer import render_template
from core.config import Settings
from core.exceptions import QueryError
from core.logging_config import get_logger
from core.retrieval import SchemaIndex
from core.schema import SearchResponse
from core.types import SchemaGraph

from . import graph as graph_module

if TYPE_CHECKING:
    from infra.db import Database
    from infra.embeddings import Embedder
    from infra.execute import Executor
    from infra.llm import LLMClient

logger = get_logger("service")

_DISALLOWED_SUBSTRINGS = ("--", "/*", "*/", ";", "\0")
_ALLOWED_CONTROL = ("\t", "\n", "\r")
_BLOCKED_CATEGORIES = ("Cc", "Cf", "Cs")


def validate_query(text: str) -> tuple[bool, str | None]:
    """Rejects control characters and the punctuation shapes of injected SQL."""
    if not text:
        return False, "The query is empty."

    for marker in _DISALLOWED_SUBSTRINGS:
        if marker in text:
            return False, "Query contains disallowed symbols or comment characters."

    for ch in text:
        if ch in _ALLOWED_CONTROL:
            continue
        if unicodedata.category(ch) in _BLOCKED_CATEGORIES:
            return False, "Query contains unsupported control or formatting characters."

    if not any(ch.isalnum() for ch in text):
        return False, "Query must contain at least one alphanumeric character."

    return True, None


class SearchService:
    """Owns the catalog, the schema index and one full search pipeline."""

    def __init__(
        self,
        settings: Settings,
        database: "Database",
        embedder: "Embedder",
        *,
        llm: "LLMClient | None" = None,
        executor: "Executor | None" = None,
    ) -> None:
        from infra.execute import Executor as _Executor
        from infra.llm import LLMClient as _LLMClient

        self.settings = settings
        self.database = database
        self.embedder = embedder
        self._llm = llm if llm is not None else _LLMClient(settings)
        self._executor = executor if executor is not None else _Executor(database, settings)
        self._graph: SchemaGraph | None = None
        self._index: SchemaIndex | None = None

    async def startup(self) -> None:
        await self.database.connect()
        await self.refresh()

    async def shutdown(self) -> None:
        await self.database.close()
        await self.embedder.aclose()
        self._graph = None
        self._index = None
        logger.debug("shutdown: catalog and schema index released")

    async def refresh(self) -> SchemaGraph:
        started = time.perf_counter()
        logger.debug("refresh: introspecting the catalog of schema %s", self.settings.db_schema)
        self._graph = await self.database.introspect()
        logger.debug(
            "refresh: %d tables introspected in %d ms; rebuilding the schema index",
            len(self._graph.tables),
            round((time.perf_counter() - started) * 1000, self.settings.elapsed_precision),
        )
        self._index = await SchemaIndex(self.embedder, self.settings).build(self._graph)
        logger.debug("refresh: schema index %s", "ready" if self._index.ready else "unavailable")
        logger.info(
            "Catalog mapped: %d tables",
            len(self._graph.tables),
        )
        return self._graph

    async def schema(self) -> SchemaGraph:
        if self._graph is None:
            logger.debug("schema: no catalog held yet; introspecting on demand")
            return await self.refresh()
        return self._graph

    def catalog_names(self) -> list[str]:
        """Names currently mapped, without touching the database."""
        return self._graph.names if self._graph is not None else []

    async def search(self, question: str, request_id: str | None = None) -> SearchResponse:
        settings = self.settings
        query = (question or "").strip()
        logger.debug("[%s] search received %d chars", request_id, len(query))
        if len(query) < settings.min_query_length:
            raise QueryError("The question is too short.")
        if len(query) > settings.max_query_length:
            raise QueryError(
                f"The question is too long (maximum {settings.max_query_length} characters)."
            )

        is_safe, reason = validate_query(query)
        if not is_safe:
            logger.warning("[%s] Risky query rejected: %s (%s)", request_id, query, reason)
            raise QueryError(f"Invalid query: {reason}")

        started = time.perf_counter()
        graph = await self.schema()
        logger.debug(
            "[%s] catalog ready: %d tables, index=%s",
            request_id,
            len(graph.tables),
            "built" if self._index is not None else "absent",
        )

        try:
            outcome = await graph_module.run(
                query, graph, self._index, settings, self._llm, self._executor
            )
        except Exception as exc:
            logger.error(
                "[%s] search failed after %d ms: %s: %s",
                request_id,
                round((time.perf_counter() - started) * 1000, settings.elapsed_precision),
                type(exc).__name__,
                exc,
                exc_info=settings.log_llm_payloads,
            )
            raise

        rows = outcome.rows

        if not rows:
            answer = settings.no_records_message
            logger.debug(
                "[%s] no rows returned; using the configured no-records message",
                request_id,
            )
        else:
            answer = render_template(
                outcome.plan.template or "",
                rows,
                outcome.plan.intent,
                max_chars=settings.error_message_max_chars,
            )
            if not answer:
                answer = settings.no_summary_message
                logger.debug(
                    "[%s] the plan carried no usable response template; using the no-summary "
                    "message",
                    request_id,
                )

        elapsed = round((time.perf_counter() - started) * 1000, settings.elapsed_precision)
        logger.debug(
            "[%s] response ready: rows=%d answer_chars=%d expose_sql=%s",
            request_id,
            len(rows),
            len(answer),
            settings.expose_sql,
        )
        logger.info(
            "[%s] intent=%s mode=%s rows=%d tables=%s sql=%s bound_params=%d",
            request_id,
            outcome.plan.intent.value,
            outcome.retrieval_mode.value,
            len(rows),
            outcome.retrieved_tables,
            outcome.sql,
            len(outcome.params),
        )

        return SearchResponse(
            query=query,
            answer=answer,
            sql=outcome.sql if settings.expose_sql else None,
            row_count=len(rows),
            elapsed_ms=elapsed,
            records=rows,
            retrieved_tables=outcome.retrieved_tables,
            retrieval_mode=outcome.retrieval_mode,
            request_id=request_id,
        )
