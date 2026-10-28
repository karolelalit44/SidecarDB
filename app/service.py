import time
import unicodedata

from . import graph as graph_module
from .answer import narrate, render_template
from .config import settings
from .exceptions import QueryError
from .logging_config import get_logger
from .retrieval import SchemaIndex
from .schema import SearchResponse
from .types import SchemaGraph

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
    def __init__(self):
        self._graph: SchemaGraph | None = None
        self._index: SchemaIndex | None = None
        self._samples: dict[str, dict[str, list[str]]] = {}

    async def startup(self) -> None:
        from .db import database

        await database.connect()
        await self.refresh()

    async def shutdown(self) -> None:
        from .db import database

        await database.close()
        self._graph = None
        self._index = None
        self._samples = {}

    async def refresh(self) -> SchemaGraph:
        from .db import database

        self._graph = await database.introspect()
        self._index = await SchemaIndex().build(self._graph)
        self._samples = {}
        logger.info(
            "Catalog mapped: %d tables, %d extension-backed columns",
            len(self._graph.tables),
            len(self._graph.indexed_columns),
        )
        return self._graph

    async def schema(self) -> SchemaGraph:
        if self._graph is None:
            await self.refresh()
        return self._graph

    async def search(self, question: str, request_id: str | None = None) -> SearchResponse:
        query = (question or "").strip()
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
            outcome = await graph_module.run(query, graph, self._index, self._samples)
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
        elif outcome.plan.template:
            answer = render_template(outcome.plan.template, rows, outcome.plan.intent)
            if answer:
                logger.debug("[%s] answered from the plan template", request_id)
            else:
                answer = await narrate(query, rows)
                logger.debug(
                    "[%s] template was unusable; answered by narration over %d rows",
                    request_id,
                    len(rows),
                )
        else:
            answer = await narrate(query, rows)
            logger.debug("[%s] answered by narration over %d rows", request_id, len(rows))

        elapsed = round((time.perf_counter() - started) * 1000, settings.elapsed_precision)
        logger.info(
            "[%s] intent=%s mode=%s rows=%d tables=%s sql=%s",
            request_id,
            outcome.plan.intent.value,
            outcome.retrieval_mode.value,
            len(rows),
            outcome.retrieved_tables,
            outcome.sql,
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


service = SearchService()
