from .config import settings
from .constants import SIMILARITY_ALIAS, VECTOR_CAST
from .embeddings import EmbeddingError, embedder
from .logging_config import get_logger
from .types import SchemaGraph

logger = get_logger("fallback")


def available(graph: SchemaGraph, tables) -> bool:
    if not settings.fallback_enabled or not embedder.enabled:
        return False
    return any(graph.embedding_column_for(name) for name in tables)


async def search(graph: SchemaGraph, tables, question: str) -> list[dict]:
    if not available(graph, tables):
        return []

    try:
        vector = (await embedder.aencode([question]))[0]
    except EmbeddingError as exc:
        logger.warning("Fallback embedding failed: %s", exc)
        return []

    for table in tables:
        column = graph.embedding_column_for(table)
        if not column:
            continue
        try:
            rows = await _query(table, column, vector)
        except Exception as exc:
            # The fallback is an optional tail to a query that already ran. If it
            # cannot run, the caller gets no rows rather than a failed request.
            logger.warning("Vector fallback query failed for %s: %s", table, exc)
            continue
        if rows:
            logger.info("Vector fallback matched %d rows in %s", len(rows), table)
            return [_strip(row, column) for row in rows]
    return []


async def _query(table: str, column: str, vector) -> list[dict]:
    from . import execute
    from .guard import vector_literal

    literal = vector_literal(vector)
    sql = (
        f'SELECT *, "{column}" <=> {literal}{VECTOR_CAST} AS {SIMILARITY_ALIAS} '
        f'FROM "{table}" ORDER BY "{column}" <=> {literal}{VECTOR_CAST} '
        f"LIMIT {settings.fallback_top_k}"
    )
    return await execute.run(execute.prepare(sql, None))


def _strip(record: dict, column: str) -> dict:
    return {k: v for k, v in record.items() if k not in (column, SIMILARITY_ALIAS)}
