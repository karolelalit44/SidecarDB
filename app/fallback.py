from .config import settings
from .constants import SIMILARITY_ALIAS, VECTOR_CAST
from .embeddings import EmbeddingError, embedder
from .guard import quote, vector_literal
from .logging_config import get_logger
from .types import SchemaGraph

logger = get_logger("fallback")

COSINE_DISTANCE = "<=>"
_reported_mismatch: set[str] = set()


def _compatible(graph: SchemaGraph, table: str, width: int) -> bool:
    """Check if table vector dimensions match query vector width."""
    declared = graph.embedding_dimensions_for(table)
    if declared is None or declared == width:
        return True
    if table not in _reported_mismatch:
        _reported_mismatch.add(table)
        logger.warning(
            "Vector fallback skipped for %s: the column is declared vector(%d) but the "
            "configured model produces %d dimensions. Re-embed the column to enable it.",
            table,
            declared,
            width,
        )
    return False


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
        if not _compatible(graph, table, len(vector)):
            continue
        try:
            rows = await _query(graph, table, column, vector)
        except Exception as exc:
            logger.warning("Vector fallback query failed for %s: %s", table, exc)
            continue
        if rows:
            logger.info("Vector fallback matched %d rows in %s", len(rows), table)
            return rows
    return []


async def _query(graph: SchemaGraph, table: str, column: str, vector) -> list[dict]:
    from . import execute

    payload = [quote(name) for name in graph.tables[table].column_names if name != column]

    sql = (
        f"SELECT {', '.join(payload) if payload else '*'}, "
        f'"{column}" {COSINE_DISTANCE} $1{VECTOR_CAST} AS {SIMILARITY_ALIAS} '
        f'FROM "{table}" '
        f'ORDER BY "{column}" {COSINE_DISTANCE} $1{VECTOR_CAST} '
        f"LIMIT {settings.fallback_top_k}"
    )
    return await execute.run(execute.prepare(sql, graph), vector_literal(vector))
