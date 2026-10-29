import time
from typing import TYPE_CHECKING

from core.exceptions import EmbeddingError
from core.logging_config import get_logger
from core.types import Hit, SchemaGraph, TableInfo

if TYPE_CHECKING:
    from core.config import Settings
    from infra.embeddings import Embedder

logger = get_logger("retrieval")


class SchemaIndex:
    """Ranks catalog tables and columns against an incoming question."""

    def __init__(self, embedder: "Embedder", settings: "Settings") -> None:
        self._embedder = embedder
        self._settings = settings
        self._keys: list[str] = []
        self._texts: list[str] = []
        self._vectors: list[list[float]] | None = None
        self._table_of: dict[str, str] = {}

    @property
    def ready(self) -> bool:
        return self._vectors is not None

    async def build(self, graph: SchemaGraph) -> "SchemaIndex":
        keys: list[str] = []
        texts: list[str] = []
        table_of: dict[str, str] = {}

        for table in graph.tables.values():
            keys.append(table.name)
            texts.append(_describe_table(table))
            table_of[table.name] = table.name

            for column in table.columns:
                key = f"{table.name}.{column.name}"
                keys.append(key)
                texts.append(_describe_column(table, column))
                table_of[key] = table.name

        self._keys, self._texts, self._table_of = keys, texts, table_of
        self._vectors = None

        if not self._embedder.enabled:
            logger.info("Embeddings disabled; schema index will report unavailable")
            return self

        logger.debug(
            "retrieval: embedding %d schema elements for %d tables",
            len(texts),
            len(graph.tables),
        )
        try:
            self._vectors = await self._embedder.aencode(texts)
            logger.info("Schema index built over %d elements", len(texts))
        except EmbeddingError as exc:
            logger.warning("Schema index unavailable: %s", exc)
            self._vectors = None
        return self

    async def search(self, question: str, top_k: int) -> list[Hit]:
        if self._vectors is None:
            logger.debug("retrieval: the index holds no vectors; search is unavailable")
            return []

        started = time.perf_counter()
        try:
            query_vector = (await self._embedder.aencode([question]))[0]
        except EmbeddingError as exc:
            logger.warning("Query embedding failed: %s", exc)
            return []

        precision = self._settings.retrieval_score_precision
        hits = [
            Hit(
                table=self._table_of[key],
                subject=key,
                score=round(dot(query_vector, vector), precision),
            )
            for key, vector in zip(self._keys, self._vectors, strict=True)
        ]
        hits.sort(key=lambda hit: hit.score, reverse=True)
        top = hits[:top_k]
        logger.debug(
            "retrieval: scored %d elements in %d ms; kept the top %d at %s",
            len(hits),
            int((time.perf_counter() - started) * 1000),
            len(top),
            ", ".join(f"{hit.subject}={hit.score}" for hit in top),
        )
        return top

    async def tables_for(self, question: str, limit: int, min_score: float) -> list[str]:
        best: dict[str, float] = {}
        for hit in await self.search(question, self._settings.retrieval_top_k):
            if hit.score < min_score:
                continue
            if hit.table not in best or hit.score > best[hit.table]:
                best[hit.table] = hit.score

        ordered = sorted(best.items(), key=lambda pair: pair[1], reverse=True)
        chosen = ordered[:limit]
        logger.debug(
            "retrieval: %d table(s) at or above %s, limited to %d: %s",
            len(ordered),
            min_score,
            limit,
            ", ".join(f"{name}={score}" for name, score in chosen),
        )
        return [name for name, _score in chosen]


def dot(a: list[float], b: list[float]) -> float:
    """Both operands are unit vectors, so cosine is the dot product."""
    return sum(x * y for x, y in zip(a, b, strict=True))


def _words(name: str) -> str:
    return name.replace("_", " ")


def _describe_table(table: TableInfo) -> str:
    parts = [_words(table.name)]
    if table.description:
        parts.append(table.description)
    parts.extend(_words(c.name) for c in table.columns)
    return " | ".join(parts)


def _describe_column(table: TableInfo, column) -> str:
    parts = [_words(column.name)]
    if column.description:
        parts.append(column.description)
    parts.append(column.data_type)
    parts.append(f"column of {_words(table.name)}")
    if table.description:
        parts.append(table.description)
    return " | ".join(parts)
