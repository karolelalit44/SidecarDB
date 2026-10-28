from .config import settings
from .embeddings import EmbeddingError, dot, embedder
from .logging_config import get_logger
from .types import Hit, SchemaGraph, TableInfo

logger = get_logger("retrieval")


class SchemaIndex:
    def __init__(self) -> None:
        self._graph: SchemaGraph | None = None
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

        self._graph = graph
        self._keys, self._texts, self._table_of = keys, texts, table_of
        self._vectors = None

        if not embedder.enabled:
            logger.info("Embeddings disabled; schema index will report unavailable")
            return self

        try:
            self._vectors = await embedder.aencode(texts)
            logger.info("Schema index built over %d elements", len(texts))
        except EmbeddingError as exc:
            logger.warning("Schema index unavailable: %s", exc)
            self._vectors = None
        return self

    async def search(self, question: str, top_k: int) -> list[Hit]:
        if self._vectors is None:
            return []

        try:
            query_vector = (await embedder.aencode([question]))[0]
        except EmbeddingError as exc:
            logger.warning("Query embedding failed: %s", exc)
            return []

        precision = settings.retrieval_score_precision
        hits = [
            Hit(
                table=self._table_of[key],
                subject=key,
                score=round(dot(query_vector, vector), precision),
            )
            for key, vector in zip(self._keys, self._vectors, strict=True)
        ]
        hits.sort(key=lambda hit: hit.score, reverse=True)
        return hits[:top_k]

    async def tables_for(self, question: str, limit: int, min_score: float) -> list[str]:
        best: dict[str, float] = {}
        for hit in await self.search(question, settings.retrieval_top_k):
            if hit.score < min_score:
                continue
            if hit.table not in best or hit.score > best[hit.table]:
                best[hit.table] = hit.score

        ordered = sorted(best.items(), key=lambda pair: pair[1], reverse=True)
        return [name for name, _score in ordered[:limit]]

    async def columns_for(self, question: str, min_score: float) -> dict[str, float]:
        """Return similarity scores for columns above the threshold."""
        scored: dict[str, float] = {}
        for hit in await self.search(question, settings.retrieval_top_k):
            if hit.score < min_score or "." not in hit.subject:
                continue
            table, column = hit.subject.rsplit(".", 1)
            scored[f"{table}.{column}"] = hit.score
        return scored


def _words(name: str) -> str:
    return name.replace("_", " ")


def _describe_table(table: TableInfo) -> str:
    parts = [_words(table.name)]
    if table.description:
        parts.append(table.description)
    parts.extend(_words(c.name) for c in table.columns if not c.primary_key)
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
