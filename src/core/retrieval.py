import re
import time
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from pydantic import BaseModel

from core.catalog import ColumnInfo, SchemaGraph, TableInfo
from core.exceptions import EmbeddingError
from utils.logging_config import get_logger

if TYPE_CHECKING:
    from core.capabilities import Embedder
    from core.plan import PlanPolicy

logger = get_logger("retrieval")


class RetrievalMode(StrEnum):
    embedding = "embedding"
    catalog = "catalog"


class Hit(BaseModel):
    table: str
    subject: str
    score: float


class SchemaIndex:
    def __init__(self, embedder: "Embedder", policy: "PlanPolicy") -> None:
        self._embedder = embedder
        self._policy = policy
        self._keys: list[str] = []
        self._texts: list[str] = []
        self._tokens_of: list[set[str]] = []
        self._matrix: np.ndarray | None = None
        self._table_of: dict[str, str] = {}

    @property
    def ready(self) -> bool:
        return self._matrix is not None

    @property
    def _vectors(self) -> np.ndarray | None:
        return self._matrix

    async def build(self, graph: SchemaGraph, cache_dir: str | Path | None = None) -> "SchemaIndex":
        keys: list[str] = []
        texts: list[str] = []
        tokens_of: list[set[str]] = []
        table_of: dict[str, str] = {}

        for table in graph.tables.values():
            keys.append(table.name)
            texts.append(_describe_table(table))
            table_of[table.name] = table.name
            tokens_of.append(_extract_tokens(table.name))

            for column in table.columns:
                key = f"{table.name}.{column.name}"
                keys.append(key)
                texts.append(_describe_column(table, column))
                table_of[key] = table.name
                tokens_of.append(_extract_tokens(column.name))

        self._keys, self._texts, self._table_of, self._tokens_of = keys, texts, table_of, tokens_of
        self._matrix = None

        logger.debug(
            "retrieval: collected %d schema elements across %d tables for indexing",
            len(keys),
            len(graph.tables),
        )

        if not self._embedder.enabled:
            logger.info("Embeddings disabled; schema index will report unavailable")
            return self

        cache_path: Path | None = None
        if cache_dir:
            cache_path = Path(cache_dir) / f"vectors_{graph.signature}.npy"
            if cache_path.is_file():
                try:
                    loaded = np.load(cache_path, mmap_mode="r")
                    if loaded.ndim == 2 and loaded.shape[0] == len(texts):
                        self._matrix = loaded
                        dim = self._matrix.shape[1]
                        logger.debug(
                            "retrieval: loaded schema vectors from cache (%s), dimensions=%d",
                            cache_path.name,
                            dim,
                        )
                        logger.info("Schema index loaded from cache over %d elements", len(texts))
                        return self
                except Exception as exc:
                    logger.debug("retrieval: cache read failed (%s); regenerating embeddings", exc)

        logger.debug(
            "retrieval: embedding %d schema elements for %d tables",
            len(texts),
            len(graph.tables),
        )
        try:
            raw_vectors = await self._embedder.aencode(texts)
            if raw_vectors:
                self._matrix = np.array(raw_vectors, dtype=np.float32)
                dim = self._matrix.shape[1] if self._matrix.ndim > 1 else 0
                if cache_path:
                    try:
                        cache_path.parent.mkdir(parents=True, exist_ok=True)
                        np.save(cache_path, self._matrix)
                        logger.debug("retrieval: saved schema vectors to cache %s", cache_path)
                    except Exception as exc:
                        logger.debug("retrieval: failed to write cache (%s)", exc)
            else:
                self._matrix = None
                dim = 0
            logger.debug("retrieval: schema index vector encoding complete, dimensions=%d", dim)
            logger.info("Schema index built over %d elements", len(texts))
        except EmbeddingError as exc:
            logger.warning("Schema index unavailable: %s", exc)
            self._matrix = None
        return self

    async def search(self, question: str, top_k: int) -> list[Hit]:
        if self._matrix is None or len(self._matrix) == 0:
            logger.debug("retrieval: the index holds no vectors; search is unavailable")
            return []

        logger.debug("retrieval: question embedding query=%r (top_k=%d)", question, top_k)
        started = time.perf_counter()
        try:
            raw_query = (await self._embedder.aencode([question]))[0]
            query_vector = np.array(raw_query, dtype=np.float32)
            logger.debug(
                "retrieval: question vector computed, scoring against %d index vectors",
                len(self._matrix),
            )
        except EmbeddingError as exc:
            logger.warning("Question embedding failed: %s", exc)
            return []

        raw_scores = self._matrix @ query_vector

        query_tokens = _extract_tokens(question)
        if query_tokens:
            boosts = np.array(
                [
                    0.20
                    if (self._tokens_of[i] and self._tokens_of[i].intersection(query_tokens))
                    else 0.0
                    for i in range(len(self._keys))
                ],
                dtype=np.float32,
            )
            raw_scores = np.minimum(1.0, raw_scores + boosts)

        precision = self._policy.retrieval_score_precision
        scores = np.round(raw_scores, precision)

        n = len(self._keys)
        k = min(top_k, n)
        if n <= k:
            top_indices = np.argsort(-scores)
        else:
            partition = np.argpartition(-scores, k)[:k]
            top_indices = partition[np.argsort(-scores[partition])]

        hits = [
            Hit(
                table=self._table_of[self._keys[idx]],
                subject=self._keys[idx],
                score=float(scores[idx]),
            )
            for idx in top_indices
        ]

        logger.debug(
            "retrieval: scored %d elements in %d ms; kept the top %d at %s",
            len(hits),
            int((time.perf_counter() - started) * 1000),
            len(hits),
            ", ".join(f"{hit.subject}={hit.score}" for hit in hits),
        )
        return hits

    async def tables_for(self, question: str, limit: int, min_score: float) -> list[str]:
        logger.debug(
            "retrieval tables_for: evaluating candidates (limit=%d, min_score=%s)",
            limit,
            min_score,
        )
        candidate_k = min(len(self._keys), max(self._policy.retrieval_top_k, limit * 5))
        best: dict[str, float] = {}
        for hit in await self.search(question, candidate_k):
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


async def link(
    question: str,
    graph: SchemaGraph,
    index: "SchemaIndex | None",
    policy: "PlanPolicy | None" = None,
) -> tuple[list[str], RetrievalMode, list[str]]:
    from core.plan import PlanPolicy as _PlanPolicy

    active_policy = policy if policy is not None else _PlanPolicy()
    logger.debug(
        "link: schema linking initiated for question: %r (max_schema_tables=%d)",
        question,
        active_policy.max_schema_tables,
    )
    if index is not None:
        tables = await index.tables_for(
            question, active_policy.max_schema_tables, active_policy.retrieval_min_score
        )
        if tables:
            logger.debug("link: resolved %d table(s) via embedding: %s", len(tables), tables)
            expanded = list(tables)
            for neighbor in graph.neighbors_of(tables):
                if neighbor not in expanded and len(expanded) < active_policy.max_schema_tables:
                    expanded.append(neighbor)
            if len(expanded) > len(tables):
                logger.debug(
                    "link: expanded %d table(s) to %d with 1-hop relational neighbors: %s",
                    len(tables),
                    len(expanded),
                    expanded,
                )
            return expanded, RetrievalMode.embedding, tables
    else:
        logger.debug("link: schema index is None; skipping embedding retrieval")

    logger.debug(
        "link: no embedding match at or above %s; falling back to catalog order",
        active_policy.retrieval_min_score,
    )
    names = rank_fallback_tables(graph, active_policy.max_schema_tables)
    logger.debug("link: fallback catalog tables: %s", names)
    return names, RetrievalMode.catalog, []


def rank_fallback_tables(graph: SchemaGraph, max_tables: int) -> list[str]:
    if not graph.tables:
        return []

    degree: dict[str, int] = {}
    for fk in graph.foreign_keys:
        degree[fk.source_table] = degree.get(fk.source_table, 0) + 1
        degree[fk.target_table] = degree.get(fk.target_table, 0) + 1

    utility_prefixes = ("audit_", "log_", "flyway_", "alembic_", "pg_", "sql_")

    def score_table(table: TableInfo) -> tuple[int, int, int, int, int]:
        is_utility = 1 if any(table.name.startswith(p) for p in utility_prefixes) else 0
        fk_degree = degree.get(table.name, 0)
        has_desc = 1 if table.description else 0
        row_score = min(table.row_estimate, 1_000_000)
        col_count = len(table.columns)
        return (-is_utility, fk_degree, has_desc, row_score, col_count)

    ranked = sorted(graph.tables.values(), key=score_table, reverse=True)
    return [t.name for t in ranked[:max_tables]]


def dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


def _words(name: str) -> str:
    return name.replace("_", " ")


def _extract_tokens(text: str) -> set[str]:
    return {w.lower() for w in re.findall(r"[A-Za-z0-9_]+", text) if len(w) > 1}


def _describe_table(table: TableInfo) -> str:
    parts = [_words(table.name)]
    if table.description:
        parts.append(table.description)
    pk_set = set(table.primary_keys)
    ordered_cols = [c for c in table.columns if c.name in pk_set] + [
        c for c in table.columns if c.name not in pk_set
    ]
    parts.extend(_words(c.name) for c in ordered_cols[:30])
    desc = " | ".join(parts)
    logger.debug("_describe_table: %s -> %r", table.name, desc)
    return desc


def _describe_column(table: TableInfo, column: ColumnInfo) -> str:
    parts = [_words(column.name)]
    if column.description:
        parts.append(column.description)
    if column.enum_values:
        parts.append("values: " + ", ".join(column.enum_values[:5]))
    parts.append(column.data_type)
    parts.append(f"in {_words(table.name)}")
    desc = " | ".join(parts)
    logger.debug("_describe_column: %s.%s -> %r", table.name, column.name, desc)
    return desc


__all__ = ["Hit", "RetrievalMode", "SchemaIndex", "link", "rank_fallback_tables"]
