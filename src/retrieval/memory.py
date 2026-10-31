import json
import re
import time
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from pydantic import BaseModel

from catalog.models import SchemaGraph, TableInfo
from core.exceptions import EmbeddingError
from core.logging import get_logger

if TYPE_CHECKING:
    from planner.models import PlanPolicy

logger = get_logger("retrieval")


class RetrievalMode(StrEnum):
    embedding = "embedding"


class Hit(BaseModel):
    table: str
    subject: str
    score: float


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(f"{path.suffix}.tmp")
    tmp_path.write_text(text, encoding="utf-8")
    tmp_path.replace(path)


def _atomic_write_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(f"{path.suffix}.tmp")
    with open(tmp_path, "wb") as f:
        np.save(f, array)
    tmp_path.replace(path)


class SchemaMemory:
    """In-memory active schema representation and semantic vector index."""

    def __init__(self, embedder: Any, policy: "PlanPolicy") -> None:
        self._embedder = embedder
        self._policy = policy
        self._keys: list[str] = []
        self._texts: list[str] = []
        self._tokens_of: list[set[str]] = []
        self._matrix: np.ndarray | None = None

    @property
    def ready(self) -> bool:
        return self._matrix is not None

    @property
    def _vectors(self) -> np.ndarray | None:
        return self._matrix

    async def build(
        self, graph: SchemaGraph, storage_dir: str | Path | None = None
    ) -> "SchemaMemory":
        keys: list[str] = []
        texts: list[str] = []
        tokens_of: list[set[str]] = []

        for table in graph.tables.values():
            keys.append(table.name)
            texts.append(_describe_table(table))
            t_tokens = _extract_tokens(table.name)
            for col in table.columns:
                t_tokens.update(_extract_tokens(col.name))
            if table.description:
                t_tokens.update(_extract_tokens(table.description))
            tokens_of.append(t_tokens)

        self._keys = keys
        self._texts = texts
        self._tokens_of = tokens_of
        self._matrix = None

        logger.debug(
            "retrieval: collected %d tables for memory indexing",
            len(keys),
        )

        dir_path = Path(storage_dir) if storage_dir else None
        meta_file = dir_path / "schema_meta.json" if dir_path else None
        vec_file = dir_path / "schema_vectors.npy" if dir_path else None
        schema_file = dir_path / "schema.json" if dir_path else None

        if meta_file and vec_file and meta_file.is_file() and vec_file.is_file():
            try:
                meta = json.loads(meta_file.read_text(encoding="utf-8"))
                if meta.get("signature") == graph.signature and meta.get("keys") == keys:
                    loaded = np.load(vec_file, mmap_mode="r")
                    if loaded.ndim == 2 and loaded.shape[0] == len(texts):
                        self._matrix = loaded
                        logger.info(
                            "Schema memory loaded from persistent storage for %d tables",
                            len(texts),
                        )
                        return self
            except Exception as exc:
                logger.debug("retrieval: failed to load persisted memory (%s); recomputing", exc)

        logger.debug(
            "retrieval: embedding %d tables",
            len(texts),
        )
        try:
            raw_vectors = await self._embedder.aencode(texts)
            if raw_vectors:
                self._matrix = np.array(raw_vectors, dtype=np.float32)
                dim = self._matrix.shape[1] if self._matrix.ndim > 1 else 0

                if dir_path and schema_file and meta_file and vec_file:
                    try:
                        _atomic_write_text(schema_file, graph.model_dump_json(indent=2))
                        _atomic_write_text(
                            meta_file,
                            json.dumps(
                                {"signature": graph.signature, "keys": keys},
                                ensure_ascii=False,
                            ),
                        )
                        _atomic_write_npy(vec_file, self._matrix)
                        logger.debug(
                            "retrieval: saved schema and vectors to %s",
                            dir_path,
                        )
                    except Exception as exc:
                        logger.warning(
                            "retrieval: could not persist schema memory to %s: %s",
                            dir_path,
                            exc,
                        )
                logger.info(
                    "Schema memory built: %d tables indexed (%d dimensions)",
                    len(texts),
                    dim,
                )
            else:
                self._matrix = np.empty((0, 0), dtype=np.float32)
                logger.info("Schema memory built: 0 tables indexed (empty schema)")
        except EmbeddingError:
            raise
        except Exception as exc:
            logger.error("retrieval: embedding %d tables failed: %s", len(texts), exc)
            raise EmbeddingError(f"Embedding failed: {exc}") from exc

        return self

    @classmethod
    def load_persisted(
        cls, storage_dir: str | Path, embedder: Any, policy: "PlanPolicy"
    ) -> tuple[SchemaGraph, "SchemaMemory"] | None:
        dir_path = Path(storage_dir)
        schema_file = dir_path / "schema.json"
        meta_file = dir_path / "schema_meta.json"
        vec_file = dir_path / "schema_vectors.npy"

        if not (schema_file.is_file() and meta_file.is_file() and vec_file.is_file()):
            return None
        try:
            graph = SchemaGraph.model_validate_json(schema_file.read_text(encoding="utf-8"))
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
            keys = meta.get("keys", [])
            matrix = np.load(vec_file)
            if matrix.shape[0] != len(keys):
                return None

            memory = cls(embedder, policy)
            memory._keys = keys
            memory._texts = [_describe_table(graph.tables[k]) for k in keys if k in graph.tables]
            tokens_of = []
            for k in keys:
                table = graph.tables[k]
                t_tokens = _extract_tokens(table.name)
                for col in table.columns:
                    t_tokens.update(_extract_tokens(col.name))
                if table.description:
                    t_tokens.update(_extract_tokens(table.description))
                tokens_of.append(t_tokens)
            memory._tokens_of = tokens_of
            memory._matrix = matrix
            logger.info("Restored schema memory for %d tables from %s", len(keys), dir_path)
            return graph, memory
        except Exception as exc:
            logger.warning("retrieval: could not restore schema from %s: %s", dir_path, exc)
            return None

    async def search(self, question: str, top_k: int) -> list[Hit]:
        if self._matrix is None or len(self._matrix) == 0:
            logger.debug("retrieval: memory holds no vectors; search is unavailable")
            return []

        logger.debug("retrieval: question embedding query=%r (top_k=%d)", question, top_k)
        started = time.perf_counter()
        try:
            raw_query = (await self._embedder.aencode([question]))[0]
            query_vector = np.array(raw_query, dtype=np.float32)
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
                table=self._keys[idx],
                subject=self._keys[idx],
                score=float(scores[idx]),
            )
            for idx in top_indices
        ]

        logger.debug(
            "retrieval: scored %d tables in %d ms; top: %s",
            len(hits),
            int((time.perf_counter() - started) * 1000),
            ", ".join(f"{hit.table}={hit.score}" for hit in hits[:k]),
        )
        return hits

    async def tables_for(self, question: str, limit: int, min_score: float) -> list[str]:
        candidate_k = min(len(self._keys), max(self._policy.retrieval_top_k, limit * 2))
        best: dict[str, float] = {}
        for hit in await self.search(question, candidate_k):
            if hit.score < min_score:
                continue
            if hit.table not in best or hit.score > best[hit.table]:
                best[hit.table] = hit.score

        ordered = sorted(best.items(), key=lambda pair: pair[1], reverse=True)
        chosen = ordered[:limit]
        return [name for name, _score in chosen]


SchemaIndex = SchemaMemory


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
    return " | ".join(parts)


__all__ = ["Hit", "RetrievalMode", "SchemaIndex", "SchemaMemory", "_describe_table"]
