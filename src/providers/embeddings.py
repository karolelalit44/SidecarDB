import asyncio
import math
import threading
import time
from typing import TYPE_CHECKING

from config.constants import EMBEDDING_BATCH_SIZE
from config.settings import Settings
from core.exceptions import EmbeddingError
from core.logging import get_logger
from providers.loader import get_embedding_model

if TYPE_CHECKING:
    pass

logger = get_logger("providers.embeddings")
_backend_lock = threading.Lock()


def _normalise(vector: list[float]) -> list[float]:
    magnitude = math.sqrt(sum(value * value for value in vector))
    if not magnitude:
        logger.debug("_normalise: zero magnitude vector, returning unchanged")
        return list(vector)
    return [value / magnitude for value in vector]


class _LocalBackend:
    def __init__(self, settings: Settings) -> None:
        self.model = get_embedding_model(
            model_name=settings.embedding_model,
            local_path=settings.embedding_local_path,
            device=settings.embedding_device,
        )

    def encode(self, texts: list[str]) -> list[list[float]]:
        logger.debug(
            "_LocalBackend.encode: encoding %d texts with batch_size=%d",
            len(texts),
            EMBEDDING_BATCH_SIZE,
        )
        return [
            list(map(float, vector))
            for vector in self.model.encode(
                texts,
                batch_size=EMBEDDING_BATCH_SIZE,
                normalize_embeddings=True,
            )
        ]

    async def aencode(self, texts: list[str]) -> list[list[float]]:
        return await asyncio.to_thread(self.encode, texts)


class Embedder:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._backend: _LocalBackend | None = None
        self._dimensions = settings.embedding_dimensions

    @property
    def enabled(self) -> bool:
        return True

    def _load(self) -> _LocalBackend:
        if self._backend is None:
            with _backend_lock:
                if self._backend is None:
                    started = time.perf_counter()
                    self._backend = _LocalBackend(self._settings)
                    logger.debug(
                        "embeddings: local backend ready in %d ms",
                        int((time.perf_counter() - started) * 1000),
                    )
        return self._backend

    async def aencode(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            logger.debug("embeddings: nothing to encode")
            return []
        started = time.perf_counter()
        try:
            vectors = await self._load().aencode(texts)
            if vectors:
                self._dimensions = len(vectors[0])
            logger.debug(
                "embeddings: encoded %d texts to width %d in %d ms",
                len(texts),
                self._dimensions,
                int((time.perf_counter() - started) * 1000),
            )
            return vectors
        except EmbeddingError:
            raise
        except Exception as exc:
            logger.debug("embeddings: encoding failed: %s: %s", type(exc).__name__, exc)
            raise EmbeddingError(f"Embedding failed: {exc}") from exc

    async def aclose(self) -> None:
        pass


__all__ = ["Embedder", "_normalise", "get_embedding_model"]
