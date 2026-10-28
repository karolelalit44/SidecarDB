import asyncio
import threading

from .config import settings
from .exceptions import EmbeddingError
from .logging_config import get_logger

logger = get_logger("embeddings")
_backend_lock = threading.Lock()


class _LocalBackend:
    def __init__(self):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(settings.embedding_model, device=settings.embedding_device)

    def encode(self, texts):
        return [
            list(map(float, vector))
            for vector in self.model.encode(
                texts,
                batch_size=settings.embedding_batch_size,
                normalize_embeddings=True,
            )
        ]

    async def aencode(self, texts):
        return await asyncio.to_thread(self.encode, texts)


class Embedder:
    def __init__(self):
        self._backend = None
        self._dimensions = settings.embedding_dimensions

    @property
    def enabled(self) -> bool:
        return settings.embedding_enabled

    def _load(self):
        if self._backend is None:
            with _backend_lock:
                if self._backend is None:
                    try:
                        self._backend = _LocalBackend()
                    except Exception as exc:
                        raise EmbeddingError(
                            f"Could not initialise the embedding backend: {exc}"
                        ) from exc
        return self._backend

    def _require_enabled(self):
        if not self.enabled:
            raise EmbeddingError("Embeddings are disabled.")

    def encode(self, texts):
        if not texts:
            return []
        self._require_enabled()
        try:
            vectors = self._load().encode(texts)
            if vectors:
                self._dimensions = len(vectors[0])
            return vectors
        except EmbeddingError:
            raise
        except Exception as exc:
            raise EmbeddingError(f"Embedding failed: {exc}") from exc

    async def aencode(self, texts):
        if not texts:
            return []
        self._require_enabled()
        try:
            vectors = await self._load().aencode(texts)
            if vectors:
                self._dimensions = len(vectors[0])
            return vectors
        except EmbeddingError:
            raise
        except Exception as exc:
            raise EmbeddingError(f"Embedding failed: {exc}") from exc


def dot(a, b) -> float:
    """Both operands are unit vectors, so cosine is the dot product."""
    return sum(x * y for x, y in zip(a, b, strict=True))


embedder = Embedder()
