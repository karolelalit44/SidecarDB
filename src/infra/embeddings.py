import asyncio
import math
import threading
import time
from pathlib import Path

import httpx

from core.config import Settings
from core.constants import AUTHORIZATION_HEADER, BEARER_PREFIX, PROVIDER_API, PROVIDER_LOCAL
from core.exceptions import ConfigurationError, EmbeddingError
from core.logging_config import get_logger

logger = get_logger("embeddings")
_backend_lock = threading.Lock()

PRELOAD_COMMAND = (
    'python -c "from sentence_transformers import SentenceTransformer; '
    "SentenceTransformer('{model}')\""
)


def _normalise(vector: list[float]) -> list[float]:
    """Scale to a unit vector; retrieval scores are dot products of unit vectors."""
    magnitude = math.sqrt(sum(value * value for value in vector))
    if not magnitude:
        return list(vector)
    return [value / magnitude for value in vector]


class _ApiBackend:
    """Embeds through an OpenAI-compatible /embeddings endpoint over HTTP."""

    def __init__(self, settings: Settings) -> None:
        base = (settings.embedding_api_base_url or settings.llm_base_url).rstrip("/")
        self._url = f"{base}/embeddings"
        self._key = settings.embedding_api_key or settings.llm_api_key
        self._model = settings.embedding_model
        self._client = httpx.AsyncClient(timeout=settings.llm_timeout_seconds)

    async def aencode(self, texts: list[str]) -> list[list[float]]:
        try:
            response = await self._client.post(
                self._url,
                headers={AUTHORIZATION_HEADER: f"{BEARER_PREFIX}{self._key}"},
                json={"model": self._model, "input": texts},
            )
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise EmbeddingError(f"The embedding endpoint could not be reached: {exc}") from exc

        items = payload.get("data")
        if not isinstance(items, list):
            raise EmbeddingError("The embedding endpoint returned no data array.")
        items = sorted(items, key=lambda item: item.get("index", 0))
        return [_normalise([float(value) for value in item["embedding"]]) for item in items]

    async def aclose(self) -> None:
        await self._client.aclose()


class _LocalBackend:
    """Runs a sentence-transformers model in-process (askdb[embeddings])."""

    def __init__(self, settings: Settings) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - depends on install extra
            raise EmbeddingError(
                "The local embedding backend needs the embeddings extra: "
                "pip install 'askdb[embeddings]'"
            ) from exc

        source, is_local = resolve_model_source(settings)
        if is_local:
            logger.info("embeddings: loading the model from the local path %s", source)
        else:
            logger.info(
                "embeddings: no local copy configured, so %s will be fetched from the Hugging "
                "Face Hub. Download it once with %s and set EMBEDDING_LOCAL_PATH to run offline.",
                settings.embedding_model,
                PRELOAD_COMMAND.format(model=settings.embedding_model),
            )
        self._settings = settings
        self.model = SentenceTransformer(source, device=settings.embedding_device)

    def encode(self, texts: list[str]) -> list[list[float]]:
        return [
            list(map(float, vector))
            for vector in self.model.encode(
                texts,
                batch_size=self._settings.embedding_batch_size,
                normalize_embeddings=True,
            )
        ]

    async def aencode(self, texts: list[str]) -> list[list[float]]:
        return await asyncio.to_thread(self.encode, texts)


def resolve_model_source(settings: Settings) -> tuple[str, bool]:
    """Prefer a local copy of the model, else the hub id. Returns (source, is_local)."""
    local = settings.embedding_local_path.strip()
    if local and Path(local).is_dir():
        return local, True
    if local:
        logger.warning(
            "embeddings: EMBEDDING_LOCAL_PATH %s is not a directory; "
            "falling back to the model id %s",
            local,
            settings.embedding_model,
        )
    return settings.embedding_model, False


class Embedder:
    """Turns text into vectors, via the configured provider."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._backend: _ApiBackend | _LocalBackend | None = None
        self._dimensions = settings.embedding_dimensions

    @property
    def enabled(self) -> bool:
        return self._settings.embedding_provider != "none"

    def _load(self):
        if self._backend is None:
            with _backend_lock:
                if self._backend is None:
                    started = time.perf_counter()
                    provider = self._settings.embedding_provider
                    if provider == PROVIDER_LOCAL:
                        self._backend = _LocalBackend(self._settings)
                    elif provider == PROVIDER_API:
                        self._backend = _ApiBackend(self._settings)
                    else:
                        raise ConfigurationError(
                            f"Unknown embedding provider {provider!r}; "
                            "use 'api', 'local' or 'none'."
                        )
                    logger.debug(
                        "embeddings: %s backend ready in %d ms, width %d",
                        provider,
                        int((time.perf_counter() - started) * 1000),
                        self._dimensions,
                    )
        return self._backend

    async def aencode(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            logger.debug("embeddings: nothing to encode")
            return []
        if not self.enabled:
            logger.debug("embeddings: the provider is 'none'")
            raise EmbeddingError("Embeddings are disabled.")
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
        backend = self._backend
        if backend is not None and hasattr(backend, "aclose"):
            await backend.aclose()
