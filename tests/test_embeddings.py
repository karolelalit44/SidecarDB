import httpx
import pytest

from core.config import Settings
from core.constants import EMBEDDING_BATCH_SIZE
from infra.embeddings import _ApiBackend, _normalise


def test_normalise():
    vec = [3.0, 4.0]
    norm = _normalise(vec)
    assert pytest.approx(norm[0], 0.001) == 0.6
    assert pytest.approx(norm[1], 0.001) == 0.8
    assert _normalise([0.0, 0.0]) == [0.0, 0.0]


@pytest.mark.asyncio
async def test_api_backend_chunking(monkeypatch):
    settings = Settings(
        database_url="postgresql://test:test@localhost:5432/test",
        embedding_api_base_url="http://mock-api.local",
        embedding_api_key="mock-key",
        embedding_model="mock-embed",
        embedding_provider="api",
    )
    backend = _ApiBackend(settings)

    num_texts = EMBEDDING_BATCH_SIZE * 2 + 5
    texts = [f"text_{i}" for i in range(num_texts)]

    called_batches: list[list[str]] = []

    async def mock_post(url, headers, json):
        input_texts = json["input"]
        called_batches.append(input_texts)
        return httpx.Response(
            200,
            json={"data": [{"index": i, "embedding": [1.0, 0.0]} for i in range(len(input_texts))]},
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(backend._client, "post", mock_post)

    result = await backend.aencode(texts)

    assert len(called_batches) == 3
    assert len(called_batches[0]) == EMBEDDING_BATCH_SIZE
    assert len(called_batches[1]) == EMBEDDING_BATCH_SIZE
    assert len(called_batches[2]) == 5
    assert len(result) == num_texts

    await backend.aclose()
