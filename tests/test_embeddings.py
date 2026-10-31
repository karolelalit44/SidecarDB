from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from config.settings import Settings
from providers.embeddings import Embedder, _normalise, get_embedding_model


def test_normalise():
    vec = [3.0, 4.0]
    norm = _normalise(vec)
    assert pytest.approx(norm[0], 0.001) == 0.6
    assert pytest.approx(norm[1], 0.001) == 0.8
    assert _normalise([0.0, 0.0]) == [0.0, 0.0]


def test_get_embedding_model_loads_locally_when_files_exist(tmp_path):
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")

    mock_st_cls = MagicMock()
    mock_instance = MagicMock()
    mock_st_cls.return_value = mock_instance

    mock_module = MagicMock(SentenceTransformer=mock_st_cls)
    with patch.dict("sys.modules", {"sentence_transformers": mock_module}):
        model = get_embedding_model("mock/model", tmp_path, device="cpu")

    mock_st_cls.assert_called_once_with(str(tmp_path), device="cpu")
    mock_instance.save.assert_not_called()
    assert model == mock_instance


def test_get_embedding_model_downloads_and_saves_on_first_run(tmp_path):
    target_dir = tmp_path / "downloaded_model"

    mock_st_cls = MagicMock()
    mock_instance = MagicMock()
    mock_st_cls.return_value = mock_instance

    mock_module = MagicMock(SentenceTransformer=mock_st_cls)
    with patch.dict("sys.modules", {"sentence_transformers": mock_module}):
        model = get_embedding_model("BAAI/bge-small-en-v1.5", target_dir, device="cpu")

    mock_st_cls.assert_called_once_with("BAAI/bge-small-en-v1.5", device="cpu")
    mock_instance.save.assert_called_once_with(str(target_dir))
    assert model == mock_instance


@pytest.mark.asyncio
async def test_embedder_encodes_via_local_backend(monkeypatch):
    settings = Settings(
        database_url="postgresql://test:test@localhost:5432/test",
        embedding_model="mock-embed",
        embedding_local_path="mock/path",
    )
    embedder = Embedder(settings)

    mock_backend = MagicMock()
    mock_backend.aencode = AsyncMock(return_value=[[0.1, 0.2], [0.3, 0.4]])
    embedder._backend = mock_backend

    res = await embedder.aencode(["hello", "world"])
    assert res == [[0.1, 0.2], [0.3, 0.4]]
    assert embedder.enabled is True
