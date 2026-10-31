from pathlib import Path
from typing import TYPE_CHECKING

from core.exceptions import EmbeddingError
from core.logging import get_logger

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logger = get_logger("providers.embeddings")


def get_embedding_model(
    model_name: str,
    local_path: str | Path,
    device: str = "cpu",
) -> "SentenceTransformer":
    """Resolve and load the singular embedding model."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise EmbeddingError(
            "The embedding subsystem requires sentence-transformers. "
            "Install with: pip install 'sidecardb[embeddings]'"
        ) from exc

    target_dir = Path(local_path)

    has_local_files = target_dir.is_dir() and (
        (target_dir / "modules.json").is_file()
        or (target_dir / "config.json").is_file()
        or any(target_dir.glob("*.safetensors"))
        or any(target_dir.glob("*.bin"))
    )

    if has_local_files:
        logger.info("embeddings: loading local model offline from %s", target_dir)
        return SentenceTransformer(str(target_dir), device=device)

    logger.info(
        "embeddings: model not found locally at %s; downloading %s from Hub on first run...",
        target_dir,
        model_name,
    )
    target_dir.mkdir(parents=True, exist_ok=True)
    model = SentenceTransformer(model_name, device=device)
    model.save(str(target_dir))
    logger.info("embeddings: model %s successfully saved to %s", model_name, target_dir)
    return model


__all__ = ["get_embedding_model"]
