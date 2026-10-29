from pathlib import Path

from core.config import Settings
from core.constants import PROVIDER_LOCAL, PROVIDER_NONE
from core.exceptions import ConfigurationError
from core.logging_config import get_logger

logger = get_logger("verification")


class StartupVerificationService:
    """Checks configuration at startup. Connections belong to the database."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def verify_database(self) -> None:
        if not self._settings.database_url:
            raise ConfigurationError(
                "Mandatory dependency missing: DATABASE_URL is not configured."
            )
        logger.debug("verification: DATABASE_URL is configured")

    def verify_llm(self) -> None:
        settings = self._settings
        logger.debug("verification: checking the model credentials for %s", settings.llm_model)
        if not settings.llm_api_key or not settings.llm_api_key.strip():
            raise ConfigurationError(
                "Mandatory dependency missing: LLM_API_KEY is not configured in environment."
            )
        logger.info("LLM configuration verification successful (model: %s).", settings.llm_model)

    def verify_embeddings(self) -> None:
        settings = self._settings
        logger.debug(
            "verification: embeddings provider=%s model=%s local=%s device=%s",
            settings.embedding_provider,
            settings.embedding_model,
            settings.embedding_local_path or "unset",
            settings.embedding_device,
        )
        if settings.embedding_provider == PROVIDER_NONE:
            logger.info("Embeddings are off; retrieval will use catalog order.")
            return
        if (
            settings.embedding_provider == PROVIDER_LOCAL
            and settings.embedding_local_path
            and not Path(settings.embedding_local_path).is_dir()
        ):
            logger.warning(
                "EMBEDDING_LOCAL_PATH %s does not exist; the model will be fetched from the Hub.",
                settings.embedding_local_path,
            )

    def verify(self) -> None:
        self.verify_database()
        self.verify_llm()
        self.verify_embeddings()
        logger.info("Startup prerequisite verification complete.")
