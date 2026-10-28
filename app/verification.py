from .config import settings
from .db import database
from .exceptions import ConfigurationError
from .logging_config import get_logger

logger = get_logger("verification")


class StartupVerificationService:
    """Service verifying system prerequisites on application startup."""

    async def verify_database(self) -> None:
        if not settings.database_url:
            raise ConfigurationError(
                "Mandatory dependency missing: DATABASE_URL is not configured."
            )

        try:
            await database.connect()
            connected = await database.ping()
        except Exception as exc:
            logger.error("Database connection verification failed: %s", exc)
            raise ConfigurationError(
                f"Mandatory dependency unavailable: Database could not be reached ({exc})."
            ) from exc

        if not connected:
            raise ConfigurationError("Mandatory dependency unavailable: Database ping failed.")

        logger.info("Database verification successful.")

    async def verify_llm(self) -> None:
        if not settings.llm_api_key or not settings.llm_api_key.strip():
            raise ConfigurationError(
                "Mandatory dependency missing: LLM_API_KEY is not configured in environment."
            )
        logger.info("LLM configuration verification successful (model: %s).", settings.llm_model)

    async def verify_embeddings(self) -> None:
        if not settings.embedding_enabled:
            logger.info("Embeddings are disabled by configuration (optional dependency).")

    async def verify(self) -> None:
        await self.verify_database()
        await self.verify_llm()
        await self.verify_embeddings()
        logger.info("Startup prerequisite verification complete.")


verification_service = StartupVerificationService()
