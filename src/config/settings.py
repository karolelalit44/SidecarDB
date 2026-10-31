from typing import Annotated, Any

from pydantic import field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

from config.constants import (
    DEFAULT_SCHEMA,
    DEFAULT_STORAGE_DIR,
    LOG_COLOR,
    LOG_DEPENDENCY_LEVEL,
    LOG_FORMAT,
    LOG_LEVEL,
    LOG_LLM_PAYLOADS,
    LOG_SQL,
)

CsvList = Annotated[list[str], NoDecode]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    database_url: str = ""
    db_schema: str = DEFAULT_SCHEMA
    storage_dir: str = DEFAULT_STORAGE_DIR
    db_ssl: str = ""
    db_password: str = ""
    db_role: str = ""

    llm_api_key: str = ""
    llm_base_url: str = ""
    llm_model: str = ""
    llm_extra_body: dict[str, Any] = {}

    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_local_path: str = "usage/embeddings/bge-small-en-v1.5"
    embedding_dimensions: int = 384
    embedding_device: str = "cpu"

    api_keys: CsvList = []

    log_level: str = LOG_LEVEL
    log_dependency_level: str = LOG_DEPENDENCY_LEVEL
    log_sql: bool = LOG_SQL
    log_llm_payloads: bool = LOG_LLM_PAYLOADS
    log_format: str = LOG_FORMAT
    log_color: str = LOG_COLOR

    @field_validator("api_keys", mode="before")
    @classmethod
    def _split_csv(cls, value: object) -> object:
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    @field_validator("llm_extra_body", mode="before")
    @classmethod
    def _blank_extra_body_is_empty(cls, value: object) -> object:
        if value is None or value == "":
            return {}
        return value

    @field_validator("llm_base_url", mode="before")
    @classmethod
    def _strip_trailing_slash(cls, value: object) -> object:
        if isinstance(value, str):
            return value.rstrip("/")
        return value


def get_settings(_env_file: str | None = None, **overrides: Any) -> Settings:
    if _env_file:
        return Settings(_env_file=_env_file, **overrides)  # type: ignore[call-arg]
    return Settings(**overrides)
