from typing import Annotated, Any, Final, Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

from utils.text import split_csv

PROVIDER_API: Final = "api"
PROVIDER_LOCAL: Final = "local"
PROVIDER_NONE: Final = "none"

CsvList = Annotated[list[str], NoDecode]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    database_url: str
    db_schema: str = "public"
    db_ssl: str = ""

    llm_api_key: str
    llm_base_url: str = "https://api.openai.com/v1"
    llm_model: str = "gpt-4o-mini"
    llm_extra_body: dict[str, Any] = {}

    embedding_provider: Literal["api", "local", "none"] = "api"
    embedding_model: str = "text-embedding-3-small"
    embedding_api_base_url: str = ""
    embedding_api_key: str = ""
    embedding_dimensions: int = 1536
    embedding_local_path: str = ""
    embedding_device: str = "cpu"

    api_keys: CsvList = []

    @field_validator("api_keys", mode="before")
    @classmethod
    def _split_csv(cls, value: object) -> object:
        if isinstance(value, str):
            return split_csv(value)
        return value

    @field_validator("llm_extra_body", mode="before")
    @classmethod
    def _blank_extra_body_is_empty(cls, value: object) -> object:
        if value is None or value == "":
            return {}
        return value

    @field_validator("llm_base_url", "embedding_api_base_url", mode="before")
    @classmethod
    def _strip_trailing_slash(cls, value: object) -> object:
        if isinstance(value, str):
            return value.rstrip("/")
        return value


def get_settings(**overrides: Any) -> Settings:
    return Settings(**overrides)
