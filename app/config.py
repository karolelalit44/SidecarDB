from typing import Annotated

from pydantic import field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

from app.constants import LogLevel

CsvList = Annotated[list[str], NoDecode]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    app_name: str
    app_version: str

    log_level: LogLevel
    log_dependency_level: LogLevel
    log_namespace: str
    log_sql: bool

    cors_allow_origins: CsvList
    cors_allow_methods: CsvList
    cors_allow_headers: CsvList
    cors_allow_credentials: bool

    database_url: str
    db_schema: str
    db_identifier_max_length: int

    pool_min_size: int
    pool_max_size: int

    introspection_timeout_seconds: float
    sandbox_statement_timeout_seconds: float
    sandbox_app_timeout_seconds: float
    sandbox_idle_timeout_ms: int
    sandbox_max_rows: int
    sandbox_transaction_read_only: bool

    min_query_length: int
    max_query_length: int
    default_limit: int
    max_limit: int
    max_join_depth: int
    max_plan_attempts: int
    max_schema_tables: int
    max_schema_columns: int

    llm_api_key: str
    llm_base_url: str
    llm_model: str
    llm_temperature: float
    llm_max_tokens: int
    llm_timeout_seconds: float
    llm_retries: int

    embedding_enabled: bool
    embedding_model: str
    embedding_dimensions: int
    embedding_device: str
    embedding_batch_size: int

    retrieval_top_k: int
    retrieval_min_score: float
    retrieval_score_precision: int

    fallback_enabled: bool
    fallback_top_k: int

    sample_values_enabled: bool
    value_sample_limit: int
    value_sample_max: int
    value_sample_max_length: int

    error_message_max_chars: int
    vector_literal_precision: int

    redact_catalog_in_errors: bool
    expose_sql: bool

    elapsed_precision: int

    narration_char_budget: int
    narration_max_tokens: int

    no_records_message: str
    no_summary_message: str

    @field_validator(
        "cors_allow_origins",
        "cors_allow_methods",
        "cors_allow_headers",
        mode="before",
    )
    @classmethod
    def _split_csv(cls, value: object) -> object:
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value


def get_settings() -> Settings:
    return Settings()


settings = get_settings()
