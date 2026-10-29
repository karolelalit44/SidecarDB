from typing import Annotated, Any, Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

from core.constants import LogColour, LogFormat, LogLevel

CsvList = Annotated[list[str], NoDecode]


class Settings(BaseSettings):
    """Every knob the engine has, with production-sane defaults.

    Only ``database_url`` and ``llm_api_key`` are required. Everything else
    carries a default, so a working configuration is three keyword arguments
    long::

        Settings(database_url="postgresql://...", llm_api_key="sk-...")

    Values are resolved in the usual pydantic-settings order: explicit keyword
    arguments, then process environment variables, then an optional ``.env``
    file in the working directory, then the defaults declared here.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    app_name: str = "askdb"

    # --- logging -----------------------------------------------------------
    log_level: LogLevel = "INFO"
    log_dependency_level: LogLevel = "WARNING"
    log_sql: bool = False
    log_llm_payloads: bool = False
    log_format: LogFormat = "text"
    log_color: LogColour = "auto"

    # --- HTTP service ------------------------------------------------------
    host: str = "0.0.0.0"
    port: int = 8000
    # Accepted API keys. Empty means the service runs without authentication
    # and says so loudly at startup; set at least one key before exposing the
    # service to a network you do not control.
    api_keys: CsvList = []
    rate_limit_enabled: bool = True
    rate_limit_requests: int = 60
    rate_limit_window_seconds: float = 60.0
    max_request_body_bytes: int = 65536
    cors_allow_origins: CsvList = []
    cors_allow_methods: CsvList = ["GET", "POST", "OPTIONS"]
    cors_allow_headers: CsvList = ["Authorization", "Content-Type", "X-API-Key", "X-Request-ID"]
    cors_allow_credentials: bool = False

    # --- database ----------------------------------------------------------
    database_url: str
    db_schema: str = "public"
    # Passed straight to asyncpg: "", "disable", "require", "verify-ca" or
    # "verify-full". Empty defers to the libpq/asyncpg default.
    db_ssl: str = ""
    pool_min_size: int = 1
    pool_max_size: int = 8
    db_connect_retries: int = 3
    db_connect_backoff_seconds: float = 0.5
    db_max_inactive_connection_lifetime: float = 300.0
    introspection_timeout_seconds: float = 15.0

    # --- execution sandbox -------------------------------------------------
    sandbox_statement_timeout_seconds: float = 5.0
    sandbox_app_timeout_seconds: float = 15.0
    sandbox_idle_timeout_ms: int = 10000
    sandbox_max_rows: int = 5000
    sandbox_transaction_read_only: bool = True

    # --- planning ----------------------------------------------------------
    min_query_length: int = 2
    max_query_length: int = 250
    default_limit: int = 25
    max_limit: int = 200
    max_plan_attempts: int = 3

    # Characters of rejection detail fed back to the model for repair. It lists
    # the valid column and operator names, so it is trimmed to keep the prompt
    # inside the context window.
    repair_char_budget: int = 800
    max_schema_tables: int = 30
    max_schema_columns: int = 120

    # --- language model ----------------------------------------------------
    llm_api_key: str
    llm_base_url: str = "https://api.openai.com/v1"
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.0
    # Temperature used when a plan is rejected and retried. At llm_temperature
    # the model reproduces the rejected reply verbatim, so a repair could never
    # differ and every retry would cost full model latency to reach the same
    # failure.
    llm_repair_temperature: float = 0.6
    llm_max_tokens: int = 2048
    llm_timeout_seconds: float = 120.0
    llm_retries: int = 2
    llm_context_tokens: int = 8192
    llm_chars_per_token: int = 4
    # Provider-specific request options, merged into every chat completion, as
    # a JSON object. A reasoning model bills its scratchpad against
    # llm_max_tokens, so set {"chat_template_kwargs": {"enable_thinking": false}}
    # to turn thinking off where the provider supports it.
    llm_extra_body: dict[str, Any] = {}

    # --- embeddings --------------------------------------------------------
    # "api": call an OpenAI-compatible /embeddings endpoint over HTTP (default,
    #        no heavyweight dependencies).
    # "local": run a sentence-transformers model in-process (needs the
    #        askdb[embeddings] extra).
    # "none": no retrieval index; every question is planned from catalog order.
    embedding_provider: Literal["api", "local", "none"] = "api"
    embedding_model: str = "text-embedding-3-small"
    # Empty means "reuse llm_base_url / llm_api_key".
    embedding_api_base_url: str = ""
    embedding_api_key: str = ""
    embedding_dimensions: int = 1536
    # A local copy of a sentence-transformers model directory. Leave empty to
    # fetch from the Hugging Face Hub on first use.
    embedding_local_path: str = ""
    embedding_device: str = "cpu"
    embedding_batch_size: int = 32

    # --- retrieval ---------------------------------------------------------
    retrieval_top_k: int = 12
    retrieval_min_score: float = 0.15
    retrieval_score_precision: int = 6

    # --- behaviour ---------------------------------------------------------
    error_message_max_chars: int = 200
    redact_catalog_in_errors: bool = True
    # Off by default: returning generated SQL to callers discloses table
    # names and query shape.
    expose_sql: bool = False
    elapsed_precision: int = 2

    no_records_message: str = "No matching records were found."
    no_summary_message: str = "The query returned records but no summary could be produced."

    # --- resilience --------------------------------------------------------
    health_ping_cache_seconds: float = 2.0
    recovery_interval_seconds: float = 15.0

    @field_validator(
        "api_keys",
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
    """Build a Settings instance; keyword arguments win over the environment."""
    return Settings(**overrides)
