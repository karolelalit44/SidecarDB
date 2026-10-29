import json
import logging
from pathlib import Path

import pytest
from pydantic import ValidationError
from pydantic.fields import PydanticUndefined

from sidecardb.config import Settings
from sidecardb.constants import COLOURS, NAMESPACE
from sidecardb.logging_config import (
    ColouredFormatter,
    JsonFormatter,
    NamespaceFilter,
    configure_logging,
    get_logger,
)

ROOT = Path(__file__).resolve().parent.parent
CREDENTIAL_FIELDS = frozenset({"database_url", "llm_api_key"})


def env_keys(path: Path) -> set[str]:
    text = path.read_text(encoding="utf-8")
    return {
        line.split("=", 1)[0].strip().lower()
        for line in text.splitlines()
        if line.strip() and not line.strip().startswith("#")
    }


def test_only_the_two_credentials_are_required():
    """ "One simple config" means two values, not a copied sixty-line file."""
    required = {name for name, field in Settings.model_fields.items() if field.is_required()}
    assert required == CREDENTIAL_FIELDS


def test_every_other_setting_has_a_python_default():
    offenders = [
        name
        for name, field in Settings.model_fields.items()
        if name not in CREDENTIAL_FIELDS
        and field.default is PydanticUndefined
        and field.default_factory is None
    ]
    assert not offenders, f"settings without a default: {sorted(offenders)}"


def test_example_env_covers_every_setting_exactly():
    declared = {name.lower() for name in Settings.model_fields}
    example = env_keys(ROOT / ".env.example")
    assert example == declared


def test_example_env_leaves_only_optional_values_blank():
    """The credential must be supplied; blanks are otherwise optional extras."""
    text = (ROOT / ".env.example").read_text(encoding="utf-8")
    values = dict(line.split("=", 1) for line in text.splitlines() if line.strip() and "=" in line)
    blank = {key for key, value in values.items() if value == ""}
    assert blank == {
        "API_KEYS",
        "CORS_ALLOW_ORIGINS",
        "DB_SSL",
        "EMBEDDING_API_BASE_URL",
        "EMBEDDING_API_KEY",
        "EMBEDDING_LOCAL_PATH",
        "LLM_API_KEY",
    }


def test_app_name_defaults_to_sidecardb():
    settings = Settings(_env_file=None, database_url="postgresql://x/x", llm_api_key="k")
    assert settings.app_name == "sidecardb"


def test_expose_sql_is_off_by_default():
    settings = Settings(_env_file=None, database_url="postgresql://x/x", llm_api_key="k")
    assert settings.expose_sql is False


def test_invalid_log_level_is_rejected():
    with pytest.raises(ValidationError):
        Settings(
            _env_file=None,
            database_url="postgresql://x/x",
            llm_api_key="k",
            log_level="chatty",
        )


def test_missing_database_url_is_a_validation_error_at_construction():
    """Importing the package never requires configuration; constructing does."""
    with pytest.raises(ValidationError):
        Settings(_env_file=None, llm_api_key="k")


def test_formatter_colours_each_level():
    formatter = ColouredFormatter()
    for level in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
        record = logging.LogRecord("t", logging.getLevelName(level), "f", 1, "msg", None, None)
        output = formatter.format(record)
        assert COLOURS[level] in output
        assert level in output
        assert "msg" in output
        assert output.endswith("msg")


def test_formatter_tolerates_percent_arguments():
    record = logging.LogRecord("t", logging.INFO, "f", 1, "rows=%d", (7,), None)
    assert ColouredFormatter().format(record).endswith("rows=7")


def test_json_formatter_emits_one_object_per_line():
    record = logging.LogRecord("t", logging.INFO, "f", 1, "rows=%d", (7,), None)
    payload = json.loads(JsonFormatter().format(record))
    assert payload["message"] == "rows=7"
    assert payload["level"] == "INFO"
    assert payload["logger"] == "t"


def test_json_formatter_carries_the_request_id():
    record = logging.LogRecord("t", logging.INFO, "f", 1, "m", None, None)
    record.request_id = "abc123"
    payload = json.loads(JsonFormatter().format(record))
    assert payload["request_id"] == "abc123"


def record(name: str, level: int) -> logging.LogRecord:
    return logging.LogRecord(name, level, "f", 1, "m", None, None)


def test_configure_attaches_one_handler_to_the_sidecarb_logger_only():
    """The library must never clear a host application's root handlers."""
    root = logging.getLogger()
    sidecar = logging.getLogger(NAMESPACE)
    previous_root_level, previous_root_handlers = root.level, list(root.handlers)
    previous_sidecar_level, previous_sidecar_handlers = sidecar.level, list(sidecar.handlers)
    try:
        settings = Settings(_env_file=None, database_url="postgresql://x/x", llm_api_key="k")
        configure_logging(settings)
        configure_logging(settings)
        assert len(sidecar.handlers) == 1
        assert sidecar.propagate is False
        assert root.handlers == previous_root_handlers
        assert root.level == previous_root_level
    finally:
        root.setLevel(previous_root_level)
        root.handlers[:] = previous_root_handlers
        sidecar.setLevel(previous_sidecar_level)
        sidecar.handlers[:] = previous_sidecar_handlers


def test_get_logger_prefixes_the_namespace():
    assert get_logger("graph").name == f"{NAMESPACE}.graph"


def test_application_debug_is_dropped_when_app_level_is_error():
    rule = NamespaceFilter(NAMESPACE, logging.ERROR, logging.DEBUG)
    assert not rule.filter(record(f"{NAMESPACE}.graph", logging.DEBUG))


def test_application_debug_passes_at_debug():
    rule = NamespaceFilter(NAMESPACE, logging.DEBUG, logging.WARNING)
    assert rule.filter(record(f"{NAMESPACE}.graph", logging.DEBUG))


def test_dependency_debug_records_are_suppressed():
    rule = NamespaceFilter(NAMESPACE, logging.DEBUG, logging.WARNING)
    assert not rule.filter(record("httpcore.connection", logging.DEBUG))


def test_dependency_debug_records_pass_when_dependency_level_is_debug():
    rule = NamespaceFilter(NAMESPACE, logging.DEBUG, logging.DEBUG)
    assert rule.filter(record("httpcore.connection", logging.DEBUG))


def test_dependency_warning_records_pass():
    rule = NamespaceFilter(NAMESPACE, logging.DEBUG, logging.WARNING)
    assert rule.filter(record("httpcore.connection", logging.WARNING))


def test_namespace_prefix_is_not_matched_by_partial_name():
    rule = NamespaceFilter("nl_search_engine", logging.DEBUG, logging.WARNING)
    assert not rule.filter(record("nl_search_engine_extra.graph", logging.DEBUG))
