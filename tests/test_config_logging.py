import logging
import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from pydantic.fields import PydanticUndefined

from app.config import Settings, settings
from app.logging_config import (
    COLOURS,
    ColouredFormatter,
    NamespaceFilter,
    configure,
    get_logger,
)
from app.main import app

ROOT = Path(__file__).resolve().parent.parent
ANSI = re.compile(r"\033\[[0-9;]*m")


def env_keys(path: Path) -> set[str]:
    text = path.read_text(encoding="utf-8")
    return {
        line.split("=", 1)[0].strip().lower()
        for line in text.splitlines()
        if line.strip() and not line.strip().startswith("#")
    }


@pytest.mark.parametrize("name", sorted(Settings.model_fields))
def test_no_python_default_on_any_setting(name):
    field = Settings.model_fields[name]
    assert field.default is PydanticUndefined
    assert field.default_factory is None


def test_example_env_covers_every_setting_exactly():
    declared = {name.lower() for name in Settings.model_fields}
    example = env_keys(ROOT / ".env.example")
    assert example == declared


def test_example_env_has_no_blank_value_for_a_required_setting():
    text = (ROOT / ".env.example").read_text(encoding="utf-8")
    values = dict(line.split("=", 1) for line in text.splitlines() if line.strip() and "=" in line)
    blank = [key for key, value in values.items() if value == ""]
    assert blank == ["LLM_API_KEY"]


def test_invalid_log_level_is_rejected():
    with pytest.raises(ValidationError):
        Settings(log_level="chatty")


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


def record(name: str, level: int) -> logging.LogRecord:
    return logging.LogRecord(name, level, "f", 1, "m", None, None)


def test_configure_is_idempotent_and_clears_previous_handlers():
    root = logging.getLogger()
    previous_level, previous_handlers = root.level, list(root.handlers)
    try:
        configure()
        assert len(root.handlers) == 1
        configure()
        assert len(root.handlers) == 1
        assert root.level == min(
            getattr(logging, settings.log_level),
            getattr(logging, settings.log_dependency_level),
        )
    finally:
        root.setLevel(previous_level)
        root.handlers[:] = previous_handlers


def test_get_logger_prefixes_the_namespace():
    assert get_logger("graph").name == f"{settings.log_namespace}.graph"


def test_application_debug_is_dropped_when_app_level_is_error():
    rule = NamespaceFilter(settings.log_namespace, logging.ERROR, logging.DEBUG)
    assert not rule.filter(record(f"{settings.log_namespace}.graph", logging.DEBUG))


def test_application_debug_passes_at_debug():
    rule = NamespaceFilter(settings.log_namespace, logging.DEBUG, logging.WARNING)
    assert rule.filter(record(f"{settings.log_namespace}.graph", logging.DEBUG))


def test_dependency_debug_records_are_suppressed():
    rule = NamespaceFilter(settings.log_namespace, logging.DEBUG, logging.WARNING)
    assert not rule.filter(record("httpcore.connection", logging.DEBUG))


def test_dependency_debug_records_pass_when_dependency_level_is_debug():
    rule = NamespaceFilter(settings.log_namespace, logging.DEBUG, logging.DEBUG)
    assert rule.filter(record("httpcore.connection", logging.DEBUG))


def test_dependency_warning_records_pass():
    rule = NamespaceFilter(settings.log_namespace, logging.DEBUG, logging.WARNING)
    assert rule.filter(record("httpcore.connection", logging.WARNING))


def test_namespace_prefix_is_not_matched_by_partial_name():
    rule = NamespaceFilter("nl_search_engine", logging.DEBUG, logging.WARNING)
    assert not rule.filter(record("nl_search_engine_extra.graph", logging.DEBUG))


def test_fallback_candidates_endpoint_is_not_exposed():
    client = TestClient(app, raise_server_exceptions=False)
    response = client.get("/api/v1/fallback/candidates")
    assert response.status_code == 404


def test_search_endpoint_is_exposed():
    assert "/api/v1/search" in [route.path for route in app.routes]


def test_vector_columns_endpoint_is_not_exposed():
    assert "/api/v1/vector-columns" not in [route.path for route in app.routes]
