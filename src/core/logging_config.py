import json
import logging
import sys
from typing import TYPE_CHECKING

from core.constants import COLOURS, DIM, NAMESPACE, RESET

if TYPE_CHECKING:
    from core.config import Settings


class ColouredFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        colour = COLOURS.get(record.levelname, "")
        timestamp = self.formatTime(record, "%Y-%m-%d %H:%M:%S")
        head = (
            f"{DIM}{timestamp}{RESET} "
            f"{colour}{record.levelname:<8}{RESET} "
            f"{DIM}{record.name}{RESET}"
        )
        return f"{head} {record.getMessage()}"


class JsonFormatter(logging.Formatter):
    """One JSON object per line, for log aggregation."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        request_id = getattr(record, "request_id", None)
        if isinstance(request_id, str):
            payload["request_id"] = request_id
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


class NamespaceFilter(logging.Filter):
    """Applies one level to askdb records and another to everything else."""

    def __init__(self, namespace: str, app_level: int, dependency_level: int) -> None:
        super().__init__()
        self._namespace = namespace
        self._app_level = app_level
        self._dependency_level = dependency_level

    def _is_ours(self, name: str) -> bool:
        return name == self._namespace or name.startswith(f"{self._namespace}.")

    def filter(self, record: logging.LogRecord) -> bool:
        if self._is_ours(record.name):
            return record.levelno >= self._app_level
        return record.levelno >= self._dependency_level


def get_logger(name: str) -> logging.Logger:
    """A logger under the askdb namespace; safe to call at import time."""
    return logging.getLogger(f"{NAMESPACE}.{name}")


def _wants_colour(settings: "Settings") -> bool:
    if settings.log_color == "on":
        return True
    if settings.log_color == "off":
        return False
    return bool(sys.stdout.isatty())


def _formatter(settings: "Settings") -> logging.Formatter:
    if settings.log_format == "json":
        return JsonFormatter()
    if _wants_colour(settings):
        return ColouredFormatter()
    return logging.Formatter("%(asctime)s %(levelname)-8s %(name)s %(message)s")


def configure_logging(settings: "Settings") -> logging.Logger:
    """Attach one handler to the askdb logger.

    The root logger is never touched: when askdb runs inside a host
    application, that application owns its own logging. The CLI and Docker
    entrypoint call this to get the standalone service's output.
    """
    namespace = NAMESPACE
    app_level = getattr(logging, settings.log_level, logging.INFO)
    dep_level = getattr(logging, settings.log_dependency_level, logging.WARNING)

    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(_formatter(settings))
    handler.addFilter(NamespaceFilter(namespace, app_level, dep_level))

    logger = logging.getLogger(namespace)
    logger.handlers.clear()
    logger.addHandler(handler)
    logger.setLevel(min(app_level, dep_level))
    logger.propagate = False
    return logger
