import logging
import sys

from .config import settings
from .constants import COLOURS, DIM, RESET


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


class NamespaceFilter(logging.Filter):
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
    return logging.getLogger(f"{settings.log_namespace}.{name}")


def configure() -> None:
    namespace = settings.log_namespace
    app_level = getattr(logging, settings.log_level, logging.DEBUG)
    dep_level = getattr(logging, settings.log_dependency_level, logging.WARNING)

    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(ColouredFormatter())
    handler.addFilter(NamespaceFilter(namespace, app_level, dep_level))

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(min(app_level, dep_level))
