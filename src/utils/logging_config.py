import json
import logging
import sys

from core.constants import LOG_COLOR, LOG_DEPENDENCY_LEVEL, LOG_FORMAT, LOG_LEVEL

NAMESPACE = "sidecardb"

RESET = "\033[0m"
DIM = "\033[2m"

COLOURS = {
    "DEBUG": "\033[36m",
    "INFO": "\033[32m",
    "WARNING": "\033[33m",
    "ERROR": "\033[31m",
    "CRITICAL": "\033[1;31m",
}


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
    return logging.getLogger(f"{NAMESPACE}.{name}")


def _wants_colour() -> bool:
    if LOG_COLOR == "on":
        return True
    if LOG_COLOR == "off":
        return False
    return bool(sys.stdout.isatty())


def _formatter() -> logging.Formatter:
    if LOG_FORMAT == "json":
        return JsonFormatter()
    if _wants_colour():
        return ColouredFormatter()
    return logging.Formatter("%(asctime)s %(levelname)-8s %(name)s %(message)s")


def configure_logging() -> logging.Logger:
    namespace = NAMESPACE
    app_level = getattr(logging, LOG_LEVEL, logging.INFO)
    dep_level = getattr(logging, LOG_DEPENDENCY_LEVEL, logging.WARNING)

    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(_formatter())
    handler.addFilter(NamespaceFilter(namespace, app_level, dep_level))

    logger = logging.getLogger(namespace)
    logger.handlers.clear()
    logger.addHandler(handler)
    logger.setLevel(min(app_level, dep_level))
    logger.propagate = False
    return logger
