import secrets
import time
import uuid
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from fastapi import FastAPI, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from core.catalog import SchemaGraph
from core.constants import (
    APP_NAME,
    CORS_ALLOW_CREDENTIALS,
    CORS_ALLOW_HEADERS,
    CORS_ALLOW_METHODS,
    CORS_ALLOW_ORIGINS,
    ELAPSED_PRECISION,
    MAX_REQUEST_BODY_BYTES,
    RATE_LIMIT_ENABLED,
    RATE_LIMIT_REQUESTS,
    RATE_LIMIT_WINDOW_SECONDS,
)
from core.exceptions import SearchError
from core.wire import (
    AUTHORIZATION_HEADER,
    BEARER_PREFIX,
    STATUS_OK,
    HealthResponse,
    SearchRequest,
    SearchResponse,
)
from utils.logging_config import configure_logging, get_logger

if TYPE_CHECKING:
    from core.client import SidecarDB

logger = get_logger("api")
error_logger = get_logger("error")

PROTECTED_PREFIX = "/api/v1"

REQUEST_ID_HEADER = "X-Request-ID"
API_KEY_HEADER = "X-API-Key"
RETRY_AFTER_HEADER = "Retry-After"
REQUEST_ID_MAX_LENGTH = 64


class RateLimiter:
    def __init__(self, limit: int, window_seconds: float) -> None:
        self._limit = limit
        self._window = window_seconds
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        window = self._hits[key]
        cutoff = now - self._window
        while window and window[0] <= cutoff:
            window.popleft()
        if len(window) >= self._limit:
            return False
        window.append(now)
        return True


def _request_id(raw: str | None) -> str:
    if (
        raw
        and len(raw) <= REQUEST_ID_MAX_LENGTH
        and all(character.isalnum() or character in "-._" for character in raw)
    ):
        return raw
    return uuid.uuid4().hex


def _presented_key(request: Request) -> str:
    authorization = request.headers.get(AUTHORIZATION_HEADER.lower(), "")
    if authorization[: len(BEARER_PREFIX)].lower() == BEARER_PREFIX:
        return authorization[len(BEARER_PREFIX) :].strip()
    return request.headers.get(API_KEY_HEADER.lower(), "")


def _key_accepted(presented: str, accepted: list[str]) -> bool:
    if not presented:
        return False
    return any(secrets.compare_digest(presented, candidate) for candidate in accepted)


def _error(
    status: int,
    kind: str,
    message: str,
    request_id: str | None,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={"error": kind, "message": message, "request_id": request_id},
        headers=headers,
    )


def create_app(sidecar_db: "SidecarDB | None" = None) -> FastAPI:
    from core.client import SidecarDB as _SidecarDB

    sidecar_db = sidecar_db if sidecar_db is not None else _SidecarDB()
    settings = sidecar_db.settings
    configure_logging()
    limiter = RateLimiter(RATE_LIMIT_REQUESTS, RATE_LIMIT_WINDOW_SECONDS)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        logger.debug("lifespan: startup begins")
        await sidecar_db.start(require_database=False)
        logger.debug("lifespan: startup complete")
        yield
        logger.debug("lifespan: shutdown begins")
        await sidecar_db.stop()
        logger.debug("lifespan: shutdown complete")

    app = FastAPI(
        title=APP_NAME,
        version=sidecar_db.version,
        lifespan=lifespan,
        description="Natural-language search over a PostgreSQL schema.",
    )
    app.state.sidecar_db = sidecar_db

    @app.middleware("http")
    async def edge(request: Request, call_next):
        request.state.request_id = _request_id(request.headers.get(REQUEST_ID_HEADER.lower()))
        request_id = request.state.request_id
        started = time.perf_counter()

        declared_length = request.headers.get("content-length")
        if (
            declared_length
            and declared_length.isdigit()
            and (int(declared_length) > MAX_REQUEST_BODY_BYTES)
        ):
            logger.warning("[%s] request body too large: %s bytes", request_id, declared_length)
            return _error(
                413,
                "PayloadTooLarge",
                f"The request body exceeds {MAX_REQUEST_BODY_BYTES} bytes.",
                request_id,
            )

        protected = request.url.path.startswith(PROTECTED_PREFIX)
        if (
            protected
            and settings.api_keys
            and not _key_accepted(_presented_key(request), settings.api_keys)
        ):
            logger.warning("[%s] rejected an unauthenticated request", request_id)
            return _error(401, "Unauthorized", "A valid API key is required.", request_id)

        if protected and RATE_LIMIT_ENABLED:
            client = request.client.host if request.client else "unknown"
            if not limiter.allow(client):
                retry_after = max(int(RATE_LIMIT_WINDOW_SECONDS), 1)
                logger.warning("[%s] rate limit reached for %s", request_id, client)
                return _error(
                    429,
                    "TooManyRequests",
                    "Rate limit exceeded; slow down and retry.",
                    request_id,
                    headers={RETRY_AFTER_HEADER: str(retry_after)},
                )

        logger.debug("[%s] %s %s", request_id, request.method, request.url.path)
        response = await call_next(request)
        elapsed = (time.perf_counter() - started) * 1000
        response.headers[REQUEST_ID_HEADER] = request_id
        response.headers["X-Process-Time-Ms"] = f"{elapsed:.{ELAPSED_PRECISION}f}"
        logger.debug(
            "[%s] %s %s answered %d in %s ms",
            request_id,
            request.method,
            request.url.path,
            response.status_code,
            f"{elapsed:.{ELAPSED_PRECISION}f}",
        )
        return response

    if CORS_ALLOW_ORIGINS:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(CORS_ALLOW_ORIGINS),
            allow_credentials=CORS_ALLOW_CREDENTIALS,
            allow_methods=list(CORS_ALLOW_METHODS),
            allow_headers=list(CORS_ALLOW_HEADERS),
        )

    @app.exception_handler(SearchError)
    async def handle_search_error(request: Request, exc: SearchError) -> JSONResponse:
        request_id = getattr(request.state, "request_id", None)
        error_logger.warning("[%s] %s: %s", request_id, type(exc).__name__, exc)
        return _error(exc.status_code, type(exc).__name__, str(exc), request_id)

    @app.exception_handler(Exception)
    async def handle_unexpected(request: Request, exc: Exception) -> JSONResponse:
        import traceback

        request_id = getattr(request.state, "request_id", None)
        error_logger.error(
            "[%s] unhandled %s\n%s",
            request_id,
            type(exc).__name__,
            "".join(traceback.format_exception(exc)),
        )
        return _error(500, "InternalServerError", "The request could not be completed.", request_id)

    @app.get("/health", response_model=HealthResponse)
    async def health() -> "HealthResponse | JSONResponse":
        result = await sidecar_db.health()
        if result.status != STATUS_OK:
            return JSONResponse(status_code=503, content=result.model_dump())
        return result

    @app.get("/api/v1/schema", response_model=SchemaGraph)
    async def schema(refresh: bool = Query(default=False)) -> SchemaGraph:
        logger.debug("schema: refresh=%s", refresh)
        graph = await sidecar_db.catalog(refresh=refresh)
        logger.debug(
            "schema: returning %d tables and %d columns",
            len(graph.tables),
            sum(len(info.column_names) for info in graph.tables.values()),
        )
        return graph

    @app.post("/api/v1/search", response_model=SearchResponse)
    async def search(payload: SearchRequest, request: Request) -> SearchResponse:
        request_id = getattr(request.state, "request_id", None)
        logger.debug("[%s] search accepted: %r", request_id, payload.query)
        result = await sidecar_db.search(payload.query, request_id)
        logger.debug(
            "[%s] search answered: rows=%d elapsed=%s ms",
            request_id,
            result.row_count,
            result.elapsed_ms,
        )
        return result

    return app
