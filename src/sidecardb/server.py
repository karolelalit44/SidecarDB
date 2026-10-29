import secrets
import time
import uuid
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from fastapi import FastAPI, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .constants import (
    API_KEY_HEADER,
    AUTHORIZATION_HEADER,
    BEARER_PREFIX,
    REQUEST_ID_HEADER,
    REQUEST_ID_MAX_LENGTH,
    RETRY_AFTER_HEADER,
)
from .exceptions import SearchError
from .logging_config import configure_logging, get_logger
from .schema import HealthResponse, SearchRequest, SearchResponse
from .types import SchemaGraph

if TYPE_CHECKING:
    from .client import SidecarDB

logger = get_logger("api")
error_logger = get_logger("error")

PROTECTED_PREFIX = "/api/v1"


class RateLimiter:
    """A sliding-window counter per client. In-process by design.

    A single instance bounds one process. Behind a load balancer, terminate
    rate limits at the edge as well.
    """

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
    """Trust a client request id only when it is short and printable."""
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


def create_app(sidecar: "SidecarDB | None" = None) -> FastAPI:
    """Build the HTTP service around a SidecarDB instance.

    With no argument the instance is built from the environment, so this is
    also a valid uvicorn factory: ``uvicorn sidecardb.server:create_app
    --factory``.
    """
    from .client import SidecarDB as _SidecarDB

    sidecar = sidecar if sidecar is not None else _SidecarDB()
    settings = sidecar.settings
    configure_logging(settings)
    limiter = RateLimiter(settings.rate_limit_requests, settings.rate_limit_window_seconds)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        logger.debug("lifespan: startup begins")
        await sidecar.start(require_database=False)
        logger.debug("lifespan: startup complete")
        yield
        logger.debug("lifespan: shutdown begins")
        await sidecar.stop()
        logger.debug("lifespan: shutdown complete")

    app = FastAPI(
        title=settings.app_name,
        version=sidecar.version,
        lifespan=lifespan,
        description="Natural-language search over a PostgreSQL schema.",
    )
    app.state.sidecar = sidecar

    @app.middleware("http")
    async def edge(request: Request, call_next):
        request.state.request_id = _request_id(request.headers.get(REQUEST_ID_HEADER.lower()))
        request_id = request.state.request_id
        started = time.perf_counter()

        declared_length = request.headers.get("content-length")
        if (
            declared_length
            and declared_length.isdigit()
            and (int(declared_length) > settings.max_request_body_bytes)
        ):
            logger.warning("[%s] request body too large: %s bytes", request_id, declared_length)
            return JSONResponse(
                status_code=413,
                content={
                    "error": "PayloadTooLarge",
                    "message": f"The request body exceeds {settings.max_request_body_bytes} bytes.",
                    "request_id": request_id,
                },
            )

        protected = request.url.path.startswith(PROTECTED_PREFIX)
        if (
            protected
            and settings.api_keys
            and not _key_accepted(_presented_key(request), settings.api_keys)
        ):
            logger.warning("[%s] rejected an unauthenticated request", request_id)
            return JSONResponse(
                status_code=401,
                content={
                    "error": "Unauthorized",
                    "message": "A valid API key is required.",
                    "request_id": request_id,
                },
            )

        if protected and settings.rate_limit_enabled:
            client = request.client.host if request.client else "unknown"
            if not limiter.allow(client):
                retry_after = max(int(settings.rate_limit_window_seconds), 1)
                logger.warning("[%s] rate limit reached for %s", request_id, client)
                return JSONResponse(
                    status_code=429,
                    content={
                        "error": "TooManyRequests",
                        "message": "Rate limit exceeded; slow down and retry.",
                        "request_id": request_id,
                    },
                    headers={RETRY_AFTER_HEADER: str(retry_after)},
                )

        logger.debug("[%s] %s %s", request_id, request.method, request.url.path)
        response = await call_next(request)
        elapsed = (time.perf_counter() - started) * 1000
        response.headers[REQUEST_ID_HEADER] = request_id
        response.headers["X-Process-Time-Ms"] = f"{elapsed:.{settings.elapsed_precision}f}"
        logger.debug(
            "[%s] %s %s answered %d in %s ms",
            request_id,
            request.method,
            request.url.path,
            response.status_code,
            f"{elapsed:.{settings.elapsed_precision}f}",
        )
        return response

    if settings.cors_allow_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_allow_origins,
            allow_credentials=settings.cors_allow_credentials,
            allow_methods=settings.cors_allow_methods,
            allow_headers=settings.cors_allow_headers,
        )

    @app.exception_handler(SearchError)
    async def handle_search_error(request: Request, exc: SearchError) -> JSONResponse:
        request_id = getattr(request.state, "request_id", None)
        error_logger.warning("[%s] %s: %s", request_id, type(exc).__name__, exc)
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": type(exc).__name__, "message": str(exc), "request_id": request_id},
        )

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
        return JSONResponse(
            status_code=500,
            content={
                "error": "InternalServerError",
                "message": "The request could not be completed.",
                "request_id": request_id,
            },
        )

    @app.get("/health", response_model=HealthResponse)
    async def health() -> "HealthResponse | JSONResponse":
        """Liveness: pings the database (cached for a moment), never runs a query."""
        result = await sidecar.health()
        if result.status != "ok":
            # 503 so container orchestrators and load balancers see the truth.
            return JSONResponse(status_code=503, content=result.model_dump())
        return result

    @app.get("/api/v1/schema", response_model=SchemaGraph)
    async def schema(refresh: bool = Query(default=False)) -> SchemaGraph:
        logger.debug("schema: refresh=%s", refresh)
        if refresh:
            graph = await sidecar.refresh()
        else:
            graph = await sidecar.schema()
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
        result = await sidecar.search(payload.query, request_id)
        logger.debug(
            "[%s] search answered: rows=%d elapsed=%s ms",
            request_id,
            result.row_count,
            result.elapsed_ms,
        )
        return result

    return app
