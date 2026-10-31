import time
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config.constants import (
    APP_NAME,
    CORS_ALLOW_CREDENTIALS,
    CORS_ALLOW_HEADERS,
    CORS_ALLOW_METHODS,
    CORS_ALLOW_ORIGINS,
    ELAPSED_PRECISION,
    MAX_REQUEST_BODY_BYTES,
    PROTECTED_PREFIX,
    RATE_LIMIT_ENABLED,
    RATE_LIMIT_REQUESTS,
    RATE_LIMIT_WINDOW_SECONDS,
    REQUEST_ID_HEADER,
    RETRY_AFTER_HEADER,
)
from core.exceptions import SearchError
from core.logging import configure_logging, get_logger
from server.middleware import (
    BodyCapMiddleware,
    RateLimiter,
    api_key_accepted,
    error_response,
    presented_api_key,
    resolve_request_id,
)
from server.routes import register_routes

if TYPE_CHECKING:
    from core.client import SidecarDB

logger = get_logger("server.app")
error_logger = get_logger("server.error")


def create_app(sidecar_db: "SidecarDB | None" = None) -> FastAPI:
    if sidecar_db is None:
        from core.client import SidecarDB

        sidecar_db = SidecarDB()

    settings = sidecar_db.settings
    configure_logging(
        level=settings.log_level,
        dependency_level=settings.log_dependency_level,
        format_type=settings.log_format,
        color=settings.log_color,
    )
    logger.debug(
        "create_app: initializing FastAPI for %s v%s (schema=%s)",
        APP_NAME,
        sidecar_db.version,
        settings.db_schema,
    )
    limiter = RateLimiter(RATE_LIMIT_REQUESTS, RATE_LIMIT_WINDOW_SECONDS)
    logger.debug(
        "create_app: rate_limiter enabled=%s (limit=%d window=%.1fs)",
        RATE_LIMIT_ENABLED,
        RATE_LIMIT_REQUESTS,
        RATE_LIMIT_WINDOW_SECONDS,
    )

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

    app.add_middleware(BodyCapMiddleware, max_bytes=MAX_REQUEST_BODY_BYTES)

    @app.middleware("http")
    async def edge(request: Request, call_next):
        raw_req_id = request.headers.get(REQUEST_ID_HEADER.lower())
        request.state.request_id = resolve_request_id(raw_req_id)
        request_id = request.state.request_id
        started = time.perf_counter()

        logger.debug(
            "[%s] HTTP incoming %s %s from %s (headers=%d, custom_id=%s)",
            request_id,
            request.method,
            request.url.path,
            request.client.host if request.client else "unknown",
            len(request.headers),
            bool(raw_req_id and raw_req_id == request_id),
        )

        protected = request.url.path.startswith(PROTECTED_PREFIX)
        if (
            protected
            and settings.api_keys
            and not api_key_accepted(presented_api_key(request), settings.api_keys)
        ):
            logger.warning("[%s] rejected an unauthenticated request", request_id)
            return error_response(401, "Unauthorized", "A valid API key is required.", request_id)
        if protected and settings.api_keys:
            logger.debug("[%s] request authentication succeeded", request_id)
        elif protected:
            logger.debug("[%s] protected route accessed with no API_KEYS configured", request_id)

        if protected and RATE_LIMIT_ENABLED:
            client = request.client.host if request.client else "unknown"
            if not limiter.allow(client):
                retry_after = max(int(RATE_LIMIT_WINDOW_SECONDS), 1)
                logger.warning("[%s] rate limit reached for %s", request_id, client)
                return error_response(
                    429,
                    "TooManyRequests",
                    "Rate limit exceeded; slow down and retry.",
                    request_id,
                    headers={RETRY_AFTER_HEADER: str(retry_after)},
                )
            logger.debug("[%s] rate limit check passed for client %s", request_id, client)

        logger.debug(
            "[%s] dispatching %s %s to endpoint",
            request_id,
            request.method,
            request.url.path,
        )
        response = await call_next(request)
        elapsed = (time.perf_counter() - started) * 1000
        response.headers[REQUEST_ID_HEADER] = request_id
        response.headers["X-Process-Time-Ms"] = f"{elapsed:.{ELAPSED_PRECISION}f}"
        logger.debug(
            "[%s] %s %s answered %d in %s ms (response headers: %d)",
            request_id,
            request.method,
            request.url.path,
            response.status_code,
            f"{elapsed:.{ELAPSED_PRECISION}f}",
            len(response.headers),
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
        logger.debug(
            "[%s] SearchError handled: status=%d error=%s message=%s",
            request_id,
            exc.status_code,
            type(exc).__name__,
            exc,
        )
        return error_response(exc.status_code, type(exc).__name__, str(exc), request_id)

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
        logger.debug(
            "[%s] unhandled exception: %s: %s",
            request_id,
            type(exc).__name__,
            exc,
        )
        return error_response(
            500, "InternalServerError", "The request could not be completed.", request_id
        )

    register_routes(app, sidecar_db)
    return app


__all__ = ["create_app"]
