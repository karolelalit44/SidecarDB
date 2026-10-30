import json
import secrets
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager

from fastapi import FastAPI, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.datastructures import Headers
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from core.catalog import SchemaGraph
from core.client import SidecarDB
from core.constants import (
    API_KEY_HEADER,
    APP_NAME,
    AUTHORIZATION_HEADER,
    BEARER_PREFIX,
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
    REQUEST_ID_MAX_LENGTH,
    RETRY_AFTER_HEADER,
    STATUS_OK,
)
from core.exceptions import SearchError
from core.wire import (
    HealthResponse,
    SearchRequest,
    SearchResponse,
)
from utils.logging_config import configure_logging, get_logger

logger = get_logger("api")
error_logger = get_logger("error")


class RateLimiter:
    def __init__(self, limit: int, window_seconds: float) -> None:
        self._limit = limit
        self._window = window_seconds
        self._hits: dict[str, deque[float]] = {}

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        cutoff = now - self._window
        window = self._hits.get(key)
        if window is not None:
            while window and window[0] <= cutoff:
                window.popleft()
            if not window:
                del self._hits[key]
                window = None

        if window is not None and len(window) >= self._limit:
            logger.debug(
                "rate_limiter: key=%s rejected (hits=%d >= limit=%d)",
                key,
                len(window),
                self._limit,
            )
            return False

        if window is None:
            window = deque()
            self._hits[key] = window
        window.append(now)
        logger.debug(
            "rate_limiter: key=%s accepted (hits=%d/%d in %.1fs)",
            key,
            len(window),
            self._limit,
            self._window,
        )
        return True


class _BodyTooLargeError(Exception):
    pass


class BodyCapMiddleware:
    def __init__(self, app: ASGIApp, *, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request_id = scope.get("state", {}).get("request_id")
        declared = Headers(scope=scope).get("content-length")
        logger.debug(
            "[%s] body cap check: declared content-length=%s (max=%d)",
            request_id,
            declared,
            self.max_bytes,
        )
        if declared and declared.isdigit() and int(declared) > self.max_bytes:
            logger.warning("[%s] request body too large: %s bytes", request_id, declared)
            await self._reject(send, request_id)
            return

        received = 0
        response_started = False

        async def guarded_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                chunk_len = len(message.get("body", b""))
                received += chunk_len
                logger.debug(
                    "[%s] received request body chunk: %d bytes (total=%d / max=%d)",
                    request_id,
                    chunk_len,
                    received,
                    self.max_bytes,
                )
                if received > self.max_bytes:
                    raise _BodyTooLargeError()
            return message

        async def guarded_send(message: Message) -> None:
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self.app(scope, guarded_receive, guarded_send)
        except _BodyTooLargeError:
            logger.warning(
                "[%s] streamed request body passed %d bytes before it ended",
                request_id,
                self.max_bytes,
            )
            if response_started:
                raise
            await self._reject(send, request_id)

    async def _reject(self, send: Send, request_id: str | None) -> None:
        payload = json.dumps(
            {
                "error": "PayloadTooLarge",
                "message": f"The request body exceeds {self.max_bytes} bytes.",
                "request_id": request_id,
            }
        ).encode()
        await send(
            {
                "type": "http.response.start",
                "status": 413,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(payload)).encode()),
                ],
            }
        )
        await send({"type": "http.response.body", "body": payload})


def _request_id(raw: str | None) -> str:
    if (
        raw
        and len(raw) <= REQUEST_ID_MAX_LENGTH
        and all(character.isalnum() or character in "-._" for character in raw)
    ):
        logger.debug("request_id: accepted client-provided id=%s", raw)
        return raw
    generated = uuid.uuid4().hex
    logger.debug(
        "request_id: generated new id=%s (raw was %s)",
        generated,
        "invalid" if raw else "none",
    )
    return generated


def _presented_key(request: Request) -> str:
    has_auth = AUTHORIZATION_HEADER.lower() in request.headers
    has_api_key = API_KEY_HEADER.lower() in request.headers
    logger.debug(
        "presented_key: checking headers (has_auth=%s, has_api_key=%s)",
        has_auth,
        has_api_key,
    )
    authorization = request.headers.get(AUTHORIZATION_HEADER.lower(), "")
    if authorization[: len(BEARER_PREFIX)].lower() == BEARER_PREFIX:
        return authorization[len(BEARER_PREFIX) :].strip()
    return request.headers.get(API_KEY_HEADER.lower(), "")


def _key_accepted(presented: str, accepted: list[str]) -> bool:
    logger.debug(
        "key_accepted: checking key (presented_len=%d, accepted_count=%d)",
        len(presented),
        len(accepted),
    )
    if not presented:
        return False
    result = any(secrets.compare_digest(presented, candidate) for candidate in accepted)
    logger.debug("key_accepted: match=%s", result)
    return result


def _error(
    status: int,
    kind: str,
    message: str,
    request_id: str | None,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    logger.debug(
        "[%s] _error: building response status=%d kind=%s message=%r",
        request_id,
        status,
        kind,
        message,
    )
    return JSONResponse(
        status_code=status,
        content={"error": kind, "message": message, "request_id": request_id},
        headers=headers,
    )


def create_app(sidecar_db: SidecarDB | None = None) -> FastAPI:
    sidecar_db = sidecar_db if sidecar_db is not None else SidecarDB()
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
        request.state.request_id = _request_id(raw_req_id)
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
            and not _key_accepted(_presented_key(request), settings.api_keys)
        ):
            logger.warning("[%s] rejected an unauthenticated request", request_id)
            return _error(401, "Unauthorized", "A valid API key is required.", request_id)
        if protected and settings.api_keys:
            logger.debug("[%s] request authentication succeeded", request_id)
        elif protected:
            logger.debug("[%s] protected route accessed with no API_KEYS configured", request_id)

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
        logger.debug(
            "[%s] unhandled exception: %s: %s",
            request_id,
            type(exc).__name__,
            exc,
        )
        return _error(500, "InternalServerError", "The request could not be completed.", request_id)

    @app.get("/health", response_model=HealthResponse)
    async def health() -> "HealthResponse | JSONResponse":
        logger.debug("health: check requested")
        result = await sidecar_db.health()
        logger.debug(
            "health: check complete: status=%s database=%s tables=%d model=%s embedding=%s",
            result.status,
            result.database,
            result.tables,
            result.model,
            result.embedding,
        )
        if result.status != STATUS_OK:
            return JSONResponse(status_code=503, content=result.model_dump())
        return result

    @app.get("/api/v1/schema", response_model=SchemaGraph)
    async def schema(refresh: bool = Query(default=False)) -> SchemaGraph:
        logger.debug("schema: requested with refresh=%s", refresh)
        graph = await sidecar_db.catalog(refresh=refresh)
        logger.debug(
            "schema: returning %d tables (%s) and %d columns",
            len(graph.tables),
            ", ".join(graph.tables.keys()),
            sum(len(info.column_names) for info in graph.tables.values()),
        )
        return graph

    @app.post("/api/v1/search", response_model=SearchResponse)
    async def search(payload: SearchRequest, request: Request) -> SearchResponse:
        request_id = getattr(request.state, "request_id", None)
        logger.debug(
            "[%s] /api/v1/search accepted question: %r (chars: %d)",
            request_id,
            payload.question,
            len(payload.question),
        )
        result = await sidecar_db.search(payload.question, request_id)
        logger.debug(
            "[%s] /api/v1/search answered: rows=%d elapsed=%s ms mode=%s",
            request_id,
            result.row_count,
            result.elapsed_ms,
            result.retrieval_mode.value,
        )
        return result

    return app
