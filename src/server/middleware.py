import json
import secrets
import time
import uuid
from collections import deque

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.datastructures import Headers
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from config.constants import (
    API_KEY_HEADER,
    AUTHORIZATION_HEADER,
    BEARER_PREFIX,
    REQUEST_ID_MAX_LENGTH,
)
from core.logging import get_logger

logger = get_logger("server.middleware")


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


def resolve_request_id(raw: str | None) -> str:
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


def presented_api_key(request: Request) -> str:
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


def api_key_accepted(presented: str, accepted: list[str]) -> bool:
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


def error_response(
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


__all__ = [
    "BodyCapMiddleware",
    "RateLimiter",
    "api_key_accepted",
    "error_response",
    "presented_api_key",
    "resolve_request_id",
]
