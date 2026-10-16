import time
import traceback
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from . import logging_config
from .config import settings
from .constants import DATABASE_DOWN, DATABASE_UP, EMBEDDING_OFF, STATUS_DEGRADED, STATUS_OK
from .db import database
from .exceptions import SearchError
from .schema import HealthResponse, SearchRequest, SearchResponse
from .service import service
from .types import SchemaGraph
from .verification import verification_service

logging_config.configure()
error_logger = logging_config.get_logger("error")


@asynccontextmanager
async def lifespan(app: FastAPI):
    await verification_service.verify()
    await service.startup()
    yield
    await service.shutdown()


app = FastAPI(title=settings.app_name, version=settings.app_version, lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allow_origins,
    allow_credentials=settings.cors_allow_credentials,
    allow_methods=settings.cors_allow_methods,
    allow_headers=settings.cors_allow_headers,
)


@app.middleware("http")
async def trace(request: Request, call_next):
    request.state.request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
    started = time.perf_counter()
    response = await call_next(request)
    elapsed = (time.perf_counter() - started) * 1000
    response.headers["X-Request-ID"] = request.state.request_id
    response.headers["X-Process-Time-Ms"] = f"{elapsed:.{settings.elapsed_precision}f}"
    return response


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
async def health() -> HealthResponse:
    try:
        graph = await service.schema()
        connected = await database.ping()
        status_text = STATUS_OK if connected else STATUS_DEGRADED
        tables = len(graph.tables)
    except Exception:
        status_text, connected, tables = STATUS_DEGRADED, False, 0

    return HealthResponse(
        status=status_text,
        app=settings.app_name,
        database=DATABASE_UP if connected else DATABASE_DOWN,
        tables=tables,
        model=settings.llm_model,
        embedding=settings.embedding_model if settings.embedding_enabled else EMBEDDING_OFF,
    )


@app.get("/api/v1/schema", response_model=SchemaGraph)
async def schema(refresh: bool = Query(default=False)) -> SchemaGraph:
    if refresh:
        return await service.refresh()
    return await service.schema()


@app.post("/api/v1/search", response_model=SearchResponse)
async def search(payload: SearchRequest, request: Request) -> SearchResponse:
    # Failures are logged once, here, by the handlers above, which already carry
    # the request id and the exception type.
    return await service.search(payload.query, getattr(request.state, "request_id", None))
