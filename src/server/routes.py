from typing import TYPE_CHECKING

from fastapi import FastAPI, Query, Request
from fastapi.responses import JSONResponse

from catalog.models import SchemaGraph
from config.constants import STATUS_OK
from core.logging import get_logger
from server.schemas import (
    HealthResponse,
    SearchRequest,
    SearchResponse,
)

if TYPE_CHECKING:
    from core.client import SidecarDB

logger = get_logger("server.routes")


def register_routes(app: FastAPI, sidecar_db: "SidecarDB") -> None:
    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse | JSONResponse:
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


__all__ = ["register_routes"]
