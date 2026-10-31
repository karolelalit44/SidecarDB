from catalog.models import SchemaGraph
from core.logging import get_logger
from core.protocols import DatabaseGateway, Embedder
from planner.models import PlanPolicy
from retrieval.memory import SchemaMemory

logger = get_logger("core.catalog")


class CatalogManager:
    def __init__(
        self,
        database: DatabaseGateway,
        embedder: Embedder,
        policy: PlanPolicy,
        storage_dir: str,
    ) -> None:
        self._database = database
        self._embedder = embedder
        self._policy = policy
        self._storage_dir = storage_dir

    async def refresh(self) -> tuple[SchemaGraph, SchemaMemory]:
        logger.debug("refresh: introspecting the catalog")
        graph = await self._database.introspect()
        logger.debug(
            "refresh: %d tables introspected; rebuilding schema memory",
            len(graph.tables),
        )
        index = await SchemaMemory(self._embedder, self._policy).build(
            graph, storage_dir=self._storage_dir
        )
        logger.info("Catalog mapped: %d tables", len(graph.tables))
        return graph, index

    def restore(self) -> tuple[SchemaGraph, SchemaMemory] | None:
        return SchemaMemory.load_persisted(self._storage_dir, self._embedder, self._policy)


__all__ = ["CatalogManager"]
