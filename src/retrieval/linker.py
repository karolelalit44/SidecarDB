from catalog.models import SchemaGraph
from core.logging import get_logger
from planner.models import PlanPolicy
from retrieval.memory import RetrievalMode, SchemaMemory

logger = get_logger("retrieval.linker")


async def link(
    question: str,
    graph: SchemaGraph,
    index: SchemaMemory | None,
    policy: PlanPolicy | None = None,
) -> tuple[list[str], RetrievalMode, list[str]]:
    active_policy = policy if policy is not None else PlanPolicy()
    logger.debug(
        "link: schema linking initiated for question: %r (max_schema_tables=%d)",
        question,
        active_policy.max_schema_tables,
    )
    if index is not None and index.ready:
        tables = await index.tables_for(
            question, active_policy.max_schema_tables, active_policy.retrieval_min_score
        )
        if tables:
            logger.debug("link: resolved %d table(s) via embedding: %s", len(tables), tables)
            expanded = list(tables)
            for neighbor in graph.neighbors_of(tables):
                if neighbor not in expanded and len(expanded) < active_policy.max_schema_tables:
                    expanded.append(neighbor)
            if len(expanded) > len(tables):
                logger.debug(
                    "link: expanded %d table(s) to %d with 1-hop relational neighbors: %s",
                    len(tables),
                    len(expanded),
                    expanded,
                )
            return expanded, RetrievalMode.embedding, tables

    logger.debug("link: no embedding match at or above %s", active_policy.retrieval_min_score)
    return [], RetrievalMode.embedding, []


def dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


__all__ = ["dot", "link"]
