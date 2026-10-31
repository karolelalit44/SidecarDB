from catalog.models import SchemaGraph
from core.logging import get_logger
from core.protocols import DatabaseGateway, Planner
from planner.compiler import statement_for
from planner.generator import generate_plan
from planner.models import Outcome, PlanPolicy
from retrieval.linker import link
from retrieval.memory import SchemaMemory

logger = get_logger("core.service")


async def run_search(
    question: str,
    graph: SchemaGraph,
    index: SchemaMemory | None,
    planner: Planner,
    database: DatabaseGateway,
    policy: PlanPolicy,
    schema: str,
) -> Outcome:
    logger.debug("pipeline: beginning execution for question: %r", question)
    logger.debug("pipeline: step 1 - schema linking (retrieval)")
    tables, retrieval_mode, retrieved = await link(question, graph, index, policy)
    logger.debug(
        "pipeline: step 1 complete - mode=%s retrieved_tables=%s final_tables=%s",
        retrieval_mode.value,
        retrieved,
        tables,
    )

    logger.debug("pipeline: step 2 - generating query plan with %d table candidate(s)", len(tables))
    plan = await generate_plan(
        question=question,
        graph=graph,
        tables=tables,
        planner=planner,
        policy=policy,
    )
    logger.debug(
        "pipeline: step 2 complete - plan accepted: intent=%s table=%s projection=%d filter=%s %s",
        plan.intent.value,
        plan.table,
        len(plan.projection),
        plan.filter_column,
        plan.filter_operator,
    )

    logger.debug("pipeline: step 3 - compiling plan to guarded SQL statement")
    statement = statement_for(plan, graph, schema)
    logger.debug(
        "pipeline: step 3 complete - statement composed: sql=%s, params_count=%d",
        statement.sql,
        len(statement.params),
    )

    logger.debug("pipeline: step 4 - executing statement via database gateway")
    rows = await database.execute(statement)
    logger.debug("pipeline: step 4 complete - fetched %d row(s)", len(rows))

    return Outcome(
        plan=plan,
        statement=statement,
        rows=rows,
        retrieved_tables=retrieved,
        retrieval_mode=retrieval_mode,
    )


__all__ = ["run_search"]
