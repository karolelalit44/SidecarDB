from typing import Any

from pydantic import ValidationError

from catalog.models import SchemaGraph
from config.constants import PLANNING_FAILED_MESSAGE
from core.exceptions import ConfigurationError, PlanningError, ProviderError
from core.logging import get_logger
from planner.models import Plan, PlanPolicy
from planner.prompts import REPAIR, SYSTEM, PlanRequest, build_prompt
from planner.validation import check_plan, normalise_plan

logger = get_logger("planner.generator")


async def make_plan(
    request: PlanRequest,
    llm: Any,
    error: str | None = None,
    attempt: int = 0,
) -> tuple[Plan | None, str | None]:
    policy = request.policy
    prompt = build_prompt(request)
    if error:
        prompt = f"{REPAIR.format(error=error[: policy.repair_char_budget])}\n\n{prompt}"

    temperature = policy.llm_temperature if attempt == 0 else policy.llm_repair_temperature

    logger.debug(
        "Planning attempt %d: tables=%d prompt_chars=%d temperature=%s repairing=%s",
        attempt + 1,
        len(request.tables),
        len(prompt),
        temperature,
        error,
    )

    try:
        payload = await llm.complete_json(SYSTEM, prompt, temperature)
        logger.debug("Plan payload: %s", payload)
        result = normalise_plan(Plan.model_validate(payload), request)
    except (ConfigurationError, ProviderError):
        raise
    except (PlanningError, ValidationError) as exc:
        logger.debug("Plan malformed: %s: %s", type(exc).__name__, exc)
        return None, " ".join(str(exc).split())[: policy.error_message_max_chars]
    except Exception as exc:
        logger.warning("Planner call failed: %s: %s", type(exc).__name__, exc)
        return None, "a usable plan could not be produced"

    reason = check_plan(result, request)
    if reason:
        logger.debug(
            "Plan rejected: %s (intent=%s table=%s)",
            reason,
            result.intent.value,
            result.table,
        )
        return None, reason

    logger.debug(
        "Plan accepted: intent=%s table=%s measure=%s/%s filter=%s %s %r projection=%d order_by=%s",
        result.intent.value,
        result.table,
        result.measure_function.value if result.measure_function else None,
        result.measure_column,
        result.filter_column,
        result.filter_operator,
        result.filter_value,
        len(result.projection),
        result.order_by,
    )
    return result, None


async def generate_plan(
    question: str,
    graph: SchemaGraph,
    tables: list[str],
    planner: Any,
    policy: PlanPolicy | None = None,
) -> Plan:
    active_policy = policy or PlanPolicy()
    request = PlanRequest(
        question=question,
        graph=graph,
        policy=active_policy,
        tables=tables,
    )
    logger.debug(
        "planning pipeline: starting planning loop (max_attempts=%d)",
        active_policy.max_plan_attempts,
    )
    error: str | None = None
    for attempt in range(active_policy.max_plan_attempts):
        plan_result, error = await make_plan(request, planner, error, attempt)
        if plan_result is not None:
            logger.debug(
                "planning pipeline: plan produced on attempt %d of %d",
                attempt + 1,
                active_policy.max_plan_attempts,
            )
            return plan_result
        logger.debug(
            "plan: attempt %d of %d failed: %s",
            attempt + 1,
            active_policy.max_plan_attempts,
            error,
        )

    logger.error("No plan survived %d attempts: %s", active_policy.max_plan_attempts, error)
    raise PlanningError(PLANNING_FAILED_MESSAGE)


__all__ = ["generate_plan", "make_plan"]
