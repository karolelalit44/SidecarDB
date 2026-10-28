import asyncio
import time

from .config import settings
from .exceptions import ConfigurationError, ExecutionError, SecurityError
from .logging_config import get_logger

logger = get_logger("execute")


async def run(sql: str, *args) -> list[dict]:
    """Run one statement with bound parameters."""
    from .db import database

    if not sql or not sql.strip():
        raise SecurityError("Empty statement.")

    started = time.perf_counter()
    if settings.log_sql:
        logger.info("Executing: %s", sql)
    else:
        logger.debug("Executing (LOG_SQL is off): %s", sql)

    pool = await database.query_pool()

    async def _execute() -> list[dict]:
        async with pool.acquire() as conn:
            if settings.sandbox_transaction_read_only:
                async with conn.transaction(readonly=True):
                    records = await conn.fetch(sql, *args)
            else:
                records = await conn.fetch(sql, *args)

        rows = [dict(record) for record in records]
        if len(rows) > settings.sandbox_max_rows:
            logger.warning(
                "Truncating %d rows to the sandbox cap of %d",
                len(rows),
                settings.sandbox_max_rows,
            )
            return rows[: settings.sandbox_max_rows]
        return rows

    try:
        rows = await asyncio.wait_for(_execute(), timeout=settings.sandbox_app_timeout_seconds)
    except TimeoutError as exc:
        logger.error(
            "Execution exceeded the %ss budget: %s",
            settings.sandbox_app_timeout_seconds,
            sql,
        )
        raise ExecutionError(
            f"Query exceeded the {settings.sandbox_app_timeout_seconds}s execution budget."
        ) from exc
    except (ConfigurationError, SecurityError):
        raise
    except Exception as exc:
        logger.error(
            "Execution failed in %d ms: %s: %s",
            int((time.perf_counter() - started) * 1000),
            type(exc).__name__,
            exc,
        )
        raise ExecutionError(_safe_message(exc)) from exc

    logger.debug(
        "Execution returned %d rows in %d ms",
        len(rows),
        int((time.perf_counter() - started) * 1000),
    )
    return rows


def _safe_message(exc: Exception) -> str:
    detail = str(exc).split("\n", 1)[0].strip()
    if settings.redact_catalog_in_errors:
        detail = detail.split("(")[0].strip()
    if not detail:
        detail = "rejected by the database"
    return f"Query rejected: {detail[: settings.error_message_max_chars]}"


def prepare(sql: str, graph) -> str:
    from .guard import validate

    return validate(sql, graph)
