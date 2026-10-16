import asyncio

from .config import settings
from .exceptions import ConfigurationError, ExecutionError, SecurityError
from .logging_config import get_logger

logger = get_logger("execute")


async def run(sql: str) -> list[dict]:
    from .db import database

    if not sql or not sql.strip():
        raise SecurityError("Empty statement.")

    if settings.log_sql:
        logger.info("Executing: %s", sql)

    pool = await database.query_pool()

    async def _execute() -> list[dict]:
        async with pool.acquire() as conn:
            if settings.sandbox_transaction_read_only:
                async with conn.transaction(readonly=True):
                    records = await conn.fetch(sql)
            else:
                records = await conn.fetch(sql)

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
        return await asyncio.wait_for(_execute(), timeout=settings.sandbox_app_timeout_seconds)
    except TimeoutError as exc:
        raise ExecutionError(
            f"Query exceeded the {settings.sandbox_app_timeout_seconds}s execution budget."
        ) from exc
    except (ConfigurationError, SecurityError):
        raise
    except Exception as exc:
        raise ExecutionError(_safe_message(exc)) from exc


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
