import asyncio
import time
from typing import TYPE_CHECKING

from core.constants import (
    ERROR_MESSAGE_MAX_CHARS,
    LOG_SQL,
    MS_PER_SECOND,
    REDACT_CATALOG_IN_ERRORS,
    SANDBOX_APP_TIMEOUT_SECONDS,
    SANDBOX_MAX_ROWS,
    SANDBOX_STATEMENT_TIMEOUT_SECONDS,
    SANDBOX_TRANSACTION_READ_ONLY,
)
from core.exceptions import ConfigurationError, ExecutionError, SecurityError
from core.guard import validate
from core.plan import Statement
from utils.logging_config import get_logger

if TYPE_CHECKING:
    from infra.db import Database

logger = get_logger("execute")

REDACTED_BY_SQLSTATE = {
    "42501": "the database denied the statement access to an object",
    "57014": "the statement exceeded the database's execution budget",
}
REDACTED_BY_SQLSTATE_CLASS = {
    "42": "the statement named an object the database does not provide",
}


class Sandbox:
    def __init__(self, database: "Database") -> None:
        self._database = database

    async def run(self, statement: Statement) -> list[dict]:
        sql = validate(statement.sql, None, "").sql
        args = statement.params

        started = time.perf_counter()
        logger.debug(
            "execute: sandbox readonly=%s statement_timeout=%ss app_timeout=%ss "
            "max_rows=%d bound_args=%d",
            SANDBOX_TRANSACTION_READ_ONLY,
            SANDBOX_STATEMENT_TIMEOUT_SECONDS,
            SANDBOX_APP_TIMEOUT_SECONDS,
            SANDBOX_MAX_ROWS,
            len(args),
        )
        if LOG_SQL:
            logger.info("Executing: %s", sql)
        else:
            logger.debug("Executing (LOG_SQL is off): %s", sql)

        await self._database.ensure_connected()
        pool = self._database.pool

        async def _execute() -> list[dict]:
            async with pool.acquire() as conn:
                if SANDBOX_TRANSACTION_READ_ONLY:
                    async with conn.transaction(readonly=True):
                        records = await conn.fetch(sql, *args)
                else:
                    records = await conn.fetch(sql, *args)

            rows = [dict(record) for record in records]
            if len(rows) > SANDBOX_MAX_ROWS:
                logger.debug(
                    "execute: %d rows fetched, capped at %d",
                    len(rows),
                    SANDBOX_MAX_ROWS,
                )
                logger.warning(
                    "Truncating %d rows to the sandbox cap of %d",
                    len(rows),
                    SANDBOX_MAX_ROWS,
                )
                return rows[:SANDBOX_MAX_ROWS]
            return rows

        try:
            rows = await asyncio.wait_for(_execute(), timeout=SANDBOX_APP_TIMEOUT_SECONDS)
        except TimeoutError as exc:
            if LOG_SQL:
                logger.error(
                    "Execution exceeded the %ss budget: %s",
                    SANDBOX_APP_TIMEOUT_SECONDS,
                    sql,
                )
            else:
                logger.error(
                    "Execution exceeded the %ss budget (LOG_SQL is off; SQL withheld)",
                    SANDBOX_APP_TIMEOUT_SECONDS,
                )
            raise ExecutionError(
                f"Statement exceeded the {SANDBOX_APP_TIMEOUT_SECONDS}s execution budget."
            ) from exc
        except (ConfigurationError, SecurityError):
            raise
        except Exception as exc:
            logger.error(
                "Execution failed in %d ms: %s: %s",
                int((time.perf_counter() - started) * MS_PER_SECOND),
                type(exc).__name__,
                exc,
            )
            raise ExecutionError(self._safe_message(exc)) from exc

        logger.debug(
            "Execution returned %d rows in %d ms",
            len(rows),
            int((time.perf_counter() - started) * MS_PER_SECOND),
        )
        return rows

    def _safe_message(self, exc: Exception) -> str:
        if REDACT_CATALOG_IN_ERRORS:
            return f"Statement rejected: {self._classified_message(exc)}"
        detail = str(exc).split("\n", 1)[0].strip() or "rejected by the database"
        return f"Statement rejected: {detail[:ERROR_MESSAGE_MAX_CHARS]}"

    def _classified_message(self, exc: Exception) -> str:
        sqlstate = getattr(exc, "sqlstate", None)
        if isinstance(sqlstate, str):
            content = REDACTED_BY_SQLSTATE.get(sqlstate) or REDACTED_BY_SQLSTATE_CLASS.get(
                sqlstate[:2]
            )
            if content:
                return content
        return "the database refused the statement"
