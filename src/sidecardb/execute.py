import asyncio
import time
from typing import TYPE_CHECKING

from .constants import MS_PER_SECOND
from .exceptions import ConfigurationError, ExecutionError, SecurityError
from .logging_config import get_logger
from .render import Rendered

if TYPE_CHECKING:
    from .config import Settings
    from .db import Database
    from .types import SchemaGraph

logger = get_logger("execute")


class Executor:
    """Validates a rendered statement, then runs it inside the sandbox."""

    def __init__(self, database: "Database", settings: "Settings") -> None:
        self._database = database
        self._settings = settings

    def prepare(self, rendered: Rendered, graph: "SchemaGraph") -> str:
        from .guard import validate

        return validate(rendered.sql, graph, self._settings)

    async def run(self, sql: str, *args) -> list[dict]:
        """Run one statement with its bound parameters."""
        settings = self._settings
        if not sql or not sql.strip():
            raise SecurityError("Empty statement.")

        started = time.perf_counter()
        logger.debug(
            "execute: sandbox readonly=%s statement_timeout=%ss app_timeout=%ss "
            "max_rows=%d bound_args=%d",
            settings.sandbox_transaction_read_only,
            settings.sandbox_statement_timeout_seconds,
            settings.sandbox_app_timeout_seconds,
            settings.sandbox_max_rows,
            len(args),
        )
        if settings.log_sql:
            logger.info("Executing: %s", sql)
        else:
            logger.debug("Executing (LOG_SQL is off): %s", sql)

        await self._database.ensure_connected()
        pool = self._database.pool

        async def _execute() -> list[dict]:
            async with pool.acquire() as conn:
                if settings.sandbox_transaction_read_only:
                    async with conn.transaction(readonly=True):
                        records = await conn.fetch(sql, *args)
                else:
                    records = await conn.fetch(sql, *args)

            rows = [dict(record) for record in records]
            if len(rows) > settings.sandbox_max_rows:
                logger.debug(
                    "execute: %d rows fetched, capped at %d",
                    len(rows),
                    settings.sandbox_max_rows,
                )
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
        detail = str(exc).split("\n", 1)[0].strip()
        if self._settings.redact_catalog_in_errors:
            detail = detail.split("(")[0].strip()
        if not detail:
            detail = "rejected by the database"
        return f"Query rejected: {detail[: self._settings.error_message_max_chars]}"
