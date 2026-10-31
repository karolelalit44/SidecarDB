import asyncio
import time
from typing import Any

from catalog.introspect import read_postgres_catalog, read_sqlite_catalog
from catalog.models import SchemaGraph
from config.constants import (
    DB_BACKOFF_FACTOR,
    DB_CONNECT_BACKOFF_SECONDS,
    DB_CONNECT_RETRIES,
    DB_MAX_INACTIVE_CONNECTION_LIFETIME,
    ERROR_MESSAGE_MAX_CHARS,
    INTROSPECTION_TIMEOUT_SECONDS,
    LOG_SQL,
    MS_PER_SECOND,
    POOL_MAX_SIZE,
    POOL_MIN_SIZE,
    REDACT_CATALOG_IN_ERRORS,
    REDACTED_BY_SQLSTATE,
    REDACTED_BY_SQLSTATE_CLASS,
    SANDBOX_APP_TIMEOUT_SECONDS,
    SANDBOX_IDLE_TIMEOUT_MS,
    SANDBOX_MAX_ROWS,
    SANDBOX_STATEMENT_TIMEOUT_SECONDS,
    SANDBOX_TRANSACTION_READ_ONLY,
)
from config.settings import Settings
from core.exceptions import CatalogError, ConfigurationError, ExecutionError, SecurityError
from core.logging import get_logger
from guard.quoting import quote_literal
from planner.models import Statement

logger = get_logger("storage.connection")


class Database:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        url = settings.database_url or ""
        self._is_sqlite = url.startswith("sqlite")
        self._pool = None
        self._sqlite_conn: Any = None
        self._lock = asyncio.Lock()

    @property
    def connected(self) -> bool:
        if self._is_sqlite:
            return self._sqlite_conn is not None
        return self._pool is not None and not self._pool.is_closing()

    @property
    def pool(self):
        if self._pool is None:
            raise ConfigurationError("Database is not connected.")
        return self._pool

    async def connect(self, *, readonly: bool = True, retries: int | None = None) -> None:
        logger.debug("db: connect requested (readonly=%s, retries=%s)", readonly, retries)
        if self.connected:
            logger.debug("db: pool already open; reusing it")
            return

        async with self._lock:
            if self.connected:
                logger.debug("db: pool already open; reusing it")
                return

            if self._is_sqlite:
                import sqlite3

                raw_path = self._settings.database_url or ""
                path = raw_path.removeprefix("sqlite:///").removeprefix("sqlite://")
                target = ":memory:" if not path or path == ":memory:" else path
                logger.debug("db: connecting to sqlite at %s", target)
                self._sqlite_conn = sqlite3.connect(target, check_same_thread=False)
                self._sqlite_conn.row_factory = sqlite3.Row
                return

            attempts = DB_CONNECT_RETRIES if retries is None else retries
            delay = DB_CONNECT_BACKOFF_SECONDS
            last_error: Exception | None = None

            for attempt in range(attempts + 1):
                try:
                    logger.debug(
                        "db: connect attempt %d of %d starting...", attempt + 1, attempts + 1
                    )
                    await self._open_pool(readonly)
                    logger.debug("db: connect attempt %d succeeded", attempt + 1)
                    return
                except Exception as exc:
                    last_error = exc
                    if attempt >= attempts:
                        break
                    logger.warning(
                        "db: connect attempt %d of %d failed: %s: %s; retrying in %.1fs",
                        attempt + 1,
                        attempts + 1,
                        type(exc).__name__,
                        exc,
                        delay,
                    )
                    await asyncio.sleep(delay)
                    delay *= DB_BACKOFF_FACTOR

            logger.error(
                "db: the pool could not be opened: %s: %s", type(last_error).__name__, last_error
            )
            raise ConfigurationError(
                f"Database could not be reached ({last_error})."
            ) from last_error

    async def _open_pool(self, readonly: bool) -> None:
        import asyncpg

        settings = self._settings
        logger.debug(
            "db: opening pool (schema=%s, min=%d, max=%d, readonly=%s)",
            settings.db_schema,
            POOL_MIN_SIZE,
            POOL_MAX_SIZE,
            readonly,
        )

        async def _init(conn):
            statement_ms = int(SANDBOX_STATEMENT_TIMEOUT_SECONDS * MS_PER_SECOND)
            idle_ms = SANDBOX_IDLE_TIMEOUT_MS
            logger.debug(
                "db: initializing pool connection (readonly=%s, statement_timeout=%dms)",
                readonly,
                statement_ms,
            )
            if readonly:
                await conn.execute("SET default_transaction_read_only = on")
            await conn.execute(f"SET statement_timeout = '{statement_ms}ms'")
            await conn.execute(f"SET idle_in_transaction_session_timeout = '{idle_ms}ms'")
            if settings.db_schema:
                schema_name = quote_literal(settings.db_schema)
                await conn.execute(f"SET search_path = {schema_name}")

        dsn = settings.database_url
        ssl = settings.db_ssl or None
        self._pool = await asyncpg.create_pool(
            dsn,
            min_size=POOL_MIN_SIZE,
            max_size=POOL_MAX_SIZE,
            max_inactive_connection_lifetime=DB_MAX_INACTIVE_CONNECTION_LIFETIME,
            ssl=ssl,
            init=_init,
        )
        logger.debug("db: asyncpg pool created successfully")

    async def close(self) -> None:
        logger.debug("db: close requested")
        if self._is_sqlite:
            if self._sqlite_conn:
                self._sqlite_conn.close()
                self._sqlite_conn = None
            return

        if self._pool is not None:
            await self._pool.close()
            self._pool = None
            logger.debug("db: connection pool closed")

    async def ensure_connected(self) -> None:
        if not self.connected:
            await self.connect()

    async def ping(self) -> bool:
        if not self.connected:
            try:
                await self.connect(retries=0)
            except Exception as exc:
                logger.debug("db: ping connect failed: %s", exc)
                return False

        if self._is_sqlite:
            try:
                cursor = self._sqlite_conn.cursor()
                cursor.execute("SELECT 1")
                return cursor.fetchone() is not None
            except Exception as exc:
                logger.debug("db: sqlite ping error: %s", exc)
                return False

        try:
            async with self.pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
            return True
        except Exception as exc:
            logger.debug("db: ping error: %s", exc)
            return False

    async def introspect(self) -> SchemaGraph:
        await self.ensure_connected()
        started = time.perf_counter()

        if self._is_sqlite:
            assert self._sqlite_conn is not None
            graph = read_sqlite_catalog(self._sqlite_conn)
            logger.info(
                "Catalog introspected in %d ms (sqlite)",
                int((time.perf_counter() - started) * MS_PER_SECOND),
            )
            return graph

        logger.debug(
            "Introspecting pg_catalog for schema %r (timeout=%ss)",
            self._settings.db_schema,
            INTROSPECTION_TIMEOUT_SECONDS,
        )
        try:
            graph = await asyncio.wait_for(
                read_postgres_catalog(self.pool, self._settings.db_schema),
                timeout=INTROSPECTION_TIMEOUT_SECONDS,
            )
        except TimeoutError as exc:
            logger.error("Introspection timed out after %ss", INTROSPECTION_TIMEOUT_SECONDS)
            raise CatalogError("Introspection timed out.") from exc
        except Exception as exc:
            logger.error("Introspection failed: %s: %s", type(exc).__name__, exc)
            raise CatalogError(f"Introspection failed: {exc}") from exc

        logger.info(
            "Catalog introspected in %d ms: %d tables, %d foreign keys",
            int((time.perf_counter() - started) * MS_PER_SECOND),
            len(graph.tables),
            len(graph.foreign_keys),
        )
        return graph

    async def execute(self, statement: Statement) -> list[dict]:
        sql = statement.sql
        args = statement.params
        started = time.perf_counter()

        logger.debug(
            "execute: statement execution started (timeout=%.1fs, cap=%d, params=%d)",
            SANDBOX_APP_TIMEOUT_SECONDS,
            SANDBOX_MAX_ROWS,
            len(args),
        )
        if LOG_SQL:
            logger.info("Executing: %s", sql)
        else:
            logger.debug("Executing (LOG_SQL is off): %s", sql)

        await self.ensure_connected()

        if self._is_sqlite:
            assert self._sqlite_conn is not None
            import sqlglot

            transpiled = sqlglot.transpile(sql, read="postgres", write="sqlite")[0]
            for i in range(len(args), 0, -1):
                transpiled = transpiled.replace(f"@{i}", "?")
            cursor = self._sqlite_conn.cursor()
            cursor.execute(transpiled, list(args))
            records = cursor.fetchall()
            rows = [dict(record) for record in records]
            if len(rows) > SANDBOX_MAX_ROWS:
                logger.warning(
                    "Truncating %d rows to the sandbox cap of %d",
                    len(rows),
                    SANDBOX_MAX_ROWS,
                )
                rows = rows[:SANDBOX_MAX_ROWS]
            logger.debug(
                "Execution returned %d rows in %d ms (sqlite)",
                len(rows),
                int((time.perf_counter() - started) * MS_PER_SECOND),
            )
            return rows

        pool = self.pool

        async def _run() -> list[dict]:
            async with pool.acquire() as conn:
                logger.debug("execute: connection acquired from pool, executing query...")
                if SANDBOX_TRANSACTION_READ_ONLY:
                    async with conn.transaction(readonly=True):
                        records = await conn.fetch(sql, *args)
                else:
                    records = await conn.fetch(sql, *args)

            logger.debug("execute: fetched %d raw record(s) from database", len(records))
            if records:
                logger.debug("execute: column keys in first row: %s", list(records[0].keys()))

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
            rows = await asyncio.wait_for(_run(), timeout=SANDBOX_APP_TIMEOUT_SECONDS)
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
        logger.debug(
            "_safe_message: formatting error message (redact=%s, exc=%s)",
            REDACT_CATALOG_IN_ERRORS,
            type(exc).__name__,
        )
        if REDACT_CATALOG_IN_ERRORS:
            return f"Statement rejected: {self._classified_message(exc)}"
        detail = str(exc).split("\n", 1)[0].strip() or "rejected by the database"
        return f"Statement rejected: {detail[:ERROR_MESSAGE_MAX_CHARS]}"

    def _classified_message(self, exc: Exception) -> str:
        sqlstate = getattr(exc, "sqlstate", None)
        logger.debug("_classified_message: evaluating sqlstate=%s", sqlstate)
        if isinstance(sqlstate, str):
            content = REDACTED_BY_SQLSTATE.get(sqlstate) or REDACTED_BY_SQLSTATE_CLASS.get(
                sqlstate[:2]
            )
            if content:
                logger.debug("_classified_message: matched classified text: %s", content)
                return content
        return "the database refused the statement"


__all__ = ["Database"]
