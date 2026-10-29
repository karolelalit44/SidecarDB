import asyncio
import time

from core.catalog import SchemaGraph, TableInfo
from core.config import Settings
from core.constants import (
    COLUMNS_SQL,
    DB_BACKOFF_FACTOR,
    DB_CONNECT_BACKOFF_SECONDS,
    DB_CONNECT_RETRIES,
    DB_MAX_INACTIVE_CONNECTION_LIFETIME,
    INTROSPECTION_TIMEOUT_SECONDS,
    MS_PER_SECOND,
    POOL_MAX_SIZE,
    POOL_MIN_SIZE,
    SANDBOX_IDLE_TIMEOUT_MS,
    SANDBOX_STATEMENT_TIMEOUT_SECONDS,
    TABLES_SQL,
)
from core.exceptions import CatalogError, ConfigurationError
from utils.logging_config import get_logger
from utils.sql import quote_literal

logger = get_logger("db")


class Database:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._pool = None
        self._lock = asyncio.Lock()

    @property
    def connected(self) -> bool:
        return self._pool is not None and not self._pool.is_closing()

    @property
    def pool(self):
        if self._pool is None:
            raise ConfigurationError("Database is not connected.")
        return self._pool

    async def connect(self, *, readonly: bool = True, retries: int | None = None) -> None:
        if self.connected:
            logger.debug("db: pool already open; reusing it")
            return

        async with self._lock:
            if self.connected:
                logger.debug("db: pool already open; reusing it")
                return

            attempts = DB_CONNECT_RETRIES if retries is None else retries
            delay = DB_CONNECT_BACKOFF_SECONDS
            last_error: Exception | None = None

            for attempt in range(attempts + 1):
                try:
                    await self._open_pool(readonly)
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

        async def _init(conn):
            statement_ms = int(SANDBOX_STATEMENT_TIMEOUT_SECONDS * MS_PER_SECOND)
            idle_ms = SANDBOX_IDLE_TIMEOUT_MS
            if readonly:
                await conn.execute("SET SESSION CHARACTERISTICS AS TRANSACTION READ ONLY")
            await conn.execute(f"SET statement_timeout = '{statement_ms}ms'")
            await conn.execute(f"SET idle_in_transaction_session_timeout = '{idle_ms}ms'")
            await conn.execute("SET search_path = " + quote_literal(settings.db_schema))

        options: dict[str, object] = {
            "dsn": settings.database_url,
            "min_size": POOL_MIN_SIZE,
            "max_size": POOL_MAX_SIZE,
            "command_timeout": SANDBOX_STATEMENT_TIMEOUT_SECONDS,
            "max_inactive_connection_lifetime": DB_MAX_INACTIVE_CONNECTION_LIFETIME,
            "init": _init,
        }
        if settings.db_ssl:
            options["ssl"] = settings.db_ssl

        started = time.perf_counter()
        self._pool = await asyncpg.create_pool(**options)
        logger.debug(
            "db: pool opened in %d ms (min=%d max=%d readonly=%s "
            "statement_timeout=%ss idle_timeout=%d ms schema=%s ssl=%s)",
            int((time.perf_counter() - started) * MS_PER_SECOND),
            POOL_MIN_SIZE,
            POOL_MAX_SIZE,
            readonly,
            SANDBOX_STATEMENT_TIMEOUT_SECONDS,
            SANDBOX_IDLE_TIMEOUT_MS,
            settings.db_schema,
            settings.db_ssl or "default",
        )

    async def ensure_connected(self) -> None:
        if self.connected:
            return
        logger.debug("db: no live pool; reconnecting on demand")
        await self.connect()

    async def close(self) -> None:
        async with self._lock:
            if self._pool is not None:
                logger.debug("db: closing the connection pool")
                await self._pool.close()
                self._pool = None

    async def introspect(self) -> SchemaGraph:
        await self.ensure_connected()
        started = time.perf_counter()
        settings = self._settings
        logger.debug(
            "db: introspecting schema %s with a %ss budget",
            settings.db_schema,
            INTROSPECTION_TIMEOUT_SECONDS,
        )
        try:
            graph = await asyncio.wait_for(
                self._read_catalog(), timeout=INTROSPECTION_TIMEOUT_SECONDS
            )
        except TimeoutError as exc:
            raise CatalogError(
                f"Catalog introspection exceeded {INTROSPECTION_TIMEOUT_SECONDS}s."
            ) from exc
        except CatalogError:
            raise
        except Exception as exc:
            raise CatalogError(f"Catalog introspection failed: {exc}") from exc
        logger.debug(
            "db: introspected %d tables and %d columns in %d ms",
            len(graph.tables),
            sum(len(info.column_names) for info in graph.tables.values()),
            int((time.perf_counter() - started) * MS_PER_SECOND),
        )
        return graph

    async def _read_catalog(self) -> SchemaGraph:
        async with self.pool.acquire() as conn:
            tables = await conn.fetch(TABLES_SQL, self._settings.db_schema)
            logger.debug("db: catalog read %d tables", len(tables))
            columns = await conn.fetch(COLUMNS_SQL, self._settings.db_schema)
            logger.debug("db: catalog read %d columns", len(columns))

        columns_by_table: dict[str, list] = {}
        for row in columns:
            table_name = row["table_name"]
            columns_by_table.setdefault(table_name, []).append(
                {
                    "name": row["name"],
                    "data_type": row["data_type"],
                    "type_category": row["type_category"],
                    "nullable": row["nullable"],
                    "description": row["description"],
                }
            )

        tables_by_name: dict[str, TableInfo] = {}
        for row in tables:
            name = row["name"]
            tables_by_name[name] = TableInfo(
                name=name,
                description=row["description"],
                columns=columns_by_table.get(name, []),
                row_estimate=max(int(row["row_estimate"] or 0), 0),
            )

        return SchemaGraph(tables=tables_by_name)

    async def ping(self) -> bool:
        if not self.connected:
            logger.debug("db: ping without an open pool reports down")
            return False
        try:
            async with self.pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
            logger.debug("db: ping ok")
            return True
        except Exception as exc:
            logger.debug("db: ping failed: %s: %s", type(exc).__name__, exc)
            return False
