import asyncio
import time

from core.config import Settings
from core.constants import BACKOFF_FACTOR, MS_PER_SECOND
from core.exceptions import ConfigurationError, SchemaError
from core.logging_config import get_logger
from core.types import SchemaGraph, TableInfo

logger = get_logger("db")

_COLUMNS_SQL = """
SELECT c.relname AS table_name,
       a.attname AS name,
       format_type(a.atttypid, a.atttypmod) AS data_type,
       base.typcategory AS type_category,
       NOT a.attnotnull AS nullable,
       col_description(c.oid, a.attnum) AS description
FROM pg_attribute a
JOIN pg_class c ON c.oid = a.attrelid
JOIN pg_namespace n ON n.oid = c.relnamespace
JOIN LATERAL (
    SELECT bt.typcategory
    FROM pg_type rt
    JOIN pg_type bt ON bt.oid = COALESCE(NULLIF(rt.typbasetype, 0), rt.oid)
    WHERE rt.oid = a.atttypid
) base ON true
WHERE n.nspname = $1
  AND c.relkind = 'r'
  AND a.attnum > 0
  AND NOT a.attisdropped
ORDER BY c.relname, a.attnum
"""

_TABLES_SQL = """
SELECT c.relname AS name,
       obj_description(c.oid, 'pg_class') AS description,
       c.reltuples AS row_estimate
FROM pg_class c
JOIN pg_namespace n ON n.oid = c.relnamespace
WHERE n.nspname = $1 AND c.relkind = 'r'
ORDER BY c.relname
"""


class Database:
    """A pooled, sandboxed connection set to one PostgreSQL schema."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._pool = None

    @property
    def connected(self) -> bool:
        """Whether a live pool is held."""
        return self._pool is not None and not self._pool.is_closing()

    @property
    def pool(self):
        if self._pool is None:
            raise ConfigurationError("Database is not connected.")
        return self._pool

    async def connect(self, *, readonly: bool = True, retries: int | None = None) -> None:
        """Open the pool, retrying with exponential backoff.

        readonly=False is for maintenance scripts that write to the database.
        """
        if self.connected:
            logger.debug("db: pool already open; reusing it")
            return

        attempts = self._settings.db_connect_retries if retries is None else retries
        delay = self._settings.db_connect_backoff_seconds
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
                delay *= BACKOFF_FACTOR

        logger.error(
            "db: the pool could not be opened: %s: %s", type(last_error).__name__, last_error
        )
        raise ConfigurationError(f"Database could not be reached ({last_error}).") from last_error

    async def _open_pool(self, readonly: bool) -> None:
        import asyncpg

        settings = self._settings

        async def _init(conn):
            statement_ms = int(settings.sandbox_statement_timeout_seconds * MS_PER_SECOND)
            idle_ms = settings.sandbox_idle_timeout_ms
            if readonly:
                await conn.execute("SET SESSION CHARACTERISTICS AS TRANSACTION READ ONLY")
            await conn.execute(f"SET statement_timeout = '{statement_ms}ms'")
            await conn.execute(f"SET idle_in_transaction_session_timeout = '{idle_ms}ms'")
            await conn.execute("SET search_path = " + _quote_literal(settings.db_schema))

        options: dict[str, object] = {
            "dsn": settings.database_url,
            "min_size": settings.pool_min_size,
            "max_size": settings.pool_max_size,
            "command_timeout": settings.sandbox_statement_timeout_seconds,
            "max_inactive_connection_lifetime": settings.db_max_inactive_connection_lifetime,
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
            settings.pool_min_size,
            settings.pool_max_size,
            readonly,
            settings.sandbox_statement_timeout_seconds,
            settings.sandbox_idle_timeout_ms,
            settings.db_schema,
            settings.db_ssl or "default",
        )

    async def ensure_connected(self) -> None:
        """Reopen the pool if it was never opened or has gone away."""
        if self.connected:
            return
        logger.debug("db: no live pool; reconnecting on demand")
        await self.connect()

    async def close(self) -> None:
        if self._pool is not None:
            logger.debug("db: closing the connection pool")
            await self._pool.close()
        self._pool = None

    async def introspect(self) -> SchemaGraph:
        _ = self.pool  # fail fast with ConfigurationError when not connected
        started = time.perf_counter()
        settings = self._settings
        logger.debug(
            "db: introspecting schema %s with a %ss budget",
            settings.db_schema,
            settings.introspection_timeout_seconds,
        )
        try:
            graph = await asyncio.wait_for(
                self._read_catalog(), timeout=settings.introspection_timeout_seconds
            )
        except TimeoutError as exc:
            raise SchemaError(
                f"Catalog introspection exceeded {settings.introspection_timeout_seconds}s."
            ) from exc
        except SchemaError:
            raise
        except Exception as exc:
            raise SchemaError(f"Catalog introspection failed: {exc}") from exc
        logger.debug(
            "db: introspected %d tables and %d columns in %d ms",
            len(graph.tables),
            sum(len(info.column_names) for info in graph.tables.values()),
            int((time.perf_counter() - started) * MS_PER_SECOND),
        )
        return graph

    async def _read_catalog(self) -> SchemaGraph:
        async with self.pool.acquire() as conn:
            tables = await conn.fetch(_TABLES_SQL, self._settings.db_schema)
            logger.debug("db: catalog read %d tables", len(tables))
            columns = await conn.fetch(_COLUMNS_SQL, self._settings.db_schema)
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

        graph: dict[str, TableInfo] = {}
        for row in tables:
            name = row["name"]
            graph[name] = TableInfo(
                name=name,
                description=row["description"],
                columns=columns_by_table.get(name, []),
                row_estimate=max(int(row["row_estimate"] or 0), 0),
            )

        return SchemaGraph(tables=graph)

    async def ping(self) -> bool:
        """Ask the server for a trivial value; the only honest liveness check."""
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


def _quote_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"
