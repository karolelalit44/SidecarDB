import asyncio
import time
from typing import Any

from core.catalog import ColumnInfo, ForeignKeyInfo, SchemaGraph, TableInfo
from core.config import Settings
from core.constants import (
    COLUMNS_SQL,
    CONSTRAINTS_SQL,
    DB_BACKOFF_FACTOR,
    DB_CONNECT_BACKOFF_SECONDS,
    DB_CONNECT_RETRIES,
    DB_MAX_INACTIVE_CONNECTION_LIFETIME,
    ENUMS_SQL,
    ERROR_MESSAGE_MAX_CHARS,
    FOREIGN_KEYS_SQL,
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
    TABLES_SQL,
)
from core.exceptions import CatalogError, ConfigurationError, ExecutionError, SecurityError
from core.guard import quote_literal
from core.plan import Statement
from utils.logging_config import get_logger

logger = get_logger("db")


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
            if self._is_sqlite:
                if self._sqlite_conn is not None:
                    logger.debug("db: closing sqlite connection")
                    self._sqlite_conn.close()
                    self._sqlite_conn = None
                return
            if self._pool is not None:
                logger.debug("db: closing the connection pool")
                await self._pool.close()
                self._pool = None

    async def introspect(self) -> SchemaGraph:
        await self.ensure_connected()
        if self._is_sqlite:
            return await self._read_sqlite_catalog()
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

    async def _read_sqlite_catalog(self) -> SchemaGraph:
        assert self._sqlite_conn is not None
        cursor = self._sqlite_conn.cursor()
        tables_query = (
            "SELECT name FROM sqlite_master "
            "WHERE type IN ('table', 'view') AND name NOT LIKE 'sqlite_%'"
        )
        tables_raw = cursor.execute(tables_query).fetchall()
        table_map: dict[str, TableInfo] = {}
        all_fks: list[ForeignKeyInfo] = []
        for r in tables_raw:
            t_name = str(r[0])
            col_rows = cursor.execute(f"PRAGMA table_info('{t_name}')").fetchall()
            cols = [
                ColumnInfo(
                    name=str(c["name"]),
                    data_type=str(c["type"] or "text"),
                    is_primary_key=bool(c["pk"]),
                    nullable=not bool(c["notnull"]),
                )
                for c in col_rows
            ]
            fk_rows = cursor.execute(f"PRAGMA foreign_key_list('{t_name}')").fetchall()
            fks = [
                ForeignKeyInfo(
                    source_table=t_name,
                    source_column=str(f["from"]),
                    target_table=str(f["table"]),
                    target_column=str(f["to"]),
                )
                for f in fk_rows
            ]
            all_fks.extend(fks)
            table_map[t_name] = TableInfo(
                name=t_name,
                columns=cols,
                foreign_keys=fks,
                primary_keys=[str(c["name"]) for c in col_rows if c["pk"]],
            )
        logger.debug(
            "db: sqlite catalog read %d tables, %d foreign keys",
            len(table_map),
            len(all_fks),
        )
        return SchemaGraph(tables=table_map, foreign_keys=all_fks)

    async def _read_catalog(self) -> SchemaGraph:
        logger.debug(
            "db: querying system catalog tables and columns for schema %s",
            self._settings.db_schema,
        )
        async with self.pool.acquire() as conn:
            tables = await conn.fetch(TABLES_SQL, self._settings.db_schema)
            logger.debug("db: catalog read %d tables", len(tables))
            columns = await conn.fetch(COLUMNS_SQL, self._settings.db_schema)
            logger.debug("db: catalog read %d columns", len(columns))
            constraints = await conn.fetch(CONSTRAINTS_SQL, self._settings.db_schema)
            logger.debug("db: catalog read %d constraints", len(constraints))
            foreign_keys_raw = await conn.fetch(FOREIGN_KEYS_SQL, self._settings.db_schema)
            logger.debug("db: catalog read %d foreign keys", len(foreign_keys_raw))
            try:
                enums_raw = await conn.fetch(ENUMS_SQL)
                logger.debug("db: catalog read %d enums", len(enums_raw))
            except Exception as exc:
                logger.debug("db: enum fetch skipped (%s)", exc)
                enums_raw = []

        enums_by_type: dict[str, list[str]] = {}
        for row in enums_raw:
            enums_by_type.setdefault(row["type_name"], []).append(row["enum_label"])

        pk_by_table: dict[str, set[str]] = {}
        uq_by_table: dict[str, set[str]] = {}
        for row in constraints:
            t_name = row["table_name"]
            c_name = row["column_name"]
            if row["constraint_type"] == "p":
                pk_by_table.setdefault(t_name, set()).add(c_name)
            elif row["constraint_type"] == "u":
                uq_by_table.setdefault(t_name, set()).add(c_name)

        foreign_keys: list[ForeignKeyInfo] = [
            ForeignKeyInfo(
                source_table=row["source_table"],
                source_column=row["source_column"],
                target_table=row["target_table"],
                target_column=row["target_column"],
            )
            for row in foreign_keys_raw
        ]
        fks_by_table: dict[str, list[ForeignKeyInfo]] = {}
        for fk in foreign_keys:
            fks_by_table.setdefault(fk.source_table, []).append(fk)

        columns_by_table: dict[str, list] = {}
        for row in columns:
            table_name = row["table_name"]
            col_name = row["name"]
            data_type = row["data_type"]
            is_pk = col_name in pk_by_table.get(table_name, set())
            is_uq = col_name in uq_by_table.get(table_name, set())
            enum_values = enums_by_type.get(data_type, [])
            columns_by_table.setdefault(table_name, []).append(
                {
                    "name": col_name,
                    "data_type": data_type,
                    "type_category": row["type_category"],
                    "nullable": row["nullable"],
                    "description": row["description"],
                    "is_primary_key": is_pk,
                    "is_unique": is_uq,
                    "enum_values": enum_values,
                }
            )

        tables_by_name: dict[str, TableInfo] = {}
        for row in tables:
            name = row["name"]
            table_pks = sorted(pk_by_table.get(name, set()))
            tables_by_name[name] = TableInfo(
                name=name,
                description=row["description"],
                columns=columns_by_table.get(name, []),
                row_estimate=max(int(row["row_estimate"] or 0), 0),
                foreign_keys=fks_by_table.get(name, []),
                primary_keys=table_pks,
            )

        logger.debug(
            "db: catalog graph assembled with %d tables, %d foreign keys",
            len(tables_by_name),
            len(foreign_keys),
        )
        return SchemaGraph(tables=tables_by_name, foreign_keys=foreign_keys)

    async def ping(self) -> bool:
        if not self.connected:
            logger.debug("db: ping without an open pool reports down")
            return False
        if self._is_sqlite:
            try:
                assert self._sqlite_conn is not None
                self._sqlite_conn.execute("SELECT 1")
                logger.debug("db: sqlite ping ok")
                return True
            except Exception as exc:
                logger.debug("db: sqlite ping failed: %s: %s", type(exc).__name__, exc)
                return False
        logger.debug("db: executing ping query (SELECT 1)")
        try:
            async with self.pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
            logger.debug("db: ping ok")
            return True
        except Exception as exc:
            logger.debug("db: ping failed: %s: %s", type(exc).__name__, exc)
            return False

    async def execute(self, statement: Statement) -> list[dict]:
        sql = statement.sql
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
