import asyncio

from .config import settings
from .exceptions import ConfigurationError, SchemaError
from .logging_config import get_logger
from .types import ForeignKeyInfo, IndexedColumn, SchemaGraph, TableInfo

logger = get_logger("db")

MS_PER_SECOND = 1000

_COLUMNS_SQL = """
SELECT c.relname AS table_name,
       a.attname AS name,
       format_type(a.atttypid, a.atttypmod) AS data_type,
       base.typcategory AS type_category,
       NOT a.attnotnull AS nullable,
       col_description(c.oid, a.attnum) AS description,
       c.reltuples AS row_estimate
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

_PRIMARY_KEYS_SQL = """
SELECT c.relname AS table_name, a.attname AS name
FROM pg_index i
JOIN pg_class c ON c.oid = i.indrelid
JOIN pg_namespace n ON n.oid = c.relnamespace
JOIN pg_attribute a ON a.attrelid = c.oid AND a.attnum = ANY(i.indkey)
WHERE n.nspname = $1 AND i.indisprimary
"""

_FOREIGN_KEYS_SQL = """
SELECT src.relname AS from_table, sa.attname AS from_column,
       tgt.relname AS to_table, ta.attname AS to_column
FROM pg_constraint con
JOIN pg_class src ON src.oid = con.conrelid
JOIN pg_class tgt ON tgt.oid = con.confrelid
JOIN pg_namespace n ON n.oid = src.relnamespace
JOIN pg_attribute sa ON sa.attrelid = con.conrelid AND sa.attnum = con.conkey[1]
JOIN pg_attribute ta ON ta.attrelid = con.confrelid AND ta.attnum = con.confkey[1]
WHERE con.contype = 'f' AND n.nspname = $1
"""

_COLUMN_STATS_SQL = """
SELECT tablename AS table_name,
       attname AS name,
       n_distinct
FROM pg_stats
WHERE schemaname = $1
"""

_EXTENSION_COLUMNS_SQL = """
SELECT c.relname AS table_name,
       a.attname AS name,
       format_type(a.atttypid, a.atttypmod) AS data_type,
       e.extname AS extension
FROM pg_attribute a
JOIN pg_class c ON c.oid = a.attrelid
JOIN pg_namespace n ON n.oid = c.relnamespace
JOIN pg_type t ON t.oid = a.atttypid
JOIN pg_depend d ON d.classid = 'pg_type'::regclass
                AND d.objid = t.oid
                AND d.deptype = 'e'
JOIN pg_extension e ON e.oid = d.refobjid
WHERE n.nspname = $1
  AND c.relkind = 'r'
  AND a.attnum > 0
  AND NOT a.attisdropped
ORDER BY c.relname, a.attnum
"""


class Database:
    def __init__(self) -> None:
        self._pool = None

    async def connect(self, *, readonly: bool = True) -> None:
        """Opens the pool.

        readonly=True is what the service must use. The maintenance script, whose
        whole job is to write embeddings back, passes readonly=False -- the flag
        exists for that one caller and defaults to the safe value.
        """
        import asyncpg

        async def _init(conn):
            statement_ms = int(settings.sandbox_statement_timeout_seconds * MS_PER_SECOND)
            idle_ms = settings.sandbox_idle_timeout_ms
            if readonly:
                await conn.execute("SET SESSION CHARACTERISTICS AS TRANSACTION READ ONLY")
            await conn.execute(f"SET statement_timeout = '{statement_ms}ms'")
            await conn.execute(f"SET idle_in_transaction_session_timeout = '{idle_ms}ms'")
            await conn.execute("SET search_path = " + _quote_literal(settings.db_schema))

        if self._pool is None:
            self._pool = await asyncpg.create_pool(
                dsn=settings.database_url,
                min_size=settings.pool_min_size,
                max_size=settings.pool_max_size,
                command_timeout=settings.sandbox_statement_timeout_seconds,
                init=_init,
            )

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
        self._pool = None

    async def introspect(self) -> SchemaGraph:
        if self._pool is None:
            raise ConfigurationError("Database is not connected.")
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
        return graph

    async def _read_catalog(self) -> SchemaGraph:
        async with self._pool.acquire() as conn:
            tables = await conn.fetch(_TABLES_SQL, settings.db_schema)
            columns = await conn.fetch(_COLUMNS_SQL, settings.db_schema)
            primary = await conn.fetch(_PRIMARY_KEYS_SQL, settings.db_schema)
            foreign = await conn.fetch(_FOREIGN_KEYS_SQL, settings.db_schema)
            indexed = await conn.fetch(_EXTENSION_COLUMNS_SQL, settings.db_schema)
            stats = await conn.fetch(_COLUMN_STATS_SQL, settings.db_schema)

        primary_by_table: dict[str, list[str]] = {}
        for row in primary:
            primary_by_table.setdefault(row["table_name"], []).append(row["name"])

        foreign_by_table: dict[str, list[ForeignKeyInfo]] = {}
        for row in foreign:
            foreign_by_table.setdefault(row["from_table"], []).append(
                ForeignKeyInfo(
                    from_table=row["from_table"],
                    from_column=row["from_column"],
                    to_table=row["to_table"],
                    to_column=row["to_column"],
                )
            )

        embedding_by_table: dict[str, list[str]] = {}
        indexed_columns: list[IndexedColumn] = []
        for row in indexed:
            embedding_by_table.setdefault(row["table_name"], []).append(row["name"])
            indexed_columns.append(
                IndexedColumn(
                    table=row["table_name"],
                    column=row["name"],
                    data_type=row["data_type"],
                    extension=row["extension"],
                )
            )

        stats_by_column: dict[tuple[str, str], float] = {}
        for row in stats:
            if row["n_distinct"] is not None:
                stats_by_column[(row["table_name"], row["name"])] = float(row["n_distinct"])

        row_estimates = {row["name"]: max(int(row["row_estimate"] or 0), 0) for row in tables}

        columns_by_table: dict[str, list] = {}
        for row in columns:
            table_name = row["table_name"]
            raw = stats_by_column.get((table_name, row["name"]))
            columns_by_table.setdefault(table_name, []).append(
                {
                    "name": row["name"],
                    "data_type": row["data_type"],
                    "type_category": row["type_category"],
                    "nullable": row["nullable"],
                    "primary_key": row["name"] in primary_by_table.get(table_name, []),
                    "description": row["description"],
                    "distinct_estimate": _distinct_count(raw, row_estimates.get(table_name, 0)),
                }
            )

        graph: dict[str, TableInfo] = {}
        for row in tables:
            name = row["name"]
            graph[name] = TableInfo(
                name=name,
                description=row["description"],
                columns=columns_by_table.get(name, []),
                primary_keys=primary_by_table.get(name, []),
                foreign_keys=foreign_by_table.get(name, []),
                row_estimate=max(int(row["row_estimate"] or 0), 0),
                embedding_columns=embedding_by_table.get(name, []),
            )

        return SchemaGraph(tables=graph, indexed_columns=indexed_columns)

    async def fetch(self, sql: str, *args):
        if self._pool is None:
            raise ConfigurationError("Database is not connected.")
        async with self._pool.acquire() as conn:
            records = await conn.fetch(sql, *args)
            return [dict(record) for record in records]

    async def query_pool(self):
        if self._pool is None:
            raise ConfigurationError("Database is not connected.")
        return self._pool

    async def ping(self) -> bool:
        if self._pool is None:
            return False
        try:
            async with self._pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
            return True
        except Exception:
            return False


def _distinct_count(n_distinct: float | None, row_estimate: int) -> float:
    if n_distinct is None or row_estimate <= 0:
        return 0.0
    if n_distinct < 0:
        return abs(n_distinct) * row_estimate
    return n_distinct


def join_path(source: str, target: str, graph: SchemaGraph, max_depth: int):
    if source == target:
        return []
    frontier = [(source, [])]
    seen = {source}
    for _ in range(max_depth):
        nxt = []
        for node, path in frontier:
            for fk in graph.foreign_keys:
                if node not in (fk.from_table, fk.to_table):
                    continue
                for neighbour in (fk.from_table, fk.to_table):
                    if neighbour in seen:
                        continue
                    extended = [*path, neighbour]
                    if neighbour == target:
                        return extended
                    seen.add(neighbour)
                    nxt.append((neighbour, extended))
        if not nxt:
            break
        frontier = nxt
    return None


def _quote_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


database = Database()
