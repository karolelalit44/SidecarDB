from typing import Any

from catalog.models import ColumnInfo, ForeignKeyInfo, SchemaGraph, TableInfo
from catalog.queries import (
    COLUMNS_SQL,
    CONSTRAINTS_SQL,
    ENUMS_SQL,
    FOREIGN_KEYS_SQL,
    TABLES_SQL,
)
from core.logging import get_logger

logger = get_logger("catalog.introspect")


def read_sqlite_catalog(connection: Any) -> SchemaGraph:
    cursor = connection.cursor()
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


async def read_postgres_catalog(pool: Any, schema: str) -> SchemaGraph:
    logger.debug("db: querying system catalog tables and columns for schema %s", schema)
    async with pool.acquire() as conn:
        tables = await conn.fetch(TABLES_SQL, schema)
        logger.debug("db: catalog read %d tables", len(tables))
        columns = await conn.fetch(COLUMNS_SQL, schema)
        logger.debug("db: catalog read %d columns", len(columns))
        constraints = await conn.fetch(CONSTRAINTS_SQL, schema)
        logger.debug("db: catalog read %d constraints", len(constraints))
        foreign_keys_raw = await conn.fetch(FOREIGN_KEYS_SQL, schema)
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
            foreign_keys=fks_by_table.get(name, []),
            primary_keys=table_pks,
        )

    logger.debug(
        "db: catalog graph assembled with %d tables, %d foreign keys",
        len(tables_by_name),
        len(foreign_keys),
    )
    return SchemaGraph(tables=tables_by_name, foreign_keys=foreign_keys)


__all__ = ["read_postgres_catalog", "read_sqlite_catalog"]
