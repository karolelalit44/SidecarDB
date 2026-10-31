COLUMNS_SQL = """SELECT c.relname AS table_name,
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
  AND c.relkind IN ('r', 'p', 'v', 'm', 'f')
  AND a.attnum > 0
  AND NOT a.attisdropped
ORDER BY c.relname, a.attnum"""

TABLES_SQL = """SELECT c.relname AS name,
       obj_description(c.oid, 'pg_class') AS description
FROM pg_class c
JOIN pg_namespace n ON n.oid = c.relnamespace
WHERE n.nspname = $1 AND c.relkind IN ('r', 'p', 'v', 'm', 'f')
ORDER BY c.relname"""

CONSTRAINTS_SQL = """SELECT c.relname AS table_name,
       a.attname AS column_name,
       con.contype AS constraint_type
FROM pg_constraint con
JOIN pg_class c ON c.oid = con.conrelid
JOIN pg_namespace n ON n.oid = c.relnamespace
JOIN pg_attribute a ON a.attrelid = c.oid AND a.attnum = ANY(con.conkey)
WHERE n.nspname = $1
  AND con.contype IN ('p', 'u')
  AND c.relkind IN ('r', 'p', 'v', 'm', 'f')
  AND a.attnum > 0
  AND NOT a.attisdropped"""

FOREIGN_KEYS_SQL = """SELECT s_rel.relname AS source_table,
       s_att.attname AS source_column,
       t_rel.relname AS target_table,
       t_att.attname AS target_column
FROM (
    SELECT conrelid,
           confrelid,
           unnest(conkey) AS s_attnum,
           unnest(confkey) AS t_attnum
    FROM pg_constraint con
    JOIN pg_namespace n ON n.oid = con.connamespace
    WHERE n.nspname = $1 AND con.contype = 'f'
) fk
JOIN pg_class s_rel ON s_rel.oid = fk.conrelid
JOIN pg_attribute s_att ON s_att.attrelid = fk.conrelid AND s_att.attnum = fk.s_attnum
JOIN pg_class t_rel ON t_rel.oid = fk.confrelid
JOIN pg_attribute t_att ON t_att.attrelid = fk.confrelid AND t_att.attnum = fk.t_attnum
WHERE NOT s_att.attisdropped AND NOT t_att.attisdropped
ORDER BY source_table, source_column"""

ENUMS_SQL = """SELECT t.typname AS type_name,
       e.enumlabel AS enum_label
FROM pg_type t
JOIN pg_enum e ON e.enumtypid = t.oid
ORDER BY t.typname, e.enumsortorder"""

__all__ = [
    "COLUMNS_SQL",
    "CONSTRAINTS_SQL",
    "ENUMS_SQL",
    "FOREIGN_KEYS_SQL",
    "TABLES_SQL",
]
