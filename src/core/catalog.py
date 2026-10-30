from datetime import UTC, datetime

from pydantic import BaseModel, Field

from core.constants import NUMERIC_DATA_TYPES, TYPE_CATEGORY_NUMERIC, TYPE_CATEGORY_UNKNOWN
from utils.logging_config import get_logger

logger = get_logger("catalog")


class ForeignKeyInfo(BaseModel):
    source_table: str
    source_column: str
    target_table: str
    target_column: str


class ColumnInfo(BaseModel):
    name: str
    data_type: str
    type_category: str = TYPE_CATEGORY_UNKNOWN
    nullable: bool = True
    description: str | None = None
    is_primary_key: bool = False
    is_unique: bool = False
    enum_values: list[str] = Field(default_factory=list)

    @property
    def is_numeric(self) -> bool:
        if self.type_category == TYPE_CATEGORY_NUMERIC:
            return True
        base_type = self.data_type.lower().split("(")[0].strip()
        return base_type in NUMERIC_DATA_TYPES


class TableInfo(BaseModel):
    name: str
    description: str | None = None
    columns: list[ColumnInfo] = Field(default_factory=list)
    row_estimate: int = 0
    foreign_keys: list[ForeignKeyInfo] = Field(default_factory=list)
    primary_keys: list[str] = Field(default_factory=list)

    @property
    def column_names(self) -> list[str]:
        return [c.name for c in self.columns]

    @property
    def numeric_columns(self) -> list[str]:
        return [c.name for c in self.columns if c.is_numeric]


def _allocate_column_budgets(tables: list[TableInfo], total_budget: int) -> dict[str, int]:
    if not tables or total_budget <= 0:
        return {}
    allocations = {t.name: 0 for t in tables}
    remaining = total_budget
    needed = {t.name: len(t.columns) for t in tables}

    while remaining > 0 and any(allocations[name] < needed[name] for name in allocations):
        active = [name for name in allocations if allocations[name] < needed[name]]
        share = max(1, remaining // len(active))
        progress = False
        for name in active:
            take = min(share, needed[name] - allocations[name], remaining)
            if take > 0:
                allocations[name] += take
                remaining -= take
                progress = True
            if remaining == 0:
                break
        if not progress:
            break
    return allocations


class SchemaGraph(BaseModel):
    tables: dict[str, TableInfo] = Field(default_factory=dict)
    foreign_keys: list[ForeignKeyInfo] = Field(default_factory=list)
    introspected_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @property
    def names(self) -> list[str]:
        return list(self.tables)

    @property
    def signature(self) -> str:
        import hashlib

        hasher = hashlib.sha256()
        for t_name in sorted(self.tables):
            table = self.tables[t_name]
            hasher.update(f"table:{table.name}:{table.description or ''}".encode())
            for col in table.columns:
                hasher.update(
                    f"col:{col.name}:{col.data_type}:{col.is_primary_key}:{col.is_unique}:"
                    f"{col.description or ''}:{col.enum_values}".encode()
                )
        for fk in sorted(self.foreign_keys, key=lambda f: (f.source_table, f.source_column)):
            hasher.update(
                f"fk:{fk.source_table}.{fk.source_column}->{fk.target_table}.{fk.target_column}".encode()
            )
        return hasher.hexdigest()

    def neighbors_of(self, tables: list[str]) -> list[str]:
        target_set = set(tables)
        neighbors: list[str] = []
        for fk in self.foreign_keys:
            if (
                fk.source_table in target_set
                and fk.target_table in self.tables
                and fk.target_table not in target_set
                and fk.target_table not in neighbors
            ):
                neighbors.append(fk.target_table)
            elif (
                fk.target_table in target_set
                and fk.source_table in self.tables
                and fk.source_table not in target_set
                and fk.source_table not in neighbors
            ):
                neighbors.append(fk.source_table)
        return neighbors

    def describe(self, tables: list[str] | None = None, max_columns: int = 0) -> str:
        selected = (
            [self.tables[n] for n in tables if n in self.tables]
            if tables
            else list(self.tables.values())
        )
        lines: list[str] = []
        budget = max_columns if max_columns > 0 else None
        budgets = (
            _allocate_column_budgets(selected, budget)
            if budget is not None
            else {t.name: len(t.columns) for t in selected}
        )
        logger.debug(
            "describe: formatting schema for %d table(s) (budget=%s columns)",
            len(selected),
            budget,
        )

        for table in selected:
            header = f"TABLE {table.name}"
            if table.description:
                header += f"  -- {table.description}"
            lines.append(header)

            allowed = budgets.get(table.name, len(table.columns))
            columns = table.columns[:allowed]
            truncated = allowed < len(table.columns)

            fk_map = {
                fk.source_column: f"{fk.target_table}.{fk.target_column}"
                for fk in table.foreign_keys
            }

            for column in columns:
                if column.enum_values:
                    type_str = f"enum: {' | '.join(repr(v) for v in column.enum_values)}"
                else:
                    type_str = column.data_type

                flags: list[str] = []
                if column.is_primary_key:
                    flags.append("[PK]")
                elif column.is_unique:
                    flags.append("[UQ]")
                if column.name in fk_map:
                    flags.append(f"[FK -> {fk_map[column.name]}]")

                flag_str = (" " + " ".join(flags)) if flags else ""
                note = f" -- {column.description}" if column.description else ""
                lines.append(f"    {column.name} ({type_str}){flag_str}{note}")

            if truncated:
                lines.append(f"    ... {len(table.columns) - allowed} more columns")

            lines.append("")

        return "\n".join(lines).strip()


__all__ = [
    "ColumnInfo",
    "ForeignKeyInfo",
    "SchemaGraph",
    "TableInfo",
]
