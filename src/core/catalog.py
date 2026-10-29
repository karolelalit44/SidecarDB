from datetime import UTC, datetime

from pydantic import BaseModel, Field

TYPE_CATEGORY_NUMERIC = "N"
TYPE_CATEGORY_UNKNOWN = "X"


class ColumnInfo(BaseModel):
    name: str
    data_type: str
    type_category: str = TYPE_CATEGORY_UNKNOWN
    nullable: bool = True
    description: str | None = None

    @property
    def is_numeric(self) -> bool:
        return self.type_category == TYPE_CATEGORY_NUMERIC


class TableInfo(BaseModel):
    name: str
    description: str | None = None
    columns: list[ColumnInfo] = Field(default_factory=list)
    row_estimate: int = 0

    @property
    def column_names(self) -> list[str]:
        return [c.name for c in self.columns]

    @property
    def numeric_columns(self) -> list[str]:
        return [c.name for c in self.columns if c.is_numeric]


class SchemaGraph(BaseModel):
    tables: dict[str, TableInfo] = Field(default_factory=dict)
    introspected_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @property
    def names(self) -> list[str]:
        return list(self.tables)

    def describe(self, tables: list[str] | None = None, max_columns: int = 0) -> str:
        selected = (
            [self.tables[n] for n in tables if n in self.tables]
            if tables
            else list(self.tables.values())
        )
        lines: list[str] = []
        budget = max_columns if max_columns > 0 else None
        emitted = 0

        for table in selected:
            header = f"TABLE {table.name}"
            if table.description:
                header += f"  -- {table.description}"
            lines.append(header)

            columns = table.columns
            truncated = False
            if budget is not None and emitted + len(columns) > budget:
                allowed = max(budget - emitted, 0)
                columns = columns[:allowed]
                truncated = True

            for column in columns:
                emitted += 1
                note = f" -- {column.description}" if column.description else ""
                lines.append(f"    {column.name} ({column.data_type}){note}")

            if truncated:
                lines.append(f"    ... {len(table.columns) - len(columns)} more columns")

            lines.append("")

        return "\n".join(lines).strip()


__all__ = [
    "TYPE_CATEGORY_NUMERIC",
    "TYPE_CATEGORY_UNKNOWN",
    "ColumnInfo",
    "SchemaGraph",
    "TableInfo",
]
