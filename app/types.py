from datetime import UTC, datetime
from enum import StrEnum
from functools import cached_property

from pydantic import BaseModel, Field, field_validator

from .constants import TYPE_CATEGORY_NUMERIC, TYPE_CATEGORY_STRING, TYPE_CATEGORY_UNKNOWN


class RetrievalMode(StrEnum):
    embedding = "embedding"
    catalog = "catalog"


class QueryIntent(StrEnum):
    count = "count"
    aggregate = "aggregate"
    retrieve = "retrieve"
    lookup = "lookup"
    distinct = "distinct"


class MeasureFunction(StrEnum):
    sum = "sum"
    avg = "avg"
    min = "min"
    max = "max"
    # Counts rows rather than reading a column, so it is the one measure that
    # needs no measure_column. Without it "count X grouped by Y" has no plan.
    count = "count"

    @property
    def needs_column(self) -> bool:
        """Whether the measure reads a column."""
        return self is not MeasureFunction.count


class ForeignKeyInfo(BaseModel):
    from_table: str
    from_column: str
    to_table: str
    to_column: str


class IndexedColumn(BaseModel):
    table: str
    column: str
    data_type: str
    extension: str


class Hit(BaseModel):
    """One catalog element's score for the incoming question."""

    table: str
    subject: str
    score: float


class QueryPlan(BaseModel):
    intent: QueryIntent
    table: str
    related_tables: list[str] = Field(default_factory=list)
    measure_column: str | None = None
    measure_function: MeasureFunction | None = None
    filter_column: str | None = None
    filter_value: str | None = None
    filter_operator: str | None = None
    projection: list[str] = Field(default_factory=list)
    order_by: str | None = None
    descending: bool = False
    group_by: list[str] = Field(default_factory=list)
    text_search_columns: list[str] = Field(default_factory=list)
    text_search_query: str | None = None
    template: str | None = Field(default=None, max_length=200)

    @field_validator(
        "related_tables", "projection", "group_by", "text_search_columns", mode="before"
    )
    @classmethod
    def _coerce_list(cls, value: object) -> object:
        """Coerce string, null, or list input to a list of strings."""
        if value is None:
            return []
        if isinstance(value, str):
            if not value.strip():
                return []
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    @field_validator("order_by", mode="before")
    @classmethod
    def _coerce_order_by(cls, value: object) -> object:
        if isinstance(value, list):
            return str(value[0]) if value else None
        return value

    @property
    def tables(self) -> list[str]:
        return [self.table, *self.related_tables]


class ColumnInfo(BaseModel):
    name: str
    data_type: str
    type_category: str = TYPE_CATEGORY_UNKNOWN
    nullable: bool = True
    primary_key: bool = False
    description: str | None = None
    distinct_estimate: float = 0.0
    dimensions: int | None = None

    @property
    def is_numeric(self) -> bool:
        return self.type_category == TYPE_CATEGORY_NUMERIC

    @property
    def is_string(self) -> bool:
        return self.type_category == TYPE_CATEGORY_STRING


class TableInfo(BaseModel):
    name: str
    description: str | None = None
    columns: list[ColumnInfo] = Field(default_factory=list)
    primary_keys: list[str] = Field(default_factory=list)
    foreign_keys: list[ForeignKeyInfo] = Field(default_factory=list)
    row_estimate: int = 0
    embedding_columns: list[str] = Field(default_factory=list)

    @property
    def column_names(self) -> list[str]:
        return [c.name for c in self.columns]

    @property
    def numeric_columns(self) -> list[str]:
        return [c.name for c in self.columns if c.is_numeric]

    @property
    def embedding_column(self) -> str | None:
        return self.embedding_columns[0] if self.embedding_columns else None


class SchemaGraph(BaseModel):
    tables: dict[str, TableInfo] = Field(default_factory=dict)
    indexed_columns: list[IndexedColumn] = Field(default_factory=list)
    introspected_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @property
    def names(self) -> list[str]:
        return list(self.tables)

    @cached_property
    def foreign_keys(self) -> list[ForeignKeyInfo]:
        """All foreign keys in the schema graph."""
        return [fk for t in self.tables.values() for fk in t.foreign_keys]

    def embedding_column_for(self, table: str) -> str | None:
        info = self.tables.get(table)
        return info.embedding_column if info else None

    def embedding_dimensions_for(self, table: str) -> int | None:
        """Declared width of the table's vector column, or None if unconstrained."""
        info = self.tables.get(table)
        if info is None or not info.embedding_column:
            return None
        column = next((c for c in info.columns if c.name == info.embedding_column), None)
        return column.dimensions if column else None

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
                marks = ["PK"] if column.primary_key else []
                marks.append(column.data_type)
                note = f" -- {column.description}" if column.description else ""
                lines.append(f"    {column.name} ({', '.join(marks)}){note}")

            if truncated:
                lines.append(f"    ... {len(table.columns) - len(columns)} more columns")

            lines.extend(
                f"    FOREIGN KEY {fk.from_column} -> {fk.to_table}.{fk.to_column}"
                for fk in table.foreign_keys
            )

            lines.append("")

        return "\n".join(lines).strip()


__all__ = [
    "ColumnInfo",
    "ForeignKeyInfo",
    "Hit",
    "IndexedColumn",
    "MeasureFunction",
    "QueryIntent",
    "QueryPlan",
    "RetrievalMode",
    "SchemaGraph",
    "TableInfo",
]
