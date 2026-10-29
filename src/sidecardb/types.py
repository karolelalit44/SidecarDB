from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, Field, field_validator

from .constants import TYPE_CATEGORY_NUMERIC, TYPE_CATEGORY_UNKNOWN


class RetrievalMode(StrEnum):
    embedding = "embedding"
    catalog = "catalog"


class QueryIntent(StrEnum):
    # `count` shadows str.count for mypy; that is the point of a StrEnum
    # member name and is safe at runtime.
    count = "count"  # type: ignore[assignment]
    aggregate = "aggregate"
    retrieve = "retrieve"
    distinct = "distinct"


class MeasureFunction(StrEnum):
    sum = "sum"
    avg = "avg"
    min = "min"
    max = "max"
    # Counts rows rather than reading a column, so it is the one measure that
    # needs no measure_column. Without it "count X grouped by Y" has no plan.
    # Shadows str.count like every count member of a StrEnum; safe at runtime.
    count = "count"  # type: ignore[assignment]

    @property
    def needs_column(self) -> bool:
        """Whether the measure reads a column."""
        return self is not MeasureFunction.count


class Hit(BaseModel):
    """One catalog element's score for the incoming question."""

    table: str
    subject: str
    score: float


class QueryPlan(BaseModel):
    intent: QueryIntent
    table: str
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

    @field_validator("projection", "group_by", "text_search_columns", mode="before")
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
    "ColumnInfo",
    "Hit",
    "MeasureFunction",
    "QueryIntent",
    "QueryPlan",
    "RetrievalMode",
    "SchemaGraph",
    "TableInfo",
]
