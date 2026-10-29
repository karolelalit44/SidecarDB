from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field, field_validator

from core.constants import (
    ERROR_MESSAGE_MAX_CHARS,
    LLM_CHARS_PER_TOKEN,
    LLM_CONTEXT_TOKENS,
    LLM_MAX_TOKENS,
    LLM_REPAIR_TEMPERATURE,
    LLM_TEMPERATURE,
    MAX_PLAN_ATTEMPTS,
    MAX_SCHEMA_COLUMNS,
    MAX_SCHEMA_TABLES,
    REPAIR_CHAR_BUDGET,
    RETRIEVAL_MIN_SCORE,
    RETRIEVAL_SCORE_PRECISION,
    RETRIEVAL_TOP_K,
)
from utils.text import split_csv

if TYPE_CHECKING:
    from core.retrieval import RetrievalMode


class QueryIntent(StrEnum):
    count = "count"  # type: ignore[assignment]
    aggregate = "aggregate"
    retrieve = "retrieve"
    distinct = "distinct"


class MeasureFunction(StrEnum):
    sum = "sum"
    avg = "avg"
    min = "min"
    max = "max"
    count = "count"  # type: ignore[assignment]

    @property
    def needs_column(self) -> bool:
        return self is not MeasureFunction.count


class Plan(BaseModel):
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
        if value is None:
            return []
        if isinstance(value, str):
            if not value.strip():
                return []
            return split_csv(value)
        return value

    @field_validator("order_by", mode="before")
    @classmethod
    def _coerce_order_by(cls, value: object) -> object:
        if isinstance(value, list):
            return str(value[0]) if value else None
        return value


@dataclass(frozen=True, slots=True)
class PlanPolicy:
    llm_temperature: float = LLM_TEMPERATURE
    llm_repair_temperature: float = LLM_REPAIR_TEMPERATURE
    repair_char_budget: int = REPAIR_CHAR_BUDGET
    max_plan_attempts: int = MAX_PLAN_ATTEMPTS
    error_message_max_chars: int = ERROR_MESSAGE_MAX_CHARS
    llm_max_tokens: int = LLM_MAX_TOKENS
    llm_chars_per_token: int = LLM_CHARS_PER_TOKEN
    llm_context_tokens: int = LLM_CONTEXT_TOKENS
    max_schema_tables: int = MAX_SCHEMA_TABLES
    max_schema_columns: int = MAX_SCHEMA_COLUMNS
    retrieval_top_k: int = RETRIEVAL_TOP_K
    retrieval_min_score: float = RETRIEVAL_MIN_SCORE
    retrieval_score_precision: int = RETRIEVAL_SCORE_PRECISION


@dataclass(frozen=True, slots=True)
class Statement:
    sql: str
    params: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class Outcome:
    plan: Plan
    statement: Statement
    rows: list[dict]
    retrieved_tables: list[str]
    retrieval_mode: "RetrievalMode"


COMPARISON_OPERATORS = ("=", "!=", "<>", "<", "<=", ">", ">=")
PATTERN_OPERATORS = ("LIKE", "ILIKE")
NULL_TESTS = ("IS NULL", "IS NOT NULL")
OPERATORS = (*COMPARISON_OPERATORS, *PATTERN_OPERATORS, *NULL_TESTS)

DEFAULT_COMPARISON = COMPARISON_OPERATORS[0]

COUNT_ALIAS = "record_count"

WILDCARD = "*"

SCALAR_INTENTS = frozenset((QueryIntent.count, QueryIntent.aggregate))
MEASURED_INTENTS = frozenset((QueryIntent.aggregate,))

__all__ = [
    "COMPARISON_OPERATORS",
    "COUNT_ALIAS",
    "DEFAULT_COMPARISON",
    "MEASURED_INTENTS",
    "NULL_TESTS",
    "OPERATORS",
    "PATTERN_OPERATORS",
    "SCALAR_INTENTS",
    "WILDCARD",
    "MeasureFunction",
    "Outcome",
    "Plan",
    "PlanPolicy",
    "QueryIntent",
    "Statement",
]
