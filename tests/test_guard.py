import pytest

from core.catalog import ColumnInfo, SchemaGraph, TableInfo
from core.exceptions import QuestionError, SecurityError
from core.guard import (
    is_safe_identifier,
    quote_identifier,
    quote_literal,
    statement_for,
    validate,
    validate_question,
)
from core.plan import JoinSpec, JoinType, Plan, QueryIntent


def test_validate_question_success():
    assert (
        validate_question("  How many orders were placed today?  ")
        == "How many orders were placed today?"
    )


def test_validate_question_rejects_too_short():
    with pytest.raises(QuestionError, match="too short"):
        validate_question("a")


def test_validate_question_rejects_disallowed_symbols():
    with pytest.raises(QuestionError, match="disallowed symbols"):
        validate_question("show orders; DROP TABLE users;--")


def test_validate_question_rejects_no_alphanumeric():
    with pytest.raises(QuestionError, match="alphanumeric"):
        validate_question("??? !!! +++")


def test_safe_identifier_and_quoting():
    assert is_safe_identifier("valid_table_name") is True
    assert is_safe_identifier("table;drop") is False
    assert is_safe_identifier("123bad") is False
    assert quote_identifier('my"table') == '"my""table"'
    assert quote_literal("O'Reilly") == "'O''Reilly'"


def test_statement_for_and_ast_guard():
    graph = SchemaGraph(
        tables={
            "users": TableInfo(
                name="users",
                columns=[
                    ColumnInfo(name="id", data_type="integer"),
                    ColumnInfo(name="email", data_type="text"),
                ],
            )
        }
    )
    plan = Plan(
        intent=QueryIntent.retrieve,
        table="users",
        projection=["id", "email"],
        template="Found {row_count} users.",
    )

    statement = statement_for(plan, graph, "public")
    assert "SELECT" in statement.sql
    assert '"users"' in statement.sql
    assert "LIMIT 25" in statement.sql


def test_ast_guard_blocks_mutations():
    with pytest.raises(SecurityError, match="Only SELECT is permitted"):
        validate("DELETE FROM users WHERE id = 1", None, "public")


def test_statement_for_multi_table_join():
    graph = SchemaGraph(
        tables={
            "orders": TableInfo(
                name="orders",
                columns=[
                    ColumnInfo(name="id", data_type="integer"),
                    ColumnInfo(name="user_id", data_type="integer"),
                    ColumnInfo(name="total", data_type="numeric"),
                ],
            ),
            "users": TableInfo(
                name="users",
                columns=[
                    ColumnInfo(name="id", data_type="integer"),
                    ColumnInfo(name="name", data_type="text"),
                ],
            ),
        }
    )
    plan = Plan(
        intent=QueryIntent.retrieve,
        table="orders",
        joins=[
            JoinSpec(
                table="orders",
                on_column="user_id",
                target_table="users",
                target_column="id",
                join_type=JoinType.inner,
            )
        ],
        projection=["orders.id", "users.name", "orders.total"],
        filter_column="users.name",
        filter_operator="=",
        filter_value="Alice",
        template="Found {row_count} orders.",
    )
    stmt = statement_for(plan, graph, "public")
    assert 'INNER JOIN "users" ON "orders"."user_id" = "users"."id"' in stmt.sql
    assert 'WHERE "users"."name" = $1' in stmt.sql
    assert stmt.params == ("Alice",)


def test_statement_for_ansi_cast_and_lower_like():
    graph = SchemaGraph(
        tables={
            "products": TableInfo(
                name="products",
                columns=[
                    ColumnInfo(name="id", data_type="integer"),
                    ColumnInfo(name="title", data_type="text"),
                    ColumnInfo(name="price", data_type="numeric"),
                ],
            )
        }
    )
    plan_cast = Plan(
        intent=QueryIntent.retrieve,
        table="products",
        filter_column="price",
        filter_operator=">",
        filter_value="50",
        template="Found {row_count} products.",
    )
    stmt_cast = statement_for(plan_cast, graph, "public")
    assert "CAST($1 AS numeric)" in stmt_cast.sql

    plan_text = Plan(
        intent=QueryIntent.retrieve,
        table="products",
        text_search_columns=["title"],
        text_search_query="phone",
        template="Found {row_count} products.",
    )
    stmt_text = statement_for(plan_text, graph, "public")
    assert 'LOWER("title") LIKE LOWER($1)' in stmt_text.sql
