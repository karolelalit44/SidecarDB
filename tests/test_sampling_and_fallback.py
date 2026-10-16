"""Which column values are allowed to reach the model, and the vector fallback."""

import pytest

from app import fallback
from app.config import settings
from app.db import _vector_width
from app.graph import _samplable
from app.types import ColumnInfo, SchemaGraph, TableInfo


def column(name, **kw):
    data_type = kw.pop("data_type", "text")
    kw.setdefault("type_category", "N" if data_type == "numeric" else "S")
    return ColumnInfo(name=name, data_type=data_type, **kw)


def samplable(col, row_estimate, graph=None, table=None):
    return _samplable(table or "t", col, row_estimate)


# --- which columns may be sampled -------------------------------------------


def test_a_repeated_text_column_is_sampled():
    col = column("status", distinct_estimate=4.0)
    assert samplable(col, 6) is True


def test_a_column_that_differs_on_every_row_is_not_sampled():
    """Identifiers and contact details look like this: one value per row.

    They carry no vocabulary for the planner, and they are the columns that leak.
    """
    col = column("reference_number", distinct_estimate=6.0)
    assert samplable(col, 6) is False


def test_near_unique_columns_are_still_skipped():
    col = column("email", distinct_estimate=999.0)
    assert samplable(col, 1000) is False


def test_a_key_is_never_sampled():
    col = column("id", data_type="uuid", primary_key=True, distinct_estimate=1.0)
    assert samplable(col, 6) is False


def test_a_non_text_column_is_never_sampled():
    col = column("amount", data_type="numeric", type_category="N", distinct_estimate=4.0)
    assert samplable(col, 6) is False


def test_a_single_valued_column_is_not_sampled():
    col = column("country", distinct_estimate=1.0)
    assert samplable(col, 6) is False


def test_a_wide_vocabulary_is_not_sampled():
    col = column("summary", distinct_estimate=500.0)
    assert samplable(col, 5000) is False


def test_the_denylist_blocks_a_low_cardinality_column(monkeypatch):
    """Cardinality alone cannot catch a sensitive column that happens to repeat."""
    monkeypatch.setattr(settings, "value_sample_denylist", ["Nationality"])
    col = column("nationality", distinct_estimate=3.0)
    assert samplable(col, 100) is False


def test_the_denylist_is_case_insensitive(monkeypatch):
    monkeypatch.setattr(settings, "value_sample_denylist", ["NATIONALITY"])
    assert samplable(column("nationality", distinct_estimate=3.0), 100) is False


def test_an_empty_denylist_blocks_nothing(monkeypatch):
    monkeypatch.setattr(settings, "value_sample_denylist", [])
    assert samplable(column("status", distinct_estimate=3.0), 100) is True


def test_a_sensitive_marker_in_the_comment_blocks_the_column(monkeypatch):
    monkeypatch.setattr(settings, "value_sample_sensitive_marker", "Protected PII")
    col = column(
        "identity_number",
        distinct_estimate=3.0,
        description="Government identity number (Protected PII).",
    )
    assert samplable(col, 100) is False


def test_the_marker_check_is_case_insensitive(monkeypatch):
    monkeypatch.setattr(settings, "value_sample_sensitive_marker", "protected pii")
    col = column("x", distinct_estimate=3.0, description="Identity (Protected PII).")
    assert samplable(col, 100) is False


def test_an_empty_marker_disables_the_comment_check(monkeypatch):
    monkeypatch.setattr(settings, "value_sample_sensitive_marker", "")
    col = column("x", distinct_estimate=3.0, description="Identity (Protected PII).")
    assert samplable(col, 100) is True


def test_an_ordinary_comment_does_not_block_the_column(monkeypatch):
    monkeypatch.setattr(settings, "value_sample_sensitive_marker", "Protected PII")
    col = column("status", distinct_estimate=3.0, description="Current stage of the case.")
    assert samplable(col, 100) is True


def test_an_unknown_row_estimate_does_not_block_the_column():
    """Without a row count there is nothing to compare distinctness against."""
    assert samplable(column("status", distinct_estimate=3.0), 0) is True


# --- vector width from the catalog ------------------------------------------


@pytest.mark.parametrize(
    ("declared", "expected"),
    [("vector(1536)", 1536), ("vector(384)", 384), ("vector", None), ("vector()", None)],
)
def test_vector_width_is_read_from_the_declared_type(declared, expected):
    assert _vector_width(declared) == expected


@pytest.mark.parametrize("declared", ["text", "numeric(15,2)", "character varying(50)"])
def test_a_non_vector_type_has_no_width(declared):
    assert _vector_width(declared) is None


def test_a_malformed_width_is_ignored():
    assert _vector_width("vector(abc)") is None


# --- the fallback refuses a mismatched column -------------------------------


def graph_with_vector(table, dimensions):
    return SchemaGraph(
        tables={
            table: TableInfo(
                name=table,
                row_estimate=10,
                embedding_columns=["embedding"],
                columns=[
                    ColumnInfo(name="title", data_type="text"),
                    ColumnInfo(name="embedding", data_type="text", dimensions=dimensions),
                ],
            )
        }
    )


def test_a_matching_width_is_accepted():
    assert fallback._compatible(graph_with_vector("docs", 384), "docs", 384) is True


def test_an_unconstrained_column_is_accepted():
    assert fallback._compatible(graph_with_vector("docs", None), "docs", 384) is True


def test_a_mismatched_width_is_refused(monkeypatch):
    monkeypatch.setattr(fallback, "_reported_mismatch", set())
    assert fallback._compatible(graph_with_vector("docs", 1536), "docs", 384) is False


def test_the_mismatch_is_reported_once(monkeypatch, caplog):
    reported: set[str] = set()
    monkeypatch.setattr(fallback, "_reported_mismatch", reported)
    graph = graph_with_vector("docs", 1536)
    with caplog.at_level("WARNING"):
        for _ in range(3):
            fallback._compatible(graph, "docs", 384)
    assert len([r for r in caplog.records if "Vector fallback skipped" in r.message]) == 1


# --- the fallback statement --------------------------------------------------


async def test_the_distance_operator_reaches_the_statement(monkeypatch, ecommerce):
    from app import execute

    captured = {}

    async def fake_run(sql, *args):
        captured["sql"] = sql
        captured["args"] = args
        return []

    monkeypatch.setattr(execute, "run", fake_run)
    graph = SchemaGraph(
        tables={
            "orders": TableInfo(
                name="orders",
                row_estimate=10,
                embedding_columns=["embedding"],
                columns=[
                    ColumnInfo(name="order_reference", data_type="text"),
                    ColumnInfo(name="embedding", data_type="text", dimensions=384),
                ],
            )
        }
    )

    await fallback._query(graph, "orders", "embedding", [0.1, 0.2, 0.3])

    assert "<=>" in captured["sql"]
    assert "IS NOT DISTINCT FROM" not in captured["sql"]
    # The vector is bound, not inlined, and the column is not selected back.
    assert captured["args"] == ("[0.10000000,0.20000000,0.30000000]",)
    assert '"embedding",' not in captured["sql"].split("AS similarity")[0]
    assert "embedding" not in captured["sql"].split("FROM")[0].split('"embedding" <=>')[0]
