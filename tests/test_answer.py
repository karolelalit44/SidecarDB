"""The answer layer: template binding."""

from core.answer import SCALAR_INTENTS, render_template
from core.types import QueryIntent

COUNT_ROWS = [{"record_count": 3}]
FETCH_ROWS = [{"order_reference": "R-1"}, {"order_reference": "R-2"}]
MAX_CHARS = 200


def _bind(template, records, intent):
    return render_template(template, records, intent, max_chars=MAX_CHARS)


# --- {row_count} on a scalar result -----------------------------------------


def test_count_intent_binds_the_aggregate_not_the_row_number():
    assert _bind("There are {row_count} cases.", COUNT_ROWS, QueryIntent.count) == (
        "There are 3 cases."
    )


def test_aggregate_intent_binds_the_measure():
    rows = [{"sum_of_amount": 99.0}]
    assert _bind("Total was {row_count}.", rows, QueryIntent.aggregate) == "Total was 99.0."


def test_count_of_zero_is_not_treated_as_absent():
    assert _bind("{row_count}", [{"record_count": 0}], QueryIntent.count) == "0"


def test_count_aliases_all_bind_to_the_aggregate():
    for token in ("row_count", "count", "value", "result", "total"):
        assert _bind("{" + token + "}", COUNT_ROWS, QueryIntent.count) == "3", token


def test_retrieve_intent_counts_rows():
    assert _bind("Returned {row_count}.", FETCH_ROWS, QueryIntent.retrieve) == "Returned 2."


def test_retrieve_intent_does_not_treat_a_single_cell_as_a_scalar():
    """A one-column retrieve is not an aggregate, so {row_count} stays the row count."""
    rows = [{"title": "only"}]
    assert _bind("Returned {row_count}.", rows, QueryIntent.retrieve) == "Returned 1."


def test_missing_intent_falls_back_to_row_semantics():
    """Callers that do not know the intent still get the row count."""
    assert _bind("Returned {row_count}.", COUNT_ROWS, None) == "Returned 1."


def test_projected_column_names_are_bound():
    assert _bind("Case {order_reference}.", FETCH_ROWS, QueryIntent.retrieve) == "Case R-1."


def test_unknown_placeholder_is_not_shown_to_the_caller():
    """Raw template syntax must never reach the user.

    A model will invent a placeholder that names nothing in the result. An empty
    string is the signal for the caller to use the configured no-summary
    message; echoing "{nope}" would present template markup as the answer.
    """
    assert _bind("{nope}", FETCH_ROWS, QueryIntent.retrieve) == ""


def test_one_unbound_placeholder_discards_the_whole_template():
    rows = [{"status": "open"}]
    assert _bind("Open: {status} ({nope})", rows, QueryIntent.retrieve) == ""


def test_bound_placeholder_alongside_none_still_renders():
    rows = [{"status": "open"}]
    assert _bind("Status {status}.", rows, QueryIntent.retrieve) == "Status open."


def test_empty_template_renders_empty():
    assert render_template("", FETCH_ROWS, QueryIntent.retrieve, max_chars=MAX_CHARS) == ""
    assert render_template("   ", FETCH_ROWS, QueryIntent.retrieve, max_chars=MAX_CHARS) == ""


def test_rendering_is_capped_at_the_configured_length():
    long_template = "x " * 100
    rendered = render_template(long_template, FETCH_ROWS, QueryIntent.retrieve, max_chars=10)
    assert len(rendered) == 10


def test_scalar_intents_are_exactly_the_ones_that_collapse_to_one_number():
    assert set(SCALAR_INTENTS) == {QueryIntent.count, QueryIntent.aggregate}


# --- the planner is told which placeholders exist ----------------------------


def test_prompt_documents_every_placeholder_the_engine_binds():
    from core.graph import SYSTEM

    for token in ("row_count", "count"):
        assert "{" + token + "}" in SYSTEM, token


def test_prompt_makes_the_template_mandatory():
    from core.graph import SYSTEM

    assert "template: required" in SYSTEM
