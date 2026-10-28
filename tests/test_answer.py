"""The answer layer: template binding and narration row-block assembly."""

import json

from app.answer import SCALAR_INTENTS, _trim, narrate, render_template
from app.types import QueryIntent

COUNT_ROWS = [{"record_count": 3}]
FETCH_ROWS = [{"order_reference": "R-1"}, {"order_reference": "R-2"}]


class Result:
    def __init__(self, text):
        self.content = text


class Chat:
    def __init__(self, text="two orders"):
        self.text = text
        self.calls = []

    async def ainvoke(self, messages, **_kwargs):
        self.calls.append(messages)
        return Result(self.text)


def _bind(template, records, intent):
    return render_template(template, records, intent)


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

    A model will invent a placeholder that names nothing in the result. An
    empty string is the signal for the caller to narrate the rows instead;
    echoing "{nope}" would present template markup as if it were the answer.
    """
    assert _bind("{nope}", FETCH_ROWS, QueryIntent.retrieve) == ""


def test_one_unbound_placeholder_discards_the_whole_template():
    rows = [{"status": "open"}]
    assert _bind("Open: {status} ({nope})", rows, QueryIntent.retrieve) == ""


def test_bound_placeholder_alongside_none_still_renders():
    rows = [{"status": "open"}]
    assert _bind("Status {status}.", rows, QueryIntent.retrieve) == "Status open."


def test_empty_template_renders_empty():
    assert render_template("", FETCH_ROWS, QueryIntent.retrieve) == ""
    assert render_template("   ", FETCH_ROWS, QueryIntent.retrieve) == ""


def test_scalar_intents_are_exactly_the_ones_that_collapse_to_one_number():
    assert set(SCALAR_INTENTS) == {QueryIntent.count, QueryIntent.aggregate}


# --- narration row block -----------------------------------------------------


def _records(n):
    return [{"ref": f"R-{i}", "amount": float(i), "note": "x" * 40} for i in range(n)]


def test_trimmed_rows_form_valid_json_that_matches_the_original():
    records = _records(20)
    for budget in (200, 5000, 10**9):
        kept, truncated = _trim(records, budget)
        block = "[" + ",".join(kept) + "]"
        assert json.loads(block) == records[: len(kept)]
        assert truncated is (len(kept) < len(records))


def test_trim_always_keeps_at_least_one_record():
    kept, truncated = _trim(_records(5), 1)
    assert len(kept) == 1
    assert truncated is True


async def test_narrate_sends_parseable_rows(monkeypatch):
    import app.answer as answer_module

    records = _records(12)
    chat = Chat("twelve orders")
    monkeypatch.setattr(answer_module, "chat_model", lambda: chat)

    assert await narrate("how many orders", records) == "twelve orders"

    rows_section = chat.calls[0][1][1].split("ROWS (")[1].split("\n", 1)[1]
    payload = rows_section[rows_section.index("[") : rows_section.rindex("]") + 1]
    assert json.loads(payload) == records


async def test_narrate_on_no_records_uses_the_configured_message(monkeypatch):
    from app.config import settings

    assert await narrate("anything", []) == settings.no_records_message
