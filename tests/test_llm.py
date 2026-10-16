import pytest

from app.exceptions import PlanningError
from app.llm import extract_json


def test_plain_object():
    assert extract_json('{"table": "t"}') == {"table": "t"}


def test_fenced_block():
    text = '```json\n{"table": "t", "intent": "count"}\n```'
    assert extract_json(text) == {"table": "t", "intent": "count"}


def test_fenced_without_language_tag():
    assert extract_json('```\n{"a": 1}\n```') == {"a": 1}


def test_preamble_before_object():
    text = 'Here is the plan you asked for:\n{"table": "t"}'
    assert extract_json(text) == {"table": "t"}


def test_trailing_prose_after_object():
    assert extract_json('{"a": 1}\nHope that helps!') == {"a": 1}


def test_braces_inside_strings_do_not_confuse_the_scanner():
    assert extract_json('{"a": "text with } brace", "b": 2}') == {
        "a": "text with } brace",
        "b": 2,
    }


def test_escaped_quote_inside_string():
    assert extract_json('{"a": "he said \\"hi\\""}') == {"a": 'he said "hi"'}


def test_nested_object():
    text = '{"a": {"b": {"c": 1}}}'
    assert extract_json(text) == {"a": {"b": {"c": 1}}}


def test_empty_response_is_rejected():
    with pytest.raises(PlanningError):
        extract_json("")


def test_non_json_is_rejected():
    with pytest.raises(PlanningError):
        extract_json("I cannot answer that.")


def test_truncated_object_is_rejected():
    with pytest.raises(PlanningError):
        extract_json('{"table": "t", "intent":')


def test_json_array_is_not_an_object():
    with pytest.raises(PlanningError):
        extract_json('["a", "b"]')
