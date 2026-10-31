from planner.models import QueryIntent
from synthesis.formatter import bind_template


def test_bind_template_scalar_count():
    records = [{"count": 42}]
    result = bind_template(
        "There are {count} items in total.",
        records,
        intent=QueryIntent.count,
        max_chars=200,
    )
    assert result == "There are 42 items in total."


def test_bind_template_record_fields():
    records = [{"name": "Widget", "price": 19.99}]
    result = bind_template(
        "Product {name} costs ${price}.",
        records,
        intent=QueryIntent.retrieve,
        max_chars=200,
    )
    assert result == "Product Widget costs $19.99."


def test_bind_template_missing_placeholder_returns_empty():
    records = [{"name": "Widget"}]
    result = bind_template(
        "Product {name} costs ${missing_field}.",
        records,
        intent=QueryIntent.retrieve,
        max_chars=200,
    )
    assert result == ""


def test_bind_template_max_chars_truncation():
    records = [{"text": "A" * 100}]
    result = bind_template(
        "Output: {text}",
        records,
        intent=QueryIntent.retrieve,
        max_chars=20,
    )
    assert len(result) <= 20
