from core.constants import SCALAR_INTENTS
from core.plan import QueryIntent
from utils.logging_config import get_logger

logger = get_logger("answer")


class _FormatDict(dict):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.missing: list[str] = []

    def __missing__(self, key: str) -> str:
        self.missing.append(key)
        return ""


def bind_template(
    template: str,
    records: list[dict],
    intent: QueryIntent | None = None,
    *,
    max_chars: int,
) -> str:
    if not template or not template.strip():
        logger.debug("template: empty template provided, returning empty string")
        return ""

    context: dict[str, object] = {}
    scalar = intent in SCALAR_INTENTS and len(records) == 1 and len(records[0]) == 1

    if scalar:
        value = next(iter(records[0].values()))
        context["row_count"] = value
        context["count"] = value
    else:
        context["row_count"] = len(records)
        context["count"] = len(records)

    if records:
        for key, value in records[0].items():
            context[str(key)] = value
            context[str(key).lower()] = value

    logger.debug(
        "template: context prepared with %d variables: %s", len(context), list(context.keys())
    )
    bindings = _FormatDict(context)
    logger.debug("template: binding %r to %d row(s)", template[:max_chars], len(records))
    try:
        filled = template.format_map(bindings).strip()
    except Exception as exc:
        logger.warning("Template could not be filled: %s; using the no-answer message.", exc)
        return ""

    if bindings.missing:
        logger.warning(
            "Template names %d unbound placeholder(s) such as {%s}; using the no-answer message.",
            len(bindings.missing),
            bindings.missing[0],
        )
        return ""

    if len(filled) > max_chars:
        logger.debug(
            "template: output length %d exceeded max %d chars; truncating",
            len(filled),
            max_chars,
        )
        filled = filled[:max_chars]
    logger.debug("template: bound %d chars: %s", len(filled), filled)
    return filled or ""
