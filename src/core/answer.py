from core.plan import QueryIntent
from utils.logging_config import get_logger

logger = get_logger("answer")

SCALAR_INTENTS = (QueryIntent.count, QueryIntent.aggregate)


class _FormatDict(dict):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.missing: list[str] = []

    def __missing__(self, key: str) -> str:
        self.missing.append(key)
        return ""


def render_template(
    template: str,
    records: list[dict],
    intent: QueryIntent | None = None,
    *,
    max_chars: int,
) -> str:
    if not template or not template.strip():
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

    bindings = _FormatDict(context)
    logger.debug("template: binding %r to %d row(s)", template[:max_chars], len(records))
    try:
        rendered = template.format_map(bindings).strip()
    except Exception:
        logger.warning("Template could not be rendered; using the no-summary message.")
        return ""

    if bindings.missing:
        logger.warning(
            "Template names %d unbound placeholder(s) such as {%s}; using the no-summary message.",
            len(bindings.missing),
            bindings.missing[0],
        )
        return ""

    if len(rendered) > max_chars:
        rendered = rendered[:max_chars]
    logger.debug("template: rendered %d chars: %s", len(rendered), rendered)
    return rendered or ""
