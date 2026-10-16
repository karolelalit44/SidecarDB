import json

from .config import settings
from .exceptions import ExecutionError
from .llm import chat_model
from .logging_config import get_logger
from .types import QueryIntent

logger = get_logger("answer")

# Intents whose result is a single number. For these {row_count} must mean the
# computed value, not len(records) -- a COUNT returns one row, so len(records) is
# always 1 and binding it to {row_count} reports "1" for every count question.
SCALAR_INTENTS = (QueryIntent.count, QueryIntent.aggregate)

SYSTEM = """\
You answer a question using only the rows you are given.

Rules:
- Every fact must come from the rows. Never add, infer or embellish.
- If the rows do not answer the question, say exactly that.
- Be brief: two or three sentences, or a short list if the answer is a list.
- Do not mention SQL, tables or columns.
- Do not apologise for the data. State what it shows.
"""

USER = """\
QUESTION
{question}

ROWS ({count})
{rows}
"""


def _encode(records: list[dict]) -> str:
    return json.dumps(records, default=str, ensure_ascii=False)


def _trim(records: list[dict], budget: int) -> tuple[list[str], bool]:
    """Selects the records that fit the budget, returning their JSON fragments.

    Each record is encoded once and the fragments are reused to build the final
    array, so nothing is serialised twice.
    """
    kept: list[str] = []
    used = 0
    for record in records:
        blob = _encode(record)
        if used + len(blob) > budget and kept:
            break
        kept.append(blob)
        used += len(blob)
    return kept, len(kept) < len(records)


async def narrate(question: str, records: list[dict]) -> str:
    if not records:
        return settings.no_records_message

    fragments, truncated = _trim(records, settings.narration_char_budget)
    rows_block = "[" + ",".join(fragments) + "]"
    if truncated:
        rows_block += f"\n[showing {len(fragments)} of {len(records)} matched rows]"

    message = USER.format(question=question, count=len(records), rows=rows_block)

    try:
        result = await chat_model().ainvoke(
            [("system", SYSTEM), ("human", message)],
            max_tokens=settings.narration_max_tokens,
        )
    except Exception as exc:
        logger.error("Narration failed: %s", exc)
        raise ExecutionError("The result was retrieved but could not be summarised.") from exc

    text = (result.content or "").strip()
    return text or settings.no_summary_message


class _FormatDict(dict):
    def __missing__(self, key: str) -> str:
        return f"{{{key}}}"


def render_template(template: str, records: list[dict], intent: QueryIntent | None = None) -> str:
    """Populates the model-decided template with data from the query execution result."""
    if not template or not template.strip():
        return ""

    context: dict[str, object] = {}
    scalar = intent in SCALAR_INTENTS and len(records) == 1 and len(records[0]) == 1

    if scalar:
        value = next(iter(records[0].values()))
        context["row_count"] = value
        context["count"] = value
        context["value"] = value
        context["result"] = value
        context["total"] = value
    else:
        context["row_count"] = len(records)
        context["count"] = len(records)

    if records:
        for key, value in records[0].items():
            context[str(key)] = value
            context[str(key).lower()] = value

    try:
        rendered = template.format_map(_FormatDict(context)).strip()
        if len(rendered) > settings.error_message_max_chars:
            rendered = rendered[: settings.error_message_max_chars]
        return rendered or template
    except Exception:
        return template
