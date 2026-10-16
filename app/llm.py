import json
from functools import lru_cache
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from .config import settings
from .constants import FENCE, JSON_INSTRUCTION, JSON_TAG
from .exceptions import ConfigurationError, PlanningError, ProviderError


def chat_model() -> BaseChatModel:
    # The credential check stays outside the cache so it runs on every call and a
    # missing key always surfaces, whatever was built earlier.
    if not settings.llm_api_key:
        raise ConfigurationError("LLM_API_KEY must be set in .env")
    return _client()


@lru_cache(maxsize=1)
def _client() -> BaseChatModel:
    """One client for the process, so the HTTP connection to the provider is reused."""
    return ChatOpenAI(
        model=settings.llm_model,
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
        timeout=settings.llm_timeout_seconds,
        max_retries=settings.llm_retries,
    )


def extract_json(text: str) -> dict[str, Any]:
    if not text:
        raise PlanningError("The model returned an empty response.")

    candidate = _strip_fence(text.strip())
    parsed = _loads(candidate)
    if isinstance(parsed, dict):
        return parsed

    start = candidate.find("{")
    while start != -1:
        end = _balanced_end(candidate, start)
        if end is not None:
            parsed = _loads(candidate[start : end + 1])
            if isinstance(parsed, dict):
                return parsed
        start = candidate.find("{", start + 1)

    raise PlanningError("The model response contained no JSON object.")


def _strip_fence(text: str) -> str:
    marker = text.find(FENCE)
    if marker == -1:
        return text
    body = text[marker + len(FENCE) :]
    if body[: len(JSON_TAG)].lower() == JSON_TAG:
        body = body[len(JSON_TAG) :]
    end = body.find(FENCE)
    return (body[:end] if end != -1 else body).strip()


def _loads(text: str):
    try:
        return json.loads(text)
    except ValueError:
        return None


def _balanced_end(text: str, start: int) -> int | None:
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        character = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                return index
    return None


async def complete_json(system_prompt: str, user_prompt: str) -> dict[str, Any]:
    # chat_model() is outside the try so ConfigurationError keeps its own meaning.
    model = chat_model()
    try:
        result = await model.ainvoke(
            [
                SystemMessage(content=f"{system_prompt}\n\n{JSON_INSTRUCTION}"),
                HumanMessage(content=user_prompt),
            ]
        )
    except Exception as exc:
        raise ProviderError("The model provider could not be reached.") from exc
    # Outside the try: a malformed reply is a planning fault, not a transport fault.
    return extract_json(result.content or "")
