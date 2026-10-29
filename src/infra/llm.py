import json
import time
from typing import TYPE_CHECKING, Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from core.constants import (
    COMPLETION_TOKENS_FIELD,
    FENCE,
    FINISH_REASON_FIELD,
    FINISH_REASON_LENGTH,
    JSON_INSTRUCTION,
    JSON_TAG,
    METADATA_SOURCES,
    PROMPT_TOKENS_FIELD,
    REASONING_BUDGET_WARNING,
    REASONING_EMPTY_RESPONSE,
    REASONING_FIELD,
    SERVED_MODEL_FIELD,
    TRUNCATION_MARK,
    USAGE_FIELD,
)
from core.exceptions import ConfigurationError, PlanningError, ProviderError
from core.logging_config import get_logger

if TYPE_CHECKING:
    from core.config import Settings

logger = get_logger("llm")


class LLMClient:
    """One lazily-opened chat client per instance, so connections are reused."""

    def __init__(self, settings: "Settings") -> None:
        self._settings = settings
        self._model: BaseChatModel | None = None

    def chat_model(self) -> BaseChatModel:
        # Outside the cache, so a missing key always surfaces.
        if not self._settings.llm_api_key:
            raise ConfigurationError("LLM_API_KEY is not configured.")
        if self._model is None:
            settings = self._settings
            logger.debug(
                "llm: opening a client for %s at %s", settings.llm_model, settings.llm_base_url
            )
            # ChatOpenAI accepts max_tokens at runtime (verified against its
            # model_fields), but released stubs drift on that field; routing
            # the drift-prone option through an Any-typed mapping keeps the
            # call type-clean against any future stub revision.
            options: dict[str, Any] = {"max_tokens": settings.llm_max_tokens}
            self._model = ChatOpenAI(
                model=settings.llm_model,
                base_url=settings.llm_base_url,
                api_key=SecretStr(settings.llm_api_key),
                temperature=settings.llm_temperature,
                timeout=settings.llm_timeout_seconds,
                max_retries=settings.llm_retries,
                extra_body=settings.llm_extra_body or None,
                **options,
            )
        return self._model

    async def complete_json(
        self, system_prompt: str, user_prompt: str, temperature: float | None = None
    ) -> dict[str, Any]:
        settings = self._settings
        # Outside the try, so ConfigurationError keeps its own meaning.
        model = self.chat_model()
        system_text = f"{system_prompt}\n\n{JSON_INSTRUCTION}"
        started = time.perf_counter()
        effective = settings.llm_temperature if temperature is None else temperature
        runnable = model if temperature is None else model.bind(temperature=effective)

        logger.debug(
            "LLM request model=%s system_chars=%d user_chars=%d total_chars=%d "
            "max_tokens=%d temperature=%s extra_body=%s",
            settings.llm_model,
            len(system_text),
            len(user_prompt),
            len(system_text) + len(user_prompt),
            settings.llm_max_tokens,
            effective,
            sorted(settings.llm_extra_body),
        )

        try:
            result = await runnable.ainvoke(
                [
                    SystemMessage(content=system_text),
                    HumanMessage(content=user_prompt),
                ]
            )
        except Exception as exc:
            logger.error(
                "LLM transport failure after %d ms: %s: %s",
                int((time.perf_counter() - started) * 1000),
                type(exc).__name__,
                exc,
                exc_info=settings.log_llm_payloads,
            )
            raise ProviderError("The model provider could not be reached.") from exc

        meta = provider_metadata(result)
        usage = meta.get(USAGE_FIELD)
        usage = usage if isinstance(usage, dict) else {}
        content = _answer(result)
        reasoning = _reasoning(meta)
        finish_reason = meta.get(FINISH_REASON_FIELD)

        logger.debug(
            "LLM reply in %d ms: finish=%s served_model=%s content_chars=%d "
            "reasoning_chars=%d prompt_tokens=%s completion_tokens=%s",
            int((time.perf_counter() - started) * 1000),
            finish_reason,
            meta.get(SERVED_MODEL_FIELD),
            len(content),
            len(reasoning),
            _as_int(usage.get(PROMPT_TOKENS_FIELD)),
            _as_int(usage.get(COMPLETION_TOKENS_FIELD)),
        )

        if not content:
            logger.warning(
                "Empty model reply: finish=%s reasoning_chars=%d prompt_tokens=%s "
                "completion_tokens=%s of max_tokens=%d",
                finish_reason,
                len(reasoning),
                _as_int(usage.get(PROMPT_TOKENS_FIELD)),
                _as_int(usage.get(COMPLETION_TOKENS_FIELD)),
                settings.llm_max_tokens,
            )
            if settings.log_llm_payloads and reasoning:
                logger.debug("Reasoning scratchpad: %s", reasoning)
            raise PlanningError(_empty_response_message(finish_reason, reasoning))

        try:
            payload = extract_json(content)
        except PlanningError as exc:
            logger.warning("Unparseable model reply (%d chars): %s", len(content), exc)
            if settings.log_llm_payloads:
                logger.debug("Raw reply: %s", content)
            raise

        logger.debug(
            "llm: reply parsed into %d field(s): %s",
            len(payload),
            ", ".join(sorted(payload)),
        )
        return payload


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


def prompt_chars(reserve: int, settings: "Settings") -> int:
    """Characters available for a prompt body on this model."""
    return max(settings.llm_context_tokens - reserve, 0) * settings.llm_chars_per_token


def trim(text: str, limit: int) -> str:
    """Shortens text to limit characters, marking that it was cut."""
    if len(text) <= limit:
        return text
    if limit <= len(TRUNCATION_MARK):
        return text[:limit]
    return text[: limit - len(TRUNCATION_MARK)] + TRUNCATION_MARK


def provider_metadata(result: Any) -> dict[str, Any]:
    """Flatten reply metadata from response_metadata or additional_kwargs."""
    merged: dict[str, Any] = {}
    for source_name in METADATA_SOURCES:
        source = getattr(result, source_name, None)
        if isinstance(source, dict):
            merged.update(source)
    return merged


def _answer(result: Any) -> str:
    content = getattr(result, "content", None)
    return content if isinstance(content, str) else ""


def _reasoning(meta: dict[str, Any]) -> str:
    scratchpad = meta.get(REASONING_FIELD)
    return scratchpad if isinstance(scratchpad, str) else ""


def _as_int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _empty_response_message(finish_reason: Any, reasoning: str) -> str:
    """Generate descriptive error message for an empty model response."""
    if reasoning:
        return REASONING_BUDGET_WARNING
    if finish_reason == FINISH_REASON_LENGTH:
        return REASONING_EMPTY_RESPONSE
    return "The model returned an empty response."
