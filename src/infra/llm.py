import json
import time
from typing import TYPE_CHECKING, Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from core.constants import (
    LLM_MAX_TOKENS,
    LLM_RETRIES,
    LLM_TEMPERATURE,
    LLM_TIMEOUT_SECONDS,
    LOG_LLM_PAYLOADS,
)
from core.exceptions import ConfigurationError, PlanningError, ProviderError
from utils.logging_config import get_logger

if TYPE_CHECKING:
    from core.config import Settings

logger = get_logger("llm")

JSON_INSTRUCTION = (
    "Reply with one JSON object and nothing else. No prose, no explanation, "
    "no markdown code fences. Start with { and end with }."
)

METADATA_SOURCES = ("response_metadata", "additional_kwargs")

FINISH_REASON_FIELD = "finish_reason"
FINISH_REASON_LENGTH = "length"
REASONING_FIELD = "reasoning_content"
USAGE_FIELD = "usage_metadata"
SERVED_MODEL_FIELD = "model_name"
COMPLETION_TOKENS_FIELD = "output_tokens"
PROMPT_TOKENS_FIELD = "input_tokens"

REASONING_BUDGET_WARNING = (
    "The model spent its entire completion budget reasoning and returned no answer. "
    "Raise the completion budget or turn the model's thinking mode off."
)
REASONING_EMPTY_RESPONSE = (
    "The model used its entire completion budget without returning an answer. "
    "Raise the completion budget."
)


class LLMClient:
    def __init__(self, settings: "Settings") -> None:
        self._settings = settings
        self._model: BaseChatModel | None = None

    def chat_model(self) -> BaseChatModel:
        if not self._settings.llm_api_key:
            raise ConfigurationError("LLM_API_KEY is not configured.")
        if self._model is None:
            settings = self._settings
            logger.debug(
                "llm: opening a client for %s at %s", settings.llm_model, settings.llm_base_url
            )
            options: dict[str, Any] = {"max_tokens": LLM_MAX_TOKENS}
            self._model = ChatOpenAI(
                model=settings.llm_model,
                base_url=settings.llm_base_url,
                api_key=SecretStr(settings.llm_api_key),
                temperature=LLM_TEMPERATURE,
                timeout=LLM_TIMEOUT_SECONDS,
                max_retries=LLM_RETRIES,
                extra_body=settings.llm_extra_body or None,
                **options,
            )
        return self._model

    async def complete_json(
        self, system_prompt: str, user_prompt: str, temperature: float | None = None
    ) -> dict[str, Any]:
        settings = self._settings
        model = self.chat_model()
        system_text = f"{system_prompt}\n\n{JSON_INSTRUCTION}"
        started = time.perf_counter()
        effective = LLM_TEMPERATURE if temperature is None else temperature
        runnable = model if temperature is None else model.bind(temperature=effective)

        logger.debug(
            "LLM request model=%s system_chars=%d user_chars=%d total_chars=%d "
            "max_tokens=%d temperature=%s extra_body=%s",
            settings.llm_model,
            len(system_text),
            len(user_prompt),
            len(system_text) + len(user_prompt),
            LLM_MAX_TOKENS,
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
                exc_info=LOG_LLM_PAYLOADS,
            )
            raise ProviderError("The model provider could not be reached.") from exc

        meta = provider_metadata(result)
        usage = meta.get(USAGE_FIELD)
        usage = usage if isinstance(usage, dict) else {}
        content = _model_content(result)
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
                LLM_MAX_TOKENS,
            )
            if LOG_LLM_PAYLOADS and reasoning:
                logger.debug("Reasoning scratchpad: %s", reasoning)
            raise PlanningError(_empty_response_message(finish_reason, reasoning))

        try:
            payload = extract_json(content)
        except PlanningError as exc:
            logger.warning("Unparseable model reply (%d chars): %s", len(content), exc)
            if LOG_LLM_PAYLOADS:
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
    start = 0
    while True:
        start = text.find("{", start)
        if start == -1:
            break
        end = _balanced_end(text, start)
        if end is not None:
            parsed = _loads(text[start : end + 1])
            if isinstance(parsed, dict):
                return parsed
        start += 1
    raise PlanningError("The model response contained no JSON object.")


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


def provider_metadata(result: Any) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for source_name in METADATA_SOURCES:
        source = getattr(result, source_name, None)
        if isinstance(source, dict):
            merged.update(source)
    return merged


def _model_content(result: Any) -> str:
    content = getattr(result, "content", None)
    return content if isinstance(content, str) else ""


def _reasoning(meta: dict[str, Any]) -> str:
    scratchpad = meta.get(REASONING_FIELD)
    return scratchpad if isinstance(scratchpad, str) else ""


def _as_int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _empty_response_message(finish_reason: Any, reasoning: str) -> str:
    if reasoning:
        return REASONING_BUDGET_WARNING
    if finish_reason == FINISH_REASON_LENGTH:
        return REASONING_EMPTY_RESPONSE
    return "The model returned an empty response."
