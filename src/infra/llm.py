import json
import time
from typing import TYPE_CHECKING, Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from core.constants import (
    COMPLETION_TOKENS_FIELD,
    FINISH_REASON_FIELD,
    FINISH_REASON_LENGTH,
    JSON_INSTRUCTION,
    LLM_MAX_TOKENS,
    LLM_RETRIES,
    LLM_TEMPERATURE,
    LLM_TIMEOUT_SECONDS,
    LOG_LLM_PAYLOADS,
    METADATA_SOURCES,
    PROMPT_TOKENS_FIELD,
    REASONING_BUDGET_WARNING,
    REASONING_EMPTY_RESPONSE,
    REASONING_FIELD,
    SERVED_MODEL_FIELD,
    USAGE_FIELD,
)
from core.exceptions import ConfigurationError, PlanningError, ProviderError
from utils.logging_config import get_logger

if TYPE_CHECKING:
    from core.config import Settings

logger = get_logger("llm")


class LLMClient:
    def __init__(self, settings: "Settings") -> None:
        self._settings = settings
        self._model: BaseChatModel | None = None

    def chat_model(self) -> BaseChatModel:
        if not self._settings.llm_api_key:
            raise ConfigurationError("LLM_API_KEY is not configured.")
        if not self._settings.llm_base_url:
            raise ConfigurationError("LLM_BASE_URL is not configured.")
        if not self._settings.llm_model:
            raise ConfigurationError("LLM_MODEL is not configured.")
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
            logger.debug("llm: ChatOpenAI client initialized")
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

        if LOG_LLM_PAYLOADS:
            logger.debug("llm payload: raw content received (%d chars): %s", len(content), content)

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
        logger.debug("extract_json: text is empty")
        raise PlanningError("The model returned an empty response.")
    logger.debug("extract_json: searching for JSON object in %d chars", len(text))
    start = 0
    while True:
        start = text.find("{", start)
        if start == -1:
            break
        end = _balanced_end(text, start)
        if end is not None:
            parsed = _loads(text[start : end + 1])
            if isinstance(parsed, dict):
                logger.debug(
                    "extract_json: extracted JSON dict (%d keys: %s)",
                    len(parsed),
                    list(parsed.keys()),
                )
                return parsed
        start += 1
    logger.debug("extract_json: failed to find valid JSON object")
    raise PlanningError("The model response contained no JSON object.")


def _loads(text: str):
    logger.debug("_loads: attempting json.loads on %d chars", len(text))
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
                logger.debug("_balanced_end: found closing bracket at index %d", index)
                return index
    return None


def provider_metadata(result: Any) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for source_name in METADATA_SOURCES:
        source = getattr(result, source_name, None)
        if isinstance(source, dict):
            merged.update(source)
    logger.debug(
        "provider_metadata: extracted %d metadata keys: %s",
        len(merged),
        list(merged.keys()),
    )
    return merged


def _model_content(result: Any) -> str:
    content = getattr(result, "content", None)
    text = content if isinstance(content, str) else ""
    logger.debug("_model_content: extracted %d chars", len(text))
    return text


def _reasoning(meta: dict[str, Any]) -> str:
    scratchpad = meta.get(REASONING_FIELD)
    text = scratchpad if isinstance(scratchpad, str) else ""
    logger.debug("_reasoning: extracted %d chars", len(text))
    return text


def _as_int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _empty_response_message(finish_reason: Any, reasoning: str) -> str:
    logger.debug(
        "_empty_response_message: finish_reason=%s, reasoning_chars=%d",
        finish_reason,
        len(reasoning),
    )
    if reasoning:
        return REASONING_BUDGET_WARNING
    if finish_reason == FINISH_REASON_LENGTH:
        return REASONING_EMPTY_RESPONSE
    return "The model returned an empty response."
