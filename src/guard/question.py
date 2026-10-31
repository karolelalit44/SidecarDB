from unicodedata import category

from config.constants import (
    ALLOWED_CONTROL,
    BLOCKED_CATEGORIES,
    DISALLOWED_SUBSTRINGS,
    MAX_QUESTION_LENGTH,
    MIN_QUESTION_LENGTH,
)
from core.exceptions import QuestionError
from core.logging import get_logger

logger = get_logger("guard.question")


def validate_question(raw_question: str) -> str:
    question = (raw_question or "").strip()
    if not question or len(question) < MIN_QUESTION_LENGTH:
        logger.debug(
            "question rejected: length %d < MIN_QUESTION_LENGTH (%d)",
            len(question),
            MIN_QUESTION_LENGTH,
        )
        raise QuestionError("The question is too short.")
    if len(question) > MAX_QUESTION_LENGTH:
        logger.debug(
            "question rejected: length %d > MAX_QUESTION_LENGTH (%d)",
            len(question),
            MAX_QUESTION_LENGTH,
        )
        raise QuestionError(f"The question is too long (maximum {MAX_QUESTION_LENGTH} characters).")
    for marker in DISALLOWED_SUBSTRINGS:
        if marker in question:
            logger.warning(
                "Rejected a %d-character question: disallowed symbols (%r)",
                len(question),
                marker,
            )
            raise QuestionError("The question contains disallowed symbols or comment characters.")
    for ch in question:
        if ch in ALLOWED_CONTROL:
            continue
        if category(ch) in BLOCKED_CATEGORIES:
            logger.warning(
                "Rejected question: unsupported control character %r (category %s)",
                ch,
                category(ch),
            )
            raise QuestionError(
                "The question contains unsupported control or formatting characters."
            )
    if not any(ch.isalnum() for ch in question):
        logger.warning("Rejected question: no alphanumeric character")
        raise QuestionError("The question must contain at least one alphanumeric character.")
    logger.debug("question validated successfully: %r (%d chars)", question, len(question))
    return question


__all__ = ["validate_question"]
