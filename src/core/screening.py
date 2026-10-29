from unicodedata import category

_DISALLOWED_SUBSTRINGS = ("--", "/*", "*/", ";", "\0")
_ALLOWED_CONTROL = ("\t", "\n", "\r")
_BLOCKED_CATEGORIES = ("Cc", "Cf", "Cs")

__all__ = ["check_question"]


def check_question(text: str) -> str | None:
    if not text:
        return "The question is empty."

    for marker in _DISALLOWED_SUBSTRINGS:
        if marker in text:
            return "The question contains disallowed symbols or comment characters."

    for ch in text:
        if ch in _ALLOWED_CONTROL:
            continue
        if category(ch) in _BLOCKED_CATEGORIES:
            return "The question contains unsupported control or formatting characters."

    if not any(ch.isalnum() for ch in text):
        return "The question must contain at least one alphanumeric character."

    return None
