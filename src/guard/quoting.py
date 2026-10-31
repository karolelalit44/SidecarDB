from config.constants import IDENTIFIER_EXTRA_CHARACTERS, PG_IDENTIFIER_MAX_LENGTH
from core.exceptions import SecurityError
from core.logging import get_logger

logger = get_logger("guard.quoting")


def is_safe_identifier(name: str) -> bool:
    if not name or len(name) > PG_IDENTIFIER_MAX_LENGTH:
        return False
    if not (name[0].isalpha() or name[0] == "_"):
        return False
    return all(
        character.isalnum() or character in IDENTIFIER_EXTRA_CHARACTERS for character in name
    )


def quote_identifier(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def quote_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def quote(name: str) -> str:
    logger.debug("guard quote: identifier=%r", name)
    if not is_safe_identifier(name):
        logger.debug("guard: refusing to compose the identifier %r", name)
        raise SecurityError("The Guard cannot write this identifier safely.")
    return quote_identifier(name)


def quote_column_ref(name: str) -> str:
    if "." in name:
        table_part, col_part = name.split(".", 1)
        return f"{quote(table_part)}.{quote(col_part)}"
    return quote(name)


def like_pattern(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace("%", r"\%").replace("_", r"\_")
    logger.debug("guard like_pattern: escaped %r -> %r", value, escaped)
    return escaped


__all__ = [
    "is_safe_identifier",
    "like_pattern",
    "quote",
    "quote_column_ref",
    "quote_identifier",
    "quote_literal",
]
