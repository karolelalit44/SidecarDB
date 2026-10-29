PG_IDENTIFIER_MAX_LENGTH = 63
IDENTIFIER_EXTRA_CHARACTERS = frozenset("_$")

__all__ = [
    "IDENTIFIER_EXTRA_CHARACTERS",
    "PG_IDENTIFIER_MAX_LENGTH",
    "is_safe_identifier",
    "quote_identifier",
    "quote_literal",
]


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
