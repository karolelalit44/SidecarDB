from core.constants import (
    DEFAULT_SCHEMA,
    ROLE_IDLE_IN_TRANSACTION_TIMEOUT_MS,
    ROLE_STATEMENT_TIMEOUT_MS,
)
from core.exceptions import ConfigurationError, SecurityError
from utils.logging_config import get_logger
from utils.sql import is_safe_identifier, quote_identifier, quote_literal

logger = get_logger("roles")


def _validated_identifier(value: str, what: str) -> str:
    if not is_safe_identifier(value):
        raise SecurityError(
            f"{what} must start with a letter or underscore and use only letters, "
            f"digits, '_' or '$' (got {value!r})"
        )
    return value


def _role_exists_sql(role_literal: str, password_literal: str) -> str:
    return (
        "DO $block$\n"
        "BEGIN\n"
        f"    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = {role_literal}) THEN\n"
        "        EXECUTE format('CREATE ROLE %I WITH LOGIN PASSWORD %L', "
        f"{role_literal}, {password_literal});\n"
        "    END IF;\n"
        "END\n"
        "$block$;"
    )


def _provision_statements(role: str, password: str, schema: str, database: str) -> list[str]:
    target = quote_identifier(role)
    scope = quote_identifier(schema)
    exists = _role_exists_sql(quote_literal(role), quote_literal(password))
    revoke = "INSERT, UPDATE, DELETE, TRUNCATE, REFERENCES, TRIGGER"
    default_grant = f"ALTER DEFAULT PRIVILEGES IN SCHEMA {scope} GRANT SELECT ON TABLES TO {target}"
    default_revoke = (
        f"ALTER DEFAULT PRIVILEGES IN SCHEMA {scope} REVOKE {revoke} ON TABLES FROM {target}"
    )
    return [
        exists,
        f"GRANT CONNECT ON DATABASE {quote_identifier(database)} TO {target}",
        f"GRANT USAGE ON SCHEMA {scope} TO {target}",
        f"GRANT SELECT ON ALL TABLES IN SCHEMA {scope} TO {target}",
        default_grant,
        f"REVOKE {revoke} ON ALL TABLES IN SCHEMA {scope} FROM {target}",
        default_revoke,
        f"ALTER ROLE {target} SET default_transaction_read_only = on",
        f"ALTER ROLE {target} SET statement_timeout = '{ROLE_STATEMENT_TIMEOUT_MS}ms'",
        (
            f"ALTER ROLE {target} SET idle_in_transaction_session_timeout = "
            f"'{ROLE_IDLE_IN_TRANSACTION_TIMEOUT_MS}ms'"
        ),
        f"ALTER ROLE {target} SET search_path = {scope}",
    ]


async def create_read_only_role(
    admin_dsn: str,
    role: str = "",
    password: str = "",
    schema: str = DEFAULT_SCHEMA,
) -> None:
    if not admin_dsn:
        raise ConfigurationError("An admin DSN is required to provision the read-only role.")
    if not role:
        raise ConfigurationError("A role name is required for the read-only role.")
    if not password:
        raise ConfigurationError("A password is required for the read-only role.")
    _validated_identifier(role, "The role name")
    _validated_identifier(schema, "The schema name")

    import asyncpg

    try:
        connection = await asyncpg.connect(admin_dsn)
    except Exception as exc:
        raise ConfigurationError(f"Could not connect as the admin: {exc}") from exc

    try:
        database = str(await connection.fetchval("SELECT current_database()"))
        statements = _provision_statements(role, password, schema, database)
        async with connection.transaction():
            for statement in statements:
                logger.debug("roles: %s", statement)
                await connection.execute(statement)
        logger.info(
            "Read-only role %s provisioned on database %s (schema %s).",
            role,
            database,
            schema,
        )
    except Exception as exc:
        raise ConfigurationError(f"Read-only role provisioning failed: {exc}") from exc
    finally:
        await connection.close()
