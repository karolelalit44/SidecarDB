import argparse
import asyncio
from collections.abc import Sequence

from core.config import Settings
from core.constants import DEFAULT_HOST, DEFAULT_PORT, DEFAULT_SCHEMA
from utils.logging_config import get_logger

logger = get_logger("cli")

__all__ = ["build_parser", "main"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sidecardb",
        description="Natural-language search over a PostgreSQL schema.",
    )
    subparsers = parser.add_subparsers(dest="command")
    serve = subparsers.add_parser("serve", help="Run the HTTP service")
    serve.add_argument("--host", default=None, help="Bind address (default: 0.0.0.0)")
    serve.add_argument("--port", type=int, default=None, help="Bind port (default: 8000)")
    serve.add_argument(
        "--env-file", default=None, help="Read configuration from this file instead of .env"
    )
    roles = subparsers.add_parser(
        "setup-roles",
        help="Create the read-only database role the engine connects as",
    )
    roles.add_argument(
        "--admin-dsn",
        required=True,
        help="DSN of a role that may create roles and grants (e.g. postgres)",
    )
    roles.add_argument(
        "--role",
        default=None,
        help="Name of the role to create (default: DB_ROLE from settings)",
    )
    roles.add_argument(
        "--schema",
        default=DEFAULT_SCHEMA,
        help=f"Schema the role may read from (default: {DEFAULT_SCHEMA})",
    )
    roles.add_argument(
        "--password",
        default=None,
        help="Password for the new role (default: DB_PASSWORD from settings)",
    )
    return parser


def serve(host: str | None, port: int | None, env_file: str | None) -> None:
    import uvicorn

    from core.client import SidecarDB
    from core.server import create_app

    effective_host = host if host is not None else DEFAULT_HOST
    effective_port = port if port is not None else DEFAULT_PORT
    logger.debug(
        "serve command initiated: host=%s port=%s env_file=%s",
        effective_host,
        effective_port,
        env_file,
    )

    sidecardb = SidecarDB(_env_file=env_file)
    app = create_app(sidecardb)
    logger.debug("launching uvicorn runner on %s:%s", effective_host, effective_port)
    uvicorn.run(
        app,
        host=effective_host,
        port=effective_port,
        log_config=None,
    )


def setup_roles(admin_dsn: str, role: str | None, schema: str, password: str | None) -> None:
    from infra.roles import create_read_only_role

    settings = Settings()
    effective_role = role or settings.db_role
    if not effective_role:
        logger.error("No role name: pass --role or set DB_ROLE in settings.")
        raise SystemExit(2)
    effective_password = password or settings.db_password
    if not effective_password:
        logger.error("No role password: pass --password or set DB_PASSWORD in settings.")
        raise SystemExit(2)
    logger.debug("setup_roles command initiated: schema=%s role=%s", schema, effective_role)
    asyncio.run(create_read_only_role(admin_dsn, effective_role, effective_password, schema))


def main(argv: Sequence[str] | None = None) -> int:
    logger.debug("CLI main invoked with argv: %s", argv)
    parser = build_parser()
    args = parser.parse_args(argv)
    logger.debug("CLI parsed command: %s", args.command)
    if args.command == "serve":
        serve(args.host, args.port, args.env_file)
        return 0
    if args.command == "setup-roles":
        setup_roles(args.admin_dsn, args.role, args.schema, args.password)
        return 0
    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
