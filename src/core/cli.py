import argparse
import asyncio
import os
from collections.abc import Sequence

from core.constants import DEFAULT_HOST, DEFAULT_PORT, DEFAULT_ROLE_NAME
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
        default=DEFAULT_ROLE_NAME,
        help=f"Name of the role to create (default: {DEFAULT_ROLE_NAME})",
    )
    roles.add_argument(
        "--schema",
        default="public",
        help="Schema the role may read from (default: public)",
    )
    roles.add_argument(
        "--password",
        default=os.environ.get("DB_PASSWORD"),
        help="Password for the new role (default: DB_PASSWORD environment variable)",
    )
    return parser


def serve(host: str | None, port: int | None, env_file: str | None) -> None:
    import uvicorn

    from core.client import SidecarDB
    from core.server import create_app

    sidecardb = SidecarDB(_env_file=env_file) if env_file else SidecarDB()
    app = create_app(sidecardb)
    uvicorn.run(
        app,
        host=host if host is not None else DEFAULT_HOST,
        port=port if port is not None else DEFAULT_PORT,
        log_config=None,
    )


def setup_roles(admin_dsn: str, role: str, schema: str, password: str | None) -> None:
    from infra.roles import create_read_only_role

    if not password:
        logger.error("No role password: pass --password or set DB_PASSWORD in the environment.")
        raise SystemExit(2)
    asyncio.run(create_read_only_role(admin_dsn, role, password, schema))


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
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
