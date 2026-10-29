import argparse
from collections.abc import Sequence

from core.logging_config import get_logger

logger = get_logger("cli")

__all__ = ["build_parser", "main"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="askdb",
        description="Natural-language search over a PostgreSQL schema.",
    )
    subparsers = parser.add_subparsers(dest="command")
    serve = subparsers.add_parser("serve", help="Run the HTTP service")
    serve.add_argument("--host", default=None, help="Bind address (default: HOST or 0.0.0.0)")
    serve.add_argument("--port", type=int, default=None, help="Bind port (default: PORT or 8000)")
    serve.add_argument(
        "--env-file", default=None, help="Read configuration from this file instead of .env"
    )
    return parser


def serve(host: str | None, port: int | None, env_file: str | None) -> None:
    import uvicorn

    from core.client import AskDB
    from core.server import create_app

    # Passing _env_file=None would disable the default .env discovery; only
    # forward the keyword when --env-file was actually given.
    askdb = AskDB(_env_file=env_file) if env_file else AskDB()
    settings = askdb.settings
    app = create_app(askdb)
    uvicorn.run(
        app,
        host=host if host is not None else settings.host,
        port=port if port is not None else settings.port,
        # The service configures its own logging; uvicorn must not replace it.
        log_config=None,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "serve":
        serve(args.host, args.port, args.env_file)
        return 0
    parser.print_help()
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
