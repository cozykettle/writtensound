"""Console entrypoint: ``writtensound`` runs the FastAPI app with uvicorn."""

from __future__ import annotations

import argparse

import uvicorn

from .config import EnvConfig


def main(argv: list[str] | None = None) -> None:
    env = EnvConfig.from_env()
    parser = argparse.ArgumentParser(prog="writtensound", description="Run the WrittenSound web app.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1).")
    parser.add_argument("--port", type=int, default=env.port, help=f"Port (default: {env.port}).")
    parser.add_argument("--reload", action="store_true", help="Restart workers when source files change (development).")
    args = parser.parse_args(argv)
    uvicorn.run(
        "backend.app:create_app",
        factory=True,
        host=args.host,
        port=args.port,
        log_level=env.log_level.lower(),
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
