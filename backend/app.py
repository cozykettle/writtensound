"""FastAPI application factory and boot sequence (docs/03 §7)."""

from __future__ import annotations

import asyncio
import base64
import logging
import secrets
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.exceptions import HTTPException as StarletteHTTPException

from . import __version__
from .catalog import Catalog, load_catalog
from .config import AppConfig, EnvConfig, Paths, effective_hf_token, load_config, mask_token
from .db import Database, JobStore, ModelStateStore, UploadStore
from .downloader import DownloadManager, hf_home_env
from .events import EventHub
from .hardware import HardwareStatus, probe_hardware
from .jobs import ApiError, job_out
from .maintenance import cleanup_all
from .renderers import hhmmss
from .routes import api, pages, partials
from .runner import JobRunner

log = logging.getLogger(__name__)
BASE_DIR = Path(__file__).resolve().parent


@dataclass
class AppState:
    env: EnvConfig
    paths: Paths
    catalog: Catalog
    config: AppConfig
    db: Database
    jobs: JobStore
    uploads: UploadStore
    models: ModelStateStore
    hub: EventHub
    hardware: HardwareStatus
    downloader: DownloadManager
    runner: JobRunner
    templates: Jinja2Templates

    def engine_flags(self) -> dict[str, bool]:
        return {engine.key: self.config.engine_enabled.get(engine.key, True) for engine in self.catalog.visible_engines}

    def model_state_map(self) -> dict[str, list[dict[str, Any]]]:
        result: dict[str, list[dict[str, Any]]] = {}
        for engine in self.catalog.visible_engines:
            result[engine.key] = [self.downloader.state_row(model) for model in engine.models]
        return result

    def snapshot(self) -> dict[str, Any]:
        jobs = [job_out(row, self.paths.jobs) for row in self.jobs.all_for_snapshot()]
        return {
            "hardware": self.hardware.model_dump(mode="json"),
            "jobs": jobs,
            "model_state": self.model_state_map(),
            "engine_enabled": self.engine_flags(),
        }


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _basic_auth_ok(request: Request, password: str) -> bool:
    header = request.headers.get("authorization", "")
    if not header.lower().startswith("basic "):
        return False
    try:
        decoded = base64.b64decode(header.split(" ", 1)[1]).decode("utf-8")
    except (ValueError, UnicodeDecodeError):
        return False
    _, _, supplied = decoded.partition(":")
    return secrets.compare_digest(supplied, password)


def _status_for(code: str) -> int:
    return {
        "invalid_settings": 400,
        "unsupported_media": 415,
        "not_found": 404,
        "conflict": 409,
        "unsupported_device": 400,
        "hf_auth": 400,
        "engine_crash": 500,
        "ffmpeg_failed": 500,
        "download_failed": 502,
        "storage": 500,
        "internal": 500,
    }.get(code, 500)


async def _periodic_cleanup(state: AppState) -> None:
    while True:
        try:
            await asyncio.sleep(3600)
            await asyncio.to_thread(cleanup_all, state)
        except asyncio.CancelledError:
            return
        except Exception:
            log.exception("periodic cleanup failed")


def create_app() -> FastAPI:
    env = EnvConfig.from_env()
    _configure_logging(env.log_level)
    paths = Paths.build(env.data_dir, env.hf_home)
    paths.create_dirs()
    hf_home_env(paths)

    catalog = load_catalog()
    if not env.demo_engine_enabled:
        catalog = catalog.model_copy(
            update={"engines": [engine for engine in catalog.engines if engine.key != "demo"]}
        )
    config = load_config(paths, catalog)
    db = Database(paths.db_file)
    db.initialize()
    jobs = JobStore(db)
    uploads = UploadStore(db)
    models = ModelStateStore(db)
    hub = EventHub()
    hardware = probe_hardware(env)
    for note in hardware.notes:
        log.info("hardware: %s", note)

    templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
    templates.env.filters["filesize"] = _filesize
    templates.env.filters["clock"] = hhmmss
    templates.env.globals["version"] = __version__
    static_root = BASE_DIR / "static"
    asset_mtime = max((int(path.stat().st_mtime) for path in static_root.iterdir()), default=0)
    templates.env.globals["asset_version"] = f"{__version__}.{asset_mtime}"
    templates.env.globals["masked_token"] = mask_token

    state = AppState(
        env=env,
        paths=paths,
        catalog=catalog,
        config=config,
        db=db,
        jobs=jobs,
        uploads=uploads,
        models=models,
        hub=hub,
        hardware=hardware,
        downloader=None,  # type: ignore[arg-type]
        runner=None,  # type: ignore[arg-type]
        templates=templates,
    )
    state.downloader = DownloadManager(
        paths=paths,
        catalog=catalog,
        states=models,
        hub=hub,
        token_provider=lambda: effective_hf_token(state.config, env),
    )
    state.downloader.scan()
    state.runner = JobRunner(
        store=jobs,
        paths=paths,
        env=env,
        catalog=catalog,
        config_provider=lambda: state.config,
        hardware_provider=lambda: state.hardware,
        hub=hub,
        downloader=state.downloader,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        hub.bind()
        state.runner.start()
        cleanup_task = asyncio.create_task(_periodic_cleanup(state))
        try:
            yield
        finally:
            cleanup_task.cancel()
            state.runner.stop()

    app = FastAPI(title="WrittenSound", version=__version__, lifespan=lifespan)
    app.state.app_state = state
    app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

    app.include_router(pages.router)
    app.include_router(partials.router)
    app.include_router(api.router)

    @app.exception_handler(ApiError)
    async def api_error_handler(request: Request, exc: ApiError) -> JSONResponse:
        return JSONResponse(status_code=_status_for(exc.code), content=exc.payload().model_dump())

    @app.exception_handler(RequestValidationError)
    async def validation_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=400,
            content={
                "code": "invalid_settings",
                "message": "Some fields need attention.",
                "details": {"errors": jsonable_encoder(exc.errors())},
            },
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_handler(request: Request, exc: StarletteHTTPException) -> Response:
        if request.url.path.startswith("/api/"):
            code = "not_found" if exc.status_code == 404 else "internal"
            return JSONResponse(status_code=exc.status_code, content={"code": code, "message": str(exc.detail)})
        return await pages.http_error(request, exc)

    @app.middleware("http")
    async def auth_middleware(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        if (
            state.env.app_password
            and request.url.path != "/healthz"
            and not _basic_auth_ok(request, state.env.app_password)
        ):
            return Response(
                status_code=401,
                headers={"WWW-Authenticate": 'Basic realm="WrittenSound"'},
                content="WrittenSound is protected. Use any username and the app password.",
            )
        return await call_next(request)

    return app


def _filesize(value: int | float | None) -> str:
    if not value:
        return "—"
    size = float(value)
    for suffix in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or suffix == "TB":
            return f"{size:.0f} {suffix}" if suffix == "B" else f"{size:.2f} {suffix}"
        size /= 1024
    return f"{size:.2f} TB"


def get_state(request: Request) -> AppState:
    state: AppState = request.app.state.app_state
    return state
