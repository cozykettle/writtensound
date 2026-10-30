"""All /api/* routes (docs/03 §6) plus the SSE stream."""

from __future__ import annotations

import asyncio
import contextlib
import json
import mimetypes
import os
import re
import shutil
import uuid
from pathlib import Path
from typing import Annotated, Any
from urllib.parse import unquote

import aiofiles
from fastapi import APIRouter, Body, File, Header, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse, Response, StreamingResponse
from pydantic import BaseModel, ConfigDict, ValidationError

from ..config import AppConfig, SettingsUpdate, save_config
from ..hardware import probe_hardware
from ..jobs import ApiError, JobSettings, job_out, validate_settings
from ..maintenance import cleanup_all
from ..pipeline.prep import probe_media
from .common import (
    aligner_ready,
    available_devices,
    catalog_payload,
    default_settings_for,
    enabled_engines,
    get_state,
    settings_payload,
)

router = APIRouter()

ALLOWED_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".m4a",
    ".ogg",
    ".opus",
    ".flac",
    ".aac",
    ".wma",
    ".aiff",
    ".mp4",
    ".mkv",
    ".mov",
    ".webm",
    ".avi",
    ".mpg",
    ".mpeg",
    ".wmv",
}
SAFE_NAME_RE = re.compile(r"[^\w\-. ()\[\]]", re.UNICODE)


class JobItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    upload_id: str
    settings: JobSettings


class JobCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[JobItem]


@router.get("/api/status")
async def status(request: Request) -> JSONResponse:
    state = get_state(request)
    return JSONResponse(state.hardware.model_dump(mode="json"))


@router.post("/api/status/refresh")
async def refresh_status(request: Request) -> JSONResponse:
    state = get_state(request)

    async def reprobe() -> None:
        fresh = await asyncio.to_thread(probe_hardware, state.env)
        state.hardware = fresh
        state.hub.publish("hardware", fresh.model_dump(mode="json"))

    asyncio.create_task(reprobe())
    return JSONResponse(state.hardware.model_dump(mode="json"))


@router.get("/api/settings")
async def get_settings(request: Request) -> JSONResponse:
    return JSONResponse(settings_payload(get_state(request)))


@router.put("/api/settings")
async def put_settings(request: Request, payload: SettingsUpdate) -> JSONResponse:
    state = get_state(request)
    updates = payload.model_dump(exclude_unset=True, mode="json")
    if "engine_enabled" in updates:
        unknown = set(updates["engine_enabled"]) - {engine.key for engine in state.catalog.engines}
        if unknown:
            raise ApiError("invalid_settings", "Unknown engine key(s).", {"engines": sorted(unknown)})
    merged = state.config.model_dump(mode="json")
    merged.update(updates)
    try:
        new_config = AppConfig.model_validate(merged)
    except ValidationError as exc:
        raise ApiError("invalid_settings", "Settings could not be saved.", {"errors": exc.errors()}) from exc
    save_config(state.paths, new_config)
    old_flags = state.engine_flags()
    state.config = new_config
    for engine in state.catalog.visible_engines:
        enabled = state.engine_flags().get(engine.key, True)
        if old_flags.get(engine.key) != enabled:
            state.hub.publish("engine", {"engine": engine.key, "enabled": enabled})
    return JSONResponse(settings_payload(state))


@router.post("/api/uploads")
async def create_upload(
    request: Request,
    file: Annotated[UploadFile, File()],
    x_relative_path: Annotated[str | None, Header()] = None,
    x_client_id: Annotated[str | None, Header()] = None,
) -> JSONResponse:
    state = get_state(request)
    upload_id = await _store_upload(state, file, x_relative_path, x_client_id)
    return JSONResponse({"upload_id": upload_id})


@router.post("/api/uploads/legacy", response_class=HTMLResponse)
async def legacy_upload(request: Request, file: Annotated[UploadFile, File()]) -> Response:
    state = get_state(request)
    upload_id = await _store_upload(state, file, None, None)
    settings = default_settings_for(state)
    try:
        job_ids = await _create_jobs(state, [JobItem(upload_id=upload_id, settings=settings)])
    except ApiError:
        return RedirectResponse("/?legacy_error=1", status_code=303)
    return RedirectResponse(f"/?queued={job_ids[0]}", status_code=303)


@router.post("/api/jobs")
async def create_jobs(request: Request, payload: JobCreateRequest) -> JSONResponse:
    state = get_state(request)
    if not payload.items:
        raise ApiError("invalid_settings", "Add at least one file to the queue.")
    job_ids = await _create_jobs(state, payload.items)
    jobs = [
        job_out(row, state.paths.jobs)
        for row in (state.jobs.get(job_id) for job_id in job_ids)
        if row is not None
    ]
    return JSONResponse({"job_ids": job_ids, "jobs": jobs})


@router.get("/api/jobs")
async def list_jobs(
    request: Request,
    status: str | None = None,
    limit: int = 100,
    offset: int = 0,
) -> JSONResponse:
    state = get_state(request)
    if status and status not in {"pending", "running", "succeeded", "failed", "cancelled", "interrupted"}:
        raise ApiError("invalid_settings", "Unknown status filter.")
    limit = max(1, min(500, limit))
    offset = max(0, offset)
    rows = state.jobs.list_jobs(status=status, limit=limit, offset=offset)
    return JSONResponse({"jobs": [job_out(row, state.paths.jobs) for row in rows]})


@router.get("/api/jobs/{job_id}")
async def get_job(request: Request, job_id: str) -> JSONResponse:
    state = get_state(request)
    row = state.jobs.get(job_id)
    if row is None:
        raise ApiError("not_found", "Job not found.")
    return JSONResponse(job_out(row, state.paths.jobs))


@router.patch("/api/jobs/{job_id}")
async def patch_job(
    request: Request, job_id: str, payload: Annotated[dict[str, Any], Body()]
) -> JSONResponse:
    state = get_state(request)
    row = state.jobs.get(job_id)
    if row is None:
        raise ApiError("not_found", "Job not found.")
    if row["status"] != "pending":
        raise ApiError("conflict", "Only pending jobs can be edited.")
    current = JobSettings.model_validate_json(row["settings"]).model_dump(mode="json")
    merged = _deep_merge(current, payload)
    try:
        settings = JobSettings.model_validate(merged)
        validate_settings(
            settings,
            state.catalog,
            available_devices=available_devices(state),
            enabled_engines=enabled_engines(state),
            aligner_ready=aligner_ready(state),
        )
    except ValidationError as exc:
        raise ApiError("invalid_settings", "The job settings are not valid.", {"errors": exc.errors()}) from exc
    state.jobs.update(job_id, settings=settings.model_dump(mode="json"))
    updated = state.jobs.get(job_id)
    assert updated is not None
    state.hub.publish("job", job_out(updated, state.paths.jobs))
    return JSONResponse(job_out(updated, state.paths.jobs))


@router.delete("/api/jobs/{job_id}")
async def delete_job(request: Request, job_id: str) -> Response:
    state = get_state(request)
    row = state.jobs.get(job_id)
    if row is None:
        raise ApiError("not_found", "Job not found.")
    if row["status"] == "running":
        state.runner.cancel(job_id)
        await asyncio.to_thread(state.runner.wait_for, job_id, 15.0)
        row = state.jobs.get(job_id)
        if row and row["status"] == "running":
            raise ApiError("conflict", "The job is still stopping; try again in a moment.")
    elif row["status"] == "pending":
        with contextlib.suppress(ApiError):
            state.runner.cancel(job_id)
    state.jobs.delete(job_id)
    state.hub.publish("queue", state.runner.queue_snapshot())
    return Response(status_code=204)


@router.post("/api/jobs/{job_id}/cancel")
async def cancel_job(request: Request, job_id: str) -> JSONResponse:
    state = get_state(request)
    state.runner.cancel(job_id)
    row = state.jobs.get(job_id)
    if row is None:
        raise ApiError("not_found", "Job not found.")
    return JSONResponse(job_out(row, state.paths.jobs))


@router.post("/api/jobs/{job_id}/retry")
async def retry_job(request: Request, job_id: str) -> JSONResponse:
    state = get_state(request)
    state.runner.retry(job_id)
    row = state.jobs.get(job_id)
    if row is None:
        raise ApiError("not_found", "Job not found.")
    return JSONResponse(job_out(row, state.paths.jobs))


@router.post("/api/jobs/{job_id}/fallback-cpu")
async def fallback_cpu(request: Request, job_id: str) -> JSONResponse:
    state = get_state(request)
    row = state.jobs.get(job_id)
    if row is None:
        raise ApiError("not_found", "Job not found.")
    error = json.loads(row["error"]) if row["error"] else {}
    if error.get("code") != "unsupported_device":
        raise ApiError("conflict", "Fallback to CPU is only offered for device failures.")
    state.runner.retry(job_id, force_cpu=True)
    updated = state.jobs.get(job_id)
    assert updated is not None
    return JSONResponse(job_out(updated, state.paths.jobs))


@router.get("/api/jobs/{job_id}/outputs")
async def outputs(request: Request, job_id: str) -> JSONResponse:
    state = get_state(request)
    row = state.jobs.get(job_id)
    if row is None:
        raise ApiError("not_found", "Job not found.")
    return JSONResponse(job_out(row, state.paths.jobs)["outputs_manifest"])


@router.get("/api/jobs/{job_id}/outputs/{name}")
async def download_output(request: Request, job_id: str, name: str) -> FileResponse:
    state = get_state(request)
    row = state.jobs.get(job_id)
    if row is None:
        raise ApiError("not_found", "Job not found.")
    manifest = job_out(row, state.paths.jobs)["outputs_manifest"]
    if not any(item["name"] == name for item in manifest):
        raise ApiError("not_found", "Output file not found.")
    path = state.paths.job_dir(job_id) / "out" / name
    return FileResponse(path, filename=name, media_type=mimetypes.guess_type(name)[0] or "application/octet-stream")


@router.get("/api/jobs/{job_id}/logs/{name}")
async def job_log(request: Request, job_id: str, name: str) -> FileResponse:
    state = get_state(request)
    if name not in {"pipeline.log", "engine-asr.log", "engine-diar.log"}:
        raise ApiError("not_found", "Log file not found.")
    path = state.paths.job_dir(job_id) / "logs" / name
    if not path.exists() or not path.is_file():
        raise ApiError("not_found", "Log file not found.")
    return FileResponse(path, media_type="text/plain; charset=utf-8")


@router.get("/api/catalog")
async def catalog(request: Request) -> JSONResponse:
    return JSONResponse(catalog_payload(get_state(request)))


@router.post("/api/models/{engine_key}/{model_id}/download")
async def download_model(request: Request, engine_key: str, model_id: str) -> JSONResponse:
    state = get_state(request)
    row = state.downloader.start(engine_key, model_id)
    return JSONResponse(row)


@router.post("/api/models/{engine_key}/{model_id}/cancel")
async def cancel_download(request: Request, engine_key: str, model_id: str) -> JSONResponse:
    state = get_state(request)
    return JSONResponse(state.downloader.cancel(engine_key, model_id))


@router.post("/api/models/{engine_key}/{model_id}/remove")
async def remove_model(request: Request, engine_key: str, model_id: str) -> JSONResponse:
    state = get_state(request)
    return JSONResponse(state.downloader.remove(engine_key, model_id))


@router.post("/api/models/scan")
async def scan_models(request: Request) -> JSONResponse:
    state = get_state(request)
    state.downloader.scan()
    return JSONResponse(catalog_payload(state))


@router.post("/api/engines/{engine_key}/enable")
async def enable_engine(request: Request, engine_key: str) -> JSONResponse:
    return await _set_engine(request, engine_key, True)


@router.post("/api/engines/{engine_key}/disable")
async def disable_engine(request: Request, engine_key: str) -> JSONResponse:
    return await _set_engine(request, engine_key, False)


@router.post("/api/maintenance/cleanup")
async def maintenance_cleanup(request: Request) -> JSONResponse:
    state = get_state(request)
    result = await asyncio.to_thread(cleanup_all, state)
    return JSONResponse(result)


@router.get("/api/events")
async def events(request: Request) -> StreamingResponse:
    state = get_state(request)
    # Subscribe before reading the snapshot: anything published while the
    # snapshot is being built lands in this queue and is delivered right after
    # it, instead of falling into the gap between "initial state" and "live".
    queue = state.hub.subscribe()
    try:
        snapshot = state.snapshot()
    except Exception:
        state.hub.unsubscribe(queue)
        raise

    async def stream() -> Any:
        payload = json.dumps(snapshot, ensure_ascii=False, separators=(",", ":"))
        try:
            yield f"event: snapshot\ndata: {payload}\n\n"
            async for chunk in state.hub.stream(queue):
                yield chunk
        finally:
            state.hub.unsubscribe(queue)

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


async def _set_engine(request: Request, engine_key: str, enabled: bool) -> JSONResponse:
    state = get_state(request)
    engine = state.catalog.engine_or_none(engine_key)
    if engine is None:
        raise ApiError("not_found", "Engine not found.")
    new_config = state.config.model_copy(
        update={"engine_enabled": {**state.config.engine_enabled, engine_key: enabled}}
    )
    save_config(state.paths, new_config)
    state.config = new_config
    state.hub.publish("engine", {"engine": engine_key, "enabled": enabled})
    return JSONResponse({"engine": engine_key, "enabled": enabled})


async def _store_upload(
    state: Any,
    file: UploadFile,
    relative_path: str | None,
    client_id: str | None,
) -> str:
    original = Path(file.filename or "upload").name
    suffix = Path(original).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise ApiError("unsupported_media", f"Unsupported file type {suffix or '(none)'}. Use audio or video.")
    safe = SAFE_NAME_RE.sub("_", original).strip(" .") or "upload"
    upload_id = uuid.uuid4().hex
    target = state.paths.staging / f"{upload_id}_{safe}"
    total = 0
    try:
        async with aiofiles.open(target, "wb") as handle:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > state.env.max_upload_bytes:
                    raise ApiError(
                        "unsupported_media",
                        f"File exceeds the {state.env.max_upload_bytes // (1024**3)} GB upload limit.",
                    )
                await handle.write(chunk)
        if total == 0:
            raise ApiError("unsupported_media", "The uploaded file is empty.")
        probe_media(target)
    except ApiError:
        target.unlink(missing_ok=True)
        state.hub.publish("upload", {"upload_id": upload_id, "state": "failed"})
        raise
    except Exception as exc:
        target.unlink(missing_ok=True)
        raise ApiError("storage", f"Upload could not be written: {exc}") from exc

    clean_relative = _safe_relative(relative_path) if relative_path else None
    state.uploads.create(
        upload_id=upload_id,
        original_name=original,
        relative_path=clean_relative,
        stored_path=str(target),
        size_bytes=total,
        content_type=file.content_type,
        client_file_id=client_id,
    )
    state.uploads.set_state(upload_id, "ready")
    state.hub.publish("upload", {"upload_id": upload_id, "state": "ready"})
    return upload_id


async def _create_jobs(state: Any, items: list[JobItem]) -> list[str]:
    errors: list[dict[str, Any]] = []
    seen: set[str] = set()
    rows: list[Any] = []
    for index, item in enumerate(items):
        try:
            if item.upload_id in seen:
                raise ApiError("invalid_settings", "The same upload appears twice.")
            seen.add(item.upload_id)
            upload = state.uploads.get(item.upload_id)
            if upload is None or upload["state"] not in {"staging", "ready"}:
                raise ApiError("not_found", "Upload not found or already queued.")
            validate_settings(
                item.settings,
                state.catalog,
                available_devices=available_devices(state),
                enabled_engines=enabled_engines(state),
                aligner_ready=aligner_ready(state),
            )
            rows.append(upload)
        except ApiError as exc:
            errors.append({"index": index, "error": exc.payload().model_dump()})
    if errors:
        raise ApiError("invalid_settings", "Some items could not be queued.", {"items_errors": errors})

    created: list[str] = []
    cleanup_paths: list[Path] = []
    try:
        for item, upload in zip(items, rows, strict=True):
            job_id = uuid.uuid4().hex
            job_dir = state.paths.job_dir(job_id)
            media_dir = job_dir / "media"
            media_dir.mkdir(parents=True, exist_ok=True)
            source = Path(upload["stored_path"])
            if not source.exists():
                raise ApiError("storage", "The staged upload is missing.")
            destination = media_dir / f"original{source.suffix.lower()}"
            try:
                os.link(source, destination)
            except OSError:
                shutil.copy2(source, destination)
            state.jobs.create(
                job_id=job_id,
                title=upload["original_name"],
                settings=item.settings.model_dump(mode="json"),
            )
            state.uploads.set_state(item.upload_id, "consumed")
            created.append(job_id)
            if state.config.cleanup_uploads:
                cleanup_paths.append(source)
            state.runner.enqueue(job_id)
    except ApiError:
        _rollback_jobs(state, created, [item.upload_id for item in items])
        raise
    except Exception as exc:
        _rollback_jobs(state, created, [item.upload_id for item in items])
        raise ApiError("storage", f"Could not create the job: {exc}") from exc
    for path in cleanup_paths:
        path.unlink(missing_ok=True)
    return created


def _rollback_jobs(state: Any, job_ids: list[str], upload_ids: list[str]) -> None:
    for job_id in job_ids:
        state.jobs.delete(job_id)
        shutil.rmtree(state.paths.job_dir(job_id), ignore_errors=True)
    for upload_id in upload_ids:
        row = state.uploads.get(upload_id)
        if row and row["state"] == "consumed":
            state.uploads.set_state(upload_id, "ready")
    state.hub.publish("queue", state.runner.queue_snapshot())


def _safe_relative(value: str) -> str:
    cleaned = unquote(value).replace("\\", "/").lstrip("/")
    parts = [part for part in cleaned.split("/") if part not in {"", ".", ".."}]
    return "/".join(parts)[:500]


def _deep_merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged
