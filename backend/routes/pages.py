"""Server-rendered pages."""

from __future__ import annotations

import json

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from starlette.exceptions import HTTPException as StarletteHTTPException

from .common import (
    default_settings_for,
    engine_payload,
    get_state,
    jobs_payload,
    page_context,
    settings_form_context,
)

router = APIRouter()


@router.get("/healthz")
async def healthz(request: Request) -> JSONResponse:
    state = get_state(request)
    ok = state.db.ping()
    return JSONResponse({"ok": ok}, status_code=200 if ok else 503)


@router.get("/favicon.ico")
async def favicon() -> Response:
    return Response(status_code=204)


@router.get("/", response_class=HTMLResponse)
async def queue_page(request: Request) -> Response:
    state = get_state(request)
    settings = default_settings_for(state)
    context = settings_form_context(request, state, settings, mode="queue")
    settings = context["settings"]
    jobs = jobs_payload(state)
    context.update(
        {
            "jobs": jobs,
            "settings_json": json.dumps(settings.model_dump(mode="json")),
            "job_count": len(jobs),
        }
    )
    context["active"] = "queue"
    return state.templates.TemplateResponse(request=request, name="queue.html", context=context)


@router.get("/engines", response_class=HTMLResponse)
async def engines_page(request: Request) -> Response:
    state = get_state(request)
    context = page_context(request, state, "engines")
    context.update(
        {
            "engines": [engine_payload(state, engine) for engine in state.catalog.visible_engines],
            "aligner_model": state.catalog.aligner_model(),
            "settings_json": json.dumps(state.config.defaults.model_dump(mode="json")),
        }
    )
    return state.templates.TemplateResponse(request=request, name="engines.html", context=context)


@router.get("/settings", response_class=HTMLResponse)
async def settings_page(request: Request) -> Response:
    state = get_state(request)
    context = settings_form_context(request, state, state.config.defaults, mode="defaults")
    settings = context["settings"]
    context.update(
        {
            "settings_json": json.dumps(settings.model_dump(mode="json")),
        }
    )
    context["active"] = "settings"
    return state.templates.TemplateResponse(request=request, name="settings.html", context=context)


async def http_error(request: Request, exc: StarletteHTTPException) -> Response:
    state = get_state(request)
    context = page_context(request, state, "")
    context.update({"status": exc.status_code, "detail": exc.detail})
    return state.templates.TemplateResponse(
        request=request,
        name="error.html",
        context=context,
        status_code=exc.status_code,
    )
