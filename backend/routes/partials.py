"""htmx fragments."""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, Response

from ..jobs import JobSettings
from .common import (
    default_settings_for,
    form_settings,
    get_state,
    settings_form_context,
)

router = APIRouter()


@router.get("/partials/settings-form", response_class=HTMLResponse)
async def settings_form(
    request: Request,
    job_id: str | None = None,
    mode: str = "queue",
) -> Response:
    state = get_state(request)
    if job_id:
        row = state.jobs.get(job_id)
        base = JobSettings.model_validate_json(row["settings"]) if row else default_settings_for(state)
        mode = "job"
    else:
        base = default_settings_for(state)
    settings = form_settings(request, base)
    context = settings_form_context(request, state, settings, mode=mode, job_id=job_id)
    return state.templates.TemplateResponse(
        request=request,
        name="partials/settings_form.html",
        context=context,
    )
