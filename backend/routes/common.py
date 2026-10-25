"""Shared route helpers."""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any, cast

from fastapi import Request

from ..catalog import EngineSpec, ModelSpec, engine_available
from ..config import mask_token
from ..jobs import JobSettings, OutputFormats, job_out, settings_warnings

if TYPE_CHECKING:
    from ..app import AppState


def get_state(request: Request) -> AppState:
    return cast("AppState", request.app.state.app_state)


def available_devices(state: AppState) -> set[str]:
    return set(state.hardware.devices.keys())


def aligner_ready(state: AppState) -> bool:
    model = state.catalog.aligner_model()
    return bool(model and state.downloader.is_ready(model))


def enabled_engines(state: AppState) -> set[str]:
    return {key for key, enabled in state.engine_flags().items() if enabled}


def model_payload(state: AppState, model: ModelSpec) -> dict[str, Any]:
    engine = state.catalog.engine(model.engine)
    return {
        "id": model.id,
        "name": model.name,
        "role": model.role,
        "engine": model.engine,
        "repo": model.repo,
        "size_bytes": model.size_bytes,
        "size_label": model.size_label,
        "languages": model.languages,
        "language_auto": model.language_auto,
        "english_only": model.english_only,
        "word_ts": model.word_ts,
        "max_speakers": model.max_speakers,
        "gated": model.gated,
        "notes": model.notes,
        "default": model.default,
        "supported_devices": list(engine.devices),
        "state": state.downloader.state_row(model),
    }


def engine_payload(state: AppState, engine: EngineSpec) -> dict[str, Any]:
    return {
        "key": engine.key,
        "name": engine.name,
        "runtime": engine.runtime,
        "kind": engine.kind,
        "devices": list(engine.devices),
        "available": engine_available(engine),
        "enabled": state.engine_flags().get(engine.key, True),
        "license": engine.license,
        "notes": engine.notes,
        "models": [model_payload(state, model) for model in engine.models],
    }


def catalog_payload(state: AppState) -> dict[str, Any]:
    return {
        "engines": [engine_payload(state, engine) for engine in state.catalog.visible_engines],
        "hardware": state.hardware.model_dump(mode="json"),
        "aligner_ready": aligner_ready(state),
        "engine_enabled": state.engine_flags(),
    }


def settings_payload(state: AppState) -> dict[str, Any]:
    payload: dict[str, Any] = state.config.model_dump(mode="json")
    token = payload.get("hf_token") or state.env.hf_token_env
    payload["hf_token"] = mask_token(token)
    return payload


def page_context(request: Request, state: AppState, active: str) -> dict[str, Any]:
    return {
        "request": request,
        "active": active,
        "hardware": state.hardware,
        "paths": state.paths,
        "env": state.env,
        "catalog": state.catalog,
        "config": state.config,
        "version": state.templates.env.globals["version"],
        "engine_flags": state.engine_flags(),
        "engine_available": {engine.key: engine_available(engine) for engine in state.catalog.visible_engines},
        "model_states": state.model_state_map(),
        "aligner_ready": aligner_ready(state),
        "defaults": state.config.defaults,
        "masked_token": mask_token(state.config.hf_token or state.env.hf_token_env),
        "model_state_by_id": {
            model.id: state.downloader.state_row(model)
            for engine in state.catalog.visible_engines
            for model in engine.models
        },
        "max_upload_bytes": state.env.max_upload_bytes,
        "model_names": {
            model.id: model.name
            for engine in state.catalog.visible_engines
            for model in engine.models
        },
    }


def jobs_payload(state: AppState) -> list[dict[str, Any]]:
    return [job_out(row, state.paths.jobs) for row in state.jobs.all_for_snapshot()]


def warnings_for(state: AppState, settings: JobSettings) -> list[str]:
    warnings = settings_warnings(settings, state.catalog)
    model = state.catalog.model_or_none(settings.asr_model_id)
    if model:
        engine = state.catalog.engine(model.engine)
        if not engine_available(engine):
            warnings.insert(0, f"The {engine.name} runtime is not installed in this environment.")
        if not state.downloader.is_ready(model):
            warnings.insert(0, f"{model.name} is not downloaded yet; the queue will fetch it on first run.")
    if settings.diar != "off" and settings.diar_model_id:
        diar = state.catalog.model_or_none(settings.diar_model_id)
        if diar and not state.downloader.is_ready(diar):
            warnings.append(f"{diar.name} is not downloaded yet; the queue will fetch it on first run.")
    if settings.word_timestamps and model and model.word_ts == "aligner":
        aligner = state.catalog.aligner_model()
        if aligner and not state.downloader.is_ready(aligner):
            warnings.append("The Qwen3 ForcedAligner model is not downloaded yet.")
    return warnings


def parse_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "on", "yes"}


def form_settings(request: Request, base: JobSettings) -> JobSettings:
    params = request.query_params
    formats = params.getlist("formats")
    values = base.model_dump(mode="json")
    if params.get("asr_model_id"):
        values["asr_model_id"] = params["asr_model_id"]
    if params.get("device") in {"auto", "cpu", "cuda", "vulkan"}:
        values["device"] = params["device"]
    if params.get("language"):
        values["language"] = params["language"]
    if params.get("diar") in {"off", "nemo", "pyannote"}:
        values["diar"] = params["diar"]
    if params.get("diar_model_id") is not None:
        values["diar_model_id"] = params.get("diar_model_id") or None
    if params.get("speaker_count"):
        raw = params["speaker_count"]
        values["speaker_count"] = "auto" if raw == "auto" else _clamp_int(raw, 2, 8, 2)
    if params.get("timestamps_mode") in {"none", "interval", "turns"}:
        values["timestamps_mode"] = params["timestamps_mode"]
    if params.get("interval_s"):
        values["interval_s"] = _clamp_int(params["interval_s"], 10, 3600, 60)
    if "word_timestamps" in params or params.get("word_timestamps_present"):
        values["word_timestamps"] = parse_bool(params.get("word_timestamps"))
    if "mark_pauses" in params or params.get("mark_pauses_present"):
        values["mark_pauses"] = parse_bool(params.get("mark_pauses"))
    if params.get("pause_min_s"):
        with contextlib.suppress(ValueError):
            values["pause_min_s"] = max(0.5, min(10.0, float(params["pause_min_s"])))
    if "verbatim" in params or params.get("verbatim_present"):
        values["verbatim"] = parse_bool(params.get("verbatim"))
    values["start_at"] = params.get("start_at") or None
    values["end_at"] = params.get("end_at") or None
    values["out_basename"] = params.get("out_basename") or None
    if params.get("formats_present"):
        chosen = set(formats)
        values["output"] = {
            "html": "html" in chosen,
            "txt": "txt" in chosen,
            "srt": "srt" in chosen,
            "vtt": "vtt" in chosen,
            "timed_txt": "timed_txt" in chosen,
            "json": "json" in chosen,
        }
    try:
        return JobSettings.model_validate(values)
    except ValueError:
        return base


def _clamp_int(value: str, minimum: int, maximum: int, fallback: int) -> int:
    try:
        return max(minimum, min(maximum, int(value)))
    except ValueError:
        return fallback


def auto_device_hint(state: AppState, engine: EngineSpec) -> str:
    order: list[str] = []
    for slug, label in (("vulkan", "Vulkan"), ("cuda", "CUDA")):
        if slug in engine.devices and slug in state.hardware.devices:
            order.append(label)
    order.append("CPU")
    return ", then ".join(order)


def default_settings_for(state: AppState) -> JobSettings:
    defaults = state.config.defaults
    model = state.catalog.model_or_none(defaults.asr_model_id)
    if model is None:
        model = state.catalog.default_asr_model()
        return defaults.model_copy(update={"asr_model_id": model.id, "language": "auto"})
    return defaults


def settings_form_context(
    request: Request,
    state: AppState,
    base: JobSettings,
    *,
    mode: str,
    job_id: str | None = None,
) -> dict[str, Any]:
    """Everything the shared settings fragment needs, from one place."""
    settings = base
    model = state.catalog.model_or_none(settings.asr_model_id)
    if model is None:
        settings = default_settings_for(state)
        model = state.catalog.model_or_none(settings.asr_model_id)
    assert model is not None
    engine = state.catalog.engine(model.engine)
    if settings.diar.value != "off":
        available_diar = state.catalog.diar_models_for(settings.diar.value)
        if available_diar and settings.diar_model_id not in {item.id for item in available_diar}:
            settings = settings.model_copy(update={"diar_model_id": available_diar[0].id})
    diar_model = state.catalog.model_or_none(settings.diar_model_id) if settings.diar_model_id else None
    aligner = state.catalog.aligner_model()
    aligned = bool(aligner and aligner_ready(state))
    context = page_context(request, state, "")
    warnings = warnings_for(state, settings)
    context.update(
        {
            "settings": settings,
            "model": model,
            "engine": engine,
            "diar_model": diar_model,
            "diar_models": state.catalog.diar_models_for(settings.diar.value),
            "warnings": warnings,
            "mode": mode,
            "job_id": job_id,
            "language_options": state.catalog.language_options(model),
            "available_devices": set(state.hardware.devices.keys()),
            "word_ts_available": model.word_ts == "native" or (model.word_ts == "aligner" and aligned),
            "verbatim_available": state.catalog.is_whisper_family(model),
            "captions_available": settings.timestamps_mode != "none",
            "speaker_max": diar_model.max_speakers if diar_model and diar_model.max_speakers else 8,
            "form_id": "job-settings-form" if mode == "queue" else f"job-settings-form-{mode}",
            "runtime_available": engine_available(engine),
            "auto_device_hint": auto_device_hint(state, engine),
        }
    )
    return context


def output_payload(formats: OutputFormats) -> dict[str, bool]:
    return formats.model_dump()
