"""JobSettings schema, validation, persistence helpers and API serializers."""

from __future__ import annotations

import json
import re
import sqlite3
import warnings
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .catalog import Catalog, ModelSpec

TIMECODE_RE = re.compile(r"^\d{1,2}:\d{2}:\d{2}$")


class TimestampsMode(StrEnum):
    none = "none"
    interval = "interval"
    turns = "turns"


class Diar(StrEnum):
    off = "off"
    nemo = "nemo"
    pyannote = "pyannote"


class Device(StrEnum):
    auto = "auto"
    cpu = "cpu"
    cuda = "cuda"
    vulkan = "vulkan"


# The docs/03 API names the JSON toggle "json"; Pydantic warns because it
# shadows BaseModel.json. Suppress only that exact message, not warnings in
# general.
warnings.filterwarnings(
    "ignore",
    message=r'Field name "json" in "OutputFormats" shadows an attribute in parent "BaseModel"',
    category=UserWarning,
)


class OutputFormats(BaseModel):
    model_config = ConfigDict(extra="forbid")

    html: bool = True
    txt: bool = True
    srt: bool = True
    vtt: bool = False
    timed_txt: bool = False
    json: bool = True  # type: ignore[assignment]  # normative docs/03 field name

    def any(self) -> bool:
        return any((self.html, self.txt, self.srt, self.vtt, self.timed_txt, self.json))


class JobSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    asr_model_id: str
    device: Device = Device.auto
    language: str = "auto"
    diar: Diar = Diar.off
    diar_model_id: str | None = None
    speaker_count: int | Literal["auto"] = "auto"
    timestamps_mode: TimestampsMode = TimestampsMode.interval
    interval_s: int = 60
    word_timestamps: bool = False
    mark_pauses: bool = False
    pause_min_s: float = 2.0
    verbatim: bool = False
    start_at: str | None = None
    end_at: str | None = None
    output: OutputFormats = Field(default_factory=OutputFormats)
    out_basename: str | None = None

    @field_validator("speaker_count")
    @classmethod
    def _clamp_speakers(cls, value: int | Literal["auto"]) -> int | Literal["auto"]:
        if value == "auto":
            return value
        return max(2, min(8, int(value)))

    @field_validator("interval_s")
    @classmethod
    def _interval_range(cls, value: int) -> int:
        if not 10 <= value <= 3600:
            raise ValueError("interval_s must be between 10 and 3600")
        return value

    @field_validator("pause_min_s")
    @classmethod
    def _pause_range(cls, value: float) -> float:
        if not 0.5 <= value <= 10:
            raise ValueError("pause_min_s must be between 0.5 and 10")
        return value

    @field_validator("start_at", "end_at")
    @classmethod
    def _timecode(cls, value: str | None) -> str | None:
        if value in (None, ""):
            return None
        if not TIMECODE_RE.match(value):
            raise ValueError("timecode must be hh:mm:ss")
        hours, minutes, seconds = (int(part) for part in value.split(":"))
        if minutes > 59 or seconds > 59:
            raise ValueError("timecode minutes/seconds must be < 60")
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    @field_validator("out_basename")
    @classmethod
    def _basename(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        if not value:
            return None
        if "/" in value or "\\" in value or ".." in value:
            raise ValueError("out_basename must be a plain filename stem")
        return value

    @model_validator(mode="after")
    def _cross_checks(self) -> JobSettings:
        if not self.output.any():
            raise ValueError("at least one output format must be selected")
        if self.start_at and self.end_at and timecode_seconds(self.end_at) <= timecode_seconds(self.start_at):
            raise ValueError("end_at must be after start_at")
        if self.diar == Diar.off:
            self.diar_model_id = None
        return self

    def is_clipped(self) -> bool:
        return bool(self.start_at or self.end_at)


def timecode_seconds(value: str) -> int:
    hours, minutes, seconds = (int(part) for part in value.split(":"))
    return hours * 3600 + minutes * 60 + seconds


class ApiErrorPayload(BaseModel):
    code: str
    message: str
    details: dict[str, Any] | None = None


class ApiError(Exception):
    """Domain error; routers translate this into the uniform JSON envelope."""

    def __init__(self, code: str, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details

    def payload(self) -> ApiErrorPayload:
        return ApiErrorPayload(code=self.code, message=self.message, details=self.details)

    def json(self) -> str:
        return self.payload().model_dump_json()


def test_timecodes(start: str | None, end: str | None) -> bool:
    return bool(start or end)


def validate_settings(
    settings: JobSettings,
    catalog: Catalog,
    *,
    available_devices: set[str],
    enabled_engines: set[str],
    aligner_ready: bool,
    check_install: bool = False,
    installed_models: set[str] | None = None,
) -> ModelSpec:
    """Server-side re-check of docs/02 §4–5. Returns the resolved ASR model."""
    model = catalog.model_or_none(settings.asr_model_id)
    if model is None or model.role != "asr":
        raise ApiError("invalid_settings", f"Unknown ASR model: {settings.asr_model_id}")
    engine = catalog.engine(model.engine)
    if engine.key not in enabled_engines:
        raise ApiError("invalid_settings", f"Engine {engine.name} is disabled. Enable it on the Engines page.")

    if settings.device != Device.auto:
        slug = settings.device.value
        if slug not in engine.devices:
            raise ApiError("invalid_settings", f"{engine.name} has no {slug.upper()} backend.")
        if slug not in available_devices:
            raise ApiError(
                "unsupported_device",
                f"{slug.upper()} is not available in this environment.",
                {"device": slug, "engine": engine.key},
            )

    if settings.word_timestamps:
        if model.word_ts == "none":
            raise ApiError("invalid_settings", f"{model.name} does not provide word timestamps.")
        if model.word_ts == "aligner" and not aligner_ready:
            raise ApiError(
                "invalid_settings",
                "Word timestamps require the Qwen3 ForcedAligner model. Install it on the Engines page.",
            )
    if settings.mark_pauses and (model.word_ts == "none" or (model.word_ts == "aligner" and not aligner_ready)):
        raise ApiError("invalid_settings", "Pause marking needs word-level timing for this model.")
    if settings.verbatim and not catalog.is_whisper_family(model):
        raise ApiError("invalid_settings", "Verbatim mode is only available for whisper-family engines.")

    if settings.timestamps_mode == TimestampsMode.none and (
        settings.output.srt or settings.output.vtt or settings.output.timed_txt
    ):
        raise ApiError(
            "invalid_settings",
            "Captions (SRT, VTT, and timed text) need a timestamp mode. Choose interval or turns.",
        )
    if settings.diar != Diar.off:
        diar_models = catalog.diar_models_for(settings.diar.value)
        ids = {m.id for m in diar_models}
        if settings.diar_model_id not in ids:
            raise ApiError("invalid_settings", "Choose a diarization model that matches the selected diarizer.")
    if check_install and installed_models is not None and model.id not in installed_models and model.engine != "demo":
        raise ApiError(
            "invalid_settings",
            f"{model.name} is not installed yet. Download it from the Engines page before queueing.",
        )
    return model


def settings_warnings(settings: JobSettings, catalog: Catalog) -> list[str]:
    warnings: list[str] = []
    model = catalog.model_or_none(settings.asr_model_id)
    if model is None:
        return ["Selected model is not in the catalog."]
    engine = catalog.engine(model.engine)
    if model.english_only and settings.language not in {"auto", "en"}:
        warnings.append(f"{model.name} is English-only; the selected language ({settings.language}) may be ignored.")
    if not model.language_auto and settings.language == "auto":
        warnings.append(f"{model.name} has no auto-detect; choose a language explicitly.")
    if settings.device.value not in {"auto", *engine.devices}:
        warnings.append(f"{engine.name} cannot run on {settings.device.value.upper()}; the device will be rejected.")
    if settings.word_timestamps and model.word_ts == "aligner":
        warnings.append("Word timing will be produced by the Qwen3 ForcedAligner, one five-minute window at a time.")
    if settings.diar != Diar.off and settings.speaker_count != "auto":
        diar_model = catalog.model_or_none(settings.diar_model_id or "")
        if diar_model and diar_model.max_speakers and int(settings.speaker_count) > diar_model.max_speakers:
            warnings.append(
                f"{diar_model.name} supports up to {diar_model.max_speakers} speakers; "
                f"{settings.speaker_count} will be capped inside the engine."
            )
    if settings.timestamps_mode == TimestampsMode.none and (settings.output.html or settings.output.txt):
        warnings.append("Raw text: captions are disabled and no timestamps will appear in HTML/TXT.")
    if settings.verbatim:
        warnings.append("Verbatim mode uses an initial prompt; engines may still normalize some disfluencies.")
    return warnings


def resolved_device(settings: JobSettings, catalog: Catalog, available: set[str], image_variant: str) -> tuple[str, str]:
    """Resolve auto device per docs/03 §5, or confirm an explicit choice."""
    model = catalog.model(settings.asr_model_id)
    engine = catalog.engine(model.engine)
    if settings.device != Device.auto:
        slug = settings.device.value
        if slug not in engine.devices:
            raise ApiError("invalid_settings", f"{engine.name} has no {slug.upper()} backend.")
        if slug not in available:
            raise ApiError("unsupported_device", f"{slug.upper()} is not available in this image.")
        return slug, compute_label(engine.key, model.id, slug)
    order = ["vulkan", "cuda", "cpu"] if engine.key in {"whisper.cpp", "nemo-speech"} else ["cuda", "cpu"]
    for slug in order:
        if slug in engine.devices and slug in available:
            if slug == "cuda" and image_variant != "cuda":
                continue
            return slug, compute_label(engine.key, model.id, slug)
    raise ApiError("unsupported_device", f"No supported device is available for {engine.name}.")


def compute_label(engine_key: str, model_id: str, device: str) -> str:
    if engine_key == "demo":
        return "synthetic"
    if device == "cpu":
        return "cpu-int8" if engine_key == "faster-whisper" else "cpu"
    if device == "cuda":
        return "cuda-fp16" if engine_key == "faster-whisper" else "cuda"
    bits = "q5_0" if "q5" in model_id else "q8_0" if "q8" in model_id else "f16"
    return f"vulkan-{bits}"


def job_settings_from_row(row: sqlite3.Row) -> JobSettings:
    return JobSettings.model_validate(json.loads(row["settings"]))


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def job_out(row: sqlite3.Row, jobs_root: Path) -> dict[str, Any]:
    settings = job_settings_from_row(row)
    outputs: list[dict[str, Any]] = []
    out_dir = jobs_root / row["id"] / "out"
    if out_dir.is_dir():
        for path in sorted(out_dir.iterdir(), key=lambda p: p.name):
            if path.is_file():
                stat = path.stat()
                outputs.append({"name": path.name, "size_ct": stat.st_size, "modified": stat.st_mtime})
    return {
        "id": row["id"],
        "seq": row["seq"],
        "created_at": row["created_at"],
        "started_at": row["started_at"],
        "finished_at": row["finished_at"],
        "duration_s": row["duration_s"],
        "title": row["title"],
        "kind": row["kind"],
        "settings": settings.model_dump(mode="json"),
        "status": row["status"],
        "stage": row["stage"],
        "progress": row["progress"],
        "message": row["message"],
        "resolved": json.loads(row["resolved"]) if row["resolved"] else None,
        "error": json.loads(row["error"]) if row["error"] else None,
        "log_paths": json.loads(row["log_paths"]) if row["log_paths"] else [],
        "outputs_manifest": outputs,
    }
