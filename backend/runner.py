"""Single-threaded serial job runner (docs/01 §2, docs/04)."""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import deque
from collections.abc import Callable
from pathlib import Path
from typing import Any

from . import __version__
from .catalog import Catalog, ModelSpec
from .config import AppConfig, EnvConfig, Paths, effective_hf_token
from .downloader import DownloadManager
from .engines import EngineContext, JobCancelledError, asr_adapter, diarizer_adapter
from .events import EventHub
from .hardware import HardwareStatus
from .jobs import (
    ApiError,
    JobSettings,
    job_out,
    job_settings_from_row,
    resolved_device,
    settings_warnings,
    utc_now,
    validate_settings,
)
from .pipeline import prep
from .pipeline.align import run_aligner
from .pipeline.merge import merge
from .pipeline.types import DiarResult, TranscriptionResult
from .renderers import render_all

log = logging.getLogger(__name__)

STAGE_WEIGHTS = {"ingest": 0.02, "prepare": 0.03, "asr": 0.65, "diarize": 0.20, "align": 0.04, "render": 0.06}
STAGE_ORDER = ["ingest", "prepare", "asr", "diarize", "align", "render"]
EMIT_INTERVAL = 0.5


class JobRunner:
    def __init__(
        self,
        *,
        store: Any,
        paths: Paths,
        env: EnvConfig,
        catalog: Catalog,
        config_provider: Callable[[], AppConfig],
        hardware_provider: Callable[[], HardwareStatus],
        hub: EventHub,
        downloader: DownloadManager,
    ) -> None:
        self.store = store
        self.paths = paths
        self.env = env
        self.catalog = catalog
        self.config_provider = config_provider
        self.hardware_provider = hardware_provider
        self.hub = hub
        self.downloader = downloader
        self._queue: deque[str] = deque()
        self._queued: set[str] = set()
        self._condition = threading.Condition()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._current: str | None = None
        self._cancel_events: dict[str, threading.Event] = {}
        self._last_emit: dict[str, float] = {}
        self._last_db: dict[str, float] = {}

    # -- lifecycle -------------------------------------------------------
    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        for row in self.store.pending():
            self._queue.append(row["id"])
            self._queued.add(row["id"])
        self._thread = threading.Thread(target=self._loop, name="writtensound-runner", daemon=True)
        self._thread.start()
        self._publish_queue()

    def stop(self) -> None:
        self._stop.set()
        with self._condition:
            self._condition.notify_all()
        if self._thread:
            self._thread.join(timeout=5)

    # -- public queue operations ----------------------------------------
    def enqueue(self, job_id: str) -> None:
        with self._condition:
            if job_id not in self._queued:
                self._queue.append(job_id)
                self._queued.add(job_id)
            self._cancel_events[job_id] = threading.Event()
            self._condition.notify()
        self._publish_queue()
        self._emit_job(job_id, force=True)

    def cancel(self, job_id: str) -> None:
        row = self.store.get(job_id)
        if row is None:
            raise ApiError("not_found", "Job not found.")
        if row["status"] == "pending":
            self.store.update(job_id, status="cancelled", message="cancelled", finished_at=utc_now())
            with self._condition:
                try:
                    self._queue.remove(job_id)
                    self._queued.discard(job_id)
                except ValueError:
                    pass
            self._emit_job(job_id, force=True)
            self._publish_queue()
            return
        if row["status"] != "running":
            raise ApiError("conflict", "Only pending or running jobs can be cancelled.")
        event = self._cancel_events.setdefault(job_id, threading.Event())
        event.set()
        self._emit_job(job_id, force=True)

    def retry(self, job_id: str, *, force_cpu: bool = False) -> None:
        row = self.store.get(job_id)
        if row is None:
            raise ApiError("not_found", "Job not found.")
        if row["status"] not in {"failed", "cancelled", "interrupted"}:
            raise ApiError("conflict", "Only failed, cancelled or interrupted jobs can be retried.")
        settings = job_settings_from_row(row)
        if force_cpu:
            settings = settings.model_copy(update={"device": "cpu"})
            self.store.update(job_id, settings=settings.model_dump(mode="json"))
        self.store.update(
            job_id,
            status="pending",
            stage=None,
            progress=0.0,
            message="queued again",
            error=None,
            finished_at=None,
            started_at=None,
            duration_s=None,
        )
        self.enqueue(job_id)

    def is_running(self, job_id: str) -> bool:
        return self._current == job_id

    def wait_for(self, job_id: str, timeout: float = 10.0) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            row = self.store.get(job_id)
            if row is None or row["status"] != "running":
                return
            time.sleep(0.1)

    def queue_snapshot(self) -> list[dict[str, Any]]:
        with self._condition:
            ids = ([self._current] if self._current else []) + list(self._queue)
        rows = []
        for job_id in ids:
            if job_id is None:
                continue
            row = self.store.get(job_id)
            if row and row["status"] in {"pending", "running"}:
                rows.append(job_out(row, self.paths.jobs))
        rows.sort(key=lambda job: job.get("seq") or 0, reverse=True)
        return rows

    # -- worker ----------------------------------------------------------
    def _loop(self) -> None:
        while not self._stop.is_set():
            with self._condition:
                if not self._queue:
                    self._condition.wait(timeout=1.0)
                    continue
                job_id = self._queue.popleft()
                self._queued.discard(job_id)
            self._current = job_id
            event = self._cancel_events.setdefault(job_id, threading.Event())
            try:
                self._process(job_id, event)
            except JobCancelledError:
                self._finish_cancelled(job_id)
            except ApiError as exc:
                self._finish_failed(job_id, exc)
            except Exception as exc:  # noqa: BLE001 - surfaced as a job failure
                log.exception("unexpected job failure")
                self._finish_failed(job_id, ApiError("internal", f"Unexpected error: {exc}"))
            finally:
                self._current = None
                self._publish_queue()
                self._last_emit.pop(job_id, None)
                self._last_db.pop(job_id, None)

    def _process(self, job_id: str, cancel: threading.Event) -> None:
        row = self.store.get(job_id)
        if row is None or row["status"] != "pending":
            return
        settings = job_settings_from_row(row)
        config = self.config_provider()
        hardware = self.hardware_provider()
        available = set(hardware.devices.keys())
        aligner_model = self.catalog.aligner_model()
        aligner_ready = bool(aligner_model and self.downloader.is_ready(aligner_model))
        model = validate_settings(
            settings,
            self.catalog,
            available_devices=available,
            enabled_engines={key for key, enabled in config.engine_enabled.items() if enabled},
            aligner_ready=aligner_ready,
        )
        device, compute = resolved_device(settings, self.catalog, available, self.env.image_variant)
        if settings.device.value == "auto" and device != "cpu":
            self._log(job_id, f"auto device resolved to {device} ({compute})")

        diar_model: ModelSpec | None = None
        if settings.diar_model_id:
            diar_model = self.catalog.model_or_none(settings.diar_model_id)
            if diar_model is None:
                raise ApiError("invalid_settings", "Diarization model is not in the catalog.")

        job_dir = self.paths.job_dir(job_id)
        job_dir.mkdir(parents=True, exist_ok=True)
        log_paths: list[str] = ["logs/pipeline.log"]
        self._log(job_id, f"starting {row['title']} with {model.name} on {device}")
        for warning in settings_warnings(settings, self.catalog):
            self._log(job_id, f"warning: {warning}")

        self.store.update(
            job_id,
            status="running",
            stage="ingest",
            progress=0.0,
            message="starting",
            started_at=utc_now(),
            resolved={
                "asr_model": model.id,
                "asr_engine": model.engine,
                "diar_model": diar_model.id if diar_model else None,
                "device": device,
                "compute": compute,
            },
        )
        self._emit_job(job_id, force=True)

        source = _find_source(job_dir)
        if source is None:
            raise ApiError("storage", "The job's media file is missing from its working directory.")

        self._stage(job_id, "ingest", 0.2, "validating media")
        media_info = prep.probe_media(source)
        if media_info.duration_s <= 0:
            raise ApiError("unsupported_media", "The media has no measurable duration.")
        self._stage(job_id, "ingest", 1.0, "media validated")

        if cancel.is_set():
            raise JobCancelledError
        self._stage(job_id, "prepare", 0.1, "preparing 16 kHz mono audio")
        prep_path, _, _ = prep.prepare_audio(
            source,
            job_dir,
            start_at=settings.start_at,
            end_at=settings.end_at,
            log=_JobLog(self._log, job_id),
        )
        self._stage(job_id, "prepare", 1.0, "audio ready")

        checkpoint = _load_checkpoint(job_dir, settings, model.id)
        transcription: TranscriptionResult
        diar: DiarResult | None = None
        if checkpoint:
            transcription = checkpoint["transcription"]
            diar = checkpoint["diar"]
            self._log(job_id, "reusing checkpointed transcript (no ASR re-run)")
            self._stage(job_id, "asr", 1.0, "checkpoint reused")
        else:
            self._ensure_model(model, job_id, cancel)
            if cancel.is_set():
                raise JobCancelledError
            adapter = asr_adapter(model.engine, self._engine_context(config))
            progress = self._progress_cb(job_id, "asr")
            self._stage(job_id, "asr", 0.0, f"transcribing with {model.name}")
            transcription = adapter.transcribe(
                prep_path,
                settings,
                model,
                device=device,
                compute=compute,
                media_id=settings.out_basename or _media_id(row["title"]),
                progress=progress,
                log_path=job_dir / "logs" / "engine-asr.log",
                cancel=lambda: cancel.is_set(),
            )
            log_paths.append("logs/engine-asr.log")
            self._stage(job_id, "asr", 1.0, "transcription complete")

            if settings.diar.value != "off" and diar_model is not None:
                if cancel.is_set():
                    raise JobCancelledError
                self._ensure_model(diar_model, job_id, cancel)
                diar_adapter = diarizer_adapter(settings.diar.value, self._engine_context(config))
                progress = self._progress_cb(job_id, "diarize")
                self._stage(job_id, "diarize", 0.0, f"diarizing with {diar_model.name}")
                diar = diar_adapter.diarize(
                    prep_path,
                    diar_model,
                    device=device,
                    speaker_count=settings.speaker_count,
                    progress=progress,
                    log_path=job_dir / "logs" / "engine-diar.log",
                    cancel=lambda: cancel.is_set(),
                )
                log_paths.append("logs/engine-diar.log")
                self._stage(job_id, "diarize", 1.0, "diarization complete")
            else:
                self._stage(job_id, "diarize", 1.0, "diarization off")

            if (
                settings.word_timestamps
                and model.word_ts == "aligner"
                and aligner_model is not None
                and aligner_ready
            ):
                if cancel.is_set():
                    raise JobCancelledError
                self._stage(job_id, "align", 0.1, "aligning words")
                transcription.segments = run_aligner(
                    prep_path,
                    transcription.segments,
                    model_dir=self.downloader.model_dir(aligner_model),
                    device=device,
                    language=transcription.language,
                    repo=aligner_model.repo,
                    token=self._engine_context(config).hf_token,
                    log=lambda message: self._log(job_id, message),
                )
                self._stage(job_id, "align", 1.0, "words aligned")
            else:
                self._stage(job_id, "align", 1.0, "alignment not needed")
            _save_checkpoint(job_dir, settings, model.id, transcription, diar)

        if cancel.is_set():
            raise JobCancelledError
        self._stage(job_id, "render", 0.1, "merging speakers and pauses")
        segments, pauses = merge(
            transcription,
            diar,
            mark_pauses=settings.mark_pauses,
            pause_min_s=settings.pause_min_s,
        )
        captions = settings.output.srt or settings.output.vtt or settings.output.timed_txt
        if captions and not any(segment.words for segment in segments):
            self._log(
                job_id,
                "caption output uses ASR segment boundaries; this transcript has no word timings",
            )
        resolved = {
            "asr_model": model.id,
            "asr_engine": model.engine,
            "diar_model": diar_model.id if diar_model else None,
            "device": device,
            "compute": compute,
        }
        self._stage(job_id, "render", 0.4, "writing output files")
        written = render_all(
            out_dir=job_dir / "out",
            title=row["title"],
            segments=segments,
            pauses=pauses,
            settings=settings,
            transcription=transcription,
            diar=diar,
            resolved=resolved,
            version=__version__,
        )
        self._log(job_id, f"wrote {len(written)} output files")
        existing_logs = [path for path in log_paths if (job_dir / path).exists()]
        self.store.update(
            job_id,
            status="succeeded",
            stage="render",
            progress=1.0,
            message="transcript ready",
            log_paths=existing_logs,
            finished_at=utc_now(),
            duration_s=_elapsed(self.store.require(job_id)["started_at"]),
        )
        self._emit_job(job_id, force=True)

    # -- downloads -------------------------------------------------------
    def _ensure_model(self, model: ModelSpec, job_id: str, cancel: threading.Event) -> None:
        if self.downloader.is_ready(model):
            return
        folder = self.downloader.model_dir(model)
        if folder.exists() and any(path.is_file() for path in folder.rglob("*")):
            self.downloader.scan()
            if self.downloader.is_ready(model):
                return
        config = self.config_provider()
        if model.gated and not effective_hf_token(config, self.env):
            raise ApiError(
                "hf_auth",
                f"{model.name} is gated. Add a Hugging Face read token in Settings, then accept the model licenses.",
            )
        self._log(job_id, f"downloading {model.name} ({model.size_label})")
        self.downloader.start(model.engine, model.id)
        while True:
            state = self.downloader.state_row(model)
            if state["status"] == "ready":
                self._log(job_id, f"{model.name} is ready")
                return
            if state["status"] == "partial" or state["status"] == "absent":
                raise ApiError(
                    "download_failed",
                    state.get("last_error") or f"{model.name} could not be downloaded.",
                )
            if cancel.is_set():
                self.downloader.cancel(model.engine, model.id)
                raise JobCancelledError
            self._progress_raw(job_id, f"downloading model {state.get('pct', 0):.0f}%", None)
            time.sleep(0.6)

    # -- progress & SSE --------------------------------------------------
    def _stage(self, job_id: str, stage: str, fraction: float, message: str) -> None:
        base = sum(STAGE_WEIGHTS[s] for s in STAGE_ORDER[: STAGE_ORDER.index(stage)])
        overall = min(0.999, base + STAGE_WEIGHTS[stage] * max(0.0, min(1.0, fraction)))
        self.store.update(job_id, stage=stage, progress=round(overall, 4), message=message)
        self._emit_job(job_id)

    def _progress_cb(self, job_id: str, stage: str) -> Callable[[float, str], None]:
        def report(fraction: float, message: str) -> None:
            self._stage(job_id, stage, fraction, message)

        return report

    def _progress_raw(self, job_id: str, message: str, progress: float | None) -> None:
        now = time.monotonic()
        last = self._last_emit.get(job_id, 0.0)
        if progress is not None:
            self.store.update(job_id, progress=progress)
        if now - last >= EMIT_INTERVAL:
            self._last_emit[job_id] = now
            self.store.update(job_id, message=message)
            self._emit_job(job_id, force=True)

    def _emit_job(self, job_id: str, *, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self._last_emit.get(job_id, 0.0) < EMIT_INTERVAL:
            return
        row = self.store.get(job_id)
        if row is None:
            return
        self._last_emit[job_id] = now
        self.hub.publish("job", job_out(row, self.paths.jobs))

    def _publish_queue(self) -> None:
        self.hub.publish("queue", self.queue_snapshot())

    # -- terminal states -------------------------------------------------
    def _finish_cancelled(self, job_id: str) -> None:
        self._log(job_id, "job cancelled")
        row = self.store.get(job_id)
        if row is None:
            return
        self.store.update(
            job_id,
            status="cancelled",
            message="cancelled",
            finished_at=utc_now(),
            duration_s=_elapsed(row["started_at"]),
        )
        self._emit_job(job_id, force=True)

    def _finish_failed(self, job_id: str, error: ApiError) -> None:
        self._log(job_id, f"FAILED [{error.code}] {error.message}")
        row = self.store.get(job_id)
        if row is None:
            return
        job_dir = self.paths.job_dir(job_id)
        logs = [path for path in ("logs/pipeline.log", "logs/engine-asr.log", "logs/engine-diar.log") if (job_dir / path).exists()]
        self.store.update(
            job_id,
            status="failed",
            error=error.payload().model_dump(),
            message=error.message,
            finished_at=utc_now(),
            duration_s=_elapsed(row["started_at"]),
            log_paths=logs,
        )
        self._emit_job(job_id, force=True)
        self.hub.publish("toast", {"level": "error", "text": f"{row['title']}: {error.message}", "source": "job"})

    # -- helpers ---------------------------------------------------------
    def _engine_context(self, config: AppConfig) -> EngineContext:
        return EngineContext(
            paths=self.paths,
            env=self.env,
            catalog=self.catalog,
            hf_token=effective_hf_token(config, self.env),
        )

    def _log(self, job_id: str, message: str) -> None:
        job_dir = self.paths.job_dir(job_id)
        logs = job_dir / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        line = f"{utc_now()} {message}\n"
        with (logs / "pipeline.log").open("a", encoding="utf-8") as handle:
            handle.write(line)
        log.info("job %s: %s", job_id, message)


class _JobLog:
    """Tiny logger shim so prep helpers can write into the job log."""

    def __init__(self, sink: Callable[[str, str], None], job_id: str) -> None:
        self.sink = sink
        self.job_id = job_id

    def warning(self, message: str, *args: Any) -> None:
        self.sink(self.job_id, "warning: " + (message % args if args else message))

    def info(self, message: str, *args: Any) -> None:
        self.sink(self.job_id, message % args if args else message)


def _find_source(job_dir: Path) -> Path | None:
    media = job_dir / "media"
    if not media.exists():
        return None
    for path in sorted(media.iterdir()):
        if path.is_file() and path.name not in {"prep.wav", "meta.json"}:
            return path
    return None


def _media_id(title: str) -> str:
    name = Path(title).name
    return name.rsplit(".", 1)[0] if "." in name else name


def _elapsed(started_at: str | None) -> float | None:
    if not started_at:
        return None
    from datetime import UTC, datetime

    try:
        start = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
        return round((datetime.now(UTC) - start).total_seconds(), 2)
    except ValueError:
        return None


def _checkpoint_path(job_dir: Path) -> Path:
    return job_dir / "checkpoint.json"


def _save_checkpoint(
    job_dir: Path,
    settings: JobSettings,
    model_id: str,
    transcription: TranscriptionResult,
    diar: DiarResult | None,
) -> None:
    payload = {
        "settings": settings.model_dump(mode="json"),
        "model_id": model_id,
        "transcription": transcription.model_dump(mode="json"),
        "diar": diar.model_dump(mode="json") if diar else None,
    }
    _checkpoint_path(job_dir).write_text(json.dumps(payload), encoding="utf-8")


def _load_checkpoint(
    job_dir: Path, settings: JobSettings, model_id: str
) -> dict[str, Any] | None:
    path = _checkpoint_path(job_dir)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("model_id") != model_id:
            return None
        if payload.get("settings") != settings.model_dump(mode="json"):
            return None
        return {
            "transcription": TranscriptionResult.model_validate(payload["transcription"]),
            "diar": DiarResult.model_validate(payload["diar"]) if payload.get("diar") else None,
        }
    except (OSError, json.JSONDecodeError, KeyError, ValueError):
        return None


def checkpoint_path(job_dir: Path) -> Path:
    return _checkpoint_path(job_dir)


def media_id_for(title: str) -> str:
    return _media_id(title)


def render_merged(
    *,
    out_dir: Path,
    title: str,
    transcription: TranscriptionResult,
    diar: DiarResult | None,
    settings: JobSettings,
    resolved: dict[str, Any],
) -> list[Path]:
    segments, pauses = merge(
        transcription,
        diar,
        mark_pauses=settings.mark_pauses,
        pause_min_s=settings.pause_min_s,
    )
    return render_all(
        out_dir=out_dir,
        title=title,
        segments=segments,
        pauses=pauses,
        settings=settings,
        transcription=transcription,
        diar=diar,
        resolved=resolved,
        version=__version__,
    )


__all__ = [
    "JobRunner",
    "checkpoint_path",
    "media_id_for",
    "render_merged",
]
