"""pyannote.audio diarization adapter."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

from ...jobs import ApiError
from ...pipeline.types import DiarResult, DiarTurn
from ..base import CancelFn, EngineContext, JobCancelledError, ProgressFn
from . import args


class PyannoteAdapter:
    name = "pyannote"

    def __init__(self, context: EngineContext) -> None:
        self.context = context

    def diarize(
        self,
        audio_path: Path,
        model: Any,
        *,
        device: str,
        speaker_count: int | str,
        progress: ProgressFn,
        log_path: Path,
        cancel: CancelFn,
    ) -> DiarResult:
        if not self.context.hf_token and model.gated:
            raise ApiError(
                "hf_auth",
                "pyannote pipelines are gated. Accept both Hugging Face licenses and add a read token in Settings.",
            )
        try:
            pyannote = importlib.import_module("pyannote.audio")
            torch = importlib.import_module("torch")
        except ImportError as exc:
            raise ApiError("engine_crash", "pyannote.audio/torch are not installed in this image.") from exc

        model_dir = self.context.model_dir(model)
        model_ref = str(model_dir) if model_dir.exists() and any(model_dir.iterdir()) else (model.repo or "")
        if not model_ref:
            raise ApiError("engine_crash", f"{model.name} has no repo or local files.")
        progress(0.02, "loading pyannote pipeline")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            pipeline = pyannote.Pipeline.from_pretrained(model_ref, token=self.context.hf_token)
            pipeline.to(torch.device("cuda" if device == "cuda" else "cpu"))
        except Exception as exc:
            raise ApiError("engine_crash", f"pyannote pipeline failed to load: {exc}") from exc

        min_speakers, max_speakers = args.speaker_bounds(speaker_count)
        kwargs: dict[str, Any] = {}
        if min_speakers is not None:
            kwargs["min_speakers"] = min_speakers
        if max_speakers is not None:
            kwargs["max_speakers"] = max_speakers
        progress(0.15, "diarizing")
        try:
            with pyannote.pipelines.utils.hook.ProgressHook() as hook:
                output = pipeline(str(audio_path), hook=hook, **kwargs)
        except Exception as exc:
            raise ApiError("engine_crash", f"pyannote diarization failed: {exc}") from exc
        if cancel():
            raise JobCancelledError

        annotation = getattr(output, "speaker_diarization", output)
        raw: list[tuple[float, float, str]] = []
        try:
            for turn, _, label in annotation.itertracks(yield_label=True):
                raw.append((float(turn.start), float(turn.end), str(label)))
        except AttributeError as exc:
            raise ApiError("engine_crash", "pyannote returned an unexpected diarization object.") from exc
        raw.sort(key=lambda row: (row[0], row[1]))
        labels: dict[str, int] = {}
        turns: list[DiarTurn] = []
        for start, end, label in raw:
            if label not in labels:
                labels[label] = len(labels)
            turns.append(DiarTurn(start=start, end=end, speaker=labels[label]))
        log_path.write_text(
            f"pyannote pipeline: {model.id}\ndevice: {device}\nturns: {len(turns)}\nspeakers: {len(labels)}\n",
            encoding="utf-8",
        )
        progress(1.0, "diarization complete")
        return DiarResult(engine="pyannote", model_id=model.id, device=device, turns=turns)
