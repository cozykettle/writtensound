"""faster-whisper adapter (in-process CTranslate2)."""

from __future__ import annotations

import importlib
import inspect
import os
from pathlib import Path
from typing import Any

from ...jobs import ApiError, JobSettings
from ...pipeline.prep import probe_media
from ...pipeline.types import Segment, TranscriptionResult, Word
from ..base import CancelFn, EngineContext, JobCancelledError, ProgressFn
from . import args


class FasterWhisperAdapter:
    name = "faster-whisper"

    def __init__(self, context: EngineContext) -> None:
        self.context = context

    def transcribe(
        self,
        audio_path: Path,
        settings: JobSettings,
        model: Any,
        *,
        device: str,
        compute: str,
        media_id: str,
        progress: ProgressFn,
        log_path: Path,
        cancel: CancelFn,
    ) -> TranscriptionResult:
        try:
            fw = importlib.import_module("faster_whisper")
        except ImportError as exc:
            raise ApiError("engine_crash", "faster-whisper is not installed in this image.") from exc

        model_dir = self.context.model_dir(model)
        if not model_dir.exists() or not any(model_dir.iterdir()):
            raise ApiError("engine_crash", f"Model files are missing at {model_dir}. Download the model first.")
        ctranslate_device = "cuda" if device == "cuda" else "cpu"
        progress(0.02, "loading faster-whisper model")
        try:
            whisper = fw.WhisperModel(
                str(model_dir),
                device=ctranslate_device,
                compute_type=args.compute_type(device),
                cpu_threads=max(2, (os.cpu_count() or 4) // 2),
            )
        except Exception as exc:
            raise ApiError("engine_crash", f"faster-whisper could not load {model.id} on {device}: {exc}") from exc

        def hook(current: int, total: int) -> None:
            if total:
                progress(min(0.98, current / total), f"asr {current}/{total}")

        signature = inspect.signature(whisper.transcribe)
        accept_hook = "progress_hook" in signature.parameters
        kwargs = args.transcribe_kwargs(
            language=settings.language,
            word_timestamps=settings.word_timestamps or settings.mark_pauses,
            verbatim=settings.verbatim,
            accept_progress_hook=accept_hook,
        )
        if accept_hook:
            kwargs["progress_hook"] = hook
        else:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text("faster-whisper 1.2.1 has no progress_hook; using heuristic progress.\n", encoding="utf-8")

        try:
            segment_iter, info = whisper.transcribe(str(audio_path), **kwargs)
            segments: list[Segment] = []
            for index, raw in enumerate(segment_iter):
                if cancel():
                    raise JobCancelledError
                words = None
                if raw.words:
                    words = [
                        Word(
                            start=float(word.start),
                            end=float(word.end),
                            text=str(word.word).strip(),
                            probability=float(word.probability) if word.probability is not None else None,
                        )
                        for word in raw.words
                    ]
                segments.append(
                    Segment(
                        start=float(raw.start),
                        end=float(raw.end),
                        text=str(raw.text).strip(),
                        words=words,
                    )
                )
                if not accept_hook and index % 8 == 0:
                    progress(0.1 + min(0.8, index * 0.04), f"asr {index} segments")
        except JobCancelledError:
            raise
        except Exception as exc:
            raise ApiError("engine_crash", f"faster-whisper failed: {exc}") from exc

        duration = float(getattr(info, "duration", 0.0) or 0.0)
        if duration <= 0:
            duration = probe_media(audio_path).duration_s
        return TranscriptionResult(
            engine="faster-whisper",
            model_id=model.id,
            device=device,
            compute=compute,
            language=getattr(info, "language", None),
            language_prob=getattr(info, "language_probability", None),
            duration_s=duration,
            media_id=media_id,
            segments=segments,
        )
