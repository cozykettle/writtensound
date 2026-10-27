"""Qwen3-ASR adapter (transformers 5.20, docs/04 §3.4)."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

from ...jobs import ApiError, JobSettings
from ...pipeline.types import Segment, TranscriptionResult
from ..base import CancelFn, EngineContext, JobCancelledError, ProgressFn
from . import args
from .audio import duration_s, read_window
from .chat_template import ensure_chat_template
from .dtype import cast_floating, inference_dtype


class Qwen3AsrAdapter:
    name = "qwen3-asr"

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
            transformers = importlib.import_module("transformers")
            torch = importlib.import_module("torch")
        except ImportError as exc:
            raise ApiError("engine_crash", "transformers/torch are not installed in this image.") from exc
        model_dir = self.context.model_dir(model)
        if not model_dir.exists() or not any(model_dir.iterdir()):
            raise ApiError("engine_crash", f"Qwen3 model files are missing at {model_dir}.")

        forced_language = None if settings.language == "auto" else settings.language

        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as log:
            log.write(f"loading {model.id} from {model_dir}\n")
            log.write(
                f"transformers {getattr(transformers, '__version__', '?')}; "
                f"torch {getattr(torch, '__version__', '?')}\n"
            )

            def write_log(message: str) -> None:
                log.write(f"{message}\n")

            try:
                processor = transformers.AutoProcessor.from_pretrained(str(model_dir))
            except Exception as exc:
                raise ApiError("engine_crash", f"Qwen3-ASR failed to load: {exc}") from exc

            ensure_chat_template(
                processor,
                model_dir=model_dir,
                repo=model.repo,
                token=self.context.hf_token,
                log=write_log,
            )
            try:
                torch_device = "cuda" if device == "cuda" else "cpu"
                dtype = inference_dtype(torch, torch_device)
                pipeline_model = transformers.AutoModelForMultimodalLM.from_pretrained(
                    str(model_dir), dtype=dtype
                )
                pipeline_model = pipeline_model.to(torch_device)
            except Exception as exc:
                raise ApiError("engine_crash", f"Qwen3-ASR failed to load: {exc}") from exc

            duration = duration_s(audio_path)
            segments: list[Segment] = []
            detected_language = None
            starts = _window_starts(duration, args.CHUNK_SECONDS, args.CHUNK_OVERLAP_SECONDS)
            for index, start in enumerate(starts):
                if cancel():
                    raise JobCancelledError
                end = min(duration, start + args.CHUNK_SECONDS)
                try:
                    request = processor.apply_transcription_request(
                        audio=read_window(audio_path, start, end),
                        language=forced_language,
                    )
                    inputs = cast_floating(request.to(torch_device), torch, dtype)
                    output = pipeline_model.generate(**inputs, max_new_tokens=args.MAX_NEW_TOKENS)
                    generated = output[:, inputs["input_ids"].shape[1] :]
                    parsed = processor.decode(generated, return_format="parsed")[0]
                except Exception as exc:
                    raise ApiError("engine_crash", f"Qwen3-ASR failed on window {index + 1}: {exc}") from exc
                text = _transcript_text(parsed)
                language = _parsed_language(parsed) or forced_language
                if text:
                    segments.append(Segment(start=start, end=end, text=text, language=language))
                detected_language = detected_language or language
                progress(min(0.98, (index + 1) / max(1, len(starts))), f"qwen3 window {index + 1}/{len(starts)}")
            log.write(f"windows: {len(starts)}\nsegments: {len(segments)}\n")
        return TranscriptionResult(
            engine="qwen3-asr",
            model_id=model.id,
            device=device,
            compute=compute,
            language=detected_language,
            language_prob=None,
            duration_s=duration,
            media_id=media_id,
            segments=segments,
        )


def _window_starts(duration: float, window: float, overlap: float) -> list[float]:
    if duration <= 0:
        return [0.0]
    starts: list[float] = []
    cursor = 0.0
    while cursor < duration:
        starts.append(round(cursor, 3))
        if cursor + window >= duration:
            break
        cursor += max(1.0, window - overlap)
    return starts


def _parsed_language(parsed: Any) -> str | None:
    if not isinstance(parsed, dict):
        return None
    value = parsed.get("language")
    return str(value) if value else None


def _transcript_text(parsed: Any) -> str:
    if not isinstance(parsed, dict):
        return str(parsed).strip()
    value = parsed.get("transcription") or parsed.get("text") or ""
    if isinstance(value, list):
        return " ".join(str(item) for item in value).strip()
    return str(value).strip()
