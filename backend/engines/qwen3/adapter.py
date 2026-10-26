"""Qwen3-ASR adapter (transformers >= 5.13)."""

from __future__ import annotations

import importlib
import tempfile
import wave
from pathlib import Path
from typing import Any, TextIO

from ...jobs import ApiError, JobSettings
from ...pipeline.prep import probe_media
from ...pipeline.types import Segment, TranscriptionResult
from ..base import CancelFn, EngineContext, JobCancelledError, ProgressFn
from . import args


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
            importlib.import_module("torch")
        except ImportError as exc:
            raise ApiError("engine_crash", "transformers/torch are not installed in this image.") from exc
        model_dir = self.context.model_dir(model)
        if not model_dir.exists() or not any(model_dir.iterdir()):
            raise ApiError("engine_crash", f"Qwen3 model files are missing at {model_dir}.")

        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as log:
            log.write(f"loading {model.id} from {model_dir}\n")
            try:
                processor = transformers.AutoProcessor.from_pretrained(str(model_dir))
            except Exception as exc:
                raise ApiError("engine_crash", f"Qwen3-ASR failed to load: {exc}") from exc
            _ensure_processor_chat_template(
                processor,
                model_dir=model_dir,
                repo=model.repo,
                token=self.context.hf_token,
                log=log,
            )
            try:
                pipeline_model = transformers.AutoModelForMultimodalLM.from_pretrained(str(model_dir))
                torch_device = "cuda" if device == "cuda" else "cpu"
                pipeline_model = pipeline_model.to(torch_device)
            except Exception as exc:
                raise ApiError("engine_crash", f"Qwen3-ASR failed to load: {exc}") from exc

            duration = probe_media(audio_path).duration_s
            segments: list[Segment] = []
            detected_language = None
            starts = _window_starts(duration, args.CHUNK_SECONDS, args.CHUNK_OVERLAP_SECONDS)
            with tempfile.TemporaryDirectory(prefix="ws-qwen-", dir=log_path.parent.parent) as temp:
                for index, start in enumerate(starts):
                    if cancel():
                        raise JobCancelledError
                    end = min(duration, start + args.CHUNK_SECONDS)
                    chunk_path = Path(temp) / f"chunk-{index:03d}.wav"
                    _slice_wav(audio_path, chunk_path, start, end)
                    try:
                        request = processor.apply_transcription_request(
                            audio=str(chunk_path),
                            language=None if settings.language == "auto" else settings.language,
                        )
                        inputs = {key: value.to(torch_device) for key, value in request.items()}
                        output = pipeline_model.generate(**inputs, max_new_tokens=args.MAX_NEW_TOKENS)
                        parsed = processor.decode(output, return_format="parsed")[0]
                    except Exception as exc:
                        raise ApiError("engine_crash", f"Qwen3-ASR failed on window {index + 1}: {exc}") from exc
                    text = _transcript_text(parsed)
                    if text:
                        segments.append(Segment(start=start, end=end, text=text, language=parsed.get("language")))
                    detected_language = detected_language or parsed.get("language")
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


def _ensure_processor_chat_template(
    processor: Any,
    *,
    model_dir: Path,
    repo: str | None,
    token: str | None,
    log: TextIO,
) -> None:
    """Repair Qwen snapshots downloaded before ``*.jinja`` was allow-listed.

    Current Transformers stores the Qwen3 chat template in a separate
    ``chat_template.jinja`` file.  The v0.2 snapshot rules omitted it, so the
    processor loads with ``chat_template=None`` and every transcription call
    fails in ``apply_chat_template``.
    """
    if getattr(processor, "chat_template", None):
        return
    tokenizer = getattr(processor, "tokenizer", None)
    template = getattr(tokenizer, "chat_template", None)
    if template:
        processor.chat_template = template
        return

    template_path = model_dir / "chat_template.jinja"
    if not template_path.exists():
        if not repo:
            raise ApiError(
                "engine_crash",
                f"Qwen3 processor has no chat template and {template_path.name} is missing.",
            )
        log.write(f"{template_path.name} missing; fetching it from {repo}\n")
        try:
            from huggingface_hub import hf_hub_download

            hf_hub_download(
                repo_id=repo,
                filename=template_path.name,
                local_dir=str(model_dir),
                token=token,
            )
        except Exception as exc:
            raise ApiError(
                "engine_crash",
                f"Qwen3 model is missing {template_path.name} and it could not be fetched: {exc}. "
                "Remove and re-download the model from the Engines page.",
            ) from exc
    try:
        template = template_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ApiError(
            "engine_crash",
            f"Qwen3 chat template could not be read from {template_path}: {exc}",
        ) from exc

    processor.chat_template = template
    if tokenizer is not None:
        tokenizer.chat_template = template
    log.write(f"loaded chat template from {template_path.name}\n")


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


def _slice_wav(source: Path, target: Path, start: float, end: float) -> None:
    with wave.open(str(source), "rb") as reader:
        rate = reader.getframerate()
        reader.setpos(min(reader.getnframes(), int(start * rate)))
        frames = max(0, int((end - start) * rate))
        payload = reader.readframes(frames)
        params = reader.getparams()
    with wave.open(str(target), "wb") as writer:
        writer.setparams(params)
        writer.writeframes(payload)


def _transcript_text(parsed: Any) -> str:
    if not isinstance(parsed, dict):
        return str(parsed).strip()
    value = parsed.get("transcription") or parsed.get("text") or ""
    if isinstance(value, list):
        return " ".join(str(item) for item in value).strip()
    return str(value).strip()
