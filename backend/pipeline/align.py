"""Qwen3-ForcedAligner orchestration (docs/04 §3.6).

The aligner runs a token-classification forward pass, not ``generate``.  Audio
is passed as a numpy window so transformers never hands a file path to
torchcodec, and word times are offset by each window's start.
"""

from __future__ import annotations

import importlib
import logging
import math
from collections.abc import Callable
from pathlib import Path

from ..engines.qwen3 import args as qwen_args
from ..engines.qwen3.audio import read_window
from ..engines.qwen3.chat_template import ensure_chat_template
from ..engines.qwen3.dtype import cast_floating, inference_dtype
from ..jobs import ApiError
from .types import Segment, Word

logger = logging.getLogger(__name__)

ALIGNER_LANGUAGES = {
    "zh": "Chinese",
    "en": "English",
    "yue": "Cantonese",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "pt": "Portuguese",
    "ru": "Russian",
    "es": "Spanish",
}
_ALIGNER_BY_NAME = {name.lower(): code for code, name in ALIGNER_LANGUAGES.items()}


def normalize_language(language: str | None) -> str | None:
    """Map a code or full language name to the aligner's canonical code."""
    if language is None:
        return None
    value = language.strip()
    if not value or value.lower() == "auto":
        return None
    if value.lower() in ALIGNER_LANGUAGES:
        return value.lower()
    return _ALIGNER_BY_NAME.get(value.lower())


def run_aligner(
    audio_path: Path,
    segments: list[Segment],
    *,
    model_dir: Path,
    device: str,
    language: str | None,
    repo: str | None = None,
    token: str | None = None,
    log: Callable[[str], None] | None = None,
    window_s: float = qwen_args.ALIGNER_WINDOW_SECONDS,
) -> list[Segment]:
    emit = log or (lambda message: logger.info(message))
    code = normalize_language(language)
    if code is None:
        supported = ", ".join(sorted(ALIGNER_LANGUAGES))
        emit(
            f"forced aligner skipped: language {language!r} is unknown or unsupported; "
            f"supported codes are {supported}. Word timestamps stay empty."
        )
        return segments

    try:
        transformers = importlib.import_module("transformers")
        torch = importlib.import_module("torch")
    except ImportError as exc:
        raise ApiError("engine_crash", "transformers/torch are not installed; install the engines extra.") from exc

    if not model_dir.exists():
        raise ApiError("engine_crash", f"ForcedAligner model is missing at {model_dir}.")

    logger.info("loading forced aligner from %s on %s", model_dir, device)
    torch_device = "cuda" if device == "cuda" else "cpu"
    try:
        processor = transformers.AutoProcessor.from_pretrained(str(model_dir))
        ensure_chat_template(
            processor,
            model_dir=model_dir,
            repo=repo,
            token=token,
            log=emit,
        )
        model = transformers.AutoModelForTokenClassification.from_pretrained(
            str(model_dir), dtype=inference_dtype(torch, torch_device)
        )
        model = model.to(torch_device)
        model.eval()
    except ApiError:
        raise
    except Exception as exc:  # pragma: no cover - requires the real model
        raise ApiError("engine_crash", f"ForcedAligner failed to load: {exc}") from exc

    aligned: list[Segment] = []
    for index, segment in enumerate(segments, start=1):
        text = segment.text.strip()
        if not text:
            aligned.append(segment)
            continue
        windows = _windows(segment, text, window_s)
        if len(windows) > 1:
            emit(
                f"forced aligner: segment {index} spans {segment.end - segment.start:.1f}s; "
                f"splitting it into {len(windows)} windows of at most {window_s:.0f}s"
            )
        words: list[Word] = []
        try:  # pragma: no cover - requires the real model
            for start, end, window_text in windows:
                audio = read_window(audio_path, start, end)
                if audio.size == 0:
                    continue
                inputs, word_lists = processor.prepare_forced_aligner_inputs(
                    audio=audio,
                    transcript=window_text,
                    language=code,
                )
                inputs = cast_floating(inputs.to(torch_device), torch, model.dtype)
                with torch.inference_mode():
                    outputs = model(**inputs)
                items = processor.decode_forced_alignment(
                    logits=outputs.logits,
                    input_ids=inputs["input_ids"],
                    word_lists=word_lists,
                    timestamp_token_id=model.config.timestamp_token_id,
                )[0]
                for item in items:
                    word_start = max(start, min(end, start + float(item["start_time"])))
                    word_end = max(word_start, min(end, start + float(item["end_time"])))
                    words.append(
                        Word(
                            start=round(word_start, 3),
                            end=round(word_end, 3),
                            text=str(item["text"]),
                        )
                    )
        except Exception as exc:
            raise ApiError("engine_crash", f"ForcedAligner failed while aligning segment {index}: {exc}") from exc
        if words:
            segment.words = words
        aligned.append(segment)
    return aligned


def _windows(segment: Segment, text: str, window_s: float) -> list[tuple[float, float, str]]:
    """Split a long segment into time windows with a proportional text split."""
    start = max(0.0, segment.start)
    end = max(start, segment.end)
    window_s = max(1.0, window_s)
    span = end - start
    if span <= window_s:
        return [(start, end, text)]

    parts = max(1, math.ceil(span / window_s))
    chunks = _split_text(text, parts)
    if not chunks:
        return [(start, end, text)]
    windows: list[tuple[float, float, str]] = []
    for offset in range(parts):
        chunk_start = start + span * offset / parts
        chunk_end = end if offset + 1 == parts else start + span * (offset + 1) / parts
        chunk = chunks[offset].strip() if offset < len(chunks) else ""
        if chunk:
            windows.append((chunk_start, chunk_end, chunk))
    return windows or [(start, end, text)]


def _split_text(text: str, parts: int) -> list[str]:
    """Split text into at most ``parts`` contiguous, roughly equal chunks."""
    stripped = text.strip()
    spaced = any(char.isspace() for char in stripped)
    units = stripped.split() if spaced else list(stripped)
    if not units:
        return []
    parts = min(max(1, parts), len(units))
    joiner = " " if spaced else ""
    total = sum(len(unit) for unit in units)
    chunks: list[str] = []
    start = 0
    consumed = 0
    for part in range(parts):
        remaining = parts - part
        if remaining == 1:
            end = len(units)
        else:
            target = (total - consumed) / remaining
            end = start
            taken = 0
            while end < len(units) - (remaining - 1):
                if end > start and taken >= target:
                    break
                taken += len(units[end])
                end += 1
        chunks.append(joiner.join(units[start:end]))
        consumed += sum(len(unit) for unit in units[start:end])
        start = end
    return chunks
