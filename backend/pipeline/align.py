"""Qwen3-ForcedAligner orchestration.

The model API is pinned in docs/04 §3.6 and verified in Phase A (S4). Until
that spike lands, the adapter fails loudly rather than inventing timings.
"""

from __future__ import annotations

import logging
from pathlib import Path

from ..jobs import ApiError
from .types import Segment, Word

log = logging.getLogger(__name__)

ALIGNER_LANGS = {"zh", "en", "yue", "fr", "de", "it", "ja", "ko", "pt", "ru", "es"}


def run_aligner(
    audio_path: Path,
    segments: list[Segment],
    *,
    model_dir: Path,
    device: str,
    language: str | None,
    window_s: float = 300.0,
) -> list[Segment]:
    if language and language not in ALIGNER_LANGS:
        raise ApiError(
            "invalid_settings",
            f"The Qwen3 ForcedAligner cannot align {language!r}. It supports 11 languages.",
        )
    try:
        from transformers import AutoModelForMultimodalLM, AutoProcessor  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ApiError("engine_crash", "transformers/torch are not installed; install the engines extra.") from exc

    if not model_dir.exists():
        raise ApiError("engine_crash", f"ForcedAligner model is missing at {model_dir}.")

    log.info("loading forced aligner from %s on %s", model_dir, device)
    try:
        processor = AutoProcessor.from_pretrained(str(model_dir))
        model = AutoModelForMultimodalLM.from_pretrained(str(model_dir))
        torch_device = "cuda" if device == "cuda" else "cpu"
        model = model.to(torch_device)
    except Exception as exc:  # pragma: no cover - requires the real model
        raise ApiError("engine_crash", f"ForcedAligner failed to load: {exc}") from exc

    aligned: list[Segment] = []
    for segment in segments:
        if not segment.text.strip():
            aligned.append(segment)
            continue
        try:  # pragma: no cover - requires the real model
            request = processor.apply_transcription_request(
                audio=str(audio_path),
                text=segment.text,
                language=language,
            )
            inputs = {key: value.to(torch_device) for key, value in request.items()}
            output = model.generate(**inputs, max_new_tokens=4096)
            parsed = processor.decode(output, return_format="parsed")[0]
            words = parsed.get("words") or []
        except Exception as exc:
            raise ApiError("engine_crash", f"ForcedAligner failed while aligning a segment: {exc}") from exc
        if words:
            segment.words = [
                Word(
                    start=float(word["start"]),
                    end=float(word["end"]),
                    text=str(word["text"]),
                )
                for word in words
            ]
        aligned.append(segment)
    return aligned
