"""Frozen faster-whisper call shape (Phase A SPIKE S3 pending)."""

from __future__ import annotations

from typing import Any

FASTER_WHISPER_VERSION = "1.2.1"
INITIAL_PROMPT_VERBATIM = "Uh, um, er, ah — transcribe every filler word exactly as spoken."
VAD_MIN_SILENCE_MS = 500


def compute_type(device: str) -> str:
    return "float16" if device == "cuda" else "int8"


def transcribe_kwargs(*, language: str, word_timestamps: bool, verbatim: bool, accept_progress_hook: bool) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "language": None if language == "auto" else language,
        "word_timestamps": word_timestamps,
        "vad_filter": True,
        "vad_parameters": {"min_silence_duration_ms": VAD_MIN_SILENCE_MS},
        "condition_on_previous_text": False,
        "temperatures": (0.0,),
        "initial_prompt": INITIAL_PROMPT_VERBATIM if verbatim else None,
        "log_progress": True,
    }
    if accept_progress_hook:
        kwargs["progress_hook"] = True  # replaced with the real callback by the adapter
    return kwargs
