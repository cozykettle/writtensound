"""Engine adapter registry."""

from __future__ import annotations

from .base import AsrAdapter, DiarizerAdapter, EngineContext, JobCancelledError
from .demo.adapter import DemoAdapter
from .fw.adapter import FasterWhisperAdapter
from .nemo.adapter import NemoSpeechAdapter
from .pyannote.adapter import PyannoteAdapter
from .qwen3.adapter import Qwen3AsrAdapter
from .wcpp.adapter import WhisperCppAdapter


def asr_adapter(engine_key: str, context: EngineContext) -> AsrAdapter:
    if engine_key == "demo":
        return DemoAdapter(context)
    if engine_key == "faster-whisper":
        return FasterWhisperAdapter(context)
    if engine_key == "whisper.cpp":
        return WhisperCppAdapter(context)
    if engine_key == "nemo-speech":
        return NemoSpeechAdapter(context)
    if engine_key == "qwen3-asr":
        return Qwen3AsrAdapter(context)
    raise KeyError(f"no ASR adapter for {engine_key}")


def diarizer_adapter(family: str, context: EngineContext) -> DiarizerAdapter:
    if family == "nemo":
        return NemoSpeechAdapter(context)
    if family == "pyannote":
        return PyannoteAdapter(context)
    raise KeyError(f"no diarizer for {family}")


__all__ = [
    "AsrAdapter",
    "DiarizerAdapter",
    "EngineContext",
    "JobCancelledError",
    "asr_adapter",
    "diarizer_adapter",
]
