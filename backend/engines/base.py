"""Adapter contracts and shared context (docs/01 §5)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from ..catalog import Catalog, ModelSpec
from ..config import EnvConfig, Paths
from ..jobs import JobSettings
from ..pipeline.types import DiarResult, TranscriptionResult

ProgressFn = Callable[[float, str], None]
CancelFn = Callable[[], bool]

MODEL_DIRS = {
    "faster-whisper": "fastwhisper",
    "whisper.cpp": "whispercpp",
    "nemo-speech": "nemo",
    "qwen3-asr": "qwen3",
    "qwen3-aligner": "qwen3",
    "pyannote": "pyannote",
    "demo": "_demo",
}


@dataclass(frozen=True)
class EngineContext:
    paths: Paths
    env: EnvConfig
    catalog: Catalog
    hf_token: str | None = None

    def model_dir(self, model: ModelSpec) -> Path:
        folder = MODEL_DIRS.get(model.engine, model.engine)
        return self.paths.models / folder / model.id


class JobCancelledError(Exception):
    """Raised cooperatively inside adapters when the user cancels a job."""


class AsrAdapter(Protocol):
    name: str

    def transcribe(
        self,
        audio_path: Path,
        settings: JobSettings,
        model: ModelSpec,
        *,
        device: str,
        compute: str,
        media_id: str,
        progress: ProgressFn,
        log_path: Path,
        cancel: CancelFn,
    ) -> TranscriptionResult: ...


class DiarizerAdapter(Protocol):
    name: str

    def diarize(
        self,
        audio_path: Path,
        model: ModelSpec,
        *,
        device: str,
        speaker_count: int | str,
        progress: ProgressFn,
        log_path: Path,
        cancel: CancelFn,
    ) -> DiarResult: ...


def find_binary(env_name: str, binary: str) -> str | None:
    import os
    import shutil

    override = os.environ.get(env_name, "")
    if override and Path(override).exists():
        return override
    return shutil.which(binary)
