"""NeMo-Speech.cpp argv construction. Exact output schema is SPIKE S2."""

from __future__ import annotations

from pathlib import Path

NEMOSPEECH_TAG = "v0.2.0"


def device_arg(device: str) -> str:
    if device == "vulkan":
        return "vulkan:0"
    if device == "cuda":
        return "cuda:0"
    return "cpu"


def build_transcribe_command(
    *,
    binary: str,
    audio_path: Path,
    model_path: Path,
    device: str,
    language: str,
) -> list[str]:
    cmd = [
        binary,
        "transcribe",
        str(audio_path),
        "--model",
        str(model_path),
        "--json",
        "--device",
        device_arg(device),
    ]
    if language != "auto":
        cmd += ["--language", language]
    return cmd


def build_diarize_command(
    *,
    binary: str,
    audio_path: Path,
    model_path: Path,
    device: str,
) -> list[str]:
    return [
        binary,
        "diarize",
        str(audio_path),
        "--model",
        str(model_path),
        "--json",
        "--device",
        device_arg(device),
    ]
