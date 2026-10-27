"""Verify the pinned Qwen3 stack at image-build time.

v0.3.0 shipped torch 2.9 with torchcodec 0.17 (pulled in by pyannote), whose
shared libraries need torch >= 2.11.  The failure only appeared when a model
job tried to decode a WAV, so this probe exercises the exact decode path and
the optional Japanese/Korean splitters before an image can be published.
"""

from __future__ import annotations

import importlib
import math
import struct
import sys
import tempfile
import wave
from pathlib import Path

TRANSFORMERS_VERSION = "5.20.2"
TORCH_SERIES = (2, 9)
TORCHAUDIO_SERIES = (2, 9)
TORCHCODEC_VERSION = "0.9.1"

PROBE_SAMPLE_RATE = 16_000
PROBE_FRAMES = 3_200


def _release(version: str) -> str:
    return version.split("+", 1)[0]


def _series(version: str) -> tuple[int, ...]:
    parts: list[int] = []
    for item in _release(version).split("."):
        if not item.isdigit():
            break
        parts.append(int(item))
    return tuple(parts[:2])


def version_errors(
    *,
    torch_version: str,
    torchcodec_version: str,
    transformers_version: str,
    torchaudio_version: str,
) -> list[str]:
    errors: list[str] = []
    if _series(torch_version) != TORCH_SERIES:
        errors.append(f"torch {torch_version} is not the pinned {'.'.join(map(str, TORCH_SERIES))}.x series")
    if _series(torchaudio_version) != TORCHAUDIO_SERIES:
        errors.append(
            f"torchaudio {torchaudio_version} does not match torch {'.'.join(map(str, TORCHAUDIO_SERIES))}.x"
        )
    if _release(torchcodec_version) != TORCHCODEC_VERSION:
        errors.append(
            f"torchcodec {torchcodec_version} is not {TORCHCODEC_VERSION} "
            f"(newer releases require torch >= 2.11; torch here is {torch_version})"
        )
    if _release(transformers_version) != TRANSFORMERS_VERSION:
        errors.append(f"transformers {transformers_version} is not {TRANSFORMERS_VERSION}")
    return errors


def _module_version(name: str) -> str:
    module = importlib.import_module(name)
    version = getattr(module, "__version__", None)
    if not isinstance(version, str) or not version:
        raise RuntimeError(f"{name} does not expose __version__")
    return version


def check_versions() -> dict[str, str]:
    versions = {
        "torch": _module_version("torch"),
        "torchaudio": _module_version("torchaudio"),
        "torchcodec": _module_version("torchcodec"),
        "transformers": _module_version("transformers"),
    }
    errors = version_errors(
        torch_version=versions["torch"],
        torchcodec_version=versions["torchcodec"],
        transformers_version=versions["transformers"],
        torchaudio_version=versions["torchaudio"],
    )
    if errors:
        raise RuntimeError("; ".join(errors))
    return versions


def probe_qwen_symbols() -> None:
    import transformers  # type: ignore[import-not-found]

    for name in ("AutoModelForMultimodalLM", "AutoModelForTokenClassification"):
        if not hasattr(transformers, name):
            raise RuntimeError(f"transformers {transformers.__version__} is missing {name}")
    from transformers.models.qwen3_asr.processing_qwen3_asr import (  # type: ignore[import-not-found]
        FORCED_ALIGNER_LANGUAGES,
        LANGUAGE_CODE_TO_NAME,
    )

    if not LANGUAGE_CODE_TO_NAME:
        raise RuntimeError("Qwen3-ASR language table is empty")
    if not FORCED_ALIGNER_LANGUAGES:
        raise RuntimeError("Qwen3 forced-aligner language table is empty")


def probe_audio_decode() -> None:
    import numpy as np
    from transformers.audio_utils import load_audio  # type: ignore[import-not-found]

    with tempfile.TemporaryDirectory(prefix="ws-qwen-probe-") as temp:
        path = Path(temp) / "probe.wav"
        with wave.open(str(path), "wb") as writer:
            writer.setnchannels(1)
            writer.setsampwidth(2)
            writer.setframerate(PROBE_SAMPLE_RATE)
            frames = b"".join(
                struct.pack("<h", int(12_000 * math.sin(2 * math.pi * 440 * index / PROBE_SAMPLE_RATE)))
                for index in range(PROBE_FRAMES)
            )
            writer.writeframes(frames)
        samples = load_audio(str(path), sampling_rate=PROBE_SAMPLE_RATE, backend="torchcodec")

    if not isinstance(samples, np.ndarray):
        raise RuntimeError(f"torchcodec returned {type(samples).__name__}, expected numpy.ndarray")
    if samples.shape != (PROBE_FRAMES,):
        raise RuntimeError(f"torchcodec decoded {samples.shape}, expected ({PROBE_FRAMES},)")
    if float(np.abs(samples).max()) <= 0:
        raise RuntimeError("torchcodec decoded a silent probe WAV")


def probe_word_splitters() -> None:
    import nagisa  # type: ignore[import-not-found]
    import soynlp  # type: ignore[import-not-found]

    if not nagisa.tagging("これはテストです").words:
        raise RuntimeError("nagisa could not split Japanese text")
    if not soynlp.__version__:
        raise RuntimeError("soynlp is missing __version__")


def probe_pyannote() -> None:
    import pyannote.audio  # type: ignore[import-not-found]  # noqa: F401


def main() -> int:
    try:
        versions = check_versions()
        probe_qwen_symbols()
        probe_audio_decode()
        probe_word_splitters()
        probe_pyannote()
    except Exception as exc:
        print(f"Qwen3 runtime probe failed: {exc}", file=sys.stderr)
        return 1
    print(
        "Qwen3 runtime probe ok: "
        f"transformers {versions['transformers']}, torch {versions['torch']}, "
        f"torchaudio {versions['torchaudio']}, torchcodec {versions['torchcodec']}, "
        "nagisa, soynlp, pyannote.audio, torchcodec WAV decode"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
