"""Verify the pinned faster-whisper + PyAV stack at image-build time.

faster-whisper 1.2.1's ``decode_audio()`` calls
``av.open(..., metadata_errors="ignore")``.  PyAV 19 removed that keyword, and
faster-whisper only requires ``av>=11``, so a fresh resolution picked 19.0.2
and every job failed in decode before transcription started.  This probe
exercises the exact decode path on a generated WAV so a broken image fails at
build time instead of on the first job.
"""

from __future__ import annotations

import importlib
import math
import struct
import sys
import tempfile
import wave
from pathlib import Path

FASTER_WHISPER_VERSION = "1.2.1"
MAX_AV_MAJOR = 18
PROBE_SAMPLE_RATE = 16_000
PROBE_FRAMES = 3_200


def _module_version(name: str) -> str:
    module = importlib.import_module(name)
    version = getattr(module, "__version__", None)
    if not isinstance(version, str) or not version:
        raise RuntimeError(f"{name} does not expose __version__")
    return version


def _av_major(version: str) -> int:
    head = version.split(".", 1)[0]
    if not head.isdigit():
        raise RuntimeError(f"av version {version!r} is not numeric")
    return int(head)


def version_errors(*, faster_whisper_version: str, av_version: str) -> list[str]:
    errors: list[str] = []
    if faster_whisper_version != FASTER_WHISPER_VERSION:
        errors.append(
            f"faster-whisper {faster_whisper_version} is not {FASTER_WHISPER_VERSION}"
        )
    if _av_major(av_version) > MAX_AV_MAJOR:
        errors.append(
            f"av {av_version} removed metadata_errors= from av.open(); "
            f"faster-whisper {FASTER_WHISPER_VERSION} needs av <= {MAX_AV_MAJOR}"
        )
    return errors


def check_versions() -> dict[str, str]:
    versions = {
        "faster-whisper": _module_version("faster_whisper"),
        "av": _module_version("av"),
        "ctranslate2": _module_version("ctranslate2"),
    }
    errors = version_errors(
        faster_whisper_version=versions["faster-whisper"],
        av_version=versions["av"],
    )
    if errors:
        raise RuntimeError("; ".join(errors))
    return versions


def probe_decode() -> None:
    from faster_whisper.audio import decode_audio  # type: ignore[import-not-found]

    with tempfile.TemporaryDirectory(prefix="ws-fw-probe-") as temp:
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
        samples = decode_audio(str(path))

    shape = getattr(samples, "shape", None)
    if shape != (PROBE_FRAMES,):
        raise RuntimeError(f"decode_audio returned {shape}, expected ({PROBE_FRAMES},)")
    if float(abs(samples).max()) <= 0:
        raise RuntimeError("decode_audio returned a silent probe WAV")


def main() -> int:
    try:
        versions = check_versions()
        probe_decode()
    except Exception as exc:
        print(f"faster-whisper runtime probe failed: {exc}", file=sys.stderr)
        return 1
    print(
        "faster-whisper runtime probe ok: "
        f"faster-whisper {versions['faster-whisper']}, av {versions['av']}, "
        f"ctranslate2 {versions['ctranslate2']}, PyAV WAV decode"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
