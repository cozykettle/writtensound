"""16 kHz PCM helpers shared by the Qwen3 engines.

``prep.wav`` is always mono 16-bit PCM at 16 kHz (docs/04 §1).  Passing numpy
windows to the processor keeps transformers away from its file/URL audio
loader, which is the path that selected torchcodec for a plain WAV file.
"""

from __future__ import annotations

import wave
from pathlib import Path
from typing import Any

from ...jobs import ApiError

SAMPLE_RATE = 16_000

_FORMAT_ERROR = "Qwen3 expects mono {rate} Hz 16-bit PCM audio"


def _check_format(reader: wave.Wave_read, path: Path) -> None:
    channels = reader.getnchannels()
    rate = reader.getframerate()
    width = reader.getsampwidth()
    if channels != 1 or rate != SAMPLE_RATE or width != 2:
        raise ApiError(
            "engine_crash",
            f"{_FORMAT_ERROR.format(rate=SAMPLE_RATE)}; {path.name} is "
            f"{channels} channel(s), {rate} Hz, {width * 8}-bit.",
        )


def duration_s(path: Path) -> float:
    """Return the duration of a validated prep WAV."""
    try:
        with wave.open(str(path), "rb") as reader:
            _check_format(reader, path)
            frames = reader.getnframes()
            rate = reader.getframerate()
    except ApiError:
        raise
    except (OSError, EOFError, wave.Error) as exc:
        raise ApiError("engine_crash", f"Qwen3 could not read {path.name}: {exc}") from exc
    return frames / float(rate)


def read_window(path: Path, start_s: float = 0.0, end_s: float | None = None) -> Any:
    """Read ``[start_s, end_s)`` as float32 mono samples in [-1, 1].

    Bounds are clamped to the file.  Numpy is imported lazily so the web app and
    the non-GPU test suite can run without the engines extra.
    """
    import numpy as np

    try:
        with wave.open(str(path), "rb") as reader:
            _check_format(reader, path)
            rate = reader.getframerate()
            total = reader.getnframes()
            first = min(total, max(0, int(round(start_s * rate))))
            last = total if end_s is None else min(total, max(first, int(round(end_s * rate))))
            reader.setpos(first)
            payload = reader.readframes(last - first)
    except ApiError:
        raise
    except (OSError, EOFError, wave.Error) as exc:
        raise ApiError("engine_crash", f"Qwen3 could not read {path.name}: {exc}") from exc

    if not payload:
        return np.zeros(0, dtype=np.float32)
    return np.frombuffer(payload, dtype="<i2").astype(np.float32) / 32768.0
