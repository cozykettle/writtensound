"""pyannote pipeline knobs (SPIKE S5)."""

from __future__ import annotations

PYANNOTE_AUDIO_MIN = "4.0"


def speaker_bounds(speaker_count: int | str) -> tuple[int | None, int | None]:
    if speaker_count == "auto":
        return None, None
    count = int(speaker_count)
    return count, count
