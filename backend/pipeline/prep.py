"""ffmpeg/ffprobe helpers and a small WAV-only development fallback.

Production images always ship ffmpeg. The stdlib path exists so the demo can
run on a bare Windows laptop; every use writes a visible warning to the job log.
"""

from __future__ import annotations

import json
import math
import shutil
import struct
import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..jobs import ApiError, timecode_seconds

SAMPLE_RATE = 16000


@dataclass(frozen=True)
class MediaInfo:
    duration_s: float
    sample_rate: int
    channels: int
    format_name: str


def ffmpeg_path() -> str | None:
    return shutil.which("ffmpeg")


def ffprobe_path() -> str | None:
    return shutil.which("ffprobe")


def probe_media(path: Path) -> MediaInfo:
    ffprobe = ffprobe_path()
    if ffprobe:
        try:
            result = subprocess.run(
                [
                    ffprobe,
                    "-v",
                    "error",
                    "-print_format",
                    "json",
                    "-show_format",
                    "-show_streams",
                    str(path),
                ],
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise ApiError("unsupported_media", f"ffprobe could not read the file: {exc}") from exc
        if result.returncode != 0:
            raise ApiError("unsupported_media", "The file is not decodable audio/video.")
        try:
            payload: dict[str, Any] = json.loads(result.stdout)
            duration = float(payload.get("format", {}).get("duration") or 0.0)
            stream = next((s for s in payload.get("streams", []) if s.get("codec_type") == "audio"), None)
            if stream is None:
                raise ApiError("unsupported_media", "No audio stream found in the file.")
            return MediaInfo(
                duration_s=duration,
                sample_rate=int(stream.get("sample_rate") or 0),
                channels=int(stream.get("channels") or 0),
                format_name=str(payload.get("format", {}).get("format_name") or "unknown"),
            )
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            raise ApiError("unsupported_media", "ffprobe returned an unreadable media description.") from exc
    try:
        with wave.open(str(path), "rb") as handle:
            frames = handle.getnframes()
            rate = handle.getframerate()
            return MediaInfo(
                duration_s=frames / rate if rate else 0.0,
                sample_rate=rate,
                channels=handle.getnchannels(),
                format_name="wav",
            )
    except (wave.Error, EOFError) as exc:
        raise ApiError(
            "unsupported_media",
            "ffprobe is not installed here, and only PCM WAV files can be inspected without it.",
        ) from exc


def _warn_stderr(stderr: str) -> str:
    stderr = stderr.strip()
    return stderr[-4000:] if stderr else ""


def prepare_audio(
    source: Path,
    job_dir: Path,
    *,
    start_at: str | None,
    end_at: str | None,
    log: Any,
) -> tuple[Path, MediaInfo, bool]:
    """Produce jobs/{id}/media/prep.wav. Returns (path, info, used_clip)."""
    media_dir = job_dir / "media"
    media_dir.mkdir(parents=True, exist_ok=True)
    prep = media_dir / "prep.wav"
    clipped = bool(start_at or end_at)
    source = source.resolve()
    if not source.exists():
        raise ApiError("storage", "The uploaded media vanished from staging.")

    wave_info = _try_wave_info(source)
    ffmpeg = ffmpeg_path()
    if not ffmpeg:
        if wave_info is None:
            raise ApiError(
                "ffmpeg_failed",
                "ffmpeg is not installed and the source is not a PCM WAV. Install ffmpeg or use a WAV file.",
            )
        log.warning("ffmpeg not found; using the stdlib WAV path (development only).")
        _wav_to_prep(source, prep, start_at, end_at)
        info = probe_media(prep)
        _write_meta(job_dir, info, source.name)
        return prep, info, clipped

    cmd = [ffmpeg, "-hide_banner", "-y"]
    if start_at:
        cmd += ["-ss", start_at]
    cmd += ["-i", str(source), "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE), "-c:a", "pcm_s16le"]
    if end_at:
        start_seconds = timecode_seconds(start_at) if start_at else 0
        clip_seconds = max(0, timecode_seconds(end_at) - start_seconds)
        cmd += ["-t", str(clip_seconds)]
    cmd.append(str(prep))
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise ApiError("ffmpeg_failed", "ffmpeg could not prepare the media.", {"stderr": _warn_stderr(result.stderr)})
    info = probe_media(prep)
    _write_meta(job_dir, info, source.name)
    return prep, info, clipped


def _write_meta(job_dir: Path, info: MediaInfo, original: str) -> None:
    meta = {
        "duration": round(info.duration_s, 3),
        "sample_rate": info.sample_rate,
        "channels": info.channels,
        "original_filename": original,
    }
    (job_dir / "media" / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def _try_wave_info(path: Path) -> MediaInfo | None:
    try:
        with wave.open(str(path), "rb") as handle:
            rate = handle.getframerate()
            return MediaInfo(
                duration_s=handle.getnframes() / rate if rate else 0.0,
                sample_rate=rate,
                channels=handle.getnchannels(),
                format_name="wav",
            )
    except (wave.Error, EOFError, OSError):
        return None


def _decode_pcm(raw: bytes, width: int) -> list[float]:
    if width == 1:
        return [(byte - 128) / 128.0 for byte in raw]
    if width == 2:
        count = len(raw) // 2
        return [value / 32768.0 for value in struct.unpack(f"<{count}h", raw[: count * 2])]
    if width == 3:
        values: list[float] = []
        for index in range(0, len(raw) - 2, 3):
            value = int.from_bytes(raw[index : index + 3], "little", signed=True)
            values.append(value / 8388608.0)
        return values
    if width == 4:
        count = len(raw) // 4
        return [value / 2147483648.0 for value in struct.unpack(f"<{count}i", raw[: count * 4])]
    raise ApiError("unsupported_media", f"Unsupported WAV sample width: {width} bytes.")


def _resample_linear(samples: list[float], src_rate: int, dst_rate: int) -> list[float]:
    if src_rate == dst_rate or not samples:
        return samples
    ratio = dst_rate / src_rate
    out_count = max(1, int(len(samples) * ratio))
    output = [0.0] * out_count
    for index in range(out_count):
        position = index / ratio
        left = int(position)
        right = min(left + 1, len(samples) - 1)
        frac = position - left
        output[index] = samples[left] * (1 - frac) + samples[right] * frac
    return output


def _wav_to_prep(source: Path, prep: Path, start_at: str | None, end_at: str | None) -> None:
    with wave.open(str(source), "rb") as handle:
        channels = handle.getnchannels()
        width = handle.getsampwidth()
        rate = handle.getframerate()
        start = timecode_seconds(start_at) if start_at else 0
        end = timecode_seconds(end_at) if end_at else None
        position = 0
        if start:
            position = min(handle.getnframes(), int(start * rate))
            handle.setpos(position)
        frames = handle.getnframes() - position
        if end is not None:
            frames = min(frames, max(0, int((end - start) * rate)))
        raw = handle.readframes(frames)

    samples = _decode_pcm(raw, width)
    if channels > 1:
        mono: list[float] = []
        for index in range(0, len(samples) - channels + 1, channels):
            mono.append(sum(samples[index : index + channels]) / channels)
        samples = mono
    samples = _resample_linear(samples, rate, SAMPLE_RATE)
    pcm = b"".join(struct.pack("<h", max(-32768, min(32767, int(sample * 32767)))) for sample in samples)
    with wave.open(str(prep), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(pcm)


def duration_of(path: Path) -> float:
    return probe_media(path).duration_s


def estimate_seconds(timecode: str) -> float:
    return float(timecode_seconds(timecode))


def rounded(value: float, places: int = 2) -> float:
    factor = 10**places
    return float(math.floor(value * factor + 0.5) / factor)
