"""whisper.cpp adapter (subprocess)."""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

from ...jobs import ApiError, JobSettings
from ...pipeline.types import Segment, TranscriptionResult, Word
from ..base import CancelFn, EngineContext, JobCancelledError, ProgressFn, find_binary
from . import args

PERCENT_RE = re.compile(r"(\d{1,3})%")


class WhisperCppAdapter:
    name = "whisper.cpp"

    def __init__(self, context: EngineContext) -> None:
        self.context = context

    def transcribe(
        self,
        audio_path: Path,
        settings: JobSettings,
        model: Any,
        *,
        device: str,
        compute: str,
        media_id: str,
        progress: ProgressFn,
        log_path: Path,
        cancel: CancelFn,
    ) -> TranscriptionResult:
        binary = find_binary("WS_WHISPER_CLI", "whisper-cli")
        if not binary:
            raise ApiError("engine_crash", "whisper-cli is not installed in this image.")
        model_dir = self.context.model_dir(model)
        model_path = _find_model(model_dir, ".bin")
        if model_path is None:
            raise ApiError("engine_crash", f"No GGML model found under {model_dir}. Download it first.")
        output_prefix = log_path.parent / f"{audio_path.stem}.wcpp"
        cmd = args.build_command(
            binary=binary,
            model_path=model_path,
            audio_path=audio_path,
            output_prefix=output_prefix,
            language=settings.language,
            device=device,
            threads=max(2, (os.cpu_count() or 4) // 2),
            word_timestamps=settings.word_timestamps or settings.mark_pauses,
            verbatim=settings.verbatim,
        )
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as log:
            log.write("$ " + " ".join(cmd) + "\n")
            log.flush()
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=os.environ.copy(),
            )
            assert process.stdout is not None
            try:
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    if cancel():
                        _terminate(process)
                        raise JobCancelledError
                    match = PERCENT_RE.search(line)
                    if match:
                        progress(min(0.97, int(match.group(1)) / 100), line.strip()[:120])
                returncode = process.wait()
            finally:
                if process.poll() is None:
                    _terminate(process)
        if returncode != 0:
            tail = log_path.read_text(encoding="utf-8", errors="replace")[-2000:]
            raise ApiError("engine_crash", "whisper-cli exited unsuccessfully.", {"returncode": returncode, "stderr": tail})

        payload = _read_json(output_prefix)
        segments = _segments_from_payload(payload)
        if not segments:
            raise ApiError("engine_crash", "whisper-cli produced no transcript segments.")
        progress(1.0, "asr complete")
        return TranscriptionResult(
            engine="whisper.cpp",
            model_id=model.id,
            device=device,
            compute=compute,
            language=payload.get("result", {}).get("language") if isinstance(payload.get("result"), dict) else None,
            language_prob=None,
            duration_s=probe_duration(payload),
            media_id=media_id,
            segments=segments,
        )


def _find_model(directory: Path, suffix: str) -> Path | None:
    if not directory.exists():
        return None
    matches = sorted(directory.glob(f"*{suffix}"))
    return matches[0] if matches else None


def _read_json(prefix: Path) -> dict[str, Any]:
    candidates = [
        prefix.with_suffix(".json"),
        Path(str(prefix) + ".json"),
        *sorted(prefix.parent.glob(f"{prefix.name}*.json")),
    ]
    for candidate in candidates:
        if candidate.exists():
            try:
                value = json.loads(candidate.read_text(encoding="utf-8"))
                if isinstance(value, dict):
                    return value
            except json.JSONDecodeError as exc:
                raise ApiError("engine_crash", f"whisper-cli JSON could not be parsed: {exc}") from exc
    raise ApiError("engine_crash", "whisper-cli did not write its JSON output file.")


def _segments_from_payload(payload: dict[str, Any]) -> list[Segment]:
    raw_segments = payload.get("transcription") or payload.get("segments") or []
    segments: list[Segment] = []
    for raw in raw_segments:
        offsets = raw.get("offsets") or {}
        start = _ms(offsets.get("from", raw.get("t0", 0)))
        end = _ms(offsets.get("to", raw.get("t1", 0)))
        text = str(raw.get("text", "")).strip()
        tokens = raw.get("tokens") or raw.get("words") or []
        words: list[Word] = []
        for token in tokens:
            token_offsets = token.get("offsets") or {}
            token_start = token_offsets.get("from", token.get("t0"))
            token_end = token_offsets.get("to", token.get("t1"))
            token_text = str(token.get("text", "")).strip()
            if not token_text:
                continue
            words.append(
                Word(
                    start=_ms(token_start) if token_start is not None else None,
                    end=_ms(token_end) if token_end is not None else None,
                    text=token_text,
                    probability=token.get("p"),
                )
            )
        segments.append(Segment(start=start, end=end, text=text, words=words or None))
    return segments


def _ms(value: Any) -> float:
    try:
        return float(value) / 1000.0
    except (TypeError, ValueError):
        return 0.0


def probe_duration(payload: dict[str, Any]) -> float:
    result = payload.get("result")
    if isinstance(result, dict):
        try:
            return float(result.get("duration") or 0.0)
        except (TypeError, ValueError):
            return 0.0
    return 0.0


def _terminate(process: subprocess.Popen[str]) -> None:
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
