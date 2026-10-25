"""NeMo-Speech.cpp adapter: Nemotron ASR and Sortformer/Nemotron diarization."""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

from ...jobs import ApiError, JobSettings
from ...pipeline.prep import probe_media
from ...pipeline.types import DiarResult, DiarTurn, Segment, TranscriptionResult, Word
from ..base import CancelFn, EngineContext, JobCancelledError, ProgressFn, find_binary
from . import args

PERCENT_RE = re.compile(r"(\d{1,3})%")


class NemoSpeechAdapter:
    name = "nemo-speech"

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
        binary = find_binary("WS_NEMO_SPEECH", "nemo-speech")
        if not binary:
            raise ApiError("engine_crash", "nemo-speech is not installed in this image.")
        model_path = _find_model(self.context.model_dir(model))
        if model_path is None:
            raise ApiError("engine_crash", f"No GGUF model found under {self.context.model_dir(model)}.")
        language = self._locale(settings.language)
        cmd = args.build_transcribe_command(
            binary=binary,
            audio_path=audio_path,
            model_path=model_path,
            device=device,
            language=language,
        )
        payload = _run_json(cmd, log_path, cancel, progress, "asr")
        segments = _segments_from_payload(payload)
        if not segments:
            raise ApiError("engine_crash", "nemo-speech produced no transcript segments.")
        duration = _float(payload.get("duration")) or probe_media(audio_path).duration_s
        return TranscriptionResult(
            engine="nemo-speech",
            model_id=model.id,
            device=device,
            compute=compute,
            language=str(payload.get("language") or (None if language == "auto" else language)),
            language_prob=None,
            duration_s=duration,
            media_id=media_id,
            segments=segments,
        )

    def diarize(
        self,
        audio_path: Path,
        model: Any,
        *,
        device: str,
        speaker_count: int | str,
        progress: ProgressFn,
        log_path: Path,
        cancel: CancelFn,
    ) -> DiarResult:
        binary = find_binary("WS_NEMO_SPEECH", "nemo-speech")
        if not binary:
            raise ApiError("engine_crash", "nemo-speech is not installed in this image.")
        model_path = _find_model(self.context.model_dir(model))
        if model_path is None:
            raise ApiError("engine_crash", f"No GGUF diarization model found under {self.context.model_dir(model)}.")
        cmd = args.build_diarize_command(
            binary=binary,
            audio_path=audio_path,
            model_path=model_path,
            device=device,
        )
        payload = _run_json(cmd, log_path, cancel, progress, "diarize")
        raw_turns = _raw_turns(payload)
        turns = _normalize_turns(raw_turns)
        if speaker_count != "auto" and model.max_speakers:
            cap = min(int(speaker_count), model.max_speakers)
            turns = [turn for turn in turns if turn.speaker < cap]
        return DiarResult(
            engine="nemo-speech",
            model_id=model.id,
            device=device,
            turns=turns,
        )

    def _locale(self, language: str) -> str:
        if language == "auto":
            return "auto"
        if "-" in language:
            return language
        return language


def _find_model(directory: Path) -> Path | None:
    if not directory.exists():
        return None
    matches = sorted(directory.glob("*.gguf"))
    return matches[0] if matches else None


def _run_json(
    cmd: list[str],
    log_path: Path,
    cancel: CancelFn,
    progress: ProgressFn,
    stage: str,
) -> dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    captured: list[str] = []
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
                captured.append(line)
                if cancel():
                    _terminate(process)
                    raise JobCancelledError
                match = PERCENT_RE.search(line)
                if match:
                    progress(min(0.98, int(match.group(1)) / 100), line.strip()[:120])
            returncode = process.wait()
        finally:
            if process.poll() is None:
                _terminate(process)
    if returncode != 0:
        tail = "".join(captured)[-2000:]
        raise ApiError("engine_crash", "nemo-speech exited unsuccessfully.", {"returncode": returncode, "stderr": tail})
    payload = _extract_json("".join(captured))
    if payload is None:
        raise ApiError("engine_crash", "nemo-speech did not return readable JSON.", {"stderr": "".join(captured)[-2000:]})
    progress(1.0, f"{stage} complete")
    return payload


def _extract_json(text: str) -> dict[str, Any] | None:
    text = text.strip()
    if not text:
        return None
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        pass

    # Engine diagnostics (for example the Vulkan shader-cache warning) share
    # stderr with the JSON document.  Scan for the first complete object instead
    # of accepting the first single-line object: NeMo emits each word as its own
    # line, so a line-based fallback can accidentally return only a word record.
    decoder = json.JSONDecoder()
    fallback: dict[str, Any] | None = None
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text, index)
        except json.JSONDecodeError:
            continue
        if not isinstance(value, dict):
            continue
        if any(key in value for key in ("text", "segments", "transcription", "words", "file")):
            return value
        if fallback is None:
            fallback = value
    return fallback


def _segments_from_payload(payload: dict[str, Any]) -> list[Segment]:
    raw = payload.get("segments") or payload.get("transcription") or []
    segments: list[Segment] = []
    for item in raw:
        start, end = _times(item)
        text = str(item.get("text") or item.get("transcript") or "").strip()
        words: list[Word] = []
        for token in item.get("words") or item.get("tokens") or []:
            token_start, token_end = _times(token)
            token_text = str(token.get("text") or token.get("word") or "").strip()
            if token_text:
                words.append(
                    Word(
                        start=token_start,
                        end=token_end,
                        text=token_text,
                        probability=_float(token.get("probability") or token.get("p")),
                    )
                )
        if text or words:
            segments.append(Segment(start=start, end=end, text=text, words=words or None))
    if not segments and isinstance(payload.get("text"), str):
        segments.append(Segment(start=0.0, end=float(payload.get("duration") or 0.0), text=str(payload["text"])))
    return segments


def _raw_turns(payload: dict[str, Any]) -> list[Any]:
    for key in ("turns", "segments", "speaker_turns", "diarization"):
        value = payload.get(key)
        if isinstance(value, list):
            return value
    return []


def _normalize_turns(raw_turns: list[Any]) -> list[DiarTurn]:
    prepared: list[tuple[float, float, Any, bool]] = []
    for item in raw_turns:
        if not isinstance(item, dict):
            continue
        start, end = _times(item)
        speaker = item.get("speaker", item.get("speaker_id", item.get("label", 0)))
        prepared.append((start, end, speaker, bool(item.get("overlap", False))))
    prepared.sort(key=lambda row: (row[0], row[1]))
    labels: dict[Any, int] = {}
    turns: list[DiarTurn] = []
    for start, end, speaker, overlap in prepared:
        key = str(speaker)
        if key not in labels:
            labels[key] = len(labels)
        turns.append(DiarTurn(start=start, end=end, speaker=labels[key], overlap=overlap))
    return turns


def _times(item: dict[str, Any]) -> tuple[float, float]:
    if "offsets" in item and isinstance(item["offsets"], dict):
        return _float(item["offsets"].get("from")) or 0.0, _float(item["offsets"].get("to")) or 0.0
    start = _float(item.get("start", item.get("t0", item.get("start_time")))) or 0.0
    end = _float(item.get("end", item.get("t1", item.get("end_time")))) or 0.0
    if end > 1000 and start > 1000:
        start /= 1000.0
        end /= 1000.0
    return start, end


def _float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _terminate(process: subprocess.Popen[str]) -> None:
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
