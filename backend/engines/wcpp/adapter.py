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
SPECIAL_TOKEN_RE = re.compile(r"^\[_[A-Za-z0-9_]+\]$")
SPECIAL_TOKEN_ANY_RE = re.compile(r"\[_[A-Za-z0-9_]+\]")
# Scripts that do not put spaces between words.  Merging their tokens would
# collapse a whole segment into one caption word, so keep the token pieces.
UNSPACED_SCRIPT_RE = re.compile(
    r"[\u0e00-\u0eff\u1000-\u109f\u1780-\u17ff\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]"
)
# whisper.cpp starts its special vocabulary at EOT (50256 on the pinned
# whisper models).  Full JSON includes those tokens even though the segment
# text leaves them out.
SPECIAL_TOKEN_ID = 50256


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
        text = _clean_text(str(raw.get("text", "")))
        tokens = raw.get("tokens") or raw.get("words") or []
        words = _words_from_tokens(tokens)
        if text or words:
            segments.append(Segment(start=start, end=end, text=text, words=words or None))
    return segments


def _words_from_tokens(tokens: Any) -> list[Word]:
    """Group whisper.cpp subword tokens into words.

    The tokenizer marks a new word with a leading space.  Punctuation and
    contractions usually arrive as unspaced pieces, so appending them keeps
    ``Don't``, ``foot-washing``, and ``unimaginable`` intact.
    """
    if not isinstance(tokens, list):
        return []
    words: list[Word] = []
    current: Word | None = None
    for token in tokens:
        if not isinstance(token, dict) or _is_special_token(token):
            continue
        token_text = str(token.get("text", "")).replace("\n", " ")
        if not token_text or not token_text.strip():
            continue
        token_offsets = token.get("offsets") or {}
        token_start = token_offsets.get("from", token.get("t0"))
        token_end = token_offsets.get("to", token.get("t1"))
        start = _ms(token_start) if token_start is not None else None
        end = _ms(token_end) if token_end is not None else None
        probability = token.get("p")
        if token_text[0].isspace():
            if current is not None:
                words.append(current)
            current = Word(
                start=start,
                end=end,
                text=token_text.lstrip(),
                probability=probability,
            )
        elif current is None:
            current = Word(
                start=start,
                end=end,
                text=token_text.lstrip(),
                probability=probability,
            )
        elif UNSPACED_SCRIPT_RE.search(current.text) or UNSPACED_SCRIPT_RE.search(token_text):
            words.append(current)
            current = Word(
                start=start,
                end=end,
                text=token_text.lstrip(),
                probability=probability,
            )
        else:
            current.text += token_text
            if end is not None:
                current.end = end
            if probability is not None and (
                current.probability is None or probability < current.probability
            ):
                current.probability = probability
    if current is not None:
        words.append(current)
    for word in words:
        word.text = word.text.strip()
    return [word for word in words if word.text]


def _is_special_token(token: dict[str, Any]) -> bool:
    token_id = token.get("id")
    if isinstance(token_id, int) and token_id >= SPECIAL_TOKEN_ID:
        return True
    return bool(SPECIAL_TOKEN_RE.fullmatch(str(token.get("text", "")).strip()))


def _clean_text(text: str) -> str:
    return " ".join(SPECIAL_TOKEN_ANY_RE.sub(" ", text).split())


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
