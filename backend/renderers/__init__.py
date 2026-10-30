"""Output renderers (docs/04 §5): TXT, HTML, SRT, VTT, timed text, JSON."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape

from ..jobs import JobSettings, TimestampsMode
from ..pipeline.types import DiarResult, MergedSegment, Pause, TranscriptionResult

TEMPLATE_DIR = Path(__file__).with_name("templates")


@dataclass
class Paragraph:
    text: str
    timestamp: str | None = None
    pause: str | None = None
    overlap: bool = False


@dataclass
class Turn:
    speaker: int | None
    start: float
    timestamp: str | None
    paragraphs: list[Paragraph]


def hhmmss(seconds: float) -> str:
    total = max(0, int(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def srt_ts(seconds: float) -> str:
    millis = max(0, int(round(seconds * 1000)))
    hours, remainder = divmod(millis, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, ms = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def vtt_ts(seconds: float) -> str:
    return srt_ts(seconds).replace(",", ".")


def _safe_base(settings: JobSettings, transcription: TranscriptionResult) -> str:
    base = settings.out_basename or transcription.media_id or "transcript"
    cleaned = "".join(char for char in base if char.isalnum() or char in " -_().[]")
    return cleaned.strip() or "transcript"


def _pause_marker(seconds: float) -> str:
    dots = min(5, max(1, int(math.ceil(seconds))))
    text = "(" + " ".join("." for _ in range(dots)) + ")"
    if seconds > 5:
        text += f" (+{int(round(seconds))} s)"
    return text


def _is_turn_start(previous: MergedSegment | None, segment: MergedSegment) -> bool:
    return previous is None or previous.speaker != segment.speaker


def _paragraphs_for_segment(
    segment: MergedSegment,
    settings: JobSettings,
    *,
    turn_start: bool,
    last_mark: float,
) -> tuple[list[Paragraph], float]:
    """Split interval markers at the first word past each threshold.

    ASR output can be one long segment (NeMo often returns one segment with
    top-level words).  A segment-only timestamp would therefore put a single
    marker on a 26-minute transcript, so interval mode cuts at word boundaries.
    """
    words = (
        [word for word in segment.words or [] if word.text.strip()]
        if settings.timestamps_mode == TimestampsMode.interval
        else []
    )
    if not words:
        marker: str | None = None
        turn_mark = turn_start and settings.timestamps_mode != TimestampsMode.none
        interval_due = (
            settings.timestamps_mode == TimestampsMode.interval
            and segment.start >= last_mark + settings.interval_s
        )
        if turn_mark or interval_due:
            marker = hhmmss(segment.start)
            last_mark = segment.start
        return [
            Paragraph(text=segment.text.strip(), timestamp=marker, overlap=segment.overlap)
        ], last_mark

    chunks: list[tuple[str | None, str]] = []
    current: list[Any] = []
    current_timestamp: str | None = None
    first = True
    for word in words:
        start = word.start if word.start is not None else segment.start
        due = (first and turn_start) or start >= last_mark + settings.interval_s
        if current and due:
            chunks.append((current_timestamp, " ".join(item.text.strip() for item in current)))
            current = []
            current_timestamp = None
        if not current:
            if due:
                marker_at = segment.start if first and turn_start else start
                current_timestamp = hhmmss(marker_at)
                last_mark = marker_at
            else:
                current_timestamp = None
        current.append(word)
        first = False
    if current:
        chunks.append((current_timestamp, " ".join(item.text.strip() for item in current)))

    # Keep the engine's original segment text when the interval does not cut
    # the segment; use word text only when a split is necessary.
    if len(chunks) == 1:
        return [
            Paragraph(
                text=segment.text.strip() or chunks[0][1],
                timestamp=chunks[0][0],
                overlap=segment.overlap,
            )
        ], last_mark

    return [
        Paragraph(text=text, timestamp=timestamp, overlap=segment.overlap)
        for timestamp, text in chunks
        if text
    ], last_mark


def _pause_for(segment: MergedSegment, pauses: list[Pause], consumed: set[int]) -> str | None:
    for index, pause in enumerate(pauses):
        if index not in consumed and segment.start - 0.01 <= pause.start <= segment.end + 0.05:
            consumed.add(index)
            return _pause_marker(pause.seconds)
    # A pause between two segments belongs to the sentence that follows it.
    for index, pause in enumerate(pauses):
        if index in consumed:
            continue
        if pause.start < segment.start and abs(segment.start - (pause.start + pause.seconds)) <= 0.2:
            consumed.add(index)
            return _pause_marker(pause.seconds)
    return None


def build_turns(
    segments: list[MergedSegment],
    pauses: list[Pause],
    settings: JobSettings,
) -> list[Turn]:
    turns: list[Turn] = []
    consumed: set[int] = set()
    last_mark = 0.0
    previous: MergedSegment | None = None
    for segment in segments:
        turn_start = _is_turn_start(previous, segment)
        if not turns or turns[-1].speaker != segment.speaker:
            turns.append(
                Turn(
                    speaker=segment.speaker,
                    start=segment.start,
                    timestamp=None,
                    paragraphs=[],
                )
            )
        paragraphs, last_mark = _paragraphs_for_segment(
            segment,
            settings,
            turn_start=turn_start,
            last_mark=last_mark,
        )
        if paragraphs:
            pause = _pause_for(segment, pauses, consumed)
            if pause:
                paragraphs[-1].pause = pause
            if turns[-1].timestamp is None and segment.speaker is not None:
                turns[-1].timestamp = next(
                    (paragraph.timestamp for paragraph in paragraphs if paragraph.timestamp),
                    None,
                )
            turns[-1].paragraphs.extend(paragraphs)
        previous = segment
    if not turns:
        turns.append(Turn(speaker=None, start=0.0, timestamp=None, paragraphs=[]))
    return turns


def render_txt(segments: list[MergedSegment], pauses: list[Pause], settings: JobSettings) -> str:
    turns = build_turns(segments, pauses, settings)
    blocks: list[str] = []
    for turn in turns:
        paragraphs: list[str] = []
        for paragraph in turn.paragraphs:
            prefix = ""
            if paragraph.timestamp:
                prefix += f"[{paragraph.timestamp}] "
            if paragraph.pause:
                prefix += f"{paragraph.pause} "
            paragraphs.append(prefix + paragraph.text)
        block = "\n".join(paragraphs)
        if len(turns) > 1 and turn.speaker is not None:
            block = f"Speaker {turn.speaker + 1}:\n{block}"
        blocks.append(block)
    return "\n\n".join(blocks).strip() + "\n"


def _reflow_html_turns(turns: list[Turn]) -> list[Turn]:
    """Merge tiny engine segments into readable HTML paragraphs.

    whisper.cpp emits progressively shorter segments late in a file, which made
    the HTML transcript look like a poem.  Merge only paragraphs that carry no
    timestamp, pause marker, or overlap change, so every explicit break and
    inline marker survives.  Turn and speaker boundaries are never crossed.
    """
    target_words = 80
    target_chars = 560
    reflowed: list[Turn] = []
    for turn in turns:
        paragraphs: list[Paragraph] = []
        for paragraph in turn.paragraphs:
            text = paragraph.text.strip()
            if not text:
                continue
            current = paragraphs[-1] if paragraphs else None
            fits = current is not None and (
                len(current.text.split()) + len(text.split()) <= target_words
                and len(current.text) + 1 + len(text) <= target_chars
            )
            can_merge = (
                current is not None
                and current.pause is None
                and paragraph.timestamp is None
                and paragraph.pause is None
                and paragraph.overlap == current.overlap
                and fits
            )
            if can_merge and current is not None:
                current.text = f"{current.text.rstrip()} {text}"
            else:
                paragraphs.append(
                    Paragraph(
                        text=text,
                        timestamp=paragraph.timestamp,
                        pause=paragraph.pause,
                        overlap=paragraph.overlap,
                    )
                )
        reflowed.append(
            Turn(
                speaker=turn.speaker,
                start=turn.start,
                timestamp=turn.timestamp,
                paragraphs=paragraphs,
            )
        )
    return reflowed


def render_html(
    *,
    title: str,
    segments: list[MergedSegment],
    pauses: list[Pause],
    settings: JobSettings,
    transcription: TranscriptionResult,
    diar: DiarResult | None,
    resolved: dict[str, Any],
    version: str,
) -> str:
    env = Environment(
        loader=FileSystemLoader(str(TEMPLATE_DIR)),
        autoescape=select_autoescape(["html", "j2"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    template = env.get_template("transcript.html.j2")
    turns = _reflow_html_turns(build_turns(segments, pauses, settings))
    speakers = {turn.speaker for turn in turns if turn.speaker is not None}
    return template.render(
        title=title,
        turns=turns,
        show_speakers=len(speakers) > 1,
        transcription=transcription,
        diar=diar,
        settings=settings,
        resolved=resolved,
        version=version,
        hhmmss=hhmmss,
        generated_on=__import__("datetime").datetime.now().astimezone().isoformat(timespec="seconds"),
    )


def _words_for_cues(segments: list[MergedSegment]) -> list[tuple[float, float, str, int | None]]:
    words: list[tuple[float, float, str, int | None]] = []
    for segment in segments:
        if segment.words:
            for word in segment.words:
                if word.start is not None and word.end is not None and word.text.strip():
                    words.append(
                        (word.start, word.end, word.text.strip(), word.speaker if word.speaker is not None else segment.speaker)
                    )
        elif segment.text.strip():
            words.append((segment.start, segment.end, segment.text.strip(), segment.speaker))
    words.sort(key=lambda word: (word[0], word[1]))
    return words


def _cue_text(text: str) -> str:
    # Captions are line-oriented: collapse whitespace and never emit a stray
    # cue terminator inside the payload.
    return " ".join(text.split()).replace("-->", "->")


def _speaker_labels(segments: list[MergedSegment], diar: DiarResult | None) -> bool:
    if diar is None:
        return False
    speakers: set[int] = set()
    for segment in segments:
        if segment.speaker is not None:
            speakers.add(segment.speaker)
        for word in segment.words or []:
            if word.speaker is not None:
                speakers.add(word.speaker)
    return len(speakers) > 1


def _group_cues(
    segments: list[MergedSegment], *, speaker_labels: bool
) -> list[tuple[float, float, str]]:
    if not any(segment.words for segment in segments):
        plain_cues: list[tuple[float, float, str]] = []
        for segment in segments:
            text = _cue_text(segment.text)
            if not text:
                continue
            if speaker_labels and segment.speaker is not None:
                text = f"Speaker {segment.speaker + 1}: {text}"
            plain_cues.append((segment.start, max(segment.start + 0.05, segment.end), text))
        return plain_cues

    words = _words_for_cues(segments)
    if not words:
        return []
    groups: list[list[tuple[float, float, str, int | None]]] = []
    current: list[tuple[float, float, str, int | None]] = []
    for word in words:
        if current:
            previous = current[-1]
            speaker_change = word[3] != previous[3] and speaker_labels
            gap = word[0] - previous[1] > 1.0
            too_many = len(current) >= 12
            too_long = word[1] - current[0][0] >= 5.0
            if speaker_change or gap or too_many or too_long:
                groups.append(current)
                current = []
        current.append(word)
    if current:
        groups.append(current)

    cues: list[tuple[float, float, str]] = []
    for index, group in enumerate(groups):
        start = group[0][0]
        end = group[-1][1] + 0.25
        if index + 1 < len(groups):
            end = min(end, groups[index + 1][0][0] - 0.01)
        end = max(start + 0.05, end)
        text = _cue_text(" ".join(word[2] for word in group))
        if speaker_labels and group[0][3] is not None:
            text = f"Speaker {group[0][3] + 1}: {text}"
        cues.append((start, end, text))
    return cues


def render_srt(segments: list[MergedSegment], settings: JobSettings, diar: DiarResult | None) -> str:
    cues = _group_cues(segments, speaker_labels=_speaker_labels(segments, diar))
    lines: list[str] = []
    for index, (start, end, text) in enumerate(cues, start=1):
        lines.extend([str(index), f"{srt_ts(start)} --> {srt_ts(end)}", text, ""])
    return "\n".join(lines)


def render_vtt(segments: list[MergedSegment], settings: JobSettings, diar: DiarResult | None) -> str:
    cues = _group_cues(segments, speaker_labels=_speaker_labels(segments, diar))
    lines: list[str] = ["WEBVTT", "NOTE WrittenSound", ""]
    for start, end, text in cues:
        lines.extend([f"{vtt_ts(start)} --> {vtt_ts(end)}", text, ""])
    return "\n".join(lines)


def render_timed_txt(
    segments: list[MergedSegment], settings: JobSettings, diar: DiarResult | None
) -> str:
    cues = _group_cues(segments, speaker_labels=_speaker_labels(segments, diar))
    lines = [f"[{vtt_ts(start)} --> {vtt_ts(end)}] {text}" for start, end, text in cues]
    return "\n".join(lines) + ("\n" if lines else "")


def render_json(
    *,
    segments: list[MergedSegment],
    pauses: list[Pause],
    settings: JobSettings,
    transcription: TranscriptionResult,
    diar: DiarResult | None,
    resolved: dict[str, Any],
    version: str,
) -> str:
    ir = transcription.model_dump(mode="json")
    # MergedSegment carries the diarization fields; serialise the actual IR
    # objects rather than slicing them back through Segment.
    ir["segments"] = [segment.model_dump(mode="json") for segment in segments]
    payload = {
        "ir": ir,
        "diar": diar.model_dump(mode="json") if diar else None,
        "pauses": [pause.model_dump(mode="json") for pause in pauses],
        "settings": settings.model_dump(mode="json"),
        "engine_meta": resolved,
        "written_sound": {"app": "WrittenSound", "version": version},
    }
    return json.dumps(payload, indent=2, ensure_ascii=False) + "\n"


def render_all(
    *,
    out_dir: Path,
    title: str,
    segments: list[MergedSegment],
    pauses: list[Pause],
    settings: JobSettings,
    transcription: TranscriptionResult,
    diar: DiarResult | None,
    resolved: dict[str, Any],
    version: str,
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    base = _safe_base(settings, transcription)
    written: list[Path] = []
    if settings.output.txt:
        path = out_dir / f"{base}.txt"
        path.write_text(render_txt(segments, pauses, settings), encoding="utf-8", newline="\n")
        written.append(path)
    if settings.output.html:
        path = out_dir / f"{base}.html"
        path.write_text(
            render_html(
                title=title,
                segments=segments,
                pauses=pauses,
                settings=settings,
                transcription=transcription,
                diar=diar,
                resolved=resolved,
                version=version,
            ),
            encoding="utf-8",
            newline="\n",
        )
        written.append(path)
    if settings.output.srt:
        path = out_dir / f"{base}.srt"
        path.write_text(render_srt(segments, settings, diar), encoding="utf-8", newline="\n")
        written.append(path)
    if settings.output.vtt:
        path = out_dir / f"{base}.vtt"
        path.write_text(render_vtt(segments, settings, diar), encoding="utf-8", newline="\n")
        written.append(path)
    if settings.output.timed_txt:
        path = out_dir / f"{base}.timed.txt"
        path.write_text(render_timed_txt(segments, settings, diar), encoding="utf-8", newline="\n")
        written.append(path)
    if settings.output.json:
        path = out_dir / "ir.json"
        path.write_text(
            render_json(
                segments=segments,
                pauses=pauses,
                settings=settings,
                transcription=transcription,
                diar=diar,
                resolved=resolved,
                version=version,
            ),
            encoding="utf-8",
            newline="\n",
        )
        written.append(path)
    return written
