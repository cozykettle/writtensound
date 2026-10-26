"""Word/turn merge, speaker labels and pause detection (docs/04 §4)."""

from __future__ import annotations

import re
from collections import Counter

from .types import DiarResult, DiarTurn, MergedSegment, Pause, Segment, TranscriptionResult, Word

# whisper.cpp exposes timestamp and control tokens in its full JSON.  They
# belong to the engine protocol, never to the transcript.  Filtering here also
# repairs checkpoints created before the adapter learned to skip them.
SPECIAL_TOKEN_RE = re.compile(r"\[_[A-Za-z0-9_]+\]")


def _clean_text(text: str) -> str:
    return " ".join(SPECIAL_TOKEN_RE.sub(" ", text).split())


def _speaker_order(turns: list[DiarTurn]) -> dict[int, int]:
    first: dict[int, float] = {}
    for turn in turns:
        first[turn.speaker] = min(first.get(turn.speaker, turn.start), turn.start)
    ordered = sorted(first, key=lambda speaker: (first[speaker], speaker))
    return {speaker: index for index, speaker in enumerate(ordered)}


def _best_turn(start: float, end: float, turns: list[DiarTurn]) -> tuple[DiarTurn | None, bool]:
    midpoint = (start + end) / 2
    candidates = [turn for turn in turns if turn.start <= midpoint <= turn.end]
    if not candidates:
        return None, False
    scored = [
        (max(0.0, min(end, turn.end) - max(start, turn.start)), turn.start, turn)
        for turn in candidates
    ]
    scored.sort(key=lambda item: (-item[0], item[1]))
    return scored[0][2], len(candidates) > 1


def merge(
    transcription: TranscriptionResult,
    diar: DiarResult | None,
    *,
    mark_pauses: bool = False,
    pause_min_s: float = 2.0,
) -> tuple[list[MergedSegment], list[Pause]]:
    turns = sorted(diar.turns, key=lambda turn: (turn.start, turn.end)) if diar else []
    order = _speaker_order(turns) if turns else {}
    merged: list[MergedSegment] = []
    all_words: list[Word] = []

    for segment in transcription.segments:
        text = _clean_text(segment.text)
        words: list[Word] = []
        for word in segment.words or []:
            word_text = _clean_text(word.text)
            if not word_text:
                continue
            if word_text != word.text:
                word.text = word_text
            words.append(word)
        generated = False
        if not words:
            if not text:
                continue
            words = [Word(start=segment.start, end=segment.end, text=text)]
            generated = True
        assigned: list[Word] = []
        for word in words:
            word_start = word.start if word.start is not None else segment.start
            word_end = word.end if word.end is not None else segment.end
            turn, overlaps = _best_turn(word_start, word_end, turns)
            word.speaker = order.get(turn.speaker) if turn else None
            assigned.append(word)
        speakers = [word.speaker for word in assigned if word.speaker is not None]
        speaker = Counter(speakers).most_common(1)[0][0] if speakers else None
        overlap = False
        for word in assigned:
            word_start = word.start if word.start is not None else segment.start
            word_end = word.end if word.end is not None else segment.end
            _, overlaps = _best_turn(word_start, word_end, turns)
            overlap = overlap or overlaps
        new_segment = MergedSegment(
            start=segment.start,
            end=segment.end,
            text=text,
            words=None if generated else assigned,
            language=segment.language,
            speaker=speaker,
            overlap=overlap,
        )
        merged.append(new_segment)
        all_words.extend(assigned)

    pauses: list[Pause] = []
    if mark_pauses:
        timed = [w for w in all_words if w.start is not None and w.end is not None]
        timed.sort(key=lambda w: (w.start or 0.0, w.end or 0.0))
        for previous, following in zip(timed, timed[1:], strict=False):
            gap = (following.start or 0.0) - (previous.end or 0.0)
            if gap >= pause_min_s:
                pauses.append(Pause(start=round(previous.end or 0.0, 2), seconds=round(gap, 2)))
    return merged, pauses


def synthetic_segment(start: float, end: float, text: str) -> Segment:
    return Segment(start=start, end=end, text=text)
