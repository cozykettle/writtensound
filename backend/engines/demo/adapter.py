"""The built-in demo engine: deterministic, local, and proud of it."""

from __future__ import annotations

import hashlib
import random
from pathlib import Path

from ...catalog import ModelSpec
from ...jobs import JobSettings
from ...pipeline.prep import probe_media
from ...pipeline.types import Segment, TranscriptionResult, Word
from ..base import CancelFn, EngineContext, JobCancelledError, ProgressFn

SCRIPT = (
    "The room is quiet for a moment, and then someone leans toward the microphone. "
    "This is a demonstration transcript, generated entirely on this machine. "
    "No audio left the building, no account was created, and no model weights were harmed. "
    "WrittenSound turns a recording into words, and words into something you can search. "
    "Imagine three voices around a kitchen table, trading stories until the coffee goes cold. "
    "The first speaker remembers a train station in winter. "
    "The second insists it was summer, and produces a photograph as evidence. "
    "The third says the season does not matter, only that everyone came home. "
    "That is the whole trick of a transcript: it keeps the ordinary afternoon from disappearing. "
    "Drop your own recording into the queue whenever you are ready. "
    "The demo will step aside, quietly, like a stagehand. "
)


class DemoAdapter:
    name = "demo"

    def __init__(self, context: EngineContext) -> None:
        self.context = context

    def transcribe(
        self,
        audio_path: Path,
        settings: JobSettings,
        model: ModelSpec,
        *,
        device: str,
        compute: str,
        media_id: str,
        progress: ProgressFn,
        log_path: Path,
        cancel: CancelFn,
    ) -> TranscriptionResult:
        info = probe_media(audio_path)
        duration = max(info.duration_s, 1.0)
        media_id = settings.out_basename or media_id
        seed = int(hashlib.sha256(media_id.encode("utf-8")).hexdigest()[:8], 16)
        rng = random.Random(seed)

        words_per_second = 2.35
        target_words = max(8, int(duration * words_per_second))
        script_words = SCRIPT.split()
        offset = rng.randrange(len(script_words))
        spoken = [script_words[(offset + index) % len(script_words)] for index in range(target_words)]

        raw_segments: list[Segment] = []
        cursor = 0.35
        index = 0
        while index < len(spoken):
            take = min(len(spoken) - index, rng.randint(7, 13))
            chunk = spoken[index : index + take]
            segment_duration = take / words_per_second
            if cursor + segment_duration > duration:
                segment_duration = max(0.5, duration - cursor - 0.1)
            if segment_duration <= 0.2:
                break
            word_duration = segment_duration / take
            words: list[Word] = []
            for word_index, text in enumerate(chunk):
                start = cursor + word_index * word_duration
                end = start + word_duration * 0.86
                words.append(
                    Word(
                        start=round(start, 3),
                        end=round(min(end, duration), 3),
                        text=text,
                        probability=round(0.86 + rng.random() * 0.12, 3),
                    )
                )
            text = " ".join(chunk)
            raw_segments.append(
                Segment(
                    start=round(cursor, 3),
                    end=round(min(cursor + segment_duration, duration), 3),
                    text=text,
                    words=words,
                    language="en",
                )
            )
            cursor += segment_duration + rng.choice((0.18, 0.35, 0.72, 1.4))
            index += take
            if cursor >= duration - 0.2:
                break
            progress(min(0.96, index / max(1, len(spoken))), f"demo {index}/{len(spoken)} words")
            if cancel():
                raise JobCancelledError

        if not raw_segments:
            raw_segments = [
                Segment(
                    start=0.0,
                    end=duration,
                    text="A short, quiet recording. Even silence has a shape.",
                    words=[
                        Word(
                            start=0.0,
                            end=min(duration, 2.0),
                            text="A short, quiet recording. Even silence has a shape.",
                            probability=1.0,
                        )
                    ],
                    language="en",
                )
            ]
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(
            "demo engine\n"
            f"media: {audio_path.name}\n"
            f"duration: {duration:.2f}s\n"
            f"words: {sum(len(s.words or []) for s in raw_segments)}\n"
            "synthetic transcript — no speech recognition was performed\n",
            encoding="utf-8",
        )
        progress(1.0, "demo transcript ready")
        return TranscriptionResult(
            engine="demo",
            model_id=model.id,
            device="cpu",
            compute="synthetic",
            language="en",
            language_prob=1.0,
            duration_s=round(duration, 3),
            media_id=settings.out_basename or media_id,
            segments=raw_segments,
        )


__all__ = ["DemoAdapter", "SCRIPT"]
