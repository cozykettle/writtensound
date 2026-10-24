"""Normalized internal IR shared by every engine and renderer (docs/04 §2)."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Word(BaseModel):
    start: float | None = None
    end: float | None = None
    text: str
    speaker: int | None = None
    probability: float | None = None


class Segment(BaseModel):
    start: float
    end: float
    text: str
    words: list[Word] | None = None
    language: str | None = None


class TranscriptionResult(BaseModel):
    engine: str
    model_id: str
    device: str
    compute: str | None = None
    language: str | None = None
    language_prob: float | None = None
    duration_s: float
    media_id: str
    segments: list[Segment] = Field(default_factory=list)


class DiarTurn(BaseModel):
    start: float
    end: float
    speaker: int
    overlap: bool = False


class DiarResult(BaseModel):
    engine: str
    model_id: str
    device: str
    turns: list[DiarTurn] = Field(default_factory=list)


class MergedSegment(Segment):
    speaker: int | None = None
    overlap: bool = False


class Pause(BaseModel):
    start: float
    seconds: float
