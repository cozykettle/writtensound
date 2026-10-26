from __future__ import annotations

from backend.jobs import JobSettings
from backend.pipeline.merge import merge
from backend.pipeline.types import DiarResult, DiarTurn, Segment, TranscriptionResult, Word


def transcription(*segments: Segment) -> TranscriptionResult:
    return TranscriptionResult(
        engine="demo",
        model_id="demo-script",
        device="cpu",
        language="en",
        duration_s=30.0,
        media_id="recording",
        segments=list(segments),
    )


def settings(**overrides) -> JobSettings:
    data = {
        "asr_model_id": "demo-script",
        "language": "en",
    }
    data.update(overrides)
    return JobSettings.model_validate(data)


def test_speaker_order_follows_first_appearance_not_id() -> None:
    diar = DiarResult(
        engine="pyannote",
        model_id="pyannote-community-1",
        device="cpu",
        turns=[
            DiarTurn(start=0.0, end=1.0, speaker=7),
            DiarTurn(start=5.0, end=6.0, speaker=2),
        ],
    )
    result = transcription(
        Segment(start=0.0, end=1.0, text="first", words=[Word(start=0.1, end=0.9, text="first")]),
        Segment(start=5.0, end=6.0, text="second", words=[Word(start=5.1, end=5.9, text="second")]),
    )
    merged, _ = merge(result, diar)
    assert [segment.speaker for segment in merged] == [0, 1]


def test_overlap_and_majority_speaker() -> None:
    diar = DiarResult(
        engine="nemo",
        model_id="nemo-diar-nemotron-3",
        device="cpu",
        turns=[
            DiarTurn(start=0.0, end=2.0, speaker=1),
            DiarTurn(start=1.5, end=4.0, speaker=2, overlap=True),
        ],
    )
    result = transcription(
        Segment(
            start=0.0,
            end=4.0,
            text="overlapping speech",
            words=[
                Word(start=1.8, end=2.0, text="quiet"),
                Word(start=2.1, end=2.3, text="overlapping"),
                Word(start=2.6, end=2.8, text="speech"),
            ],
        )
    )
    merged, _ = merge(result, diar)
    assert merged[0].overlap is True
    # Two words in turn 2 outvote one word in turn 1.
    assert merged[0].speaker == 1
    assert merged[0].words is not None
    assert [word.speaker for word in merged[0].words] == [0, 1, 1]


def test_pause_detection_uses_threshold() -> None:
    result = transcription(
        Segment(
            start=0.0,
            end=8.0,
            text="hello ... world",
            words=[
                Word(start=0.1, end=1.0, text="hello"),
                Word(start=5.0, end=6.0, text="world"),
            ],
        )
    )
    _, pauses = merge(result, None, mark_pauses=True, pause_min_s=2.0)
    assert len(pauses) == 1
    assert pauses[0].start == 1.0
    assert pauses[0].seconds == 4.0

    _, no_pauses = merge(result, None, mark_pauses=True, pause_min_s=5.0)
    assert no_pauses == []
    _, disabled = merge(result, None, mark_pauses=False, pause_min_s=0.5)
    assert disabled == []


def test_special_tokens_are_removed_from_segment_text_and_words() -> None:
    result = transcription(
        Segment(
            start=0.0,
            end=2.0,
            text="Hello [_TT_10] world.",
            words=[
                Word(start=0.0, end=0.5, text="Hello"),
                Word(start=0.5, end=0.8, text="[_TT_10]"),
                Word(start=0.8, end=2.0, text="world."),
            ],
        )
    )

    merged, _ = merge(result, None)

    assert merged[0].text == "Hello world."
    assert merged[0].words is not None
    assert [word.text for word in merged[0].words] == ["Hello", "world."]


def test_wordless_segment_gets_one_synthetic_word_and_speaker() -> None:
    diar = DiarResult(
        engine="pyannote",
        model_id="pyannote-community-1",
        device="cpu",
        turns=[DiarTurn(start=0.0, end=3.0, speaker=4)],
    )
    result = transcription(Segment(start=0.0, end=3.0, text="unworded"))
    merged, _ = merge(result, diar)
    assert merged[0].speaker == 0
    assert merged[0].words is None
    assert merged[0].overlap is False
