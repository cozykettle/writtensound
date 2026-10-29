import json

from backend.engines.nemo.adapter import _extract_json, _segments_from_payload, _times


def test_extract_json_ignores_engine_warning_and_word_lines() -> None:
    payload = {
        "file": "/data/jobs/example/media/prep.wav",
        "text": "Hello world.",
        "confidence": 0.91,
        "duration": 2.5,
        "languages": ["en-US"],
        "words": [
            {"word": "Hello", "start": 0.0, "end": 1.0, "confidence": 0.9},
            {"word": "world.", "start": 1.0, "end": 2.5, "confidence": 0.92},
        ],
    }
    log = (
        "Failed to create /home/writtensound for shader cache "
        "(Permission denied)---disabling.\n" + json.dumps(payload, indent=2)
    )

    parsed = _extract_json(log)

    assert parsed == payload
    assert parsed is not None
    segments = _segments_from_payload(parsed)
    assert len(segments) == 1
    assert segments[0].text == "Hello world."
    assert segments[0].start == 0.0
    assert segments[0].end == 2.5
    assert segments[0].words is not None
    assert [word.text for word in segments[0].words] == ["Hello", "world."]
    assert segments[0].words[0].probability == 0.9


def test_top_level_words_are_not_collapsed_into_one_text_segment() -> None:
    payload = {
        "file": "/data/jobs/example/media/prep.wav",
        "text": "One two three.",
        "duration": 31.0,
        "words": [
            {"word": "One", "start": 0.0, "end": 1.0},
            {"word": "two", "start": 11.0, "end": 12.0},
            {"word": "three.", "start": 22.0, "end": 23.0},
        ],
    }

    segments = _segments_from_payload(payload)

    assert len(segments) == 1
    assert segments[0].start == 0.0
    assert segments[0].end == 23.0
    assert segments[0].words is not None
    assert [word.start for word in segments[0].words] == [0.0, 11.0, 22.0]


def test_late_word_times_stay_in_seconds_past_sixteen_minutes() -> None:
    payload = {
        "file": "/data/jobs/example/media/prep.wav",
        "text": "Early late.",
        "duration": 1602.0,
        "words": [
            {"word": "Early", "start": 2.0, "end": 2.5},
            {"word": "late.", "start": 1600.0, "end": 1601.5},
        ],
    }

    segments = _segments_from_payload(payload)

    assert segments[0].words is not None
    assert [word.start for word in segments[0].words] == [2.0, 1600.0]
    assert [word.end for word in segments[0].words] == [2.5, 1601.5]
    assert segments[0].end == 1601.5


def test_millisecond_payloads_are_still_normalized() -> None:
    item = {"start": 120_000, "end": 121_500}

    assert _times(item) == (120.0, 121.5)
