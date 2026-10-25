import json

from backend.engines.nemo.adapter import _extract_json, _segments_from_payload


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
