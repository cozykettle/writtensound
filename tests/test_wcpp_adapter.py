from __future__ import annotations

from pathlib import Path

from backend.engines.wcpp.adapter import _segments_from_payload
from backend.engines.wcpp.args import build_command


def test_special_tokens_are_excluded_and_word_pieces_are_merged() -> None:
    payload = {
        "transcription": [
            {
                "offsets": {"from": 0, "to": 2_500},
                "text": " Don't hide your light. [_TT_100] [_BEG_]",
                "tokens": [
                    {"text": " Don", "offsets": {"from": 0, "to": 300}, "id": 100},
                    {"text": "'t", "offsets": {"from": 300, "to": 500}, "id": 101},
                    {"text": " hide", "offsets": {"from": 500, "to": 900}, "id": 102},
                    {"text": " your", "offsets": {"from": 900, "to": 1_200}, "id": 103},
                    {"text": " light", "offsets": {"from": 1_200, "to": 1_800}, "id": 104},
                    {"text": ".", "offsets": {"from": 1_800, "to": 1_900}, "id": 105},
                    {"text": "[_TT_100]", "offsets": {"from": 1_900, "to": 2_000}, "id": 50_364},
                    {"text": "[_BEG_]", "offsets": {"from": 2_000, "to": 2_100}, "id": 50_363},
                ],
            }
        ]
    }

    segments = _segments_from_payload(payload)

    assert len(segments) == 1
    assert segments[0].text == "Don't hide your light."
    assert segments[0].words is not None
    assert [word.text for word in segments[0].words] == ["Don't", "hide", "your", "light."]
    assert [word.start for word in segments[0].words] == [0.0, 0.5, 0.9, 1.2]
    assert [word.end for word in segments[0].words] == [0.5, 0.9, 1.2, 1.9]


def test_subword_tokens_become_readable_words() -> None:
    payload = {
        "transcription": [
            {
                "offsets": {"from": 0, "to": 4_000},
                "text": "unimaginable friendliness foot-washing",
                "tokens": [
                    {"text": " un"},
                    {"text": "imagin"},
                    {"text": "able"},
                    {"text": " friend"},
                    {"text": "liness"},
                    {"text": " foot"},
                    {"text": "-"},
                    {"text": "was"},
                    {"text": "hing"},
                ],
            }
        ]
    }

    segments = _segments_from_payload(payload)

    assert segments[0].words is not None
    assert [word.text for word in segments[0].words] == [
        "unimaginable",
        "friendliness",
        "foot-washing",
    ]


def test_unspaced_script_tokens_are_not_collapsed_into_one_word() -> None:
    payload = {
        "transcription": [
            {
                "offsets": {"from": 0, "to": 1_000},
                "text": "你好世界",
                "tokens": [
                    {"text": "你", "offsets": {"from": 0, "to": 200}},
                    {"text": "好", "offsets": {"from": 200, "to": 400}},
                    {"text": "世界", "offsets": {"from": 400, "to": 900}},
                ],
            }
        ]
    }

    segments = _segments_from_payload(payload)

    assert segments[0].words is not None
    assert [word.text for word in segments[0].words] == ["你", "好", "世界"]


def test_segment_with_only_special_tokens_is_not_emitted() -> None:
    payload = {
        "transcription": [
            {
                "offsets": {"from": 0, "to": 1_000},
                "text": "[_TT_12]",
                "tokens": [
                    {"text": "[_TT_12]", "id": 50_376},
                    {"text": "[_BEG_]", "id": 50_363},
                ],
            }
        ]
    }

    assert _segments_from_payload(payload) == []


def test_command_keeps_full_json_without_destructive_max_len() -> None:
    cmd = build_command(
        binary="whisper-cli",
        model_path=Path("/models/ggml.bin"),
        audio_path=Path("/tmp/prep.wav"),
        output_prefix=Path("/tmp/out"),
        language="auto",
        device="cpu",
        threads=4,
        verbatim=False,
    )

    assert "--output-json-full" in cmd
    assert "--max-len" not in cmd
