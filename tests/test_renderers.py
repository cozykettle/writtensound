from __future__ import annotations

import json
from pathlib import Path

from backend.jobs import JobSettings
from backend.pipeline.types import DiarResult, DiarTurn, MergedSegment, Pause, TranscriptionResult, Word
from backend.renderers import (
    _reflow_html_turns,
    build_turns,
    render_all,
    render_html,
    render_json,
    render_srt,
    render_timed_txt,
    render_txt,
    render_vtt,
    srt_ts,
    vtt_ts,
)


def settings(**overrides) -> JobSettings:
    data = {
        "asr_model_id": "demo-script",
        "language": "en",
        "timestamps_mode": "interval",
        "interval_s": 10,
        "word_timestamps": True,
        "mark_pauses": True,
        "pause_min_s": 2.0,
        "output": {"html": True, "txt": True, "srt": True, "vtt": True, "json": True},
    }
    data.update(overrides)
    return JobSettings.model_validate(data)


def transcription() -> TranscriptionResult:
    return TranscriptionResult(
        engine="demo",
        model_id="demo-script",
        device="cpu",
        compute="cpu",
        language="en",
        duration_s=30.0,
        media_id="kitchen-table",
        segments=[],
    )


def merged_segments() -> list[MergedSegment]:
    return [
        MergedSegment(
            start=0.0,
            end=2.0,
            text="Hello there.",
            speaker=0,
            words=[Word(start=0.0, end=0.8, text="Hello"), Word(start=0.9, end=2.0, text="there.")],
        ),
        MergedSegment(
            start=12.0,
            end=14.0,
            text="General Kenobi.",
            speaker=1,
            overlap=True,
            words=[
                Word(start=12.0, end=13.0, text="General"),
                Word(start=13.0, end=14.0, text="Kenobi."),
            ],
        ),
    ]


def diar_result() -> DiarResult:
    return DiarResult(
        engine="pyannote",
        model_id="pyannote-community-1",
        device="cpu",
        turns=[
            DiarTurn(start=0.0, end=2.0, speaker=0),
            DiarTurn(start=12.0, end=14.0, speaker=1),
        ],
    )


def test_timestamp_formatters() -> None:
    assert srt_ts(3661.234) == "01:01:01,234"
    assert vtt_ts(3661.234) == "01:01:01.234"
    assert srt_ts(-1) == "00:00:00,000"


def test_turn_grouping_keeps_one_heading_per_speaker_turn() -> None:
    turns = build_turns(merged_segments(), [Pause(start=2.0, seconds=10.0)], settings())
    assert len(turns) == 2
    assert turns[0].speaker == 0
    assert len(turns[0].paragraphs) == 1
    assert turns[1].speaker == 1
    assert turns[0].paragraphs[0].pause is not None

    # Same speaker across an interval stays in one turn but gets a second timestamp.
    same_speaker = [
        MergedSegment(start=0.0, end=2.0, text="first", speaker=0),
        MergedSegment(start=12.0, end=14.0, text="second", speaker=0),
    ]
    turn = build_turns(same_speaker, [], settings())[0]
    assert len(turn.paragraphs) == 2
    assert turn.paragraphs[1].timestamp == "00:00:12"
    assert turn.timestamp == "00:00:00"


def test_one_long_segment_is_split_at_word_speaker_changes() -> None:
    # NeMo returns one segment for the whole file with top-level words, so the
    # segment-level speaker is only the majority label.  TXT/HTML must still
    # split at word-level speaker changes like the caption renderers do.
    segment = MergedSegment(
        start=0.0,
        end=6.0,
        text="Hello there. General Kenobi.",
        speaker=0,
        words=[
            Word(start=0.0, end=1.0, text="Hello", speaker=0),
            Word(start=1.0, end=2.0, text="there.", speaker=0),
            Word(start=3.0, end=4.0, text="General", speaker=1),
            Word(start=4.0, end=6.0, text="Kenobi.", speaker=1),
        ],
    )
    turns = build_turns([segment], [], settings())
    assert [turn.speaker for turn in turns] == [0, 1]
    assert [paragraph.text for paragraph in turns[0].paragraphs] == ["Hello there."]
    assert [paragraph.text for paragraph in turns[1].paragraphs] == ["General Kenobi."]

    text = render_txt([segment], [], settings())
    assert "Speaker 1:\n[00:00:00] Hello there." in text
    assert "Speaker 2:\n[00:00:03] General Kenobi." in text

    html = render_html(
        title="Interview",
        segments=[segment],
        pauses=[],
        settings=settings(),
        transcription=transcription(),
        diar=diar_result(),
        resolved={"asr_model": "demo-script", "device": "cpu", "compute": "cpu"},
        version="0.5.0",
    )
    assert html.count("<h3>Speaker 1") == 1
    assert html.count("<h3>Speaker 2") == 1


def test_txt_has_speaker_blocks_timestamps_and_pause_marker() -> None:
    text = render_txt(merged_segments(), [Pause(start=2.0, seconds=8.0)], settings())
    assert "Speaker 1:" in text
    assert "Speaker 2:" in text
    assert "Hello there." in text
    assert "[00:00:00]" in text
    assert "[00:00:12]" in text
    assert "(. . . . .) (+8 s)" in text
    assert text.endswith("\n")


def test_html_escapes_content_and_marks_overlap() -> None:
    segments = merged_segments()
    segments[1].text = "General <Kenobi>."
    html = render_html(
        title="Kitchen <Table>",
        segments=segments,
        pauses=[],
        settings=settings(),
        transcription=transcription(),
        diar=diar_result(),
        resolved={"asr_model": "demo-script", "device": "cpu", "compute": "cpu"},
        version="0.1.0",
    )
    assert "Kitchen &lt;Table&gt;" in html
    assert "General &lt;Kenobi&gt;." in html
    assert '<span class="overlap">' in html
    assert html.count("<h3>Speaker 1") == 1
    assert html.count("<h3>Speaker 2") == 1
    assert "00:00:12" in html
    assert "pyannote-community-1" in html
    assert "your words stayed home" in html


def test_html_reflows_short_engine_segments_without_losing_text() -> None:
    segments = [
        MergedSegment(start=float(index * 3), end=float(index * 3 + 2), text=f"word{index} alpha beta gamma", speaker=None)
        for index in range(12)
    ]

    html = render_html(
        title="Reflow",
        segments=segments,
        pauses=[],
        settings=settings(timestamps_mode="none"),
        transcription=transcription(),
        diar=None,
        resolved={"asr_model": "demo-script", "device": "cpu"},
        version="0.4.0",
    )

    assert html.count("<p>") == 1
    assert "word0 alpha beta gamma word1 alpha beta gamma" in html
    assert "word11 alpha beta gamma" in html


def test_html_reflow_caps_paragraph_size_and_keeps_pauses_at_the_end() -> None:
    segments = [
        MergedSegment(start=float(index * 2), end=float(index * 2 + 1), text=f"a{index} b{index} c{index}", speaker=None)
        for index in range(40)
    ]
    pauses = [Pause(start=19.0, seconds=3.0)]

    turns = _reflow_html_turns(build_turns(segments, pauses, settings(timestamps_mode="none")))
    paragraphs = turns[0].paragraphs

    assert len(paragraphs) >= 2
    for paragraph in paragraphs:
        assert len(paragraph.text.split()) <= 80
    paused = [paragraph for paragraph in paragraphs if paragraph.pause]
    assert len(paused) == 1
    assert paused[0].pause == "(. . .)"
    assert paused[0].text.endswith("c9")
    assert paragraphs[-1].text.endswith("c39")


def test_html_reflow_keeps_interval_markers_as_paragraph_starts() -> None:
    segments = [
        MergedSegment(start=float(index * 11), end=float(index * 11 + 10), text=f"block{index}", speaker=None)
        for index in range(3)
    ]

    html = render_html(
        title="Markers",
        segments=segments,
        pauses=[],
        settings=settings(interval_s=10),
        transcription=transcription(),
        diar=None,
        resolved={"asr_model": "demo-script", "device": "cpu"},
        version="0.4.0",
    )

    assert html.count("<p>") == 3
    assert '<p><span class="ts">[00:00:00]</span>' in html
    assert '<span class="ts">[00:00:11]</span>' in html
    assert '<span class="ts">[00:00:22]</span>' in html


def test_html_meta_can_wrap_dates_and_shows_duration() -> None:
    data = transcription().model_copy(update={"duration_s": 3725.0})
    html = render_html(
        title="Meta",
        segments=merged_segments(),
        pauses=[],
        settings=settings(),
        transcription=data,
        diar=None,
        resolved={"asr_model": "demo-script", "device": "cpu"},
        version="0.4.0",
    )

    assert "01:02:05" in html
    assert "min-width: 0" in html
    assert "overflow-wrap: anywhere" in html


def test_srt_vtt_cues_and_wordless_segments() -> None:
    srt = render_srt(merged_segments(), settings(), diar_result())
    assert "00:00:00,000 --> 00:00:02,250" in srt
    assert "Speaker 1: Hello there." in srt
    assert "Speaker 2: General Kenobi." in srt

    vtt = render_vtt(merged_segments(), settings(), diar_result())
    assert vtt.startswith("WEBVTT\n")
    assert "00:00:00.000 --> 00:00:02.250" in vtt

    wordless = [
        MergedSegment(start=0.0, end=1.0, text="one", speaker=0),
        MergedSegment(start=1.0, end=2.0, text="two", speaker=1),
    ]
    cues = render_srt(wordless, settings(), diar_result()).strip().split("\n\n")
    assert len(cues) == 2
    assert cues[0].splitlines()[1] == "00:00:00,000 --> 00:00:01,000"
    assert cues[1].splitlines()[1] == "00:00:01,000 --> 00:00:02,000"


def test_single_speaker_captions_do_not_repeat_speaker_one() -> None:
    segments = [
        MergedSegment(
            start=0.0,
            end=2.0,
            text="Hello there.",
            speaker=0,
            words=[Word(start=0.0, end=0.8, text="Hello"), Word(start=0.9, end=2.0, text="there.")],
        )
    ]
    diar = DiarResult(
        engine="pyannote",
        model_id="pyannote-community-1",
        device="cpu",
        turns=[DiarTurn(start=0.0, end=2.0, speaker=0)],
    )

    srt = render_srt(segments, settings(), diar)
    vtt = render_vtt(segments, settings(), diar)
    timed = render_timed_txt(segments, settings(), diar)

    assert "Speaker 1:" not in srt
    assert "Speaker 1:" not in vtt
    assert "Speaker 1:" not in timed
    assert "Hello there." in timed

    html = render_html(
        title="Single speaker",
        segments=segments,
        pauses=[],
        settings=settings(),
        transcription=transcription(),
        diar=diar,
        resolved={"asr_model": "demo-script", "device": "cpu", "compute": "cpu"},
        version="0.2.0",
    )
    assert "<h3>Speaker 1" not in html


def test_timed_text_uses_caption_cues_without_indices_or_header() -> None:
    text = render_timed_txt(merged_segments(), settings(), diar_result())

    assert text == (
        "[00:00:00.000 --> 00:00:02.250] Speaker 1: Hello there.\n"
        "[00:00:12.000 --> 00:00:14.250] Speaker 2: General Kenobi.\n"
    )


def test_interval_markers_split_a_long_word_segment() -> None:
    segment = MergedSegment(
        start=0.0,
        end=24.0,
        text="one two three",
        speaker=0,
        words=[
            Word(start=0.0, end=1.0, text="one"),
            Word(start=11.0, end=12.0, text="two"),
            Word(start=22.0, end=23.0, text="three"),
        ],
    )

    turn = build_turns([segment], [], settings(interval_s=10))[0]

    assert [paragraph.timestamp for paragraph in turn.paragraphs] == [
        "00:00:00",
        "00:00:11",
        "00:00:22",
    ]
    assert turn.timestamp == "00:00:00"


def test_json_retains_merged_speaker_overlap_and_settings() -> None:
    payload = json.loads(
        render_json(
            segments=merged_segments(),
            pauses=[Pause(start=2.0, seconds=8.0)],
            settings=settings(),
            transcription=transcription(),
            diar=diar_result(),
            resolved={"asr_model": "demo-script", "device": "cpu"},
            version="0.1.0",
        )
    )
    assert payload["ir"]["segments"][1]["speaker"] == 1
    assert payload["ir"]["segments"][1]["overlap"] is True
    assert payload["ir"]["media_id"] == "kitchen-table"
    assert payload["diar"]["turns"][0]["speaker"] == 0
    assert payload["pauses"][0]["seconds"] == 8.0
    assert payload["settings"]["asr_model_id"] == "demo-script"
    assert payload["written_sound"]["app"] == "WrittenSound"


def test_render_all_writes_selected_formats_and_sanitizes_basename(tmp_path: Path) -> None:
    written = render_all(
        out_dir=tmp_path,
        title="A talk",
        segments=merged_segments(),
        pauses=[],
        settings=settings(out_basename="talk: one?"),
        transcription=transcription(),
        diar=None,
        resolved={"asr_model": "demo-script"},
        version="0.1.0",
    )
    names = {path.name for path in written}
    assert names == {"talk one.html", "talk one.srt", "talk one.txt", "talk one.vtt", "ir.json"}
    assert all(path.exists() for path in written)


def test_render_all_writes_timed_text_format(tmp_path: Path) -> None:
    timed_settings = settings(
        out_basename="talk",
        output={"html": False, "txt": False, "srt": False, "vtt": False, "timed_txt": True, "json": False},
    )

    written = render_all(
        out_dir=tmp_path,
        title="A talk",
        segments=merged_segments(),
        pauses=[],
        settings=timed_settings,
        transcription=transcription(),
        diar=diar_result(),
        resolved={"asr_model": "demo-script"},
        version="0.2.0",
    )

    assert [path.name for path in written] == ["talk.timed.txt"]
    assert written[0].read_text(encoding="utf-8").startswith("[00:00:00.000 --> ")


def test_no_timestamps_mode_omits_markers() -> None:
    plain = settings(timestamps_mode="none", output={"html": True, "txt": True, "srt": False, "vtt": False, "json": True})
    text = render_txt(merged_segments(), [], plain)
    assert "[00:" not in text
    assert "Speaker 1:" in text
