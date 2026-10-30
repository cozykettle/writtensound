from __future__ import annotations

from backend.engines.fw.args import FASTER_WHISPER_VERSION, transcribe_kwargs


def test_faster_whisper_121_uses_the_singular_temperature() -> None:
    kwargs = transcribe_kwargs(
        language="en",
        word_timestamps=True,
        verbatim=False,
        accept_progress_hook=False,
    )

    assert FASTER_WHISPER_VERSION == "1.2.1"
    assert kwargs["temperature"] == 0.0
    assert "temperatures" not in kwargs
    assert "progress_hook" not in kwargs


def test_progress_hook_is_only_added_when_the_runtime_accepts_it() -> None:
    kwargs = transcribe_kwargs(
        language="auto",
        word_timestamps=False,
        verbatim=True,
        accept_progress_hook=True,
    )

    assert kwargs["progress_hook"] is True
    assert kwargs["language"] is None
    assert kwargs["initial_prompt"]
