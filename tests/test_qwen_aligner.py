from __future__ import annotations

import contextlib
import struct
import sys
import types
import wave
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from backend.jobs import ApiError
from backend.pipeline.align import normalize_language, run_aligner


class DummyTokenizer:
    chat_template: str | None = None


class FakeInputs(dict[str, Any]):
    def to(self, device: str) -> FakeInputs:
        self.device = device
        return self


class FakeProcessor:
    instance: FakeProcessor

    def __init__(self) -> None:
        self.chat_template: str | None = "template"
        self.tokenizer = DummyTokenizer()
        self.calls: list[tuple[int, str, str]] = []

    @classmethod
    def from_pretrained(cls, path: str) -> FakeProcessor:
        return cls.instance

    def prepare_forced_aligner_inputs(
        self, *, audio: Any, transcript: str, language: str
    ) -> tuple[FakeInputs, list[list[str]]]:
        self.calls.append((len(audio), transcript, language))
        return FakeInputs(input_ids="ids", attention_mask="mask"), [transcript.split()]

    def decode_forced_alignment(
        self,
        *,
        logits: Any,
        input_ids: Any,
        word_lists: list[list[str]],
        timestamp_token_id: int,
    ) -> list[list[dict[str, Any]]]:
        return [
            [
                {"text": "hello", "start_time": 0.25, "end_time": 0.5},
                {"text": "world", "start_time": 0.5, "end_time": 0.9},
            ]
        ]


class FakeModel:
    instance: FakeModel

    def __init__(self) -> None:
        self.config = types.SimpleNamespace(timestamp_token_id=99)
        self.device = ""
        self.dtype: Any = "float32"
        self.evaluated = False
        self.seen: dict[str, Any] | None = None

    @classmethod
    def from_pretrained(cls, path: str, dtype: Any = None) -> FakeModel:
        cls.instance.dtype = dtype
        return cls.instance

    def to(self, device: str) -> FakeModel:
        self.device = device
        return self

    def eval(self) -> FakeModel:
        self.evaluated = True
        return self

    def __call__(self, **inputs: Any) -> types.SimpleNamespace:
        self.seen = inputs
        return types.SimpleNamespace(logits="logits")


def install_fakes(monkeypatch: pytest.MonkeyPatch, processor: FakeProcessor, model: FakeModel) -> None:
    FakeProcessor.instance = processor
    FakeModel.instance = model
    transformers = types.ModuleType("transformers")
    transformers.AutoProcessor = FakeProcessor  # type: ignore[attr-defined]
    transformers.AutoModelForTokenClassification = FakeModel  # type: ignore[attr-defined]
    torch = types.ModuleType("torch")
    torch.inference_mode = lambda: contextlib.nullcontext()  # type: ignore[attr-defined]
    torch.float32 = "float32"  # type: ignore[attr-defined]
    torch.bfloat16 = "bfloat16"  # type: ignore[attr-defined]
    torch.float16 = "float16"  # type: ignore[attr-defined]
    torch.cuda = types.SimpleNamespace(is_bf16_supported=lambda: False)  # type: ignore[attr-defined]
    torch.is_tensor = lambda value: False  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setitem(sys.modules, "torch", torch)


def write_wav(path: Path, seconds: float = 2.0) -> None:
    frames = int(16_000 * seconds)
    payload = struct.pack("<h", 0) * frames
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(16_000)
        writer.writeframes(payload)


def test_normalize_language_accepts_codes_names_and_auto() -> None:
    assert normalize_language("en") == "en"
    assert normalize_language("EN") == "en"
    assert normalize_language("English") == "en"
    assert normalize_language("yue") == "yue"
    assert normalize_language("auto") is None
    assert normalize_language(None) is None
    assert normalize_language("uk") is None


def test_aligner_repairs_template_from_old_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "chat_template.jinja").write_text("template from disk", encoding="utf-8")
    processor = FakeProcessor()
    processor.chat_template = None
    install_fakes(monkeypatch, processor, FakeModel())
    messages: list[str] = []

    result = run_aligner(
        tmp_path / "audio.wav",
        [],
        model_dir=tmp_path,
        device="cpu",
        language="en",
        log=messages.append,
    )

    assert result == []
    assert messages == ["loaded chat template from chat_template.jinja"]


def test_aligner_missing_template_fails_loudly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    processor = FakeProcessor()
    processor.chat_template = None
    install_fakes(monkeypatch, processor, FakeModel())

    with pytest.raises(ApiError, match="chat_template.jinja"):
        run_aligner(
            tmp_path / "audio.wav",
            [],
            model_dir=tmp_path,
            device="cpu",
            language="en",
        )


def test_aligner_uses_numpy_windows_and_offsets_word_times(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from backend.pipeline.types import Segment

    audio_path = tmp_path / "audio.wav"
    write_wav(audio_path)
    processor = FakeProcessor()
    model = FakeModel()
    install_fakes(monkeypatch, processor, model)

    result = run_aligner(
        audio_path,
        [Segment(start=0.5, end=1.5, text="hello world")],
        model_dir=tmp_path,
        device="cpu",
        language="English",
        log=lambda message: None,
    )

    assert len(processor.calls) == 1
    samples, transcript, language = processor.calls[0]
    assert samples == 16_000
    assert transcript == "hello world"
    assert language == "en"
    assert model.device == "cpu"
    assert model.evaluated
    assert model.seen is not None
    words = result[0].words or []
    assert [(word.text, word.start, word.end) for word in words] == [
        ("hello", 0.75, 1.0),
        ("world", 1.0, 1.4),
    ]


def test_aligner_splits_long_segments_and_logs_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from backend.pipeline import align
    from backend.pipeline.types import Segment

    processor = FakeProcessor()
    model = FakeModel()
    install_fakes(monkeypatch, processor, model)
    monkeypatch.setattr(align, "read_window", lambda path, start, end: np.zeros(1600, dtype=np.float32))
    messages: list[str] = []

    result = run_aligner(
        tmp_path / "audio.wav",
        [Segment(start=0.0, end=700.0, text="one two three four five six")],
        model_dir=tmp_path,
        device="cpu",
        language="en",
        log=messages.append,
        window_s=300.0,
    )

    assert len(processor.calls) == 3
    assert all(samples == 1600 for samples, _, _ in processor.calls)
    assert any("splitting it into 3 windows" in message for message in messages)
    assert len(result[0].words or []) == 6


def test_aligner_never_sends_a_window_longer_than_the_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from backend.pipeline import align
    from backend.pipeline.types import Segment

    processor = FakeProcessor()
    model = FakeModel()
    install_fakes(monkeypatch, processor, model)
    spans: list[tuple[float, float]] = []

    def fake_read_window(path: Path, start: float, end: float) -> np.ndarray:
        spans.append((start, end))
        return np.zeros(1600, dtype=np.float32)

    monkeypatch.setattr(align, "read_window", fake_read_window)

    result = run_aligner(
        tmp_path / "audio.wav",
        [Segment(start=0.0, end=700.0, text="hello")],
        model_dir=tmp_path,
        device="cpu",
        language="en",
        log=lambda message: None,
        window_s=300.0,
    )

    assert spans
    assert len(spans) == 3
    assert all(end - start <= 300.0 for start, end in spans)
    assert len(processor.calls) == len(spans)
    assert len(result[0].words or []) == 6


def test_aligner_skips_unsupported_language_without_failing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from backend.pipeline.types import Segment

    messages: list[str] = []
    segment = Segment(start=0.0, end=1.0, text="hello")

    result = run_aligner(
        tmp_path / "audio.wav",
        [segment],
        model_dir=tmp_path,
        device="cpu",
        language="uk",
        log=messages.append,
    )

    assert result == [segment]
    assert result[0].words is None
    assert any("forced aligner skipped" in message and "'uk'" in message for message in messages)
