from __future__ import annotations

import struct
import sys
import types
import wave
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from backend.engines.qwen3.adapter import Qwen3AsrAdapter
from backend.jobs import JobSettings


class FakeIds:
    def __init__(self, length: int) -> None:
        self.shape = (1, length)


class FakeRequest:
    def __init__(self, prompt_len: int) -> None:
        self.input_ids = FakeIds(prompt_len)
        self.device = ""

    def keys(self) -> list[str]:
        return ["input_ids"]

    def items(self) -> Any:
        return {"input_ids": self.input_ids}.items()

    def __getitem__(self, key: str) -> Any:
        assert key == "input_ids"
        return self.input_ids

    def to(self, device: str) -> FakeRequest:
        self.device = device
        return self


class FakeGenerated:
    def __init__(self, length: int) -> None:
        self.shape = (1, length)


class FakeOutput:
    def __init__(self, length: int) -> None:
        self.shape = (1, length)
        self.slice: slice | None = None

    def __getitem__(self, key: tuple[slice, slice]) -> FakeGenerated:
        assert key[0] == slice(None)
        self.slice = key[1]
        start = key[1].start or 0
        stop = key[1].stop or self.shape[1]
        return FakeGenerated(stop - start)


class FakeProcessor:
    instance: FakeProcessor

    def __init__(self) -> None:
        self.chat_template: str | None = "template"
        self.tokenizer = types.SimpleNamespace(chat_template=None)
        self.audio: Any = None
        self.language: str | None = "unset"
        self.decoded: FakeGenerated | None = None
        self.detected_language: str | None = "English"

    @classmethod
    def from_pretrained(cls, path: str) -> FakeProcessor:
        return cls.instance

    def apply_transcription_request(self, *, audio: Any, language: str | None) -> FakeRequest:
        self.audio = audio
        self.language = language
        return FakeRequest(5)

    def decode(self, generated: FakeGenerated, return_format: str) -> list[dict[str, Any]]:
        assert return_format == "parsed"
        self.decoded = generated
        return [{"language": self.detected_language, "transcription": "hello"}]


class FakeModel:
    instance: FakeModel

    def __init__(self) -> None:
        self.device = ""
        self.dtype: Any = None
        self.output: FakeOutput | None = None
        self.inputs: dict[str, Any] | None = None

    @classmethod
    def from_pretrained(cls, path: str, dtype: Any = None) -> FakeModel:
        cls.instance.dtype = dtype
        return cls.instance

    def to(self, device: str) -> FakeModel:
        self.device = device
        return self

    def generate(self, **inputs: Any) -> FakeOutput:
        self.inputs = inputs
        prompt_len = inputs["input_ids"].shape[1]
        self.output = FakeOutput(prompt_len + 3)
        return self.output


class FakeContext:
    hf_token = None

    def __init__(self, model_dir: Path) -> None:
        self._model_dir = model_dir

    def model_dir(self, model: Any) -> Path:
        return self._model_dir


def install_fakes(monkeypatch: pytest.MonkeyPatch, processor: FakeProcessor, model: FakeModel) -> None:
    FakeProcessor.instance = processor
    FakeModel.instance = model
    transformers = types.ModuleType("transformers")
    transformers.__version__ = "5.20.2"  # type: ignore[attr-defined]
    transformers.AutoProcessor = FakeProcessor  # type: ignore[attr-defined]
    transformers.AutoModelForMultimodalLM = FakeModel  # type: ignore[attr-defined]
    torch = types.ModuleType("torch")
    torch.__version__ = "2.9.1+cpu"  # type: ignore[attr-defined]
    torch.float32 = "float32"  # type: ignore[attr-defined]
    torch.bfloat16 = "bfloat16"  # type: ignore[attr-defined]
    torch.float16 = "float16"  # type: ignore[attr-defined]
    torch.cuda = types.SimpleNamespace(is_bf16_supported=lambda: False)  # type: ignore[attr-defined]
    torch.is_tensor = lambda value: False  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setitem(sys.modules, "torch", torch)


def write_wav(path: Path, seconds: float = 1.0) -> None:
    frames = int(16_000 * seconds)
    payload = struct.pack("<h", 1000) * frames
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(16_000)
        writer.writeframes(payload)


def transcribe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    language: str = "auto",
    detected_language: str | None = "English",
) -> tuple[Any, FakeProcessor, FakeModel, str]:
    audio_path = tmp_path / "prep.wav"
    write_wav(audio_path)
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    processor = FakeProcessor()
    processor.detected_language = detected_language
    model = FakeModel()
    install_fakes(monkeypatch, processor, model)
    log_path = tmp_path / "engine-asr.log"
    adapter = Qwen3AsrAdapter(FakeContext(model_dir))  # type: ignore[arg-type]
    result = adapter.transcribe(
        audio_path,
        JobSettings(asr_model_id="qwen3-asr-0.6b", language=language),
        types.SimpleNamespace(id="qwen3-asr-0.6b", repo="Qwen/Qwen3-ASR-0.6B-hf"),
        device="cpu",
        compute="cpu",
        media_id="media-1",
        progress=lambda fraction, message: None,
        log_path=log_path,
        cancel=lambda: False,
    )
    return result, processor, model, log_path.read_text(encoding="utf-8")


def test_qwen_asr_passes_numpy_audio_and_slices_generated_tokens(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result, processor, model, log = transcribe(tmp_path, monkeypatch)

    assert isinstance(processor.audio, np.ndarray)
    assert processor.audio.dtype == np.float32
    assert processor.audio.shape == (16_000,)
    assert processor.language is None
    assert model.inputs is not None
    assert model.dtype == "float32"
    assert model.inputs["input_ids"].shape == (1, 5)
    assert model.output is not None
    assert model.output.slice is not None
    assert model.output.slice.start == 5
    assert model.output.slice.stop is None
    assert processor.decoded is not None
    assert processor.decoded.shape == (1, 3)
    assert len(result.segments) == 1
    assert result.segments[0].text == "hello"
    assert result.segments[0].start == 0.0
    assert result.segments[0].end == 1.0
    assert result.language == "English"
    assert "transformers 5.20.2" in log


def test_qwen_asr_reports_forced_language_when_output_has_no_tag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result, processor, _, _ = transcribe(
        tmp_path, monkeypatch, language="ja", detected_language=None
    )

    assert processor.language == "ja"
    assert result.language == "ja"
    assert result.segments[0].language == "ja"
