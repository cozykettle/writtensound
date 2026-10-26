from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from backend.jobs import ApiError
from backend.pipeline.align import run_aligner


class DummyTokenizer:
    chat_template: str | None = None


class DummyProcessor:
    def __init__(self) -> None:
        self.chat_template: str | None = None
        self.tokenizer = DummyTokenizer()

    @classmethod
    def from_pretrained(cls, path: str) -> DummyProcessor:
        return cls()


class DummyModel:
    @classmethod
    def from_pretrained(cls, path: str) -> DummyModel:
        return cls()

    def to(self, device: str) -> DummyModel:
        return self


def install_fake_transformers(monkeypatch: pytest.MonkeyPatch) -> None:
    module = types.ModuleType("transformers")
    module.AutoProcessor = DummyProcessor  # type: ignore[attr-defined]
    module.AutoModelForMultimodalLM = DummyModel  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "transformers", module)


def test_aligner_repairs_template_from_old_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "chat_template.jinja").write_text("template from disk", encoding="utf-8")
    install_fake_transformers(monkeypatch)
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
    install_fake_transformers(monkeypatch)

    with pytest.raises(ApiError, match="chat_template.jinja"):
        run_aligner(
            tmp_path / "audio.wav",
            [],
            model_dir=tmp_path,
            device="cpu",
            language="en",
        )
