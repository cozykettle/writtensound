from __future__ import annotations

import io
from pathlib import Path

import pytest

from backend.engines.qwen3.adapter import _ensure_processor_chat_template
from backend.jobs import ApiError


class DummyTokenizer:
    chat_template: str | None = None


class DummyProcessor:
    def __init__(self, *, chat_template: str | None = None) -> None:
        self.chat_template = chat_template
        self.tokenizer = DummyTokenizer()


def test_missing_template_is_restored_from_snapshot(tmp_path: Path) -> None:
    (tmp_path / "chat_template.jinja").write_text("template from disk", encoding="utf-8")
    processor = DummyProcessor()

    _ensure_processor_chat_template(
        processor,
        model_dir=tmp_path,
        repo=None,
        token=None,
        log=io.StringIO(),
    )

    assert processor.chat_template == "template from disk"
    assert processor.tokenizer.chat_template == "template from disk"


def test_tokenizer_template_is_reused_when_processor_lacks_one() -> None:
    processor = DummyProcessor()
    processor.tokenizer.chat_template = "template from tokenizer"

    _ensure_processor_chat_template(
        processor,
        model_dir=Path("unused"),
        repo=None,
        token=None,
        log=io.StringIO(),
    )

    assert processor.chat_template == "template from tokenizer"


def test_missing_template_is_fetched_from_repo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import huggingface_hub

    calls: dict[str, str | None] = {}

    def fake_download(**kwargs: str | None) -> None:
        calls.update(kwargs)
        (tmp_path / "chat_template.jinja").write_text("fetched template", encoding="utf-8")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    processor = DummyProcessor()

    _ensure_processor_chat_template(
        processor,
        model_dir=tmp_path,
        repo="Qwen/Qwen3-ASR-1.7B-hf",
        token="hf_test",
        log=io.StringIO(),
    )

    assert calls["repo_id"] == "Qwen/Qwen3-ASR-1.7B-hf"
    assert calls["filename"] == "chat_template.jinja"
    assert calls["token"] == "hf_test"
    assert processor.chat_template == "fetched template"


def test_missing_template_without_repo_fails_loudly(tmp_path: Path) -> None:
    processor = DummyProcessor()

    with pytest.raises(ApiError, match="chat_template.jinja"):
        _ensure_processor_chat_template(
            processor,
            model_dir=tmp_path,
            repo=None,
            token=None,
            log=io.StringIO(),
        )
