from __future__ import annotations

from pathlib import Path

import pytest

from backend.engines.qwen3.chat_template import ensure_chat_template
from backend.jobs import ApiError


class DummyTokenizer:
    chat_template: str | None = None


class DummyProcessor:
    def __init__(self, *, chat_template: str | None = None) -> None:
        self.chat_template = chat_template
        self.tokenizer = DummyTokenizer()


def ignore_log(message: str) -> None:
    pass


def test_missing_template_is_restored_from_snapshot(tmp_path: Path) -> None:
    (tmp_path / "chat_template.jinja").write_text("template from disk", encoding="utf-8")
    processor = DummyProcessor()
    messages: list[str] = []

    ensure_chat_template(
        processor,
        model_dir=tmp_path,
        repo=None,
        token=None,
        log=messages.append,
    )

    assert processor.chat_template == "template from disk"
    assert processor.tokenizer.chat_template == "template from disk"
    assert messages == ["loaded chat template from chat_template.jinja"]


def test_tokenizer_template_is_reused_when_processor_lacks_one() -> None:
    processor = DummyProcessor()
    processor.tokenizer.chat_template = "template from tokenizer"

    ensure_chat_template(
        processor,
        model_dir=Path("unused"),
        repo=None,
        token=None,
        log=ignore_log,
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
    messages: list[str] = []

    ensure_chat_template(
        processor,
        model_dir=tmp_path,
        repo="Qwen/Qwen3-ASR-1.7B-hf",
        token="hf_test",
        log=messages.append,
    )

    assert calls["repo_id"] == "Qwen/Qwen3-ASR-1.7B-hf"
    assert calls["filename"] == "chat_template.jinja"
    assert calls["token"] == "hf_test"
    assert processor.chat_template == "fetched template"
    assert messages == [
        "chat_template.jinja missing; fetching it from Qwen/Qwen3-ASR-1.7B-hf",
        "loaded chat template from chat_template.jinja",
    ]


def test_missing_template_without_repo_fails_loudly(tmp_path: Path) -> None:
    processor = DummyProcessor()

    with pytest.raises(ApiError, match="chat_template.jinja"):
        ensure_chat_template(
            processor,
            model_dir=tmp_path,
            repo=None,
            token=None,
            log=ignore_log,
        )
