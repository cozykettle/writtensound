"""Repair Qwen chat templates in snapshots downloaded before ``*.jinja``.

Current Transformers stores the Qwen3 chat template in a separate
``chat_template.jinja`` file.  The v0.2 snapshot rules omitted it, so affected
processors load with ``chat_template=None`` and every transcription or
alignment call fails in ``apply_chat_template``.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from ...jobs import ApiError

LogFn = Callable[[str], None]


def ensure_chat_template(
    processor: Any,
    *,
    model_dir: Path,
    repo: str | None,
    token: str | None,
    log: LogFn,
) -> None:
    if getattr(processor, "chat_template", None):
        return
    tokenizer = getattr(processor, "tokenizer", None)
    template = getattr(tokenizer, "chat_template", None)
    if template:
        processor.chat_template = template
        return

    template_path = model_dir / "chat_template.jinja"
    if not template_path.exists():
        if not repo:
            raise ApiError(
                "engine_crash",
                f"Qwen3 processor has no chat template and {template_path.name} is missing.",
            )
        log(f"{template_path.name} missing; fetching it from {repo}")
        try:
            from huggingface_hub import hf_hub_download

            hf_hub_download(
                repo_id=repo,
                filename=template_path.name,
                local_dir=str(model_dir),
                token=token,
            )
        except Exception as exc:
            raise ApiError(
                "engine_crash",
                f"Qwen3 snapshot is missing {template_path.name} and it could not be fetched: {exc}. "
                "Remove and re-download the model from the Engines page.",
            ) from exc
    try:
        template = template_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ApiError(
            "engine_crash",
            f"Qwen3 chat template could not be read from {template_path}: {exc}",
        ) from exc

    processor.chat_template = template
    if tokenizer is not None:
        tokenizer.chat_template = template
    log(f"loaded chat template from {template_path.name}")
