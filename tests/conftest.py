"""Shared fixtures: disposable data directories, a booted app and tiny WAVs."""

from __future__ import annotations

import io
import wave
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.app import create_app


def wav_bytes(seconds: float = 4.0, rate: int = 16_000) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(b"\x00\x00" * int(rate * seconds))
    return buffer.getvalue()


@pytest.fixture
def app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("WS_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("WS_DISABLE_HW_PROBE", "1")
    monkeypatch.delenv("APP_PASSWORD", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    return create_app()


@pytest.fixture
def client(app) -> Iterator[TestClient]:
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def wav_file() -> bytes:
    return wav_bytes()
