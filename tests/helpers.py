"""Small builders shared by API and renderer tests."""

from __future__ import annotations

from typing import Any

from backend.jobs import JobSettings, OutputFormats


def demo_settings(**overrides: Any) -> dict[str, Any]:
    settings = JobSettings(
        asr_model_id="demo-script",
        device="auto",
        language="en",
        timestamps_mode="interval",
        interval_s=30,
        word_timestamps=True,
        mark_pauses=True,
        pause_min_s=2.0,
        output=OutputFormats(html=True, txt=True, srt=True, vtt=True, json=True),
    )
    data = settings.model_dump(mode="json")
    data.update(overrides)
    return data


def upload_wav(client, wav: bytes, name: str = "tour.wav", relative: str | None = None) -> str:
    headers = {"X-Relative-Path": relative} if relative else {}
    response = client.post(
        "/api/uploads",
        files={"file": (name, wav, "audio/wav")},
        headers=headers,
    )
    assert response.status_code == 200, response.text
    return response.json()["upload_id"]


def queue_one(client, wav: bytes, *, name: str = "tour.wav", settings: dict[str, Any] | None = None) -> str:
    upload_id = upload_wav(client, wav, name=name)
    response = client.post(
        "/api/jobs",
        json={"items": [{"upload_id": upload_id, "settings": settings or demo_settings()}]},
    )
    assert response.status_code == 200, response.text
    return response.json()["job_ids"][0]
