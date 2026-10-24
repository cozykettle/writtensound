from __future__ import annotations

import json

from fastapi.testclient import TestClient

from .helpers import demo_settings, upload_wav


def test_settings_defaults_and_mask(client: TestClient) -> None:
    response = client.get("/api/settings")
    assert response.status_code == 200
    payload = response.json()
    assert payload["defaults"]["asr_model_id"] == "wcpp-large-v3-turbo-q8_0"
    assert payload["hf_token"] is None
    assert payload["keep_completed_jobs_days"] == 14


def test_settings_round_trip_and_token_storage(client: TestClient, app) -> None:
    token = "hf_" + "a" * 32
    defaults = demo_settings(out_basename="my-default")
    response = client.put(
        "/api/settings",
        json={
            "defaults": defaults,
            "hf_token": token,
            "keep_completed_jobs_days": 7,
            "cleanup_uploads": False,
        },
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["defaults"]["asr_model_id"] == "demo-script"
    assert payload["defaults"]["out_basename"] == "my-default"
    assert payload["hf_token"] == "hf_****" + "a" * 3 + "a"
    assert payload["keep_completed_jobs_days"] == 7
    assert payload["cleanup_uploads"] is False

    state = app.state.app_state
    stored = json.loads(state.paths.config_file.read_text(encoding="utf-8"))
    assert stored["hf_token"] == token
    assert stored["defaults"]["asr_model_id"] == "demo-script"

    # The settings page must never leak the raw token into HTML.
    page = client.get("/settings")
    assert page.status_code == 200
    assert token not in page.text
    assert "hf_****" in page.text


def test_invalid_settings_are_rejected(client: TestClient, app) -> None:
    token = client.put("/api/settings", json={"hf_token": "not-a-token"})
    assert token.status_code == 400
    assert token.json()["code"] == "invalid_settings"

    unknown = client.put("/api/settings", json={"engine_enabled": {"nope": False}})
    assert unknown.status_code == 400
    assert unknown.json()["code"] == "invalid_settings"
    assert app.state.app_state.paths.config_file.exists() is False


def test_engine_toggle_and_unknown_engine(client: TestClient) -> None:
    disabled = client.post("/api/engines/whisper.cpp/disable")
    assert disabled.status_code == 200
    assert disabled.json() == {"engine": "whisper.cpp", "enabled": False}
    assert client.get("/api/catalog").json()["engine_enabled"]["whisper.cpp"] is False

    assert client.post("/api/engines/whisper.cpp/enable").json()["enabled"] is True
    assert client.post("/api/engines/not-real/disable").status_code == 404


def test_job_validation_errors_surface_item_details(client: TestClient, wav_file: bytes) -> None:
    upload_id = upload_wav(client, wav_file, name="bad.wav")

    bad_formats = demo_settings()
    bad_formats["output"] = {"html": False, "txt": False, "srt": False, "vtt": False, "json": False}
    response = client.post("/api/jobs", json={"items": [{"upload_id": upload_id, "settings": bad_formats}]})
    assert response.status_code == 400
    assert response.json()["code"] == "invalid_settings"

    unknown_model = demo_settings(asr_model_id="not-a-model")
    response = client.post("/api/jobs", json={"items": [{"upload_id": upload_id, "settings": unknown_model}]})
    assert response.status_code == 400
    detail = response.json()["details"]["items_errors"][0]["error"]
    assert detail["code"] == "invalid_settings"
    assert "Unknown ASR model" in detail["message"]


def test_caption_validation_rejects_missing_timestamp_mode(client: TestClient, wav_file: bytes) -> None:
    upload_id = upload_wav(client, wav_file, name="captions.wav")
    settings = demo_settings(timestamps_mode="none")
    response = client.post("/api/jobs", json={"items": [{"upload_id": upload_id, "settings": settings}]})
    assert response.status_code == 400
    assert "captions" in response.text.lower()
