from __future__ import annotations

import time
from urllib.parse import quote

from fastapi.testclient import TestClient

from .helpers import demo_settings, queue_one, upload_wav


def wait_for_terminal(client: TestClient, job_id: str, timeout: float = 20.0) -> dict:
    deadline = time.monotonic() + timeout
    seen_snapshot = None
    while time.monotonic() < deadline:
        response = client.get(f"/api/jobs/{job_id}")
        assert response.status_code == 200
        data = response.json()
        seen_snapshot = data
        if data["status"] in {"succeeded", "failed", "cancelled", "interrupted"}:
            return data
        time.sleep(0.1)
    raise AssertionError(f"job did not finish: {seen_snapshot}")


def test_upload_validation_and_relative_path(client: TestClient, wav_file: bytes) -> None:
    upload_id = upload_wav(client, wav_file, name="field recording.wav", relative=quote("folder/ünïcode.wav"))
    assert upload_id

    bad = client.post(
        "/api/uploads",
        files={"file": ("notes.txt", b"hello", "text/plain")},
    )
    assert bad.status_code == 415
    assert bad.json()["code"] == "unsupported_media"

    empty = client.post(
        "/api/uploads",
        files={"file": ("empty.wav", b"", "audio/wav")},
    )
    assert empty.status_code == 415
    assert empty.json()["code"] == "unsupported_media"


def test_queue_response_carries_full_rows(client: TestClient, wav_file: bytes) -> None:
    upload_id = upload_wav(client, wav_file, name="quick-failure.wav")
    response = client.post(
        "/api/jobs",
        json={"items": [{"upload_id": upload_id, "settings": demo_settings()}]},
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert len(payload["job_ids"]) == len(payload["jobs"]) == 1
    job = payload["jobs"][0]
    assert job["id"] == payload["job_ids"][0]
    assert job["title"] == "quick-failure.wav"
    assert job["seq"] > 0
    assert job["status"] in {"pending", "running", "succeeded", "failed"}
    assert job["settings"]["asr_model_id"] == "demo-script"


def test_demo_job_end_to_end_outputs_and_logs(client: TestClient, wav_file: bytes) -> None:
    job_id = queue_one(client, wav_file)
    data = wait_for_terminal(client, job_id)
    assert data["status"] == "succeeded", data
    assert data["resolved"]["asr_engine"] == "demo"
    assert data["resolved"]["device"] == "cpu"
    assert data["resolved"]["compute"] == "synthetic"
    names = {item["name"] for item in data["outputs_manifest"]}
    assert {"tour.html", "tour.srt", "tour.txt", "tour.vtt", "ir.json"} <= names

    for item in data["outputs_manifest"]:
        response = client.get(f"/api/jobs/{job_id}/outputs/{item['name']}")
        assert response.status_code == 200, item
        assert response.content
        assert response.headers["content-type"]

    assert data["log_paths"] == ["logs/pipeline.log", "logs/engine-asr.log"]
    log = client.get(f"/api/jobs/{job_id}/logs/pipeline.log")
    assert log.status_code == 200
    assert "starting tour.wav" in log.text
    assert client.get(f"/api/jobs/{job_id}/logs/engine-asr.log").status_code in {200, 404}
    assert client.get(f"/api/jobs/{job_id}/logs/other.log").status_code == 404

    list_response = client.get("/api/jobs")
    listed = list_response.json()["jobs"]
    assert [item["id"] for item in listed] == [job_id]


def test_timed_text_output_is_manifested_and_downloadable(client: TestClient, wav_file: bytes) -> None:
    timed = demo_settings()
    timed["output"]["timed_txt"] = True
    job_id = queue_one(client, wav_file, settings=timed)
    data = wait_for_terminal(client, job_id)
    assert data["status"] == "succeeded", data

    names = {item["name"] for item in data["outputs_manifest"]}
    assert "tour.timed.txt" in names
    response = client.get(f"/api/jobs/{job_id}/outputs/tour.timed.txt")
    assert response.status_code == 200
    assert response.text.startswith("[")
    assert " --> " in response.text


def test_pending_job_edit_cancel_and_retry(client: TestClient, wav_file: bytes, app) -> None:
    state = app.state.app_state
    state.runner.stop()

    first = queue_one(client, wav_file, name="first.wav")
    second = queue_one(client, wav_file, name="second.wav", settings=demo_settings(out_basename="second"))
    assert client.get(f"/api/jobs/{first}").json()["status"] == "pending"
    assert client.get(f"/api/jobs/{second}").json()["status"] == "pending"

    first_data = client.get(f"/api/jobs/{first}").json()
    second_data = client.get(f"/api/jobs/{second}").json()
    assert second_data["seq"] > first_data["seq"]
    listed = client.get("/api/jobs").json()["jobs"]
    assert [item["id"] for item in listed] == [second, first]
    queue_snapshot = state.runner.queue_snapshot()
    assert [item["id"] for item in queue_snapshot] == [second, first]
    assert queue_snapshot[0]["title"] == "second.wav"

    edited = client.patch(
        f"/api/jobs/{second}",
        json={"out_basename": "edited-second", "interval_s": 10},
    )
    assert edited.status_code == 200, edited.text
    assert edited.json()["settings"]["out_basename"] == "edited-second"

    cancelled = client.post(f"/api/jobs/{second}/cancel")
    assert cancelled.status_code == 200
    assert cancelled.json()["status"] == "cancelled"

    retried = client.post(f"/api/jobs/{second}/retry")
    assert retried.status_code == 200
    assert retried.json()["status"] == "pending"

    state.runner.start()
    finished_first = wait_for_terminal(client, first)
    finished_second = wait_for_terminal(client, second)
    assert finished_first["status"] == "succeeded"
    assert finished_second["status"] == "succeeded"
    assert "edited-second.html" in {item["name"] for item in finished_second["outputs_manifest"]}


def test_delete_and_unknown_job(client: TestClient, wav_file: bytes) -> None:
    job_id = queue_one(client, wav_file)
    wait_for_terminal(client, job_id)
    assert client.delete(f"/api/jobs/{job_id}").status_code == 204
    assert client.get(f"/api/jobs/{job_id}").status_code == 404
    assert client.post(f"/api/jobs/{job_id}/cancel").status_code == 404
    assert client.get("/api/jobs/does-not-exist").json()["code"] == "not_found"
    assert client.get(f"/api/jobs/{job_id}/outputs/nope.txt").status_code == 404


def test_legacy_upload_is_a_plain_form_post(client: TestClient, wav_file: bytes) -> None:
    response = client.post(
        "/api/uploads/legacy",
        files={"file": ("plain.wav", wav_file, "audio/wav")},
        follow_redirects=False,
    )
    assert response.status_code == 303
    assert response.headers["location"].startswith("/?queued=")
