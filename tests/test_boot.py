from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from backend.app import create_app


def test_healthz_and_pages(client: TestClient) -> None:
    assert client.get("/healthz").json() == {"ok": True}

    queue = client.get("/")
    assert queue.status_code == 200
    assert "Add jobs" in queue.text
    assert 'id="job-settings-form"' in queue.text
    assert 'class="engine-strip"' in queue.text

    engines = client.get("/engines")
    assert engines.status_code == 200
    assert "<h1>Engines</h1>" in engines.text
    assert 'id="engine-demo"' in engines.text
    assert 'data-action="use-model"' in engines.text

    settings = client.get("/settings")
    assert settings.status_code == 200
    assert "Retention" in settings.text
    assert 'id="job-settings-form-defaults"' in settings.text


def test_pages_show_configured_data_dir(client: TestClient) -> None:
    root = str(client.app.state.app_state.paths.root)
    assert root in client.get("/engines").text
    assert root in client.get("/settings").text


def test_static_assets_and_favicon(client: TestClient) -> None:
    css = client.get("/static/app.css")
    assert css.status_code == 200
    assert "text/css" in css.headers["content-type"]
    assert "--paper" in css.text

    js = client.get("/static/app.js")
    assert js.status_code == 200
    assert "window.WS.app" in js.text

    assert client.get("/favicon.ico").status_code == 204


def test_openapi_contains_normative_routes(client: TestClient) -> None:
    paths = client.get("/openapi.json").json()["paths"]
    for path in (
        "/api/status",
        "/api/status/refresh",
        "/api/settings",
        "/api/uploads",
        "/api/jobs",
        "/api/jobs/{job_id}",
        "/api/jobs/{job_id}/cancel",
        "/api/jobs/{job_id}/retry",
        "/api/events",
        "/api/catalog",
        "/api/models/{engine_key}/{model_id}/download",
        "/api/maintenance/cleanup",
    ):
        assert path in paths


def test_basic_auth_gate(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("WS_DATA_DIR", str(tmp_path / "secured"))
    monkeypatch.setenv("WS_DISABLE_HW_PROBE", "1")
    monkeypatch.setenv("APP_PASSWORD", "s3cret")
    app = create_app()
    with TestClient(app) as client:
        assert client.get("/healthz").status_code == 200
        denied = client.get("/")
        assert denied.status_code == 401
        assert denied.headers["www-authenticate"].startswith("Basic")
        allowed = client.get("/", auth=("anyone", "s3cret"))
        assert allowed.status_code == 200
