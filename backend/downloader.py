"""Hugging Face model downloads with progress, cancellation and disk reconciliation."""

from __future__ import annotations

import os
import shutil
import threading
import time
from pathlib import Path
from typing import Any

from .catalog import Catalog, ModelSpec
from .config import Paths
from .db import ModelStateStore
from .engines.base import MODEL_DIRS
from .events import EventHub
from .jobs import ApiError

POLL_SECONDS = 0.8
EMIT_SECONDS = 1.0


class DownloadManager:
    def __init__(
        self,
        *,
        paths: Paths,
        catalog: Catalog,
        states: ModelStateStore,
        hub: EventHub,
        token_provider: Any,
    ) -> None:
        self.paths = paths
        self.catalog = catalog
        self.states = states
        self.hub = hub
        self.token_provider = token_provider
        self._lock = threading.Lock()
        self._active: dict[tuple[str, str], threading.Event] = {}
        self._slots = threading.Semaphore(2)

    def model_dir(self, model: ModelSpec) -> Path:
        folder = MODEL_DIRS.get(model.engine, model.engine)
        return self.paths.models / folder / model.id

    def state_row(self, model: ModelSpec) -> dict[str, Any]:
        if model.engine == "demo":
            return _state(model, "ready", 0, 0.0, 100.0, None)
        row = self.states.get(model.engine, model.id)
        if row is None:
            return _state(model, "absent", model.size_bytes, 0.0, 0.0, None)
        return {
            "engine": row["engine"],
            "model_id": row["model_id"],
            "status": row["status"],
            "bytes_total": row["bytes_total"],
            "bytes_done": row["bytes_done"],
            "pct": row["pct"],
            "last_error": row["last_error"],
            "updated_at": row["updated_at"],
        }

    def is_ready(self, model: ModelSpec) -> bool:
        if model.engine == "demo":
            return True
        return bool(self.state_row(model)["status"] == "ready")

    def downloaded_models(self) -> set[str]:
        return {
            model.id
            for engine in self.catalog.engines
            for model in engine.models
            if self.is_ready(model)
        }

    def start(self, engine_key: str, model_id: str) -> dict[str, Any]:
        engine = self.catalog.engine_or_none(engine_key)
        model = self.catalog.model_or_none(model_id)
        if engine is None or model is None or model.engine != engine_key:
            raise ApiError("not_found", "Model is not in the catalog.")
        if model.engine == "demo":
            return self.state_row(model)
        if not model.downloadable or not model.repo:
            raise ApiError("invalid_settings", f"{model.name} has no fetch rule; place its files on disk and scan.")
        key = (model.engine, model.id)
        with self._lock:
            if key in self._active:
                raise ApiError("conflict", f"{model.name} is already downloading.")
            event = threading.Event()
            self._active[key] = event
        self._publish(model, "downloading", model.size_bytes, 0.0, 0.0, None)
        thread = threading.Thread(
            target=self._run,
            args=(model, event),
            name=f"download-{model.engine}-{model.id}",
            daemon=True,
        )
        thread.start()
        return self.state_row(model)

    def cancel(self, engine_key: str, model_id: str) -> dict[str, Any]:
        model = self.catalog.model_or_none(model_id)
        if model is None:
            raise ApiError("not_found", "Model is not in the catalog.")
        with self._lock:
            event = self._active.get((engine_key, model_id))
        if event is not None:
            event.set()
        return self.state_row(model)

    def remove(self, engine_key: str, model_id: str) -> dict[str, Any]:
        model = self.catalog.model_or_none(model_id)
        if model is None:
            raise ApiError("not_found", "Model is not in the catalog.")
        with self._lock:
            if (engine_key, model_id) in self._active:
                raise ApiError("conflict", "Cancel the download before removing the model.")
        target = self.model_dir(model)
        if target.exists():
            shutil.rmtree(target, ignore_errors=True)
        self.states.delete(model.engine, model.id)
        self._publish(model, "absent", model.size_bytes, 0.0, 0.0, None)
        return self.state_row(model)

    def scan(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        with self._lock:
            active = set(self._active)
        for engine in self.catalog.engines:
            for model in engine.models:
                if model.engine == "demo":
                    rows.append(self.state_row(model))
                    continue
                if (model.engine, model.id) in active:
                    rows.append(self.state_row(model))
                    continue
                target = self.model_dir(model)
                if target.exists() and any(path.is_file() for path in target.rglob("*")):
                    size = _dir_size(target)
                    self.states.upsert(
                        model.engine,
                        model.id,
                        status="ready",
                        bytes_total=max(size, model.size_bytes),
                        bytes_done=float(size),
                        pct=100.0,
                    )
                    rows.append(self.state_row(model))
                else:
                    existing = self.states.get(model.engine, model.id)
                    if existing is None or existing["status"] == "ready":
                        self.states.upsert(model.engine, model.id, status="absent", pct=0.0)
                    rows.append(self.state_row(model))
        return rows

    def _run(self, model: ModelSpec, cancel: threading.Event) -> None:
        target = self.model_dir(model)
        target.mkdir(parents=True, exist_ok=True)
        error: list[BaseException] = []
        token = self.token_provider()

        def fetch() -> None:
            try:
                self._slots.acquire()
                try:
                    self._fetch_files(model, target, token)
                finally:
                    self._slots.release()
            except BaseException as exc:  # noqa: BLE001 - reported through model_state
                error.append(exc)

        worker = threading.Thread(target=fetch, name=f"hf-{model.id}", daemon=True)
        worker.start()
        last_emit = 0.0
        try:
            while worker.is_alive():
                worker.join(timeout=POLL_SECONDS)
                if cancel.is_set():
                    self._publish(
                        model,
                        "partial",
                        model.size_bytes,
                        float(_dir_size(target)),
                        _pct(_dir_size(target), model.size_bytes),
                        "download cancelled; partial files kept",
                    )
                    return
                now = time.monotonic()
                if now - last_emit >= EMIT_SECONDS:
                    done = _dir_size(target)
                    self._publish(model, "downloading", model.size_bytes, float(done), _pct(done, model.size_bytes), None)
                    last_emit = now
            worker.join()
        finally:
            with self._lock:
                self._active.pop((model.engine, model.id), None)

        if error:
            message = str(error[0])[:2000]
            self._publish(model, "partial", model.size_bytes, float(_dir_size(target)), 0.0, message)
            return
        if cancel.is_set():
            self._publish(model, "partial", model.size_bytes, float(_dir_size(target)), 0.0, "download cancelled")
            return
        done = _dir_size(target)
        self._publish(model, "ready", max(done, model.size_bytes), float(done), 100.0, None)

    def _fetch_files(self, model: ModelSpec, target: Path, token: str | None) -> None:
        try:
            from huggingface_hub import hf_hub_download, snapshot_download
        except ImportError as exc:
            raise ApiError("download_failed", "huggingface_hub is not installed.") from exc
        assert model.repo is not None
        assert model.fetch is not None
        if model.fetch.kind == "single":
            if not model.fetch.filename:
                raise ApiError("download_failed", f"{model.id} has no filename in its fetch rule.")
            hf_hub_download(
                repo_id=model.repo,
                filename=model.fetch.filename,
                local_dir=str(target),
                token=token,
            )
        else:
            snapshot_download(
                repo_id=model.repo,
                local_dir=str(target),
                allow_patterns=model.fetch.allow_patterns or None,
                token=token,
            )

    def _publish(
        self,
        model: ModelSpec,
        status: str,
        total: int | None,
        done: float,
        pct: float,
        error: str | None,
    ) -> None:
        self.states.upsert(
            model.engine,
            model.id,
            status=status,
            bytes_total=total,
            bytes_done=done,
            pct=pct,
            last_error=error,
        )
        self.hub.publish("model", self.state_row(model))


def _state(
    model: ModelSpec,
    status: str,
    total: int | None,
    done: float,
    pct: float,
    error: str | None,
) -> dict[str, Any]:
    return {
        "engine": model.engine,
        "model_id": model.id,
        "status": status,
        "bytes_total": total,
        "bytes_done": done,
        "pct": pct,
        "last_error": error,
        "updated_at": None,
    }


def _dir_size(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
    except OSError:
        return 0


def _pct(done: int, total: int) -> float:
    if total <= 0:
        return min(99.0, done / 1_000_000.0)
    return min(99.5, max(0.0, done / total * 100.0))


def hf_home_env(paths: Paths) -> None:
    os.environ.setdefault("HF_HOME", str(paths.hf_cache))
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
