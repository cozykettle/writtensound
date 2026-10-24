"""Staging retention and housekeeping."""

from __future__ import annotations

import shutil
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from .jobs import utc_now


def cleanup_all(state: Any) -> dict[str, int]:
    return {
        "staging_removed": cleanup_staging(state),
        "jobs_removed": cleanup_completed(state),
    }


def cleanup_staging(state: Any) -> int:
    referenced = {
        row["stored_path"]
        for row in state.uploads.list_by_state("staging") + state.uploads.list_by_state("ready")
    }
    removed = 0
    now = time.time()
    if state.paths.staging.exists():
        for path in state.paths.staging.iterdir():
            if not path.is_file():
                continue
            if str(path) in referenced:
                continue
            try:
                if now - path.stat().st_mtime < 3600:
                    continue
            except OSError:
                continue
            path.unlink(missing_ok=True)
            removed += 1
    for state_name in ("consumed", "failed"):
        for row in state.uploads.list_by_state(state_name):
            if not Path(row["stored_path"]).exists():
                state.uploads.delete(row["id"])
    return removed


def cleanup_completed(state: Any, *, now: datetime | None = None) -> int:
    days = int(state.config.keep_completed_jobs_days)
    if days <= 0:
        return 0
    cutoff = (now or datetime.now(UTC)) - timedelta(days=days)
    removed = 0
    for row in state.jobs.list(limit=1000):
        if row["status"] not in {"succeeded", "failed", "cancelled", "interrupted"}:
            continue
        finished = row["finished_at"] or row["created_at"]
        try:
            moment = datetime.fromisoformat(finished.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            continue
        if moment >= cutoff:
            continue
        job_dir = state.paths.job_dir(row["id"])
        shutil.rmtree(job_dir, ignore_errors=True)
        state.jobs.delete(row["id"])
        removed += 1
    return removed


def cleanup_upload(store: Any, paths: Any, upload_id: str) -> None:
    row = store.get(upload_id)
    if row is None:
        return
    Path(row["stored_path"]).unlink(missing_ok=True)
    store.delete(upload_id)


def stale_upload_rows(state: Any) -> None:
    for row in state.uploads.list_by_state("staging"):
        if not Path(row["stored_path"]).exists():
            state.uploads.set_state(row["id"], "failed")


def now_iso() -> str:
    return utc_now()
