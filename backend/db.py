"""SQLite persistence. DDL is normative from docs/03 §3 (WAL enabled)."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .jobs import utc_now

SCHEMA_VERSION = 1

DDL = """
CREATE TABLE IF NOT EXISTS meta (
  key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS uploads (
  id TEXT PRIMARY KEY,
  client_file_id TEXT,
  original_name TEXT NOT NULL,
  relative_path TEXT,
  stored_path TEXT NOT NULL,
  size_bytes INTEGER NOT NULL,
  content_type TEXT,
  state TEXT NOT NULL DEFAULT 'staging',
  created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS jobs (
  id TEXT PRIMARY KEY,
  created_at TEXT NOT NULL,
  title TEXT NOT NULL,
  kind TEXT NOT NULL,
  settings TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'pending',
  stage TEXT,
  progress REAL,
  message TEXT,
  resolved TEXT,
  error TEXT,
  log_paths TEXT,
  started_at TEXT,
  finished_at TEXT,
  duration_s REAL);
CREATE TABLE IF NOT EXISTS model_state (
  engine TEXT, model_id TEXT,
  status TEXT NOT NULL DEFAULT 'absent',
  bytes_total INTEGER, bytes_done REAL, pct REAL,
  last_error TEXT, updated_at TEXT,
  PRIMARY KEY (engine, model_id));
CREATE INDEX IF NOT EXISTS jobs_status ON jobs (status, created_at);
"""


class Database:
    def __init__(self, path: Path) -> None:
        self.path = path

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=30000")
        conn.execute("PRAGMA synchronous=NORMAL")
        try:
            yield conn
        finally:
            conn.close()

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as conn:
            conn.executescript(DDL)
            conn.execute(
                "INSERT INTO meta(key, value) VALUES('schema_version', ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (str(SCHEMA_VERSION),),
            )
            conn.execute(
                "UPDATE jobs SET status='interrupted', message='app restarted during run', finished_at=? "
                "WHERE status='running'",
                (utc_now(),),
            )

    def ping(self) -> bool:
        try:
            with self.connect() as conn:
                conn.execute("SELECT 1").fetchone()
            return True
        except sqlite3.Error:
            return False

    def query(self, sql: str, params: tuple[Any, ...] = ()) -> list[sqlite3.Row]:
        with self.connect() as conn:
            return list(conn.execute(sql, params).fetchall())

    def query_one(self, sql: str, params: tuple[Any, ...] = ()) -> sqlite3.Row | None:
        rows = self.query(sql, params)
        return rows[0] if rows else None

    def execute(self, sql: str, params: tuple[Any, ...] = ()) -> None:
        with self.connect() as conn:
            conn.execute(sql, params)


class JobStore:
    def __init__(self, db: Database) -> None:
        self.db = db

    def create(
        self,
        *,
        job_id: str,
        title: str,
        settings: dict[str, Any],
        kind: str = "upload",
        created_at: str | None = None,
    ) -> None:
        self.db.execute(
            "INSERT INTO jobs(id, created_at, title, kind, settings, status, progress) "
            "VALUES(?, ?, ?, ?, ?, 'pending', 0.0)",
            (job_id, created_at or utc_now(), title, kind, json.dumps(settings)),
        )

    def get(self, job_id: str) -> sqlite3.Row | None:
        return self.db.query_one("SELECT * FROM jobs WHERE id=?", (job_id,))

    def require(self, job_id: str) -> sqlite3.Row:
        row = self.get(job_id)
        if row is None:
            raise KeyError(job_id)
        return row

    def list_jobs(self, *, status: str | None = None, limit: int = 100, offset: int = 0) -> list[sqlite3.Row]:
        if status:
            return self.db.query(
                "SELECT * FROM jobs WHERE status=? ORDER BY created_at DESC, id DESC LIMIT ? OFFSET ?",
                (status, limit, offset),
            )
        return self.db.query(
            "SELECT * FROM jobs ORDER BY created_at DESC, id DESC LIMIT ? OFFSET ?",
            (limit, offset),
        )

    def pending(self) -> list[sqlite3.Row]:
        return self.db.query("SELECT * FROM jobs WHERE status='pending' ORDER BY created_at ASC, id ASC")

    def all_for_snapshot(self, limit: int = 250) -> list[sqlite3.Row]:
        return self.list_jobs(limit=limit)

    def update(self, job_id: str, **fields: Any) -> None:
        allowed = {
            "status",
            "stage",
            "progress",
            "message",
            "resolved",
            "error",
            "log_paths",
            "started_at",
            "finished_at",
            "duration_s",
            "settings",
            "title",
        }
        updates: dict[str, Any] = {}
        for key, value in fields.items():
            if key not in allowed:
                raise ValueError(f"cannot update job field {key!r}")
            if key in {"resolved", "error", "log_paths", "settings"} and value is not None:
                value = json.dumps(value)
            updates[key] = value
        if not updates:
            return
        assignments = ", ".join(f"{key}=?" for key in updates)
        self.db.execute(f"UPDATE jobs SET {assignments} WHERE id=?", (*updates.values(), job_id))

    def set_error(self, job_id: str, error: dict[str, Any], message: str) -> None:
        self.update(job_id, status="failed", error=error, message=message, finished_at=utc_now())

    def delete(self, job_id: str) -> None:
        self.db.execute("DELETE FROM jobs WHERE id=?", (job_id,))

    def counts(self) -> dict[str, int]:
        rows = self.db.query("SELECT status, COUNT(*) AS n FROM jobs GROUP BY status")
        return {row["status"]: int(row["n"]) for row in rows}


class UploadStore:
    def __init__(self, db: Database) -> None:
        self.db = db

    def create(
        self,
        *,
        upload_id: str,
        original_name: str,
        relative_path: str | None,
        stored_path: str,
        size_bytes: int,
        content_type: str | None,
        client_file_id: str | None,
    ) -> None:
        self.db.execute(
            "INSERT INTO uploads(id, client_file_id, original_name, relative_path, stored_path, size_bytes, "
            "content_type, state, created_at) VALUES(?, ?, ?, ?, ?, ?, ?, 'staging', ?)",
            (
                upload_id,
                client_file_id,
                original_name,
                relative_path,
                stored_path,
                size_bytes,
                content_type,
                utc_now(),
            ),
        )

    def get(self, upload_id: str) -> sqlite3.Row | None:
        return self.db.query_one("SELECT * FROM uploads WHERE id=?", (upload_id,))

    def require(self, upload_id: str) -> sqlite3.Row:
        row = self.get(upload_id)
        if row is None:
            raise KeyError(upload_id)
        return row

    def set_state(self, upload_id: str, state: str) -> None:
        self.db.execute("UPDATE uploads SET state=? WHERE id=?", (state, upload_id))

    def list_by_state(self, state: str) -> list[sqlite3.Row]:
        return self.db.query("SELECT * FROM uploads WHERE state=? ORDER BY created_at", (state,))

    def delete(self, upload_id: str) -> None:
        self.db.execute("DELETE FROM uploads WHERE id=?", (upload_id,))


class ModelStateStore:
    def __init__(self, db: Database) -> None:
        self.db = db

    def get(self, engine: str, model_id: str) -> sqlite3.Row | None:
        return self.db.query_one(
            "SELECT * FROM model_state WHERE engine=? AND model_id=?", (engine, model_id)
        )

    def list(self) -> list[sqlite3.Row]:
        return self.db.query("SELECT * FROM model_state ORDER BY engine, model_id")

    def upsert(
        self,
        engine: str,
        model_id: str,
        *,
        status: str,
        bytes_total: int | None = None,
        bytes_done: float | None = None,
        pct: float | None = None,
        last_error: str | None = None,
    ) -> None:
        self.db.execute(
            "INSERT INTO model_state(engine, model_id, status, bytes_total, bytes_done, pct, last_error, updated_at) "
            "VALUES(?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(engine, model_id) DO UPDATE SET status=excluded.status, bytes_total=excluded.bytes_total, "
            "bytes_done=excluded.bytes_done, pct=excluded.pct, last_error=excluded.last_error, "
            "updated_at=excluded.updated_at",
            (engine, model_id, status, bytes_total, bytes_done, pct, last_error, utc_now()),
        )

    def delete(self, engine: str, model_id: str) -> None:
        self.db.execute("DELETE FROM model_state WHERE engine=? AND model_id=?", (engine, model_id))

    def state_map(self) -> dict[str, dict[str, Any]]:
        result: dict[str, dict[str, Any]] = {}
        for row in self.list():
            result.setdefault(row["engine"], {})[row["model_id"]] = {
                "engine": row["engine"],
                "model_id": row["model_id"],
                "status": row["status"],
                "bytes_total": row["bytes_total"],
                "bytes_done": row["bytes_done"],
                "pct": row["pct"],
                "last_error": row["last_error"],
                "updated_at": row["updated_at"],
            }
        return result
