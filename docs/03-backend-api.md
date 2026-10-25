# WrittenSound — Backend & API Specification

Normative for `backend/app` HTTP surface, persistence, and event streaming. Pydantic model names in this doc are the *actual* names to create. FastAPI serves OpenAPI at `/docs` automatically; the UI never relies on OpenAPI.

---

## 1. Environment variables (image contract)

| Var | Default | Meaning |
|---|---|---|
| `WS_DATA_DIR` | `/data` | root of all state (see 01 §3) |
| `WS_PORT` | `8080` | uvicorn port inside container |
| `APP_PASSWORD` | _(unset → auth off)_ | when set, HTTP Basic gate on everything except `/healthz`; user name irrelevant (levelapp "WrittenSound") |
| `HF_TOKEN` | _(unset)_ | optional bootstrap; prefer Settings-first (§6) |
| `HF_HOME` | `/data/models/hf-cache` | hub cache |
| `WS_LOG_LEVEL` | `INFO` | `DEBUG|INFO|WARNING` |
| `WS_DISABLE_HW_PROBE` | `0` | skip probes at boot (unit tests) |
| `CUDA_VISIBLE_DEVICES` | unset | passthrough for cuda variant |

Everything else lives in `/data/config/app-config.json` (§6).

## 2. App config file (`/data/config/app-config.json`)

```jsonc
{
  "schema_version": 1,
  "hf_token": null,                    // overrides env when set (unmasked; warning shown in UI §6)
  "defaults": { "$ref": "#job-settings-below" },   // seed for new job forms
  "engine_enabled": { "faster-whisper": true, "whisper.cpp": true, "nemo-speech": true,
                      "qwen3-asr": true, "pyannote": true, "qwen3-aligner": true },
  "keep_completed_jobs_days": 14,
  "cleanup_uploads": true,             // remove staging after success
  "default_preselect": { "asr_model_id": "wcpp-large-v3-turbo-q8_0",
                         "device": "auto", "language": "auto" }
}
```

- Written by `PUT /api/settings` (full replace of editable keys). Read at boot; missing keys fall back to baked defaults.
- HF_TOKEN precedence: `app-config.json.hf_token` > env `HF_TOKEN`.

## 3. SQLite schema (DDL, applies at boot; WAL on)

```sql
CREATE TABLE IF NOT EXISTS meta (
  key TEXT PRIMARY KEY, value TEXT NOT NULL);            -- schema_version, …
CREATE TABLE IF NOT EXISTS uploads (
  id TEXT PRIMARY KEY,                                   -- uuid4 hex
  client_file_id TEXT,                                   -- client dedupe key
  original_name TEXT NOT NULL,
  relative_path TEXT,                                    -- from folder upload ('2025_interviews/a.wav')
  stored_path TEXT NOT NULL,                             -- /data/jobs/staging/{id}_{name}
  size_bytes INTEGER NOT NULL,
  content_type TEXT,
  state TEXT NOT NULL DEFAULT 'staging',                 -- staging|ready|consumed|failed
  created_at TEXT NOT NULL);                             -- ISO-8601 UTC
CREATE TABLE IF NOT EXISTS jobs (
  id TEXT PRIMARY KEY,                                   -- uuid4 hex == job_id used on disk
  created_at TEXT NOT NULL,
  title TEXT NOT NULL,                                   -- basename(s)
  kind TEXT NOT NULL,                                    -- 'upload'  (watch-folder rev later)
  settings TEXT NOT NULL,                                -- JSON JobSettings (frozen at start; editable pre-start)
  status TEXT NOT NULL DEFAULT 'pending',                -- pending|running|succeeded|failed|cancelled|interrupted
  stage TEXT,                                            -- ingest|prepare|asr|diarize|align|render
  progress REAL, message TEXT,                           -- 0..1
  resolved TEXT,                                         -- JSON: {asr_model, diar_model, device, compute}
  error TEXT,                                            -- JSON ApiError
  log_paths TEXT,                                        -- JSON list under /data/jobs/{id}/logs
  started_at TEXT, finished_at TEXT, duration_s REAL);
CREATE TABLE IF NOT EXISTS model_state (
  engine TEXT, model_id TEXT,                            -- PK(engine, model_id)
  status TEXT NOT NULL DEFAULT 'absent',                 -- absent|partial|downloading|ready
  bytes_total INTEGER, bytes_done REAL, pct REAL,
  last_error TEXT, updated_at TEXT,
  PRIMARY KEY (engine, model_id));
CREATE INDEX IF NOT EXISTS jobs_status ON jobs (status, created_at);
```

- `uploads.state == 'consumed'` — referenced by a queued job (interned path; cleanup honored post-job).

## 4. Job settings schema `JobSettings` (normative, pydantic v2)

```python
class TimestampsMode(str, Enum): none="none"; interval="interval"; turns="turns"
class Diar(str, Enum): off="off"; nemo="nemo"; pyannote="pyannote"
class Device(str, Enum): auto="auto"; cpu="cpu"; cuda="cuda"; vulkan="vulkan"

class OutputFormats(BaseModel):
    html: bool = True; txt: bool = True; srt: bool = True
    vtt: bool = False; timed_txt: bool = False; json: bool = True

class JobSettings(BaseModel):
    schema_version: int = 1
    asr_model_id: str                        # catalog id (any engine)
    device: Device = Device.auto
    language: str = "auto"                   # "auto" | ISO-639-1 (whisper) | locale-code (nemotron) | human name (qwen)
    diar: Diar = Diar.off
    diar_model_id: str | None = None         # required when diar != off
    speaker_count: int | Literal["auto"] = "auto"     # 2..8 when int
    timestamps_mode: TimestampsMode = TimestampsMode.interval
    interval_s: int = 60                     # 10..3600 when mode == interval
    word_timestamps: bool = False            # words against every word; captions use them
    mark_pauses: bool = False
    pause_min_s: float = 2.0                 # 0.5..10
    verbatim: bool = False                   # fillers (whisper-family only)
    start_at: str | None = None              # hh:mm:ss  → clip
    end_at: str | None = None
    output: OutputFormats = OutputFormats()
    out_basename: str | None = None          # override default naming
```

Validation rules (enforced by pydantic + route checks, error code `invalid_settings`):
- `diar_model_id` required & from `diar` engine family when `diar != off`.
- `diar_model_id`/'pyannote-community-1'|'pyannote-3.1' only for `diar=pyannote`; nemo ids for `diar=nemo`.
- `speaker_count` int → clamp 2..8 (nemo nemo-3 supports 8; sortformer 4; request still allowed — diarizer caps internally, warning logged).
- `interval_s` int 10..3600.
- `start_at`/`end_at` regex `^\d{1,2}:\d{2}:\d{2}$`; end > start.
- At least one output format selected.
- SRT, VTT, and timed text are caption formats: `timestamps_mode=none` is rejected when any is selected (`invalid_settings`); the UI disables those checkboxes in that mode instead of inventing times.
- Feature gate errors (server-side re-check of `02 § 5 matrix`): e.g. `word_timestamps=true` requires model capability `native` or aligner installed; `verbatim` requires whisper-family → `error.invalid_settings` with human `message` the UI shows inline.

## 5. Hardware/engine status report `HardwareStatus`

```python
class DeviceInfo(BaseModel): platform: str; slug: str   # slug: cpu|cuda|vulkan; platform free-form ("Intel Arc A380")
class HardwareStatus(BaseModel):
    image_variant: str            # "standard" | "cuda" (baked env WS_IMAGE_VARIANT)
    devices: dict[str, DeviceInfo]    # merged cpu/vulkan/cuda/silence
    cpu_cores: int; ram_gb: float
    vulkan_ok: bool; cuda_ok: bool
    notes: list[str]              # human warnings ("CUDA image supports CUDA; standard image cannot enable it")
```

Producers: boot + `POST /api/status/refresh`. Resolution logic (`device=auto`): **whisper.cpp / nemo-speech** order `vulkan > cuda > cpu`; **faster-whisper / qwen3 / pyannote** order `cuda > cpu`. Chosen device stored into `jobs.resolved` when the job starts.

## 6. HTTP API (paths, verbs, request/response)

All JSON; all require auth when enabled; all return `application/json` except files. Prefix `/api`.

| # | Method Path | Purpose & notes |
|---|---|---|
| 1 | `GET /healthz` | liveness/readiness — no auth. Returns `{"ok": true}` (+ DB ping). |
| 2 | `GET /` | queue page HTML (htmx). |
| 3 | `GET /settings` `GET /engines` | static pages (htmx). |
| 4 | `GET /api/status` → `HardwareStatus` | cached; `POST /api/status/refresh` re-probes async (emits `hardware` SSE when done). |
| 5 | `GET /api/settings` → current config (hf_token **masked** `hf_****last4`) |  |
| 6 | `PUT /api/settings` body: subset of §2 keys, `hf_token: str|null` (**full value**) → 200 echo masked | validates: hf_token must match `^hf_[A-Za-z0-9]{30,}`; wrong → `invalid_settings` |
| 7 | `POST /api/uploads` (multipart, field `file`; headers `X-Relative-Path`, `X-Client-Id`) → `{"upload_id": "…"}` | streams to `/data/jobs/staging/{upload_id}_{basename}`; SAFE filename (`re.sub(r'[^\w\-. ()\[\]]','_')`); whitelist by extension from catalog: `wav mp3 m4a ogg opus flac aac wma aiff mp4 mkv mov webm avi mpg mpeg wmv`; ffprobe check BEFORE returning 200; content >0 bytes; on reject → 415 `unsupported_media` (reason text). Big files fine (uvicorn spools to disk). |
| 8 | `POST /api/jobs` body `{"items":[{"upload_id":str, "settings":JobSettings}, …]}` → `{"job_ids":[…]}` | settings validated **per item** (per-user per-item overrides); bulk response 400 with `items_errors: [{index, ApiError}]` partial acceptance **not allowed** (atomically all-or-nothing to avoid half-queues). |
| 9 | `GET /api/jobs?status=&limit=&offset=` → `{"jobs":[JobOut]}` newest-first | `JobOut` = job row + `outputs_manifest: [ {name, size_ct, modified} ]` when present. |
| 10 | `GET /api/jobs/{id}` → `JobOut` | includes `settings` JSON (client opens modal to re-edit when pending). |
| 11 | `PATCH /api/jobs/{id}` body: partial JobSettings | only when `status == pending`; emits `job` SSE. |
| 12 | `DELETE /api/jobs/{id}` | when pending → remove; when running → cancel first then remove (async; SSE events); outputs/logs retained per retention policy. |
| 13 | `POST /api/jobs/{id}/cancel` | running → cancel (S2A below), pending → becomes `cancelled`. |
| 14 | `POST /api/jobs/{id}/retry` | re-queues job **at the tail**; on `failed` status ID keeps history (now `interrupted`→status reset pending). |
| 15 | `POST /api/jobs/{id}/fallback-cpu` | failed-with-device-error; sets `device=cpu` and re-queues. Only enabled when device error category. |
| 16 | `GET /api/jobs/{id}/outputs` → manifest list (fresh from disk) | names: `{job_id}{suffix}.{ext}` foldered. |
| 17 | `GET /api/jobs/{id}/outputs/{name}` | `FileResponse` w/ `Content-Disposition: attachment`; names validated against sanitized manifest only. |
| 18 | `GET /api/catalog` → merged catalog + per-model `state` (from model_state) + per-device support (resolved against HardwareStatus) | single source feed for engines page and settings dropdowns. |
| 19 | `POST /api/models/{engine}/{model_id}/download` | async task; model_state → `downloading` with SSE `model` events; concurrent downloads allowed (max 2 by config); returns 409 if already downloading. |
| 20 | `POST /api/models/{engine}/{model_id}/cancel` | cancels download; state → `partial`. |
| 21 | `POST /api/models/{engine}/{model_id}/remove` | deletes files (only `ready`); state → absent. |
| 22 | `POST /api/models/scan` | rescans `/data/models/*` on disk; binds loose files; returns merged catalog. |
| 23 | `POST /api/engines/{key}/enable` `…/disable` (mirror of config toggle) | disables → engine UI grey; pending jobs selecting it stay valid (warn only). |

### 5.1 SSE: `GET /api/events` (EventSource)

Content-Type: `text/event-stream`, `Cache-Control: no-cache`. First event on connect is always:

```
event: snapshot
data: {"hardware": HardwareStatus, "jobs": [JobOut…], "model_state": {engine:[ModelState…]}, "engine_enabled": {…}}
```

Then incremental:

| event | payload | frequency |
|---|---|---|
| `job` | Single `JobOut` (full replace for that row) | on any change (status/progress/stage/message) ≤2 Hz per job |
| `queue` | minimal `[{job_id, status, progress}]` array for ordering | whenever order changes (rare; also emitted on boot) |
| `hardware` | full `HardwareStatus` | on refresh |
| `model` | one `ModelState` row | when model download progresses (throttled ≤1 Hz), completes, or aborts |
| `upload` | `{upload_id, state, error?}` | when an upload's staging state changes (server-side ffprobe verdict etc.) |
| `engine` | `{engine, enabled}` | on toggle |
| `toast` | `{level, text, source}` | rare (warnings, HW probe surprises, engine crash tails) |

SSE heartbeats: `: hb\n\n` every 15 s so proxies keep the connection. Reconnect: client re-open receives fresh `snapshot`. **All queue UI updates come from these events**, no polling.

### 5.2 Error model (uniform)

```python
class ApiError(BaseModel):
    code: str     # one-of: invalid_settings | unsupported_media | not_found | conflict |
                  # unsupported_device | engine_crash | ffmpeg_failed | download_failed |
                  # hf_auth | storage | internal
    message: str  # human, self-explaining
    details: dict | None
```

HTTP mapping: 400 invalid_settings/unsupported_media; 404; 409 conflict; 415 unsupported_media; 500 internal. Every job `error` column uses ApiError JSON verbatim so the UI can style by `code` (e.g. `hf_auth` inline shows the pyannote setup pane link; `engine_crash` suggests device swap; `unsupported_device` offers `fallback-cpu`).

## 7. App boot sequence (deterministic)

1. Load config (§2) + validate; create dirs; open DB (WAL).
2. Load catalog `backend/db/catalog.yaml` (frozen copy mirrored in section `02`; live entries override `SPIKE` flags).
3. Detect hardware (unless disabled) → cache HardwareStatus.
4. Scan `/data/models/*` → reconcile `model_state` rows from disk presence.
5. Start DownloadManager (idle), queue worker thread, SSE hub.
6. Rows with `status=running` from a previous process → set `interrupted` (message "app restarted during run").
7. Bind uvicorn.

Order matters for tests: boot is exercised end-to-end in `tests/test_boot.py` with `WS_DATA_DIR` pointed into tmp_path.
