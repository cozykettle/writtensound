# WrittenSound — Architecture

## 1. Component diagram (inside one container)

```
┌────────────────────────── writtensound container ──────────────────────────┐
│                                                                            │
│  uvicorn (FastAPI app, port 8080)                                          │
│   ├── / (Jinja2 templates + htmx) ── static (htmx, css, tiny JS)           │
│   ├── REST API /api/*            ├── SSE /api/events                       │
│   ├── UploadStreamer (multipart → /data/jobs/staging)                      │
│   └── AppState (hardware probe, engine registry, config, auth)             │
│                                                                            │
│  JobRunner (single-threaded queue owner)                                   │
│   ├── serial loop: pick job → run pipeline → emit progress → save outputs  │
│   ├── Pipeline per job:                                                    │
│   │     clip (ffmpeg) → ENGINE ADAPTER → diar ADAPTER → merge → render     │
│   └── EngineModelCache (LRU, keep ≤1 warm model by default)                │
│                                                                            │
│  EnginePackagers:                                                          │
│   ├── FasterWhisperAdapter  (in-process, CTranslate2)                      │
│   ├── WhisperCppAdapter     (subprocess: whisper-cli)                      │
│   ├── NemoSpeechAdapter     (subprocess: nemo-speech; diarize + asr)       │
│   ├── Qwen3AsrAdapter       (in-process, transformers)                     │
│   └── PyannoteAdapter       (in-process, pyannote.audio)                   │
│                                                                            │
│  DownloadManager (huggingface_hub snapshot_download w/ progress events)    │
│  SQLite (WAL): jobs, settings, engine state, logs   + /data filesystems    │
└────────────────────────────────────────────────────────────────────────────┘
      │                        │                       │
   /data/jobs/*           /data/models/*           GPU devices
   (staging, outputs,   (per-engine layout;      (/dev/dri/renderD128 for
    logs, db, config)    HF cache subtree)        Vulkan; merged /dev/dri via
                                                  nvidia runtime & cudnn/devices
                                                  in cuda variant)
```

## 2. Process & threading model (exact)

- **uvicorn** runs 1 async event loop. All HTTP/SSE handling is async.
- **JobRunner** owns a dedicated `threading.Thread` — the only place blocking engine calls happen. The API touches runner state only via thread-safe methods on a `QueueManager` (mutex-guarded).
- **Engine adapters** are sync classes. Subprocess adapters stream `stdout` line-by-line on a reader thread to feed progress into a `Profiler` object.
- **Model load is serialized** under a global lock: only one model (ASR or diar) is resident at a time. After a job finishes, the current model is kept in cache unless `settings.unload_model_after_job` is set (default: keep last one).
- **No background jobs other than**: hardware/engine probe (startup + on demand), downloads (async task, guarded), staging cleanup (periodic, hourly).

## 3. Data layout (volumes)

```
/data
├── config
│   └── app-config.json          # persisted UI settings, HF token, defaults
├── jobs
│   ├── staging/                 # upload landing area (chunked whole files)
│   │   └── {upload_id}_{original_name}
│   ├── {job_id}/
│   │   ├── media/               # job's working input (may be hardlink/symlink
│   │   │   ├── original.*       #  to staging; plus WAV extracted for engines)
│   │   │   └── prep.wav         # 16k mono s16 wav (C++ engines, diarization)
│   │   ├── out/                 # final outputs named {base}.{ext}
│   │   └── logs/                # engine stdout/stderr + renderer log
│   └── writtensound.db          # SQLite (WAL)
└── models                       # everything weight-related
    ├── whispercpp/              # ggml files
    ├── nemo/                     # one dir per model, one *.gguf each
    ├── fastwhisper/             # ct2 model dirs
    ├── qwen3/                   # transformers snapshots (HF cache layout)
    ├── pyannote/                # HF-cache format
    └── hf-cache/                # generic HF cache (HF_HOME points here)
```

Rules:
- `/data/jobs` DB holds **every** state; the filesystem is rebuildable from the DB (logs aside).
- Deleted/editable jobs: renames allowed while `status=pending`.
- Retention policy setting: `keep_completed_jobs_days` (default 14) cleanup in staging/out.
- Settings snapshots are stored inline in the `jobs` row (JSON) — never mutated after start.

## 4. Queue & lifecycle overview (details in 03/04)

```
ids: pending → starting → prepared → loading_model → transcribing
     → diarizing → aligning → rendering → done
                                            ↘ failed  (retain to retry)
       cancel from any state → cancelled
```

- Progress: adapters report `(stage, 0..1, message)`. Exact numbers where the engine provides them; heuristic (audio seek position / duration) otherwise. Emitted over SSE as `job_progress` events at ≤2 Hz.
- Retry re-enters at then-current `pipeline checkpoint` (only `rendering` is recomputed without re-running ASR).
- **Serial worker only.** No concurrent jobs (G-goals); downloads may proceed in parallel with a running job (they are GPU-idle-bound).

## 5. Engine adapter interface (single contract)

```python
class EngineAdapter(Protocol):
    name: str                                   # "faster-whisper", ...
    binary_or_module_available: bool

    def detect_devices(self) -> DeviceReport
    def list_models(self) -> list[ModelMeta]    # from catalog + downloaded scan
    def download_model(self, model_id, on_progress) -> None
    def ensure_model_loaded(self, model_id, device) -> LoadedModel
    def transcribe(self, loaded_model, audio_path,
                   job_settings) -> TranscriptionResult
```

- `TranscriptionResult` is **the** normalized internal IR (see `04-pipelines.md §2`): segments with words, optional speaker labels, language, timings.
- Diarization has its own narrow interface:

```python
class DiarizerAdapter(Protocol):
    def diarize(self, audio_path, stream_cfg: DiarSettings) -> DiarResult
    # DiarResult: turns [{start, end, speaker, overlap: bool}]
```

- Both adapters may throw `DeviceUnsupportedError` (UI surfaces as a warning, rules from `02 § capability matrix`) — CPU re-runs only on user-completed "Fallback to CPU" action (explicit, not silent; note in job log).

## 6. Hardware & engine detection at boot

At startup (and on `POST /api/status/refresh`):

- `infer_hardware()`: detects `/dev/dri/renderD*` (Vulkan potential), `torch.cuda.is_available()`, vulkan loader (`libvulkan.so.1` + ICD JSONs), CUDA (ct2), CPU count/RAM.
- Result vector → `AppState.device_report` delivered to UI via SSE `hardware` event; forms use it to grey devices.
- Probe order and costs are kept <2 s; nothing has to make network calls.

## 7. Security posture (v1)

- Binds to `0.0.0.0:8080` inside container. On unraid typically exposed via host port; recommend reverse-proxy/TLS by *user's own setup*. Optional `APP_PASSWORD` → HTTP Basic on everything except `/healthz`.
- Uploads streamed to disk; never parsed into memory (`httpClient`-independent); extension allowlist (audio/video) + `ffprobe` sanity check before acceptance.
- HF token stored in `/data/config/app-config.json` — container-internal; document unmask-the-risk in `06` warning box.

## 8. Failure & retry semantics

| Failure class | Handling |
|---|---|
| Engine crash (subprocess exit ≠0) | job → `failed` with pinned stderr tail in UI; one `Retry` button |
| CUDA/Vulkan failure at load or mid-run (OOM / loader) | job → `failed`; job log suggests: device swap button |
| Bad audio/undecodable | mark `failed` with message EARLY stage (`prepared`) |
| Download failure | model stays `partial`; job remains pending-resumable: user retries download or switches model |
| Cancel | cooperative: adapters checked at segment boundaries; subprocesses killed by process-handle when possible |
| App container death | DB says queued → surfaced as `interrupted` on boot → user Retry |

## 9. Repository strategy

Single repo, single branch `main`, trunk-based squash commits made by the coding agent; docs in `docs/`; special `AGENTS.md` at root with agent instructions. Optional CI (build both image variants on tags).
