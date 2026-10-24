# WrittenSound — Development Guide

Coding-agent-facing implementation plan: exact repo layout, module responsibilities, coding standards, dependency pins, test strategy and a phase-by-phase build order with acceptance criteria. When this document conflicts with docs 00–06 in a *design* matter, fix this doc; when it conflicts on *facts* (versions, flags), consult `02 § 8` spikes which are the authoritative last word.

---

## 1. Repository layout (complete, target end-state)

```
writtensound/
├── AGENTS.md                       # agent entrypoint (08 agent brief distilled)
├── README.md                       # human-facing quick start (compose up, screenshots)
├── LICENSE                         # MIT
├── pyproject.toml                  # uv-managed; pins in [project.dependencies]
├── uv.lock
├── ruff.toml, mypy.ini
├── Dockerfile
├── docker-compose.yml
├── docker-build-args.env           # WHISPERCPP_TAG=…, NEMOSPEECH_TAG=…
├── .env.example
├── .github/workflows/docker.yml    # optional tag builds
├── unraid/writtensound.xml
├── docs/                           # docs 00–09
├── backend/
│   ├── pyproject.toml  uv.lock     # (the same as root; keep only root pyproject, backend package)
│   ├── app.py                      # create_app(), boot wiring
│   ├── entrypoint.sh               # PUID/PGID drop, exec uvicorn
│   ├── config.py                   # AppConfig load/save, env merge
│   ├── db.py                       # sqlite open/migrate (DDL from 03 §3), jobs/DAO fns
│   ├── catalog.py                  # load catalog.yaml → EngineSpec/ModelSpec dataclasses
│   ├── db/catalog.yaml             # normative catalog (content == docs/02 §2)
│   ├── hardware.py                 # DeviceReport probe (vulkan/cuda/ffprobe/core-count)
│   ├── events.py                   # SSE hub (subscribe/publish), event dataclasses
│   ├── uploads.py                  # streaming multipart → staging, ffprobe validate
│   ├── jobs.py                     # JobStore DAO + JobOut serializers
│   ├── runner.py                   # queue thread: pick/execute/cancel/retry; stage weights
│   ├── renderers/                  # 04 §5 contracts
│   │   ├── html.j2  html.py  txt.py  srt.py  vtt.py  json_out.py
│   ├── pipeline/
│   │   ├── merge.py                # 04 §4.1 algorithm + Pause pass
│   │   ├── align.py                # qwen3-aligner orchestration
│   │   └── prep.py                 # ffmpeg clip + wav extract helpers
│   ├── engines/
│   │   ├── base.py                 # EngineAdapter/DiarizerAdapter Protocols (01 §5)
│   │   ├── fw/       adapter.py args.py
│   │   ├── wcpp/     adapter.py args.py
│   │   ├── nemo/     adapter.py args.py
│   │   ├── qwen3/    adapter.py args.py
│   │   └── pyannote/adapter.py args.py
│   ├── downloader.py               # hf snapshot fetch, progress, cancel, verify
│   ├── routes/
│   │   ├── api.py                  # all /api/* from 03 §6 (router factories)
│   │   ├── pages.py                # /, /engines, /settings
│   │   └── partials.py             # htmx fragments (settings form, queue rows)
│   ├── templates/ (queue.html engines.html settings.html partials/*.html)
│   ├── static/       (app.css app.js htmx.min.js htmx-ext-sse.js)
│   └── tests_mocks/  (stub binaries for subprocess engines)
├── tests/
│   ├── conftest.py                 # tmp data dir, fake hardware, tiny audios
│   ├── test_settings.py test_jobs_api.py test_merge.py test_renderers.py
│   ├── test_catalog.py test_sse.py test_boot.py
│   └── integration/  (GPU-gated, env-gated)
├── tools/
│   ├── probe_spike.sh              # Phase A spike collection script
│   └── e2e_smoke.sh                # boots compose, runs a job, asserts outputs
└── samples/jfk.wav                 # public-domain test clip (whisper.cpp sample)
```

The tree above is the target end-state. The current preview ships
`backend/`, `tests/`, `docs/`, `unraid/`, and the Docker/release files; the
optional CI workflow and Phase A `tools/`/`samples/` helpers are not shipped
yet, so Phase A and a scripted real-engine E2E remain open.

## 2. Dependency pins (`pyproject.toml`)

```toml
[project]
name = "writtensound"
version = "0.1.0"
requires-python = ">=3.12,<3.13"
dependencies = [
  "fastapi>=0.118,<1",
  "uvicorn[standard]>=0.37",
  "pydantic>=2.11,<3",
  "jinja2>=3.1",
  "python-multipart>=0.0.20",
  "aiofiles>=24",
  "PyYAML>=6.0",
  "huggingface-hub>=1.0,<2",
]
[project.optional-dependencies]
engines = [
  "faster-whisper==1.2.1",
  "ctranslate2>=4.5,<5",
  "transformers>=5.13,<6",
  "torch==2.9.*",
  "torchaudio==2.9.*",
  "pyannote.audio>=4.0,<5",
]
dev = ["pytest>=9", "pytest-asyncio", "httpx", "ruff>=0.14", "mypy>=1.18", "types-aiofiles", "types-PyYAML"]
```

`uv sync --frozen` in image builds. The committed lock resolves `torch`/`torchaudio` from the PyTorch CPU index declared in `pyproject.toml`; the CUDA final layer replaces those two wheels with cu128 builds. Do not add `ffmpeg-python` or any Python ffmpeg wrapper — use the system `ffmpeg`/`ffprobe` binaries only.

## 3. Coding standards

- **Typed everywhere**; `mypy --strict` clean on `backend/` (engine internals exempted via `# type: ignore[no-untyped-call]` only where interop requires).
- **ruff**: line 120, `E,F,I,UP,N,B,SIM` selected; no `try-except-pass`; imports sorted.
- **No comments** unless explaining a *why* (ties to the repo agent rules). Docstrings only on public API of adapters.
- **No global mutable state** besides `AppState` singleton created in `app.create_app()`.
- **Filesystem ops**: `pathlib` only, no bare `open()` without encoding exceptions for binary.
- **Subprocess**: always `subprocess.Popen` with explicit `env` copy + `bufsize=1` text mode; never `shell=True`.
- **Time**: store UTC ISO-8601 with `Z`; display local, convert server-side only for UI strings.
- **Frontend**: no build step; JS must not reference external CDNs; all templates compile server-side; `{{ }}` escaping always via `|e` where variables come from job content.
- **Errors**: raise `ApiError(code, message, details)` in domain layer; routers never invent error payloads.

## 4. Phase plan (build order with acceptance criteria)

Each phase ends in a committed, tested increment. Phases may not start before their prerequisites; SPIKE gating applies.

| Phase | Delivers | Acceptance criteria (tests must pass) |
|---|---|---|
| **P0 scaffold** | repo skeleton, `pyproject`+lock, config/db/catalog loaders, healthz, static bootstrap of `/`, `uv sync` in CI-like script, lint/type gates wired | `pytest tests/test_boot.py` green; `ruff check .` + `mypy backend` clean; compose builds at least the standard target with *stub* engine binaries (mock placeholders that echo valid JSON) |
| **P-A spike** | `tools/probe_spike.sh` runs against pinned engine binaries/libs; findings appended to `docs/02 § 2` with concrete JSON/args; `SPIKE` flags in catalog turned into fixed fields | every `SPIKE:` resolved or explicitly "absent" recorded in a table in docs 02; `args.py` files updated with the frozen strings |
| **P1 core loops** | uploads (staging+ffprobe), jobs CRUD + settings validation, SSE hub, hardware probe, queue runner skeleton, `_stub` engine that emits known-word JSON for tests | `test_jobs_api.py`, `test_sse.py`, `test_settings.py` green; a `_stub` job runs end-to-end producing `out/*` files via the renderers |
| **P2 faster-whisper** | fw adapter (+device resolution), real words IR from info + merge, starts from config defaults; pyannote OUT of scope here | `test_merge.py` green with synthetic fw dict; manual e2e on jfk.wav via `POST /api/jobs` produces ir.json + txt |
| **P3 whisper.cpp + nemo** | subprocess adapters with parsed JSON per frozen args; device index selection; progress parsing | jfk.wav e2e on CPU for both engines: outputs equal or better than fw baseline text (manual diff, no auto golden), progress events no gaps |
| **P4 diarization + merge** | pyannote + nemo diar adapters; merge pipeline; speaker labels; turn/interval ts passes; pause pass | `test_merge_diar.py` — synthetic turns + words produce expected grouping (fixtures in JSON); engine-gated integration marked `-m gpu` includes nemo-on-A380 |
| **P5 renderers** | all 5 renderers with fixtures: transcript HTML snapshot test (signed), srt/vtt timing, txt formatting, JSON IR shape | `test_renderers.py` full green, incl. word-cue grouping edge cases (words with `end=None`, overlapping turns, non-latin scripts) |
| **P6 UI complete** | queue page, engines page, settings page, settings fragment, modal, drag/folder, SSE wiring, toasts, noscript fallback | full e2e smoke `tools/e2e_smoke.sh` green (spins up app, uploads jfk.wav through the HTTP path, cancels, retries, downloads outputs) |
| **P7 docker + docs** | actual Dockerfile builds standard+cuda on CI-ish machine with `docker-build-args.env` pinned tags; unraid template; README; version bump | `docker compose up -d` fresh clone works; both images present and healthchecks pass; user-facing docs complete |
| **P8 optional extras** | qwen3-aligner engine (word-ts fallback), parakeet-tdt as bonus model, watch-folder ingestion, `apply below` button, batched fw toggle, editor groundwork | each feature behind its own flag with tests |

## 5. Test strategy

- **Unit (fast, hermetic)**: merge algorithm, renderer fixtures, catalog capability matrix consistency, settings validation, config precedence. No GPU/network.
- **Contract/gate (hermetic)**: catalog YAML must satisfy the *capability matrix* in docs 02 §5 — consistency test asserts each engine's flags match doc rows (parse the doc? no: a hand-checked roster test replicating docs' table in code with a comment citing the doc line).
- **Integration (marked, hermetic-safe)**: `_stub` engine always; live engines via env `WS_TEST_ENGINE=fw|wcpp|nemo|qwen3|pyannote` making pytest skip otherwise. Test container mounts `samples/jfk.wav`.
- **E2E**: `tools/e2e_smoke.sh` — compose standard build → wait `/healthz` → upload `samples/jfk.wav` through real HTTP → poll SSE (`curl`-based) → assert all 5 output files exist and HTML loosely contains expected transcript fragments ("my fellow Americans").
- **Manual A380 checklist** after P5 (record in `docs/process-notes.md`): vulkan engine run, pause marking, 8-speaker nemo diar, OOM fallback path.

## 6. Process notes

- All code-level decisions not covered by this doc must go to `docs/process-notes.md` (created in P0) with one-line rationale — the agent appends, never rewrites.
- Commits: Conventional Commits (`feat:`, `fix:`, `docs:`, `spike:`, `test:`, `chore:`). Squash into main frequently, ideally one commit per phase milestone.
- When engine docs/CLI change upstream, update `02 § 2` + the relevant `args.py` **in the same commit** that consumes the change.
- CI (optional, cheap): ruff + mypy + pytest non-gpu on every push; docker build on tags only.
- Never commit model weights, upload media, or logs. `.gitignore` includes `/data`, `samples/*.wav` except `samples/jfk.wav`, `.venv/`.
- Keep `docs/` updated: any behavioral change lands with a matching doc edit — a failing doc/implementation consistency check is a review blocker.
