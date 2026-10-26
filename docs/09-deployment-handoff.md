# WrittenSound - deployment handoff and change queue

Use this file when continuing in a fresh agent session. It records the verified
baseline, where to make changes, and the final Unraid path. Detailed behavior
is still owned by docs 00-08.

## 1. Current verified baseline (2026-10-25)

- Source gates: `ruff`, `mypy`, and `pytest -m "not gpu"` (53 tests) pass on
  the `v0.3.0` tree.
- `writtensound:standard` was built and smoke-tested: CPU torch
  `2.9.1+cpu`, both pinned engine binaries answer `--help`, the demo job
  produced the default five output formats, and the container ran as UID/GID 99/100.
- `v0.3.0` adds whisper.cpp special-token filtering and subword merging, and
  repairs Qwen3 `chat_template.jinja` for both ASR and the ForcedAligner.
  Paragraph output still follows engine segment boundaries.
- `v0.2.0` adds NeMo word preservation, multi-cue captions, timed text,
  single-speaker label suppression, word-boundary interval markers, and the
  restored Add jobs engine/model selector. The compact UI overhaul is
  `a986c26`; the `v0.2.0` tag adds the transcript changes on top.
- Tag `v0.1.0` was built by GitHub Actions in 8m21s and published publicly to
  GHCR as `ghcr.io/cozykettle/writtensound:latest` (digest
  `sha256:e95fceeb2efa8b50df3668e437243eae8226c1223b2ce0605e4faed18e98735b`).
  An anonymous pull, health check, and updated Engines page were verified.
- Tag `v0.1.1` published `latest`, `0.1.1`, `0.1`, and `sha-e75f3a7` (digest
  `sha256:a844d0d2425ca4f1fdc358b298ffab2eb45a34d6023ee84e4561a13d0ff45f7a`).
  The image starts as UID/GID 99/100, creates `/home/writtensound` for the
  Vulkan shader cache, reaches `healthy`, and parses a NeMo transcript preceded
  by the shader warning and line-delimited `words` output.
- Tag `v0.2.0` was built by GitHub Actions in 9m and published `latest`,
  `0.2.0`, `0.2`, and `sha-f0a1126` (digest
  `sha256:d4f56f8a0a4cd5a8ddaf79a286f6a4a62d625f32d908f164b1d2678f983dce43`).
  Registry manifests for all four tags resolve to that digest. The container
  started as UID/GID 99/100, reached `healthy`, returned HTTP 200 from
  `/healthz`, and served the queue page reporting `0.2.0` with the model
  selector and timed-text option.
- Tag `v0.3.0` was built by GitHub Actions in 8m41s and published `latest`,
  `0.3.0`, `0.3`, and `sha-9b935e1` (digest
  `sha256:e64334c7f7e4745c1ba086a149881d094fefb3f0702c00104322bda4203f7efc`).
  Registry manifests for all four tags resolve to that digest. The container
  started as UID/GID 99/100, reached `healthy`, returned HTTP 200 from
  `/healthz`, and served the queue page reporting `0.3.0`.
- Docker Desktop on Windows cannot pass `/dev/dri`; use
  `docker-compose.cpu.yml` there. CPU + the demo engine are the supported
  Windows test path.
- CUDA is **not** the release target here. The CUDA Python layer installs
  `torch 2.9.1+cu128`; the full `final-cuda` image was not completed. No
  `writtensound:cuda` image is claimed.
- NeMo-Speech CPU inference produced transcript text on a real Parakeet TDT v3
  job. Full-context Parakeet on the 6 GB Arc A380 still fails with
  `ErrorOutOfDeviceMemory` (7.27 GB graph); use whisper.cpp for long A380 jobs.
  Other real-model inference and A380 Vulkan execution remain Phase A work; see
  `docs/process-notes.md`.

## 2. Where requested changes belong

| Change | Primary files |
|---|---|
| Wording, layout, controls, colors | `backend/templates/`, `backend/static/app.css`, `backend/static/app.js` |
| Default model, device, output style | `backend/config.py`, `backend/jobs.py`, `backend/db/catalog.yaml` |
| Model list, sizes, repos, capabilities | `backend/db/catalog.yaml` and `docs/02-engines-and-models.md` together |
| Engine CLI flags | `backend/engines/*/args.py`, only after Phase A verification |
| Ports, data path, auth, PUID/PGID | `.env.example`, `docker-compose.yml`, README, docs/06 |
| Unraid defaults | `unraid/writtensound.xml` |
| Docker/runtime behavior | `Dockerfile`, `backend/entrypoint.sh`, docs/06 |

Do not hard-code deployment paths in templates; use the passed `paths` values.

## 3. Fast iteration loop

Run the app directly from source while editing:

```bash
uv run writtensound --reload --port 8091
```

Use a disposable data directory on Windows:

```powershell
$env:WS_DATA_DIR = Join-Path $env:TEMP "ws-browser"
$env:WS_DISABLE_HW_PROBE = "1"
uv run writtensound --reload --port 8091
```

Then run the gates:

```bash
uv run ruff check .
uv run mypy backend
uv run pytest -m "not gpu"
```

Rebuild only after the source is settled:

```bash
docker build --target final-standard -t writtensound:standard .
```

On Windows Docker Desktop, test the image without `/dev/dri`:

```bash
docker compose -f docker-compose.yml -f docker-compose.cpu.yml up -d --build
```

## 4. Final Unraid deployment

Preferred: pull the public standard image from GHCR with
`unraid/writtensound.xml`:

```text
ghcr.io/cozykettle/writtensound:latest
/mnt/user/appdata/writtensound -> /data
```

The template passes `/dev/dri`; remove that mapping for CPU-only Unraid.
No registry login is needed once the GHCR package is public. Use a version
tag instead of `latest` for deliberate updates.

Alternative: build the standard target on Unraid from the release source and
point the template at the local `writtensound:standard` image. This avoids a
registry dependency but uses local build cache and takes substantially longer.
Do not use a `docker save` tarball as the release artifact; the GHCR image is
already the transfer mechanism.

Before starting on Unraid:

- ensure the Docker image file or directory has room for the roughly 3.3 GB
  image plus update headroom;
- set `PUID=99` and `PGID=100` (or the share owner);
- set `WS_DATA_DIR`/the template data path;
- set `APP_PASSWORD` if the LAN is not trusted;
- confirm `/dev/dri/renderD128` exists for Vulkan;
- choose and download the intended model only after the container is healthy.

## 5. Change queue and log

Add each requested change here before implementing it, then update its status.
This is the durable list when chat context is lost.

| # | Change | Files | Status |
|---|---|---|---|
| 1 | Scrub third-party app references from docs, UI, package metadata, and deployment metadata | README, docs, `backend/templates/engines.html`, `pyproject.toml`, `unraid/writtensound.xml` | done |
| 2 | Add the AI-slop warning to the README | `README.md` | done |
| 3 | Publish the standard CPU + Vulkan image through GitHub Actions to public GHCR and point the Unraid template at it | `.github/workflows/docker.yml`, `unraid/writtensound.xml`, README, docs | done `v0.1.0` |
| 4 | Fix NeMo JSON parsing after stderr diagnostics and create the runtime home for the Vulkan shader cache | `backend/engines/nemo/adapter.py`, `backend/entrypoint.sh`, `tests/test_nemo_adapter.py` | done `v0.1.1` |
| 5 | Preserve NeMo top-level word timestamps and stop collapsing SRT/VTT to one whole-file cue | `backend/engines/nemo/adapter.py`, `tests/test_nemo_adapter.py`, renderer fixtures | done `0.2` |
| 6 | Add the timed-text output format (`[HH:MM:SS.mmm --> HH:MM:SS.mmm] text`) and its settings/API/UI contract | `docs/00`, `docs/03`, `docs/04`, `docs/05`, `backend/jobs.py`, `backend/config.py`, `backend/routes/common.py`, `backend/renderers/__init__.py`, settings template, tests | done `0.2` |
| 7 | Suppress the `Speaker N:` prefix for single-speaker HTML/TXT/SRT/VTT/timed output; keep it for multi-speaker files | renderers, `docs/04`, `docs/05`, tests | done `0.2` |
| 8 | UI overhaul using Impeccable | `backend/templates/`, `backend/static/`, Impeccable `PRODUCT.md`/`DESIGN.md` and surface briefs | done `0.2` (`a986c26`) |
| 9 | Register Impeccable with OpenCode so the UI overhaul can use the skill tool | `~/.config/opencode/skills/impeccable`, `~/.config/opencode/commands/impeccable.md` | done 2026-10-25 |
| 10 | Restore a visible engine/model selector to the Add jobs form and remove the read-only model strip | `backend/templates/partials/settings_form.html`, `backend/templates/queue.html`, `docs/05`, `tests/test_boot.py` | done `0.2` |
| 11 | Split interval timestamps at the first word past each interval so long single-segment transcripts get multiple markers | `backend/renderers/__init__.py`, `tests/test_renderers.py`, `docs/04` | done `0.2` |
| 12 | Filter whisper.cpp special/timestamp tokens and merge subword tokens into words; stop using `--max-len 1`, which wraps tokens into separate segments | `backend/engines/wcpp/adapter.py`, `backend/engines/wcpp/args.py`, `backend/pipeline/merge.py`, `docs/04`, tests | done `0.3` |
| 13 | Download Qwen3's separate `chat_template.jinja` (`*.jinja` was excluded from the snapshot allow-list) and repair older Qwen snapshots at load time | `backend/db/catalog.yaml`, `backend/engines/qwen3/adapter.py`, `docs/02`, tests | done `0.3` |
| 14 | Apply the same chat-template repair when loading the Qwen3 ForcedAligner so older aligner snapshots do not need a manual re-download | `backend/engines/qwen3/chat_template.py`, `backend/pipeline/align.py`, `backend/runner.py`, `docs/02`, `docs/04`, tests | done `0.3` |

Rows 12–14 ship in `v0.3.0`; the `v0.2.0` image predates them.

### 0.2 implementation notes (2026-10-25)

`0.2` combines transcript timestamp correctness with the compact UI. The
normative contract is now in docs 00, 03, 04, and 05; this section records the
decisions that are easy to lose.

**Transcript/timestamp work.**

- NeMo Parakeet returns top-level `words[]` with no `segments[]`. The adapter
  now maps those words into one `Segment` with its `words` list intact and uses
  the payload text when present. It only falls back to a word-less segment when
  the payload has no words; caption output then uses ASR segment boundaries and
  logs that limitation.
- SRT, VTT, and timed text share one cue-grouping path. With words, cues start
  at speaker changes (multi-speaker files only), gaps > 1 s, 12 words, or 5 s.
  Without words, ASR segments are the cues.
- `timed_txt` (`{base}.timed.txt`) is one line per cue:
  `[HH:MM:SS.mmm --> HH:MM:SS.mmm] text`; no indices, no `WEBVTT` header, no
  blank line between cues. It is a caption format for validation, so
  `timestamps_mode=none` rejects it.
- Speaker prefixes are emitted only when the file has more than one speaker.
  Single-speaker HTML, TXT, SRT, VTT, and timed text do not show
  `Speaker 1:`. Multi-speaker files keep labels.
- Interval mode splits long segments at the first word at or after each
  interval threshold, so a one-segment NeMo transcript gets multiple markers.
  Turn starts still get a marker. HTML/TXT produce one paragraph per interval
  chunk.
- No silent fallback: if an engine has no word timings, segment-level cues are
  used; the limitation remains visible in the job log.

**UI.** The Add jobs form has the visible `Engine and model` select grouped by
engine; selecting a model re-renders the shared settings fragment and updates
device/language/word-timing dependencies. The read-only model strip and hidden
`asr_model_id` input are gone. Output formats are `HTML | TXT | SRT | VTT |
Timed text | JSON`; caption formats are disabled when timestamp style is
`None`.

**Impeccable.** Skill 4.7.0 / engine v0.1.16 remains registered globally at
`~/.config/opencode/skills/impeccable` (plus
`~/.config/opencode/commands/impeccable.md`). No `PRODUCT.md` exists; the init
interview was canceled. Do not commit the roughly 20 MB engine binary or the
global install.

**0.3 release notes (2026-10-25).** `v0.3.0` is a fix release. whisper.cpp
transcripts no longer contain `[_TT_*]`/`[_BEG_]` tokens or split subword
pieces, and `--max-len 1` is gone. Qwen3 snapshots now include
`chat_template.jinja`; both the ASR and ForcedAligner loaders repair older
snapshots by fetching only that file. The source gates pass with 53 non-GPU
tests. Tag verification, including the image digest, is recorded in
`docs/process-notes.md`; `v0.2.0` remains published and unchanged.

At the end of a change set, append the decision and verification date to
`docs/process-notes.md`.

## 6. Version-control note

The repository is initialized on `main` and tracks the public remote. Keep
changes in small Conventional Commits and never include `.venv`, `data`,
models, media, logs, or credentials. A source-only rollback archive was kept
outside the repository during the initial build-up; do not add such archives
to Git or GitHub Releases.
