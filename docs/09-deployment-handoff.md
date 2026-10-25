# WrittenSound - deployment handoff and change queue

Use this file when continuing in a fresh agent session. It records the verified
baseline, where to make changes, and the final Unraid path. Detailed behavior
is still owned by docs 00-08.

## 1. Current verified baseline (2026-10-25)

- Source gates: `ruff`, `mypy`, and `pytest -m "not gpu"` (34 tests) pass.
- `writtensound:standard` was built and smoke-tested: CPU torch
  `2.9.1+cpu`, both pinned engine binaries answer `--help`, the demo job
  produced all five output formats, and the container ran as UID/GID 99/100.
- Tag `v0.1.0` was built by GitHub Actions in 8m21s and published publicly to
  GHCR as `ghcr.io/cozykettle/writtensound:latest` (digest
  `sha256:e95fceeb2efa8b50df3668e437243eae8226c1223b2ce0605e4faed18e98735b`).
  An anonymous pull, health check, and updated Engines page were verified.
- Tag `v0.1.1` published `latest`, `0.1.1`, `0.1`, and `sha-e75f3a7` (digest
  `sha256:a844d0d2425ca4f1fdc358b298ffab2eb45a34d6023ee84e4561a13d0ff45f7a`).
  The image starts as UID/GID 99/100, creates `/home/writtensound` for the
  Vulkan shader cache, reaches `healthy`, and parses a NeMo transcript preceded
  by the shader warning and line-delimited `words` output.
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
| 5 | *(next requested change)* | | pending |

At the end of a change set, append the decision and verification date to
`docs/process-notes.md`.

## 6. Version-control note

The repository is initialized on `main` and tracks the public remote. Keep
changes in small Conventional Commits and never include `.venv`, `data`,
models, media, logs, or credentials. A source-only rollback archive was kept
outside the repository during the initial build-up; do not add such archives
to Git or GitHub Releases.
