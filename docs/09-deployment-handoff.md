# WrittenSound - deployment handoff and change queue

Use this file when continuing in a fresh agent session. It records the verified
baseline, where to make changes, and the final Unraid path. Detailed behavior
is still owned by docs 00-08.

## 1. Current verified baseline (2026-10-23)

- Source gates: `ruff`, `mypy`, and `pytest -m "not gpu"` (33 tests) pass.
- `writtensound:standard` was built and smoke-tested: CPU torch
  `2.9.1+cpu`, both pinned engine binaries answer `--help`, the demo job
  produced all five output formats, and the container ran as UID/GID 99/100.
- Docker Desktop on Windows cannot pass `/dev/dri`; use
  `docker-compose.cpu.yml` there. CPU + the demo engine are the supported
  Windows test path.
- CUDA is **not** the release target here. The CUDA Python layer installs
  `torch 2.9.1+cu128`; the full `final-cuda` image was not completed. No
  `writtensound:cuda` image is claimed.
- Real engine inference, A380 Vulkan execution, and real model weights remain
  Phase A work; see `docs/process-notes.md`.

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

Preferred: build the standard target on the Unraid host from the release
source, then use `unraid/writtensound.xml` and mount:

```text
/mnt/user/appdata/writtensound -> /data
```

The template passes `/dev/dri`; remove that mapping for CPU-only Unraid.

Alternative: build on Windows, export, copy, and load:

```powershell
docker save writtensound:standard -o writtensound-standard.tar
```

```bash
docker load -i writtensound-standard.tar
```

Before starting on Unraid:

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
| 3 | *(next requested change)* | | pending |

At the end of a change set, append the decision and verification date to
`docs/process-notes.md`.

## 6. Version-control note

This workspace currently has no Git executable or repository. Install Git for
Windows and initialize a baseline before making a large change set, or keep a
source-only snapshot archive outside the repository. Never include `.venv`,
`data`, models, media, or logs in either.
