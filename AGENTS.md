# WrittenSound — agent instructions

## Before ANY work

Read, in order: `docs/00-overview.md` → `docs/01-architecture.md` → `docs/02-engines-and-models.md` → `docs/03-backend-api.md` → `docs/04-pipelines.md` → `docs/05-frontend-ui.md` → `docs/06-docker-deployment.md` → `docs/07-development-guide.md` → `docs/08-agent-brief.md` → `docs/09-deployment-handoff.md`.

## Non-negotiables

- Follow `docs/07 § 4` phase order; P-A spike (engine CLI/library verification) before touching engine adapters.
- Engine flags live ONLY in `backend/engines/*/args.py` after being verified against pinned upstream releases (`docker-build-args.env`).
- `backend/db/catalog.yaml` must stay in sync with `docs/02 § 2`; any change updates both in one commit.
- No silent fallbacks (device switches, capability downgrades) — surface via job log/SSE toast.
- Tests: `ruff check`, `mypy backend`, `pytest -m "not gpu"` must pass before every commit.
- No scope creep; no extra endpoints/fields beyond `docs/03`; no frontend framework.
- Conventional Commits; never commit models, media, or logs.

## Project facts

- Python 3.12, FastAPI + Jinja2 + htmx (server-rendered), SQLite (WAL), uv-package-managed, no node toolchain.
- Engines: faster-whisper (CUDA/CPU), whisper.cpp (Vulkan/CUDA/CPU, GGML), NeMo-Speech.cpp (Vulkan/CUDA/CPU, GGUF — Nemotron ASR + diarization), Qwen3-ASR + Qwen3-ForcedAligner (CUDA/CPU via transformers), pyannote (CUDA/CPU).
- Target deployment: Unraid + Intel Arc A380 (Vulkan); optional CUDA image variant.
- Docs are authoritative; missing decisions go to `docs/process-notes.md` with a one-line rationale.
