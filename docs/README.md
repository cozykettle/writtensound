# WrittenSound — Documentation Index

A self-hosted, local-first web app for speech-to-text with a multi-engine model catalog (Whisper-family, NVIDIA Nemotron ASR + Sortformer/Nemotron-3 diarization, Qwen3-ASR), a serial job queue with per-job settings, and Vulkan/CUDA/CPU flexibility.

| Doc | Contents |
|---|---|
| `docs/00-overview.md` | Vision, goals/non-goals, stack rationale, desktop workflow mapping, risks, glossary |
| `docs/01-architecture.md` | Component diagram, process model, data layout, adapter contract, detection, failure semantics |
| `docs/02-engines-and-models.md` | Engine registry, model catalog (repos, sizes, langs), device matrix + graying rules, capability matrix, spikes |
| `docs/03-backend-api.md` | Env & config, SQLite DDL, JobSettings schema, all REST routes, SSE events, error model, boot sequence |
| `docs/04-pipelines.md` | Stage machine, IR models, exact engine invocations, merge/pause algorithms, renderers, cancellation, logging |
| `docs/05-frontend-ui.md` | Pages, queue layout, settings form, drag-drop/upload mechanics, engines & settings pages, styling |
| `docs/06-docker-deployment.md` | Image matrix, Dockerfile skeleton, docker-compose, Unraid notes, CI, runbook |
| `docs/07-development-guide.md` | Repo tree, dependency pins, coding standards, phase plan w/ acceptance criteria, tests |
| `docs/08-agent-brief.md` | Operating instructions for the coding agent |
| `docs/09-deployment-handoff.md` | Verified baseline, change queue, iteration loop, and final Unraid deployment checklist |

**Quick start**: see the repository [`README.md`](../README.md). With Docker available: `cp .env.example .env && docker compose up -d --build` → open `http://<host>:8080`. For a local demo without Docker: `uv sync && uv run writtensound`.

Status: **working preview, October 2026.** The web app, demo engine, persistence, uploads, queue, renderers, settings, and browser UI are implemented and tested. Real-engine adapters and Docker engine images are present, but Phase A runtime verification (pinned CLI output schemas, GPU behavior, model downloads) and CUDA/A380 hardware checks remain. See [`process-notes.md`](process-notes.md) for the verification record and known limitations.
