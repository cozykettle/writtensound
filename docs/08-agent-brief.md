# WrittenSound — Agent Brief (for the coding agent)

This file is the operational instruction set for an autonomous coding agent (opencode + DeepSeek 4.1 Flash or similar) building WrittenSound from these specs.

**Rule #1 — You build, you don't design.** All design decisions live in `docs/00–07`. If a decision is genuinely missing AND not derivable from those docs, append it to `docs/process-notes.md` with a one-line rationale and proceed with the *conservative* choice.

---

## Read order (in this sequence, before any code)

1. `README.md` (repo root, if present)
2. `AGENTS.md` (repo root) — this file's condensed sibling
3. `docs/00-overview.md` — what & why, desktop workflow mapping
4. `docs/01-architecture.md` — components, threading, storage, adapter contracts
5. `docs/02-engines-and-models.md` — the catalog: THE source of truth for engines/models/devices/grays; note every `SPIKE:` item
6. `docs/03-backend-api.md` — HTTP/SSE contract, DB DDL, JobSettings schema
7. `docs/04-pipelines.md` — IR, per-engine invocation, merge algorithm, renderers
8. `docs/05-frontend-ui.md` — pages, forms, drag-drop mechanics, graying rules
9. `docs/06-docker-deployment.md` — image matrix, Dockerfile skeleton, compose, Unraid
10. `docs/07-development-guide.md` — repo tree, pins, standards, phase plan (your roadmap)

## Operating rules

1. **Follow the phase order** in `docs/07 § 4`. Do not start a phase before its prerequisites. Phase P-A (the spike) comes before P2.
2. **Spike first, freeze strings**: any engine flag (`--output-json-full`, `--device vulkan:0`, fw kwargs, transformers calls) must be verified against the real pinned binary/library in P-A and then frozen in `backend/engines/*/args.py`. Engineers-in-your-head flags are forbidden elsewhere.
3. **Catalog is dynamic**: `backend/db/catalog.yaml` is generated/mirrored from docs 02 §2; if the spike changes a fact (word-ts capability, file list), update BOTH the YAML and the corresponding row in `docs/02 § 2` in the same commit.
4. **No silent fallbacks**: when a device fails or a feature is gated, the app must surface the warning/error (SSE toast or job error) — never auto-retry on a different device without a user-visible record (job log line).
5. **No scope additions** (no watch-folder, no editor, no i18n…) unless the phase plan explicitly says so in a later phase. Extra "polish"501 unrequested features are review-blocked.
6. **Tests drive each acceptance criterion** in docs 07 §4. Run `ruff`, `mypy`, and `pytest` (non-gpu) before every commit.
7. **Commits**: conventional commits, one commit per coherent milestone. Never commit models/audio/logs; `.gitignore` from docs 07 §6 is established in P0.
8. **Docker tags pinned** via `docker-build-args.env`; versions resolved in P0 (whisper.cpp release, NeMo-Speech.cpp release); being stale is fine, drifting to `main` is not.
9. **Where you work**: root repo; on Windows hosts keep paths POSIX inside code/docs; on Linux containers paths are POSIX anyway. Tooling: uv (python), node is NOT part of this project.
10. **HF token privacy never in git**; only in `/data/config` or env at runtime.

## Definition of Done (P7)

- Fresh `git clone` → `docker compose up -d` (standard) works; UI reachable; queue works with real model download from HF.
- Both image variants build on a fat build host; Unraid template validates.
- Docs 00–07 remain consistent with code.
- `tools/e2e_smoke.sh` passes without manual steps.

## Anti-pattern list (these will be considered bugs)

- Inventing API fields not in docs 03 (e.g. extra `speaker_names` out of nowhere).
- Hard-coding engine flags outside `args.py`.
- Using `requests`/`urllib` when `httpx` is available; using `shutil.copy` when streaming uploads matter.
- Frontend frameworks (React/Vue/Alpine) or a node build step.
- Storing tts of progress in job row more often than needed (SSE >2 Hz per job).
- Auto-pulling a gated model without the token flow; crashing on missing token instead of the pyannote setup pane.
