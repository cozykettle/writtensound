# WrittenSound - deployment handoff and change queue

Use this file when continuing in a fresh agent session. It records the verified
baseline, where to make changes, and the final Unraid path. Detailed behavior
is still owned by docs 00-08.

## 1. Current verified baseline (2026-10-29)

- Source gates: `ruff`, `mypy`, and `pytest -m "not gpu"` (77 tests) pass on
  the `v0.4.0` tree.
- `writtensound:standard` (0.4.0) was built and smoke-tested locally: 3.54 GB,
  build-time Qwen probe reports `transformers 5.20.2`, `torch 2.9.1+cpu`,
  `torchaudio 2.9.1+cpu`, `torchcodec 0.9.1`, nagisa, soynlp, pyannote.audio,
  and a torchcodec WAV decode; both pinned engine binaries answer `--help`; the
  container runs the app as UID 99, reaches `healthy`, and returns HTTP 200
  from `/healthz`. The smoke run checked the 0.4.0 version string, SVG favicon,
  live `queue` SSE with full rows newest-first, a 12 s demo transcript with a
  real `00:00:12` duration, and the wrapping meta grid.
- `v0.4.0` is a fix release: live queue updates and insertion order (`seq`),
  faster-whisper 1.2.1 `temperature`, HTML date wrapping, whisper.cpp duration
  fallback and short-paragraph reflow, NeMo second/millisecond handling, the
  documented Parakeet dropped-utterance limitation, and a local SVG favicon.
  CUDA is parked: no CUDA image was built or validated for `0.4.0`.
- `v0.3.1` fixes the released-image Qwen failure: torchcodec 0.9.1,
  transformers 5.20.2, numpy 2, nagisa/soynlp, the `libpython3.12` OS package,
  float32 PCM windows, token slicing past the prompt, the 5.20.2 aligner API
  with segment windowing, and device-driven dtype. `v0.3.0` and `v0.2.0`
  remain published and unchanged.
- `v0.3.0` adds whisper.cpp special-token filtering and subword merging, and
  repairs Qwen3 `chat_template.jinja` for both ASR and the ForcedAligner.
  Paragraph output still follows engine segment boundaries (HTML reflow arrived
  in `v0.4.0`).
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
- Tag `v0.3.1` was built by GitHub Actions in 8m56s and published `latest`,
  `0.3.1`, `0.3`, and `sha-814eb59` (digest
  `sha256:ed651b4de76e029a468dfcaf8e9cdc5eea20a282fc174f7d144f82f7331d9800`).
  Registry manifests for all four tags resolve to that digest. The published
  image's Qwen probe passed; a real CPU smoke run inside it produced the exact
  transcript and 16 aligned words. The container started as UID 99, reached
  `healthy`, returned HTTP 200 from `/healthz`, and served the queue page
  reporting `0.3.1`.
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
| 15 | Fix the Qwen3 runtime failure in the released image: pin the torch 2.9-compatible torchcodec, exact transformers, numpy 2 and Japanese/Korean splitters; pass PCM windows instead of file paths; slice the generated tokens; update the aligner to the 5.20.2 API | `pyproject.toml`, `uv.lock`, `backend/engines/qwen3/`, `backend/pipeline/align.py`, `backend/db/catalog.yaml`, docs 02/04, tests | done `0.3.1` |
| 16 | Fail image builds when the Qwen runtime is broken: probe versions, Qwen imports, torchcodec WAV decode, nagisa/soynlp and pyannote in both app layers | `backend/engines/qwen3/runtime.py`, `Dockerfile`, docs 02/06/07 | done `0.3.1` |
| 17 | Add a basic WrittenSound browser icon (speech bubble + sound bars) | `backend/static/favicon.svg`, `backend/templates/base.html`, `docs/05` | done `0.4.0` |
| 18 | Keep the queue live: consume the `queue` SSE event, show jobs added behind a running job, and order rows by insertion (newest first, status never sorts) | `backend/static/app.js`, `backend/runner.py`, `backend/jobs.py`, `backend/db.py`, `docs/03`, `docs/05`, tests | done `0.4.0` |
| 19 | Fix the faster-whisper 1.2.1 call shape (`temperature`, no `progress_hook`) | `backend/engines/fw/args.py`, `backend/engines/fw/adapter.py`, `docs/04`, tests | done `0.4.0` |
| 20 | Stop the HTML date from overlapping `duration`; write the real duration when whisper.cpp omits it | `backend/renderers/templates/transcript.html.j2`, `backend/renderers/__init__.py`, `backend/engines/wcpp/adapter.py`, `docs/04`, tests | done `0.4.0` |
| 21 | Reflow short whisper.cpp HTML paragraphs into readable blocks without losing interval/turn markers | `backend/renderers/__init__.py`, `docs/04`, tests | done `0.4.0` |
| 22 | Stop treating NeMo word times above 1000 s as milliseconds; document Parakeet's dropped-sentence limitation and the absence of an ASR precision flag | `backend/engines/nemo/adapter.py`, `docs/02`, `docs/04`, tests | done `0.4.0` |
| 23 | Fix faster-whisper 1.2.1 audio decode in the released image: `decode_audio()` passes `metadata_errors` to `av.open`, which PyAV 19 removed; pin PyAV 18 and probe the call in image builds | `pyproject.toml`, `uv.lock`, `backend/engines/fw/runtime.py`, `Dockerfile`, `docs/02`, `docs/04`, `docs/07`, tests | done `0.5.0` |
| 24 | Make queue updates fail-safe: return full `JobOut` rows from `POST /api/jobs`, upsert them immediately, subscribe before building the SSE snapshot, and reconcile active rows that disappear without a terminal event | `backend/routes/api.py`, `backend/static/upload.js`, `backend/static/app.js`, `docs/03`, `docs/05`, tests | done `0.5.0` |
| 25 | Replace full-filename output links with downloads grouped under the file name and clear per-format buttons (HTML, VTT, TXT, SRT, timed text, JSON) | `backend/static/app.js`, `backend/static/app.css`, `docs/05`, tests | done `0.5.0` |
| 26 | Fix speaker labels missing from HTML/TXT when an engine returns one long segment (NeMo): split turns at word-level speaker changes, not only at `segment.speaker` | `backend/renderers/__init__.py`, `docs/04`, tests | done `0.5.0` |

Rows 12-14 ship in `v0.3.0`; rows 15-16 ship in `v0.3.1`; rows 17-22 ship in
`v0.4.0`; rows 23-26 ship in `v0.5.0`. The CUDA target stays parked: no CUDA
build or validation is part of `0.4.0` or `0.5.0`.

### 0.4.0 plan (2026-10-28)

`0.4.0` is a fix release driven by eight user reports. The source of truth for
each item is the row above plus docs 02–05. Implementation order:

1. **Docs first.** Record the root causes and the queue/HTML contracts here and
   in docs 02–05 before editing code.
2. **Icon.** A local SVG in `backend/static/` linked from `base.html`; no remote
   asset, no font, no build step.
3. **Queue live update and order.** The server already publishes a `queue` event
   whenever a job is enqueued, but the client had no listener for it, so jobs
   added behind a running job stayed invisible until they started. The client
   will merge `queue` payloads into its job list. `queue` now carries full
   `JobOut` objects instead of the unused minimal shape. Every `JobOut` gains
   `seq` (SQLite rowid) and rows sort by `seq` descending, so the newest job is
   on top and a running job sits below jobs added after it. Status is not a sort
   key.
4. **faster-whisper.** Version 1.2.1 takes singular `temperature` and has no
   `progress_hook`; the adapter already detects the hook, so only the frozen
   kwargs and the hard-coded version message change. Heuristic progress remains
   and is logged.
5. **HTML meta.** The date cell overflowed because the flex child kept its
   min-content width. Add `min-width: 0` and `overflow-wrap: anywhere` to the
   meta cells. whisper.cpp's JSON has no `result.duration`, so the adapter falls
   back to `probe_media(prep.wav)`; duration is no longer `00:00:00`.
6. **HTML reflow.** whisper.cpp emits progressively shorter segments late in a
   file. Merge consecutive short paragraphs inside a turn (target ~80 words /
   560 characters), breaking on timestamps, pauses, and speaker turns. This is a
   renderer mitigation only; engine segments stay unchanged for captions/JSON.
7. **NeMo.** Fix the seconds/milliseconds heuristic so words after 1000 s keep
   their real times. Document that Parakeet can drop short utterances, that
   NeMo-Speech.cpp splits long offline input automatically, and that the CLI has
   no ASR precision flag. No engine substitution, no silent fallback.
8. **Verify and release.** `ruff`, `mypy backend`, `pytest -m "not gpu"`, build
   the standard image, run the demo smoke test, tag/publish `v0.4.0`, and record
   the digest and verification in `docs/process-notes.md`.

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

**0.3.1 release notes (2026-10-27).** `v0.3.1` fixes Qwen3 in the published
image. The engines extra pins `torchcodec==0.9.1` for torch 2.9,
`transformers==5.20.2`, `numpy>=2,<3`, `nagisa==0.3.0`, and `soynlp==0.0.493`;
both app layers install `libpython3.12` for torchcodec's FFmpeg-6 extension.
ASR and the aligner pass 16 kHz float32 PCM windows, slice generated tokens
past the prompt, use the 5.20.2 aligner API with 300 s windowing, and choose
float32 on CPU or bf16/float16 on CUDA. Image builds now run a Qwen runtime
probe. The source gates pass with 68 non-GPU tests, and the published image
reproduced the exact CPU transcript plus 16 aligner words. Tag verification and
the digest are in `docs/process-notes.md`; `v0.3.0` and `v0.2.0` remain
published and unchanged.

**0.4.0 release notes (2026-10-29).** `v0.4.0` is a fix release for eight
reported issues. The queue consumes the `queue` SSE event (full `JobOut` rows
for `pending` and `running` jobs) and orders rows by `seq` (SQLite rowid)
descending, so a newly added job appears immediately on top and a running job
sits below later additions. faster-whisper 1.2.1 is called with
`temperature=0.0` (not `temperatures`) and keeps heuristic progress. HTML
transcript metadata wraps long ISO dates; whisper.cpp duration falls back to
`probe_media(prep.wav)`; short whisper.cpp paragraphs reflow to roughly 80
words / 560 characters without losing timestamps, pauses, or speaker turns.
NeMo word times above 1000 s stay in seconds, and Parakeet's dropped-utterance
behaviour is documented as an upstream limitation with no CLI precision flag.
A local SVG favicon ships. CUDA remains parked: no CUDA image was built or
validated for this release. The source gates pass with 77 non-GPU tests. Tag
verification and the published digest are in `docs/process-notes.md`;
`v0.3.1` and earlier remain published and unchanged.

**0.5.0 plan (2026-10-30).** `0.5.0` fixes the remaining faster-whisper crash
and two UI gaps reported after `0.4.0`:

1. **faster-whisper decode.** faster-whisper 1.2.1 calls
   `av.open(..., metadata_errors="ignore")` inside `decode_audio()`. PyAV 19
   removed that keyword, so the pinned engine stack resolved `av 19.0.2` and
   every transcription failed with `open() got an unexpected keyword argument
   'metadata_errors'`. Pin `av==18.0.0` in the engines extra, add
   `backend/engines/fw/runtime.py`, and run it in both app layers so a broken
   PyAV build fails the image, not the first job.
2. **Queue after a fast failure.** A job row was still only created by SSE.
   `POST /api/jobs` will return the full new `JobOut` rows and the upload client
   will upsert them before the toast; `/api/events` will subscribe before
   building the snapshot so no event can fall into the connect gap; and the
   client will re-fetch any `pending`/`running` row that vanishes from a `queue`
   payload without a terminal `job` event. This heals the case where a job
   failed while the stream was reconnecting, which previously required a page
   refresh.
3. **Downloads.** The detail row will group outputs by base file name and show
   one clear button per format (`HTML`, `TXT`, `VTT`, `SRT`, timed text, `JSON`)
   with the full filename and size in the tooltip, instead of one link per
   complete filename.
4. **NeMo speaker labels in HTML/TXT.** NeMo returns the whole file as one
   segment with top-level words, so `segment.speaker` is only the majority
   label. Timed text/SRT/VTT already split cues at word-level speaker changes;
   TXT and HTML grouped turns by `segment.speaker` and lost the labels. The
   turn builder now splits a segment into speaker-homogeneous runs first, so
   all formats show the same `Speaker N` boundaries.

The release uses the supplied 30-minute MP3 (kept untracked) and a trimmed copy
for local engine tests. CUDA stays parked. Version bump, docs 02-07, tests, image
build, tag workflow, and published-image verification follow the `0.4.0` path.

**0.5.0 release notes (2026-10-30).** `v0.5.0` fixes the remaining
faster-whisper crash and three output problems reported after `0.4.0`.
faster-whisper 1.2.1's `decode_audio()` passes `metadata_errors` to `av.open`;
PyAV 19 removed the keyword, and the engines extra had resolved `av 19.0.2`.
The engines extra now pins `av==18.0.0`, and both app layers run
`python -m backend.engines.fw.runtime`, which imports the engine and decodes a
probe WAV through the exact `decode_audio()` path. `POST /api/jobs` returns the
full new `JobOut` rows, the upload client upserts them immediately,
`/api/events` subscribes before building its snapshot, and the client re-fetches
an active row that disappears from a `queue` payload without a terminal `job`
event. Outputs are grouped under the base file name with one button per format
(`HTML`, `TXT`, `SRT`, `VTT`, timed text, `JSON`). TXT/HTML turns now split at
word-level speaker changes, so NeMo's one-segment transcripts keep their
`Speaker N` headings. The source gates pass with 84 non-GPU tests. The local
`final-standard` image passes both runtime probes; inside it, faster-whisper's
`decode_audio` decoded the supplied 26-minute MP3, and a 90 s clip transcribed
with Whisper small on CPU into all six formats. The tag workflow built in 8m31s
and published `latest`, `0.5.0`, `0.5`, and `sha-21d59a6` at digest
`sha256:8222ab737895f5b0af778f6ab26d420d87c67cca0cfa3817475f87446026692b`;
all four registry manifests resolve to that digest. The published image runs as
UID 99, reaches `healthy`, serves `0.5.0`, passes the faster-whisper probe,
decodes the full MP3, and renders all six outputs for a demo job. Tag
verification is recorded in `docs/process-notes.md`; `v0.4.0` and earlier
remain published and unchanged. CUDA remains parked: no CUDA image was built or
validated for this release.

At the end of a change set, append the decision and verification date to
`docs/process-notes.md`.

## 6. Version-control note

The repository is initialized on `main` and tracks the public remote. Keep
changes in small Conventional Commits and never include `.venv`, `data`,
models, media, logs, or credentials. A source-only rollback archive was kept
outside the repository during the initial build-up; do not add such archives
to Git or GitHub Releases.
