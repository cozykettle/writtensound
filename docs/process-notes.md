# Process Notes

Decisions that are not fully specified in `docs/00–07` are appended here, one
line each, with the conservative rationale. This file is append-only.

## 2026-10-23 — first implementation pass

### Runtime and persistence

- **App factory only.** `backend.app:create_app` is the sole application entry
  point; no module-level app is constructed. This keeps tests, the console
  entrypoint, and uvicorn's factory mode on one lifecycle path.
- **Optional engine stack.** Heavy inference dependencies live in the
  `engines` extra so the demo, API, renderers, and hermetic tests install and
  run on a small machine. Docker images install the extra explicitly.
- **Demo engine.** The built-in `demo` engine is deterministic and exists to
  exercise uploads → queue → outputs without downloading weights. It reports
  resolved compute as `synthetic` rather than claiming CPU/CUDA/Vulkan work.
- **Default word timestamps.** `word_timestamps` stays false in the default
  recipe because word-level rendering is optional and can be expensive; the
  demo test enables it to exercise the full IR path.
- **WAV fallback is visible.** If ffmpeg is unavailable, WAV uploads continue
  but a warning is attached to the job log/response. The production image
  requires ffmpeg; a silent fallback would hide a degraded environment.
- **In-memory pending deque.** Queue ordering is owned by the process. Retries
  append to the tail without changing the original `created_at`, which matches
  the user-visible "retry is a new attempt" expectation.
- **Partial settings PATCH.** Job edits accept a direct partial settings object
  at `PATCH /api/jobs/{id}`. This is the narrowest shape that supports the
  editor and avoids inventing a wrapper field outside docs/03.
- **Aligner fails loudly.** The Qwen3 aligner is structurally present but
  raises a clear unsupported/verification error until Phase A confirms its
  command and output schema. It never silently drops word timestamps.
- **Settings context is centralized.** `settings_form_context()` builds the
  queue form, modal, and partial from one payload so disabled engines,
  device hints, and model choices cannot drift between pages.
- **Hidden means hidden.** `[hidden] { display: none !important; }` prevents
  component display rules from revealing the token-clear control.
- **Cache-busted assets.** The static URL version is the package version plus
  the newest static-file mtime. A server restart is required after static
  edits; this is intentional and avoids stale JS in long-lived tabs.
- **Token UI updates immediately.** Save/remove updates the masked pill,
  placeholder, and clear control without a page reload.
- **Accessible engine switches.** Engine toggles expose `role="switch"`,
  `aria-checked`, and an engine-name label; control state is synchronized by
  both the click handler and SSE.
- **Sub-second durations.** Durations below one second render as `<1s` instead
  of misleading `0s`.
- **Configured data path in the UI.** Engines and Settings display
  `paths.root`/`paths.config_dir`/`paths.models` rather than a hard-coded
  `/data`, so local runs and containers with a custom data directory tell the
  truth.

### Engines and source verification

- **whisper.cpp pin.** `WHISPERCPP_TAG = v1.9.5`; source inspection confirmed
  `--output-json-full`, `--output-file`, `--print-progress`, `--max-len`, and
  the GPU selector `--device` (the earlier `--gpu-device` spelling was invalid
  in the pinned source). The DTW preset names and JSON field names still
  require a Phase A run.
- **NeMo-Speech.cpp pin.** `NEMOSPEECH_TAG = v0.2.0`; its published source
  confirms the `transcribe`/`diarize` subcommands, global `--json`, `--device`
  aliases, `--model`, and `--language`. The output schema and GPU behavior
  remain Phase A items.
- **whisper.cpp DTW is a preset, not a model directory.** The pinned v1.9.5
  source has no `models/dtw` tree; `--dtw MODEL` takes a preset/model name.
  The image does not copy a nonexistent directory, and the adapter keeps token
  offsets from the full JSON output until Phase A validates DTW presets.
- **One CPU lock, CUDA overlay.** The committed `uv.lock` pins
  `torch==2.9.*`/`torchaudio==2.9.*` from the PyTorch CPU index so the
  standard image does not carry CUDA wheels. The CUDA Docker layer installs
  the matching cu128 wheels and asserts the CUDA version, keeping one lock file
  while making both final images honest.
- **No silent device fallback.** Device resolution is explicit. A failure is
  reported through the job error/log and SSE toast; the app never retries on a
  different device without a visible record.
- **Docker engine builds.** Standard uses CPU + Vulkan builds of both C++
  engines. CUDA builds request both CUDA and Vulkan; if an upstream build
  cannot combine them, that must be surfaced as a Docker build failure rather
  than silently shipping a CUDA-only or Vulkan-only binary.

### Deployment

- **Explicit Docker targets, not a PROFILE build arg.** One Dockerfile exposes
  `final-standard` and `final-cuda`. Compose selects the target explicitly;
  this is more reliable than interpolating a final stage name and keeps the two
  runtime bases honest (Ubuntu 24.04 Python for both).
- **CUDA base is 12.8.1 on Ubuntu 24.04.** NVIDIA does not publish the older
  12.4.1 CUDA images on Ubuntu 24.04, and Python 3.12 is required. CUDA 12.8.1
  keeps the shared Python/runtime contract and supports current PyTorch cu128
  wheels.
- **Standard compose service is unprofiled.** `docker compose up -d` therefore
  starts the standard image as documented. The CUDA service is behind the
  `cuda` profile and must be started by service name. Both services retain the
  same container name; running both at once is intentionally rejected.
- **CPU-only compose override.** `docker-compose.cpu.yml` resets the standard
  service's `devices` list with `!reset []`. This covers hosts without
  `/dev/dri` and Docker Desktop on Windows, where Linux render nodes cannot be
  passed into the container. GPU/Vulkan remains a Linux/Unraid deployment.
- **Windows Docker Desktop CUDA is separate from Vulkan.** The WSL2 backend
  exposes NVIDIA GPUs through the `nvidia` runtime, while `/dev/dri` Vulkan
  passthrough remains unavailable. The CUDA image includes `sm_75`, so a
  T600/4 GB can run it, subject to model-size limits; the image still has to be
  built locally because no CUDA artifact is published.
- **CUDA architectures are a build argument.** The release default remains the
  six-architecture fat image; a local single-GPU build can set
  `CUDA_ARCHITECTURES=75` (T600) to avoid compiling unused fatbin variants.
  The resulting image is intentionally architecture-specific.
- **Engine pins live in `docker-build-args.env`.** Compose passes matching
  defaults so a plain `docker compose build` works without sourcing the file;
  changing a tag still requires updating the env file (and `args.py`) together.
- **Extra operational environment variables.** `WS_IMAGE_VARIANT`,
  `WS_MAX_UPLOAD_BYTES`, `WS_ENABLE_DEMO_ENGINE`, `WS_DISABLE_HW_PROBE`,
  `WS_WHISPER_CLI`, and `WS_NEMO_SPEECH` are image/operations controls, not API
  fields. They are documented in the README rather than added to the
  docs/03 API schema.
- **Entrypoint drops privileges after startup.** The image starts as root only
  to create the PUID/PGID user, grant access to the mounted `/dev/dri` group,
  and chown the known data directories; the app process runs as the mapped
  user via `gosu`.

### Verification record (2026-10-23)

- The demo path, settings, token controls, queue SSE, output downloads, job
  editor, and responsive UI were exercised in a real browser. The engine
  switches render as `role="switch"` with synchronized `aria-checked` values.
- `uv run writtensound --help` works through the console entry point.
- `uv run ruff check .` passes.
- `uv run mypy backend` passes (42 source files).
- `uv run pytest -m "not gpu"` passes: 33 tests, 18.88 s, with only dependency
  deprecation warnings.
- `docker build --target final-standard -t writtensound:standard .` succeeds
  from the checked-in Dockerfile. The resulting image is 3.28 GB and contains
  the CPU torch wheel, `whisper-cli` v1.9.5, and `nemo-speech` v0.2.0; both
  `--help` invocations succeed and torch reports `2.9.1+cpu`.
- The standard container starts under UID/GID 99/100, passes `/healthz`, and
  completes an uploaded WAV through the demo engine with
  `ir.json`, `tour.html`, `tour.srt`, `tour.txt`, and `tour.vtt` plus pipeline
  and engine logs. The final image also renders the configured `/data` paths
  and uses 33 tests after adding coverage for the data-directory display.
- `uv build` produces the wheel and sdist; the wheel contains the templates,
  static assets, `catalog.yaml`, and `LICENSE`.
- The CUDA target build was attempted and is recorded below. No CUDA device is
  present in this environment, so CUDA kernel execution cannot be tested here.

## CUDA build attempt (2026-10-23)

- The `app-cuda` layer completed: it synced the committed CPU lock, replaced
  only `torch`/`torchaudio` with `2.9.1+cu128`, installed the cu12 NVIDIA
  wheels, and passed the build-time CUDA 12.8 assertion.
- Docker was still compiling the CUDA+Vulkan engine stages after roughly
  thirty minutes (`whisper-cuda-build` at 113/514 objects,
  `nemo-cuda-build` at 112/560). The build was stopped to free the CPU build
  host, so `final-cuda` did not produce an image. This is a build-time
  limitation, not a runtime result; no CUDA kernels or engine binaries were
  exercised.
- The standard target remains the verified deployment for the Intel Arc A380
  (Vulkan) and CPU-only hosts.

## Pending Phase A checks

These are intentionally unresolved and must not be described as production
verified:

- Run the pinned whisper.cpp and NeMo-Speech.cpp binaries on CPU, Vulkan, and
  CUDA; capture actual JSON field names, progress output, and failure modes.
- Verify the `--dtw` preset names and output behavior for whisper.cpp word
  timestamps; v1.9.5 has no `models/dtw` directory.
- Verify NeMo diarization JSON and speaker-count behavior.
- Verify faster-whisper and pyannote with real weights, including gated-token
  flows.
- Verify the Qwen3-ASR and aligner adapters against `transformers`/model output.
- Run the CUDA image on an NVIDIA host and the standard image on the target
  Intel Arc A380 host.

## GHCR release path (2026-10-24)

- Release builds use GitHub Actions on tag pushes (`v*`) plus manual dispatch.
  The workflow publishes only the standard CPU + Vulkan target; CUDA remains a
  local/optional build.
- `v0.1.0` produced public image tags `latest`, `0.1.0`, `0.1`, and
  `sha-961fb42` at digest
  `sha256:e95fceeb2efa8b50df3668e437243eae8226c1223b2ce0605e4faed18e98735b`.
  The public package was pulled anonymously and the container passed its
  health check.
- Unraid uses the prebuilt GHCR image through `unraid/writtensound.xml`; the
  standard image is roughly 3.3 GB locally. Models remain in `/data`.
- No image tarball is attached to the GitHub release. Public GHCR storage is
  currently free, and the source repository stays small.

## NeMo runtime fixes (2026-10-25)

- NeMo writes a plain-text shader-cache warning to stderr before its JSON
  result. Its `words` array also emits one JSON object per line. The previous
  line-based JSON fallback could return the last word object instead of the
  transcript, which surfaced as `nemo-speech produced no transcript segments`
  even when the CPU transcription succeeded. The parser now scans for the first
  complete transcript-shaped object.
- The runtime user is created without a home directory, so Vulkan logged
  `Failed to create /home/writtensound for shader cache (Permission denied)`.
  The entrypoint now creates and owns the passwd home before dropping
  privileges; the warning is non-fatal but disabling the cache costs speed.
- Parakeet TDT v3 is offline/full-utterance. On the 6 GB Arc A380 a real file
  requested a 7.27 GB Vulkan graph and failed with `ErrorOutOfDeviceMemory`.
  Use whisper.cpp for long files on the A380, or chunk NeMo input; do not
  describe Parakeet full-context Vulkan as A380-compatible without chunking.
- `v0.1.1` published `latest`, `0.1.1`, `0.1`, and `sha-e75f3a7` at digest
  `sha256:a844d0d2425ca4f1fdc358b298ffab2eb45a34d6023ee84e4561a13d0ff45f7a`;
  the release has no assets. The image reports version `0.1.1`, starts as
  UID/GID 99/100, creates `/home/writtensound`, reaches `healthy`, and the
  parser regression was exercised inside the image. The source gates pass with
  34 non-GPU tests.

## 0.2 planning — timestamps, VTT, and Impeccable (2026-10-25, historical)

This section records the pre-implementation state and plan; the implementation
is in the next section.

- **The original scope** combined transcript timestamp correctness, the
  timed-text output format, and the UI overhaul.
- **The one-cue VTT is a NeMo data-loss bug.** NeMo returns top-level `words[]`
  with per-token times, but `_segments_from_payload()` only reads
  `segments`/`transcription` and falls back to one `0 → duration` segment.
  `_group_cues()` then takes its word-less path; one speaker adds only the
  `Speaker 1:` prefix. Word timestamps must be preserved before cue rendering
  can improve.
- **Timed text is planned, not normative.** The proposed `timed_txt` format is
  `[HH:MM:SS.mmm --> HH:MM:SS.mmm] text`, one line per cue, reusing the SRT/VTT
  grouping when words exist and falling back to ASR segments otherwise. The
  exact API/UI contract must be added to docs 00/03/04/05 in the implementing
  commit; until then docs 00–07 remain authoritative and the feature is absent.
- **Single-speaker labels are noise.** Current VTT prefixes every cue with
  `Speaker 1:` whenever diarization is enabled, even with one speaker. The
  planned timed/VTT behavior suppresses the prefix for one speaker or makes it
  optional; this is a renderer/UI decision, not a diarization change.
- **Interval timestamps currently use segment starts only.** docs/04 § 4.2
  describes interval marks at the first word past each interval; a one-segment
  file therefore gets one mark. This gap is part of the timestamp work, not a
  reason to reinterpret the format.
- **Impeccable is now registered with OpenCode.** The skill is 4.7.0 / engine
  v0.1.16. It was originally installed under `.github/skills/impeccable`;
  OpenCode 2.1.3 does not scan that path. It was re-installed globally with
  `--providers=opencode --scope=global --no-hooks` into
  `~/.config/opencode/skills/impeccable`, and OpenCode now lists it in the
  `skill` tool. The CLI runs (`impeccable context` reports no `PRODUCT.md`).
  Run its init flow before UI work; keep the roughly 20 MB engine binary out of
  Git.
- **Superseded by the implementation section below.** The planning order was
  spec amendment, NeMo word preservation/cue fix, timed-text renderer, then the
  Impeccable UI overhaul.

## 0.2 implementation — timestamps, timed text, and selector (2026-10-25)

- **NeMo words are preserved.** The adapter reads top-level `words[]` when
  `segments[]` is absent, keeps each word's `start`/`end`/`confidence`, and
  uses the payload text for the segment. It returns a word-less segment only
  when the payload has no words at all.
- **Caption grouping is shared.** SRT, VTT, and timed text use the same cue
  grouping: speaker changes, >1 s gaps, 12 words, or 5 s spans. Without words,
  ASR segments are the cues. The runner writes a pipeline-log line when caption
  output falls back to segment boundaries because no word timings exist.
- **Timed text is normative.** `timed_txt` (`{base}.timed.txt`) is one line per
  cue: `[HH:MM:SS.mmm --> HH:MM:SS.mmm] text`; no indices, no `WEBVTT` header,
  no blank line between cues. Settings, API, manifest, download, and UI all
  include it. `timestamps_mode=none` rejects it with the other captions.
- **Single-speaker labels are suppressed.** HTML, TXT, SRT, VTT, and timed text
  omit `Speaker 1`/`Speaker 1:` when the transcript has one speaker.
  Multi-speaker files retain `Speaker N:` prefixes.
- **Interval markers split at words.** A long segment is cut into paragraphs at
  the first word at or after each interval threshold, so a one-segment NeMo
  transcript gets multiple `[hh:mm:ss]` markers. Turn starts still get markers.
- **The Add jobs selector is back.** The shared settings form has a visible
  `Engine and model` select grouped by engine; the read-only queue strip and
  hidden `asr_model_id` are removed. The htmx dependent-field refresh is kept.
- **Verification:** `ruff`, `mypy`, and 40 non-GPU tests pass. The UI overhaul
  commit is `a986c26`; the `v0.2.0` tag includes both the UI and transcript
  changes.
- **Release:** tag `v0.2.0` built in 9m (Actions run 40233714913) and published
  `latest`, `0.2.0`, `0.2`, and `sha-f0a1126` at digest
  `sha256:d4f56f8a0a4cd5a8ddaf79a286f6a4a62d625f32d908f164b1d2678f983dce43`.
  Registry manifests for all four tags match. The source app smoke test on port
  8098 reported `0.2.0` with the model selector and Timed text option. Docker's
  pull path on this Windows host failed with `tls: bad record MAC`; `crane`
  fetched the same digest and `docker load` completed the container check. The
  app process ran as UID/GID 99/100, the healthcheck reached `healthy`,
  `/healthz` returned 200, and the queue page reported `0.2.0` with the model
  selector and Timed text option.

## Post-0.2 fixes - whisper.cpp tokens and Qwen chat template (2026-10-25)

- **whisper.cpp special tokens are engine protocol, not transcript text.**
  `--output-json-full` emits timestamp/control tokens (`[_TT_*]`, `[_BEG_]`)
  in `tokens[]`; treating every token as a `Word` leaked them into captions,
  timed text, and interval-split HTML. The adapter now drops token ids at or
  above EOT (50256) and any `[_...]` marker before building words. `merge()`
  repeats the cleanup defensively so checkpoints written before this fix also
  render cleanly.
- **whisper.cpp tokens are subword pieces.** `tokens[]` is not a word list.
  Adjacent pieces are merged at leading-space boundaries, producing
  `unimaginable`, `friendliness`, `Don't`, and `foot-washing` instead of token
  fragments.
- **`--max-len 1` is destructive and was removed.** In whisper.cpp v1.9.5,
  `--output-json-full` already enables token timestamps; `--max-len` only wraps
  the decoder output into one-token segments, which turns paragraph output into
  token-per-line noise. Paragraphs continue to follow the engine's real segment
  boundaries, not inferred speaker pauses.
- **Qwen3's chat template is a separate snapshot file.** The Qwen repos ship
  `chat_template.jinja` outside `tokenizer_config.json`; the v0.2 snapshot
  allow-list omitted `*.jinja`, so the processor loaded with no template and
  `apply_transcription_request` failed. The catalog now fetches `*.jinja` for
  both ASR sizes and the aligner. The ASR adapter also repairs an older local
  snapshot by fetching the single missing template file, and raises an
  actionable error if that is impossible.
- **The ForcedAligner uses the same repair.** The shared loader helper is used
  by both the ASR and aligner paths. The runner passes the aligner repo and the
  effective HF token, and surfaces repair progress in `logs/pipeline.log`, so an
  older aligner snapshot no longer requires a manual remove/re-download.
- **Verification:** `ruff check`, `mypy backend`, and 53 non-GPU tests pass.
- **Release:** included in `v0.3.0`; see the release section below.

## 0.3.0 release (2026-10-25)

- **Version:** `pyproject.toml`, `backend/__init__.py`, and `uv.lock` report
  `0.3.0`.
- **Contents:** whisper.cpp special-token filtering and subword merging, Qwen3
  `chat_template.jinja` repair for ASR and the ForcedAligner, and removal of the
  destructive `--max-len 1` flag. `v0.2.0` remains published and unchanged.
- **Verification:** `uv run ruff check .`, `uv run mypy backend`, and
  `uv run pytest -m "not gpu"` (53 tests) pass on the tagged source tree.
- **Image:** tag workflow run 40278271970 built in 8m41s and published
  `latest`, `0.3.0`, `0.3`, and `sha-9b935e1` at digest
  `sha256:e64334c7f7e4745c1ba086a149881d094fefb3f0702c00104322bda4203f7efc`.
  Registry manifests for all four tags resolve to that digest. The container
  started as UID/GID 99/100, reached `healthy`, returned HTTP 200 from
  `/healthz`, and served the queue page reporting `0.3.0`.
- **GitHub release:** https://github.com/cozykettle/writtensound/releases/tag/v0.3.0
