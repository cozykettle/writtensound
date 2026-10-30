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

## Post-0.3 fixes - Qwen3 runtime and image dependencies (2026-10-27)

- **The released 0.3.0 image could not run Qwen3 at all.** The first job failed
  with `[engine_crash] Qwen3-ASR failed on window 1: Could not load this
  library: .../torchcodec/libtorchcodec_image.so`. pyannote pulls
  `torchcodec 0.17`, which requires torch 2.11, while the image pins torch
  2.9.1. `torchcodec==0.9.1` is the last release built for torch 2.9 and is now
  an explicit engines pin. The lock resolves torch/torchaudio `2.9.1+cpu`,
  torchcodec `0.9.1`, transformers `5.20.2`, and numpy `2.5.4`.
- **torchcodec also needs an OS package, not just a Python pin.** After the
  downgrade the probe still failed, and the FFmpeg-6 extension reported
  `libpython3.12.so.1.0: cannot open shared object file`. Ubuntu 24.04's
  `python3-venv` does not install `libpython3.12`; both app layers now apt
  install it. The build-time probe failed loudly until this was fixed, which is
  why the broken dependency set did not reach a tag.
- **Qwen audio must be PCM, not a path.** `apply_transcription_request`
  accepts a path, but transformers then routes it through torchcodec's audio
  decoder, which loads the same fragile native extension (and on the first
  rebuilt image requested `libnvrtc.so.13`). `backend/engines/qwen3/audio.py`
  reads 16 kHz mono s16le `prep.wav` windows into float32 numpy arrays, and both
  ASR and the aligner pass those arrays directly. Decoding is shared with the
  aligner so both engines see identical samples.
- **Generated tokens include the prompt.** `generate()` returns prompt plus
  completion; decoding the whole tensor made the processor parse chat-template
  tokens as transcript. Both paths now slice
  `output[:, inputs["input_ids"].shape[1]:]` before `decode`.
- **The pinned transformers 5.20.2 aligner API is different from the one
  sketched during planning.** The aligner uses
  `prepare_forced_aligner_inputs(audio=..., transcript=..., language=...)` and
  `decode_forced_alignment(logits=..., input_ids=..., word_lists=...,
  timestamp_token_id=...)`. Codes and full names are normalized; `auto` and
  unsupported languages log and skip instead of crashing. Segments longer than
  300 s are split into `ceil(span / 300)` windows so neither the model nor the
  audio tensor is fed an oversized input.
- **Dtype follows the device, not the checkpoint.** The Qwen snapshots declare
  bf16. On CPU that fails with `Input type (float) and bias type
  (c10::BFloat16) should be the same`; on Turing (sm_75) bf16 is not
  executable. `backend/engines/qwen3/dtype.py` selects float32 on CPU, bf16 on
  CUDA when `torch.cuda.is_bf16_supported()`, and float16 otherwise, then casts
  only floating inputs so ids and masks stay integer.
- **The catalog language list was wrong.** The old `qwen3` list mixed
  unsupported codes (bn, ur, ta, te, mr, gu, he, uk) with supported ones. It
  now matches the processor's 30 codes exactly: ar, yue, zh, cs, da, nl, en,
  fil, fi, fr, de, el, hi, hu, id, it, ja, ko, mk, ms, fa, pl, pt, ro, ru, es,
  sv, th, tr, vi. The aligner list stays at its 11 languages.
- **The build now proves the runtime.** `backend/engines/qwen3/runtime.py`
  checks the pinned versions, imports the Qwen stack plus nagisa/soynlp and
  pyannote.audio, and decodes a generated WAV through torchcodec. Both
  `app-standard` and `app-cuda` run it, so a broken Qwen dependency set fails
  the image build instead of the first user job. The `_series` check compares
  only major.minor, so `2.9.1+cpu` and `2.9.1+cu128` satisfy `(2, 9)`.
- **apt downloads were also unreliable.** Several image builds died with
  `Hash Sum mismatch` on Ubuntu mirrors. All eight `apt-get` invocations now
  pass `-o Acquire::http::Pipeline-Depth=0 -o Acquire::http::No-Cache=true` in
  addition to the existing retries.
- **Verification:** `ruff check`, `mypy backend` (46 files), and 68 non-GPU
  tests pass. The rebuilt `final-standard` image is 3.54 GB and prints `Qwen3
  runtime probe ok: transformers 5.20.2, torch 2.9.1+cpu, torchaudio 2.9.1+cpu,
  torchcodec 0.9.1, nagisa, soynlp, pyannote.audio, torchcodec WAV decode`. A
  real CPU smoke run inside that image transcribed a 6.69 s clip with
  Qwen3-ASR-0.6B (exact text, `language='en'`) and aligned 16 words with
  Qwen3-ForcedAligner-0.6B.
- **Compatibility:** `v0.3.0` and `v0.2.0` remain published and unchanged;
  `v0.3.1` is additive.
- **Release:** included in `v0.3.1`; see the release section below.

## 0.3.1 release (2026-10-27)

- **Version:** `pyproject.toml`, `backend/__init__.py`, and `uv.lock` report
  `0.3.1`.
- **Contents:** torchcodec 0.9.1, transformers 5.20.2, numpy 2, nagisa/soynlp,
  the `libpython3.12` OS package, float32 PCM windows, generated-token slicing,
  the 5.20.2 aligner API with 300 s windowing, device-driven dtype, the
  corrected 30-code Qwen language list, and the build-time Qwen runtime probe.
  `v0.3.0` and `v0.2.0` remain published and unchanged.
- **Verification:** `uv run ruff check .`, `uv run mypy backend`, and
  `uv run pytest -m "not gpu"` (68 tests) pass on the tagged source tree.
- **Image:** tag workflow run 40458105813 built in 8m56s and published
  `latest`, `0.3.1`, `0.3`, and `sha-814eb59` at digest
  `sha256:ed651b4de76e029a468dfcaf8e9cdc5eea20a282fc174f7d144f82f7331d9800`.
  Registry manifests for all four tags resolve to that digest. The published
  image's probe reports `transformers 5.20.2`, `torch 2.9.1+cpu`,
  `torchaudio 2.9.1+cpu`, `torchcodec 0.9.1`, nagisa, soynlp, pyannote.audio,
  and a passing torchcodec WAV decode. A real CPU smoke run inside the
  published image transcribed the 6.69 s clip exactly (`language='en'`) and
  aligned 16 words. The container started as UID 99, reached `healthy`,
  returned HTTP 200 from `/healthz`, and served the queue page reporting
  `0.3.1`. Docker's pull path on this Windows host again failed with `tls: bad
  record MAC`; `crane` fetched the digest and `docker load` completed the
  container check.
- **GitHub release:** https://github.com/cozykettle/writtensound/releases/tag/v0.3.1

## 0.4.0 plan and root causes (2026-10-28)

`0.4.0` is a fix release for eight user reports. The durable change list lives
in `docs/09-deployment-handoff.md` §5; this section records the decisions and
the evidence behind them. The CUDA target stays parked and is not built or
validated for this release.

- **Queue updates stopped behind a running job.** The server publishes a
  `queue` event from `JobRunner.enqueue()` and `_publish_queue()`, but
  `backend/static/app.js` only listened for `job`, `snapshot`, `hardware`,
  `model`, `engine`, and `toast`. `job` is emitted when a job starts, changes
  progress, or finishes, so newly enqueued `pending` jobs never reached the
  client until their turn. The client now consumes `queue`.
- **The `queue` payload was minimal and unused.** It carried
  `[{job_id,status,progress}]`, which cannot render a row and does not match the
  `JobOut.id` field used elsewhere. It now carries full `JobOut` objects for
  `pending` and `running` jobs. No extra request per row.
- **Queue order is insertion order, newest first.** Sorting by status rank put
  `running` above newer `pending` jobs and used `created_at`, which can collide
  inside one batch. `JobOut` now includes `seq` (the SQLite rowid), and the
  client sorts by `seq` descending. This satisfies "newest on top" and
  "processing jobs below new ones" with one rule. `created_at` remains the
  human timestamp; `seq` is never user-editable.
- **faster-whisper 1.2.1 signature.** The installed 1.2.1 source declares
  `temperature: Union[float, List[float], Tuple[float, ...]]` and has no
  `progress_hook` parameter. The frozen kwargs now pass `temperature: 0.0`;
  the adapter's existing hook detection keeps the segment-count heuristic and
  writes a log line using the pinned version constant, not a hard-coded string.
- **HTML date overlap is CSS, not data.** The `.meta` grid uses
  `minmax(140px, 1fr)` and each cell is a flex row. The `dd` child keeps its
  min-content width, so a 25-character ISO date overflows into the duration
  cell. `min-width: 0` on the cell and `overflow-wrap: anywhere` on the value
  let the date wrap inside its own column.
- **HTML duration `00:00:00` is missing metadata.** `probe_duration(payload)`
  only reads `payload["result"]["duration"]`; whisper.cpp's full JSON does not
  provide it. The adapter now falls back to `probe_media(audio_path)` and keeps
  the payload value when present.
- **Shrinking whisper.cpp lines are engine-driven.** The attached repro has
  early paragraphs of ~20 words and late paragraphs of 2–5 words because the
  engine emits progressively smaller segments. HTML now merges consecutive
  paragraphs inside a turn up to ~80 words / 560 characters, breaking on a
  timestamp, a pause marker, or a speaker turn. TXT/SRT/VTT/timed/JSON keep the
  original boundaries so captions do not become long cues.
- **NeMo late-word timestamps were mangled.** `_times()` divided by 1000 when
  both endpoints were above 1000, which is true for any word after 16:40 in a
  file. The heuristic now only treats values above 100000 as milliseconds, so
  words keep real seconds.
- **Parakeet dropped sentences are an upstream limitation.** NeMo-Speech.cpp
  has no ASR precision flag; the GGUF quantization is the quality choice. The
  runtime already splits long offline input at quiet points when it exceeds the
  encoder's positional budget, and Parakeet TDT is offline-only. NVIDIA's model
  card documents full attention to about 24 minutes (local attention to about 3
  hours) and the model discussion board reports occasional dropped short
  utterances. WrittenSound logs the limitation and documents whisper.cpp as the
  long-form alternative; it does not switch engines or silently degrade.

## 0.4.0 release (2026-10-29)

- **Version:** `pyproject.toml`, `backend/__init__.py`, and `uv.lock` report
  `0.4.0`.
- **Contents:** live `queue` SSE with full `JobOut` rows and `seq` ordering,
  faster-whisper `temperature=0.0`, HTML meta wrapping and duration fallback,
  whisper.cpp paragraph reflow (80 words / 560 characters), the NeMo
  seconds/milliseconds fix, the local SVG favicon, and the documented Parakeet
  dropped-utterance limitation. CUDA stays parked: no CUDA image was built or
  validated. `v0.3.1` and earlier remain published and unchanged.
- **Verification:** `uv run ruff check .`, `uv run mypy backend`, and
  `uv run pytest -m "not gpu"` (77 tests) pass on the tagged source tree.
  A source-server E2E run on port 8091 queued six jobs and received 12 `queue`
  events; the largest snapshot carried six full rows in newest-first order.
- **Local image:** `writtensound:standard` built to
  `sha256:ab4268ad9694ffec8c69d7a58ac09bdc1f56fc1496203e6a1754892b1a728ad2`
  (3.54 GB) with the Qwen runtime probe passing. The container ran the app as
  UID 99, reached `healthy`, returned HTTP 200 from `/healthz`, served the
  0.4.0 page and favicon, streamed a live newest-first queue with full rows,
  and produced a 12 s demo transcript with `00:00:12` and the wrapping meta
  grid.
- **Image:** tag workflow run 40985079706 built in 9m45s and published
  `latest`, `0.4.0`, `0.4`, and `sha-e6ae998` at digest
  `sha256:c4e1dd4fc6763e49f287961da6aec6a8884754062c4d54c0fcd33ab3dfd6b2eb`.
  Registry manifests for all four tags resolve to that digest by `crane`.
  `docker pull` succeeded on this host (no TLS fallback needed). The published
  image (3,534,988,499 bytes) re-ran the container smoke test successfully;
  the app process runs as UID 99, the container reaches `healthy`, `/healthz`
  returns HTTP 200, and both pinned engine binaries answer `--help`.
- **GitHub release:** https://github.com/cozykettle/writtensound/releases/tag/v0.4.0

## 0.5.0 plan and root causes (2026-10-30)

`0.5.0` is a follow-up fix release for three reports after `0.4.0`. The durable
change list lives in `docs/09-deployment-handoff.md` section 5 (rows 23-25);
CUDA stays parked.

- **faster-whisper failed with `metadata_errors`.** The `0.4.0` call-shape fix
  reached transcription, then failed inside faster-whisper's audio decode.
  faster-whisper 1.2.1 `decode_audio()` calls
  `av.open(input_file, mode="r", metadata_errors="ignore")`. PyAV 19 removed the
  `metadata_errors` keyword (`av 19.0.2` raises `open() got an unexpected
  keyword argument 'metadata_errors'`), and the engines extra let uv resolve
  `av 19.0.2` because faster-whisper only requires `av>=11`. PyAV 16, 17, and 18
  still accept the keyword; 19 does not. The fix pins `av==18.0.0`, adds a
  build-time/runtime probe, and keeps faster-whisper at 1.2.1.
- **Queue missed a fast failure.** New rows were created only from SSE events,
  and `/api/events` built its snapshot before subscribing to the hub, so events
  published in that gap were lost; a job that failed while the stream was
  reconnecting therefore never reached the client and needed a refresh.
  `POST /api/jobs` now returns the full new `JobOut` rows, the client upserts
  them immediately, the SSE snapshot is built after subscription, and the client
  re-fetches any active row that disappears from a `queue` payload without a
  terminal `job` event.
- **Downloads were full filenames.** The detail row listed one link per complete
  filename. It now groups outputs by base name and shows one clear format
  button per file (`HTML`, `TXT`, `VTT`, `SRT`, timed text, `JSON`), with the
  full name and size in the tooltip.
- **NeMo speaker labels were missing from HTML/TXT.** NeMo returns the whole
  file as one segment with top-level words, so `segment.speaker` is only the
  majority label (usually one speaker for the whole interview). Timed text,
  SRT, and VTT already split cues at word-level speaker changes, which is why
  only those formats showed `Speaker N`. TXT/HTML grouped turns by
  `segment.speaker`, collapsing the file into a single unlabeled turn. The turn
  builder now splits each segment into speaker-homogeneous word runs before
  grouping, so all formats share the same boundaries.
- **Test media.** A 30-minute MP3 supplied by the user at the repository root is
  used for local engine tests only. It stays untracked; trimmed copies go to the
  temporary directory. E2E coverage uses the real MP3 with faster-whisper on CPU
  in the built image, plus a trimmed clip for the full output/UI check.
- **Local verification (2026-10-30).** `uv run ruff check .`, `uv run mypy
  backend`, and `uv run pytest -m "not gpu"` pass with 84 tests. The local
  `final-standard` image (3.54 GB) passes both build-time probes; its
  faster-whisper probe reports `av 18.0.0`. Inside the running container,
  `faster_whisper.audio.decode_audio` decoded the full supplied MP3 (25,349,193
  samples), which is the exact call that raised `metadata_errors` in `0.4.0`. A
  90 s trimmed clip transcribed with Whisper small on CPU in 14.43 s of worker
  time and produced `clip.html`, `clip.txt`, `clip.srt`, `clip.vtt`,
  `clip.timed.txt`, and `ir.json`; all six download as attachments. A
  NeMo-shaped one-segment transcript with two word-level speakers rendered
  `Speaker 1`/`Speaker 2` blocks in TXT and both HTML headings. The container is
  healthy, `/healthz` returns 200, and the page reports `0.5.0`.

- **Release verification (2026-10-30).** The `v0.5.0` tag workflow built in
  8m31s and published `latest`, `0.5.0`, `0.5`, and `sha-21d59a6` at digest
  `sha256:8222ab737895f5b0af778f6ab26d420d87c67cca0cfa3817475f87446026692b`.
  `docker buildx imagetools inspect` resolves all four tags to that digest, and
  `docker pull` succeeded on this host. The published image starts as UID 99,
  reaches `healthy`, returns HTTP 200 from `/healthz`, serves `0.5.0`, and its
  faster-whisper probe reports `av 18.0.0`. Inside the published image,
  `decode_audio` decoded the full supplied MP3 (25,349,193 samples), and a demo
  job produced `full.html`, `full.txt`, `full.srt`, `full.vtt`,
  `full.timed.txt`, and `ir.json`. Both pinned engine binaries answer `--help`.
  GitHub release: https://github.com/cozykettle/writtensound/releases/tag/v0.5.0
