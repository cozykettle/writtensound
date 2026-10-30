# WrittenSound

> [!WARNING]
> **⚠️ AI SLOP ALERT ⚠️** At absolutely no point did a human write a single line of code or documentation for this entire application. A human spent roughly two whole minutes of strenuous brain power sketching out an idea for a program they sorta wanted to try for a project, then offloaded every remaining thought to the machines and called it a day. Not even *good* machines, mind you: the human was too lazy to pay for an actual flagship model, so this was vomited out by promotional-tier GLM 5.3 Flash and DeepSeek 4.1 Flash instead. This repository is the result. Abandon all hope, ye who enter here, for all that shall meet you is slop to wade through.

Self-hosted, local-first speech-to-text with a focused transcript view, a serial job queue, and a multi-engine model catalog. WrittenSound never sends audio to a hosted service: uploads, inference, model files, and job outputs stay on the machine running the app.

The project is currently a **working preview**. The demo engine, queue, upload flow, SSE updates, settings, renderers, and job output/download paths are implemented and covered by a hermetic test suite. Real-engine adapters and Docker engine builds exist, but the Phase A runtime verification described in [`docs/02`](docs/02-engines-and-models.md) is still required before they can be called production-ready.

## What works today

- Server-rendered queue with drag-and-drop or file-picker uploads, folder-relative paths, cancellation, retry, per-job settings, and output downloads.
- A deterministic built-in **Demo** engine for a zero-download end-to-end run.
- HTML, TXT, SRT, VTT, timed-text, and merged JSON renderers, including speaker labels, interval and turn timestamps, word-level timing, and pause markers.
- Engine catalog and model download UI for faster-whisper, whisper.cpp, NeMo-Speech.cpp, Qwen3-ASR, and pyannote.
- SQLite job/config storage, a single serial worker, SSE queue events, hardware probing, retention cleanup, and optional HTTP Basic auth.
- A standard Docker image target with CPU + Vulkan engine builds and a CUDA target with CUDA + Vulkan builds.

## Quick start (demo, no model download)

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/). `ffmpeg`/`ffprobe` are recommended; WAV uploads work without them and show a warning.

```bash
uv sync
uv run writtensound --host 127.0.0.1 --port 8080
```

Open <http://127.0.0.1:8080>, select a **Demo** model in the settings form, add a WAV file, and queue it. The demo engine produces a synthetic transcript and the default HTML, TXT, SRT, VTT, and JSON outputs; select **Timed text** for the extra caption-style format.

For the full local engine stack:

```bash
uv sync --extra engines
```

Real engine binaries (`whisper-cli`, `nemo-speech`) must be installed or pointed to with `WS_WHISPER_CLI` and `WS_NEMO_SPEECH`. Model weights are never baked into the repository; download them from the **Engines** page.

## Docker

### Prebuilt image (recommended)

The standard CPU + Vulkan image is built by GitHub Actions and published to the public GitHub Container Registry:

```bash
docker pull ghcr.io/cozykettle/writtensound:latest
```

Use this image with `unraid/writtensound.xml` or any Docker host with `/dev/dri` for Intel/AMD Vulkan. The package is public, so Unraid does not need a registry login. Container tags such as `0.3.1` are published alongside `latest`; pin a version if you do not want automatic updates.

### Build from source

Copy the example environment file and set a data path. The default data path is `./data`; on Unraid use an appdata or share path.

```bash
cp .env.example .env
docker compose up -d --build
```

Open <http://localhost:8080>. The standard service builds the CPU + Vulkan image and passes through `/dev/dri` for Intel/AMD Vulkan on a Linux host. For a CPU-only host, or for Docker Desktop on Windows (which cannot pass a Linux `/dev/dri` render node into the container), use the CPU override:

```bash
docker compose -f docker-compose.yml -f docker-compose.cpu.yml up -d --build
```

The app still runs, but Vulkan stays unavailable. GPU/Vulkan acceleration requires a Linux/Unraid host with `/dev/dri`; it is not available through Docker Desktop on Windows.

CUDA hosts need the NVIDIA Container Toolkit. Docker Desktop on Windows can also expose an NVIDIA GPU through its WSL2 backend; verify that Docker sees it with:

```bash
docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi
```

Stop the standard service first because both services use the same container name:

```bash
docker compose down
docker compose --profile cuda up -d --build writtensound-cuda
```

The CUDA image contains both CUDA and Vulkan engine builds. The standard image does not expose CUDA devices. The CUDA image is not prebuilt in this preview, so this command compiles it locally and may take a long time. On a Windows laptop with a 4 GB GPU such as a T600, use smaller or quantized models; large models and simultaneous ASR plus diarization may exhaust VRAM.

For a local build targeting one GPU, set that GPU's CUDA architecture to avoid compiling six fatbin variants. A T600 is `75`:

```powershell
$env:CUDA_ARCHITECTURES = "75"
docker compose --profile cuda up -d --build writtensound-cuda
```

The resulting image only contains kernels for the selected architecture; use the default list for a redistributable release image.

### Image targets

The Dockerfile builds two explicit targets from one file:

| target | image | use |
|---|---|---|
| `final-standard` | `writtensound:standard` | CPU + Vulkan (Intel Arc A380, AMD, CPU-only) |
| `final-cuda` | `writtensound:cuda` | NVIDIA CUDA + Vulkan |

```bash
docker build --target final-standard -t writtensound:standard .
docker build --target final-cuda -t writtensound:cuda .
```

Engine source tags are pinned in [`docker-build-args.env`](docker-build-args.env). Models are downloaded at runtime into `/data/models`.

### Verification status

As of 2026-10-27, the `final-standard` image reports `torch 2.9.1+cpu`, and both pinned engine binaries answer `--help`. The local `0.1.1` container started as UID/GID 99/100, reached `healthy`, returned HTTP 200 from `/healthz`, and the NeMo parser regression was exercised inside the image. The `0.2.0` source tree preserves NeMo top-level word timings, groups captions into multiple cues, adds timed-text output, and splits interval markers at word boundaries; `uv run ruff check .`, `uv run mypy backend`, and the 40 non-GPU tests pass. The `0.3.0` source tree filters whisper.cpp special tokens, merges subword pieces into words, and repairs Qwen3 `chat_template.jinja` for both ASR and the ForcedAligner; the same gates pass with 53 non-GPU tests.

The `0.3.1` source tree fixes the Qwen3 runtime failure in the released image. It pins `torchcodec 0.9.1`, `transformers 5.20.2`, `numpy 2`, and the Japanese/Korean splitters, installs `libpython3.12` for torchcodec's FFmpeg-6 extension, passes 16 kHz float32 PCM windows instead of file paths, slices generated tokens past the prompt, updates the aligner to the 5.20.2 API with segment windowing, and chooses float32 on CPU / bf16 or float16 on CUDA. The source gates pass with 68 non-GPU tests. A local `final-standard` image (3.54 GB) runs the build-time Qwen probe (`transformers 5.20.2`, `torch 2.9.1+cpu`, `torchcodec 0.9.1`, nagisa, soynlp, pyannote.audio, torchcodec WAV decode), and a real CPU smoke run inside the image transcribed a 6.69 s clip with Qwen3-ASR-0.6B (exact text) and aligned 16 words with Qwen3-ForcedAligner-0.6B.

The `0.4.0` source tree fixes live queue updates and ordering, faster-whisper 1.2.1's `temperature` keyword, HTML metadata date overlap, whisper.cpp's missing duration and short-paragraph reflow, and late NeMo word timestamps; Parakeet's dropped-utterance behaviour is documented as an upstream limitation. It adds a local SVG favicon and leaves CUDA parked, with no CUDA image built or validated. The source gates pass with 77 non-GPU tests, and the local `final-standard` image (3.54 GB) passes the build-time Qwen probe and a container smoke test covering the 0.4.0 version string, favicon, live newest-first queue with full rows, a real `00:00:12` duration, and the wrapping meta grid.

The `0.5.0` source tree pins PyAV 18 for faster-whisper 1.2.1 (PyAV 19 removed the `metadata_errors` keyword that `decode_audio()` passes to `av.open`), adds a build-time faster-whisper decode probe to both app layers, returns full `JobOut` rows from `POST /api/jobs` and reconciles missed SSE events so a fast failure appears without a refresh, groups outputs under the base file name with one clear download button per format, and restores NeMo speaker headings in HTML/TXT by splitting turns at word-level speaker changes. The source gates pass with 84 non-GPU tests. The local `final-standard` image passes both runtime probes, decodes the supplied 26-minute MP3 through faster-whisper's `decode_audio`, and transcribes a 90 s clip with Whisper small on CPU, producing all six output formats.

The `v0.2.0` tag workflow published the standard image as `ghcr.io/cozykettle/writtensound:latest`, `0.2.0`, `0.2`, and `sha-f0a1126` (digest `sha256:d4f56f8a0a4cd5a8ddaf79a286f6a4a62d625f32d908f164b1d2678f983dce43`). Registry manifests for all four tags match. The image started as UID/GID 99/100, reached `healthy`, returned HTTP 200 from `/healthz`, and served `0.2.0` with the restored model selector and Timed text option.

The `v0.3.0` tag workflow built in 8m41s and published `latest`, `0.3.0`, `0.3`, and `sha-9b935e1` (digest `sha256:e64334c7f7e4745c1ba086a149881d094fefb3f0702c00104322bda4203f7efc`). Registry manifests for all four tags match. The image started as UID/GID 99/100, reached `healthy`, returned HTTP 200 from `/healthz`, and served the queue page reporting `0.3.0`.

The `v0.3.1` tag workflow built in 8m56s and published `latest`, `0.3.1`, `0.3`, and `sha-814eb59` (digest `sha256:ed651b4de76e029a468dfcaf8e9cdc5eea20a282fc174f7d144f82f7331d9800`). Registry manifests for all four tags match. The published image's Qwen probe passed, and a real CPU smoke run inside it transcribed a 6.69 s clip with Qwen3-ASR-0.6B (exact text, `language='en'`) and aligned 16 words with Qwen3-ForcedAligner-0.6B. The container started as UID 99, reached `healthy`, returned HTTP 200 from `/healthz`, and served the queue page reporting `0.3.1`.

The `v0.4.0` tag workflow built in 9m45s and published `latest`, `0.4.0`, `0.4`, and `sha-e6ae998` (digest `sha256:c4e1dd4fc6763e49f287961da6aec6a8884754062c4d54c0fcd33ab3dfd6b2eb`). Registry manifests for all four tags match. The published image re-ran the container smoke test: the app process runs as UID 99, the container reaches `healthy`, `/healthz` returns HTTP 200, the page reports 0.4.0 with the SVG favicon, the `queue` event streams full rows newest-first, and a demo transcript renders with a real `00:00:12` duration and wrapping metadata. CUDA is parked for this release.

The `v0.5.0` tag workflow built in 8m31s and published `latest`, `0.5.0`, `0.5`, and `sha-21d59a6` (digest `sha256:8222ab737895f5b0af778f6ab26d420d87c67cca0cfa3817475f87446026692b`). Registry manifests for all four tags match. The published image runs as UID 99, reaches `healthy`, serves `0.5.0`, passes the faster-whisper probe with `av 18.0.0`, decodes the full supplied MP3 through `decode_audio`, and renders all six output formats for a demo job. CUDA remains parked.

The GitHub Actions build published `v0.1.1` to GHCR as `ghcr.io/cozykettle/writtensound:latest` and `0.1.1` (digest `sha256:a844d0d2425ca4f1fdc358b298ffab2eb45a34d6023ee84e4561a13d0ff45f7a`). This fixes parsing NeMo output when a shader-cache warning precedes the JSON and when the `words` array is line-delimited; the entrypoint also creates the runtime user's home so Vulkan can use its shader cache. A real CPU Parakeet TDT v3 run produced transcript text. Full-context Parakeet on the 6 GB Arc A380 still fails with `ErrorOutOfDeviceMemory` (a 7.27 GB Vulkan graph), so use whisper.cpp for long files there. A CUDA target build was attempted; its Python layer installed and asserted `torch 2.9.1+cu128`, but the long nvcc compilation of the CUDA+Vulkan engine stages was stopped on this CPU-only host, so no `writtensound:cuda` image is claimed. See [`docs/process-notes.md`](docs/process-notes.md) for the exact record.

### Security

`APP_PASSWORD` is unset by default, which leaves the app open to anyone who can reach the port. Set it before exposing the container beyond a trusted network. The health endpoint remains unauthenticated so container healthchecks work.

## Configuration

All settings are optional. Docker variables can be placed in `.env`; local runs can use the same environment variables.

| Variable | Default | Meaning |
|---|---|---|
| `WS_DATA_DIR` | `./data` | Root for config, jobs, staging, models, and the SQLite database |
| `WS_PORT` | `8080` | HTTP port and default console port |
| `APP_PASSWORD` | empty | Optional HTTP Basic password for all routes except health |
| `HF_TOKEN` | empty | Hugging Face token fallback for gated model downloads |
| `HF_HOME` | `<data>/models/hf-cache` | Hugging Face cache location |
| `WS_LOG_LEVEL` | `INFO` | Python/uvicorn log level |
| `WS_MAX_UPLOAD_BYTES` | `17179869184` (16 GiB) | Upload size ceiling |
| `WS_ENABLE_DEMO_ENGINE` | `1` | Set to `0` to hide the demo engine |
| `WS_WHISPER_CLI` | empty | Override the `whisper-cli` executable path |
| `WS_NEMO_SPEECH` | empty | Override the `nemo-speech` executable path |
| `WS_DISABLE_HW_PROBE` | `0` | Skip the startup hardware probe (tests/diagnostics) |

The Hugging Face token can also be saved from the Settings page. A token saved there takes precedence over `HF_TOKEN`.

## Development

```bash
uv sync --extra dev
uv run ruff check .
uv run mypy backend
uv run pytest -m "not gpu"
```

The test suite is hermetic and uses temporary data directories plus the demo engine. GPU or live-engine tests are opt-in with the `gpu` marker. See [`docs/07-development-guide.md`](docs/07-development-guide.md) for module responsibilities and the phase plan.

## Project layout

```text
backend/
  app.py            FastAPI factory and startup wiring
  runner.py         serial job runner
  jobs.py           settings, validation, job serialization
  engines/          one adapter + args module per engine
  pipeline/         merge, pause, prep, and alignment stages
  renderers/        HTML/TXT/SRT/VTT/JSON output
  routes/           pages, partials, and JSON API
  templates/        Jinja2 server-rendered UI
  static/           CSS and vanilla JS (no build step)
docs/               specifications and process notes
tests/              hermetic unit/contract tests
unraid/             Unraid container template
```

## Known limitations

- **Phase A is incomplete.** whisper.cpp and NeMo-Speech.cpp flags were checked against their pinned upstream sources, but their JSON output schemas and GPU behavior have not been exercised against a local runtime in this environment. Qwen3-ASR + ForcedAligner CPU inference is verified in `0.3.1`; CUDA/Vulkan execution of the Qwen aligner and other real-model GPU output remain unverified. No engine silently falls back to another device.
- pyannote is structurally present but still requires Phase A output verification and gated model weights. Qwen3-ASR/aligner CPU inference is exercised by the local image smoke test; CUDA Qwen execution is not.
- The standard Docker image was built and smoke-tested as part of this preview; see [`docs/process-notes.md`](docs/process-notes.md) for the exact verification record. A CUDA target build was started and its cu128 Python layer completed, but CUDA+Vulkan engine compilation did not finish in the available build window. CUDA hardware/output and model-weight inference were not available in the verification environment.
- Model download actions are not exercised by the hermetic test suite, and gated pyannote models require accepting their Hugging Face terms.

## License

MIT. See [LICENSE](LICENSE). Engine binaries and model weights keep their own upstream licenses; the Engines page lists them.
