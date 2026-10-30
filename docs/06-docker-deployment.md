# WrittenSound — Docker, GPU & Deployment Specification

Target flow: `git clone https://github.com/CozyKettle/writtensound && cd writtensound && docker compose up -d` on a machine with (or without) GPUs. Everything else — engine binaries, python deps — is baked; **models are never baked** and download on demand via the UI.

---

## 1. Image matrix

| tag | target | contents | approx size | build-on-VM cost |
|---|---|---|---|---|
| `writtensound:standard` | `final-standard` | CPU+**Vulkan** whisper.cpp + CPU+Vulkan NeMo-Speech.cpp + faster-whisper (CT2 CPU) + Qwen3/pyannote (CPU torch) + ffmpeg | ~3.3 GB (built) | first CI/manual build ~10–20 min |
| `writtensound:cuda` | `final-cuda` | identical app stack, plus CUDA+Vulkan builds of whisper.cpp & NeMo-Speech.cpp + cu12 torch + CTranslate2/CUDA dependencies (cuDNN) in the SAME image | ~8-12 GB (estimated) | ~25-45 min (not completed here) |

The standard target is published as `ghcr.io/cozykettle/writtensound:latest` (and version/SHA tags) by the GitHub Actions workflow. `writtensound:standard` remains the local-build tag. The CUDA target is not published.

`cuda` is the superset for NVIDIA users; A380/other-Vulkan users pick `standard`. The app detects hardware at boot and greys device options that its image cannot honor (standard image: no CUDA radio option).

Both variants are built from **one Dockerfile** with explicit final targets, `final-standard` and `final-cuda`. Compose selects the target; there is no `PROFILE` build argument.

## 2. Dockerfile structure (normative)

The checked-in [`Dockerfile`](../Dockerfile) is authoritative. Its stages are:

- `whisper-vulkan-build` and `nemo-vulkan-build`: Ubuntu 24.04, CMake/Ninja, `glslc` + Vulkan headers, pinned upstream tags. The binaries are copied to `/opt/engines/whispercpp/bin/whisper-cli` and `/opt/engines/nemo/bin/nemo-speech`; `libggml*.so*` and other shared objects are copied beside them under `lib/`.
- `whisper-cuda-build` and `nemo-cuda-build`: `nvidia/cuda:12.8.1-devel-ubuntu24.04`, CUDA architectures `75;80;86;89;90;120`, both `GGML_CUDA=ON` and `GGML_VULKAN=ON`.
- `app-standard` / `app-cuda`: Ubuntu 24.04 Python 3.12, `uv sync --frozen --no-dev --extra engines`, system ffmpeg, Vulkan loader/ICD, and the Python engine stack. The committed lock uses the PyTorch CPU index (`torch==2.9.*` / `torchaudio==2.9.*`). Both layers install `libpython3.12` because torchcodec 0.9.1's FFmpeg-6 extension needs `libpython3.12.so.1.0`, which `python3-venv` does not provide. The CUDA layer force-reinstalls `torch==2.9.1+cu128` and `torchaudio==2.9.1+cu128` in the same layer and asserts `torch.version.cuda == "12.8"`.
- `final-standard` / `final-cuda`: environment contract, engine copies, `/data` mount point, healthcheck, and `CMD ["/app/entrypoint.sh"]`.

**Build contract rules:**
- `WHISPERCPP_TAG=v1.9.5`, `NEMOSPEECH_TAG=v0.2.0`, and `CUDA_VERSION=12.8.1` are pinned in [`docker-build-args.env`](../docker-build-args.env). Never use `main` or an unverified moving tag.
- `CUDA_ARCHITECTURES` defaults to `75;80;86;89;90;120` for the release fat image. Local single-GPU builds can set one value (for example `75` for a T600) to cut nvcc compile time; that image only runs on the selected architecture.
- The default recipe and catalog are data-driven; models are **never** baked into an image. The app writes downloads under `/data/models`.
- uvicorn runs with a single worker (`backend.app:create_app --factory --host 0.0.0.0 --port ${WS_PORT}`). A single worker is required because the queue, SSE hub, and SQLite writer are process-local.
- The entrypoint starts as root only long enough to create the `PUID`/`PGID` user, grant access to the host `/dev/dri` group, and chown the top-level `/data` directories. It then `exec`s the app under `gosu`; the application process is not root.
- `APP_PASSWORD` enables HTTP Basic auth for everything except `/healthz`. It is unset by default; reverse-proxy/TLS remains the operator's responsibility.
- CUDA and Vulkan are built into the CUDA image. If an upstream build cannot combine those backends, the image build fails loudly; the app never silently presents a CUDA device without a CUDA-capable binary.

## 3. `docker-compose.yml` (normative)

The checked-in [`docker-compose.yml`](../docker-compose.yml) is authoritative:

- `writtensound` is the **unprofiled standard service**. `docker compose up -d`
  starts it with the CPU + Vulkan image. It exposes `/dev/dri` and maps
  `${WS_DATA_DIR:-./data}` to `/data`.
- `writtensound-cuda` is behind the `cuda` profile and builds
  `final-cuda`. It reserves NVIDIA GPUs through the Compose device
  reservation block and sets `NVIDIA_DRIVER_CAPABILITIES`.
- Both services deliberately use `container_name: writtensound`. Start only
  one at a time; the named-service command is the safe way to switch.
- Ports and the in-container `WS_PORT` both follow `${WS_PORT:-8080}`.
- PUID/PGID, `APP_PASSWORD`, `HF_TOKEN`, log level, upload limit, and the demo
  engine switch are passed from `.env` or the shell.

Standard (default):

```bash
cp .env.example .env
docker compose up -d --build
```

On a host without `/dev/dri` - including Docker Desktop on Windows, which
cannot pass a Linux render node into the container - use the CPU override:

```bash
docker compose -f docker-compose.yml -f docker-compose.cpu.yml up -d --build
```

The same image still runs the CPU paths. The base compose file intentionally
fails loudly on a missing device rather than silently omitting GPU access.
GPU/Vulkan acceleration requires a Linux/Unraid host with `/dev/dri`.

CUDA (stop the standard container first):

```bash
docker compose down
docker compose --profile cuda up -d --build writtensound-cuda
```

On Docker Desktop for Windows, CUDA works through the WSL2 backend when the
Windows NVIDIA driver is installed; `/dev/dri` Vulkan passthrough still does
not. Confirm GPU visibility before building with:

```bash
docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi
```

The CUDA target is not prebuilt in this preview and must be built locally. Its
Turing-class `sm_75` architecture is included, so a T600 can run CUDA kernels;
its 4 GB VRAM still limits usable model sizes. Set `CUDA_ARCHITECTURES=75` for
a T600-only local build to avoid compiling all six fatbin variants.

Do not run `docker compose --profile cuda up -d` without the service name:
that also selects the unprofiled standard service and both services cannot
own the same container name. There is no `PROFILE` environment variable
switch; the target is part of each service definition.

`.env` can set `WS_DATA_DIR`, `WS_PORT`, password, token, and engine pins. On
Unraid point `WS_DATA_DIR` at a share or appdata path; the compose default is
the repository-local `./data` so a fresh checkout works on any host.

## 4. Unraid specifics (6.x)

Unraid Docker templates point at a prebuilt image; they do not clone a repository or run a build. The repository's [`unraid/writtensound.xml`](../unraid/writtensound.xml) therefore targets the published GHCR image directly.

1. **Recommended — prebuilt image**: use `ghcr.io/cozykettle/writtensound:latest` in the template or Add Container form. The public GHCR package needs no registry login. The template passes `/dev/dri` via `ExtraParams`; remove that field for a CPU-only host. Use a container version tag such as `0.4.0` instead of `latest` when you want deliberate updates.
2. **Local build (alternative)**: build on Unraid with `docker build --target final-standard -t writtensound:standard .`, then point the template at `writtensound:standard`. This keeps no registry dependency but uses local build cache and takes substantially longer.
3. **Docker Compose Manager**: optional for people who prefer compose. A checkout can run `docker compose up -d --build`; the checked-in compose file builds from source unless an image is supplied. It is not required for the Unraid template.
4. **Intel A380 passthrough**: ensure `intel-iommu=on` and `/dev/dri/renderD128` exists on host; no `LOCALAPP` secret. Host driver: depends on kernel (Linux ≥5.x ships `xe`/`i915`); **mesa-vulkan-drivers inside the container + host ICD (via /dev/dri) is the only requirement** — no host mesa install needed.
5. **NVIDIA on unraid**: attach the NVIDIA Driver plugin + NFD (nvidia-docker). Unraid 7.x template settings include "GPU: NVIDIA" dropdown — mapped equivalent = `deploy.resources.reservations.devices` above.
6. Data share: `/mnt/user/appdata/writtensound` (the Unraid template default) on a ZFS/BTRFS share or cache-only pool — user must ensure sub-directories get created with correct permissions (PUID/PGID approach handles it).

## 5. Runtime flow (post-boot, standard profile, A380)

1. Container boots → hardware probe: `/dev/dri/renderD*` seen, `vulkaninfo --summary` reports `Intel(R) Arc A380` under "device 0". Hardware chips in the header show `Vulkan ok`. `torch.cuda.is_available() False` → CUDA chips grey.
2. User drops 3 files, adds queue, settings show default model `wcpp-large-v3-turbo-q8_0`.
3. `model_state` for that id is `absent` → on start, worker triggers FetchModel → downloads 874 MB from HF (`ggerganov/whisper.cpp` file `ggml-large-v3-turbo-q8_0.bin`) with SSE progress; state → `ready`.
4. Pipeline then runs at stage `ASR`, spawning `whisper-cli` with vulkan flags; UI shows stage bar.
5. On done, out fileset listed in job row; clickable to download.

## 6. Publishing workflow (`.github/workflows/docker.yml`)

- Trigger: tag push `v*` (release) or `workflow_dispatch` (manual test).
- Builds `final-standard` on `ubuntu-latest` for `linux/amd64` and pushes to `ghcr.io/cozykettle/writtensound` with:
  - `latest` on version tags;
  - the semantic version and major/minor tags (for example `0.3.1`, `0.3`);
  - a `sha-<commit>` tag for every build.
- Authenticates with the built-in `GITHUB_TOKEN` (`packages: write`); no personal access token is stored.
- The standard package is public and free under GitHub's public-container policy. Do not upload a `docker save` tarball to Releases or Actions artifacts; that duplicates the image and defeats the storage savings.
- CUDA is deliberately not published. It remains an optional local build for NVIDIA hosts; the A380 deployment uses the standard image.
- Verify the GHCR package visibility is **Public** after the first push; this package is public and was pulled anonymously. A private package would require Unraid registry credentials.

## 7. Operational runbook (condensed)

| Symptom | Action |
|---|---|
| Vulkan chips grey on A380 | confirm host renderD device exists; `--device /dev/dri` in container; `mesa-vulkan-drivers` present |
| HF downloads stall | check `HF_TOKEN` for gated models only; retries resumable (`huggingface_hub` aio); server bandwidth in container |
| GPU OOM on whisper large | pick smaller/quantized model (q5_0/q8_0 on turbo); or force CPU |
| Engine CLI crash | job error tray lines + `logs/engine-asr.log` on job page; fix `SPIKE` flags in args.py |
| Data dir on disk full | staging cleanup service + retention; consider `WS_DATA_DIR` on bigger share |
| pyannote bad token | settings page token save; retry download; check two-repo acceptance |

## 8. Verification status (2026-10-29)

- The `v0.4.0` source tree fixes live queue updates and insertion order,
  faster-whisper 1.2.1's `temperature` keyword, HTML meta date wrapping,
  whisper.cpp's missing duration and short-paragraph reflow, and late NeMo word
  timestamps; Parakeet's dropped-utterance behaviour is documented as an
  upstream limitation with no CLI precision flag. The source gates pass with
  77 non-GPU tests. The local `final-standard` image (3.54 GB) passes the
  build-time Qwen probe; the container runs the app as UID 99, reaches
  `healthy`, serves the 0.4.0 queue page and SVG favicon, streams a live
  newest-first `queue` snapshot with full rows, and renders a demo transcript
  with a real `00:00:12` duration and wrapping meta grid. CUDA remains parked:
  no CUDA image was built or validated for this release.

- `docker build --target final-standard -t writtensound:standard .` completed on Docker 29.9.2 / Linux x86_64. The image is 3.54 GB, reports `torch 2.9.1+cpu`, and both pinned engine binaries (`whisper-cli` v1.9.5 and `nemo-speech` v0.2.0) answer `--help`.
- The local standard container starts as UID/GID 99/100, reaches `healthy`, serves the queue and Engines pages, and completed an uploaded WAV through the demo engine with `ir.json`, `tour.html`, `tour.srt`, `tour.txt`, and `tour.vtt` plus pipeline/engine logs.
- The GitHub Actions workflow built `v0.1.0` in 8m21s and published `ghcr.io/cozykettle/writtensound:latest` plus `0.1.0`, `0.1`, and `sha-961fb42` tags. The package is public; an anonymous pull succeeded and the resulting container reached `healthy` with `/healthz` returning 200. Published digest: `sha256:e95fceeb2efa8b50df3668e437243eae8226c1223b2ce0605e4faed18e98735b`.
- The GitHub Actions workflow built `v0.1.1` and published `latest`, `0.1.1`, `0.1`, and `sha-e75f3a7`. The public image pulled successfully; the local container started as UID/GID 99/100, created `/home/writtensound` for the Vulkan shader cache, reached `healthy`, and returned HTTP 200 from `/healthz`. Digest: `sha256:a844d0d2425ca4f1fdc358b298ffab2eb45a34d6023ee84e4561a13d0ff45f7a`.
- The `v0.2.0` source tree preserves NeMo top-level words, groups captions into multiple cues, adds `{base}.timed.txt`, suppresses single-speaker labels, and splits interval markers at word boundaries. The source gates pass with 40 non-GPU tests. The tag workflow built in 9m and published `latest`, `0.2.0`, `0.2`, and `sha-f0a1126` at digest `sha256:d4f56f8a0a4cd5a8ddaf79a286f6a4a62d625f32d908f164b1d2678f983dce43`; registry manifests for all four tags resolve to that digest. The `0.2.0` container started as UID/GID 99/100, reached `healthy`, returned HTTP 200 from `/healthz`, and served the queue page reporting `0.2.0` with the model selector and timed-text option.
- The `v0.3.0` source tree filters whisper.cpp special tokens, merges subword pieces into words, and repairs Qwen3 `chat_template.jinja` for both ASR and the ForcedAligner. The source gates pass with 53 non-GPU tests. The tag workflow built in 8m41s and published `latest`, `0.3.0`, `0.3`, and `sha-9b935e1` at digest `sha256:e64334c7f7e4745c1ba086a149881d094fefb3f0702c00104322bda4203f7efc`; registry manifests for all four tags resolve to that digest. The `0.3.0` container started as UID/GID 99/100, reached `healthy`, returned HTTP 200 from `/healthz`, and served the queue page reporting `0.3.0`.
- The `v0.3.1` source tree fixes the released-image Qwen failure: `torchcodec==0.9.1` (0.17 needs torch >= 2.11), `transformers==5.20.2`, `numpy>=2,<3`, `nagisa`/`soynlp`, and the `libpython3.12` OS package; ASR/aligner receive 16 kHz float32 PCM windows instead of file paths, generated tokens are sliced past the prompt, the 5.20.2 aligner API is used with segment windowing, and dtype follows the device (float32 CPU, bf16/float16 CUDA). The source gates pass with 68 non-GPU tests. The local `final-standard` image is 3.54 GB; both app layers fail the build unless `python -m backend.engines.qwen3.runtime` reports `Qwen3 runtime probe ok: transformers 5.20.2, torch 2.9.1+cpu, torchaudio 2.9.1+cpu, torchcodec 0.9.1, nagisa, soynlp, pyannote.audio, torchcodec WAV decode`. A real CPU smoke run inside that image transcribed a 6.69 s clip with Qwen3-ASR-0.6B (exact text, `language='en'`) and aligned 16 words with Qwen3-ForcedAligner-0.6B.
- The `v0.3.1` tag workflow built in 8m56s and published `latest`, `0.3.1`, `0.3`, and `sha-814eb59` at digest `sha256:ed651b4de76e029a468dfcaf8e9cdc5eea20a282fc174f7d144f82f7331d9800`; registry manifests for all four tags resolve to that digest. The published image's probe reports the pinned Qwen stack and a passing torchcodec WAV decode. A real CPU smoke run inside the published image produced the exact 6.69 s transcript and 16 aligned words. The container started as UID 99, reached `healthy`, returned HTTP 200 from `/healthz`, and served the queue page reporting `0.3.1`.
- The `v0.4.0` source tree fixes live queue updates and insertion order, faster-whisper 1.2.1's `temperature` keyword, HTML meta date wrapping, whisper.cpp's missing duration and short-paragraph reflow, and late NeMo word timestamps; it adds the SVG favicon and documents Parakeet's dropped-utterance limitation. The source gates pass with 77 non-GPU tests. A source-server E2E run received 12 `queue` events across six jobs, with the largest snapshot carrying all six full rows newest-first. The local `final-standard` image (3.54 GB) passed the Qwen probe and a container smoke test covering the version string, favicon, live queue, `00:00:12` duration, and wrapping meta grid.
- The `v0.4.0` tag workflow built in 9m45s and published `latest`, `0.4.0`, `0.4`, and `sha-e6ae998` at digest `sha256:c4e1dd4fc6763e49f287961da6aec6a8884754062c4d54c0fcd33ab3dfd6b2eb`; registry manifests for all four tags resolve to that digest. `docker pull` succeeded on this host. The published image re-ran the container smoke test successfully; the app process runs as UID 99, the container reaches `healthy`, `/healthz` returns HTTP 200, and both pinned engine binaries answer `--help`. CUDA remains parked: no CUDA image was built or validated for this release.
- The CUDA target's Python layer installed `torch 2.9.1+cu128`, installed the expected cu12 NVIDIA wheels, and asserted `torch.version.cuda == "12.8"`. The full `final-cuda` target was stopped during nvcc compilation of the CUDA+Vulkan engine stages on this CPU-only host, so no `writtensound:cuda` image is claimed. CUDA hardware execution remains unverified.
- NeMo-Speech CPU inference produced transcript text for a real Parakeet TDT v3 job; the parser regression test covers the stderr warning plus line-delimited `words` output. A380 Vulkan inference is not verified: full-context Parakeet requested a 7.27 GB graph on the 6 GB card and failed with `ErrorOutOfDeviceMemory`. Use whisper.cpp for long A380 jobs.
- `uv run ruff check .`, `uv run mypy backend`, and `uv run pytest -m "not gpu"` (68 tests) pass on the `v0.3.1` source tree. See [`process-notes.md`](process-notes.md) for the exact record.
- The app uses no silent device fallback: a failed engine invocation becomes a visible job error. Qwen3 ASR + aligner CPU inference is now verified end-to-end; CUDA and Vulkan execution still need hardware verification.
