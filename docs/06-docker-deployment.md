# WrittenSound — Docker, GPU & Deployment Specification

Target flow: `git clone https://github.com/<you>/writtensound && cd writtensound && docker compose up -d` on a machine with (or without) GPUs. Everything else — engine binaries, python deps — is baked; **models are never baked** and download on demand via the UI.

---

## 1. Image matrix

| tag | target | contents | approx size | build-on-VM cost |
|---|---|---|---|---|
| `writtensound:standard` | `final-standard` | CPU+**Vulkan** whisper.cpp + CPU+Vulkan NeMo-Speech.cpp + faster-whisper (CT2 CPU) + Qwen3/pyannote (CPU torch) + ffmpeg | ~3.3 GB (built) | first CI/manual build ~10–20 min |
| `writtensound:cuda` | `final-cuda` | identical app stack, plus CUDA+Vulkan builds of whisper.cpp & NeMo-Speech.cpp + cu12 torch + CTranslate2/CUDA dependencies (cuDNN) in the SAME image | ~8-12 GB (estimated) | ~25-45 min (not completed here) |

`cuda` is the superset for NVIDIA users; A380/other-Vulkan users pick `standard`. The app detects hardware at boot and greys device options that its image cannot honor (standard image: no CUDA radio option).

Both variants are built from **one Dockerfile** with explicit final targets, `final-standard` and `final-cuda`. Compose selects the target; there is no `PROFILE` build argument.

## 2. Dockerfile structure (normative)

The checked-in [`Dockerfile`](../Dockerfile) is authoritative. Its stages are:

- `whisper-vulkan-build` and `nemo-vulkan-build`: Ubuntu 24.04, CMake/Ninja, `glslc` + Vulkan headers, pinned upstream tags. The binaries are copied to `/opt/engines/whispercpp/bin/whisper-cli` and `/opt/engines/nemo/bin/nemo-speech`; `libggml*.so*` and other shared objects are copied beside them under `lib/`.
- `whisper-cuda-build` and `nemo-cuda-build`: `nvidia/cuda:12.8.1-devel-ubuntu24.04`, CUDA architectures `75;80;86;89;90;120`, both `GGML_CUDA=ON` and `GGML_VULKAN=ON`.
- `app-standard` / `app-cuda`: Ubuntu 24.04 Python 3.12, `uv sync --frozen --no-dev --extra engines`, system ffmpeg, Vulkan loader/ICD, and the Python engine stack. The committed lock uses the PyTorch CPU index (`torch==2.9.*` / `torchaudio==2.9.*`). The CUDA layer force-reinstalls `torch==2.9.1+cu128` and `torchaudio==2.9.1+cu128` in the same layer and asserts `torch.version.cuda == "12.8"`.
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

1. **Preferred**: install the *Docker Compose Manager* plugin → point it to `docker-compose.yml` in a cloned copy of the repo (e.g. `/mnt/user/appdata/writtensound/compose`) → up.
2. **Alternative**: build the image once on Unraid (`docker build --target final-standard -t writtensound:standard .`) and add the container by hand with the mappings above (template source file `unraid/writtensound.xml` ships in the repo; fill icon/template icon fields manually on the Unraid form). The template passes through `/dev/dri` via `ExtraParams`; remove that field for a CPU-only host.
3. **Intel A380 passthrough**: ensure `intel-iommu=on` and `/dev/dri/renderD128` exists on host; no `LOCALAPP` secret. Host driver: depends on kernel (Linux ≥5.x ships `xe`/`i915`); **mesa-vulkan-drivers inside the container + host ICD (via /dev/dri) is the only requirement** — no host mesa install needed.
4. **NVIDIA on unraid**: attach the NVIDIA Driver plugin + NFD (nvidia-docker). Unraid 7.x template settings include "GPU: NVIDIA" dropdown — mapped equivalent = `deploy.resources.reservations.devices` above.
5. Data share: `/mnt/user/appdata/writtensound` (the Unraid template default) on a ZFS/BTRFS share or cache-only pool — user must ensure sub-directories get created with correct permissions (PUID/PGID approach handles it).

## 5. Runtime flow (post-boot, standard profile, A380)

1. Container boots → hardware probe: `/dev/dri/renderD*` seen, `vulkaninfo --summary` reports `Intel(R) Arc A380` under "device 0". Hardware chips in the header show `Vulkan ok`. `torch.cuda.is_available() False` → CUDA chips grey.
2. User drops 3 files, adds queue, settings show default model `wcpp-large-v3-turbo-q8_0`.
3. `model_state` for that id is `absent` → on start, worker triggers FetchModel → downloads 874 MB from HF (`ggerganov/whisper.cpp` file `ggml-large-v3-turbo-q8_0.bin`) with SSE progress; state → `ready`.
4. Pipeline then runs at stage `ASR`, spawning `whisper-cli` with vulkan flags; UI shows stage bar.
5. On done, out fileset listed in job row; clickable to download.

## 6. Optional CI workflow (`.github/workflows/docker.yml`)

- Trigger: `push` + tags `v*` (save preview). Job 1: build `final-standard` (runs on `ubuntu-latest` with a vulkan-capable builder). Job 2: build `final-cuda`. Push both to `ghcr.io/<owner>/writtensound:{standard,cuda}-{sha}` (+ `latest` for standard on tag events).
- Optional `workflow_dispatch` for spike runs. Artifact supply keeps repo self-sufficient without publishing.

## 7. Operational runbook (condensed)

| Symptom | Action |
|---|---|
| Vulkan chips grey on A380 | confirm host renderD device exists; `--device /dev/dri` in container; `mesa-vulkan-drivers` present |
| HF downloads stall | check `HF_TOKEN` for gated models only; retries resumable (`huggingface_hub` aio); server bandwidth in container |
| GPU OOM on whisper large | pick smaller/quantized model (q5_0/q8_0 on turbo); or force CPU |
| Engine CLI crash | job error tray lines + `logs/engine-asr.log` on job page; fix `SPIKE` flags in args.py |
| Data dir on disk full | staging cleanup service + retention; consider `WS_DATA_DIR` on bigger share |
| pyannote bad token | settings page token save; retry download; check two-repo acceptance |

## 8. Verification status (2026-10-23)

- `docker build --target final-standard -t writtensound:standard .` completed on Docker 29.9.2 / Linux x86_64. The image is 3.28 GB, reports `torch 2.9.1+cpu`, and both pinned engine binaries (`whisper-cli` v1.9.5 and `nemo-speech` v0.2.0) answer `--help`.
- The standard container starts as UID/GID 99/100, reaches `healthy`, serves the queue and Engines pages, and completed an uploaded WAV through the demo engine with `ir.json`, `tour.html`, `tour.srt`, `tour.txt`, and `tour.vtt` plus pipeline/engine logs.
- The CUDA target's Python layer installed `torch 2.9.1+cu128`, installed the expected cu12 NVIDIA wheels, and asserted `torch.version.cuda == "12.8"`. The full `final-cuda` target was stopped during nvcc compilation of the CUDA+Vulkan engine stages on this CPU-only host, so no `writtensound:cuda` image is claimed. CUDA hardware execution, A380/Vulkan execution, and real model inference remain unverified. Do not present those paths as production-verified.
- `uv run ruff check .`, `uv run mypy backend`, and `uv run pytest -m "not gpu"` (33 tests) pass on the source tree. See [`process-notes.md`](process-notes.md) for the exact record.
- The app uses no silent device fallback: a failed engine invocation becomes a visible job error, and the aligner refuses to run until Phase A confirms its CLI contract.
