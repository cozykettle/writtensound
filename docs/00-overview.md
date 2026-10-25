# WrittenSound — Overview, Goals & Stack

**WrittenSound** is a self-hosted, Docker-deployed web app for local, private speech-to-text transcription and speaker diarization. It follows the simple, functional philosophy of desktop transcription tools — no cloud, no accounts-as-a-product, no fancy charts — but runs on your server, is driven through a browser, and offers a far broader model catalog than the usual two-preset desktop app.

- **Target deployment:** Unraid server with an Intel Arc A380 (Vulkan-capable) GPU and optional fallback NVIDIA/CUDA machines.
- **Primary interface:** one web page (queue-centric), simple, htmx-driven, keyboard-light.
- **Everything local:** audio never leaves the machine except to download model weights from Hugging Face.

---

## 1. What it does (v1 scope summary)

1. Upload one file, many files, or a whole folder (drag-and-drop or picker) from the browser.
2. For each item (or a selection of items), configure, **per job**:
   - ASR **engine** (faster-whisper / whisper.cpp / Nemotron / Qwen3) and **model**, downloaded from Hugging Face on demand or loaded from an existing GGUF on disk.
   - **Device**: auto / CPU / CUDA / Vulkan, with warnings when a model or engine cannot use a chosen device.
   - **Output style**: raw text only, timestamps every *N* seconds, timestamps on speaker turns, word-level timestamps, SRT/VTT captions, and timed text.
   - **Diarization**: off / NVIDIA Nemotron-3 Diarization / pyannote; optional number of speakers.
   - Additional transcript settings: language + auto-detect (where supported), start/stop range, mark pauses, verbatim/fillers mode (where supported).
3. A single serial queue processes jobs one at a time and writes all selected output formats to a shared output directory.
4. An **Engines** page lists every supported engine/model with install size, device support and notes; downloading an engine's weights is one click. Engines may be installed but kept disabled — a disabled-but-installed engine only appears when explicitly chosen.

---

## 2. Goals

| # | Goal |
|---|---|
| G1 | Focused, queue-first UX mapped to the web: settings per file, queue, simple results. |
| G2 | Multiple ASR engines + models, selected in the UI, weights auto-downloaded from HF or loaded from local GGUF. |
| G3 | GPU flexibility: **Vulkan** (Intel/AMD — A380 primary target) and **CUDA**, with automatic CPU fallback and device warnings. |
| G4 | Serial queue with per-job settings snapshots, folder/multi-file upload, cancel/retry. |
| G5 | Complete output set: HTML transcript, TXT, SRT, VTT, timed text, JSON (+ pause marking, timestamps, speaker labels). |
| G6 | Zero-config-ish Docker: `docker compose up -d` (or Unraid template) from the repo; models are the only thing downloaded at runtime, on demand, via the UI. |
| G7 | Everything auditable: precise specs so a coding agent can implement without design decisions on the fly. |

## 3. Non-goals (v1)

- **No in-browser audio-sync editor.** Revisit after v1; transcripts are static HTML.
- No multi-user accounts, no public-internet exposure story beyond optional password.
- No training/fine-tuning, no TTS, no translation.
- No cross-language speech translation.
- No distributed processing / cluster mode.
- No automatic model *conversion* (custom `.nemo`/CT2 conversion) — supports published/standard formats only, plus user-supplied files that already match runtime formats.

## 4. Stack decision (and why)

| Layer | Choice | Rationale |
|---|---|---|
| Backend | **Python 3.12 + FastAPI + uvicorn** | Every interesting ASR/diarization library is Python-native (`faster-whisper`, `transformers` for Qwen3-ASR, `pyannote.audio`); subprocesses for the two C++ runtimes (whisper.cpp, NeMo-Speech.cpp). One language = one container layer. |
| UI | **Server-rendered Jinja2 + htmx + a little vanilla JS** (no SPA, no node build chain) | Queue updates stream over SSE; forms are regular posts; drag-and-drop and upload sequencing is small vanilla JS. trivially debuggable, minimal dependency surface. |
| DB | **SQLite** (WAL) single file under a volume | one writer (serial queue), tiny schema, zero ops. |
| Queue execution | **Single worker thread + subprocess orchestration**, asyncio for SSE/upload only | one job at a time; avoids GPU contention and OOM on 6 GB A380. |
| C++ engines | **whisper.cpp** (whisper family, GGUF/GGML) and **NVIDIA NeMo-Speech.cpp** (Nemotron ASR + Sortformer/Nemotron-3 diarization, Silero VAD) | The only maintained paths to **Vulkan** for these models; both are CLI-friendly with JSON/structured output; NeMo-Speech.cpp is NVIDIA's official ggml runtime with day-0 model support and a documented vulkan backend (`--device vulkan:0`). |
| Deep Learning engines | **faster-whisper** (CT2) and **Qwen3-ASR via `transformers>=5.13`** | faster-whisper: fastest CPU/CUDA whisper path, word timestamps, batched mode. Qwen3: top open-source ASR quality (`AutoModelForMultimodalLM`), plus **Qwen3-ForcedAligner** for word timestamps on any transcript. |
| Diarization | **Nemotron-3-Diarization (GGUF via NeMo-Speech.cpp)** and **pyannote** (`community-1`, fallback `3.1`) | Nemotron: 8 speakers, GPU via Vulkan — works on the A380. pyannote: mature, torch CPU/CUDA. |
| Audio prep | **ffmpeg** (in image) | extract 16 kHz mono WAV for the C++ engines; clipping start/stop ranges. |
| Auth | Optional HTTP basic auth from `APP_PASSWORD` env | LAN tool; frictionless when unset. |

**Device matrix per runtime** (drives graying and warnings):

| Runtime | CPU | Vulkan | CUDA | Sources |
|---|---|---|---|---|
| faster-whisper (CT2) | yes | **no** | yes | CTranslate2 supports CUDA/CPU only |
| whisper.cpp (GGML) | yes | yes | yes | `GGML_VULKAN=1`, `GGML_CUDA=1` builds |
| NeMo-Speech.cpp (GGUF) | yes | yes | yes | `--device cpu / vulkan:0 / cuda:0` |
| Qwen3-ASR (torch) | yes | **no** | yes | torch backends |
| pyannote (torch) | yes | **no** | yes | torch backends |

Two image variants carry this: `writtensound:latest` ships **CPU + Vulkan** builds (A380 default, ~2 GB class image); `writtensound:cuda` adds CUDA-capable builds + cu12 torch/CT2 for the NVIDIA case. The app **detects** what is available at startup and greys/annotates the UI accordingly in either variant. Models are never baked into images.

## 5. Engine roster (v1 curated catalog)

- **faster-whisper** — `large-v3`, `large-v3-turbo` (CT2), `distil-large-v3.5` (CT2, EN-only) [CPU/CUDA]
- **whisper.cpp** — `large-v3-turbo` (`f16`/`q5_0`/`q8_0` GGML), `large-v3-q5_0`, `distil-large-v3` (GGML, EN-only) [CPU/**Vulkan**/CUDA]
- **NeMo-Speech.cpp** — `nemotron-3.5-asr-streaming-0.6b` (GGUF q8_0, 40 locales) [CPU/**Vulkan**/CUDA]
- **Qwen3-ASR** — `Qwen3-ASR-0.6B-hf`, `Qwen3-ASR-1.7B-hf` (transformers) [CPU/CUDA] — with optional **Qwen3-ForcedAligner-0.6B** for word timestamps
- **Diarization** — `nvidia/Nemotron-3-Diarization` (GGUF via NeMo-Speech.cpp, ≤8 spk) [CPU/**Vulkan**/CUDA]; `pyannote/speaker-diarization-community-1` (fallback `3.1`) [CPU/CUDA]
- **Bonus (optional, cheap)** — `nvidia/parakeet-tdt-0.6b-v3` GGUF via the same NeMo runtime (25 langs, native per-token time spans → word timestamps without an aligner)

Details, repo IDs, sizes, capability matrices and exact download mechanics live in `02-engines-and-models.md`.

## 6. Desktop transcription workflow mapping

| Common desktop-app setting | WrittenSound implementation |
|---|---|
| Choose audio file(s) | Upload list + folder drag/drop (+ picker with `webkitdirectory`) |
| Save transcript as (html/txt/vtt) | Per-job checkboxes: HTML, TXT, SRT, VTT, timed text, JSON |
| Start/Stop range | `start_at`/`end_at` (hh:mm:ss) → ffmpeg clip |
| Language + Auto | Language select; greyed if engine lacks auto-detect |
| Quality (precise/fast) | Replaced by explicit model choice (quality is the model catalog's job). Default model preselect: `whisper.cpp large-v3-turbo-q8_0` for A380 |
| Mark pause (toggle/dots) | `mark_pauses` + `pause_min_s`; requires word/word-gap data → greyed otherwise |
| Speaker detection (auto/n/none) | `diar: off / nemo / pyannote` + `speaker_count: auto|n` |
| Overlapping speech marking | **v1: not supported** (diarizers' overlap info is unreliable); feature is present in UI as disabled with a tooltip "planned" |
| Disfluencies (um/uh) | `verbatim` toggle — supported by whisper-family engines only (prompt/hotwords), greyed otherwise |
| Timestamps [hh:mm:ss] every 60 s / per speaker | `timestamps_mode: none | interval | turns` + interval seconds |
| Start/Stop; queue tab; ✔/⟲/X actions | Jobs page with same actionable verbs |
| Editor | out of scope v1 (Section 3) |

## 7. Key risks & their mitigations

1. **Terminal timestamp support engines are unknown quantities** (RNNT/Nemotron GGUF word times). → Phase A spike verifies actual JSON output of `nemo-speech transcribe --json` and `whisper-cli --output-json-full`; catalog marks word-timestamps `native | aligner | none`; the Qwen3-ForcedAligner serves as universal fallback (≤5-min windows, 11 languages).
2. **Vulkan maturity differences** between images (Intel vs NVIDIA). → A380 is a first-class test target; quantized `q5_0/q8_0` GGUF are the default; CPU always available as fallback.
3. **pyannote gating** → explicit setup modal: accept two HF repo licenses, paste token in Settings (stored in `config/db`), app retries download after token is set.
4. **Image size creep** → base image only carries runtimes + small python deps (~2 GB class); CUDA variant separate tag; models always external in `/data/models`.
5. **Agent implementation drift** → specs below fix module names, schemas, CLI flags and acceptance criteria; the agent brief (`08-agent-brief.md`) requires reading docs 01–08 before coding.

## 8. Glossary

- **Engine** — a runtime able to execute one or more models (e.g. "faster-whisper", "nemo-speech").
- **Model** — a concrete weight set (e.g. "large-v3-turbo-q8_0") with declared capabilities.
- **Job** — one media file + one settings snapshot + status + outputs.
- **Queue** — the ordered set of jobs and the single service position.
- **Device** — `cpu`, `cuda`, `vulkan` (+ index).
- **GGUF/GGML** — weight formats consumed by NeMo-Speech.cpp / whisper.cpp respectively.
- **CT2** — CTranslate2 format consumed by faster-whisper.
- **Diarization** — "who spoke when"; produces speaker turns merged with ASR output.
- **RTTM** — text format for speaker turn segments.
