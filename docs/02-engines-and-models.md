# WrittenSound — Engines & Model Catalog

This document is normative for what appears in the **Engines** page, the model dropdowns and the per-feature graying rules. Everything cross-referenced here is sourced from the model cards/runtime docs as of **October 2026**; entries marked `SPIKE:` must be verified against actual CLI/library output in Phase A before the catalog ships defaults (they are stored as flags in the catalog YAML so behavior can be trivially corrected).

---

## 1. Engine registry (canonical order)

| engine key | runtime | ships in image | device support | involvement |
|---|---|---|---|---|
| `faster-whisper` | CTranslate2 (`faster_whisper`) | both variants | CPU, CUDA (**no Vulkan**) | whisper-family ASR, word timestamps, batched |
| `whisper.cpp` | whisper-cli (GGML) | both variants | CPU, **Vulkan**, CUDA | whisper-family ASR, token timestamps, SRT native |
| `nemo-speech` | NeMo-Speech.cpp (GGUF) | both variants | CPU, **Vulkan**, CUDA | Nemotron/Parakeet ASR **and** diarization (Sortformer / Nemotron-3) |
| `qwen3-asr` | `transformers` ==5.20.2 | both variants (CUDA deps only in cuda image) | CPU, CUDA (**no Vulkan**) | Qwen ASR; requires the ForcedAligner for word timestamps |
| `pyannote` | pyannote.audio | both variants | CPU, CUDA (**no Vulkan**) | diarization only |
| `qwen3-aligner` | `transformers` ==5.20.2 | both | CPU, CUDA | **aligner**: word timestamps for ANY transcript (≤5-min windows, 11 langs) |

An engine is *available to the UI* only if (a) its binary/module is present and (b) the engine is toggled on in **Engines** (default: all on; user may disable any engine entirely).

## 2. Canonical catalog table (normative)

`id` = what the DB & API store. `repo` = Hugging Face repo ID. Sizes = repository-reported file size.

### 2.1 faster-whisper (CT2)

| id | repo | size | langs | word-ts | notes |
|---|---|---|---|---|---|
| `fw-large-v3` | `Systran/faster-whisper-large-v3` | 3.09 GB | 99 multilingual | native (`word_timestamps=True`) | quality ceiling; ~4.5 GB VRAM fp16, ~2.9 GB int8 |
| `fw-large-v3-turbo` | `Purfview/faster-whisper-large-v3-turbo-ct2` | 1.62 GB | 99 multilingual | native | decoder pruned to 4 layers — fast, near-v3 quality |
| `fw-distil-large-v3.5` | `distil-whisper/distil-large-v3.5-ct2` | 1.51 GB | **EN only** | native | best EN speed/quality tradeoff; added fw 1.2.0 |
| `fw-small` | `Systran/faster-whisper-small` | 0.48 GB | 99 | native | fallback for very weak host |

Warnings to render: `fw-distil-*` → "English only". **Vulkan selected** → engine entirely unavailable (see §4).

### 2.2 whisper.cpp (GGML)

| id | repo / file | size | langs | word-ts | notes |
|---|---|---|---|---|---|
| `wcpp-large-v3-turbo-f16` | `ggerganov/whisper.cpp` → `ggml-large-v3-turbo.bin` | 1.62 GB | 99 | native (DTW/token) | best Vulkan quality, VRAM-heavy |
| `wcpp-large-v3-turbo-q8_0` | same repo → `ggml-large-v3-turbo-q8_0.bin` | 874 MB | 99 | native | **default recommendation on A380** |
| `wcpp-large-v3-turbo-q5_0` | same repo → `ggml-large-v3-turbo-q5_0.bin` | 574 MB | 99 | native | weakest A380, cheapest |
| `wcpp-large-v3-q5_0` | same repo → `ggml-large-v3-q5_0.bin` | 1.08 GB | 99 | native | max quality quantized |
| `wcpp-distil-large-v3` | `distil-whisper/distil-large-v3-ggml` → `ggml-distil-large-v3.bin` | ~0.8 GB | **EN only** | native | english-only, ~5× faster than large-v3 |

Warnings: English-only models say so; CUDA device selected → fine, but pick EITHER the CT2 or the GGML copy of the same model (duplicates overlap); DTW metadata required for token-level word timestamps comes from whisper.cpp's `models/dtw/` files — ensure the dir ships in the image (verified in Dockerfile § build).

### 2.3 NeMo-Speech.cpp (GGUF)

| id | repo / file | size | langs | word-ts | notes |
|---|---|---|---|---|---|
| `nemo-nemotron-3.5-asr` | `nvidia/nemotron-3.5-asr-streaming-0.6b` → `*.q8_0.gguf` | 742 MB | 40 locales | `SPIKE: native vs aligner` | Cache-aware FastConformer-RNNT; punctuation+capitalization native; en-US / de-DE / fr-FR / … (tier list in model card) |
| `nemo-nemotron-en` | `nvidia/nemotron-speech-streaming-en-0.6b` → `*.q8_0.gguf` | ~0.6 GB | EN | `SPIKE:` | English-only predecessor, streaming |
| `nemo-parakeet-tdt-v3` *(bonus)* | `nvidia/parakeet-tdt-0.6b-v3` → `*.q8_0.gguf` | ~0.6 GB | 25 EU langs | **native** (TDT per-token spans) | offline only; self-punctuating; bonus model cheap to include |

Diarization models under this engine:

| id | repo / file | max speakers | langs | notes |
|---|---|---|---|---|
| `nemo-diar-nemotron-3` | `nvidia/Nemotron-3-Diarization` → `*.gguf` `SPIKE: exact file` | 8 | EN-centric | newest (Sept 2026), big DER improvement, AOSC streaming + offline |
| `nemo-diar-sortformer-4spk-v2.1` | `nvidia/diar_streaming_sortformer_4spk-v2.1` → `*.gguf` | 4 | EN-centric | fallback |

NeMo runtime extras used internally: Silero VAD (bundled/gguf via the runtime), automatic punctuation/capitalization post-pass, endpointing. ITN and word-boosting are **out of scope in v1** (kept as documented hidden flags).

### 2.4 Qwen3-ASR (transformers)

| id | repo | size | langs | word-ts | notes |
|---|---|---|---|---|---|
| `qwen3-asr-0.6b` | `Qwen/Qwen3-ASR-0.6B-hf` | ~1.0 GB | 30 + 22 Cn dialects | via aligner | f32 on CPU; bf16/f16 on CUDA |
| `qwen3-asr-1.7b` | `Qwen/Qwen3-ASR-1.7B-hf` | ~3.4 GB | 30 + 22 Cn dialects | via aligner | SOTA open-source quality (mean WER 5.59); f32 on CPU; bf16/f16 on CUDA |
| `qwen3-aligner-0.6b` | `Qwen/Qwen3-ForcedAligner-0.6B-hf` | ~1.0 GB | aligns 11 langs | — | aligns any transcript; **word-timestamp fallback engine**; same dtype rule |

Aligned langs (aligner): zh, en, yue, fr, de, it, ja, ko, pt, ru, es.

ASR languages (catalog `qwen3` list, in model-card order): zh, en, yue, ar, de,
fr, es, pt, id, it, ko, ru, th, vi, ja, tr, hi, ms, nl, sv, da, fi, pl, cs,
fil, fa, el, hu, mk, ro. The processor rejects anything outside this set (and
the 22 Chinese dialects are handled by the model inside `zh`/`yue`); do not
advertise the wider whisper/NeMo code lists for this engine.

Runtime pins: `transformers==5.20.2`, `torch==2.9.*`, `torchaudio==2.9.*`,
`torchcodec==0.9.1`, `numpy>=2,<3`, `nagisa==0.3.0`, `soynlp==0.0.493`.
torchcodec moved to torch >= 2.11 in 0.17, so the 0.9.1 pin is mandatory:
without it `torchcodec` imports but its shared libraries fail on the first WAV
decode. `nagisa`/`soynlp` are required by the aligner's Japanese/Korean word
splitter. `Dockerfile` runs `python -m backend.engines.qwen3.runtime` in both
app layers, which checks the pins and decodes a probe WAV through torchcodec.

The image must also install the **`libpython3.12`** apt package. torchcodec
0.9.1's FFmpeg-6 extension (`libtorchcodec_custom_ops6.so`) links against
`libpython3.12.so.1.0`, which Ubuntu 24.04's `python3-venv` does not pull in;
without it the probe fails with `Could not load libtorchcodec` and the
FFmpeg-6 library reports the missing `libpython3.12.so.1.0` symbol. This is an
OS-package dependency, not a Python pin, so it lives in both app-layer apt
lists rather than `uv.lock`.

Dtype is chosen per device, not from the checkpoint config. The Qwen snapshots
declare bf16; CPU inference loads float32 (bf16 convolution is not portable
across CPU kernels), CUDA uses bf16 when `torch.cuda.is_bf16_supported()` and
float16 otherwise (Turing/sm_75 cannot run bf16). Floating processor inputs are
cast to that dtype; ids and masks stay integer. A bf16 checkpoint fed float32
features fails with `Input type (float) and bias type (c10::BFloat16) should be
the same`.

Snapshots must also fetch `chat_template.jinja` (`*.jinja` in the catalog
allow-list). It is a separate file in the Qwen repos, not part of
`tokenizer_config.json`; without it `apply_transcription_request` fails with
"processor does not have a chat template". The ASR and aligner loaders both
repair an older local snapshot from the file, or fetch that one file when it
is missing.

Warnings: CUDA/CPU only (no Vulkan/torch backend); use the native `transformers`
`apply_transcription_request` pipeline rather than the separate `qwen-asr` pip
package where possible (fewer deps, better typing).

### 2.5 pyannote (torch)

| id | repo | max spk | langs | notes |
|---|---|---|---|---|
| `pyannote-community-1` | `pyannote/speaker-diarization-community-1` | ~configurable | multi | current OSS flagship (better than 3.1, esp count/assign) |
| `pyannote-3.1` | `pyannote/speaker-diarization-3.1` | 4 | multi | fallback when community-1 unavailable / stable legacy |

**Gated**: user must (1) accept conditions on TWO pages (pipeline + its `segmentation` model — exact per README), (2) supply an HF read token (Settings). The **Engines** page shows a dedicated "Not yet accepted on HuggingFace?" pane with the steps + links.

## 3. Model & file layout on disk

```
/data/models
├── whispercpp/{model_id}/ggml-*.bin
├── nemo/{model_id}/*.gguf                 # flat: one GGUF per model
├── fastwhisper/{model_id}/                # ct2 dir (snapshot layout)
├── qwen3/{model_id}/                      # transformers snapshot (HF cache layout or folder copy)
├── pyannote/{model_id}/                   # HF cache subtree
└── hf-cache/                              # HF_HOME
```

Download procedure — unified `DownloadManager`:
- Uses `huggingface_hub.snapshot_download(..., allow_patterns=[…])` per catalog `fetch` rules — **excluding** files the runtime won't need (doc/README, *.mlmodelc.zip etc) to save GBs.
- `fetch` rules per catalog entry (see `models_catalog.yaml` in repo `backend/db/catalog.yaml`); custom models: user drops files under `models/<engine>/<dir>` → `POST /api/models/scan` finds loose files and binds them to nearest-fitting engine.
- GGUF monoliths are single-file fetches; CT2/HF dirs are snapshots. All downloads receive 4 `pct` progress events per file → UI bar.
- SHA/size verification where the source lists a sha (whisper.cpp model table) or HF does implicitly.

## 4. Device support & graying rules (normative)

Matrix from §1. Hard rules the UI enforces:

1. `device=Vulkan` with `faster-whisper` / `qwen3-asr` / `pyannote` → **not selectable** (greyed, tooltip: "engine has no Vulkan backend").
2. `CUDA` selects only render in the **cuda image variant** (the app detects; base variant shows those options greyed with "image not built for CUDA").
3. If chosen model's `word-ts == aligner` but `qwen3-aligner` model is not installed → word-level & captions grayed with tooltip "requires the Qwen3 ForcedAligner model (install on Engines page)".
4. `verbatim/fillers` only for whisper-family engines (implemented via initial prompt / hotwords); grayed otherwise.
5. `_auto` language select → greyed when model lacks auto-detection (whisper has it; `qwen-asr` has it; Nemotron 3.5 has `auto` via prompt).
6. If the user forces a device that the runtime probe marked unavailable → **inline warning strip** at settings panel top (not silent fallback).
7. English-only model + non-English language selected → inline warning strip (not greyed — user may want it anyway).

## 5. Capability matrix (what gray-controls in the settings)

Legend: `native` executes via the engine itself; `aligner` via qwen3-aligner; `no` = hide/disable feature.

| engine+model family | languages | lang auto | word timestamps | SRT/VTT captions | interval timestamps | turn timestamps | pauses mark | verbatim |
|---|---|---|---|---|---|---|---|---|
| faster-whisper | 99 | yes | native | native | native | native | yes | yes |
| whisper.cpp (whisper models) | 99 | yes | native (DTW) | native | native | native | yes | yes |
| nemo-speech (Nemotron 3.5) | 40 (locale-tier) | yes | `SPIKE:` likely `aligner` | via word/seg ts | yes | yes | if word-ts | no |
| nemo-speech (Parakeet TDT v3) | 25 | no (explicit) | native | yes | yes | yes | yes | no |
| qwen3-asr | 52 | yes | aligner | via aligner | yes | yes | if argmax aligner | no |
| diar: nemo (Sortformer/Nemotron-3) | 4 / 8 spk | — | n/a | n/a | n/a | yes (turns) | n/a | n/a |
| diar: pyannote comm-1/3.1 | — | — | n/a | n/a | n/a | yes (turns) | n/a | n/a |

Captions require **segment-level timestamps at minimum** — all engines supply those → captions always possible; **word-precision** captions additionally require word-ts (per row above).

## 6. Defaults (first run, A380 tuned)

- Test-install disabled by default; `Engines` page shows "Install" — but `nemo-nemotron-3.5-asr` and `wcpp-large-v3-turbo-q8_0` are **pre-selected** in the model dropdowns (not downloaded — that happens on first use/download click).
- Default job template: language auto, device auto, ASR → `wcpp-large-v3-turbo-q8_0`, diar off, timestamps interval 60 s, formats html+txt+srt+vtt+json, no verbatim, no start/stop.
- `speaker_count` auto; `pyannote-community-1` default diarizer when diar on (CPU-friendly) — recommend Nemotron-3 on GPU hosts.

## 7. Custom/local models

- Every engine supports pointing at a user-provided path if the file/dir type matches the expected runtime format (GGML bin, GGUF, CT2 dir, HF dir).
- The UI offers a "Local file" row in each engine's model select: path within `/data/models/<engine>/`; rescan button.
- No conversion pipeline ships in v1. If the file fails engine sanity check → job fails with that error surfaced.

## 8. Spike items (Phase A deliverable, gates Phase 2+)

| # | What to verify | How |
|---|---|---|
| S1 | `whisper-cli -f audio.wav -m m.bin -ojf` output field names for token/word timestamps; confirm flag names on the pinned tag (`--output-json-full`, `--dtw …`, progress hooks). Do not add `--max-len`; it fragments segment boundaries. | grep `whisper.h` on the pinned tag + run against jfk.wav |
| S2 | `nemo-speech transcribe --json` field schema; whether token/word timestamps exist for Nemotron RNNT and Parakeet TDT; exact `diarize` subcommand + args; `--device vulkan:0` behavior; `--language auto` availability offline | `nemo-speech --help`, `transcribe --help`, `doctor`, run |
| S3 | faster-whisper 1.2.1 batched-pipeline: exact progress callback API (`log_progress`), word dict shape | small python probe |
| S4 | `qwen-asr` pip package stability vs the pinned raw `transformers==5.20.2` path — **the transformers path** is selected; aligner windowing API call pattern follows the pinned release | follow model card snippets + pinned processor source |
| S5 | pyannote community-1 installability + license surface; fix versions/pins | pip install in-builder + acceptance test |
| S6 | NeMo-Speech.cpp release binary availability (bins w/ vulkan+cuda in one?) vs builder; pick primary | check GH release assets |

Each spike result is a one-line patch to this document §2 and to the engine adapter's fixed code paths (exact arg strings live in `backend/engines/<name>/args.py`).
