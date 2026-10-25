# WrittenSound — Pipelines, IR & Renderers

Normative for everything between "queue picks job" and "files land in `out/`". Engine arg strings marked `SPIKE` are verified in Phase A and then frozen in `backend/engines/*/args.py` (single source of truth; nothing else may hard-code engine flags).

Path convention inside this doc: `JOBS = WS_DATA_DIR/jobs`, `MODELS = WS_DATA_DIR/models`, job dir = `JOBS/{job_id}`.

---

## 1. Job stage list (state machine)

```
ingest      ffprobe validation; decide basename; error out early on bad media
prepare     clip (if start_at/end_at) → prep.wav (16k mono s16le) for ALL engines
              * clips use: ffmpeg -hide_banner -ss {start} -to {end} -i src -vn -> temp
              * prep wave: ffmpeg -hide_banner -i in -vn -ac 1 -ar 16000 -f s16le? NO: 16-bit PCM wav:
                ffmpeg -hide_banner -i in -vn -ac 1 -ar 16000 -c:a pcm_s16le out.wav
              * write meta.json: duration, sample_rate, original filename
asr         engine adapter → TranscriptionResult (IR below)
diarize     engine adapter → DiarResult (skipped when off)
align       word-level info fix-up: attach speaker labels to segments/words;
              optional qwen3-aligner pass when `word_timestamps` needed by engines lacking them
render      HTML / TXT / SRT / VTT / timed text / JSON per settings.output
```

Stage→progress weights (worker-internal normalization): ingest 2%, prepare 3%, asr 65%, diarize 20%, align 4%, render 6%.

## 2. Internal IR (single source of truth)

```python
class Word(BaseModel):
    start: float | None; end: float | None
    text: str
    speaker: int | None = None       # 0-based index → "Speaker 1"
    probability: float | None = None

class Segment(BaseModel):
    start: float; end: float
    text: str
    words: list[Word] | None = None
    language: str | None = None

class TranscriptionResult(BaseModel):
    engine: str; model_id: str
    device: str                      # resolved
    compute: str | None              # e.g. "cuda-fp16", "vulkan-q5_0"
    language: str | None             # detected/forced; None = unknown
    language_prob: float | None
    duration_s: float
    media_id: str                    # basename sans extension of source
    segments: list[Segment]

class DiarTurn(BaseModel):
    start: float; end: float
    speaker: int                     # 0-based arrival order
    overlap: bool = False            # overlaps previous/prior speaker turn

class DiarResult(BaseModel):
    engine: str; model_id: str; device: str
    turns: list[DiarTurn]

class MergedSegment(Segment):        # final IR to render
    speaker: int | None
    overlap: bool = False            # any word indeterminate for this segment
```

## 3. Engine adapters — exact invocation

### 3.1 faster-whisper (in-process)

```python
from faster_whisper import WhisperModel, BatchedInferencePipeline   # v1.2.1

model = WhisperModel(models_dir_for(model_id), device="cuda"|"cpu", compute_type="float16"|"int8",
                     cpu_threads=max(2, os.cpu_count()//2))    # int8 on CPU, float16 on CUDA
seg_iter, info = model.transcribe(prep_wav_path,
        language=None if settings.language == "auto" else settings.language,
        word_timestamps=settings.word_timestamps, vad_filter=True,
        vad_parameters=VadOptions(min_silence_duration_ms=500),
        condition_on_previous_text=False, temperatures=(0.0,),
        initial_prompt=INITIAL_PROMPT_VERBATIM if settings.verbatim else None,
        log_progress=True, progress_hook=hook)   # sequential (batched OFF in v1 by default)
```

`hook` = callable(frac, total) → `(frac/total, f"asr {frac}/{total}")`. `info.language[_probability]`, `info.duration`. Words include `prob` (0..1). No diarization integration.

Options not exposed in UI v1: `beam_size=1` fixed default, `patience=1.0`, `suppress_tokens` default, `chunk_length=30`.
`SPIKE S3` — re-check exact `progress_hook` signature and word dict keys on 1.2.1.

### 3.2 whisper.cpp (subprocess `whisper-cli`)

```
whisper-cli -m {ggml_path} -f {prep_wav} \
  --language {auto|code} \
  --output-json-full --output-file {job_dir}/out/{base}.full \
  --progress    # prints percent lines that the adapter parses SPIKE S1 (exact flag)
  [GPU flags per device/kernel: e.g. --gpu-device 0 / --no-gpu  → SPIKE S1 exact names]
  [if word_timestamps: --max-len 1 --token-timestamps?? + --dtw {model-tiny?}  → SPIKE S1]
  [if verbatim: --prompt "…uh, um —"?? → SPIKE S1 (initial prompt support)]
```

- stderr captured verbatim into logs; stdout progress lines parsed for the `job` SSE.
- Adapter converts the full JSON (segments[]→tokens[] with t0/t1 in ms) into `Segment/Word` (t0/1000). Words carry `p` (probability).
- DTW metadata: whisper.cpp requires per-token timestamps enabled AND a `dtw` alignment file matching the mels model on some modes; catalog sets `word-ts` accordingly per S1 finding (input tokens time offsets exist natively with DTW; else fall back to `aligner` capability when qwen3-aligner installed, else `no`).
- **Vulkan device selection**: `--gpu-device {idx}` `SPIKE S1`; enforce via resolved device index.

### 3.3 NeMo-Speech.cpp (subprocess `nemo-speech`)

```
nemo-speech transcribe {prep_wav} --model {gguf_path} --json \
  --device {vulkan:0|cuda:0|cpu} --language {auto|code-locale}
```

- `--language` maps UI code → catalog locale code (en-US, de-DE, …); `auto` allowed. JSON structure verified by S2 (fields to be mapped into `Segment/Word`).
- Parakeet TDT returns top-level `words[]` without `segments[]`. The adapter maps those words into one `Segment` with its `words` list intact and uses the payload `text` when present; it must not fall back to a word-less `0 → duration` segment. Renderers group the preserved words into cues.
- `SPIKE S2` — whether per-token timestamps exist for Nemotron RNNT & Parakeet TDT in offline (`transcribe`) mode (TDT appears ability-inherent). If RNNT gives none: catalog marks this engine `word-ts: aligner`.
- Diarization:

```
nemo-speech diarize {prep_wav} --model {diar_gguf} --json --device {vulkan:0|cuda:0|cpu}
```

- `SPIKE S2` — confirm exact verb/args/output schema turns[{start,end,speaker}] and max-speaker clamping for both `nemotron-3` (8) and `sortformer 4spk-v2.1` (4). Config params (chunk_len/fifo_len/spkcache…/right context) set ALL to the **offline** "very-high-latency" profile from the model card for best quality (Nemotron-3: `spkcache 264, fifo 40, chunk 340, right 40, update 300`) — the CLI may already default to it; freeze in args.py after S2.

### 3.4 Qwen3-ASR (in-process, transformers ≥5.13)

```python
from transformers import AutoProcessor, AutoModelForMultimodalLM
proc = AutoProcessor.from_pretrained(model_dir) ; model = AutoModelForMultimodalLM.from_pretrained(model_dir, dtype="auto" bf16/cuda/cpu)
inputs = proc.apply_transcription_request(audio=wav_path, language=None|locale → model's language names)
out = model.generate(**{k: v.to(dev)}, max_new_tokens=4096)
parsed = proc.decode(out_ids, return_format="parsed")[0]   # dict: language, transcription
```

- Long audio: the processor (`lingua`) handles up to? Unclear → own chunking: split wav at silence-ish boundaries every **240 s with 1 s overlap**, transcribe chunks, stitch segments (offset += chunk start; drop <0.3 s overlap dup), report combined `language`.
- Lexical mapping `translation`/`language auto` per model card list.
- **Word timestamps**: NOT natively. When `word_timestamps` or separate `words` required AND `qwen3-aligner` model installed → run align pass (§ 3.6). Else auto-error via gate check (error message leads to engines page).

### 3.5 pyannote.audio (in-process)

```python
from pyannote.audio import Pipeline; from pyannote.audio.pipelines.utils.hook import ProgressHook
pl = Pipeline.from_pretrained(model_id, token=token) ; pl.to(torch.device(device all))  # cuda|cpu
with ProgressHook() as hook:
    out = pl(prep_wav, min_speakers=? (int only), max_speakers=?, hook=hook)
turns = [(turn.start, turn.end, int(label.split("_")[-1])) for turn, _, label in out.speaker_diarization]
```

- Percent progress via hook: `(completion fraction, steps)`.
- `speaker_count` int maps to min=max=n; "auto" leaves both unset.
- `pyannote-community-1` uses 30 ms frames and torchcodec under the hood — ffmpeg is in the image (hence prep.wav remains cheap).
- `word_timestamps` irrelevant.

### 3.6 Qwen3-ForcedAligner (in-process, optional engine)

- Applied when: settings `word_timestamps=true` and the chosen ASR model's catalog capability is `aligner` (Nemotron 3.5 and Qwen-ASR).
- Inputs: wav + CURRENT merged transcript text (ASR output) restricted to alignable segments: chunk the (segment, audio) into ≤300 s windows (5-min model limit), 1-s overlap discarded.
- Output per window: word → (start, end); assemble full IR words.
- Languages (11): zh, en, yue, fr, de, it, ja, ko, pt, ru, es. If transcript language ∉ list or auto-detected unknown → warn toast; words stay empty and word-based features fall back to segment times (UI notified, not silent).

## 4. Merge & post-processing

### 4.1 Word→speaker merge (normative algorithm)

```
sort turns by start; sort words by start.
for each segment:
    words = segment.words or [synthetic mid (start+end)/2]
    for each word:
        mid = (w.start + w.end)/2 when w has times else segment mid
        cand = turns where turn.start <= mid <= turn.end
        best = cand with max overlap length; tie → earliest start
        w.speaker = best.speaker; else None → segment.speaker inherits from first
    segment.speaker = majority of words' speakers (ties → earliest)
    segment.overlap = any word whose mid ∈ 2+ turns   (v1: no special marking; info only for JSON)
map labels: label(i) = f"Speaker {i+1}" (arrival order = speaker index ascending by first start)
```

- turn gaps: optional `pauses` pass — between consecutive words `w1`,`w2`, when `w2.start - w1.end >= pause_min_s` a `Pause(start=gap_start, seconds=gap_len)` is appended to IR field `pauses: [Pause]` (consumed by TXT/HTML renderers).
- Whole segment without word times: per-segment assignment rule mid-in-turn; merge diar turns first, then segments; segment texts concatenated when all-words-of-only-one-turn.

### 4.2 Timestamps rules by `timestamps_mode`

- `none` — raw text only (TXT mode strips all; SRT/VTT/timed text impossible → gate rejected in settings validation: emit toast always).
- `interval` — insert `[hh:mm:ss]` at the first word at or after every N seconds of media time (not per segment), also at each speaker turn start. A long segment is split into paragraphs at those word boundaries so a one-segment NeMo transcript still gets multiple markers.
- `turns` — timestamp only at turn starts.

## 5. Output renderers (each takes `MergedSegment[]`, `pauses`, settings, engine meta)

Naming: `out_basename` else `media_id`+suffix when clip or collisions (`_clip`)? ALWAYS `{basename}[.{suffix}].{ext}` — suffix from `start_at/end_at` clip (`-clip0102-0105` style mm-ss suffix separately: `-clip-{startAT(format mmss)}-{endMMss}`); jobs never collide due to per-job dir; final visible names name stable = `{basename}.{ext}` even for clip jobs UNLESS same basename already exists in that job → `-clip` suffix appended.

### 5.1 TXT

- One paragraph per segment. In interval mode, a segment with word times is split into multiple paragraphs at interval markers (§4.2).
- Turn boundaries get a blank line + `Speaker N:` prefix when diarization is used and the file has more than one speaker.
- `timestamps_mode != none` → `[hh:mm:ss]` per §4.2.
- `mark_pauses` → one dot per second of pause in brackets: `( .)` per second, capped at 5 dots + `(+{N} s)` for longer.
- Encoding UTF-8, `\n` LF line endings.

### 5.2 HTML

Jinja2 template `out/transcript.html.j2` — self-contained (inline CSS, no remote assets), clean and print friendly:
- Header block: title (media name), job meta (date ISO, engine, model, device, language, duration, diarizer, version row "WrittenSound vX").
- Body per speaker turn: `<h3>Speaker N <span>[hh:mm:ss]</span></h3>` (when diarization produced more than one speaker; turn timestamps follow § 4.2 rules), then `<p>` per segment or interval chunk; line numbers in a left gutter via CSS counters. A single-speaker transcript has no `Speaker 1` heading.
- Interjection/overlap: reserved class, v1 simply bold-italicize `//…//` when `overlap` segments get wrapped in `<span class=overlap>` (no "//"; separated: Informational note, colored hover) — final UX text decided with preview.
- Pause dots inline as `( .)` styled muted.
- `<meta charset>`; reflect dark/bright print.

### 5.3 SRT / 5.4 VTT / 5.5 Timed text

- File names: `{base}.srt`, `{base}.vtt`, `{base}.timed.txt`.
- Word-less source: cues = segments.
- Word-level present: **caption cue grouping** rule (normative):
  1. start new cue at : speaker turn change OR gap > 1.0 s OR word count ≥ 12 OR cue span ≥ 5.0 s — whichever comes first;
  2. cue.start = first word .START (no rounding BEFORE 1 ms), cue.end = last word end + up to +250 ms (bounded by next cue start − 10 ms);
  3. cue text = words joined by spaces; 1 escape handles special tokens.
- Speaker prefix: emit `Speaker N: ` only when the file contains more than one speaker. A single-speaker caption has no prefix; do not repeat `Speaker 1:` on every cue.
- SRT: indices 1-based `HH:MM:SS,mmm`. VTT: `WEBVTT`, optional `NOTE WrittenSound` header, `HH:MM:SS.mmm`. Cues sorted chronologically; overlap handled by interleaving chronological with correct absolute times, not clamping (SRT allows overlapping times in practice; VTT too; UI note suggests care).
- Timed text: one cue per line, exactly `[HH:MM:SS.mmm --> HH:MM:SS.mmm] text`; no SRT indices, no `WEBVTT` header, no blank line between cues. Reuse the SRT/VTT cue grouping and speaker-prefix rule. Treat it as a caption format: `timestamps_mode=none` rejects it.

### 5.6 JSON

```jsonc
{ "ir": TranscriptionResult-merged list, "diar": DiarResult|null (raw),
  "pauses": [Pause], "settings": JobSettings, "engine_meta": HardwareStatus excerpt,
  "written_sound": { "app": "WrittenSound", "version": "1.0.0" } }
```

Path: `{out}/ir.json` (also `ir.jsonl` rows per segment? No — single object).

## 6. Cancellation semantics per stage

- `ingest/prepare`: instant abort.
- `asr`: fw loop checks cancel between segments; subprocesses get SIGTERM → if >5 s alive then SIGKILL.
- `diarize`: per-region hooks; same kill rules.
- `align/render`: negligible.

## 7. Progress mapping (normalized per stage)

- Each adapter exposes `progress_cb(stage_frac 0..1, msg)`, coalesced into worker: `total = sum(stage_weight * frac)`; SSE ≤2 Hz.
- Heuristic fallback: elapsed-time estimator `frac = min(0.99, t_elapsed / (0.18 * audio_seconds))` for engines without hooks (nemo CLI without `--verbose`? SPIKE S2 adjust when parsing available), progress values stay monotonic.

## 8. Logging contract

Per job: `logs/pipeline.log` (human-readable stage events), `logs/engine-asr.log` (stdout+stderr verbatim), `logs/engine-diar.log` (when used). Level from config; DEBUG exposes a UI "show engine log" that streams these files (end-point `GET /api/jobs/{id}/logs/{name}` — Range-capable `FileResponse`).

## 9. Determinism / idempotency

- Re-render after failure never requires re-ASR: worker keeps `ir.json` in job dir after `asr`/`diarize` so `retry` from `render` re-uses it.
- All floats rounded on output only (2 decimals seconds, 3 decimals SRT ms).
- Same settings + same model + same media → byte-comparable outputs except timestamp-bearing values within 1 ms (engines non-deterministic across devices; expected).
