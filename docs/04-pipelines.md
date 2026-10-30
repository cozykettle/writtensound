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
        vad_parameters={"min_silence_duration_ms": 500},
        condition_on_previous_text=False, temperature=0.0,
        initial_prompt=INITIAL_PROMPT_VERBATIM if settings.verbatim else None,
        log_progress=True)   # sequential (batched OFF in v1); 1.2.1 has no progress_hook
```

`info.language[_probability]`, `info.duration`. Words include `prob` (0..1). No diarization integration. The adapter derives progress from the number of emitted segments because 1.2.1 has no `progress_hook`; the job log records the heuristic.

Options not exposed in UI v1: `beam_size=1` fixed default, `patience=1.0`, `suppress_tokens` default, `chunk_length=30`.
`SPIKE S3` — faster-whisper 1.2.1 accepts the singular `temperature` (float
or list), not `temperatures`, and has no `progress_hook` parameter. The adapter
passes `temperature=0.0` and uses segment-count heuristic progress when the
hook is absent.

### 3.2 whisper.cpp (subprocess `whisper-cli`)

```
whisper-cli -m {ggml_path} -f {prep_wav} \
  --language {auto|code} \
  --output-json-full --output-file {job_dir}/out/{base}.full \
  --progress    # prints percent lines that the adapter parses (v1.9.5 name)
  [GPU flags per device/kernel: --device 0 / --no-gpu]
  [if verbatim: --prompt "…uh, um —"]
```

- stderr captured verbatim into logs; stdout progress lines parsed for the `job` SSE.
- `--output-json-full` already enables token timestamps in v1.9.5. Do **not**
  add `--max-len`: it wraps tokens into separate segments and destroys the
  engine's paragraph boundaries.
- Adapter converts the full JSON (segments[]→tokens[] with offsets in ms) into
  `Segment/Word` (offsets/1000). Words carry `p` (probability).
- whisper.cpp `tokens[]` are subword pieces, not words. Skip special tokens
  (`[_TT_*]`, `[_BEG_]`, ids at/above EOT) and merge adjacent pieces into words
  at leading-space boundaries. Otherwise captions contain timestamp markers and
  text like `un imagin able`.
- DTW metadata: whisper.cpp requires per-token timestamps enabled AND a `dtw` alignment file matching the mels model on some modes; catalog sets `word-ts` accordingly per S1 finding (input tokens time offsets exist natively with DTW; else fall back to `aligner` capability when qwen3-aligner installed, else `no`).
- **Vulkan device selection**: `--device {idx}` (or `--no-gpu` for CPU); enforce via the resolved device index.

### 3.3 NeMo-Speech.cpp (subprocess `nemo-speech`)

```
nemo-speech transcribe {prep_wav} --model {gguf_path} --json \
  --device {vulkan:0|cuda:0|cpu} --language {auto|code-locale}
```

- `--language` maps UI code → catalog locale code (en-US, de-DE, …); `auto` allowed. JSON structure verified by S2 (fields to be mapped into `Segment/Word`).
- Parakeet TDT returns top-level `words[]` without `segments[]`. The adapter maps those words into one `Segment` with its `words` list intact and uses the payload `text` when present; it must not fall back to a word-less `0 → duration` segment. Renderers group the preserved words into cues.
- NeMo-Speech.cpp times are seconds. A seconds value above 1000 (any clip over ~16.6 minutes) must not be mistaken for milliseconds; the parser only divides by 1000 when the value is implausibly large for seconds (`> 100000`).
- Long-audio limitation: NeMo-Speech.cpp splits offline input at quiet points when it exceeds the encoder's positional budget, and Parakeet TDT is offline-only. The upstream model card/community reports occasional dropped short utterances ("the model sometimes drops full sentences"); there is no ASR precision flag, only the GGUF quantization. This is a model/runtime limitation, not a settings bug, and WrittenSound must not silently substitute another engine. Use whisper.cpp for critical long-form work.
- `SPIKE S2` — whether per-token timestamps exist for Nemotron RNNT & Parakeet TDT in offline (`transcribe`) mode (TDT appears ability-inherent). If RNNT gives none: catalog marks this engine `word-ts: aligner`.
- Diarization:

```
nemo-speech diarize {prep_wav} --model {diar_gguf} --json --device {vulkan:0|cuda:0|cpu}
```

- `SPIKE S2` — confirm exact verb/args/output schema turns[{start,end,speaker}] and max-speaker clamping for both `nemotron-3` (8) and `sortformer 4spk-v2.1` (4). Config params (chunk_len/fifo_len/spkcache…/right context) set ALL to the **offline** "very-high-latency" profile from the model card for best quality (Nemotron-3: `spkcache 264, fifo 40, chunk 340, right 40, update 300`) — the CLI may already default to it; freeze in args.py after S2.

### 3.4 Qwen3-ASR (in-process, transformers 5.20)

```python
from transformers import AutoProcessor, AutoModelForMultimodalLM
proc = AutoProcessor.from_pretrained(model_dir)
dtype = torch.float32 if dev == "cpu" else (
    torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
)
model = AutoModelForMultimodalLM.from_pretrained(model_dir, dtype=dtype).to(dev).eval()
# prep.wav is 16 kHz mono s16le; read a window as float32 numpy samples
audio = qwen_audio.read_window(prep_wav, start, end)
request = proc.apply_transcription_request(audio=audio, language=None|code|name)
inputs = {k: v.to(dtype) if v.is_floating_point() else v for k, v in request.to(dev).items()}
out = model.generate(**inputs, max_new_tokens=4096)
generated = out[:, inputs["input_ids"].shape[1]:]            # drop the prompt/prefill
parsed = proc.decode(generated, return_format="parsed")[0]   # language, transcription
```

- Audio is handed to the processor as a numpy array, never as a file path.
  `transformers.audio_utils.load_audio` short-circuits ndarrays, so the WAV
  path (and torchcodec's decoders) is not selected for `prep.wav`.
- **Dtype follows the device, not the checkpoint.** The Qwen snapshots declare
  bf16. CPU loads float32 because bf16 convolution is not portable across CPU
  kernels; CUDA uses bf16 when the GPU reports support and float16 otherwise
  (Turing/sm_75, e.g. T600, cannot execute bf16). Floating inputs are cast to
  the model dtype; token ids and masks stay integer. Loading a bf16 checkpoint
  and feeding float32 features fails with `Input type (float) and bias type
  (c10::BFloat16) should be the same`.
- Language accepts ISO codes or full names; `auto` passes `None`. The catalog
  `qwen3` list is the supported 30-code set.
- The generated ids must be sliced by the request's prompt length. Decoding
  the full sequence puts the chat prompt before `<asr_text>`, which makes the
  parser read the language from the prompt instead of the model output.
- Long audio: split `prep.wav` every **240 s with 1 s overlap**, transcribe
  each window, stitch segments (`start`/`end` are window times). Qwen returns
  no word times, so the overlap is not text-deduplicated; a boundary word may
  repeat rather than be clipped.
- **Chat template**: the snapshot must contain `chat_template.jinja`; it is a
  separate file in the Qwen repos. The catalog fetches `*.jinja`, and the
  adapter restores it for older local snapshots before calling
  `apply_transcription_request`. A missing template is a loud `engine_crash`,
  never a silent prompt fallback.
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
- Inputs: wav + CURRENT merged transcript text (ASR output), one segment at a time. Audio is sliced from `prep.wav` as a float32 numpy window. Segments at or under the 5-minute model limit go in one pass; longer segments are split into consecutive windows of ≤300 s, with the segment text split at word/character boundaries in proportion to the window durations. The job log names the segment and window count, so the approximation is visible.
- API (transformers 5.20, verified against the pinned release):

```python
from transformers import AutoProcessor, AutoModelForTokenClassification
proc = AutoProcessor.from_pretrained(aligner_dir)
dtype = torch.float32 if dev == "cpu" else (
    torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
)
model = AutoModelForTokenClassification.from_pretrained(aligner_dir, dtype=dtype).to(dev).eval()
inputs, word_lists = proc.prepare_forced_aligner_inputs(
    audio=window, transcript=segment_text, language=code,
)
inputs = {k: v.to(dtype) if v.is_floating_point() else v for k, v in inputs.to(dev).items()}
with torch.inference_mode():
    logits = model(**inputs).logits
items = proc.decode_forced_alignment(
    logits=logits,
    input_ids=inputs["input_ids"],
    word_lists=word_lists,
    timestamp_token_id=model.config.timestamp_token_id,
)[0]  # [{text, start_time, end_time}] in seconds, relative to the window
```

- Output per window: word → (start, end); add the window start, clamp to the window, and assemble full IR words. Empty text segments keep `words = None`.
- The aligner uses the same device dtype rule as ASR (§ 3.4): float32 on CPU, bf16 on CUDA when supported, float16 otherwise. Cast floating inputs only.
- Languages (11): zh, en, yue, fr, de, it, ja, ko, pt, ru, es. Codes and full names are both accepted. If the transcript language is missing, `auto`, or outside that list, the aligner logs which language was refused and every segment keeps `words = None`; word-based features fall back to segment times (surfaced in the job log, never a hard job failure).
- Japanese/Korean splitting needs `nagisa==0.3.0` / `soynlp==0.0.493`; the image probe imports both.
- **Chat template**: when the aligner processor has no template, the loader repairs it from `chat_template.jinja` or fetches that one file for an older snapshot, exactly like the ASR path (§ 3.4). A missing template is a loud `engine_crash`.

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
- Header block: title (media name), job meta (date ISO, engine, model, device, language, duration, diarizer, version row "WrittenSound vX"). The meta grid lets each value wrap and shrink (`min-width: 0`, `overflow-wrap: anywhere`) so a full ISO date cannot overlap the next cell. Duration comes from `TranscriptionResult.duration_s`; when an engine omits it, the adapter probes `prep.wav` and stores the real media duration.
- Body per speaker turn: `<h3>Speaker N <span>[hh:mm:ss]</span></h3>` (when diarization produced more than one speaker; turn timestamps follow § 4.2 rules), then `<p>` per paragraph. HTML reflows short engine segments inside a turn into readable paragraphs: consecutive paragraphs merge until roughly 80 words or 560 characters, and a timestamp or pause marker starts a new paragraph. Interval and turn markers from § 4.2 are preserved. TXT, SRT, VTT, timed text, and JSON keep the unmerged segment boundaries. Line numbers stay in a left gutter via CSS counters. A single-speaker transcript has no `Speaker 1` heading.
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
