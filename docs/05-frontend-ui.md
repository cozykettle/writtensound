# WrittenSound — Frontend / UI Specification

Plain server-rendered pages (Jinja2) upgraded with **htmx** for partial swaps and one `EventSource` connection for live updates. A single hand-rolled `app.js` (no framework, no build step) covers upload sequencing, drag-and-drop folder walk, and settings-modal glue. Vendored assets only: `htmx.min.js` (~14 KB), `htmx-ext-sse`, open font optional (system stack). Distinctly **not fancy** — deliberate minimalism: flat cards, one accent color, compact spacing, system fonts.

Brand constant: `WRITTENSOUND` wordmark = text "WrittenSound" + no icon (keep CSS-only dot accent).

---

## 1. Pages

| Path | File | Purpose |
|---|---|---|
| `/` | `templates/queue.html` | Add jobs, per-batch audio settings, and the job queue |
| `/engines` | `templates/engines.html` | Model selection (`Use`), engine toggles, model downloads, licenses |
| `/settings` | `templates/settings.html` | HF token, default job settings, retention, machine, paths |
| About box | card on `/engines` | version, licenses, data paths |

Model selection lives on **Engines** (per-model `Use`) and **Settings** (defaults form). The queue page shows the configured model as a read-only strip and does not repeat the model picker. One `EventSource` at `/api/events` drives all pages; the top bar shows `CPU`, `Vulkan`, and `CUDA` chips plus `Refresh`.

## 2. Queue page layout (single column, top-flow)

```
┌ topbar ─────────────────────────────────────────────────────┐
│ WrittenSound          [hardware chips]   Queue Engines Settings │
├ page head ───────────────────────────────────────────────────┤
│ Queue                                         [N jobs] [Live] │
├ Add jobs card ───────────────────────────────────────────────┤
│ Add jobs                 MODEL whisper.cpp · Large v3 q8_0    │
│                          [not installed] [Change]              │
│ ┌ drop row ─────────────────────────────────────────────────┐ │
│ │ Drop files or a folder here         [Choose files] [Choose folder] │
│ │ Audio or video. Up to 16 GB per file.                    │ │
│ └───────────────────────────────────────────────────────────┘ │
│ [warnings strip, only when settings need attention]            │
│ Device | Language | Speaker labels                            │
│ Timestamp style | Interval | Pause | timing and filler checks  │
│ Clip start | Clip stop | Output files                         │
│ Advanced (file name prefix)                                    │
│ [staged file list, when files are staged]                      │
│ N files · size · M uploaded       [Save as defaults] [Add N]  │
├ queue table ──────────────────────────────────────────────────┤
│ File | Status | Model | Output | Actions                      │
│   row click → detail row: outputs, logs, error, settings       │
└───────────────────────────────────────────────────────────────┘
```

The model strip is read-only on this page. `Change` opens `/engines`; the queue form carries `asr_model_id` as a hidden input so every queued job has a complete `JobSettings`.

**Queue behaviors (normative):**
- `Add N to queue` uploads pending files one at a time (sequential client loop with a per-file progress bar), then `POST /api/jobs` with `items[]` and the settings from the form. On success the staged list clears and the page scrolls to the queue.
- The Add button is disabled while no files are staged, during upload, or when a staged file failed. Its label is `Add to queue` with no files, `Add N to queue` with files, and `Uploading…` during upload.
- **Per-job settings**: pending rows have `Edit` → the same settings form in the modal (`PATCH /api/jobs/{id}`), plus `Cancel` and `Delete`. Failed rows have `Retry`, optional `CPU retry` for `unsupported_device`, and `Delete`. Other rows have `Delete`; running rows have `Cancel`.
- Row click expands a detail row (outputs, logs, error, resolved settings, timestamps). The detail row spans all five columns.
- The queue table header is `File | Status | Model | Output | Actions`. Status shows an inline progress bar for running jobs and the duration for finished jobs. Model shows the display name and `engine · device`. Output shows format tags.
- Empty state: `No jobs in the queue.` with one hint line. No decorative illustration.
- The queue page must stay operable without JavaScript for a basic add-file via the `<noscript>` form (`POST /api/uploads/legacy`).

## 3. Settings form (JobSettings; server-rendered fragment `partials/settings_form.html`)

The same fragment renders in three modes:

| Mode | Where | Model field | Save control |
|---|---|---|---|
| `queue` | Add jobs card on `/` | hidden `asr_model_id` + read-only strip | `Add to queue`; footer `Save as defaults` |
| `defaults` | Settings page | model select | `Save defaults` |
| `job` | edit modal from a pending job | model select | `Save changes` / `Cancel` |

The form re-renders on any control change (`hx-trigger="change delay:180ms"`). Queue and defaults modes swap `#settings-card-body`; job mode swaps `#modal-body`. The response is the whole fragment, so warnings and dependent fields update together. The `Add to queue` button and staged-file panel stay outside the swapped node. Form values survive the swap because the server re-renders from the posted form, and the queue mode re-emits the hidden model value.

Controls, in order:
1. **Model** (defaults/job only): select grouped by engine. Option text shows model name, size, word-timing capability (`words+` / `aligner`), `EN only`, and `not installed` state. The hint shows engine runtime, notes, and an aligner install link when needed.
2. **Device**: segmented `Auto | CPU | CUDA | Vulkan`. A device is disabled with a title when the engine has no backend for it or the machine does not expose it. `Auto` hint names the chosen device. A non-auto device hint says a failure stops the job and does not retry on CPU by itself.
3. **Language**: `Auto-detect` plus catalog language codes. Auto is disabled when the model has no language detection. The hint shows the model language range.
4. **Speaker labels**: segmented `Off | Nemotron | pyannote`. When on, a speaker-model select and `Auto count | 2…8 speakers` select appear. Counts above the model limit are disabled with a title. The hint explains that `Off` writes no speaker names.
5. **Timestamp style**: `None (plain text) | Every N seconds and each turn | Each turn`. `Interval (s)` is enabled only for the interval mode. `Pause (s)` is enabled only when word timing exists.
   - `Word timestamps` (disabled without word timing; title explains)
   - `Mark pauses (one dot per second)` (needs word timing)
   - `Keep fillers (um, uh)` (whisper-family only)
6. **Clip start / Clip stop**: optional `hh:mm:ss` inputs; blank on both means the whole file.
7. **Output files**: `HTML | TXT | SRT | VTT | JSON`. SRT and VTT require a timestamp style.
8. **Advanced**: `File name prefix`; blank uses the audio file name.
9. **Warnings strip** at the top of the form (`role="status"`): auto-computed warnings from the settings rules (`02 §4`) and missing model/runtime state. The strip is absent when there is nothing to report.

All labels and helper text use ASD-STE100 Simplified Technical English. Micro-labels are uppercase mono; control text and helper text are sentence case.

The queue form values are the settings for the next batch. `Save as defaults` sends `PUT /api/settings` with `{defaults: …}`. The Settings page saves the same `defaults` object. The job modal sends `PATCH /api/jobs/{id}` with the full `JobSettings`.

## 4. Drag-and-drop / upload mechanics (`static/upload.js`, exact behaviors)

1. `dragenter`/`dragover` add the drag state; `drop` walks `DataTransfer.items` with `webkitGetAsEntry()` and recurses into directories (`createReader` reads entries in batches). Browsers without entry support fall back to `dataTransfer.files`. Folder selection needs Chrome or Edge; the button title says so.
2. `Choose files` uses a normal hidden `input[type=file][multiple]`; `Choose folder` uses a hidden `input[webkitdirectory][multiple]`.
3. The staged list uses the file handle, `webkitRelativePath || name`, and byte size. Insertion order is the order the files arrive; there is no client-side sort.
4. Files are deduped inside one batch by `path + size + lastModified`. Re-adding the same file is ignored; removing a row from the list allows adding it again.
5. Uploads are sequential `XMLHttpRequest` POSTs to `/api/uploads` with `X-Client-Id` and an encoded `X-Relative-Path`. `upload.onprogress` drives the per-file bar; there is no chunking.
6. The add button uploads any pending files when pressed, then waits for every file to be ready. A failed upload keeps its row red, shows the error in a toast, and keeps the add button disabled until the row is removed or the batch is cleared.
7. `Add N to queue` then POSTs `/api/jobs` once with `items[]` and the settings from `#job-settings-form`. On success the staged list clears, the file panel hides, and the page scrolls to the queue.
8. The client checks the file extension against the audio/video allowlist and the size against `WS_MAX_UPLOAD` (default 16 GB, from `data-max-upload`). Skipped files produce one toast.
9. `Clear` empties the batch; the per-row `×` removes one file. Both are disabled during upload.

## 5. Engines page (`templates/engines.html`)

Page head: `Engines` + one-line explanation. Actions: `Scan model folder`, `Re-probe hardware`.

Grid of cards, one per engine: two columns at ≥1100px, one column below. Card order = registry order.
- Header: engine name, runtime-present or runtime-missing pill, runtime and license line, and an on/off switch. A disabled engine gets a `disabled` pill and reduced opacity; it does not appear in the queue.
- Device chips: `CPU | Vulkan | CUDA` from hardware ∩ engine support. A chip is green when the device is live on this machine, otherwise quiet with a title.
- Models table: `Model | Details | State | Actions`.
  - Model: display name + catalog id.
  - Details: size, role (`speaker labels`, `aligner`), word timing (`words+`, `aligner`), `EN`, `gated`, and language range.
  - State: `absent | downloading | partial | ready | error`, with a mini progress bar while downloading and a short error when partial.
  - Actions: `Install` / `Resume` / `Cancel` / `Remove`, plus `Use` for ASR models. The current default row is highlighted and shows a `default` pill instead of `Use`.
- `Use` sends `PUT /api/settings` with the current defaults plus the new `asr_model_id` and reloads the page. If the new engine does not support the current device, the device resets to `auto` and the toast says so. A disabled engine cannot be selected (toast: `Turn on <engine> before you use its models.`).
- Model state updates over SSE preserve the `Use` and `default` actions (the client rebuilds the action cell for each state change).
- `pyannote` card extra pane when gated: numbered steps with links to both HF license pages and a pointer to the token on Settings.
- `qwen3-aligner` card is marked as the word-timestamp fallback.
- Demo engine card has a footnote: fixed sample transcript, does not read audio.
- About card at the bottom: version, license, data directory, default model, hardware summary, model folder, and engine licenses.

## 6. Settings page (`templates/settings.html`)

Page head: `Settings` + one-line explanation. The data directory is shown in the Paths card, not as a header pill.

Layout: one column below 1100px; two columns for the lower cards at wider sizes.
- **Hugging Face token**: password field, masked current value, `Save token`, and `Remove the stored token` when a token is stored. Hint: only gated files need a token; the token is stored as plain text in `app-config.json`; keep the data directory private.
- **Defaults for new jobs**: the same settings fragment in `defaults` mode, including the model select. `Save defaults` writes the `defaults` object. This is the second place (besides Engines `Use`) where the queue model is chosen.
- **Retention**: `Keep finished jobs for [N] days`; `0` keeps all. `Delete staged files after a job is created` checkbox. `Save retention` and `Clean staging now`.
- **This machine**: image variant, CPU cores, RAM, Vulkan/CUDA availability, device list, hardware notes, `Re-probe`.
- **Paths**: data, jobs, models, HF cache, database, catalog path. Backup hint: back up the data directory only.

## 7. Styling contract

- htmx + SSE extensions vendored; no CDN, no framework, no build step.
- Warm paper palette: `--paper #f6efe3`, `--paper-2 #fdf8ef`, `--ink #211b15`, `--muted #6f6355`, `--line #e0d1bb`, `--accent #b5430d`, `--accent-2 #d97706`, plus green/red/blue/gold state tokens. One orange accent, quiet state colors. `--muted` passes 4.5:1 on paper, card, tinted status, and warning surfaces.
- Type: serif headings (`Iowan Old Style` / Palatino fallback), sans body (`Segoe UI` / system), mono micro-labels (`Cascadia Code` / Consolas). Base 14px/1.5. Functional text is at least 11px (micro-labels, tags, table headers, timestamps); helper text is 12px. Disabled controls may be dimmed but keep a `title`.
- Shape: 10px card radius, 6–8px controls, 1px warm borders, minimal shadow. No gradients as decoration beyond the paper wash and subtle card fill.
- Density: page starts with a single-line `Page` heading; no hero, no steps band, no decorative glyphs or empty-state art. Cards use 14–16px padding, table rows 8px vertical padding, 9–12px grid gaps.
- Layout: max width 1200px; single-column content flow. Breakpoints:
  - `1100px`: settings grid 2 columns; engine grid 1 column; lower Settings cards 1 column.
  - `800px`: top bar wraps; controls stack; page head aligns top.
  - `700px`: engine model rows switch to a two-area layout (name/actions, details/state).
  - `620px`: queue table scrolls horizontally; dropzone stacks; metadata grid 1 column; toasts full width.
  - `520px`: settings grid 1 column.
- All interactive elements show a `title` when disabled (why grey). Focus-visible outline uses the accent. `prefers-reduced-motion` disables transitions.
- No dark mode, no theme switch, no marketing copy. UI labels and interface text use ASD-STE100 Simplified Technical English; helper text may stay descriptive.

## 8. Accessibility & error surfaces

- Every control has a visible label or `aria-label`; disabled controls carry a `title` that explains why.
- Settings warnings render at the top of the form (`role="status"`). Toasts stack bottom-right and auto-dismiss; hardware notes appear in the top-bar chip title and the Settings `This machine` card.
- Status uses text in addition to color (`queued`, `running`, `done`, `failed`, `cancelled`, `interrupted`); format tags and model state badges are text.
- The `Live` chip reports Live/Offline and explains reconnection in its title.
- The queue should remain operable without JS for basic add-file via the `<noscript>` form (`POST /api/uploads/legacy`). The settings form is server-rendered; htmx only refreshes it.
- Focus-visible outlines are retained on buttons, links, inputs, and the dropzone.

## 9. Out of scope (locked)

- No dark mode toggle (system-bright only). No theming. No i18n UI (English only). No account/session analytics. No router-level history beyond `#settings`.
