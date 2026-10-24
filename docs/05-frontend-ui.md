# WrittenSound — Frontend / UI Specification

Plain server-rendered pages (Jinja2) upgraded with **htmx** for partial swaps and one `EventSource` connection for live updates. A single hand-rolled `app.js` (< ~500 lines, no framework, no build step) covers upload sequencing, drag-and-drop folder walk, and settings-modal glue. Vendored assets only: `htmx.min.js` (~14 KB), `htmx-ext-sse`, open font optional (system stack). Distinctly **not fancy** — noScribe-like minimalism: flat cards, one accent color, generous spacing, system fonts.

Brand constant: `WRITTENSOUND` wordmark = text "WrittenSound" + no icon (keep CSS-only dot accent).

---

## 1. Pages

| Path | File | Purpose |
|---|---|---|
| `/` | `templates/queue.html` | Everything-day page: uploader + settings + queue |
| `/engines` | `templates/engines.html` | Engine cards, model downloads, toggles, pyannote gating pane |
| `/settings` | `templates/settings.html` | HF token, defaults, retention, cleanup |
| About box | modal on `/engines` footer | version, licenses, links |

One `EventSource` at `/api/events` drives all three pages (offline-safe: on reconnect backfill via `snapshot` payload; subscribe page-template independent). A top bar renders `HardwareStatus` chips: `CPU ok · Vulkan ok · CUDA —` and an info-dot with hover for `HardwareStatus.notes`.

## 2. Queue page layout (single column, top-flow)

```
┌ topbar ─────────────────────────────────────────────────────┐
│ WrittenSound           [status chips]   [Engines] [Settings]│
├ upload zone ────────────────────────────────────────────────┤
│ 777px drop area: "Drag & drop files or folders here —      │
│   or [Choose files] [Choose folder]"                        │
│   → on drop: file list panel (name, size, path-from-root)   │
│      [Remove] per row  ·  total size  ·  [Add N to queue]   │
├ settings card ("Applies to files added next") ──────────────┤
│   (see §3; two columns on ≥900px, one column below)          │
├ queue table ────────────────────────────────────────────────┤
│ Table: [⃟] Title | Status+stage | Engine/model | Device |    │
│        Timestamps | Diar | Formats | Progress bar | Log | ⋮ │
│   row expands (click) → result files inline when done        │
└─────────────────────────────────────────────────────────────┘
```

**Queue behaviors (normative):**
- `Add N to queue` → `POST /api/uploads` one file at a time (sequential client-driven loop; shows per-file progress bar in the file list panel), then `POST /api/jobs` with `items[]` (the file's scoped settings if any, else the settings card).
- After upload completes successfully, staging rows clear automatically.
- **Per-job settings**: the ⋮ menu on a pending row → `Edit settings` opens the *same settings form* as a modal pre-populated from the job → `Save` `PATCH /api/jobs/{id}`; buttons inside modal: `[Apply below]` = adds `.apply_below=true` → PATCH all subsequent pending jobs to identical settings (server expands).  Additional row action `Duplicate settings to all` = same but `apply_below=true` semantics applied to every pending job.
- Row actions conform to § table in 00 §6 parity: `⃓` Cancel/Remove, `⟲` Retry, `↓` outputs appear as file links when succeeded; failed rows show error strip + `Fallback to CPU` when `error.code == unsupported_device`.
- Buttons disable while uploader is mid-flight for the same client-batch (reuse of staging pending rows).

## 3. Settings form (JobSettings; server-rendered fragment `partials/settings_form.html`)

Fragment re-render trigger = any dropdown/source change `hx-get="/partials/settings-form?engine=X&…"` (auto-refreshes only the form and warning strip — no full reload). Fragment state, in this widget order:

1. **Model**: select grouped by engine family: group label = engine display name + `{device chips}`. Groups sorted: whisper.cpp, faster-whisper, NeMo, Qwen3. Options: `{installed? state}`, otherwise "not installed — will download after queue". *Local-file custom* row at the bottom: path input (`/data/models/...` sanitized) + "Scan" button (opens page `/engines` on errors).
2. **Device**: radio `Auto` / `CPU` / `CUDA` / `Vulkan` (rendered from HardwareStatus availability; disabled+tooltip when impossible per 02 §4 rules 1–2).
3. **Language**: select `auto` + code list (model's languages from catalog `languages`); greyed `auto` when unsupported; after selection the code→locale mapping renders from catalog (`en-US` for nemo etc.). Non-native codes or unsupported field → **inline warning text** (`02 §4 rule 7`).
4. **Diarization**: segmented control: `Off | Nemotron | pyannote` — suboptions slide-in: model select (model count n), speakers `Auto | 2..8` (grey `>4` for nemo-diar-sortformer; nullable for nemo; `5..8` only when nemo-diar-nemotron-3).
5. **Output details**: 
   - Timestamps mode select: `None (raw text) | Timestamps [hh:mm:ss] — every N s + turn starts | Turn starts only`.
   - `N` seconds number input (10..3600, vis only when interval).
   - checkbox `Word-level timestamps` (grayed when capability `none` or aligner missing; tooltip explains).
   - checkbox `Mark pauses (one dot per second)` + `min pause s` number (0.5..10).
   - checkbox `Verbatim (keep um/uh)` — grayed for non-whisper-family.
   - Start / Stop clip inputs (`hh:mm:ss`, optional).
6. **Formats**: checkbox row: HTML / TXT / SRT / VTT / JSON (SRT/VTT grayed when `timestamps_mode == none`).
7. **Sticky warnings strip** (top of card): bullet list of auto-computed warnings per job settings (from rules table `02 § 4` + §5 below). Empty state = no strip.

Form values use plain field names matching `JobSettings` (`asr_model_id`, `diar`, …). The **Settings card on the queue page** stores into `/api/settings` key `defaults` (via `PUT /api/settings`), so on reload last-used values come back.

## 4. Drag-and-drop / upload mechanics (`static/js/upload.js`, exact behaviors)

1. `dropzone.ondragover` → visual state; `drop` → `DataTransfer.items` walk with `webkitGetAsEntry()` recursion (`entry.isDirectory → createReader, readEntries loop` handle `readEntries` chunking caveat 100 max/read). Firefox/Safari fallback: `files` only (note shaken in UI text "Folder picking needs Chrome/Edge").
2. `[Choose folder]` button = hidden `input[type=file][webkitdirectory][multiple]`; `[Choose files]` normal input.
3. Flat list assembled client-side with three fields: file handle, `webkitRelativePath || name`, `byte size`. Sort Folder-then-name (natural).
4. Dedupe by full relative name within the batch. Existing staged name collision (same target file already uploaded/unconsumed) → number suffix `_2`, confirmed visually.
5. Progress: files >64 MB get chunked fetch? **No** — single `fetch POST /api/uploads` sequential per file; bar per-file from `upload.onprogress` (fetch lacks it) → use `XMLHttpRequest` (gives progress events) exclusively.
6. Server-side post steps for "ready/failed" reflect in the files panel via the same SSE `upload` reuse (event name `upload` on `/api/events`, payload `UploadRow`).
7. Add-to-queue button POSTs once ALL staged files reached state ready; errors keep rows red with retry link (re-POST same file).
8. Client disk-quota guard: if a file >  env `WS_MAX_UPLOAD` (default 16 GB) → greyed with reason (config-overridable at boot: `WS_MAX_UPLOAD_BYTES`).

## 5. Engines page (`templates/engines.html`)

Grid of cards, one per engine (order = registry order):
- Header: engine name + toggle (disabled state grey/pill "disabled").
- Device chips from HardwareStatus ∩ engine support (e.g. NVIDIA `CU` chip pops only in cuda image).
- Models table rows: name, meta badges (langs, ~size, word-ts badge: `words+` `aligner` `-`), installed state, actions `Install / Cancel / Remove`.
- `pyannote` card extra pane when not ready: numbered instructions with **links to accept each HF repo page** + token pointer. Refresh hook on `model` SSE event.
- `qwen3-aligner` card marked "Word-timestamp fallback for Nemotron/Qwen3".
- Footer note linking back to catalog → "add a model manually" (open engines dir path note).

## 6. Settings page (`templates/settings.html`)

- HF token: password-type field + hint (full value stored). Show masked current value; save triggers `PUT /api/settings`; **warning frame: token is stored UNSAFE-plain in `app-config.json` — the file/disk/data dir must remain private**.
- Defaults panel = same settings form fragment (binds `defaults` on save).
- Housekeeping: retention days, cleanup toggles, "Clean staging now" button.
- App info: version, image variant, source license links.

## 7. Styling contract

- htmx + SSE extensions vendored; no CDN.
- CSS custom props: `--bg #fafafa; --fg #111; --muted #667; --accent #b45309 (noScribe-ish orange); --ok #166534; --warn #b45309; --err #991b1b; --card #fff; --chip #eee; radius 10px; shadow minimal`.
- One media query breakpoint (900px) collapses to single column; queue table scrolls horizontally on narrow.
- All interactive elements show `title` tooltips on disabled state (why grey). Focus-visible outline.

## 8. Accessibility & error surfaces

- Every control labelled (no bare gif icons; text or aria-label).
- Errors presented at the annoyance-free top strip of active card; toasts stack bottom-right, 6 s auto-dismiss; persistent warnings (HW) only in topbar chip tooltip.
- The queue must be operable without JS for basic add-file via form POST fallback (`<noscript>` plain upload form to `/api/uploads/legacy`).

## 9. Out of scope (locked)

- No dark mode toggle (system-bright only). No theming. No i18n UI (English only). No account/session analytics. No router-level history beyond `#settings`.
