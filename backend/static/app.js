/* WrittenSound client: one EventSource, a few small renderers, no framework. */
(() => {
  "use strict";

  const WS = (window.WS = window.WS || {});
  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => Array.from(root.querySelectorAll(selector));
  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  const state = {
    jobs: Array.isArray(WS.jobs) ? WS.jobs : [],
    hardware: WS.hardware || {},
    modelState: {},
    engineEnabled: WS.engineEnabled || {},
    settings: WS.settings || null,
    currentJobId: null,
    expanded: new Set(),
    live: false,
    offlineToasted: false,
  };

  /* ---------------------------------------------------------------- utils */

  function el(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined && text !== null) node.textContent = String(text);
    return node;
  }

  function toast(message, level = "info", timeout = 6500) {
    const host = $("#toasts");
    if (!host) return;
    const node = el("div", `toast toast-${level}`);
    node.append(el("span", "", message));
    const close = el("button", "", "×");
    close.type = "button";
    close.setAttribute("aria-label", "Dismiss");
    close.addEventListener("click", () => node.remove());
    node.append(close);
    host.append(node);
    if (timeout > 0) {
      window.setTimeout(() => {
        node.style.opacity = "0";
        node.style.transform = "translateY(6px)";
        window.setTimeout(() => node.remove(), 180);
      }, timeout);
    }
  }

  function formatDate(iso) {
    if (!iso) return "—";
    const date = new Date(iso);
    if (Number.isNaN(date.getTime())) return iso;
    return date.toLocaleString(undefined, {
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
  }

  function formatBytes(value) {
    if (value === null || value === undefined) return "—";
    let size = Number(value);
    if (!Number.isFinite(size)) return "—";
    const units = ["B", "KB", "MB", "GB", "TB"];
    let index = 0;
    while (size >= 1024 && index < units.length - 1) {
      size /= 1024;
      index += 1;
    }
    return `${index === 0 ? size.toFixed(0) : size.toFixed(size >= 10 ? 1 : 2)} ${units[index]}`;
  }

  function formatDuration(seconds) {
    if (seconds === null || seconds === undefined) return "—";
    const value = Number(seconds);
    if (!Number.isFinite(value)) return "—";
    if (value > 0 && value < 1) return "<1s";
    const total = Math.max(0, Math.round(value));
    const h = Math.floor(total / 3600);
    const m = Math.floor((total % 3600) / 60);
    const s = total % 60;
    return h ? `${h}h ${String(m).padStart(2, "0")}m` : m ? `${m}m ${String(s).padStart(2, "0")}s` : `${s}s`;
  }

  function statusLabel(status) {
    return {
      pending: "queued",
      running: "running",
      succeeded: "done",
      failed: "failed",
      cancelled: "cancelled",
      interrupted: "interrupted",
    }[status] || status;
  }

  function stageLabel(stage) {
    return {
      ingest: "reading media",
      prepare: "preparing audio",
      asr: "transcribing",
      diarize: "finding speakers",
      align: "aligning words",
      render: "writing files",
    }[stage] || stage || "";
  }

  function percent(job) {
    if (job.status === "succeeded") return 100;
    const value = Number(job.progress || 0);
    return Math.max(0, Math.min(100, Math.round(value * 100)));
  }

  function apiError(payload, fallback = "Something went wrong.") {
    if (!payload) return fallback;
    if (typeof payload === "string") return payload;
    if (payload.message) return payload.message;
    if (payload.detail) return typeof payload.detail === "string" ? payload.detail : JSON.stringify(payload.detail);
    return fallback;
  }

  async function request(url, options = {}) {
    const response = await fetch(url, {
      headers: { "Content-Type": "application/json", ...(options.headers || {}) },
      ...options,
    });
    if (!response.ok) {
      let payload = null;
      try { payload = await response.json(); } catch { /* text or empty */ }
      if (!payload) {
        try { payload = await response.text(); } catch { /* ignore */ }
      }
      const error = new Error(apiError(payload, `HTTP ${response.status}`));
      error.status = response.status;
      error.payload = payload;
      throw error;
    }
    if (response.status === 204) return null;
    const kind = response.headers.get("content-type") || "";
    return kind.includes("application/json") ? response.json() : response.text();
  }

  function post(url, body) {
    return request(url, { method: "POST", body: body === undefined ? "{}" : JSON.stringify(body) });
  }

  /* ------------------------------------------------------------ hardware */

  function renderHardware(hardware) {
    if (!hardware) return;
    state.hardware = hardware;
    WS.hardware = hardware;
    const chips = $("#status-chips");
    if (!chips) return;
    const set = (slug, ok, text) => {
      const chip = chips.querySelector(`[data-device="${slug}"]`);
      if (!chip) return;
      chip.classList.toggle("chip-ok", Boolean(ok));
      chip.classList.toggle("chip-off", !ok);
      const label = chip.childNodes[chip.childNodes.length - 1];
      if (label && label.nodeType === Node.TEXT_NODE) label.textContent = ` ${text}`;
      else chip.append(document.createTextNode(` ${text}`));
    };
    set("vulkan", hardware.vulkan_ok, `Vulkan ${hardware.vulkan_ok ? "ok" : "—"}`);
    set("cuda", hardware.cuda_ok, `CUDA ${hardware.cuda_ok ? "ok" : "—"}`);
    const cpu = chips.querySelector('[data-device="cpu"]');
    if (cpu) {
      const label = cpu.childNodes[cpu.childNodes.length - 1];
      if (label && label.nodeType === Node.TEXT_NODE) label.textContent = ` CPU ${hardware.cpu_cores}c`;
    }
    if (hardware.notes && hardware.notes.length) chips.title = hardware.notes.join(" · ");
  }

  async function refreshHardware(button) {
    if (button) button.disabled = true;
    try {
      const hardware = await post("/api/status/refresh");
      renderHardware(hardware);
      toast("Hardware probe finished.", "ok", 3200);
    } catch (error) {
      toast(error.message, "error");
    } finally {
      if (button) button.disabled = false;
    }
  }

  /* --------------------------------------------------------------- queue */

  function actionButton(label, action, id, quiet = true) {
    const button = el("button", `btn btn-small ${quiet ? "btn-quiet" : ""}`, label);
    button.type = "button";
    button.dataset.action = action;
    button.dataset.jobId = id;
    return button;
  }

  function jobEngine(job) {
    const resolved = job.resolved || {};
    const engine = resolved.asr_engine || "";
    const model = resolved.asr_model || (job.settings && job.settings.asr_model_id) || "";
    return { engine, model };
  }

  function modelName(modelId) {
    const names = WS.modelNames || {};
    return names[modelId] || modelId || "—";
  }

  function formatTags(job) {
    const out = (job.settings && job.settings.output) || {};
    return Object.entries(out)
      .filter(([, value]) => value)
      .map(([key]) => (key === "timed_txt" ? "timed text" : key));
  }

  function buildRow(job) {
    const row = el("tr", `job-row status-${job.status}${job.status === "running" ? " is-running" : ""}`);
    row.dataset.jobId = job.id;
    row.title = "Open details";

    const titleCell = el("td", "col-title");
    titleCell.append(el("div", "job-title", job.title));
    titleCell.append(el("div", "job-sub", formatDate(job.created_at)));
    row.append(titleCell);

    const statusCell = el("td", "col-status");
    statusCell.append(el("span", `status-pill status-${job.status}`, statusLabel(job.status)));
    if (job.status === "running") {
      const wrap = el("div", "progress-wrap");
      const bar = el("div", "progress-bar");
      const fill = el("span");
      fill.style.width = `${percent(job)}%`;
      bar.append(fill);
      wrap.append(bar);
      const text = el("div", "progress-text");
      text.append(el("span", "", stageLabel(job.stage) || "working"));
      text.append(el("span", "", `${percent(job)}%`));
      wrap.append(text);
      statusCell.append(wrap);
    } else if (job.status === "pending") {
      statusCell.append(el("div", "job-sub", "waiting for the runner"));
    } else if (job.status === "succeeded" && job.duration_s) {
      statusCell.append(el("div", "job-sub", formatDuration(job.duration_s)));
    } else if (job.status === "failed" && job.error && job.error.code) {
      statusCell.append(el("div", "job-sub", job.error.code));
    }
    row.append(statusCell);

    const engineCell = el("td", "col-engine");
    const info = jobEngine(job);
    const resolved = job.resolved || {};
    const device = resolved.device || (job.settings && job.settings.device) || "auto";
    engineCell.append(el("div", "", modelName(info.model)));
    engineCell.append(el("span", "mono", `${info.engine || "—"} · ${device}`));
    row.append(engineCell);

    const formatCell = el("td", "col-format");
    const tags = el("div", "format-tags");
    for (const tag of formatTags(job)) tags.append(el("span", "tag", tag));
    formatCell.append(tags);
    row.append(formatCell);

    const actions = el("td", "col-actions");
    const holder = el("div", "actions");
    if (job.status === "pending") {
      holder.append(actionButton("Edit", "job-edit", job.id));
      holder.append(actionButton("Cancel", "job-cancel", job.id));
      holder.append(actionButton("Delete", "job-delete", job.id));
    } else if (job.status === "running") {
      holder.append(actionButton("Cancel", "job-cancel", job.id));
    } else if (job.status === "failed") {
      holder.append(actionButton("Retry", "job-retry", job.id));
      if (job.error && job.error.code === "unsupported_device") {
        holder.append(actionButton("CPU retry", "job-fallback", job.id));
      }
      holder.append(actionButton("Delete", "job-delete", job.id));
    } else if (job.status === "cancelled" || job.status === "interrupted") {
      holder.append(actionButton("Retry", "job-retry", job.id));
      holder.append(actionButton("Delete", "job-delete", job.id));
    } else {
      holder.append(actionButton("Delete", "job-delete", job.id));
    }
    actions.append(holder);
    row.append(actions);
    return row;
  }

  function outputFormat(name) {
    const lower = name.toLowerCase();
    const checks = [
      [".timed.txt", "Timed text", 4],
      [".html", "HTML", 0],
      [".txt", "TXT", 1],
      [".srt", "SRT", 2],
      [".vtt", "VTT", 3],
      [".json", "JSON", 5],
    ];
    for (const [suffix, label, rank] of checks) {
      if (!lower.endsWith(suffix)) continue;
      // ir.json is job-level, not <base>.json; group it with the transcript.
      const base = suffix === ".json" && lower === "ir.json" ? null : name.slice(0, -suffix.length);
      return { label, rank, base };
    }
    const dot = lower.lastIndexOf(".");
    return {
      label: dot > 0 ? name.slice(dot + 1).toUpperCase() : "File",
      rank: 6,
      base: dot > 0 ? name.slice(0, dot) : name,
    };
  }

  function outputGroups(job) {
    const settings = job.settings || {};
    const fallback = String(
      settings.out_basename
        || String(job.title || "").replace(/\.[^.]+$/, "")
        || "transcript"
    );
    const groups = new Map();
    for (const output of job.outputs_manifest || []) {
      const format = outputFormat(output.name);
      const base = format.base || fallback;
      if (!groups.has(base)) groups.set(base, []);
      groups.get(base).push({ ...output, label: format.label, rank: format.rank });
    }
    return Array.from(groups, ([base, files]) => ({
      base,
      files: files.sort((a, b) => a.rank - b.rank || a.name.localeCompare(b.name)),
    })).sort((a, b) => a.base.localeCompare(b.base));
  }

  function buildDetailRow(job) {
    const row = el("tr", "detail-row");
    row.dataset.detailFor = job.id;
    const cell = el("td");
    cell.colSpan = 5;
    const grid = el("div", "detail-cell");

    const left = el("div");
    left.append(el("h4", "", "Outputs"));
    if (job.outputs_manifest && job.outputs_manifest.length) {
      const groups = el("div", "output-groups");
      for (const group of outputGroups(job)) {
        const block = el("div", "output-group");
        block.append(el("div", "output-group-name", group.base));
        const buttons = el("div", "output-buttons");
        for (const file of group.files) {
          const link = el("a", "output-btn", file.label);
          link.href = `/api/jobs/${encodeURIComponent(job.id)}/outputs/${encodeURIComponent(file.name)}`;
          link.setAttribute("download", file.name);
          link.title = `${file.name} · ${formatBytes(file.size_ct)} · download`;
          link.setAttribute("aria-label", `Download ${file.label}: ${file.name}`);
          buttons.append(link);
        }
        block.append(buttons);
        groups.append(block);
      }
      left.append(groups);
    } else {
      left.append(el("p", "hint", job.status === "succeeded" ? "The manifest is empty." : "Outputs appear when rendering finishes."));
    }

    if (job.error) {
      const box = el("div", "error-box");
      box.append(el("strong", "", job.error.code || "error"));
      box.append(el("div", "", job.error.message || JSON.stringify(job.error)));
      if (job.error.details && job.error.details.stderr) {
        box.append(el("div", "hint", String(job.error.details.stderr).slice(-700)));
      }
      left.append(box);
    }

    if (job.log_paths && job.log_paths.length) {
      const logs = el("div", "log-links");
      for (const path of job.log_paths) {
        const name = String(path).split("/").pop();
        const link = el("a", "", name);
        link.href = `/api/jobs/${encodeURIComponent(job.id)}/logs/${encodeURIComponent(name)}`;
        link.target = "_blank";
        link.rel = "noreferrer";
        logs.append(link);
      }
      left.append(el("h4", "", "Logs"));
      left.append(logs);
    }

    const right = el("div");
    right.append(el("h4", "", "Settings"));
    const lines = el("div", "settings-chipline");
    const resolved = job.resolved || {};
    const settings = job.settings || {};
    const timestampLabel = settings.timestamps_mode === "interval"
      ? `timestamps: every ${settings.interval_s || 60} s`
      : settings.timestamps_mode === "turns" ? "timestamps: each turn" : "timestamps: none";
    const tags = [
      modelName(resolved.asr_model || settings.asr_model_id),
      resolved.device || settings.device,
      resolved.compute,
      settings.diar && settings.diar !== "off" ? `speaker labels: ${settings.diar}` : "no speaker labels",
      timestampLabel,
      settings.word_timestamps ? "word timing" : null,
      settings.mark_pauses ? "pause marks" : null,
    ].filter((tag, index, all) => Boolean(tag) && all.indexOf(tag) === index);
    for (const tag of tags) lines.append(el("span", "tag", tag));
    right.append(lines);

    const meta = el("dl", "meta-grid");
    const add = (label, value) => {
      const div = el("div");
      div.append(el("dt", "", label));
      div.append(el("dd", "", value));
      meta.append(div);
    };
    add("created", formatDate(job.created_at));
    add("started", formatDate(job.started_at));
    add("finished", formatDate(job.finished_at));
    add("duration", formatDuration(job.duration_s));
    add("message", job.message || "—");
    add("clip", `${settings.start_at || "start"} → ${settings.end_at || "end"}`);
    right.append(meta);

    grid.append(left);
    grid.append(right);
    cell.append(grid);
    row.append(cell);
    return row;
  }

  function renderQueue() {
    const body = $("#queue-body");
    if (!body) return;
    const wrap = $("#queue-wrap");
    const empty = $("#queue-empty");
    const count = $("#queue-count");
    const jobs = state.jobs
      .slice()
      .sort((a, b) => {
        const seqA = Number(a.seq);
        const seqB = Number(b.seq);
        if (Number.isFinite(seqA) && Number.isFinite(seqB) && seqA !== seqB) {
          return seqB - seqA;
        }
        return String(b.created_at || "").localeCompare(String(a.created_at || ""));
      });
    body.replaceChildren();
    for (const job of jobs) {
      body.append(buildRow(job));
      if (state.expanded.has(job.id)) body.append(buildDetailRow(job));
    }
    if (empty) empty.hidden = jobs.length > 0;
    if (wrap) wrap.hidden = jobs.length === 0;
    if (count) count.textContent = `${jobs.length} job${jobs.length === 1 ? "" : "s"}`;
  }

  function upsertJob(job) {
    if (!job || !job.id) return;
    const index = state.jobs.findIndex((item) => item.id === job.id);
    if (index >= 0) state.jobs[index] = job;
    else state.jobs.push(job);
  }

  function ingestJobs(jobs) {
    if (!Array.isArray(jobs) || !jobs.length) return;
    for (const job of jobs) upsertJob(job);
    renderQueue();
  }

  const refreshing = new Set();

  // A pending/running row that drops out of a `queue` payload without a
  // terminal `job` event means we missed the outcome (typically while the
  // EventSource was reconnecting). Ask the API for the real row; a 404 means
  // the job was deleted in the meantime.
  async function refreshJob(jobId) {
    if (refreshing.has(jobId)) return;
    refreshing.add(jobId);
    let changed = false;
    try {
      const response = await fetch(`/api/jobs/${encodeURIComponent(jobId)}`, {
        headers: { Accept: "application/json" },
      });
      if (response.status === 404) {
        state.jobs = state.jobs.filter((item) => item.id !== jobId);
        state.expanded.delete(jobId);
        changed = true;
      } else if (response.ok) {
        upsertJob(await response.json());
        changed = true;
      }
    } catch (error) {
      console.error("job refresh failed", error);
    } finally {
      refreshing.delete(jobId);
      if (changed) renderQueue();
    }
  }

  /* ------------------------------------------------------------- engines */

  function modelRow(engine, modelId) {
    return document.querySelector(`.model-row[data-engine="${CSS.escape(engine)}"][data-model="${CSS.escape(modelId)}"]`);
  }

  function applyModelState(data) {
    if (!data) return;
    const row = modelRow(data.engine, data.model_id);
    if (!row) return;
    const badge = $(".state", row);
    if (badge) {
      badge.textContent = data.status;
      badge.dataset.state = data.status;
      badge.className = `state state-${data.status}`;
    }
    let mini = $(".mini-bar", row);
    if (data.status === "downloading") {
      if (!mini) {
        mini = el("span", "mini-bar");
        mini.append(el("span"));
        $(".model-state", row).append(mini);
      }
      const fill = $("span", mini);
      if (fill) fill.style.width = `${Math.max(0, Math.min(100, Number(data.pct || 0)))}%`;
    } else if (mini) {
      mini.remove();
    }
    const actions = $(".model-actions", row);
    if (!actions) return;
    actions.replaceChildren();
    const engineKey = data.engine;
    const modelId = data.model_id;
    const make = (label, action, quiet) => {
      const button = el("button", `btn btn-small ${quiet ? "btn-quiet" : ""}`, label);
      button.type = "button";
      button.dataset.action = action;
      button.dataset.engine = engineKey;
      button.dataset.model = modelId;
      return button;
    };
    if (engineKey === "demo") {
      actions.append(el("span", "muted", "bundled"));
    } else if (data.status === "downloading") {
      actions.append(make("Cancel", "model-cancel", true));
    } else if (data.status === "ready") {
      actions.append(make("Remove", "model-remove", true));
    } else {
      actions.append(make(data.status === "partial" ? "Resume" : "Install", "model-download", false));
    }
    if (row.dataset.role === "asr") {
      if (row.dataset.default === "true") {
        actions.append(el("span", "pill pill-ok", "default"));
      } else {
        const use = el("button", "btn btn-small btn-quiet", "Use");
        use.type = "button";
        use.dataset.action = "use-model";
        use.dataset.model = modelId;
        use.dataset.label = row.dataset.label || modelId;
        use.title = "Use this model for new jobs.";
        actions.append(use);
      }
    }
    if (data.status === "partial" && data.last_error) {
      let error = $(".state-error", row);
      if (!error) {
        error = el("span", "state-error");
        $(".model-state", row).append(error);
      }
      error.textContent = String(data.last_error).slice(0, 90);
      error.title = data.last_error;
    }
  }

  function setLive(ok) {
    state.live = ok;
    const chip = $("#live-chip");
    if (!chip) return;
    chip.classList.toggle("chip-ok", ok);
    chip.classList.toggle("chip-off", !ok);
    chip.textContent = ok ? "Live" : "Offline";
    chip.title = ok ? "The page receives status updates from the server." : "The page cannot reach the server. It keeps trying.";
  }

  /* ------------------------------------------------------------- settings */

  function values(form) {
    const data = new FormData(form);
    return data;
  }

  function collectSettings(form) {
    const data = values(form);
    const checked = (name) => {
      const input = form.querySelector(`input[name="${name}"]`);
      return Boolean(input && input.checked && !input.disabled);
    };
    const text = (name) => String(data.get(name) || "").trim();
    const device = form.querySelector('input[name="device"]:checked');
    const formats = data.getAll("formats").map(String);
    if (!formats.length) {
      throw new Error("Choose at least one output format.");
    }
    const speaker = text("speaker_count") || "auto";
    const interval = Number(text("interval_s") || 60);
    const pause = Number(text("pause_min_s") || 2);
    return {
      asr_model_id: text("asr_model_id"),
      device: device ? device.value : "auto",
      language: text("language") || "auto",
      diar: text("diar") || "off",
      diar_model_id: text("diar") === "off" ? null : text("diar_model_id") || null,
      speaker_count: speaker === "auto" ? "auto" : Number(speaker),
      timestamps_mode: text("timestamps_mode") || "interval",
      interval_s: Number.isFinite(interval) ? interval : 60,
      word_timestamps: checked("word_timestamps"),
      mark_pauses: checked("mark_pauses"),
      pause_min_s: Number.isFinite(pause) ? pause : 2,
      verbatim: checked("verbatim"),
      start_at: text("start_at") || null,
      end_at: text("end_at") || null,
      output: {
        html: formats.includes("html"),
        txt: formats.includes("txt"),
        srt: formats.includes("srt"),
        vtt: formats.includes("vtt"),
        timed_txt: formats.includes("timed_txt"),
        json: formats.includes("json"),
      },
      out_basename: text("out_basename") || null,
    };
  }

  async function saveDefaults(button) {
    const form = button.closest("form") || document.getElementById("job-settings-form");
    if (!form) return;
    let settings;
    try {
      settings = collectSettings(form);
    } catch (error) {
      toast(error.message, "error");
      return;
    }
    button.disabled = true;
    try {
      const saved = await request("/api/settings", {
        method: "PUT",
        body: JSON.stringify({ defaults: settings }),
      });
      WS.settings = saved.defaults;
      state.settings = saved.defaults;
      toast("Saved as defaults for new jobs.", "ok");
    } catch (error) {
      toast(error.message, "error");
    } finally {
      button.disabled = false;
    }
  }

  async function useModel(button) {
    const modelId = button.dataset.model;
    const label = button.dataset.label || modelId;
    const current = state.settings || WS.settings || {};
    if (!modelId || !current.asr_model_id) {
      toast("The defaults did not load. Reload the page.", "error");
      return;
    }
    const engines = Array.isArray(WS.engines) ? WS.engines : [];
    const engine = engines.find((item) => (item.models || []).some((model) => model.id === modelId));
    if (!engine) {
      toast("The model list did not load. Reload the page.", "error");
      return;
    }
    const enabled = state.engineEnabled[engine.key] ?? engine.enabled;
    if (!enabled) {
      toast(`Turn on ${engine.name} before you use its models.`, "error");
      return;
    }
    const supported = Array.isArray(engine.devices) ? engine.devices : [];
    const device = supported.includes(current.device) ? current.device : "auto";
    const deviceChanged = device !== current.device;
    button.disabled = true;
    try {
      const next = { ...current, asr_model_id: modelId, device };
      await request("/api/settings", { method: "PUT", body: JSON.stringify({ defaults: next }) });
      state.settings = next;
      WS.settings = next;
      toast(
        `${label} is the default model now.${deviceChanged ? " Device changed to Auto." : ""}`,
        "ok",
        deviceChanged ? 4200 : 2600,
      );
      window.setTimeout(() => window.location.reload(), 700);
    } catch (error) {
      toast(error.message, "error");
      button.disabled = false;
    }
  }

  async function saveToken(button) {
    const form = button.closest("form");
    if (!form) return;
    const input = form.querySelector('input[name="hf_token"]');
    const clear = form.querySelector('input[name="clear_token"]');
    const body = {};
    if (clear && clear.checked) body.hf_token = null;
    else if (input && input.value.trim()) body.hf_token = input.value.trim();
    else {
      toast("Paste a token first, or tick remove.", "error");
      return;
    }
    button.disabled = true;
    try {
      const saved = await request("/api/settings", { method: "PUT", body: JSON.stringify(body) });
      WS.settings = saved;
      state.settings = saved.defaults || state.settings;
      if (input) {
        input.value = "";
        input.placeholder = saved.hf_token || "hf_…";
      }
      if (clear) clear.checked = false;
      const card = button.closest(".token-card");
      const pill = card && card.querySelector(".card-head .pill");
      const removed = body.hf_token === null;
      if (pill) {
        pill.textContent = removed ? "not set" : saved.hf_token ? "token present" : "not set";
        pill.classList.toggle("pill-ok", !removed && Boolean(saved.hf_token));
        pill.classList.toggle("pill-quiet", removed || !saved.hf_token);
      }
      const clearWrap = card && card.querySelector("#clear-token-wrap");
      if (clearWrap) clearWrap.hidden = removed || !saved.hf_token;
      toast(removed ? "Stored token removed." : "Token saved.", "ok");
    } catch (error) {
      toast(error.message, "error");
    } finally {
      button.disabled = false;
    }
  }

  async function saveHousekeeping(button) {
    const form = button.closest("form");
    if (!form) return;
    const days = Number(form.querySelector('[name="keep_completed_jobs_days"]').value || 14);
    const cleanup = form.querySelector('[name="cleanup_uploads"]').checked;
    button.disabled = true;
    try {
      const saved = await request("/api/settings", {
        method: "PUT",
        body: JSON.stringify({ keep_completed_jobs_days: days, cleanup_uploads: cleanup }),
      });
      WS.settings = saved;
      toast("Housekeeping saved.", "ok");
    } catch (error) {
      toast(error.message, "error");
    } finally {
      button.disabled = false;
    }
  }

  async function runCleanup(button) {
    button.disabled = true;
    try {
      const result = await post("/api/maintenance/cleanup");
      toast(`Removed ${result.staging_removed || 0} staging file(s) and ${result.jobs_removed || 0} old job(s).`, "ok");
    } catch (error) {
      toast(error.message, "error");
    } finally {
      button.disabled = false;
    }
  }

  /* ----------------------------------------------------------------- modal */

  function closeModal() {
    const modal = $("#modal");
    if (!modal) return;
    modal.hidden = true;
    $("#modal-body").replaceChildren();
    state.currentJobId = null;
  }

  async function openJobEditor(jobId) {
    const modal = $("#modal");
    const body = $("#modal-body");
    if (!modal || !body) return;
    body.replaceChildren(el("p", "hint", "Loading settings…"));
    modal.hidden = false;
    try {
      const response = await fetch(`/partials/settings-form?job_id=${encodeURIComponent(jobId)}&mode=job`, {
        headers: { Accept: "text/html" },
      });
      if (!response.ok) throw new Error(`Could not load settings (HTTP ${response.status}).`);
      body.innerHTML = await response.text();
      state.currentJobId = jobId;
      if (window.htmx) window.htmx.process(body);
      const first = body.querySelector("select, input, button");
      if (first) first.focus();
    } catch (error) {
      body.replaceChildren(el("p", "error-box", error.message));
    }
  }

  async function saveJob(button) {
    const form = button.closest("form");
    if (!form) return;
    const jobId = button.dataset.jobId || state.currentJobId;
    if (!jobId) return;
    let settings;
    try {
      settings = collectSettings(form);
    } catch (error) {
      toast(error.message, "error");
      return;
    }
    button.disabled = true;
    try {
      const job = await request(`/api/jobs/${encodeURIComponent(jobId)}`, {
        method: "PATCH",
        body: JSON.stringify(settings),
      });
      upsertJob(job);
      closeModal();
      renderQueue();
      toast("Job settings saved.", "ok");
    } catch (error) {
      toast(error.message, "error");
    } finally {
      button.disabled = false;
    }
  }

  /* --------------------------------------------------------------- actions */

  async function handleAction(button) {
    const action = button.dataset.action;
    const jobId = button.dataset.jobId;
    if (action === "refresh-hardware") return refreshHardware(button);
    if (action === "save-defaults") return saveDefaults(button);
    if (action === "use-model") return useModel(button);
    if (action === "save-token") return saveToken(button);
    if (action === "save-housekeeping") return saveHousekeeping(button);
    if (action === "cleanup") return runCleanup(button);
    if (action === "job-edit") return openJobEditor(jobId);
    if (action === "job-save") return saveJob(button);
    if (action === "job-cancel") {
      button.disabled = true;
      try {
        const job = await post(`/api/jobs/${encodeURIComponent(jobId)}/cancel`);
        upsertJob(job);
        renderQueue();
        toast("Cancelling…", "info", 3000);
      } catch (error) {
        toast(error.message, "error");
        button.disabled = false;
      }
      return;
    }
    if (action === "job-retry") {
      button.disabled = true;
      try {
        const job = await post(`/api/jobs/${encodeURIComponent(jobId)}/retry`);
        upsertJob(job);
        renderQueue();
        toast("Queued again at the tail.", "ok");
      } catch (error) {
        toast(error.message, "error");
        button.disabled = false;
      }
      return;
    }
    if (action === "job-fallback") {
      if (!window.confirm("Retry this job on CPU? The device failure will not repeat silently; this is an explicit choice.")) return;
      button.disabled = true;
      try {
        const job = await post(`/api/jobs/${encodeURIComponent(jobId)}/fallback-cpu`);
        upsertJob(job);
        renderQueue();
        toast("Queued again on CPU.", "ok");
      } catch (error) {
        toast(error.message, "error");
        button.disabled = false;
      }
      return;
    }
    if (action === "job-delete") {
      if (!window.confirm("Delete this job from the queue? Files already written stay on disk until retention cleanup.")) return;
      button.disabled = true;
      try {
        await request(`/api/jobs/${encodeURIComponent(jobId)}`, { method: "DELETE" });
        state.jobs = state.jobs.filter((job) => job.id !== jobId);
        state.expanded.delete(jobId);
        renderQueue();
        toast("Job removed.", "ok", 3200);
      } catch (error) {
        toast(error.message, "error");
        button.disabled = false;
      }
      return;
    }
    if (action === "model-download" || action === "model-cancel" || action === "model-remove") {
      const engine = button.dataset.engine;
      const model = button.dataset.model;
      const endpoint = action === "model-download" ? "download" : action === "model-cancel" ? "cancel" : "remove";
      button.disabled = true;
      try {
        const result = await post(`/api/models/${encodeURIComponent(engine)}/${encodeURIComponent(model)}/${endpoint}`);
        applyModelState(result);
        toast(action === "model-download" ? "Download started." : action === "model-cancel" ? "Cancelling download…" : "Model removed.", "ok");
      } catch (error) {
        toast(error.message, "error");
        button.disabled = false;
      }
      return;
    }
    if (action === "scan-models") {
      button.disabled = true;
      try {
        await post("/api/models/scan");
        toast("Model shelf scanned. Reloading…", "ok", 1500);
        window.setTimeout(() => window.location.reload(), 500);
      } catch (error) {
        toast(error.message, "error");
        button.disabled = false;
      }
      return;
    }
    if (action === "engine-toggle") {
      const engine = button.dataset.engine;
      const enabled = button.dataset.enabled === "true";
      const endpoint = enabled ? "disable" : "enable";
      button.disabled = true;
      try {
        const result = await post(`/api/engines/${encodeURIComponent(engine)}/${endpoint}`);
        state.engineEnabled[result.engine] = result.enabled;
        button.dataset.enabled = String(result.enabled);
        button.setAttribute("aria-checked", String(result.enabled));
        button.classList.toggle("is-on", result.enabled);
        const text = button.querySelector(".switch-text");
        if (text) text.textContent = result.enabled ? "on" : "off";
        const card = button.closest(".engine-card");
        if (card) card.classList.toggle("is-disabled", !result.enabled);
        toast(`${engine} ${result.enabled ? "enabled" : "disabled"}.`, "ok", 3200);
      } catch (error) {
        toast(error.message, "error");
      } finally {
        button.disabled = false;
      }
    }
  }

  /* ------------------------------------------------------------- SSE glue */

  function startEvents() {
    if (!window.EventSource) {
      setLive(false);
      return;
    }
    const source = new EventSource("/api/events");
    source.onopen = () => {
      setLive(true);
      state.offlineToasted = false;
    };
    source.onerror = () => {
      setLive(false);
      if (!state.offlineToasted) {
        state.offlineToasted = true;
        toast("Live updates paused. The browser will keep trying.", "error", 5000);
      }
    };
    source.addEventListener("snapshot", (event) => {
      try {
        const data = JSON.parse(event.data);
        state.jobs = Array.isArray(data.jobs) ? data.jobs : state.jobs;
        state.engineEnabled = data.engine_enabled || state.engineEnabled;
        for (const [engine, models] of Object.entries(data.model_state || {})) {
          for (const model of models) state.modelState[`${engine}/${model.model_id}`] = model;
        }
        renderHardware(data.hardware);
        renderQueue();
        refreshEngineStates();
      } catch (error) {
        console.error("bad snapshot", error);
      }
    });
    source.addEventListener("job", (event) => {
      try {
        upsertJob(JSON.parse(event.data));
        renderQueue();
      } catch (error) {
        console.error("bad job event", error);
      }
    });
    source.addEventListener("queue", (event) => {
      try {
        const jobs = JSON.parse(event.data);
        if (!Array.isArray(jobs)) return;
        const active = new Set(jobs.map((job) => job.id));
        for (const job of jobs) upsertJob(job);
        renderQueue();
        for (const job of state.jobs.slice()) {
          if (!active.has(job.id) && (job.status === "pending" || job.status === "running")) {
            refreshJob(job.id);
          }
        }
      } catch (error) {
        console.error("bad queue event", error);
      }
    });
    source.addEventListener("hardware", (event) => {
      try { renderHardware(JSON.parse(event.data)); } catch (error) { console.error(error); }
    });
    source.addEventListener("model", (event) => {
      try {
        const data = JSON.parse(event.data);
        state.modelState[`${data.engine}/${data.model_id}`] = data;
        applyModelState(data);
      } catch (error) { console.error(error); }
    });
    source.addEventListener("engine", (event) => {
      try {
        const data = JSON.parse(event.data);
        state.engineEnabled[data.engine] = data.enabled;
        const button = document.querySelector(`.switch[data-engine="${CSS.escape(data.engine)}"]`);
        if (button) {
          button.dataset.enabled = String(data.enabled);
          button.setAttribute("aria-checked", String(data.enabled));
          button.classList.toggle("is-on", data.enabled);
          const text = button.querySelector(".switch-text");
          if (text) text.textContent = data.enabled ? "on" : "off";
          const card = button.closest(".engine-card");
          if (card) card.classList.toggle("is-disabled", !data.enabled);
        }
      } catch (error) { console.error(error); }
    });
    source.addEventListener("toast", (event) => {
      try {
        const data = JSON.parse(event.data);
        toast(data.text || "Update", data.level || "info");
      } catch (error) { console.error(error); }
    });
  }

  function refreshEngineStates() {
    for (const model of Object.values(state.modelState)) applyModelState(model);
  }

  /* -------------------------------------------------------------- listeners */

  document.addEventListener("click", (event) => {
    const close = event.target.closest("[data-close-modal]");
    if (close) {
      event.preventDefault();
      closeModal();
      return;
    }
    const action = event.target.closest("[data-action]");
    if (action) {
      event.preventDefault();
      event.stopPropagation();
      handleAction(action);
      return;
    }
    const row = event.target.closest("tr.job-row");
    if (row && row.dataset.jobId) {
      const id = row.dataset.jobId;
      if (state.expanded.has(id)) state.expanded.delete(id);
      else state.expanded.add(id);
      renderQueue();
    }
  });

  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") closeModal();
  });

  document.addEventListener("htmx:responseError", (event) => {
    let message = "The settings request failed.";
    try {
      const payload = JSON.parse(event.detail.xhr.responseText);
      message = apiError(payload, message);
    } catch { /* keep default */ }
    toast(message, "error");
  });

  document.addEventListener("htmx:afterSwap", (event) => {
    if (event.detail.target && event.detail.target.id === "modal-body") {
      const first = event.detail.target.querySelector("select, input, button");
      if (first) first.focus();
    }
  });

  let initialized = false;
  function init() {
    if (initialized) return;
    initialized = true;
    renderHardware(state.hardware);
    renderQueue();
    startEvents();
    for (const id of ["refresh-hardware", "refresh-hardware-2"]) {
      const button = document.getElementById(id);
      if (button) button.addEventListener("click", () => refreshHardware(button));
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init, { once: true });
  } else {
    init();
  }

  window.WS.app = { toast, renderQueue, ingestJobs, collectSettings, openJobEditor, closeModal };
})();
