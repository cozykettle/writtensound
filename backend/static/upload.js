/* Drag, drop, upload sequentially, then hand the batch to the queue. */
(() => {
  "use strict";

  const dropzone = document.getElementById("dropzone");
  const fileInput = document.getElementById("file-input");
  const folderInput = document.getElementById("folder-input");
  const chooseFiles = document.getElementById("choose-files");
  const chooseFolder = document.getElementById("choose-folder");
  const panel = document.getElementById("file-panel");
  const list = document.getElementById("file-list");
  const addButton = document.getElementById("add-to-queue");
  const clearButton = document.getElementById("clear-batch");
  const summary = document.getElementById("batch-summary");
  const batchSize = document.getElementById("batch-size");
  if (!dropzone || !fileInput || !folderInput || !panel || !addButton) return;

  const MAX_UPLOAD = Number(document.body.dataset.maxUpload || 16 * 1024 ** 3);
  const ALLOWED = new Set([
    ".wav", ".mp3", ".m4a", ".ogg", ".opus", ".flac", ".aac", ".wma", ".aiff",
    ".mp4", ".mkv", ".mov", ".webm", ".avi", ".mpg", ".mpeg", ".wmv",
  ]);
  const CLIENT_ID = (window.crypto && window.crypto.randomUUID) ? window.crypto.randomUUID() : `batch-${Date.now()}`;
  const items = [];
  const seen = new Set();
  let uploading = false;
  let dragDepth = 0;
  let renderQueued = false;

  const sizes = (bytes) => {
    let value = Number(bytes);
    const units = ["B", "KB", "MB", "GB", "TB"];
    let index = 0;
    while (value >= 1024 && index < units.length - 1) { value /= 1024; index += 1; }
    return `${index === 0 ? value.toFixed(0) : value.toFixed(value >= 10 ? 1 : 2)} ${units[index]}`;
  };

  function relativePath(file) {
    return file.webkitRelativePath || file.name;
  }

  function suffix(name) {
    const dot = String(name).lastIndexOf(".");
    return dot >= 0 ? String(name).slice(dot).toLowerCase() : "";
  }

  function addFiles(files) {
    let added = 0;
    let rejected = 0;
    for (const file of Array.from(files)) {
      const path = relativePath(file);
      const key = `${path}\u0000${file.size}\u0000${file.lastModified}`;
      if (seen.has(key)) continue;
      if (!ALLOWED.has(suffix(file.name))) {
        rejected += 1;
        continue;
      }
      if (file.size > MAX_UPLOAD) {
        window.WS.app?.toast(`${file.name} is larger than the upload limit (${sizes(MAX_UPLOAD)}).`, "error");
        rejected += 1;
        continue;
      }
      seen.add(key);
      items.push({
        key,
        file,
        path,
        size: file.size,
        status: "staged",
        progress: 0,
        uploadId: null,
        error: null,
      });
      added += 1;
    }
    if (rejected) window.WS.app?.toast(`${rejected} file(s) skipped by type or size.`, "error");
    if (added) {
      render();
      if (panel.hidden) {
        panel.hidden = false;
        window.WS.app?.toast(`${added} file${added === 1 ? "" : "s"} staged.`, "info", 2600);
      }
    }
  }

  function scheduleRender() {
    if (renderQueued) return;
    renderQueued = true;
    window.requestAnimationFrame(() => {
      renderQueued = false;
      render();
    });
  }

  function render() {
    if (!list) return;
    const fragment = document.createDocumentFragment();
    for (const item of items) {
      const row = document.createElement("li");
      row.className = "file-row";
      row.dataset.key = item.key;
      const name = document.createElement("div");
      name.className = "file-name";
      name.textContent = item.file.name;
      const path = document.createElement("span");
      path.className = "file-path";
      path.textContent = `${item.path} · ${sizes(item.size)}`;
      name.append(path);
      row.append(name);

      const barWrap = document.createElement("div");
      barWrap.className = "file-bar";
      const bar = document.createElement("span");
      bar.style.width = `${Math.max(0, Math.min(100, item.progress || 0))}%`;
      if (item.status === "ready") bar.style.width = "100%";
      if (item.status === "failed") bar.style.background = "var(--red)";
      barWrap.append(bar);
      row.append(barWrap);

      const status = document.createElement("span");
      status.className = `file-state is-${item.status}`;
      status.textContent = item.status === "uploading" ? `${Math.round(item.progress)}%` : item.status;
      row.append(status);

      const remove = document.createElement("button");
      remove.type = "button";
      remove.className = "file-remove";
      remove.dataset.removeKey = item.key;
      remove.title = "Remove from this batch";
      remove.setAttribute("aria-label", `Remove ${item.file.name}`);
      remove.textContent = "×";
      remove.disabled = uploading;
      row.append(remove);
      fragment.append(row);
    }
    list.replaceChildren(fragment);

    const ready = items.filter((item) => item.status === "ready").length;
    const failed = items.filter((item) => item.status === "failed").length;
    const pending = items.length - ready - failed;
    panel.hidden = items.length === 0;
    addButton.disabled = uploading || items.length === 0 || failed > 0;
    addButton.textContent = uploading
      ? "Uploading…"
      : items.length
        ? `Add ${items.length} to queue`
        : "Add to queue";
    if (summary) {
      if (!items.length) summary.textContent = "No files staged.";
      else {
        const pieces = [`${items.length} file${items.length === 1 ? "" : "s"}`, `${sizes(items.reduce((total, item) => total + item.size, 0))}`];
        if (ready) pieces.push(`${ready} uploaded`);
        if (pending) pieces.push(`${pending} waiting`);
        if (failed) pieces.push(`${failed} failed`);
        summary.textContent = pieces.join(" · ");
      }
    }
    if (batchSize) {
      batchSize.textContent = items.length ? `${items.length} staged` : "nothing staged yet";
    }
    if (clearButton) clearButton.disabled = uploading;
    dropzone.classList.toggle("is-busy", uploading);
  }

  function uploadOne(item) {
    return new Promise((resolve) => {
      item.status = "uploading";
      item.progress = 0;
      item.error = null;
      scheduleRender();
      const xhr = new XMLHttpRequest();
      xhr.open("POST", "/api/uploads", true);
      xhr.setRequestHeader("X-Client-Id", CLIENT_ID);
      xhr.setRequestHeader("X-Relative-Path", encodeURIComponent(item.path));
      xhr.upload.onprogress = (event) => {
        if (!event.lengthComputable) return;
        item.progress = Math.max(1, (event.loaded / event.total) * 100);
        scheduleRender();
      };
      xhr.onerror = () => {
        item.status = "failed";
        item.error = "Network error while uploading.";
        scheduleRender();
        resolve(false);
      };
      xhr.onload = () => {
        let payload = null;
        try { payload = JSON.parse(xhr.responseText); } catch { /* no JSON */ }
        if (xhr.status >= 200 && xhr.status < 300 && payload && payload.upload_id) {
          item.uploadId = payload.upload_id;
          item.status = "ready";
          item.progress = 100;
          resolve(true);
        } else {
          item.status = "failed";
          item.error = (payload && (payload.message || payload.detail)) || `Upload failed (HTTP ${xhr.status}).`;
          resolve(false);
        }
        scheduleRender();
      };
      xhr.send(makeFormData(item));
    });
  }

  function makeFormData(item) {
    const form = new FormData();
    form.append("file", item.file, item.file.name);
    return form;
  }

  async function uploadPending() {
    if (uploading) return false;
    uploading = true;
    render();
    let ok = true;
    for (const item of items) {
      if (item.status === "ready") continue;
      if (item.status === "failed") {
        // A retry is explicit: failed rows stay failed until the user removes them
        // and re-adds them, which avoids accidental duplicate jobs.
        if (!item.uploadId) {
          // allow one retry when the user presses the button again
        }
      }
      const success = await uploadOne(item);
      if (!success) {
        ok = false;
        window.WS.app?.toast(item.error || `${item.file.name} could not be uploaded.`, "error");
      }
    }
    uploading = false;
    render();
    return ok;
  }

  async function addToQueue() {
    if (uploading) return;
    const pending = items.filter((item) => item.status !== "ready");
    if (pending.length) {
      const ok = await uploadPending();
      if (!ok) {
        window.WS.app?.toast("Some uploads failed. Remove them and try again.", "error");
        return;
      }
    }
    const ready = items.filter((item) => item.status === "ready" && item.uploadId);
    if (!ready.length) {
      window.WS.app?.toast("No uploaded files are ready.", "error");
      return;
    }
    let settings;
    try {
      const form = document.getElementById("job-settings-form");
      if (!form || !window.WS.app) throw new Error("The settings form is not ready.");
      settings = window.WS.app.collectSettings(form);
    } catch (error) {
      window.WS.app?.toast(error.message, "error");
      return;
    }
    addButton.disabled = true;
    addButton.textContent = "Queueing…";
    try {
      const response = await fetch("/api/jobs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          items: ready.map((item) => ({ upload_id: item.uploadId, settings })),
        }),
      });
      let payload = null;
      try { payload = await response.json(); } catch { /* ignore */ }
      if (!response.ok) {
        const message = (payload && (payload.message || payload.detail)) || `Queueing failed (HTTP ${response.status}).`;
        const detail = payload && payload.details && payload.details.items_errors;
        throw new Error(detail ? `${message} ${detail.map((row) => row.error?.message || "").join(" ")}`.trim() : message);
      }
      const count = (payload && payload.job_ids && payload.job_ids.length) || ready.length;
      window.WS.app?.toast(`${count} job${count === 1 ? "" : "s"} added to the queue.`, "ok");
      items.length = 0;
      seen.clear();
      render();
      panel.hidden = true;
      const queueSection = document.querySelector(".queue-section");
      if (queueSection) queueSection.scrollIntoView({ behavior: "smooth", block: "start" });
    } catch (error) {
      window.WS.app?.toast(error.message, "error");
    } finally {
      addButton.disabled = false;
      render();
    }
  }

  /* ------------------------------------------------------------- entries */

  async function readDirectory(entry, prefix, out) {
    const reader = entry.createReader();
    while (true) {
      const batch = await new Promise((resolve, reject) => reader.readEntries(resolve, reject));
      if (!batch.length) break;
      for (const child of batch) {
        if (child.isFile) {
          const file = await new Promise((resolve, reject) => child.file(resolve, reject));
          out.push({ file, path: `${prefix}${file.name}` });
        } else if (child.isDirectory) {
          await readDirectory(child, `${prefix}${child.name}/`, out);
        }
      }
    }
  }

  async function filesFromDrop(dataTransfer) {
    const entries = [];
    if (dataTransfer.items && dataTransfer.items.length) {
      for (const item of Array.from(dataTransfer.items)) {
        if (item.kind !== "file") continue;
        const entry = item.webkitGetAsEntry && item.webkitGetAsEntry();
        if (entry) entries.push(entry);
      }
    }
    if (!entries.length) {
      return Array.from(dataTransfer.files || []).map((file) => ({ file, path: relativePath(file) }));
    }
    const out = [];
    for (const entry of entries) {
      if (entry.isFile) {
        const file = await new Promise((resolve, reject) => entry.file(resolve, reject));
        out.push({ file, path: file.name });
      } else if (entry.isDirectory) {
        await readDirectory(entry, `${entry.name}/`, out);
      }
    }
    return out;
  }

  /* ------------------------------------------------------------ listeners */

  chooseFiles?.addEventListener("click", (event) => {
    event.stopPropagation();
    fileInput.click();
  });
  chooseFolder?.addEventListener("click", (event) => {
    event.stopPropagation();
    folderInput.click();
  });
  dropzone.addEventListener("click", (event) => {
    if (event.target.closest("button, input")) return;
    fileInput.click();
  });
  dropzone.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      fileInput.click();
    }
  });
  fileInput.addEventListener("change", () => {
    addFiles(fileInput.files);
    fileInput.value = "";
  });
  folderInput.addEventListener("change", () => {
    addFiles(folderInput.files);
    folderInput.value = "";
  });

  for (const type of ["dragenter", "dragover"]) {
    dropzone.addEventListener(type, (event) => {
      event.preventDefault();
      event.stopPropagation();
      if (uploading) return;
      if (type === "dragenter") dragDepth += 1;
      dropzone.classList.add("is-dragging");
    });
  }
  dropzone.addEventListener("dragleave", (event) => {
    event.preventDefault();
    dragDepth = Math.max(0, dragDepth - 1);
    if (!dragDepth) dropzone.classList.remove("is-dragging");
  });
  dropzone.addEventListener("drop", async (event) => {
    event.preventDefault();
    event.stopPropagation();
    dragDepth = 0;
    dropzone.classList.remove("is-dragging");
    if (uploading) return;
    try {
      const entries = await filesFromDrop(event.dataTransfer);
      addFiles(entries.map((row) => {
        try {
          Object.defineProperty(row.file, "webkitRelativePath", { value: row.path });
        } catch { /* read-only in some browsers; fall back to name */ }
        return row.file;
      }));
    } catch (error) {
      window.WS.app?.toast("Could not read the dropped items.", "error");
      console.error(error);
    }
  });

  window.addEventListener("dragover", (event) => {
    if (event.dataTransfer && Array.from(event.dataTransfer.types || []).includes("Files")) {
      event.preventDefault();
    }
  });
  window.addEventListener("drop", (event) => {
    if (!dropzone.contains(event.target)) event.preventDefault();
  });

  list.addEventListener("click", (event) => {
    const remove = event.target.closest("[data-remove-key]");
    if (!remove || uploading) return;
    const index = items.findIndex((item) => item.key === remove.dataset.removeKey);
    if (index >= 0) {
      seen.delete(items[index].key);
      items.splice(index, 1);
      render();
    }
  });

  clearButton?.addEventListener("click", () => {
    if (uploading) return;
    items.length = 0;
    seen.clear();
    render();
    panel.hidden = true;
  });

  addButton.addEventListener("click", addToQueue);

  render();
})();
