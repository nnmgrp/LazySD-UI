/* ============================================================
   QuantFunc WebUI — main app JavaScript
   Bridges frontend HTML/CSS with Python backend via REST API.

   Backend endpoints (defined in app.py):
     GET  /api/config              → {settings, gpu, engine_status, models}
     POST /api/generate            → starts generation job, returns job_id
     GET  /api/jobs/<id>           → polls job status
     POST /api/jobs/<id>/cancel    → cancels running job
     GET  /api/models              → list available models
     GET  /api/models/<id>/state   → check cache state for a model
     POST /api/models/<id>/download → download missing files
     GET  /api/history             → list output files
     PATCH /api/settings           → update settings
     POST /api/engine/download     → download engine binary
     POST /api/engine/verify       → verify engine checksum
     GET  /locale/<lang>.json      → translations
   ============================================================ */

function escapeHtml(str) {
  if (!str) return "";
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

const App = {
  // ---------- State ----------
  state: {
    settings: null,
    gpuInfo: null,
    engineStatus: null,
    models: [],
    currentTab: "generate",
    currentJob: null,
    pollInterval: null,
    selectedModel: null,
    selectedVariant: null,
    historyItems: [],
    selectedHistoryItem: null,
    img2imgRef: null,
    lastPreviewName: null,
    lastPreviewUrl: null,
  },

  // ---------- Init ----------
  async init() {
    console.log("[app] QuantFunc WebUI initializing...");

    // Load i18n first
    await window.initI18n();

    // Load initial config from backend
    await this.loadConfig();

    // Render initial UI
    this.renderGpuIndicator();
    this.renderModels();
    this.bindEvents();

    console.log("[app] Ready. GPU:", this.state.gpuInfo?.name || "not detected");
  },

  // ---------- API helpers ----------
  async apiGet(path) {
    const res = await fetch(path);
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || `HTTP ${res.status}`);
    }
    return res.json();
  },

  async apiPost(path, body = null) {
    const opts = {
      method: "POST",
      headers: body ? { "Content-Type": "application/json" } : {},
    };
    if (body) opts.body = JSON.stringify(body);
    const res = await fetch(path, opts);
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || `HTTP ${res.status}`);
    }
    return res.json();
  },

  async apiPatch(path, body) {
    const res = await fetch(path, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || `HTTP ${res.status}`);
    }
    return res.json();
  },

  // ---------- Config ----------
  async loadConfig() {
    try {
      const config = await this.apiGet("/api/config");
      this.state.settings = config.settings;
      this.state.gpuInfo = config.gpu;
      this.state.engineStatus = config.engine_status;
      this.state.models = config.models || [];
      // Inject into window for i18n detection
      window.QF_SETTINGS = config.settings;
    } catch (e) {
      console.error("[app] Failed to load config:", e);
      this.toast("error", t("errors.network_error") + ": " + e.message);
    }
  },

  async refreshConfig() {
    // Re-fetch config from backend after a state change (e.g. engine download).
    try {
      const config = await this.apiGet("/api/config");
      this.state.settings = config.settings;
      this.state.gpuInfo = config.gpu;
      this.state.engineStatus = config.engine_status;
      this.state.models = config.models || [];
      window.QF_SETTINGS = config.settings;
      this.renderGpuIndicator();
      this.renderModels();
      this.updateEngineStatusUI();
    } catch (e) {
      console.error("[app] Failed to refresh config:", e);
    }
  },

  updateEngineStatusUI() {
    // Update Settings tab engine status display after download.
    const statusEl = document.getElementById("setting-engine-status");
    if (!statusEl) return;
    const status = this.state.engineStatus?.status;
    if (status === "ok") {
      statusEl.innerHTML = '<span class="state-badge installed">✓ installed</span>';
    } else if (status === "missing") {
      statusEl.innerHTML = '<span class="state-badge missing">✗ missing</span>';
    } else {
      statusEl.textContent = status || "—";
    }
  },

  // ---------- Renderers ----------
  renderGpuIndicator() {
    const nameEl = document.getElementById("gpu-name");
    const dotEl = document.getElementById("gpu-status");
    if (!nameEl || !dotEl) return;

    if (this.state.gpuInfo) {
      const gpu = this.state.gpuInfo;
      nameEl.textContent = `${gpu.name} · ${gpu.tier} · ${Math.round(gpu.vram_mb / 1024)} GB`;
      dotEl.classList.remove("offline");
      dotEl.title = `SM ${gpu.sm} · Driver ${gpu.driver_version}`;
    } else {
      nameEl.textContent = t("gpu.not_detected");
      dotEl.classList.add("offline");
    }
  },

  renderModels() {
    const select = document.getElementById("model-select");
    if (!select) return;

    select.innerHTML = "";
    if (!this.state.models.length) {
      select.innerHTML = `<option value="">${t("common.loading")}</option>`;
      return;
    }

    // Filter models compatible with current GPU
    const gpuSm = this.state.gpuInfo?.sm || 0;
    const compatible = this.state.models.filter(m =>
      m.requirements.gpu_min_sm <= gpuSm
    );

    if (!compatible.length) {
      select.innerHTML = `<option value="">No compatible models</option>`;
      return;
    }

    for (const model of compatible) {
      const opt = document.createElement("option");
      opt.value = model.id;
      opt.textContent = `${model.name} (${model.family})`;
      select.appendChild(opt);
    }

    // Auto-select first recommended
    if (compatible[0]) {
      select.value = compatible[0].id;
      this.state.selectedModel = compatible[0].id;
      this.onModelChange();
    }
  },

  // ---------- Event handlers ----------
  bindEvents() {
    // Sidebar navigation
    document.querySelectorAll(".nav-item").forEach(btn => {
      btn.addEventListener("click", () => {
        if (btn.disabled) return;
        this.switchTab(btn.dataset.tab);
      });
    });

    // Model select
    const modelSelect = document.getElementById("model-select");
    if (modelSelect) {
      modelSelect.addEventListener("change", () => this.onModelChange());
    }

    // Generate button
    const genBtn = document.getElementById("generate-btn");
    if (genBtn) {
      genBtn.addEventListener("click", () => this.onGenerate());
    }

    // Keyboard shortcut: Ctrl/Cmd+Enter
    document.addEventListener("keydown", (e) => {
      if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
        e.preventDefault();
        this.onGenerate();
      }
      // ESC: close history modal first, then cancel generation
      if (e.key === "Escape") {
        if (this.state.selectedHistoryItem) {
          this.closeHistoryModal();
        } else if (this.state.currentJob) {
          this.cancelJob();
        }
      }
    });

    // Cancel button
    const cancelBtn = document.getElementById("progress-cancel");
    if (cancelBtn) {
      cancelBtn.addEventListener("click", () => this.cancelJob());
    }

    // History search filter
    const historySearch = document.getElementById("history-search");
    if (historySearch) {
      historySearch.addEventListener("input", () => this.renderHistory());
    }

    // Dropzone setup for Img2Img upload
    const previewArea = document.getElementById("tab-generate");
    if (previewArea && window.Dropzone) {
      this.dropzone = new window.Dropzone(previewArea, {
        onFile: (fileInfo) => {
          this.uploadImg2ImgFile(fileInfo.file);
        },
        acceptedTypes: ["image/png", "image/jpeg", "image/webp"],
        maxSizeMb: 25,
      });
    }

    // Modal background click to close
    const historyModal = document.getElementById("history-modal");
    if (historyModal) {
      historyModal.addEventListener("click", (e) => {
        if (e.target === historyModal) this.closeHistoryModal();
      });
    }

    // Language toggle
    document.querySelectorAll(".lang-btn").forEach(btn => {
      btn.addEventListener("click", () => window.setLocale(btn.dataset.lang));
    });
  },

  switchTab(tabName) {
    this.state.currentTab = tabName;

    // Update sidebar active state
    document.querySelectorAll(".nav-item").forEach(btn => {
      btn.classList.toggle("active", btn.dataset.tab === tabName);
    });

    // Hide all tab panels
    document.querySelectorAll(".tab-panel").forEach(panel => {
      panel.style.display = "none";
    });

    // Map tab aliases (e.g. img2img / image_edit -> tab-generate)
    const targetId = (tabName === "img2img" || tabName === "image_edit") ? "tab-generate" : `tab-${tabName}`;

    // Show selected panel
    const activePanel = document.getElementById(targetId);
    if (activePanel) {
      activePanel.style.display = (targetId === "tab-generate") ? "flex" : "block";
    }

    // Auto-load history when history tab opens
    if (tabName === "history") {
      this.loadHistory();
    }
  },

  async onModelChange() {
    const select = document.getElementById("model-select");
    if (!select || !select.value) return;

    this.state.selectedModel = select.value;

    // Fetch model state (cache check)
    try {
      const state = await this.apiGet(`/api/models/${select.value}/state`);
      this.renderVariants(state);
    } catch (e) {
      console.error("[app] Failed to fetch model state:", e);
    }
  },

  renderVariants(modelState) {
    const variantSelect = document.getElementById("variant-select");
    if (!variantSelect) return;

    variantSelect.innerHTML = "";

    const variants = modelState.transformers || [];
    if (!variants.length) {
      variantSelect.innerHTML = `<option>${t("common.loading")}</option>`;
      return;
    }

    // Check if any files are missing — if so, show download button
    const hasMissing = variants.some(v => v.state === "missing" || v.state === "downloadable")
                    || !modelState.base_model_installed;
    this.updateDownloadButton(hasMissing, modelState);

    // Group: recommended first
    const recommended = variants.filter(v => v.recommended);
    const others = variants.filter(v => !v.recommended);

    if (recommended.length) {
      const grp = document.createElement("optgroup");
      grp.label = t("model.recommended");
      for (const v of recommended) {
        const opt = document.createElement("option");
        opt.value = v.id;
        opt.textContent = `${v.backend} · ${v.steps} steps · ${v.size_gb.toFixed(1)} GB · ${v.state_label}`;
        grp.appendChild(opt);
      }
      variantSelect.appendChild(grp);
    }

    if (others.length) {
      const grp = document.createElement("optgroup");
      grp.label = t("model.available");
      for (const v of others) {
        const opt = document.createElement("option");
        opt.value = v.id;
        opt.textContent = `${v.backend} · ${v.steps} steps · ${v.size_gb.toFixed(1)} GB · ${v.state_label}`;
        grp.appendChild(opt);
      }
      variantSelect.appendChild(grp);
    }

    // Auto-select first recommended
    if (recommended[0]) {
      variantSelect.value = recommended[0].id;
      this.state.selectedVariant = recommended[0].id;
    } else if (variants[0]) {
      variantSelect.value = variants[0].id;
      this.state.selectedVariant = variants[0].id;
    }
  },

  updateDownloadButton(showButton, modelState) {
    // Find or create a download button next to the variant select
    let dlBtn = document.getElementById("download-model-btn");
    if (!showButton) {
      if (dlBtn) dlBtn.style.display = "none";
      return;
    }
    if (!dlBtn) {
      const container = document.querySelector(".controls-row");
      if (!container) return;
      dlBtn = document.createElement("button");
      dlBtn.id = "download-model-btn";
      dlBtn.className = "generate-btn";
      dlBtn.style.cssText = "background: var(--info); height: 36px; padding: 0 16px; font-size: 12px;";
      dlBtn.textContent = "⬇ Download model";
      dlBtn.addEventListener("click", () => this.downloadModel());
      container.appendChild(dlBtn);
    }
    dlBtn.style.display = "flex";
    const totalSize = (modelState.transformers || [])
      .filter(v => v.recommended)
      .reduce((sum, v) => sum + (v.size_gb || 0), 0);
    dlBtn.textContent = `⬇ Download (~${totalSize.toFixed(1)} GB)`;
  },

  async downloadModel() {
    if (!this.state.selectedModel) return;
    const dlBtn = document.getElementById("download-model-btn");
    if (dlBtn) { dlBtn.disabled = true; dlBtn.textContent = "Downloading..."; }

    try {
      const result = await this.apiPost(
        `/api/models/${this.state.selectedModel}/download`,
        { variant_id: this.state.selectedVariant }
      );
      this.toast("info", "Downloading model files...");
      // Poll for download progress
      this.startDownloadPolling(result.job_id);
    } catch (e) {
      this.toast("error", "Download failed: " + e.message);
      if (dlBtn) { dlBtn.disabled = false; dlBtn.textContent = "⬇ Download model"; }
    }
  },

  startDownloadPolling(jobId) {
    const dlBtn = document.getElementById("download-model-btn");
    const overlay = document.getElementById("progress-overlay");
    if (overlay) overlay.style.display = "flex";

    let pollCount = 0;
    const MAX_POLLS = 3600;  // 30 minutes at 500ms interval — fail-safe against stuck jobs

    const interval = setInterval(async () => {
      pollCount++;
      if (pollCount > MAX_POLLS) {
        clearInterval(interval);
        if (dlBtn) { dlBtn.disabled = false; dlBtn.textContent = "⬇ Download model"; }
        if (overlay) overlay.style.display = "none";
        this.toast("error", "Download timed out (30 min) — check server logs");
        return;
      }
      try {
        const status = await this.apiGet(`/api/jobs/${jobId}`);
        this.updateProgress(status);

        if (status.status === "complete") {
          clearInterval(interval);
          if (dlBtn) { dlBtn.disabled = false; dlBtn.style.display = "none"; }
          if (overlay) overlay.style.display = "none";
          this.toast("success", "Model files downloaded ✓");
          // Refresh model state
          await this.onModelChange();
        } else if (status.status === "error") {
          clearInterval(interval);
          if (dlBtn) { dlBtn.disabled = false; dlBtn.textContent = "⬇ Download model"; }
          if (overlay) overlay.style.display = "none";
          this.toast("error", "Download failed: " + (status.error || status.message || ""));
        } else if (status.status === "cancelled") {
          clearInterval(interval);
          if (dlBtn) { dlBtn.disabled = false; dlBtn.textContent = "⬇ Download model"; }
          if (overlay) overlay.style.display = "none";
          this.toast("info", "Download cancelled");
        }
      } catch (e) {
        console.error("[app] Download polling failed:", e);
        // If we get HTTP 404 (job not found), stop polling
        clearInterval(interval);
        if (dlBtn) { dlBtn.disabled = false; dlBtn.textContent = "⬇ Download model"; }
        if (overlay) overlay.style.display = "none";
        this.toast("error", "Download job not found — try again");
      }
    }, 500);
  },

  // ---------- Generation ----------
  async onGenerate() {
    const prompt = document.getElementById("prompt-input")?.value?.trim();
    if (!prompt) {
      this.toast("warning", t("prompt.placeholder"));
      return;
    }

    if (!this.state.selectedModel) {
      this.toast("warning", t("controls.model"));
      return;
    }

    if (!this.state.engineStatus || this.state.engineStatus.status === "missing") {
      this.toast("error", t("errors.engine_not_installed"));
      this.switchTab("settings");
      return;
    }

    const res = this.parseResolution();

    // Collect params
    const params = {
      model_id: this.state.selectedModel,
      variant_id: this.state.selectedVariant,
      prompt: prompt,
      seed: this.parseSeed(),
      width: res.width,
      height: res.height,
      steps: parseInt(document.getElementById("steps-input")?.value || "4", 10),
      backend: "lighting",
    };

    if (this.state.img2imgRef) {
      params.init_image_path = this.state.img2imgRef.path;
      params.init_image_strength = parseFloat(
        document.getElementById("img2img-strength")?.value || "0.7"
      );
    }

    // Disable button + show progress
    this.setGenerating(true);

    try {
      // Start generation
      const result = await this.apiPost("/api/generate", params);
      this.state.currentJob = result.job_id;

      // Poll for status
      this.startPolling(result.job_id);

    } catch (e) {
      console.error("[app] Generate failed:", e);
      this.toast("error", t("errors.missing_files") + ": " + e.message);
      this.setGenerating(false);
    }
  },

  parseResolution() {
    const val = document.getElementById("resolution-select")?.value?.trim() || "1024x1024";
    if (val.includes("x")) {
      const parts = val.split("x");
      const w = parseInt(parts[0], 10);
      const h = parseInt(parts[1], 10);
      return {
        width: !isNaN(w) && w > 0 ? w : 1024,
        height: !isNaN(h) && h > 0 ? h : 1024,
      };
    }
    const num = parseInt(val, 10);
    const sz = !isNaN(num) && num > 0 ? num : 1024;
    return { width: sz, height: sz };
  },

  parseSeed() {
    const val = document.getElementById("seed-input")?.value?.trim();
    if (!val || val === "random" || val === t("controls.seed.random")) {
      return null;  // Backend will randomize
    }
    const n = parseInt(val, 10);
    return isNaN(n) ? null : n;
  },

  startPolling(jobId) {
    if (this.state.pollInterval) {
      clearInterval(this.state.pollInterval);
    }

    let pollCount = 0;
    const MAX_POLLS = 9000;  // 30 min at 200ms — fail-safe against stuck jobs

    this.state.pollInterval = setInterval(async () => {
      pollCount++;
      if (pollCount > MAX_POLLS) {
        this.stopPolling();
        this.setGenerating(false);
        this.toast("error", "Generation timed out (30 min) — check server logs");
        return;
      }
      try {
        const status = await this.apiGet(`/api/jobs/${jobId}`);
        this.updateProgress(status);

        if (status.status === "complete") {
          this.stopPolling();
          this.setGenerating(false);
          this.showPreview(status.image_path);
          this.toast("success", t("toast.generation_complete"));
        } else if (status.status === "error") {
          this.stopPolling();
          this.setGenerating(false);
          this.toast("error", t("toast.generation_error") + ": " + (status.error || status.message || ""));
        } else if (status.status === "cancelled") {
          this.stopPolling();
          this.setGenerating(false);
          this.toast("info", t("toast.cancelled"));
        }
      } catch (e) {
        console.error("[app] Polling failed:", e);
        // If job endpoint returns 404 (job lost), stop polling
        this.stopPolling();
        this.setGenerating(false);
        this.toast("error", "Generation job lost — try again");
      }
    }, 200);  // Poll every 200ms
  },

  stopPolling() {
    if (this.state.pollInterval) {
      clearInterval(this.state.pollInterval);
      this.state.pollInterval = null;
    }
  },

  async cancelJob() {
    if (!this.state.currentJob) return;
    try {
      await this.apiPost(`/api/jobs/${this.state.currentJob}/cancel`);
    } catch (e) {
      console.error("[app] Cancel failed:", e);
    }
  },

  updateProgress(status) {
    const overlay = document.getElementById("progress-overlay");
    const fill = document.getElementById("progress-fill");
    const text = document.getElementById("progress-text");

    if (!overlay || !fill || !text) return;

    overlay.style.display = "flex";

    const pct = (status.progress || 0) * 100;
    fill.style.width = `${pct}%`;

    let msg = t(`status.${status.status}`) || status.status;
    if (status.step && status.total_steps) {
      msg += ` · ${status.step}/${status.total_steps}`;
    }
    if (status.message) {
      msg += ` · ${status.message}`;
    }
    text.textContent = msg;
  },

  setGenerating(busy) {
    const btn = document.getElementById("generate-btn");
    const overlay = document.getElementById("progress-overlay");
    if (!btn) return;

    if (busy) {
      btn.classList.add("busy");
      btn.disabled = true;
      btn.textContent = t("controls.busy");
      if (overlay) overlay.style.display = "flex";
    } else {
      btn.classList.remove("busy");
      btn.disabled = false;
      btn.textContent = t("controls.generate");
      if (overlay) overlay.style.display = "none";
    }
  },

  showPreview(imagePath) {
    const placeholder = document.getElementById("preview-placeholder");
    const wrap = document.getElementById("preview-image-wrap");
    const img = document.getElementById("preview-img");

    if (placeholder) placeholder.style.display = "none";
    if (wrap) wrap.style.display = "flex";
    if (img) {
      // imagePath is an absolute system path like "G:\\...\\outputs\\filename.png".
      // The server mounts /outputs/ as a static directory (app.py), so we only
      // need the filename. Cache-bust query forces the browser to reload.
      const filename = imagePath.split(/[\/\\]/).pop();
      const url = `/outputs/${encodeURIComponent(filename)}?t=${Date.now()}`;
      img.src = url;
      this.state.lastPreviewName = filename;
      this.state.lastPreviewUrl = url;
    }
  },

  // ---------- History ----------
  async loadHistory() {
    try {
      const items = await this.apiGet("/api/history");
      this.state.historyItems = items || [];
      const countEl = document.getElementById("history-count");
      if (countEl) countEl.textContent = this.state.historyItems.length;
      this.renderHistory();
    } catch (e) {
      console.error("[app] Failed to load history:", e);
      this.toast("error", t("errors.history_load_failed") + ": " + e.message);
    }
  },

  renderHistory() {
    const grid = document.getElementById("history-grid");
    if (!grid) return;

    const query = (document.getElementById("history-search")?.value || "").toLowerCase().trim();
    let items = this.state.historyItems || [];

    if (query) {
      items = items.filter(item =>
        (item.prompt || "").toLowerCase().includes(query) ||
        (item.model_id || "").toLowerCase().includes(query) ||
        (item.name || "").toLowerCase().includes(query)
      );
    }

    grid.innerHTML = "";

    if (!items.length) {
      grid.innerHTML = `
        <div class="preview-placeholder" style="grid-column: 1 / -1;">
          <div class="placeholder-text" data-i18n="history.empty">${t("history.empty")}</div>
        </div>
      `;
      return;
    }

    for (const item of items) {
      const card = document.createElement("div");
      card.className = "history-item";

      const timeStr = item.modified ? new Date(item.modified * 1000).toLocaleDateString(undefined, {
        month: "short", day: "numeric", hour: "2-digit", minute: "2-digit"
      }) : "";

      const promptDisplay = item.prompt ? item.prompt : item.name;

      card.innerHTML = `
        <div class="history-item-thumb-wrapper">
          <img src="${escapeHtml(item.url)}" alt="${escapeHtml(promptDisplay)}" loading="lazy">
          <div class="history-item-overlay">
            <div class="history-item-prompt-preview">${escapeHtml(promptDisplay)}</div>
          </div>
        </div>
        <div class="history-item-info">
          <div class="history-item-title" title="${escapeHtml(promptDisplay)}">${escapeHtml(promptDisplay)}</div>
          <div class="history-item-meta">
            <span>${escapeHtml(item.model_id || "QuantFunc")}</span>
            <span>${timeStr}</span>
          </div>
        </div>
      `;

      card.addEventListener("click", () => this.openHistoryModal(item));
      grid.appendChild(card);
    }
  },

  openHistoryModal(item) {
    this.state.selectedHistoryItem = item;
    const modal = document.getElementById("history-modal");
    if (!modal) return;

    const img = document.getElementById("history-modal-img");
    const promptEl = document.getElementById("history-modal-prompt");
    const modelEl = document.getElementById("history-modal-model");
    const seedEl = document.getElementById("history-modal-seed");
    const resEl = document.getElementById("history-modal-res");
    const stepsEl = document.getElementById("history-modal-steps");
    const backendEl = document.getElementById("history-modal-backend");
    const timeEl = document.getElementById("history-modal-time");

    if (img) img.src = item.url;
    if (promptEl) promptEl.textContent = item.prompt || t("history.no_prompt");
    if (modelEl) modelEl.textContent = item.model_id || "—";
    if (seedEl) seedEl.textContent = item.seed !== null && item.seed !== undefined ? item.seed : "—";
    if (resEl) resEl.textContent = item.width && item.height ? `${item.width} × ${item.height}` : "—";
    if (stepsEl) stepsEl.textContent = item.steps || "—";
    if (backendEl) backendEl.textContent = item.backend || "—";
    if (timeEl) timeEl.textContent = item.generation_time_sec ? `${item.generation_time_sec}s` : "—";

    // Bind modal action buttons
    const reuseBtn = document.getElementById("history-modal-reuse-btn");
    const sendImg2ImgBtn = document.getElementById("history-modal-send-img2img-btn");
    const copyBtn = document.getElementById("history-modal-copy-btn");
    const deleteBtn = document.getElementById("history-modal-delete-btn");

    if (reuseBtn) {
      reuseBtn.onclick = () => {
        this.reuseHistoryItem(item);
        this.closeHistoryModal();
      };
    }

    if (sendImg2ImgBtn) {
      sendImg2ImgBtn.onclick = () => {
        this.setImg2ImgRef({ path: item.name, url: item.url });
        this.closeHistoryModal();
      };
    }

    if (copyBtn) {
      copyBtn.onclick = () => {
        if (item.prompt) {
          navigator.clipboard.writeText(item.prompt)
            .then(() => this.toast("success", t("history.copy_prompt") + " ✓"))
            .catch(() => this.toast("error", t("errors.clipboard_denied")));
        }
      };
    }

    if (deleteBtn) {
      deleteBtn.onclick = async () => {
        if (confirm(t("history.confirm_delete"))) {
          await this.deleteHistoryItem(item.name);
          this.closeHistoryModal();
        }
      };
    }

    modal.style.display = "flex";
  },

  closeHistoryModal() {
    const modal = document.getElementById("history-modal");
    if (modal) modal.style.display = "none";
    this.state.selectedHistoryItem = null;
  },

  reuseHistoryItem(item) {
    if (!item) return;

    // Set prompt
    const promptInput = document.getElementById("prompt-input");
    if (promptInput && item.prompt) {
      promptInput.value = item.prompt;
    }

    // Set model
    if (item.model_id) {
      const modelSelect = document.getElementById("model-select");
      if (modelSelect) {
        modelSelect.value = item.model_id;
        this.onModelChange();
      }
    }

    // Set resolution
    if (item.width) {
      const resSelect = document.getElementById("resolution-select");
      if (resSelect) {
        const combo = `${item.width}x${item.height || item.width}`;
        const square = item.width.toString();
        if (resSelect.querySelector(`option[value="${combo}"]`)) {
          resSelect.value = combo;
        } else if (resSelect.querySelector(`option[value="${square}"]`)) {
          resSelect.value = square;
        }
      }
    }

    // Set seed
    if (item.seed !== null && item.seed !== undefined) {
      const seedInput = document.getElementById("seed-input");
      if (seedInput) seedInput.value = item.seed;
    }

    // Show preview of selected image in preview area
    this.showPreview(item.name);

    // Switch to generate tab
    this.switchTab("generate");
    this.toast("info", t("history.reuse") + " ✓");
  },

  async deleteHistoryItem(filename) {
    try {
      const res = await fetch(`/api/history/${encodeURIComponent(filename)}`, { method: "DELETE" });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      this.toast("success", t("common.delete") + " ✓");
      await this.loadHistory();
    } catch (e) {
      console.error("[app] Delete history error:", e);
      this.toast("error", t("errors.history_delete_failed") + ": " + e.message);
    }
  },

  // ---------- Toast notifications ----------
  toast(kind, message) {
    let container = document.querySelector(".toast-container");
    if (!container) {
      container = document.createElement("div");
      container.className = "toast-container";
      document.body.appendChild(container);
    }
    const toast = document.createElement("div");
    toast.className = `toast ${kind}`;
    toast.textContent = message;
    container.appendChild(toast);

    // Auto-remove after 4 seconds
    setTimeout(() => {
      toast.style.opacity = "0";
      toast.style.transform = "translateX(20px)";
      toast.style.transition = "all 200ms";
      setTimeout(() => toast.remove(), 200);
    }, 4000);
  },

  // ---------- Img2Img ----------
  async uploadImg2ImgFile(file) {
    const formData = new FormData();
    formData.append("file", file);

    this.toast("info", t("common.loading"));
    try {
      const res = await fetch("/api/upload", {
        method: "POST",
        body: formData,
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(err.detail || `HTTP ${res.status}`);
      }
      const data = await res.json();
      this.setImg2ImgRef({
        path: data.path,
        url: data.url,
        filename: data.filename,
      });
      this.toast("success", t("img2img.title") + " ✓");
    } catch (e) {
      console.error("[app] Upload failed:", e);
      this.toast("error", t("errors.upload_failed") + ": " + e.message);
    }
  },

  setImg2ImgRef(refData) {
    this.state.img2imgRef = refData;
    const container = document.getElementById("img2img-container");
    const thumb = document.getElementById("img2img-thumb");

    if (refData && refData.url) {
      if (thumb) thumb.src = refData.url;
      if (container) container.style.display = "flex";
      this.switchTab("generate");
    } else {
      if (container) container.style.display = "none";
    }
  },

  clearImg2ImgRef() {
    this.state.img2imgRef = null;
    const container = document.getElementById("img2img-container");
    if (container) container.style.display = "none";
  },

  updateStrengthLabel(val) {
    const label = document.getElementById("img2img-strength-val");
    if (label) label.textContent = parseFloat(val).toFixed(2);
  },

  sendCurrentPreviewToImg2Img() {
    if (!this.state.lastPreviewName && !this.state.lastPreviewUrl) {
      const img = document.getElementById("preview-img");
      if (img && img.src && !img.src.endsWith("#")) {
        const src = img.src;
        const filename = src.split("/").pop();
        if (filename) {
          this.setImg2ImgRef({
            path: filename,
            url: src,
            filename: filename,
          });
          this.toast("success", t("img2img.send_to") + " ✓");
          return;
        }
      }
      this.toast("warning", t("history.empty"));
      return;
    }
    this.setImg2ImgRef({
      path: this.state.lastPreviewName || (this.state.lastPreviewUrl ? this.state.lastPreviewUrl.split("/").pop() : ""),
      url: this.state.lastPreviewUrl,
    });
    this.toast("success", t("img2img.send_to") + " ✓");
  },
};

// Global handlers for inline HTML onclick attributes
window.loadHistory = () => App.loadHistory();
window.closeHistoryModal = () => App.closeHistoryModal();
window.clearImg2ImgRef = () => App.clearImg2ImgRef();
window.updateStrengthLabel = (val) => App.updateStrengthLabel(val);
window.sendCurrentPreviewToImg2Img = () => App.sendCurrentPreviewToImg2Img();

// ---------- Boot ----------
document.addEventListener("DOMContentLoaded", () => {
  App.init();
});

// Export for inline scripts
window.App = App;
