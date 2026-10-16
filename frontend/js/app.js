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
    modelPickerSearch: "",
    modelPickerFilter: "all",
    isModelPickerOpen: false,
    selectedVae: "texture_fix",
  },

  // ---------- Init ----------
  async init() {
    console.log("[app] LazySD UI initializing...");

    // Initialize sidebar state
    this.initSidebar();

    // Load i18n first
    await window.initI18n();

    // Load initial config from backend
    await this.loadConfig();

    // Render initial UI
    this.renderGpuIndicator();
    this.renderSystemStats(this.state.config?.system_stats);
    this.startStatsPolling();
    this.renderModels();
    this.updateEngineStatusUI();
    this.bindEvents();

    console.log("[app] Ready. GPU:", this.state.gpuInfo?.name || "not detected");
  },

  toggleSidebar() {
    const shell = document.querySelector(".app-shell");
    if (!shell) return;
    const isCollapsed = shell.classList.toggle("sidebar-collapsed");
    try {
      localStorage.setItem("lazysd_sidebar_collapsed", isCollapsed ? "1" : "0");
    } catch (_) {}
  },

  initSidebar() {
    try {
      const saved = localStorage.getItem("lazysd_sidebar_collapsed");
      if (saved === "1") {
        document.querySelector(".app-shell")?.classList.add("sidebar-collapsed");
      }
    } catch (_) {}
  },

  async toggleQuantFuncModels(checked) {
    try {
      await this.apiPatch("/api/settings", { enable_quantfunc_models: checked });
      if (this.state.settings) {
        this.state.settings.enable_quantfunc_models = checked;
      }
      this.toast("info", checked ? "Модели QuantFunc включены" : "Модели QuantFunc отключены");
      await this.refreshConfig();
      if (this.state.isModelPickerOpen) {
        this.renderModelPickerModal();
      }
    } catch (err) {
      console.error("[app] Failed to toggle QuantFunc models:", err);
      this.toast("error", `Ошибка: ${err.message}`);
    }
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
      this.state.config = config;
      this.state.settings = config.settings;
      this.state.gpuInfo = config.gpu;
      this.state.engineStatus = config.engine_status;
      this.state.models = config.models || [];
      if (config.settings?.qwen_vae_type) {
        this.state.selectedVae = config.settings.qwen_vae_type;
      }
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
      this.state.config = config;
      this.state.settings = config.settings;
      this.state.gpuInfo = config.gpu;
      this.state.engineStatus = config.engine_status;
      this.state.models = config.models || [];
      if (config.settings?.qwen_vae_type) {
        this.state.selectedVae = config.settings.qwen_vae_type;
      }
      window.QF_SETTINGS = config.settings;
      this.renderGpuIndicator();
      this.renderSystemStats(config.system_stats);
      this.renderModels();
      this.updateEngineStatusUI();
    } catch (e) {
      console.error("[app] Failed to refresh config:", e);
    }
  },

  updateEngineStatusUI() {
    // Update Settings tab engine status display after download.
    const config = this.state.config || window.QF_INITIAL_CONFIG || {};
    const engines = config.engines || {};

    const qf17El = document.getElementById("setting-engine-qf17-status");
    if (qf17El) {
      const qf17 = engines.quantfunc_0_17 || {};
      const isInst = qf17.installed ?? (this.state.engineStatus?.status === "ok");
      qf17El.innerHTML = isInst
        ? '<span class="state-badge installed">✓ installed (v0.0.17)</span>'
        : '<span class="state-badge missing">✗ missing</span>';
    }

    const qf12El = document.getElementById("setting-engine-qf12-status");
    if (qf12El) {
      const qf12 = engines.quantfunc_0_12 || {};
      const isInst = qf12.installed ?? (config.installed_engines?.includes("0.0.12"));
      qf12El.innerHTML = isInst
        ? '<span class="state-badge installed">✓ installed (v0.0.12)</span>'
        : '<span class="state-badge missing">✗ missing</span>';
    }

    const sdCppEl = document.getElementById("setting-engine-sdcpp-status");
    if (sdCppEl) {
      const sd = engines.sd_cpp || {};
      const isInst = sd.installed ?? config.sd_cpp_installed;
      const ver = sd.version || config.sd_cpp_version || 'ready';
      sdCppEl.innerHTML = isInst
        ? `<span class="state-badge installed">✓ installed (${ver})</span>`
        : '<span class="state-badge missing">✗ missing</span>';
    }

    const sdnqEl = document.getElementById("setting-engine-sdnq-status");
    if (sdnqEl) {
      const sdnq = engines.sdnq || {};
      const isInst = sdnq.installed ?? config.sdnq_installed;
      const ver = sdnq.version || config.sdnq_version || 'ready';
      sdnqEl.innerHTML = isInst
        ? `<span class="state-badge installed">✓ installed (${ver})</span>`
        : '<span class="state-badge missing">✗ missing</span>';
    }

    const qfToggleEl = document.getElementById("setting-enable-quantfunc");
    const qfBadgeEl = document.getElementById("setting-qf-toggle-badge");
    const qfEnabled = !!(this.state.settings?.enable_quantfunc_models);
    if (qfToggleEl) {
      qfToggleEl.checked = qfEnabled;
    }
    if (qfBadgeEl) {
      qfBadgeEl.innerHTML = qfEnabled
        ? '<span class="state-badge installed">✓ Включено</span>'
        : '<span class="state-badge missing">Отключено</span>';
    }
  },

  // ---------- Renderers ----------
  renderGpuIndicator() {
    const nameEl = document.getElementById("gpu-name");
    const dotEl = document.getElementById("gpu-status");
    if (!nameEl || !dotEl) return;

    if (this.state.gpuInfo && this.state.gpuInfo.name) {
      const gpu = this.state.gpuInfo;
      nameEl.textContent = `${gpu.name} · ${gpu.tier} · ${Math.round(gpu.vram_mb / 1024)} GB`;
      dotEl.classList.remove("offline");
      dotEl.title = `SM ${gpu.sm} · Driver ${gpu.driver_version}`;
    } else {
      nameEl.textContent = t("gpu.not_detected");
      dotEl.classList.add("offline");
    }
  },

  renderSystemStats(stats) {
    if (!stats) return;
    const vramEl = document.getElementById("vram-stat-text");
    const ramEl = document.getElementById("ram-stat-text");

    if (vramEl) {
      if (stats.vram_total_gb > 0) {
        const vramPct = stats.vram_percent || 0;
        const color = vramPct > 90 ? "#f87171" : (vramPct > 75 ? "#fbbf24" : "var(--text-primary)");
        vramEl.innerHTML = `VRAM: <strong style="color:${color}">${stats.vram_used_gb.toFixed(1)}/${stats.vram_total_gb.toFixed(1)} GB</strong> (${Math.round(vramPct)}%)`;
      } else {
        vramEl.textContent = "VRAM: —";
      }
    }

    if (ramEl) {
      if (stats.ram_total_gb > 0) {
        const ramPct = stats.ram_percent || 0;
        const color = ramPct > 90 ? "#f87171" : (ramPct > 80 ? "#fbbf24" : "var(--text-primary)");
        ramEl.innerHTML = `RAM: <strong style="color:${color}">${stats.ram_used_gb.toFixed(1)}/${stats.ram_total_gb.toFixed(1)} GB</strong> (${Math.round(ramPct)}%)`;
      } else {
        ramEl.textContent = "RAM: —";
      }
    }
  },

  startStatsPolling() {
    if (this._statsInterval) clearInterval(this._statsInterval);
    this._statsInterval = setInterval(async () => {
      if (document.hidden) return;
      try {
        const stats = await this.apiGet("/api/system/stats");
        this.renderSystemStats(stats);
      } catch (_) {}
    }, 3000);
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
    let compatible = this.state.models.filter(m =>
      m.requirements.gpu_min_sm <= gpuSm
    );

    // Graceful fallback: if GPU SM is not detected or no models match, show all models
    if (!compatible.length) {
      compatible = this.state.models;
    }

    if (!compatible.length) {
      select.innerHTML = `<option value="">No models available</option>`;
      return;
    }

    for (const model of compatible) {
      const opt = document.createElement("option");
      opt.value = model.id;
      if (model.disabled) {
        opt.disabled = true;
        opt.textContent = `⛔ ${model.name} (${model.family}) [Отключено]`;
      } else {
        opt.textContent = `${model.name} (${model.family})`;
      }
      select.appendChild(opt);
    }

    // Auto-select first non-disabled model
    const firstActive = compatible.find(m => !m.disabled) || compatible[0];
    if (firstActive) {
      select.value = firstActive.id;
      this.state.selectedModel = firstActive.id;
      this.onModelChange();
    }
    this.updateModelTrigger();
    this.renderModelsTab();
  },

  updateModelTrigger() {
    const nameEl = document.getElementById("model-trigger-name");
    const subEl = document.getElementById("model-trigger-sub");
    const badgeEl = document.getElementById("model-trigger-badge");
    if (!nameEl) return;

    const current = this.state.models.find(m => m.id === this.state.selectedModel);
    if (!current) {
      nameEl.textContent = "—";
      if (subEl) subEl.textContent = "—";
      return;
    }

    nameEl.textContent = current.name;
    const variant = this.state.selectedVariant || current.transformers?.[0]?.id || "default";
    let engineVer = "";
    if (current.engine === "sd_cpp") {
      engineVer = "SD.CPP";
    } else if (current.engine === "sdnq") {
      engineVer = "SDNQ";
    } else if (current.engine_version) {
      engineVer = `v${current.engine_version}`;
    }
    const steps = document.getElementById("steps-input")?.value || current.transformers?.[0]?.steps || 4;
    
    if (subEl) {
      if (current.disabled) {
        subEl.innerHTML = `<span style="color: #f87171;">⚠️ Отключено (веса с дефектами)</span>`;
      } else {
        const parts = [];
        if (engineVer) parts.push(engineVer);
        parts.push(`${steps} steps`);
        if (variant) parts.push(variant.replace(/-(?:50x|40x|30x-below)(?:-base)?/g, ""));
        subEl.textContent = parts.join(" · ");
      }
    }

    if (badgeEl) {
      if (current.disabled) {
        badgeEl.textContent = "⛔ Отключено";
      } else if (current.engine === "sd_cpp") {
        badgeEl.textContent = "⚡ SD.CPP";
      } else if (current.engine === "sdnq") {
        badgeEl.textContent = "⚡ SDNQ";
      } else {
        badgeEl.textContent = current.engine_version ? `⚡ v${current.engine_version}` : "⚡";
      }
    }

    // Dynamic CFG & Noise recommendations
    let recCfg = 1.0;
    if (current.default_cfg !== undefined && current.default_cfg !== null) {
      recCfg = Number(current.default_cfg);
    } else if (current.id.includes("turbo") || current.engine === "sdnq") {
      recCfg = 1.0;
    } else if (current.id.startsWith("qwen_image_2_1") || current.family === "qwen") {
      recCfg = 3.0;
    } else if (current.family === "sdxl" || current.id.includes("sdxl")) {
      recCfg = 7.0;
    } else if (current.family === "flux" && current.id.includes("dev")) {
      recCfg = 3.5;
    } else {
      recCfg = 1.0;
    }

    const cfgBadge = document.getElementById("cfg-rec-badge");
    if (cfgBadge) {
      const recLabel = t("controls.rec") || "Реком.";
      cfgBadge.textContent = `(${recLabel}: ${recCfg.toFixed(1)})`;
    }

    const noiseBadge = document.getElementById("noise-rec-badge");
    if (noiseBadge) {
      const recLabel = t("controls.rec") || "Реком.";
      noiseBadge.textContent = `(${recLabel}: 0.0)`;
    }
  },

  // ---------- Model Hub Modal ----------
  openModelPickerModal() {
    const modal = document.getElementById("model-picker-modal");
    if (!modal) return;
    this.state.isModelPickerOpen = true;
    modal.style.display = "flex";
    this.renderModelPickerModal();
    const searchInput = document.getElementById("model-picker-search");
    if (searchInput) {
      searchInput.value = this.state.modelPickerSearch || "";
      searchInput.focus();
    }
  },

  closeModelPickerModal() {
    const modal = document.getElementById("model-picker-modal");
    if (!modal) return;
    this.state.isModelPickerOpen = false;
    modal.style.display = "none";
  },

  onModelPickerSearch(query) {
    this.state.modelPickerSearch = (query || "").toLowerCase().trim();
    this.renderModelPickerModal();
  },

  setModelPickerFilter(filter, btnEl) {
    this.state.modelPickerFilter = filter;
    document.querySelectorAll("#model-picker-filters .filter-chip").forEach(el => {
      el.classList.toggle("active", el === btnEl);
    });
    this.renderModelPickerModal();
  },

  renderModelPickerModal() {
    const grid = document.getElementById("model-picker-grid");
    if (!grid) return;

    if (!this.state.models || !this.state.models.length) {
      grid.innerHTML = `<div class="placeholder-text">${t("common.loading")}</div>`;
      return;
    }

    const gpuSm = this.state.gpuInfo?.sm || 0;
    const query = this.state.modelPickerSearch || "";
    const filter = this.state.modelPickerFilter || "all";

    const filtered = this.state.models.filter(m => {
      // Search query
      if (query) {
        const matchesName = m.name?.toLowerCase().includes(query);
        const matchesFamily = m.family?.toLowerCase().includes(query);
        const matchesDesc = m.description?.toLowerCase().includes(query);
        if (!matchesName && !matchesFamily && !matchesDesc) return false;
      }
      // Category filter
      if (filter === "installed") {
        return m.installed === true;
      } else if (filter === "custom") {
        return m.is_custom === true;
      } else if (filter === "klein") {
        return m.family === "klein" || m.id.includes("klein");
      } else if (filter === "qwen") {
        return m.family === "qwen" || m.id.includes("qwen");
      } else if (filter === "z-image") {
        return m.family === "z-image" || m.id.includes("z_image");
      } else if (filter === "sd_cpp") {
        return m.engine === "sd_cpp" || m.id.includes("gguf");
      } else if (filter === "sdnq") {
        return m.engine === "sdnq" || m.id.includes("sdnq");
      }
      return true;
    });

    if (!filtered.length) {
      grid.innerHTML = `<div class="placeholder-text" style="grid-column: 1 / -1; padding: 40px 0;">${t("models.no_results")}</div>`;
      return;
    }

    grid.innerHTML = "";
    for (const model of filtered) {
      const isSelected = model.id === this.state.selectedModel;
      const isInstalled = model.installed === true;
      const isDisabled = model.disabled === true;

      const card = document.createElement("div");
      card.className = `model-picker-card ${isSelected ? "active" : ""} ${isDisabled ? "model-card-qf-disabled" : ""}`;

      let engineBadge = "";
      if (model.is_custom) {
        engineBadge = `<span class="badge-custom">🎨 Custom</span>`;
      } else if (model.engine === "sd_cpp") {
        engineBadge = `<span class="badge-engine sd-cpp" style="background: rgba(16, 185, 129, 0.15); color: #34d399; border-color: rgba(16, 185, 129, 0.3);">⚡ Engine: SD.CPP</span>`;
      } else if (model.engine === "sdnq") {
        engineBadge = `<span class="badge-engine sdnq" style="background: rgba(139, 92, 246, 0.18); color: #c4b5fd; border-color: rgba(139, 92, 246, 0.35);">⚡ Engine: SDNQ</span>`;
      } else if (model.engine_version) {
        engineBadge = `<span class="badge-engine ${model.engine_version === '0.0.12' ? 'v12' : ''}">⚡ Engine v${escapeHtml(model.engine_version)}</span>`;
      }

      let statusBadge = "";
      if (isDisabled) {
        statusBadge = `<span class="badge-unsupported">⚠️ Не поддерживается</span>`;
      } else if (isInstalled) {
        statusBadge = `<span class="badge-installed">✓ ${t("models.status_ready")}</span>`;
      } else {
        statusBadge = `<span class="badge-missing">⬇ ${t("models.status_download")}</span>`;
      }

      const keyBadge = model.key_protected
        ? `<span class="model-badge key-protected" style="font-size: 10px; padding: 2px 6px;">🔑 Key Protected</span>`
        : "";

      const vramMin = model.requirements?.vram_min_gb ? `${model.requirements.vram_min_gb} GB` : "8 GB";
      const defaultSteps = model.transformers?.[0]?.steps || 4;

      let actionBtnHtml = "";
      let deleteBtnHtml = "";
      if (model.is_custom) {
        deleteBtnHtml = `<button class="delete-custom-btn" onclick="App.deleteCustomModel('${escapeHtml(model.id)}', event)" title="${t('custom_model.delete') || 'Delete'}">🗑 ${t('custom_model.delete') || 'Delete'}</button>`;
      }

      if (isDisabled) {
        actionBtnHtml = `<button class="card-action-btn disabled" disabled title="${escapeHtml(model.disabled_reason || 'Отключено')}">⛔ Отключено (веса с дефектами)</button>`;
      } else if (isSelected) {
        actionBtnHtml = `<button class="card-action-btn active" disabled>✓ ${t("models.active_model")}</button>`;
      } else if (isInstalled) {
        actionBtnHtml = `<button class="card-action-btn primary" onclick="App.selectModelFromModal('${escapeHtml(model.id)}')">${t("models.select_model")}</button>`;
      } else {
        actionBtnHtml = `<button class="card-action-btn download" onclick="App.downloadModelVariant('${escapeHtml(model.id)}', null, this)">⬇ ${t("models.download_and_select")}</button>`;
      }

      const varCount = model.transformers?.length || 1;
      const teCount = model.text_encoders?.length || 0;
      const varInfo = varCount > 1 ? `<span>⚡ ${varCount} ${t("models.variants_count") || "variants"}</span>` : "";
      const teInfo = teCount > 0 ? `<span>📝 ${teCount} ${t("models.encoders_count") || "encoders"}</span>` : "";

      const noticeHtml = isDisabled
        ? `<div style="margin-top: 8px; padding: 6px 10px; background: rgba(239, 68, 68, 0.1); border: 1px solid rgba(239, 68, 68, 0.25); border-radius: 6px; font-size: 11px; color: #f87171;">⚠️ ${escapeHtml(model.disabled_reason || "Отключено из-за дефектов весов upstream. Включить можно в Настройках.")}</div>`
        : "";

      card.innerHTML = `
        <div class="card-top">
          <div class="card-header-row">
            <div class="card-title-group">
              <span class="card-model-name">${escapeHtml(model.name)}</span>
              <span class="card-family-tag">${escapeHtml(model.family || model.pipeline)}</span>
            </div>
            <div class="card-badges">
              ${engineBadge}
              ${statusBadge}
              ${keyBadge}
            </div>
          </div>
          <p class="card-desc">${escapeHtml(model.description || "")}</p>
          ${noticeHtml}
          <div class="card-specs">
            <span>📼 ${t("models.vram_min")}: ${vramMin}</span>
            <span>⏱ ${t("models.steps")}: ${defaultSteps}</span>
            <span>💻 Min SM: ${model.requirements?.gpu_min_sm || 75}+</span>
            ${varInfo}
            ${teInfo}
          </div>
        </div>
        <div class="card-footer" style="display: flex; gap: 8px; align-items: center;">
          ${actionBtnHtml}
          ${deleteBtnHtml}
        </div>
      `;

      // Allow clicking card to select if installed and not disabled
      if (!isSelected && isInstalled && !isDisabled) {
        card.style.cursor = "pointer";
        card.addEventListener("click", (e) => {
          if (e.target.tagName === "BUTTON") return;
          this.selectModelFromModal(model.id);
        });
      } else if (isDisabled) {
        card.style.cursor = "not-allowed";
        card.addEventListener("click", (e) => {
          if (e.target.tagName === "BUTTON") return;
          this.toast("warning", model.disabled_reason || "Модели QuantFunc временно отключены.");
        });
      }

      grid.appendChild(card);
    }
  },

  async selectModelFromModal(modelId) {
    const model = this.state.models.find(m => m.id === modelId);
    if (model && model.disabled) {
      this.toast("warning", model.disabled_reason || "Модели QuantFunc временно отключены. Включите их в Настройках.");
      return;
    }

    this.state.selectedModel = modelId;

    const select = document.getElementById("model-select");
    if (select) {
      select.value = modelId;
    }

    if (this.state.currentTab !== "generate") {
      this.switchTab("generate");
    }

    await this.onModelChange();
    this.updateModelTrigger();
    this.closeModelPickerModal();
    this.toast("info", `${t("models.active_model")}: ${modelId}`);
  },

  renderModelsTab() {
    const listEl = document.getElementById("models-list");
    if (!listEl) return;

    if (!this.state.models.length) {
      listEl.innerHTML = `<div class="placeholder-text">${t("common.loading")}</div>`;
      return;
    }

    listEl.innerHTML = "";
    const grid = document.createElement("div");
    grid.className = "models-grid";

    for (const model of this.state.models) {
      const isDisabled = model.disabled === true;
      const card = document.createElement("div");
      card.className = `model-card ${isDisabled ? "model-card-qf-disabled" : ""}`;

      const keyBadge = model.key_protected
        ? `<span class="model-badge key-protected">🔑 Key Protected</span>`
        : `<span class="model-badge free">🔓 Free Access</span>`;
      
      const pipelineBadge = `<span class="model-badge pipeline">${escapeHtml(model.pipeline)}</span>`;
      const familyBadge = `<span class="model-badge family">${escapeHtml(model.family)}</span>`;
      let engineBadge = "";
      if (model.engine === "sd_cpp") {
        engineBadge = `<span class="badge-engine sd-cpp" style="background: rgba(16, 185, 129, 0.15); color: #34d399; border-color: rgba(16, 185, 129, 0.3);">⚡ SD.CPP</span>`;
      } else if (model.engine === "sdnq") {
        engineBadge = `<span class="badge-engine sdnq" style="background: rgba(139, 92, 246, 0.18); color: #c4b5fd; border-color: rgba(139, 92, 246, 0.35);">⚡ SDNQ</span>`;
      } else if (model.engine_version) {
        engineBadge = `<span class="badge-engine ${model.engine_version === '0.0.12' ? 'v12' : ''}">⚡ v${escapeHtml(model.engine_version)}</span>`;
      }

      let installedBadge = "";
      if (isDisabled) {
        installedBadge = `<span class="badge-unsupported">⚠️ Не поддерживается</span>`;
      } else if (model.installed) {
        installedBadge = `<span class="badge-installed">✓ ${t("models.status_ready")}</span>`;
      } else {
        installedBadge = `<span class="badge-missing">⬇ ${t("models.status_download")}</span>`;
      }

      const statusBadge = model.available === false || model.status === "coming_soon"
        ? `<span class="model-badge key-protected" style="background: rgba(245, 158, 11, 0.2); color: #f59e0b; border-color: rgba(245, 158, 11, 0.4);">⏳ Coming Soon</span>`
        : "";

      const vramStr = model.requirements?.vram_min_gb ? `${model.requirements.vram_min_gb} GB VRAM min` : "";

      let variantsHtml = "";
      if (model.transformers && model.transformers.length) {
        for (const v of model.transformers) {
          const btn = isDisabled
            ? `<button class="setting-btn" disabled style="opacity: 0.6; cursor: not-allowed;" title="${escapeHtml(model.disabled_reason || 'Отключено')}">⛔ Отключено</button>`
            : `<button class="icon-btn" data-model-id="${escapeHtml(model.id)}" data-variant-id="${escapeHtml(v.id)}" onclick="App.downloadModelVariant(this.dataset.modelId, this.dataset.variantId, this)">⬇ Download</button>`;
          variantsHtml += `
            <div class="variant-item">
              <div class="variant-info">
                <span class="variant-name">${escapeHtml(v.id)}</span>
                <span class="variant-sub">${escapeHtml(v.backend)} · ${v.steps} steps · ${(v.size_gb || 0).toFixed(1)} GB</span>
              </div>
              ${btn}
            </div>
          `;
        }
      } else {
        variantsHtml = `<div class="placeholder-text" style="padding: 8px 0; font-size: 13px; text-align: left;">Weights pending release on ModelScope</div>`;
      }

      const isSelected = model.id === this.state.selectedModel;
      let selectBtnHtml = "";
      if (isDisabled) {
        selectBtnHtml = `<button class="card-action-btn disabled" style="margin-top: 12px; width: auto;" disabled title="${escapeHtml(model.disabled_reason || 'Отключено')}">⛔ Модель отключена (веса с дефектами)</button>`;
      } else if (isSelected) {
        selectBtnHtml = `<button class="card-action-btn active" style="margin-top: 12px; width: auto;" disabled>✓ ${t("models.active_model")}</button>`;
      } else {
        selectBtnHtml = `<button class="card-action-btn primary" style="margin-top: 12px; width: auto;" onclick="App.selectModelFromModal('${escapeHtml(model.id)}')">⚡ ${t("models.select_model")}</button>`;
      }

      const noticeHtml = isDisabled
        ? `<div style="margin-top: 10px; padding: 8px 12px; background: rgba(239, 68, 68, 0.1); border: 1px solid rgba(239, 68, 68, 0.25); border-radius: 6px; font-size: 12px; color: #f87171;">⚠️ ${escapeHtml(model.disabled_reason || "Отключено из-за дефектов весов upstream. Включить можно в Настройках.")}</div>`
        : "";

      card.innerHTML = `
        <div>
          <div class="model-card-header">
            <span class="model-card-title">${escapeHtml(model.name)}</span>
            <div class="model-card-badges">
              ${engineBadge}
              ${installedBadge}
              ${statusBadge}
              ${keyBadge}
              ${pipelineBadge}
              ${familyBadge}
            </div>
          </div>
          <p class="model-card-description" style="margin-top: 8px;">${escapeHtml(model.description || "")}</p>
          ${noticeHtml}
          <div class="model-card-reqs" style="margin-top: 10px;">
            <span>💻 Min SM: ${model.requirements?.gpu_min_sm || 75}+</span>
            ${vramStr ? `<span>📼 ${vramStr}</span>` : ""}
          </div>
          <div class="model-variants-title">Transformers (${model.transformers?.length || 0})</div>
          <div class="model-variants-list">${variantsHtml}</div>
          <div style="margin-top: 8px;">
            ${selectBtnHtml}
          </div>
        </div>
      `;

      grid.appendChild(card);
    }

    listEl.appendChild(grid);
  },

  async downloadModelVariant(modelId, variantId, btnEl) {
    const model = this.state.models.find(m => m.id === modelId);
    if (model && model.disabled) {
      this.toast("warning", model.disabled_reason || "Модели QuantFunc временно отключены из-за дефектов весов. Включите их в Настройках.");
      return;
    }

    this.state.selectedModel = modelId;
    if (variantId) {
      this.state.selectedVariant = variantId;
    }
    
    // Sync model select dropdown in generate tab if present
    const modelSelect = document.getElementById("model-select");
    if (modelSelect) {
      modelSelect.value = modelId;
      await this.onModelChange(variantId || null);
    }

    if (btnEl) {
      btnEl.disabled = true;
      btnEl.textContent = "Downloading...";
    }
    await this.downloadModel();
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

    // Model picker modal trigger
    const modelPickerTrigger = document.getElementById("model-picker-trigger");
    if (modelPickerTrigger) {
      modelPickerTrigger.addEventListener("click", () => this.openModelPickerModal());
    }

    // Variant select
    const variantSelect = document.getElementById("variant-select");
    if (variantSelect) {
      variantSelect.addEventListener("change", () => {
        this.state.selectedVariant = variantSelect.value;
        const currentVariant = this.state.modelState?.transformers?.find(v => v.id === variantSelect.value);
        if (currentVariant) {
          const stepsInput = document.getElementById("steps-input");
          if (stepsInput && currentVariant.steps) {
            stepsInput.value = currentVariant.steps;
          }
        }
        this.updateModelTrigger();
        if (this.state.modelState) {
          const activeTe = this.state.modelState.text_encoders?.find(te => te.id === this.state.selectedTextEncoder);
          const activeVar = currentVariant || (this.state.modelState.transformers || [])[0];
          const activeVarMissing = activeVar && (activeVar.state === "missing" || activeVar.state === "downloadable" || activeVar.state === "corrupted");
          const hasMissing = activeVarMissing
                          || !this.state.modelState.base_model_installed
                          || (activeTe && !activeTe.installed && this.state.selectedTextEncoder !== "custom");
          this.updateDownloadButton(hasMissing, this.state.modelState);
        }
      });
    }

    // Steps input
    const stepsInput = document.getElementById("steps-input");
    if (stepsInput) {
      stepsInput.addEventListener("input", () => this.updateModelTrigger());
    }

    // CFG slider & input sync
    const cfgSlider = document.getElementById("cfg-slider");
    const cfgInput = document.getElementById("cfg-input");
    if (cfgSlider && cfgInput) {
      cfgSlider.addEventListener("input", () => {
        cfgInput.value = cfgSlider.value;
      });
      cfgInput.addEventListener("input", () => {
        cfgSlider.value = cfgInput.value;
      });
    }

    // Noise slider & input sync (+ auto-enable stochastic sampler when noise > 0)
    const noiseSlider = document.getElementById("noise-slider");
    const noiseInput = document.getElementById("noise-input");
    const autoSyncStochasticSampler = (valStr) => {
      const n = parseFloat(valStr);
      const samplerSelect = document.getElementById("sampler-select");
      if (!isNaN(n) && n > 0 && samplerSelect && samplerSelect.value === "euler") {
        samplerSelect.value = "euler_a";
      }
    };
    if (noiseSlider && noiseInput) {
      noiseSlider.addEventListener("input", () => {
        noiseInput.value = noiseSlider.value;
        autoSyncStochasticSampler(noiseSlider.value);
      });
      noiseInput.addEventListener("input", () => {
        noiseSlider.value = noiseInput.value;
        autoSyncStochasticSampler(noiseInput.value);
      });
    }

    // Negative prompt input dot indicator
    const negInput = document.getElementById("negative-prompt-input");
    const negDot = document.getElementById("negative-prompt-dot");
    if (negInput && negDot) {
      negInput.addEventListener("input", () => {
        negDot.style.display = negInput.value.trim().length > 0 ? "inline-block" : "none";
      });
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
      // ESC: close modals first, then cancel generation
      if (e.key === "Escape") {
        if (this.state.isModelPickerOpen) {
          this.closeModelPickerModal();
        } else if (this.state.selectedHistoryItem) {
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

    const modelPickerModal = document.getElementById("model-picker-modal");
    if (modelPickerModal) {
      modelPickerModal.addEventListener("click", (e) => {
        if (e.target === modelPickerModal) this.closeModelPickerModal();
      });
    }

    // Language toggle
    document.querySelectorAll(".lang-btn").forEach(btn => {
      btn.addEventListener("click", () => window.setLocale(btn.dataset.lang));
    });

    // Custom downward prompt resize handler (prevents Flexbox upward expansion bug)
    this.initPromptResize();
  },

  initPromptResize() {
    const handle = document.getElementById("prompt-resize-handle");
    const textarea = document.getElementById("prompt-input");
    const previewArea = document.querySelector("#tab-generate .preview-area");
    if (!handle || !textarea) return;

    let isDragging = false;
    let startY = 0;
    let startHeight = 0;

    handle.addEventListener("mousedown", (e) => {
      e.preventDefault();
      isDragging = true;
      startY = e.clientY;
      startHeight = textarea.offsetHeight;
      // Lock preview-area height so expanding prompt-input pushes downward instead of upward
      if (previewArea && !previewArea.dataset.lockedHeight) {
        const h = Math.max(180, previewArea.offsetHeight);
        previewArea.style.minHeight = `${h}px`;
        previewArea.style.flex = "0 0 auto";
        previewArea.dataset.lockedHeight = "1";
      }
      document.body.style.userSelect = "none";
      document.body.style.cursor = "ns-resize";
    });

    document.addEventListener("mousemove", (e) => {
      if (!isDragging) return;
      const dy = e.clientY - startY;
      const nextHeight = Math.max(76, Math.min(360, startHeight + dy));
      textarea.style.height = `${nextHeight}px`;
    });

    document.addEventListener("mouseup", () => {
      if (!isDragging) return;
      isDragging = false;
      document.body.style.userSelect = "";
      document.body.style.cursor = "";
      if (textarea.offsetHeight <= 84 && previewArea) {
        previewArea.style.minHeight = "";
        previewArea.style.flex = "";
        delete previewArea.dataset.lockedHeight;
      }
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

    // Auto-load history or models tab
    if (tabName === "history") {
      this.loadHistory();
    } else if (tabName === "models") {
      this.renderModelsTab();
    }
  },

  toggleNegativePrompt() {
    const drawer = document.getElementById("negative-prompt-drawer");
    const content = document.getElementById("negative-prompt-content");
    const arrow = document.getElementById("negative-prompt-arrow");
    const input = document.getElementById("negative-prompt-input");
    if (!content || !arrow) return;
    const isHidden = content.style.display === "none";
    content.style.display = isHidden ? "block" : "none";
    arrow.textContent = isHidden ? "▼" : "▶";
    if (drawer) {
      if (isHidden) drawer.classList.add("open");
      else drawer.classList.remove("open");
    }
    if (isHidden && input) {
      input.focus();
    }
  },

  async saveQwenVae(value) {
    this.state.selectedVae = value;
    const vaeSelect = document.getElementById("vae-select");
    if (vaeSelect) vaeSelect.value = value;
    try {
      await this.apiPatch("/api/settings", { qwen_vae_type: value });
      this.toast("success", `${t("settings.vae.qwen_title") || "Qwen VAE"}: ${value} ✓`);
    } catch (e) {
      this.toast("error", e.message);
    }
  },

  async onModelChange(preferredVariantId = null) {
    const select = document.getElementById("model-select");
    if (!select || !select.value) return;

    this.state.selectedModel = select.value;

    // Fetch model state (cache check)
    try {
      const state = await this.apiGet(`/api/models/${select.value}/state`);
      this.renderVariants(state, preferredVariantId);
    } catch (e) {
      console.error("[app] Failed to fetch model state:", e);
    }

    // Dynamic CFG recommendation sync
    const activeModel = this.state.models.find(m => m.id === select.value);
    let recCfg = 1.0;
    if (activeModel && typeof activeModel.default_cfg === "number") {
      recCfg = activeModel.default_cfg;
    } else if (activeModel && (activeModel.id.includes("turbo") || activeModel.engine === "sdnq")) {
      recCfg = 1.0;
    } else if (activeModel && (activeModel.id.startsWith("qwen_image_2_1") || activeModel.family === "qwen")) {
      recCfg = 3.0;
    } else if (activeModel && (activeModel.family === "sdxl" || activeModel.id.includes("sdxl"))) {
      recCfg = 7.0;
    } else if (activeModel && activeModel.family === "flux" && activeModel.id.includes("dev")) {
      recCfg = 3.5;
    }

    const cfgInput = document.getElementById("cfg-input");
    const cfgSlider = document.getElementById("cfg-slider");
    if (cfgInput && cfgSlider) {
      cfgInput.value = recCfg.toFixed(1);
      cfgSlider.value = recCfg.toFixed(1);
    }

    this.updateModelTrigger();
  },

  renderVariants(modelState, preferredVariantId = null) {
    this.state.modelState = modelState;
    const variantSelect = document.getElementById("variant-select");
    if (!variantSelect) return;

    variantSelect.innerHTML = "";

    const variants = modelState.transformers || [];
    if (!variants.length) {
      variantSelect.innerHTML = `<option>${t("common.loading")}</option>`;
      return;
    }

    // Text Encoder select & Guardrails
    const teGroup = document.getElementById("text-encoder-group");
    const teSelect = document.getElementById("text-encoder-select");
    const teHint = document.getElementById("text-encoder-hint");
    const customTeInput = document.getElementById("custom-text-encoder-input");
    const teLink = document.getElementById("toggle-custom-te-link");
    const backendLabel = document.getElementById("backend-label");
    const activeModelObj = this.state.models.find(m => m.id === modelState.model_id);
    const isSDCpp = activeModelObj && activeModelObj.engine === "sd_cpp";
    const isSDNQ = activeModelObj && activeModelObj.engine === "sdnq";
    const isQuantFuncQwen = activeModelObj && activeModelObj.engine === "quantfunc" && activeModelObj.id === "qwen_image_2_1";

    if (backendLabel) {
      backendLabel.textContent = (isSDCpp || isSDNQ) ? (t("controls.quantization") || "Quantization") : (t("controls.backend") || "Backend");
    }

    let teMissing = false;

    if (teGroup && teSelect) {
      if (modelState.text_encoders && modelState.text_encoders.length > 0) {
        teGroup.style.display = "flex";
        teSelect.innerHTML = "";
        
        if (isQuantFuncQwen || isSDNQ) {
          teSelect.disabled = true;
          teSelect.style.display = "block";
          if (customTeInput) customTeInput.style.display = "none";
          if (teLink) teLink.style.display = "none";
          teGroup.classList.add("locked");
        } else {
          teSelect.disabled = false;
          if (teLink) teLink.style.display = "inline";
          teGroup.classList.remove("locked");
        }
        if (teHint) teHint.style.display = "none";

        let activeTe = null;
        for (const te of modelState.text_encoders) {
          const opt = document.createElement("option");
          opt.value = te.id;
          opt.textContent = `${te.name} (${te.installed ? '✓' : '⬇'})`;
          if (this.state.selectedTextEncoder === te.id) {
            opt.selected = true;
            activeTe = te;
          } else if (!this.state.selectedTextEncoder && te.recommended) {
            opt.selected = true;
            this.state.selectedTextEncoder = te.id;
            activeTe = te;
          }
          teSelect.appendChild(opt);
        }

        if (!isQuantFuncQwen && !isSDNQ) {
          const customOpt = document.createElement("option");
          customOpt.value = "custom";
          customOpt.textContent = `Custom file / ComfyUI...`;
          if (this.state.selectedTextEncoder === "custom") {
            customOpt.selected = true;
            if (customTeInput) customTeInput.style.display = "block";
          } else {
            teSelect.style.display = "block";
            if (customTeInput) customTeInput.style.display = "none";
            if (teLink) teLink.textContent = t("controls.custom_path") || "Custom path";
          }
          teSelect.appendChild(customOpt);
        }

        if (!activeTe && modelState.text_encoders.length > 0) {
          activeTe = modelState.text_encoders[0];
          this.state.selectedTextEncoder = activeTe.id;
        }

        if (this.state.selectedTextEncoder !== "custom" && activeTe && !activeTe.installed) {
          teMissing = true;
        }
      } else {
        teGroup.style.display = "none";
        teGroup.classList.remove("locked");
        if (teHint) teHint.style.display = "none";
        if (customTeInput) customTeInput.style.display = "none";
      }
    }

    // VAE select
    const vaeGroup = document.getElementById("vae-group");
    const vaeSelect = document.getElementById("vae-select");
    const customVaeInput = document.getElementById("custom-vae-input");
    const vaeLink = document.getElementById("toggle-custom-vae-link");
    if (vaeGroup && vaeSelect) {
      if (activeModelObj && (activeModelObj.family === "qwen" || activeModelObj.id.includes("qwen"))) {
        vaeGroup.style.display = "flex";
        const curVae = this.state.selectedVae || this.state.settings?.qwen_vae_type || "texture_fix";
        vaeSelect.value = curVae;
        if (!customVaeInput || !customVaeInput.value.trim()) {
          vaeSelect.style.display = "block";
          if (customVaeInput) customVaeInput.style.display = "none";
          if (vaeLink) vaeLink.textContent = t("controls.custom_path") || "Custom path";
        }
      } else {
        vaeGroup.style.display = "none";
        vaeSelect.style.display = "block";
        if (customVaeInput) customVaeInput.style.display = "none";
        if (vaeLink) vaeLink.textContent = t("controls.custom_path") || "Custom path";
      }
    }

    // Group: recommended first
    const recommended = variants.filter(v => v.recommended);
    const others = variants.filter(v => !v.recommended);

    if (recommended.length) {
      const grp = document.createElement("optgroup");
      grp.label = t("model.recommended");
      for (const v of recommended) {
        const opt = document.createElement("option");
        opt.value = v.id;
        opt.textContent = `${v.id} (${v.backend}, ${v.steps} steps, ${(v.size_gb || 0).toFixed(1)} GB) · ${v.state_label}`;
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
        opt.textContent = `${v.id} (${v.backend}, ${v.steps} steps, ${(v.size_gb || 0).toFixed(1)} GB) · ${v.state_label}`;
        grp.appendChild(opt);
      }
      variantSelect.appendChild(grp);
    }

    // Select preferred variant if provided, otherwise first recommended
    const preferred = preferredVariantId ? variants.find(v => v.id === preferredVariantId) : null;
    const activeVar = preferred || recommended[0] || others[0] || variants[0];
    if (activeVar) {
      variantSelect.value = activeVar.id;
      this.state.selectedVariant = activeVar.id;
      const stepsInput = document.getElementById("steps-input");
      if (stepsInput && activeVar.steps) {
        stepsInput.value = activeVar.steps;
      }
    }

    // Check if selected variant / base model / text encoder are missing
    const activeVarMissing = activeVar && (activeVar.state === "missing" || activeVar.state === "downloadable" || activeVar.state === "corrupted");
    const hasMissing = activeVarMissing
                    || !modelState.base_model_installed
                    || teMissing;
    this.updateDownloadButton(hasMissing, modelState);
    this.updateModelTrigger();
  },

  onTextEncoderChange() {
    const teSelect = document.getElementById("text-encoder-select");
    const customTeInput = document.getElementById("custom-text-encoder-input");
    const link = document.getElementById("toggle-custom-te-link");
    if (!teSelect) return;
    this.state.selectedTextEncoder = teSelect.value;
    if (teSelect.value === "custom") {
      if (customTeInput) {
        customTeInput.style.display = "block";
        customTeInput.focus();
      }
      if (link) link.textContent = t("controls.standard_select") || "List";
    } else {
      if (customTeInput) customTeInput.style.display = "none";
      if (link) link.textContent = t("controls.custom_path") || "Custom path";
    }
    if (this.state.modelState) {
      const activeTe = this.state.modelState.text_encoders?.find(te => te.id === teSelect.value);
      const activeVar = this.state.modelState.transformers?.find(v => v.id === this.state.selectedVariant) || (this.state.modelState.transformers || [])[0];
      const activeVarMissing = activeVar && (activeVar.state === "missing" || activeVar.state === "downloadable" || activeVar.state === "corrupted");
      const hasMissing = activeVarMissing
                      || !this.state.modelState.base_model_installed
                      || (activeTe && !activeTe.installed && teSelect.value !== "custom");
      this.updateDownloadButton(hasMissing, this.state.modelState);
    }
  },

  toggleCustomTeInput(e) {
    if (e) e.preventDefault();
    const customTeInput = document.getElementById("custom-text-encoder-input");
    const teSelect = document.getElementById("text-encoder-select");
    const link = document.getElementById("toggle-custom-te-link");
    if (!customTeInput || !teSelect) return;
    const isHidden = customTeInput.style.display === "none";
    customTeInput.style.display = isHidden ? "block" : "none";
    teSelect.style.display = isHidden ? "none" : "block";
    if (isHidden) {
      this.state.selectedTextEncoder = "custom";
      if (teSelect.querySelector('option[value="custom"]')) {
        teSelect.value = "custom";
      }
      customTeInput.focus();
    } else {
      const firstOpt = teSelect.querySelector('option:not([value="custom"])');
      if (firstOpt) {
        teSelect.value = firstOpt.value;
        this.state.selectedTextEncoder = firstOpt.value;
      }
    }
    if (link) link.textContent = isHidden ? (t("controls.standard_select") || "List") : (t("controls.custom_path") || "Custom path");
  },

  toggleCustomVaeInput(e) {
    if (e) e.preventDefault();
    const customVaeInput = document.getElementById("custom-vae-input");
    const vaeSelect = document.getElementById("vae-select");
    const link = document.getElementById("toggle-custom-vae-link");
    if (!customVaeInput || !vaeSelect) return;
    const isHidden = customVaeInput.style.display === "none";
    customVaeInput.style.display = isHidden ? "block" : "none";
    vaeSelect.style.display = isHidden ? "none" : "block";
    if (link) link.textContent = isHidden ? (t("controls.standard_select") || "List") : (t("controls.custom_path") || "Custom path");
    if (isHidden) customVaeInput.focus();
  },

  async onVaeChange() {
    const val = document.getElementById("vae-select")?.value || "texture_fix";
    this.state.selectedVae = val;
    try {
      await this.apiPatch("/api/settings", { qwen_vae_type: val });
    } catch (e) {
      console.warn("Failed to persist VAE setting:", e);
    }
  },

  updateDownloadButton(showButton, modelState) {
    // Find or create a download button next to the variant select
    let dlBtn = document.getElementById("download-model-btn");
    const activeModelObj = this.state.models.find(m => m.id === modelState.model_id);
    if (!showButton || (activeModelObj && activeModelObj.disabled)) {
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
    const activeVar = (modelState.transformers || []).find(v => v.id === this.state.selectedVariant) || (modelState.transformers || [])[0];
    const varSize = (activeVar && (activeVar.state === "missing" || activeVar.state === "downloadable" || activeVar.state === "corrupted")) ? (activeVar.size_gb || 0) : 0;
    const teMissing = modelState?.text_encoders?.find(te => te.id === this.state.selectedTextEncoder && !te.installed);
    const teSize = teMissing ? (teMissing.size_gb || 0) : 0;
    const baseSize = (!modelState.base_model_installed && activeModelObj?.engine === "sdnq") ? 0.65 : 0;
    dlBtn.textContent = `⬇ Download (~${(varSize + teSize + baseSize).toFixed(1)} GB)`;
  },

  async downloadModel() {
    if (!this.state.selectedModel) return;
    const activeModel = this.state.models.find(m => m.id === this.state.selectedModel);
    if (activeModel && activeModel.disabled) {
      this.toast("warning", activeModel.disabled_reason || "Модели QuantFunc временно отключены из-за дефектов весов. Включите их в Настройках.");
      return;
    }
    const dlBtn = document.getElementById("download-model-btn");
    if (dlBtn) { dlBtn.disabled = true; dlBtn.textContent = "Downloading..."; }

    try {
      const teSelect = document.getElementById("text-encoder-select");
      const teId = teSelect && teSelect.value && teSelect.value !== "custom" ? teSelect.value : (this.state.selectedTextEncoder || null);
      const result = await this.apiPost(
        `/api/models/${this.state.selectedModel}/download`,
        {
          variant_id: this.state.selectedVariant,
          text_encoder_id: teId,
        }
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

    const activeModel = this.state.models.find(m => m.id === this.state.selectedModel);
    if (activeModel && activeModel.disabled) {
      this.toast("error", activeModel.disabled_reason || "Модели QuantFunc временно отключены из-за дефектов весов. Включите их в Настройках.");
      return;
    }
    const engineType = activeModel?.engine || "quantfunc";
    if (engineType === "quantfunc" && (!this.state.engineStatus || this.state.engineStatus.status === "missing")) {
      this.toast("error", t("errors.engine_not_installed"));
      this.switchTab("settings");
      return;
    }

    const res = this.parseResolution();

    const variantSelect = document.getElementById("variant-select");
    const selectedVariantId = variantSelect?.value || this.state.selectedVariant;
    const currentVariant = this.state.modelState?.transformers?.find(v => v.id === selectedVariantId);
    const backend = currentVariant ? currentVariant.backend : "lighting";

    // Collect params
    const cfgVal = parseFloat(document.getElementById("cfg-input")?.value);
    const noiseVal = parseFloat(document.getElementById("noise-input")?.value);
    const negPrompt = document.getElementById("negative-prompt-input")?.value?.trim() || "";
    const samplerVal = document.getElementById("sampler-select")?.value || "euler";

    const params = {
      model_id: this.state.selectedModel,
      variant_id: selectedVariantId,
      prompt: prompt,
      negative_prompt: negPrompt,
      seed: this.parseSeed(),
      width: res.width,
      height: res.height,
      steps: parseInt(document.getElementById("steps-input")?.value || (currentVariant ? String(currentVariant.steps) : "4"), 10),
      cfg: !isNaN(cfgVal) ? cfgVal : 1.0,
      noise: !isNaN(noiseVal) ? noiseVal : 0.0,
      sampler: samplerVal,
      backend: backend,
    };

    if (this.state.img2imgRef) {
      params.init_image_path = this.state.img2imgRef.path;
      params.init_image_strength = parseFloat(
        document.getElementById("img2img-strength")?.value || "0.85"
      );
      params.ref_mode = document.getElementById("img2img-mode")?.value || "control_ref";
      const cnInput = document.getElementById("controlnet-custom-path");
      const cnPath = cnInput && cnInput.style.display !== "none" ? cnInput.value.trim() : "";
      if (cnPath) {
        params.control_net_path = cnPath;
      }
    }

    const teGroup = document.getElementById("text-encoder-group");
    if (teGroup && teGroup.style.display !== "none") {
      const teSelect = document.getElementById("text-encoder-select");
      const customTeInput = document.getElementById("custom-text-encoder-input");
      const customTeVal = customTeInput && customTeInput.style.display !== "none" ? customTeInput.value.trim() : "";
      if (customTeVal) {
        params.custom_text_encoder_path = customTeVal;
      } else if (teSelect && teSelect.value && teSelect.value !== "custom") {
        params.text_encoder_id = teSelect.value;
      }
    }

    const vaeGroup = document.getElementById("vae-group");
    if (vaeGroup && vaeGroup.style.display !== "none") {
      const customVaeInput = document.getElementById("custom-vae-input");
      const customVaeVal = customVaeInput && customVaeInput.style.display !== "none" ? customVaeInput.value.trim() : "";
      if (customVaeVal) {
        params.custom_vae_path = customVaeVal;
      } else {
        const vaeSelect = document.getElementById("vae-select");
        if (vaeSelect && vaeSelect.value) {
          params.vae = vaeSelect.value;
        }
      }
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

  snapDimension(val, multiple = 64) {
    const v = parseInt(val, 10);
    if (isNaN(v) || v <= 0) return 1024;
    return Math.max(multiple, Math.round(v / multiple) * multiple);
  },

  parseResolution() {
    const val = document.getElementById("resolution-select")?.value?.trim() || "1024x1024";
    if (val.includes("x")) {
      const parts = val.split("x");
      const w = parseInt(parts[0], 10);
      const h = parseInt(parts[1], 10);
      return {
        width: this.snapDimension(!isNaN(w) && w > 0 ? w : 1024, 64),
        height: this.snapDimension(!isNaN(h) && h > 0 ? h : 1024, 64),
      };
    }
    const num = parseInt(val, 10);
    const sz = this.snapDimension(!isNaN(num) && num > 0 ? num : 1024, 64);
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
    let consecutiveErrors = 0;
    const MAX_CONSECUTIVE_ERRORS = 10;
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
        consecutiveErrors = 0;
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
        consecutiveErrors++;
        console.warn(`[app] Polling warning (${consecutiveErrors}/${MAX_CONSECUTIVE_ERRORS}):`, e);
        const errMsg = String(e?.message || "");
        if (errMsg.includes("404") || errMsg.includes("Not Found")) {
          this.stopPolling();
          this.setGenerating(false);
          this.toast("error", "Generation job lost (404) — try again");
        } else if (consecutiveErrors >= MAX_CONSECUTIVE_ERRORS) {
          this.stopPolling();
          this.setGenerating(false);
          this.toast("error", "Network connection lost during generation — try again");
        }
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
    const statusPill = document.getElementById("status-pill");
    const statusMsg = document.getElementById("status-message");

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

    if (statusPill) {
      statusPill.className = "status-pill warning";
      statusPill.textContent = status.step && status.total_steps ? `${status.step}/${status.total_steps}` : "Busy";
    }
    if (statusMsg) {
      statusMsg.textContent = status.message || "";
    }
  },

  setGenerating(busy) {
    const btn = document.getElementById("generate-btn");
    const overlay = document.getElementById("progress-overlay");
    const statusPill = document.getElementById("status-pill");
    const statusMsg = document.getElementById("status-message");
    if (!btn) return;

    if (busy) {
      btn.classList.add("busy");
      btn.disabled = true;
      btn.textContent = t("controls.busy");
      if (overlay) overlay.style.display = "flex";
      if (statusPill) {
        statusPill.className = "status-pill warning";
        statusPill.textContent = t("controls.busy") || "Generating...";
      }
    } else {
      btn.classList.remove("busy");
      btn.disabled = false;
      btn.textContent = t("controls.generate");
      if (overlay) overlay.style.display = "none";
      if (statusPill) {
        statusPill.className = "status-pill success";
        statusPill.textContent = t("status.ready") || "Ready";
      }
      if (statusMsg) {
        statusMsg.textContent = "";
      }
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
    const promptBox = document.getElementById("history-modal-prompt-box");
    const promptToggle = document.getElementById("history-prompt-toggle");
    const modelEl = document.getElementById("history-modal-model");
    const seedEl = document.getElementById("history-modal-seed");
    const resEl = document.getElementById("history-modal-res");
    const stepsEl = document.getElementById("history-modal-steps");
    const backendEl = document.getElementById("history-modal-backend");
    const timeEl = document.getElementById("history-modal-time");

    const promptText = item.prompt || t("history.no_prompt");
    if (img) img.src = item.url;
    if (promptEl) {
      promptEl.textContent = promptText;
      promptEl.scrollTop = 0;
    }
    if (promptBox) {
      promptBox.classList.remove("collapsed");
    }
    if (promptToggle) {
      promptToggle.style.display = promptText.length > 110 ? "inline-block" : "none";
      promptToggle.textContent = "Свернуть ▴";
    }
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

  toggleHistoryPrompt() {
    const box = document.getElementById("history-modal-prompt-box");
    const btn = document.getElementById("history-prompt-toggle");
    if (!box || !btn) return;
    const isCollapsed = box.classList.toggle("collapsed");
    btn.textContent = isCollapsed ? "Развернуть ▾" : "Свернуть ▴";
  },

  closeHistoryModal() {
    const modal = document.getElementById("history-modal");
    if (modal) modal.style.display = "none";
    this.state.selectedHistoryItem = null;
  },

  async reuseHistoryItem(item) {
    if (!item) return;

    // Set prompt
    const promptInput = document.getElementById("prompt-input");
    if (promptInput && item.prompt) {
      promptInput.value = item.prompt;
    }

    // Set model and await variant rendering first so steps/CFG aren't overwritten
    if (item.model_id) {
      const modelSelect = document.getElementById("model-select");
      if (modelSelect) {
        modelSelect.value = item.model_id;
        await this.onModelChange(item.variant_id || null);
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

    // Set negative prompt
    const negPromptInput = document.getElementById("negative-prompt-input");
    const negDot = document.getElementById("negative-prompt-dot");
    if (negPromptInput) {
      negPromptInput.value = item.negative_prompt || "";
      if (negDot) {
        negDot.style.display = (item.negative_prompt && item.negative_prompt.trim().length > 0) ? "inline-block" : "none";
      }
    }

    // Set steps
    if (item.steps) {
      const stepsInput = document.getElementById("steps-input");
      if (stepsInput) stepsInput.value = item.steps;
    }

    // Set CFG
    if (item.cfg !== undefined && item.cfg !== null) {
      const cfgInput = document.getElementById("cfg-input");
      const cfgSlider = document.getElementById("cfg-slider");
      if (cfgInput) cfgInput.value = item.cfg;
      if (cfgSlider) cfgSlider.value = item.cfg;
    }

    // Set Noise
    if (item.noise !== undefined && item.noise !== null) {
      const noiseInput = document.getElementById("noise-input");
      const noiseSlider = document.getElementById("noise-slider");
      if (noiseInput) noiseInput.value = item.noise;
      if (noiseSlider) noiseSlider.value = item.noise;
    }

    // Set Sampler
    if (item.sampler) {
      const samplerSelect = document.getElementById("sampler-select");
      if (samplerSelect && samplerSelect.querySelector(`option[value="${item.sampler}"]`)) {
        samplerSelect.value = item.sampler;
      }
    }

    // Set VAE
    if (item.vae) {
      const vaeSelect = document.getElementById("vae-select");
      if (vaeSelect && vaeSelect.querySelector(`option[value="${item.vae}"]`)) {
        vaeSelect.value = item.vae;
        this.state.selectedVae = item.vae;
      }
    }

    // Set seed
    if (item.seed !== null && item.seed !== undefined) {
      const seedInput = document.getElementById("seed-input");
      if (seedInput) seedInput.value = item.seed;
    }

    this.updateModelTrigger();

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

  // ---------- Img2Img / Native ControlNet Reference ----------
  onRefModeChange() {
    const mode = document.getElementById("img2img-mode")?.value || "control_ref";
    const hintEl = document.getElementById("img2img-mode-hint");
    const strengthLabel = document.getElementById("img2img-strength-label");
    const strengthSlider = document.getElementById("img2img-strength");
    const strengthVal = document.getElementById("img2img-strength-val");

    if (mode === "control_ref") {
      if (hintEl) {
        hintEl.style.color = "var(--success, #22c55e)";
        hintEl.innerHTML = "✓ Изоляция шагов активна: <code>--init-img</code> и <code>--strength</code> отключены. Модель выполнит 100% шагов с чистого листа, перенося лайн и скринтоны через нативный Reference / ControlNet.";
      }
      if (strengthLabel) strengthLabel.textContent = "Сила контроля:";
      if (strengthSlider) { strengthSlider.max = "1.00"; strengthSlider.value = "0.85"; }
      if (strengthVal) strengthVal.textContent = "0.85";
    } else if (mode === "controlnet_canny") {
      if (hintEl) {
        hintEl.style.color = "var(--info, #38bdf8)";
        hintEl.innerHTML = "✒️ Canny / Lineart препроцессор: выделяет контуры туши и передает в ControlNet / Reference при 100% шагов с чистого белого шума.";
      }
      if (strengthLabel) strengthLabel.textContent = "Сила контура:";
      if (strengthSlider) { strengthSlider.max = "1.00"; strengthSlider.value = "0.90"; }
      if (strengthVal) strengthVal.textContent = "0.90";
    } else {
      if (hintEl) {
        hintEl.style.color = "var(--warning, #fbbf24)";
        hintEl.innerHTML = "⚠️ Классический Img2Img (зашумление исходника через <code>--init-img</code> + <code>--strength</code>). Может размывать белый фон и тонкие скринтоны.";
      }
      if (strengthLabel) strengthLabel.textContent = "Сила шума:";
      if (strengthSlider) { strengthSlider.max = "0.95"; strengthSlider.value = "0.70"; }
      if (strengthVal) strengthVal.textContent = "0.70";
    }
  },

  toggleControlNetInput(e) {
    if (e) e.preventDefault();
    const input = document.getElementById("controlnet-custom-path");
    if (!input) return;
    const isHidden = input.style.display === "none";
    input.style.display = isHidden ? "block" : "none";
    if (isHidden) input.focus();
  },

  async uploadImg2ImgFile(file) {
    if (!file) return;

    const ext = "." + (file.name ? file.name.split(".").pop().toLowerCase() : "");
    const validExts = [".png", ".jpg", ".jpeg", ".webp"];
    const validMimes = ["image/png", "image/jpeg", "image/webp"];
    if (!validMimes.includes(file.type) && !validExts.includes(ext)) {
      this.toast("error", "Unsupported file format. Allowed: PNG, JPEG, WEBP");
      return;
    }
    const sizeMb = file.size / 1024 / 1024;
    if (sizeMb > 25) {
      this.toast("error", `File too large (${sizeMb.toFixed(1)} MB). Maximum allowed size is 25 MB.`);
      return;
    }

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
      this.onRefModeChange();
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
        const cleanSrc = src.split("?")[0];
        const filename = cleanSrc.split("/").pop();
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
    const cleanUrl = this.state.lastPreviewUrl ? this.state.lastPreviewUrl.split("?")[0] : "";
    const cleanFilename = this.state.lastPreviewName ? this.state.lastPreviewName.split("?")[0] : (cleanUrl ? cleanUrl.split("/").pop() : "");
    this.setImg2ImgRef({
      path: cleanFilename,
      url: this.state.lastPreviewUrl,
      filename: cleanFilename,
    });
    this.toast("success", t("img2img.send_to") + " ✓");
  },

  // ---------- Custom Model Modal ----------
  async openCustomModelModal() {
    const modal = document.getElementById("custom-model-modal");
    if (!modal) return;

    try {
      const res = await fetch("/api/custom-models/files");
      if (res.ok) {
        const data = await res.json();
        this._customFilesData = data;
        const dirPathEl = document.getElementById("custom-models-dir-path");
        if (dirPathEl) dirPathEl.textContent = data.custom_dir || "models/custom";

        this.populateCustomFileSelects(data.files || []);
      }
    } catch (err) {
      console.warn("Failed to fetch custom files:", err);
    }

    modal.style.display = "flex";
  },

  closeCustomModelModal() {
    const modal = document.getElementById("custom-model-modal");
    if (modal) modal.style.display = "none";
  },

  populateCustomFileSelects(files) {
    const modelSel = document.getElementById("custom-model-file-select");
    const teSel = document.getElementById("custom-te-file-select");
    const vaeSel = document.getElementById("custom-vae-file-select");

    if (modelSel) {
      modelSel.innerHTML = `<option value="">— Select file from models/custom/ —</option><option value="__custom__">Custom path on PC...</option>`;
      for (const f of files) {
        const opt = document.createElement("option");
        opt.value = f.rel_path;
        opt.textContent = `${f.name} (${f.size_gb} GB, ${f.ext.toUpperCase()})`;
        modelSel.appendChild(opt);
      }
    }

    if (teSel) {
      teSel.innerHTML = `<option value="">— None / Bundled in model —</option><option value="__custom__">Custom path on PC...</option>`;
      for (const f of files) {
        const opt = document.createElement("option");
        opt.value = f.rel_path;
        opt.textContent = `${f.name} (${f.size_gb} GB)`;
        teSel.appendChild(opt);
      }
    }

    if (vaeSel) {
      vaeSel.innerHTML = `<option value="">— None / Bundled in model —</option><option value="__custom__">Custom path on PC...</option>`;
      for (const f of files) {
        const opt = document.createElement("option");
        opt.value = f.rel_path;
        opt.textContent = `${f.name} (${f.size_gb} GB)`;
        vaeSel.appendChild(opt);
      }
    }
  },

  onCustomFileSelectChange(type) {
    const sel = document.getElementById(`custom-${type === 'model' ? 'model-file' : type}-file-select`);
    const input = document.getElementById(`custom-${type === 'model' ? 'model-file' : type}-file-input`);
    if (!sel || !input) return;

    if (sel.value === "__custom__") {
      input.style.display = "block";
      input.focus();
    } else {
      input.style.display = "none";
    }

    // Auto-fill model name if empty and a file was selected
    if (type === "model" && sel.value && sel.value !== "__custom__") {
      const nameInput = document.getElementById("custom-model-name");
      if (nameInput && !nameInput.value.trim()) {
        const stem = sel.value.split(/[\\/]/).pop().replace(/\.[^/.]+$/, "");
        nameInput.value = stem.replace(/[-_]/g, " ");
      }
    }
  },

  onCustomEngineChange() {
    // Engine switcher hook
  },

  onCustomFamilyChange() {
    const familySel = document.getElementById("custom-model-family");
    const stepsInput = document.getElementById("custom-model-steps");
    const cfgInput = document.getElementById("custom-model-cfg");
    if (!familySel || !stepsInput || !cfgInput) return;

    const fam = familySel.value;
    if (fam === "sdxl") {
      stepsInput.value = "1";
      cfgInput.value = "1.0";
    } else if (fam === "flux") {
      stepsInput.value = "4";
      cfgInput.value = "3.5";
    } else if (fam === "qwen") {
      stepsInput.value = "20";
      cfgInput.value = "3.5";
    } else {
      stepsInput.value = "20";
      cfgInput.value = "7.0";
    }
  },

  async saveCustomModel() {
    const nameInput = document.getElementById("custom-model-name");
    const engineSel = document.getElementById("custom-model-engine");
    const familySel = document.getElementById("custom-model-family");
    const modelSel = document.getElementById("custom-model-file-select");
    const modelInput = document.getElementById("custom-model-file-input");
    const teSel = document.getElementById("custom-te-file-select");
    const teInput = document.getElementById("custom-te-file-input");
    const vaeSel = document.getElementById("custom-vae-file-select");
    const vaeInput = document.getElementById("custom-vae-file-input");
    const stepsInput = document.getElementById("custom-model-steps");
    const cfgInput = document.getElementById("custom-model-cfg");

    const name = nameInput ? nameInput.value.trim() : "";
    if (!name) {
      this.toast("error", t("custom_model.error_name") || "Please enter a model name");
      return;
    }

    let modelFile = modelSel ? modelSel.value : "";
    if (modelFile === "__custom__") {
      modelFile = modelInput ? modelInput.value.trim() : "";
    }
    if (!modelFile) {
      this.toast("error", t("custom_model.error_file") || "Please select or specify a model file");
      return;
    }

    let teFile = teSel ? teSel.value : "";
    if (teFile === "__custom__") teFile = teInput ? teInput.value.trim() : "";

    let vaeFile = vaeSel ? vaeSel.value : "";
    if (vaeFile === "__custom__") vaeFile = vaeInput ? vaeInput.value.trim() : "";

    const payload = {
      name: name,
      engine: engineSel ? engineSel.value : "sd_cpp",
      family: familySel ? familySel.value : "flux",
      model_file: modelFile,
      steps: parseInt(stepsInput?.value || "20", 10),
      cfg: parseFloat(cfgInput?.value || "3.5"),
      checkpoint_type: familySel?.value === "sdxl" ? "checkpoint" : "diffusion_model",
    };
    if (teFile) {
      if (familySel?.value === "qwen") payload.llm_file = teFile;
      else payload.t5_file = teFile;
    }
    if (vaeFile) payload.vae_file = vaeFile;

    try {
      const res = await fetch("/api/custom-models", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.detail || "Failed to register custom model");
      }

      this.toast("success", `✓ ${name} registered!`);
      this.closeCustomModelModal();

      // Refresh models
      await this.init();
      if (data.model && data.model.id) {
        await this.selectModelFromModal(data.model.id);
      }
    } catch (err) {
      this.toast("error", err.message);
    }
  },

  async deleteCustomModel(modelId, event) {
    if (event) event.stopPropagation();
    if (!confirm(`Delete custom model ${modelId}?`)) return;

    try {
      const res = await fetch(`/api/custom-models/${encodeURIComponent(modelId)}`, {
        method: "DELETE",
      });
      if (!res.ok) {
        const data = await res.json();
        throw new Error(data.detail || "Failed to delete");
      }
      this.toast("info", `Model ${modelId} removed`);
      await this.init();
      this.renderModelPickerModal();
    } catch (err) {
      this.toast("error", err.message);
    }
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
