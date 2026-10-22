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
    presets: [],
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
    activeDockPopover: null,
    hfInspectedRepo: null,
    activeLoras: [],
    loraLibrary: [],
    compareActive: false,
    comparePercentage: 50,
    previousPreviewUrl: null,
    currentPreviewUrl: null,
    currentPreviewFilename: null,
    dockPosition: "center",
    dockMinimized: false,
    dockOpacity: 0.88,
    dockBlur: 20,
    shareStatus: null,
    extrasStatus: null,
    account: null,
    guestAccounts: [],
    historyFilter: "all",
  },

  // ---------- 3-Tier Model Sorting ----------
  sortModelsList(models) {
    if (!Array.isArray(models)) return [];
    const tierScore = (m) => {
      if (m.disabled === true) return 2;
      if (m.installed === true) return 0;
      return 1;
    };
    return [...models].sort((a, b) => {
      const ta = tierScore(a);
      const tb = tierScore(b);
      if (ta !== tb) return ta - tb;
      return String(a.name || "").localeCompare(String(b.name || ""), undefined, { sensitivity: "base" });
    });
  },

  // ---------- Init ----------
  async init() {
    console.log("[app] LazySD UI initializing...");

    try { this.initSidebar(); } catch (e) { console.warn("[init] initSidebar error:", e); }
    try { this.initInspector(); } catch (e) { console.warn("[init] initInspector error:", e); }

    if (window.initI18n) {
      try { await window.initI18n(); } catch (e) { console.warn("[init] initI18n error:", e); }
    }

    try { await this.loadConfig(); } catch (e) { console.warn("[init] loadConfig error:", e); }
    try { await this.loadPresets(); } catch (e) { console.warn("[init] loadPresets error:", e); }

    try { this.renderGpuIndicator(); } catch (e) { console.warn("[init] renderGpuIndicator error:", e); }
    try { this.renderSystemStats(this.state.config?.system_stats); } catch (e) { console.warn("[init] renderSystemStats error:", e); }
    try { this.startStatsPolling(); } catch (e) { console.warn("[init] startStatsPolling error:", e); }
    try { this.renderModels(); } catch (e) { console.warn("[init] renderModels error:", e); }
    try { this.updateEngineStatusUI(); } catch (e) { console.warn("[init] updateEngineStatusUI error:", e); }
    try { this.updateDockPills(); } catch (e) { console.warn("[init] updateDockPills error:", e); }
    try { this.initDockSettings(); } catch (e) { console.warn("[init] initDockSettings error:", e); }
    try { this.loadLoras(); } catch (e) { console.warn("[init] loadLoras error:", e); }
    try { this.initCompareEvents(); } catch (e) { console.warn("[init] initCompareEvents error:", e); }
    try { this.loadShareStatus(); } catch (e) { console.warn("[init] loadShareStatus error:", e); }
    try { this.refreshExtrasStatus(); } catch (e) { console.warn("[init] refreshExtrasStatus error:", e); }
    try { this.loadHistory(); } catch (e) { console.warn("[init] loadHistory error:", e); }
    try { this.bindEvents(); } catch (e) { console.warn("[init] bindEvents error:", e); }
    try { this.initAllViewportZoomPan(); } catch (e) { console.warn("[init] initAllViewportZoomPan error:", e); }
    try { this.initViewportHudOpacity(); } catch (e) { console.warn("[init] initViewportHudOpacity error:", e); }

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

  toggleInspector() {
    const inspector = document.getElementById("topaz-inspector");
    const btn = document.getElementById("inspector-toggle-btn");
    if (!inspector) return;
    const isCollapsed = inspector.classList.toggle("collapsed");
    if (btn) btn.classList.toggle("active", !isCollapsed);
    try {
      localStorage.setItem("lazysd_inspector_collapsed", isCollapsed ? "1" : "0");
    } catch (_) {}
  },

  initInspector() {
    const inspector = document.getElementById("topaz-inspector");
    const btn = document.getElementById("inspector-toggle-btn");
    if (!inspector) return;
    try {
      const saved = localStorage.getItem("lazysd_inspector_collapsed");
      const isCollapsed = saved === "1";
      inspector.classList.toggle("collapsed", isCollapsed);
      if (btn) btn.classList.toggle("active", !isCollapsed);
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

  async apiDelete(path) {
    const res = await fetch(path, { method: "DELETE" });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || `HTTP ${res.status}`);
    }
    return res.json();
  },

  // ---------- Config & Identity ----------
  updateAccountRoleUI() {
    const acc = this.state.account || { role: "host", id: "host", username: "Хост", is_guest: false };
    const isGuest = Boolean(acc.is_guest);

    if (isGuest) {
      document.body.classList.add("role-guest");
      const badge = document.getElementById("user-role-badge");
      if (badge) {
        badge.className = "user-role-badge guest";
        badge.title = "Гостевой режим: генерация и просмотр своих работ";
      }
      const icon = document.getElementById("user-role-icon");
      if (icon) icon.textContent = "👤";
      const title = document.getElementById("user-role-title");
      if (title) title.textContent = "Гость: " + (acc.username || "Друг");
      const bannerName = document.getElementById("guest-username-display");
      if (bannerName) bannerName.textContent = acc.username || "Гость";
      const guestProfileUser = document.getElementById("guest-profile-username");
      if (guestProfileUser) guestProfileUser.textContent = acc.username || "Гость";
    } else {
      document.body.classList.remove("role-guest");
      const badge = document.getElementById("user-role-badge");
      if (badge) {
        badge.className = "user-role-badge host";
        badge.title = "Пользовательский режим (Хост / Администратор)";
      }
      const icon = document.getElementById("user-role-icon");
      if (icon) icon.textContent = "👑";
      const title = document.getElementById("user-role-title");
      if (title) title.textContent = "Хост";

      // Host can view and manage guest accounts
      this.loadGuestAccounts();

      // Populate HuggingFace token in Settings
      const hfInput = document.getElementById("setting-hf-token");
      if (hfInput && this.state.settings?.huggingface_token) {
        hfInput.value = this.state.settings.huggingface_token;
      }
    }
  },

  async saveGuestPassword() {
    const input = document.getElementById("guest-input-new-password");
    const pwd = (input?.value || "").trim();
    if (!pwd || pwd.length < 3) {
      this.toast("warning", "Пароль должен содержать не менее 3 символов");
      return;
    }
    try {
      await this.apiPost("/api/account/change-password", { new_password: pwd });
      if (input) input.value = "";
      this.toast("success", "Ваш пароль успешно сохранён! ✓");
    } catch (e) {
      this.toast("error", "Не удалось сменить пароль: " + e.message);
    }
  },

  async loadConfig() {
    try {
      const config = await this.apiGet("/api/config");
      this.state.config = config;
      this.state.account = config.account || { role: "host", id: "host", username: "Хост", is_guest: false };
      this.state.settings = config.settings;
      this.state.gpuInfo = config.gpu;
      this.state.engineStatus = config.engine_status;
      this.state.models = this.sortModelsList(config.models || []);
      if (config.settings?.qwen_vae_type) {
        this.state.selectedVae = config.settings.qwen_vae_type;
      }
      // Inject into window for i18n detection
      window.QF_INITIAL_CONFIG = config;
      window.QF_SETTINGS = config.settings;
      this.updateAccountRoleUI();
      const hideUninstalledChk = document.getElementById("setting-hide-uninstalled-gen");
      if (hideUninstalledChk && config.settings) {
        hideUninstalledChk.checked = Boolean(config.settings.hide_uninstalled_in_generator);
      }
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
      this.state.account = config.account || { role: "host", id: "host", username: "Хост", is_guest: false };
      this.state.settings = config.settings;
      this.state.gpuInfo = config.gpu;
      this.state.engineStatus = config.engine_status;
      this.state.models = this.sortModelsList(config.models || []);
      if (config.settings?.qwen_vae_type) {
        this.state.selectedVae = config.settings.qwen_vae_type;
      }
      window.QF_INITIAL_CONFIG = config;
      window.QF_SETTINGS = config.settings;
      this.updateAccountRoleUI();
      this.renderGpuIndicator();
      this.renderSystemStats(config.system_stats);
      this.renderModels();
      this.updateEngineStatusUI();
      this.updateDockPills();
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
    const cpuEl = document.getElementById("cpu-stat-text");
    const gpuStatEl = document.getElementById("gpu-stat-text");
    const gpuSepEl = document.getElementById("gpu-stat-sep");
    const vramEl = document.getElementById("vram-stat-text");
    const ramEl = document.getElementById("ram-stat-text");

    if (cpuEl && stats.cpu_percent !== undefined) {
      const cpuPct = Math.round(stats.cpu_percent);
      const color = cpuPct > 90 ? "#f87171" : (cpuPct > 70 ? "#fbbf24" : "var(--text-primary)");
      const cores = stats.cpu_cores ? ` (${stats.cpu_cores}C)` : "";
      cpuEl.innerHTML = `CPU: <strong style="color:${color}">${cpuPct}%</strong>${cores}`;
    }

    if (gpuStatEl) {
      const parts = [];
      if (stats.gpu_temp_c != null) parts.push(`${stats.gpu_temp_c}°C`);
      if (stats.gpu_power_w != null) parts.push(`${stats.gpu_power_w}W`);
      if (stats.gpu_util != null) parts.push(`${stats.gpu_util}%`);
      if (parts.length > 0) {
        gpuStatEl.innerHTML = `GPU: <strong>${parts.join(" · ")}</strong>`;
        gpuStatEl.style.display = "";
        if (gpuSepEl) gpuSepEl.style.display = "";
      } else {
        gpuStatEl.style.display = "none";
        if (gpuSepEl) gpuSepEl.style.display = "none";
      }
    }

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
    }, 2000);
  },

  renderModels() {
    const select = document.getElementById("model-select");
    if (!select) return;

    this.state.models = this.sortModelsList(this.state.models || []);

    select.innerHTML = "";
    if (!this.state.models.length) {
      select.innerHTML = `<option value="">${t("common.loading")}</option>`;
      return;
    }

    // Filter models compatible with current GPU
    const gpuSm = this.state.gpuInfo?.sm || 0;
    let compatible = this.state.models.filter(m =>
      !m.hidden_from_host && m.requirements.gpu_min_sm <= gpuSm
    );
    if (this.state.settings?.hide_uninstalled_in_generator) {
      const installedOnly = compatible.filter(m => m.installed);
      if (installedOnly.length > 0) {
        compatible = installedOnly;
      }
    }

    // Graceful fallback: if GPU SM is not detected or no models match, show all models
    if (!compatible.length) {
      compatible = this.state.models;
    }

    compatible = this.sortModelsList(compatible);

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
        const instMark = model.installed ? "✓ " : "⬇ ";
        opt.textContent = `${instMark}${model.name} (${model.family})`;
      }
      select.appendChild(opt);
    }

    // Auto-select first non-disabled model (or preserve existing selection)
    const existing = this.state.selectedModel && compatible.find(m => m.id === this.state.selectedModel && !m.disabled);
    const firstActive = existing || compatible.find(m => !m.disabled) || compatible[0];
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

    const current = this.state.models.find(m => m.id === this.state.selectedModel);
    if (!current) {
      if (nameEl) nameEl.textContent = "—";
      if (subEl) subEl.textContent = "—";
      this.updateDockPills();
      return;
    }

    if (nameEl) nameEl.textContent = current.name;
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

    this.updateDockPills();
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

    const query = this.state.modelPickerSearch || "";
    const filter = this.state.modelPickerFilter || "all";

    const filtered = this.sortModelsList(this.state.models).filter(m => {
      // Search query
      if (query) {
        const matchesName = m.name?.toLowerCase().includes(query);
        const matchesFamily = m.family?.toLowerCase().includes(query);
        const matchesDesc = m.description?.toLowerCase().includes(query);
        if (!matchesName && !matchesFamily && !matchesDesc) return false;
      }
      // Visibility filter
      if (filter === "hidden") {
        return m.hidden_from_host === true || m.hidden_for_guests === true;
      } else if (m.hidden_from_host === true) {
        return false;
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

      let visBadges = "";
      if (model.hidden_from_host) {
        visBadges += `<span class="model-badge" style="background: rgba(239, 68, 68, 0.2); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.4);" title="Скрыта из общего списка">🙈 Скрыта</span>`;
      }
      if (model.hidden_for_guests) {
        visBadges += `<span class="model-badge" style="background: rgba(245, 158, 11, 0.2); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.4);" title="Скрыта от гостей">🔒 Только хост</span>`;
      }

      const vramMin = model.requirements?.vram_min_gb ? `${model.requirements.vram_min_gb} GB` : "8 GB";
      const defaultSteps = model.transformers?.[0]?.steps || 4;

      let actionBtnHtml = "";
      let deleteBtnHtml = "";
      let visBtnHtml = "";
      if (!this.state.isGuest) {
        visBtnHtml = `<button class="card-action-btn secondary" style="width: auto; padding: 0 10px; font-size: 11px; height: 32px;" onclick="App.openModelVisibilityModal('${escapeHtml(model.id)}', event)" title="Видимость модели">👁️ Видимость</button>`;
      }
      if (model.is_custom && !this.state.isGuest) {
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
              ${visBadges}
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
        <div class="card-footer" style="display: flex; gap: 8px; align-items: center; flex-wrap: wrap;">
          ${actionBtnHtml}
          ${visBtnHtml}
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

    for (const model of this.sortModelsList(this.state.models)) {
      const isDisabled = model.disabled === true;
      const card = document.createElement("div");
      card.className = `model-card ${isDisabled ? "model-card-qf-disabled" : ""}`;

      const keyBadge = model.key_protected
        ? `<span class="model-badge key-protected">🔑 Key Protected</span>`
        : `<span class="model-badge free">🔓 Free Access</span>`;
      
      const pipelineBadge = `<span class="model-badge pipeline">${escapeHtml(model.pipeline)}</span>`;
      const familyBadge = `<span class="model-badge family">${escapeHtml(model.family)}</span>`;
      let engineBadge = "";
      if (model.is_custom) {
        engineBadge = `<span class="badge-custom">🎨 Custom</span>`;
      } else if (model.engine === "sd_cpp") {
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

      const deleteCustomBtn = model.is_custom
        ? `<button class="delete-custom-btn" style="margin-top: 12px; margin-left: 8px;" onclick="App.deleteCustomModel('${escapeHtml(model.id)}', event)">🗑 Удалить</button>`
        : "";

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
          <div style="margin-top: 8px; display: flex; align-items: center;">
            ${selectBtnHtml}
            ${deleteCustomBtn}
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
    if (this._eventsBound) return;
    this._eventsBound = true;

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

    // Steps, Sampler, Resolution, Seed sync with Dock pills
    const stepsInput = document.getElementById("steps-input");
    if (stepsInput) {
      stepsInput.addEventListener("input", () => this.updateModelTrigger());
    }
    const samplerSelect = document.getElementById("sampler-select");
    if (samplerSelect) {
      samplerSelect.addEventListener("change", () => this.updateDockPills());
    }
    const resSelect = document.getElementById("resolution-select");
    if (resSelect) {
      resSelect.addEventListener("change", () => this.updateDockPills());
    }
    const seedInput = document.getElementById("seed-input");
    if (seedInput) {
      seedInput.addEventListener("input", () => this.updateDockPills());
    }

    // CFG slider & input sync
    const cfgSlider = document.getElementById("cfg-slider");
    const cfgInput = document.getElementById("cfg-input");
    if (cfgSlider && cfgInput) {
      cfgSlider.addEventListener("input", () => {
        cfgInput.value = cfgSlider.value;
        this.updateDockPills();
      });
      cfgInput.addEventListener("input", () => {
        cfgSlider.value = cfgInput.value;
        this.updateDockPills();
      });
    }

    // Noise slider & input sync (+ auto-enable stochastic sampler when noise > 0)
    const noiseSlider = document.getElementById("noise-slider");
    const noiseInput = document.getElementById("noise-input");
    const autoSyncStochasticSampler = (valStr) => {
      const n = parseFloat(valStr);
      const sSel = document.getElementById("sampler-select");
      if (!isNaN(n) && n > 0 && sSel && sSel.value === "euler") {
        sSel.value = "euler_a";
      }
      this.updateDockPills();
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

    // Negative prompt input dot indicator & dock pill highlight
    const negInput = document.getElementById("negative-prompt-input");
    const negDot = document.getElementById("negative-prompt-dot");
    if (negInput) {
      negInput.addEventListener("input", () => {
        const hasNeg = negInput.value.trim().length > 0;
        if (negDot) negDot.style.display = hasNeg ? "inline-block" : "none";
        this.updateDockPills();
      });
    }

    // Generate button
    const genBtn = document.getElementById("generate-btn");
    if (genBtn) {
      genBtn.addEventListener("click", () => this.onGenerate());
    }

    // Close dock popovers on outside click
    document.addEventListener("click", (e) => {
      if (!this.state.activeDockPopover) return;
      const dockWrap = document.getElementById("gemini-dock-wrap");
      if (dockWrap && !dockWrap.contains(e.target)) {
        this.closeAllDockPopovers();
      }
    });

    // Keyboard shortcut: Ctrl/Cmd+Enter
    document.addEventListener("keydown", (e) => {
      if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
        e.preventDefault();
        this.onGenerate();
      }
      // ESC: close popovers/modals first, then cancel generation
      if (e.key === "Escape") {
        if (this.state.activeDockPopover) {
          this.closeAllDockPopovers();
        } else if (this.state.isModelPickerOpen) {
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

    // Global dragover & drop prevention so dropped files NEVER open in the browser
    window.addEventListener("dragover", (e) => {
      e.preventDefault();
    }, false);
    window.addEventListener("drop", (e) => {
      if (!e.target.closest("#tab-generate") && !e.target.closest("#ht-dropzone") && !e.target.closest("#manga-dropzone")) {
        e.preventDefault();
      }
    }, false);

    // Dropzone setup for Generator tab
    const previewArea = document.getElementById("tab-generate");
    if (previewArea && window.Dropzone) {
      this.dropzone = new window.Dropzone(previewArea, {
        onFile: (fileInfo) => {
          this.uploadImg2ImgFile(fileInfo.file);
        },
        acceptedTypes: ["image/png", "image/jpeg", "image/webp"],
        maxSizeMb: 50,
      });
    }

    // Dropzone setup for Halftone Studio tab
    const htArea = document.getElementById("ht-dropzone");
    if (htArea && window.Dropzone) {
      this.dropzoneHt = new window.Dropzone(htArea, {
        onFile: (fileInfo) => {
          this.loadHalftoneSource(fileInfo.dataUrl, fileInfo.name);
        },
        acceptedTypes: ["image/png", "image/jpeg", "image/webp"],
        maxSizeMb: 50,
      });
    }

    // Dropzone setup for Manga Screen AI tab
    const mangaArea = document.getElementById("manga-dropzone");
    if (mangaArea && window.Dropzone) {
      this.dropzoneManga = new window.Dropzone(mangaArea, {
        onFile: (fileInfo) => {
          this.loadMangaSource(fileInfo.dataUrl, fileInfo.name);
        },
        acceptedTypes: ["image/png", "image/jpeg", "image/webp"],
        maxSizeMb: 50,
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

    // Auto-load history, models, or templates tab
    if (tabName === "history") {
      this.loadHistory();
    } else if (tabName === "models") {
      this.renderModelsTab();
    } else if (tabName === "templates") {
      this.renderPresetsUI();
    } else if (tabName === "halftone") {
      this.initHalftoneStudio();
    } else if (tabName === "manga") {
      this.initMangaStudio();
    }
  },

  toggleNegativePrompt() {
    const content = document.getElementById("negative-prompt-content");
    const input = document.getElementById("negative-prompt-input");
    const pill = document.getElementById("dock-neg-pill");
    if (!content) return;
    const isHidden = content.style.display === "none";
    content.style.display = isHidden ? "block" : "none";
    if (pill) pill.classList.toggle("active", isHidden);
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

        // Sort text encoders: installed first, then recommended, then others
        const sortedEncoders = [...modelState.text_encoders].sort((a, b) => {
          if (Boolean(a.installed) !== Boolean(b.installed)) {
            return a.installed ? -1 : 1;
          }
          if (Boolean(a.recommended) !== Boolean(b.recommended)) {
            return a.recommended ? -1 : 1;
          }
          return 0;
        });

        // Determine active encoder: pick first installed if available
        let activeTe = sortedEncoders.find(te => te.id === this.state.selectedTextEncoder);
        if (!activeTe || (!activeTe.installed && sortedEncoders.some(te => te.installed))) {
          activeTe = sortedEncoders.find(te => te.installed) || sortedEncoders.find(te => te.recommended) || sortedEncoders[0];
          if (activeTe) {
            this.state.selectedTextEncoder = activeTe.id;
          }
        }

        for (const te of sortedEncoders) {
          const opt = document.createElement("option");
          opt.value = te.id;
          opt.textContent = `${te.name} (${te.installed ? '✓' : '⬇'})`;
          if (this.state.selectedTextEncoder === te.id) {
            opt.selected = true;
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
    this.updateDockPills();
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
    // Hide intrusive separate button inside dock
    const dlBtn = document.getElementById("download-model-btn");
    if (dlBtn) dlBtn.style.display = "none";

    // Manage canvas floating notice (#gen-uninstalled-notice)
    const notice = document.getElementById("gen-uninstalled-notice");
    const noticeText = document.getElementById("gen-uninstalled-notice-text");
    if (notice) {
      const activeModelObj = this.state.models.find(m => m.id === (modelState?.model_id || this.state.selectedModel));
      const activeTe = modelState?.text_encoders?.find(t => t.id === this.state.selectedTextEncoder);
      if (showButton) {
        if (activeModelObj && !activeModelObj.installed) {
          if (noticeText) noticeText.textContent = `Модель «${activeModelObj.name}» не скачана. Скачайте её в Model Hub перед генерацией.`;
        } else if (activeTe && !activeTe.installed && this.state.selectedTextEncoder !== "custom") {
          if (noticeText) noticeText.textContent = `Текстовый энкодер «${activeTe.name}» не скачан. Скачайте его в Model Hub.`;
        } else if (modelState && !modelState.base_model_installed) {
          if (noticeText) noticeText.textContent = `Базовые веса модели не скачаны. Скачайте их в Model Hub.`;
        } else {
          if (noticeText) noticeText.textContent = `Компоненты модели не скачаны. Перейдите в Model Hub для загрузки.`;
        }
        notice.style.display = "flex";
      } else {
        notice.style.display = "none";
      }
    }
  },

  async downloadModel() {
    if (!this.state.selectedModel) return;
    const activeModel = this.state.models.find(m => m.id === this.state.selectedModel);
    if (activeModel && activeModel.disabled) {
      this.toast("error", activeModel.disabled_reason || "Модель отключена");
      return;
    }
    if (activeModel && !activeModel.installed) {
      const notice = document.getElementById("gen-uninstalled-notice");
      const noticeText = document.getElementById("gen-uninstalled-notice-text");
      if (notice) {
        if (noticeText) noticeText.textContent = `Модель «${activeModel.name}» не скачана. Перейдите в Model Hub для загрузки.`;
        notice.style.display = "flex";
      }
      this.toast("warning", `Модель «${activeModel.name}» не скачана. Скачайте её в Model Hub.`);
      return;
    }
    if (false) {
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

    if (activeModel && activeModel.installed === false) {
      this.toast("warning", `Модель "${activeModel.name}" не скачана. Перейдите в Model Hub для загрузки.`);
      const noticeEl = document.getElementById("gen-uninstalled-notice");
      const noticeText = document.getElementById("gen-uninstalled-notice-text");
      if (noticeText) noticeText.textContent = `Модель «${activeModel.name}» не скачана. Перейдите в Model Hub для загрузки.`;
      if (noticeEl) noticeEl.style.display = "flex";
      return;
    }

    const teSelect = document.getElementById("text-encoder-select");
    const activeTe = this.state.modelState?.text_encoders?.find(t => t.id === teSelect?.value);
    if (activeTe && !activeTe.installed && teSelect?.value !== "custom") {
      this.toast("warning", `Текстовый энкодер "${activeTe.name}" не скачан. Скачайте его в Model Hub.`);
      const noticeEl = document.getElementById("gen-uninstalled-notice");
      const noticeText = document.getElementById("gen-uninstalled-notice-text");
      if (noticeText) noticeText.textContent = `Текстовый энкодер «${activeTe.name}» не скачан. Скачайте его в Model Hub.`;
      if (noticeEl) noticeEl.style.display = "flex";
      return;
    }

    if (activeModel?.family === "flux3") {
      const extrasEnv = this.state.extrasStatus?.env;
      if (extrasEnv?.nunchaku && extrasEnv.has_flux3 === false) {
        this.toast("error", "FLUX.3 SVDQuant требует Nunchaku >= 1.4.0. Обновите Nunchaku в Настройки → Движки или выберите 'Flux.2 Klein (Nunchaku)'.");
        return;
      }
      if (extrasEnv?.diffusers && extrasEnv.has_diffusers_flux3 === false) {
        this.toast("error", "FLUX.3 требует diffusers из GitHub main (стабильный 0.41.0 не поддерживает FLUX.3). Переустановите Nunchaku в Настройки → Движки — установщик поставит diffusers main.");
        return;
      }
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

    // Attach active LoRA stack
    if (this.state.activeLoras && this.state.activeLoras.length > 0) {
      params.loras = this.state.activeLoras.map(l => ({
        name: l.name,
        strength: l.strength !== undefined ? parseFloat(l.strength) : 1.0,
      }));
    }

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
      if (this.state.currentPreviewUrl) {
        this.state.previousPreviewUrl = this.state.currentPreviewUrl;
      }
      const filename = imagePath.split(/[\/\\]/).pop();
      const url = `/outputs/${encodeURIComponent(filename)}?t=${Date.now()}`;
      img.src = url;
      this.state.currentPreviewFilename = filename;
      this.state.currentPreviewUrl = url;
      this.state.lastPreviewName = filename;
      this.state.lastPreviewUrl = url;

      if (this.state.compareActive) {
        this.updateCompareImages();
      }
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

  setHistoryFilter(filter) {
    this.state.historyFilter = filter || "all";
    document.querySelectorAll(".history-filter-chip").forEach(btn => {
      btn.classList.toggle("active", btn.dataset.filter === this.state.historyFilter);
    });
    this.renderHistory();
  },

  renderHistory() {
    const grid = document.getElementById("history-grid");
    if (!grid) return;

    const query = (document.getElementById("history-search")?.value || "").toLowerCase().trim();
    let items = this.state.historyItems || [];

    // Filter by author/shared filter
    const activeFilter = this.state.historyFilter || "all";
    const myId = this.state.account?.id || "host";

    if (activeFilter === "mine") {
      items = items.filter(it => (it.owner_id || "host") === myId);
    } else if (activeFilter === "guests") {
      items = items.filter(it => (it.owner_id || "host") !== "host");
    } else if (activeFilter === "shared") {
      items = items.filter(it => Boolean(it.shared));
    }

    if (query) {
      items = items.filter(item =>
        (item.prompt || "").toLowerCase().includes(query) ||
        (item.model_id || "").toLowerCase().includes(query) ||
        (item.name || "").toLowerCase().includes(query) ||
        (item.owner_name || "").toLowerCase().includes(query)
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
      const isHostItem = (item.owner_id || "host") === "host";
      const authorBadge = isHostItem ? "👑 Хост" : `👤 ${escapeHtml(item.owner_name || "Гость")}`;
      const sharedBadge = item.shared ? `<span class="history-shared-badge" title="Видна всем пользователям">🌐 Общая</span>` : "";

      card.innerHTML = `
        <div class="history-item-thumb-wrapper">
          <div class="history-item-badges">
            <span class="history-author-badge">${authorBadge}</span>
            ${sharedBadge}
          </div>
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
    const authorEl = document.getElementById("history-modal-author");
    const sharedStatusEl = document.getElementById("history-modal-shared-status");
    const shareBtn = document.getElementById("history-modal-share-btn");
    const shareText = document.getElementById("history-modal-share-text");

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

    const isHostItem = (item.owner_id || "host") === "host";
    const authorName = isHostItem ? "👑 Хост" : `👤 ${item.owner_name || "Гость"}`;
    if (authorEl) authorEl.textContent = authorName;
    if (sharedStatusEl) {
      sharedStatusEl.textContent = item.shared ? "🌐 Доступна всем" : "🔒 Личная (только для автора)";
      sharedStatusEl.style.color = item.shared ? "#34d399" : "var(--text-dim)";
    }

    if (shareBtn) {
      const myId = this.state.account?.id || "host";
      const canShare = !this.state.account?.is_guest || (item.owner_id === myId);
      shareBtn.style.display = canShare ? "inline-flex" : "none";
      if (shareText) {
        shareText.textContent = item.shared ? "Скрыть из общей истории" : "Поделиться в общую историю";
      }
    }

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

    const openFolderBtn = document.getElementById("history-modal-open-folder-btn");
    if (openFolderBtn) {
      openFolderBtn.onclick = () => {
        this.openFolder("outputs", item.name);
      };
    }

    modal.style.display = "flex";
  },

  async toggleShareCurrentHistoryItem() {
    const item = this.state.selectedHistoryItem;
    if (!item) return;
    try {
      const res = await this.apiPost(`/api/history/${encodeURIComponent(item.name)}/share`, {
        shared: !item.shared,
      });
      if (res && res.status === "ok") {
        item.shared = res.shared;
        this.toast("success", item.shared ? "Картинка добавлена в общую историю 🌐" : "Картинка скрыта из общей истории 🔒");
        this.openHistoryModal(item);
        this.renderHistory();
      }
    } catch (e) {
      this.toast("error", "Не удалось изменить статус доступа: " + e.message);
    }
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

  // ---------- Hugging Face Token & Guest Accounts ----------
  toggleTokenVisibility(inputId) {
    const input = document.getElementById(inputId);
    if (!input) return;
    input.type = input.type === "password" ? "text" : "password";
  },

  async saveHfToken() {
    const token = document.getElementById("setting-hf-token")?.value?.trim() || "";
    try {
      await this.apiPatch("/api/settings", { huggingface_token: token });
      if (this.state.settings) this.state.settings.huggingface_token = token;
      this.toast("success", "Токен Hugging Face сохранён ✓");
    } catch (e) {
      this.toast("error", "Ошибка сохранения токена: " + e.message);
    }
  },

  async testHfToken() {
    const token = document.getElementById("setting-hf-token")?.value?.trim();
    const btn = document.getElementById("btn-test-hf-token");
    if (btn) btn.textContent = "⏳ Проверка...";
    try {
      const res = await this.apiPost("/api/hf/test-token", { token: token || undefined });
      if (res.valid) {
        this.toast("success", res.message || "Токен действителен ✓");
      } else {
        this.toast("error", res.message || "Неверный токен Hugging Face");
      }
    } catch (e) {
      this.toast("error", "Ошибка проверки: " + e.message);
    } finally {
      if (btn) btn.textContent = "🔍 Проверить";
    }
  },

  async loadGuestAccounts() {
    if (this.state.account?.is_guest) return;
    try {
      const res = await this.apiGet("/api/accounts");
      if (res && res.accounts) {
        this.state.guestAccounts = res.accounts;
        this.renderGuestAccounts(res.accounts);
      }
    } catch (e) {
      console.warn("[accounts] Could not load guest accounts:", e);
    }
  },

  renderGuestAccounts(accounts) {
    const list = document.getElementById("guest-accounts-list");
    if (!list) return;

    if (!accounts || !accounts.length) {
      list.innerHTML = `
        <div style="color: var(--text-dim); font-size: 12.5px; padding: 12px; text-align: center; background: var(--bg-card); border-radius: 8px; border: 1px dashed var(--border);">
          Нет созданных гостевых аккаунтов. Нажмите «+ Новый гость», чтобы пригласить друга с постоянным паролем и изолированной историей.
        </div>
      `;
      return;
    }

    list.innerHTML = "";
    for (const acc of accounts) {
      const card = document.createElement("div");
      card.className = "guest-account-card";

      const createdDate = acc.created_at ? new Date(acc.created_at * 1000).toLocaleDateString() : "";
      const pwdDisplay = acc.initial_password
        ? `<span style="font-family: monospace; color: #a78bfa;">${escapeHtml(acc.initial_password)}</span>`
        : `<span style="color: var(--text-dim); font-size: 11px;">(изменён гостем)</span>`;

      card.innerHTML = `
        <div class="guest-account-info">
          <div class="guest-account-name">
            <span>👤 ${escapeHtml(acc.username)}</span>
            <small style="color: var(--text-dim); font-weight: normal; font-size: 11px;">(${createdDate})</small>
          </div>
          <div class="guest-account-token" style="display: flex; gap: 12px; font-size: 11px;">
            <span>Пароль: ${pwdDisplay}</span>
            <span style="color: var(--text-dim);">Токен: ${escapeHtml((acc.token || '').slice(0, 10))}...</span>
          </div>
        </div>
        <div class="guest-account-actions">
          <button type="button" class="setting-btn" onclick="App.copyGuestCredentials('${escapeHtml(acc.username)}', '${escapeHtml(acc.initial_password || '')}')" title="Скопировать ссылку, логин и пароль">📋 Данные</button>
          <button type="button" class="setting-btn" onclick="App.showCreateGuestModal('${acc.id}', '${escapeHtml(acc.username)}', '${escapeHtml(acc.initial_password || '')}')" title="Редактировать имя или пароль">✏️</button>
          <button type="button" class="setting-btn danger" onclick="App.deleteGuestAccount('${acc.id}')" title="Удалить аккаунт">🗑️</button>
        </div>
      `;
      list.appendChild(card);
    }
  },

  generateRandomGuestPassword() {
    const chars = "abcdefghjkmnpqrstuvwxyz23456789";
    let pass = "guest_";
    for (let i = 0; i < 6; i++) {
      pass += chars.charAt(Math.floor(Math.random() * chars.length));
    }
    const pwdInput = document.getElementById("guest-account-password-input");
    if (pwdInput) pwdInput.value = pass;
    return pass;
  },

  showCreateGuestModal(accountId = null, currentName = "", currentPassword = "") {
    const modal = document.getElementById("guest-account-modal");
    const title = document.getElementById("guest-modal-title");
    const idInput = document.getElementById("guest-account-edit-id");
    const nameInput = document.getElementById("guest-account-username-input");
    const pwdInput = document.getElementById("guest-account-password-input");
    if (!modal) return;

    if (accountId) {
      if (title) title.textContent = "Редактировать гостевой аккаунт";
      if (idInput) idInput.value = accountId;
      if (nameInput) nameInput.value = currentName || "";
      if (pwdInput) pwdInput.value = currentPassword || "";
    } else {
      if (title) title.textContent = "Новый гостевой аккаунт";
      if (idInput) idInput.value = "";
      if (nameInput) nameInput.value = "";
      this.generateRandomGuestPassword();
    }
    modal.style.display = "flex";
    if (nameInput) nameInput.focus();
  },

  closeCreateGuestModal() {
    const modal = document.getElementById("guest-account-modal");
    if (modal) modal.style.display = "none";
  },

  async confirmCreateGuestAccount() {
    const idInput = document.getElementById("guest-account-edit-id");
    const nameInput = document.getElementById("guest-account-username-input");
    const pwdInput = document.getElementById("guest-account-password-input");
    const username = (nameInput?.value || "").trim();
    const password = (pwdInput?.value || "").trim();

    if (!username) {
      this.toast("error", "Укажите имя гостя");
      return;
    }
    const editId = idInput?.value;
    try {
      if (editId) {
        await this.apiPatch(`/api/accounts/${encodeURIComponent(editId)}`, { username, password });
        this.toast("success", "Аккаунт обновлён ✓");
      } else {
        const res = await this.apiPost("/api/accounts", { username, password });
        this.toast("success", `Аккаунт '${username}' создан ✓`);
        if (res && res.account) {
          this.copyGuestCredentials(username, password || res.account.initial_password);
        }
      }
      this.closeCreateGuestModal();
      await this.loadGuestAccounts();
    } catch (e) {
      this.toast("error", "Ошибка: " + e.message);
    }
  },

  async deleteGuestAccount(accountId) {
    if (!confirm("Удалить этот гостевой аккаунт? Его ссылка и пароль перестанут работать.")) return;
    try {
      await this.apiDelete(`/api/accounts/${encodeURIComponent(accountId)}`);
      this.toast("success", "Аккаунт удалён ✓");
      await this.loadGuestAccounts();
    } catch (e) {
      this.toast("error", "Не удалось удалить аккаунт: " + e.message);
    }
  },

  copyGuestCredentials(username, password) {
    // Determine active tunnel url or current host
    const tunnelInput = document.getElementById("setting-share-url");
    let tunnelUrl = tunnelInput && tunnelInput.value && !tunnelInput.value.includes("...")
      ? tunnelInput.value.trim()
      : window.location.origin;

    // Strip any query parameters or trailing slashes to guarantee clean URL
    try {
      const parsed = new URL(tunnelUrl);
      tunnelUrl = `${parsed.protocol}//${parsed.host}`;
    } catch (_) {
      tunnelUrl = tunnelUrl.split("?")[0].replace(/\/+$/, "");
    }

    const pwdText = password ? password : "(сохранённый вами пароль)";
    const text = `⚡ LazySD UI — Доступ к генератору\nСсылка: ${tunnelUrl}\nЛогин: ${username}\nПароль: ${pwdText}`;

    navigator.clipboard.writeText(text)
      .then(() => this.toast("success", "Данные доступа (ссылка, логин, пароль) скопированы в буфер! 📋"))
      .catch(() => {
        prompt("Скопируйте данные для гостя:", text);
      });
  },

  copyGuestLink() {
    const url = window.location.origin;
    navigator.clipboard.writeText(url)
      .then(() => this.toast("success", "Ссылка на вход скопирована в буфер обмена! 📋"))
      .catch(() => this.toast("info", `Ссылка: ${url}`));
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

  async openFolder(folderType = "outputs", filename = null) {
    if (!filename && folderType === "outputs") {
      filename = this.state.currentPreviewFilename || null;
      if (!filename) {
        try {
          const previewImg = document.getElementById("preview-img");
          if (previewImg && previewImg.src) {
            const u = new URL(previewImg.src, window.location.href);
            const parts = u.pathname.split("/");
            const last = parts[parts.length - 1];
            if (last && (last.endsWith(".png") || last.endsWith(".jpg") || last.endsWith(".webp"))) {
              filename = decodeURIComponent(last);
            }
          }
        } catch (e) {}
      }
    }

    try {
      const res = await this.apiPost("/api/system/open-folder", {
        folder_type: folderType,
        filename: filename,
      });
      if (res.status === "ok") {
        const label = filename
          ? `Файл «${filename}» открыт в проводнике`
          : `Папка открыта: ${res.folder || res.path || ""}`;
        this.toast("success", "📂 " + label);
      } else if (res.status === "remote") {
        this.toast("info", "Удалённая сессия: открытие проводника на сервере недоступно. Переход в Историю...");
        this.switchTab("history");
      } else {
        this.toast("error", res.message || "Не удалось открыть папку");
      }
    } catch (e) {
      console.warn("openFolder error:", e);
      this.toast("error", "Не удалось открыть папку: " + e.message);
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
  onRefModeChange(preserveStrength = false) {
    const mode = document.getElementById("img2img-mode")?.value || "control_ref";
    const hintEl = document.getElementById("img2img-mode-hint");
    const statusDot = document.getElementById("img2img-status-dot");
    const strengthLabel = document.getElementById("img2img-strength-label");
    const strengthSlider = document.getElementById("img2img-strength");
    const strengthVal = document.getElementById("img2img-strength-val");

    if (mode === "control_ref") {
      const tipText = "✓ Изоляция шагов активна: --init-img и --strength отключены. Модель выполнит 100% шагов с чистого листа, перенося лайн и скринтоны через нативный Reference / ControlNet.";
      if (hintEl) {
        hintEl.style.color = "var(--success, #22c55e)";
        hintEl.innerHTML = "✓ Изоляция шагов активна: <code>--init-img</code> и <code>--strength</code> отключены. Модель выполнит 100% шагов с чистого листа, перенося лайн и скринтоны через нативный Reference / ControlNet.";
      }
      if (statusDot) {
        statusDot.style.background = "#10B981";
        statusDot.title = tipText;
      }
      if (strengthLabel) strengthLabel.textContent = "Сила:";
      if (strengthSlider) {
        strengthSlider.max = "1.00";
        if (!preserveStrength) strengthSlider.value = "0.85";
      }
      if (strengthVal && !preserveStrength) strengthVal.textContent = "0.85";
    } else if (mode === "controlnet_canny") {
      const tipText = "✒️ Canny / Lineart препроцессор: выделяет контуры туши и передает в ControlNet / Reference при 100% шагов с чистого белого шума.";
      if (hintEl) {
        hintEl.style.color = "var(--info, #38bdf8)";
        hintEl.innerHTML = tipText;
      }
      if (statusDot) {
        statusDot.style.background = "#38bdf8";
        statusDot.title = tipText;
      }
      if (strengthLabel) strengthLabel.textContent = "Контур:";
      if (strengthSlider) {
        strengthSlider.max = "1.00";
        if (!preserveStrength) strengthSlider.value = "0.90";
      }
      if (strengthVal && !preserveStrength) strengthVal.textContent = "0.90";
    } else {
      const tipText = "⚠️ Классический Img2Img (зашумление исходника через --init-img + --strength). Может размывать белый фон и тонкие скринтоны.";
      if (hintEl) {
        hintEl.style.color = "var(--warning, #fbbf24)";
        hintEl.innerHTML = "⚠️ Классический Img2Img (зашумление исходника через <code>--init-img</code> + <code>--strength</code>). Может размывать белый фон и тонкие скринтоны.";
      }
      if (statusDot) {
        statusDot.style.background = "#F59E0B";
        statusDot.title = tipText;
      }
      if (strengthLabel) strengthLabel.textContent = "Шум:";
      if (strengthSlider) {
        strengthSlider.max = "0.95";
        if (!preserveStrength) strengthSlider.value = "0.70";
      }
      if (strengthVal && !preserveStrength) strengthVal.textContent = "0.70";
    }
  },

  toggleControlNetInput(e) {
    if (e) e.preventDefault();
    const wrap = document.getElementById("controlnet-custom-wrap");
    const input = document.getElementById("controlnet-custom-path");
    if (!input) return;
    if (wrap) {
      const isHidden = wrap.style.display === "none";
      wrap.style.display = isHidden ? "block" : "none";
      input.style.display = isHidden ? "block" : "none";
      if (isHidden) input.focus();
    } else {
      const isHidden = input.style.display === "none";
      input.style.display = isHidden ? "block" : "none";
      if (isHidden) input.focus();
    }
  },

  onDockFileSelected(inputEl) {
    const file = inputEl?.files?.[0];
    if (file) {
      this.uploadImg2ImgFile(file);
      inputEl.value = "";
    }
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
      this.onRefModeChange(true);
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

  // ---------- Gemini Floating Dock & Popovers ----------
  autoResizePrompt(textarea) {
    if (!textarea) return;
    textarea.style.height = "auto";
    const nextH = Math.max(46, Math.min(150, textarea.scrollHeight));
    textarea.style.height = `${nextH}px`;
  },

  updateDockPills() {
    // 1. Model Pill Text
    const modelPillText = document.getElementById("dock-model-pill-text");
    const activeModel = this.state.models.find(m => m.id === this.state.selectedModel);
    if (modelPillText) {
      if (activeModel) {
        const shortVar = (this.state.selectedVariant || "")
          .replace(/^(flux2_klein_|flux3_klein_|qwen_image_2_1_)/i, "")
          .replace(/-(?:50x|40x|30x-below)(?:-base)?/g, "");
        modelPillText.textContent = shortVar ? `${activeModel.name} (${shortVar})` : activeModel.name;
      } else {
        modelPillText.textContent = "Модель";
      }
    }

    // 2. Resolution Pill Text & Chips
    const resVal = document.getElementById("resolution-select")?.value || "1024x1024";
    const resPillText = document.getElementById("dock-res-pill-text");
    if (resPillText) {
      resPillText.textContent = resVal.replace("x", "×");
    }
    document.querySelectorAll("#dock-res-grid .dock-res-chip").forEach(chip => {
      chip.classList.toggle("active", chip.dataset.res === resVal);
    });

    // 3. Params Pill Text
    const samplerVal = document.getElementById("sampler-select")?.value || "euler";
    const stepsVal = document.getElementById("steps-input")?.value || "4";
    const cfgVal = document.getElementById("cfg-input")?.value || "1.0";
    const noiseVal = document.getElementById("noise-input")?.value || "0.0";
    const seedVal = document.getElementById("seed-input")?.value || "";
    const paramsPillText = document.getElementById("dock-params-pill-text");
    if (paramsPillText) {
      const sLabel = samplerVal === "euler_a" ? "Euler A" : (samplerVal === "heun" ? "Heun" : "Euler");
      paramsPillText.textContent = `${sLabel} · ${stepsVal} · CFG ${parseFloat(cfgVal || 1).toFixed(1)}`;
    }

    // 4. Negative Prompt Pill State
    const negVal = document.getElementById("negative-prompt-input")?.value?.trim() || "";
    const negPill = document.getElementById("dock-neg-pill");
    if (negPill) {
      negPill.classList.toggle("has-neg", negVal.length > 0);
    }

    // 5. Sync Quick Popover Selects with Inspector Controls
    const mainModelSel = document.getElementById("model-select");
    const dockModelSel = document.getElementById("dock-model-quick-select");
    if (mainModelSel && dockModelSel) {
      dockModelSel.innerHTML = mainModelSel.innerHTML;
      dockModelSel.value = mainModelSel.value;
    }

    const mainVarSel = document.getElementById("variant-select");
    const dockVarSel = document.getElementById("dock-variant-quick-select");
    if (mainVarSel && dockVarSel) {
      dockVarSel.innerHTML = mainVarSel.innerHTML;
      dockVarSel.value = mainVarSel.value;
    }

    const mainTeGroup = document.getElementById("text-encoder-group");
    const mainTeSel = document.getElementById("text-encoder-select");
    const dockTeRow = document.getElementById("dock-te-quick-field") || document.getElementById("dock-te-quick-row");
    const dockTeSel = document.getElementById("dock-te-quick-select");
    if (mainTeGroup && mainTeSel && dockTeRow && dockTeSel) {
      const visible = mainTeGroup.style.display !== "none";
      dockTeRow.style.display = visible ? "flex" : "none";
      if (visible) {
        dockTeSel.innerHTML = mainTeSel.innerHTML;
        dockTeSel.value = mainTeSel.value;
        dockTeSel.disabled = mainTeSel.disabled;
      }
    }

    const dockSampler = document.getElementById("dock-sampler-quick");
    if (dockSampler) dockSampler.value = samplerVal;
    const dockSteps = document.getElementById("dock-steps-quick");
    if (dockSteps) dockSteps.value = stepsVal;
    const dockCfg = document.getElementById("dock-cfg-quick");
    if (dockCfg) dockCfg.value = cfgVal;
    const dockCfgLbl = document.getElementById("dock-cfg-val-label");
    if (dockCfgLbl) dockCfgLbl.textContent = cfgVal;
    const dockNoise = document.getElementById("dock-noise-quick");
    if (dockNoise) dockNoise.value = noiseVal;
    const dockNoiseLbl = document.getElementById("dock-noise-val-label");
    if (dockNoiseLbl) dockNoiseLbl.textContent = noiseVal;
    const dockSeed = document.getElementById("dock-seed-quick");
    if (dockSeed) dockSeed.value = seedVal;
  },

  toggleDockPopover(name, event) {
    if (event) event.stopPropagation();
    const map = {
      model: "dock-popover-model",
      res: "dock-popover-res",
      params: "dock-popover-params",
      presets: "dock-popover-presets",
    };
    const targetId = map[name];
    if (!targetId) return;

    if (this.state.activeDockPopover === name) {
      this.closeAllDockPopovers();
      return;
    }

    this.closeAllDockPopovers();
    const el = document.getElementById(targetId);
    if (el) {
      this.updateDockPills();
      if (name === "presets") this.renderPresetsUI();
      el.style.display = "block";
      this.state.activeDockPopover = name;
    }
  },

  closeAllDockPopovers() {
    ["dock-popover-model", "dock-popover-res", "dock-popover-params", "dock-popover-presets"].forEach(id => {
      const el = document.getElementById(id);
      if (el) el.style.display = "none";
    });
    this.state.activeDockPopover = null;
  },

  async onDockModelSelect(modelId) {
    const mainModelSel = document.getElementById("model-select");
    if (mainModelSel) {
      mainModelSel.value = modelId;
      await this.onModelChange();
    }
  },

  onDockVariantSelect(variantId) {
    const mainVarSel = document.getElementById("variant-select");
    if (mainVarSel) {
      mainVarSel.value = variantId;
      mainVarSel.dispatchEvent(new Event("change"));
    }
  },

  onDockTeSelect(teId) {
    const mainTeSel = document.getElementById("text-encoder-select");
    if (mainTeSel) {
      mainTeSel.value = teId;
      this.onTextEncoderChange();
    }
  },

  selectResolutionChip(resValue) {
    const resSel = document.getElementById("resolution-select");
    if (resSel) {
      resSel.value = resValue;
      this.updateDockPills();
    }
    this.closeAllDockPopovers();
  },

  syncFromDockParams(field, val) {
    if (field === "sampler") {
      const el = document.getElementById("sampler-select");
      if (el) el.value = val;
    } else if (field === "steps") {
      const el = document.getElementById("steps-input");
      if (el) el.value = val;
    } else if (field === "cfg") {
      const inp = document.getElementById("cfg-input");
      const sld = document.getElementById("cfg-slider");
      const lbl = document.getElementById("dock-cfg-val-label");
      if (inp) inp.value = val;
      if (sld) sld.value = val;
      if (lbl) lbl.textContent = val;
    } else if (field === "noise") {
      const inp = document.getElementById("noise-input");
      const sld = document.getElementById("noise-slider");
      const lbl = document.getElementById("dock-noise-val-label");
      if (inp) inp.value = val;
      if (sld) sld.value = val;
      if (lbl) lbl.textContent = val;
      const n = parseFloat(val);
      const sSel = document.getElementById("sampler-select");
      if (!isNaN(n) && n > 0 && sSel && sSel.value === "euler") {
        sSel.value = "euler_a";
      }
    } else if (field === "seed") {
      const el = document.getElementById("seed-input");
      if (el) el.value = val;
    }
    this.updateModelTrigger();
  },

  // ---------- Presets / Templates System ----------
  async loadPresets() {
    try {
      const res = await this.apiGet("/api/presets");
      this.state.presets = (res && res.presets) ? res.presets : (Array.isArray(res) ? res : []);
      this.renderPresetsUI();
    } catch (e) {
      console.warn("[app] Failed to load presets:", e);
      this.state.presets = [];
      this.renderPresetsUI();
    }
  },

  renderPresetsUI() {
    const presets = this.state.presets || [];

    // 1. Render inside Gemini Dock Popover (#dock-presets-list)
    const dockList = document.getElementById("dock-presets-list");
    if (dockList) {
      dockList.innerHTML = "";
      if (presets.length === 0) {
        dockList.innerHTML = `
          <div style="padding: 16px; text-align: center; color: var(--text-dim); font-size: 12px;">
            У вас пока нет сохранённых шаблонов.
            <div style="margin-top: 10px;">
              <button type="button" class="generate-btn" style="height: 30px; padding: 0 14px; font-size: 11.5px;" onclick="App.openSavePresetModal(); App.closeAllDockPopovers();">💾 Сохранить текущие настройки в шаблон</button>
            </div>
          </div>
        `;
      }
      for (const p of presets) {
        const row = document.createElement("div");
        row.className = "dock-preset-item";
        const editBtn = p.can_edit
          ? `<button class="history-action-btn" style="padding: 2px 6px; font-size: 11px; height: 22px; width: auto;" onclick="App.openEditPresetModal('${escapeHtml(p.id)}', event)" title="Редактировать шаблон">✏️</button>`
          : "";
        const delBtn = p.can_delete
          ? `<button class="delete-custom-btn" onclick="App.deletePreset('${escapeHtml(p.id)}', event)" title="Удалить шаблон">🗑</button>`
          : "";
        row.innerHTML = `
          <div class="dock-preset-info">
            <span class="dock-preset-icon">${escapeHtml(p.icon || "✨")}</span>
            <div style="min-width: 0;">
              <div class="dock-preset-title">${escapeHtml(p.title)}</div>
              <div class="dock-preset-sub">${escapeHtml(p.description || "")}</div>
            </div>
          </div>
          <div style="display: flex; gap: 4px; align-items: center;">
            ${editBtn}
            ${delBtn}
          </div>
        `;
        row.addEventListener("click", (e) => {
          if (e.target.tagName === "BUTTON") return;
          this.applyPreset(p.id);
          this.closeAllDockPopovers();
        });
        dockList.appendChild(row);
      }
    }

    // 2. Render inside Templates Tab (#templates-grid)
    const grid = document.getElementById("templates-grid");
    if (grid) {
      grid.innerHTML = "";
      for (const p of presets) {
        const card = document.createElement("div");
        card.className = "model-card";
        const ownerBadge = p.owner_id === "host"
          ? `<span class="badge-installed">👑 Шаблон хоста</span>`
          : `<span class="badge-custom">👤 ${escapeHtml(p.owner_name || "Гость")}</span>`;
        const privBadge = p.is_private
          ? `<span class="badge-custom" style="background: rgba(239, 68, 68, 0.15); border-color: rgba(239, 68, 68, 0.3); color: #f87171;">🔒 Скрытый</span>`
          : "";
        const editBtn = p.can_edit
          ? `<button class="card-action-btn" style="width: auto;" onclick="App.openEditPresetModal('${escapeHtml(p.id)}')">✏️ Изменить</button>`
          : "";
        const delBtn = p.can_delete
          ? `<button class="delete-custom-btn" onclick="App.deletePreset('${escapeHtml(p.id)}', event)">🗑 Удалить</button>`
          : "";
        card.innerHTML = `
          <div>
            <div class="model-card-header">
              <span class="model-card-title">${escapeHtml(p.icon || "✨")} ${escapeHtml(p.title)}</span>
              <div class="model-card-badges" style="display: flex; gap: 6px; flex-wrap: wrap;">
                ${ownerBadge}
                ${privBadge}
              </div>
            </div>
            <p class="model-card-description" style="margin-top: 8px;">${escapeHtml(p.description || "")}</p>
            <div class="model-card-reqs" style="margin-top: 10px;">
              <span>📐 ${escapeHtml(p.resolution || "1024x1024")}</span>
              <span>⏱ ${p.steps || 20} steps</span>
              <span>🎛️ CFG ${p.cfg ?? 3.5}</span>
              <span>🖼️ ${escapeHtml(p.ref_mode || "control_ref")}</span>
            </div>
            ${p.prompt ? `<div style="margin-top: 10px; padding: 8px 10px; background: var(--bg-input); border: 1px solid var(--border); border-radius: 6px; font-size: 11px; color: var(--text-muted); max-height: 64px; overflow: hidden;">${escapeHtml(p.prompt)}</div>` : ""}
            <div style="margin-top: 12px; display: flex; gap: 8px; align-items: center; flex-wrap: wrap;">
              <button class="card-action-btn primary" style="width: auto;" onclick="App.applyPreset('${escapeHtml(p.id)}')">⚡ Применить шаблон</button>
              ${editBtn}
              ${delBtn}
            </div>
          </div>
        `;
        grid.appendChild(card);
      }
    }
  },

  async applyPreset(presetId) {
    const preset = (this.state.presets || []).find(p => p.id === presetId);
    if (!preset) return;

    // 1. Model & Variant (if specified and available)
    if (preset.model_id) {
      const targetModel = this.state.models.find(m => m.id === preset.model_id && !m.disabled);
      if (targetModel) {
        const modelSel = document.getElementById("model-select");
        if (modelSel) {
          modelSel.value = targetModel.id;
          await this.onModelChange(preset.variant_id || null);
        }
      }
    }

    // 2. Prompt & Negative Prompt
    const promptInput = document.getElementById("prompt-input");
    if (promptInput && preset.prompt) {
      const cur = promptInput.value.trim();
      if (preset.prompt_mode === "append" && cur) {
        promptInput.value = `${cur}, ${preset.prompt}`;
      } else {
        promptInput.value = preset.prompt;
      }
      this.autoResizePrompt(promptInput);
    }

    const negInput = document.getElementById("negative-prompt-input");
    const negDot = document.getElementById("negative-prompt-dot");
    if (negInput && preset.negative_prompt !== undefined) {
      negInput.value = preset.negative_prompt || "";
      if (negDot) {
        negDot.style.display = negInput.value.trim().length > 0 ? "inline-block" : "none";
      }
    }

    // 3. Resolution, Steps, CFG, Sampler, Noise
    if (preset.resolution) {
      const resSel = document.getElementById("resolution-select");
      if (resSel && resSel.querySelector(`option[value="${preset.resolution}"]`)) {
        resSel.value = preset.resolution;
      }
    }
    if (preset.steps) {
      const stepsEl = document.getElementById("steps-input");
      if (stepsEl) stepsEl.value = preset.steps;
    }
    if (preset.cfg !== undefined && preset.cfg !== null) {
      const cfgInp = document.getElementById("cfg-input");
      const cfgSld = document.getElementById("cfg-slider");
      if (cfgInp) cfgInp.value = preset.cfg;
      if (cfgSld) cfgSld.value = preset.cfg;
    }
    if (preset.sampler) {
      const sampEl = document.getElementById("sampler-select");
      if (sampEl && sampEl.querySelector(`option[value="${preset.sampler}"]`)) {
        sampEl.value = preset.sampler;
      }
    }
    if (preset.noise !== undefined && preset.noise !== null) {
      const nInp = document.getElementById("noise-input");
      const nSld = document.getElementById("noise-slider");
      if (nInp) nInp.value = preset.noise;
      if (nSld) nSld.value = preset.noise;
    }

    // 4. Reference / ControlNet mode & strength
    if (preset.ref_mode) {
      const modeEl = document.getElementById("img2img-mode");
      if (modeEl && modeEl.querySelector(`option[value="${preset.ref_mode}"]`)) {
        modeEl.value = preset.ref_mode;
        this.onRefModeChange(true);
      }
    }
    if (preset.strength !== undefined && preset.strength !== null) {
      const strSld = document.getElementById("img2img-strength");
      const strVal = document.getElementById("img2img-strength-val");
      if (strSld) strSld.value = preset.strength;
      if (strVal) strVal.textContent = Number(preset.strength).toFixed(2);
    }

    this.updateModelTrigger();
    if (this.state.currentTab !== "generate") {
      this.switchTab("generate");
    }
    this.toast("success", `Шаблон «${preset.title}» применён ✓`);
  },

  openSavePresetModal() {
    this.closeAllDockPopovers();
    const modal = document.getElementById("save-preset-modal");
    if (modal) modal.style.display = "flex";

    const idInput = document.getElementById("preset-save-id");
    if (idInput) idInput.value = "";
    const titleHeader = document.getElementById("preset-modal-title");
    if (titleHeader) titleHeader.textContent = "⭐ Сохранить шаблон (Preset)";
    const titleInput = document.getElementById("preset-save-name");
    if (titleInput) {
      titleInput.value = "";
      titleInput.focus();
    }
    const iconInput = document.getElementById("preset-save-icon");
    if (iconInput) iconInput.value = "⭐";
    const catInput = document.getElementById("preset-save-category");
    if (catInput) catInput.value = "Пользовательские";
    const descInput = document.getElementById("preset-save-desc");
    if (descInput) descInput.value = "";

    const promptPre = document.getElementById("preset-save-prompt");
    if (promptPre) promptPre.value = document.getElementById("prompt-input")?.value || "";
    const negPre = document.getElementById("preset-save-neg");
    if (negPre) negPre.value = document.getElementById("negative-prompt-input")?.value || "";

    const stepsPre = document.getElementById("preset-save-steps");
    if (stepsPre) stepsPre.value = document.getElementById("steps-input")?.value || "20";
    const cfgPre = document.getElementById("preset-save-cfg");
    if (cfgPre) cfgPre.value = document.getElementById("cfg-input")?.value || "3.5";
    const noisePre = document.getElementById("preset-save-noise");
    if (noisePre) noisePre.value = document.getElementById("noise-input")?.value || "0.0";
    const privPre = document.getElementById("preset-save-private");
    if (privPre) privPre.checked = false;
  },

  openEditPresetModal(presetId, event) {
    if (event) event.stopPropagation();
    this.closeAllDockPopovers();
    const preset = (this.state.presets || []).find(p => p.id === presetId);
    if (!preset) return;

    const modal = document.getElementById("save-preset-modal");
    if (modal) modal.style.display = "flex";

    const idInput = document.getElementById("preset-save-id");
    if (idInput) idInput.value = preset.id;
    const titleHeader = document.getElementById("preset-modal-title");
    if (titleHeader) titleHeader.textContent = `✏️ Редактировать шаблон: «${preset.title}»`;

    const titleInput = document.getElementById("preset-save-name");
    if (titleInput) {
      titleInput.value = preset.title || "";
      titleInput.focus();
    }
    const iconInput = document.getElementById("preset-save-icon");
    if (iconInput) iconInput.value = preset.icon || "⭐";
    const catInput = document.getElementById("preset-save-category");
    if (catInput) catInput.value = preset.category || "Пользовательские";
    const descInput = document.getElementById("preset-save-desc");
    if (descInput) descInput.value = preset.description || "";

    const promptPre = document.getElementById("preset-save-prompt");
    if (promptPre) promptPre.value = preset.prompt || preset.prompt_template || "";
    const negPre = document.getElementById("preset-save-neg");
    if (negPre) negPre.value = preset.negative_prompt || "";

    const stepsPre = document.getElementById("preset-save-steps");
    if (stepsPre) stepsPre.value = preset.steps || 20;
    const cfgPre = document.getElementById("preset-save-cfg");
    if (cfgPre) cfgPre.value = preset.cfg ?? 3.5;
    const noisePre = document.getElementById("preset-save-noise");
    if (noisePre) noisePre.value = preset.noise ?? 0.0;
    const privPre = document.getElementById("preset-save-private");
    if (privPre) privPre.checked = Boolean(preset.is_private);
  },

  closeSavePresetModal() {
    const modal = document.getElementById("save-preset-modal");
    if (modal) modal.style.display = "none";
  },

  async saveCurrentAsPreset() {
    const titleEl = document.getElementById("preset-save-name");
    const title = titleEl?.value?.trim();
    if (!title) {
      this.toast("warning", "Введите название шаблона");
      return;
    }
    const editId = document.getElementById("preset-save-id")?.value?.trim() || null;
    const promptVal = document.getElementById("preset-save-prompt")?.value?.trim()
      ?? document.getElementById("prompt-input")?.value?.trim()
      ?? "";
    const negVal = document.getElementById("preset-save-neg")?.value?.trim()
      ?? document.getElementById("negative-prompt-input")?.value?.trim()
      ?? "";

    const stepsVal = parseInt(document.getElementById("preset-save-steps")?.value || "20", 10);
    const cfgVal = parseFloat(document.getElementById("preset-save-cfg")?.value || "3.5");
    const noiseVal = parseFloat(document.getElementById("preset-save-noise")?.value || "0.0");
    const isPrivate = Boolean(document.getElementById("preset-save-private")?.checked);

    const payload = {
      id: editId,
      title: title,
      icon: document.getElementById("preset-save-icon")?.value?.trim() || "⭐",
      category: document.getElementById("preset-save-category")?.value?.trim() || "Пользовательские",
      description: document.getElementById("preset-save-desc")?.value?.trim() || "",
      prompt: promptVal,
      negative_prompt: negVal,
      model_id: this.state.selectedModel || null,
      variant_id: document.getElementById("variant-select")?.value || this.state.selectedVariant || null,
      ref_mode: document.getElementById("img2img-mode")?.value || "control_ref",
      strength: parseFloat(document.getElementById("img2img-strength")?.value || "0.85"),
      resolution: document.getElementById("resolution-select")?.value || "1024x1024",
      steps: stepsVal,
      cfg: cfgVal,
      sampler: document.getElementById("sampler-select")?.value || "euler",
      noise: noiseVal,
      is_private: isPrivate,
    };

    try {
      await this.apiPost("/api/presets", payload);
      this.closeSavePresetModal();
      await this.loadPresets();
      this.toast("success", editId ? `Шаблон «${title}» обновлён ✓` : `Шаблон «${title}» сохранён ✓`);
    } catch (e) {
      this.toast("error", e.message);
    }
  },

  async deletePreset(presetId, event) {
    if (event) event.stopPropagation();
    if (!confirm("Удалить этот шаблон?")) return;
    try {
      const res = await fetch(`/api/presets/${encodeURIComponent(presetId)}`, { method: "DELETE" });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || `HTTP ${res.status}`);
      }
      await this.loadPresets();
      this.toast("info", "Шаблон удалён ✓");
    } catch (e) {
      this.toast("error", e.message);
    }
  },

  // ---------- Universal Custom Model Importer (Local + Hugging Face Hub) ----------
  async openCustomModelModal(tab = "local") {
    const modal = document.getElementById("custom-model-modal");
    if (!modal) return;

    this.switchCustomModelTab(tab);

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
    if (tab === "hf") {
      document.getElementById("hf-search-input")?.focus();
    }
  },

  switchCustomModelTab(tab) {
    const isHf = tab === "hf";
    const tabLocal = document.getElementById("importer-tab-btn-local") || document.getElementById("custom-tab-local");
    const tabHf = document.getElementById("importer-tab-btn-hf") || document.getElementById("custom-tab-hf");
    const paneLocal = document.getElementById("custom-model-pane-local");
    const paneHf = document.getElementById("custom-model-pane-hf");

    if (tabLocal) tabLocal.classList.toggle("active", !isHf);
    if (tabHf) tabHf.classList.toggle("active", isHf);
    if (paneLocal) paneLocal.style.display = !isHf ? "block" : "none";
    if (paneHf) paneHf.style.display = isHf ? "block" : "none";
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
      modelSel.innerHTML = `<option value="">— Выберите файл из models/custom/ —</option><option value="__custom__">Указать полный путь на ПК...</option>`;
      for (const f of files) {
        const opt = document.createElement("option");
        opt.value = f.rel_path;
        const roleTag = f.role === "text_encoder" ? " [Encoder]" : (f.role === "vae" ? " [VAE]" : "");
        opt.textContent = `${f.name} (${f.size_gb} GB, ${f.quant || f.ext.toUpperCase()})${roleTag}`;
        modelSel.appendChild(opt);
      }
    }

    if (teSel) {
      teSel.innerHTML = `<option value="">— Auto / Встроен в модель —</option><option value="__custom__">Указать полный путь на ПК...</option>`;
      for (const f of files) {
        const opt = document.createElement("option");
        opt.value = f.rel_path;
        opt.textContent = `${f.name} (${f.size_gb} GB)`;
        teSel.appendChild(opt);
      }
    }

    if (vaeSel) {
      vaeSel.innerHTML = `<option value="">— Auto / Встроен в модель —</option><option value="__custom__">Указать полный путь на ПК...</option>`;
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

    // Smart auto-detection of family, steps, CFG, and name when a model file is chosen
    if (type === "model" && sel.value && sel.value !== "__custom__") {
      const nameInput = document.getElementById("custom-model-name");
      if (nameInput && !nameInput.value.trim()) {
        const stem = sel.value.split(/[\\/]/).pop().replace(/\.[^/.]+$/, "");
        nameInput.value = stem.replace(/[-_]/g, " ");
      }
      const fileMeta = (this._customFilesData?.files || []).find(f => f.rel_path === sel.value);
      if (fileMeta) {
        const familySel = document.getElementById("custom-model-family");
        const stepsInput = document.getElementById("custom-model-steps");
        const cfgInput = document.getElementById("custom-model-cfg");
        if (familySel && fileMeta.family) familySel.value = fileMeta.family;
        if (stepsInput && fileMeta.steps) stepsInput.value = fileMeta.steps;
        if (cfgInput && fileMeta.cfg !== undefined) cfgInput.value = fileMeta.cfg;
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
      stepsInput.value = "20";
      cfgInput.value = "7.0";
    } else if (fam === "flux") {
      stepsInput.value = "4";
      cfgInput.value = "1.0";
    } else if (fam === "qwen") {
      stepsInput.value = "20";
      cfgInput.value = "3.0";
    } else {
      stepsInput.value = "8";
      cfgInput.value = "2.5";
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
      this.toast("error", t("custom_model.error_name") || "Введите название модели");
      return;
    }

    let modelFile = modelSel ? modelSel.value : "";
    if (modelFile === "__custom__") {
      modelFile = modelInput ? modelInput.value.trim() : "";
    }
    if (!modelFile) {
      this.toast("error", t("custom_model.error_file") || "Выберите или укажите файл модели");
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

      this.toast("success", `✓ Модель «${name}» добавлена!`);
      this.closeCustomModelModal();

      await this.refreshConfig();
      if (data.model && data.model.id) {
        await this.selectModelFromModal(data.model.id);
      }
    } catch (err) {
      this.toast("error", err.message);
    }
  },

  async onHfSearchOrInspect() {
    const raw = document.getElementById("hf-search-input")?.value?.trim() || "";
    if (!raw) {
      this.toast("warning", "Введите поисковый запрос или ссылку на Hugging Face");
      return;
    }

    // Direct repo link or owner/repo format -> inspect directly
    const cleanRepo = raw
      .replace(/^https?:\/\/huggingface\.co\//i, "")
      .replace(/\/(tree|blob)\/.*$/i, "")
      .replace(/^\/+|\/+$/g, "");

    if (cleanRepo.includes("/") && !cleanRepo.includes(" ") && cleanRepo.split("/").length === 2) {
      await this.inspectHfRepo(cleanRepo);
      return;
    }

    const resultsEl = document.getElementById("hf-search-results");
    if (resultsEl) {
      resultsEl.style.display = "flex";
      resultsEl.innerHTML = `<div style="padding: 12px; font-size: 12px; color: var(--text-muted);">🔎 Поиск на Hugging Face...</div>`;
    }

    try {
      const data = await this.apiGet(`/api/hf/search?q=${encodeURIComponent(raw)}`);
      const results = data.results || [];
      if (!resultsEl) return;

      if (!results.length) {
        resultsEl.innerHTML = `<div style="padding: 12px; font-size: 12px; color: var(--text-dim);">Ничего не найдено. Попробуйте вставить прямую ссылку <code>author/repo</code>.</div>`;
        return;
      }

      resultsEl.innerHTML = "";
      for (const item of results) {
        const div = document.createElement("div");
        div.className = "hf-result-item";
        div.innerHTML = `
          <div style="min-width: 0;">
            <div style="font-size: 12.5px; font-weight: 600; color: var(--text-bright);">${escapeHtml(item.repo_id)}</div>
            <div style="font-size: 11px; color: var(--text-dim);">⬇ ${item.downloads || 0} скачиваний · ❤️ ${item.likes || 0}</div>
          </div>
          <button class="setting-btn" style="height: 26px; font-size: 11px;">Открыть файлы →</button>
        `;
        div.addEventListener("click", () => this.inspectHfRepo(item.repo_id));
        resultsEl.appendChild(div);
      }
    } catch (e) {
      if (resultsEl) {
        resultsEl.innerHTML = `<div style="padding: 12px; font-size: 12px; color: #f87171;">Ошибка: ${escapeHtml(e.message)}</div>`;
      }
    }
  },

  async inspectHfRepo(repoId) {
    const inspectorEl = document.getElementById("hf-repo-inspector");
    const filesListEl = document.getElementById("hf-repo-files-list") || document.getElementById("hf-files-checklist");
    const titleEl = document.getElementById("hf-inspected-repo-title") || document.getElementById("hf-repo-title");
    if (inspectorEl) inspectorEl.style.display = "block";
    if (titleEl) titleEl.textContent = `📦 Загрузка списка файлов ${repoId}...`;
    if (filesListEl) filesListEl.innerHTML = `<div style="font-size: 12px; color: var(--text-muted);">Запрос метаданных репозитория...</div>`;

    try {
      const info = await this.apiGet(`/api/hf/repo-tree?repo_id=${encodeURIComponent(repoId)}`);
      this.state.hfInspectedRepo = info;

      if (titleEl) titleEl.textContent = `📦 ${info.repo_id}`;
      const nameInp = document.getElementById("hf-import-name") || document.getElementById("hf-custom-name");
      const modelDisplayName = info.name || info.suggested_name || repoId.split("/").pop();
      if (nameInp) nameInp.value = `${modelDisplayName} (HF)`;
      const famSel = document.getElementById("hf-import-family") || document.getElementById("hf-custom-family");
      if (famSel && (info.family || info.suggested_family)) famSel.value = info.family || info.suggested_family;
      const stepsInp = document.getElementById("hf-import-steps") || document.getElementById("hf-custom-steps");
      if (stepsInp && (info.steps || info.suggested_steps)) stepsInp.value = info.steps || info.suggested_steps;
      const cfgInp = document.getElementById("hf-import-cfg") || document.getElementById("hf-custom-cfg");
      const suggestedCfg = info.cfg !== undefined ? info.cfg : info.suggested_cfg;
      if (cfgInp && suggestedCfg !== undefined) cfgInp.value = suggestedCfg;

      if (filesListEl) {
        filesListEl.innerHTML = "";
        const diffFiles = info.diffusion_files || [];
        if (!diffFiles.length) {
          filesListEl.innerHTML = `<div style="font-size: 12px; color: #f87171;">В репозитории не найдено файлов .gguf или .safetensors.</div>`;
        } else {
          diffFiles.forEach((f, idx) => {
            const isRec = f.quant === "Q8_0" || f.quant === "Q4_K_M" || idx === 0;
            const row = document.createElement("label");
            row.className = "hf-file-row";
            row.innerHTML = `
              <span style="display: flex; align-items: center; gap: 8px; min-width: 0;">
                <input type="checkbox" class="hf-diff-checkbox" data-path="${escapeHtml(f.path)}" data-quant="${escapeHtml(f.quant || '')}" data-size="${f.size_gb || 0}" ${isRec ? "checked" : ""}>
                <span style="overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">${escapeHtml(f.name)}</span>
              </span>
              <span style="font-size: 11px; color: var(--text-muted); white-space: nowrap;">${escapeHtml(f.quant || f.ext.toUpperCase())} · ${f.size_gb} GB</span>
            `;
            filesListEl.appendChild(row);
          });
        }
      }
    } catch (e) {
      if (titleEl) titleEl.textContent = `❌ Ошибка загрузки ${repoId}`;
      if (filesListEl) filesListEl.innerHTML = `<div style="font-size: 12px; color: #f87171;">${escapeHtml(e.message)}</div>`;
    }
  },

  async registerHfModel() {
    const info = this.state.hfInspectedRepo;
    if (!info || !info.repo_id) {
      this.toast("warning", "Сначала выберите репозиторий Hugging Face");
      return;
    }

    const checkedBoxes = Array.from(document.querySelectorAll(".hf-diff-checkbox:checked"));
    if (!checkedBoxes.length) {
      this.toast("warning", "Отметьте хотя бы один файл модели (.gguf / .safetensors)");
      return;
    }

    const stepsEl = document.getElementById("hf-import-steps") || document.getElementById("hf-custom-steps");
    const nameEl = document.getElementById("hf-import-name") || document.getElementById("hf-custom-name");
    const famEl = document.getElementById("hf-import-family") || document.getElementById("hf-custom-family");
    const cfgEl = document.getElementById("hf-import-cfg") || document.getElementById("hf-custom-cfg");

    const selectedFiles = checkedBoxes.map(cb => ({
      path: cb.dataset.path,
      quant: cb.dataset.quant,
      size_gb: parseFloat(cb.dataset.size || "0"),
      steps: parseInt(stepsEl?.value || "20", 10),
    }));

    const payload = {
      repo_id: info.repo_id,
      name: nameEl?.value?.trim() || `${info.name || info.suggested_name || 'Model'} (HF)`,
      family: famEl?.value || info.family || info.suggested_family || "flux",
      steps: parseInt(stepsEl?.value || "20", 10),
      cfg: parseFloat(cfgEl?.value || "3.5"),
      selected_files: selectedFiles,
      variants: selectedFiles,
      text_encoder_files: (info.text_encoder_files || []).slice(0, 2),
    };

    try {
      const res = await this.apiPost("/api/hf/register", payload);
      this.toast("success", `✓ Модель «${payload.name}» добавлена в Model Hub!`);
      this.closeCustomModelModal();
      await this.refreshConfig();
      if (res.model && res.model.id) {
        await this.selectModelFromModal(res.model.id);
      }
    } catch (e) {
      this.toast("error", e.message);
    }
  },

  async deleteCustomModel(modelId, event) {
    if (event) event.stopPropagation();
    const confirmed = confirm(`Удалить модель «${modelId}» из реестра Model Hub?`);
    if (!confirmed) return;

    try {
      const res = await fetch(`/api/custom-models/${encodeURIComponent(modelId)}?delete_files=true`, {
        method: "DELETE",
      });
      if (!res.ok) {
        const data = await res.json();
        throw new Error(data.detail || "Failed to delete");
      }
      this.toast("info", `Модель ${modelId} удалена из реестра`);
      await this.refreshConfig();
      if (this.state.isModelPickerOpen) {
        this.renderModelPickerModal();
      }
    } catch (err) {
      this.toast("error", err.message);
    }
  },

  openModelVisibilityModal(modelId, event) {
    if (event) event.stopPropagation();
    const modal = document.getElementById("model-visibility-modal");
    if (!modal) return;
    const model = (this.state.models || []).find(m => m.id === modelId);
    if (!model) return;

    const idInput = document.getElementById("model-vis-id");
    const descEl = document.getElementById("model-vis-title-desc");
    const chkGuests = document.getElementById("model-vis-hidden-guests");
    const chkHost = document.getElementById("model-vis-hidden-host");

    if (idInput) idInput.value = model.id;
    if (descEl) descEl.textContent = `Модель: «${model.name}» (${model.id})`;
    if (chkGuests) chkGuests.checked = !!model.hidden_for_guests;
    if (chkHost) chkHost.checked = !!model.hidden_from_host;

    modal.style.display = "flex";
  },

  closeModelVisibilityModal() {
    const modal = document.getElementById("model-visibility-modal");
    if (modal) modal.style.display = "none";
  },

  async saveModelVisibility() {
    const idInput = document.getElementById("model-vis-id");
    const chkGuests = document.getElementById("model-vis-hidden-guests");
    const chkHost = document.getElementById("model-vis-hidden-host");
    const modelId = idInput?.value;
    if (!modelId) return;

    const payload = {
      hidden_for_guests: !!chkGuests?.checked,
      hidden_from_host: !!chkHost?.checked,
    };

    try {
      await this.apiPost(`/api/models/${encodeURIComponent(modelId)}/visibility`, payload);
      this.toast("success", `✓ Настройки видимости для «${modelId}» сохранены`);
      this.closeModelVisibilityModal();
      await this.refreshConfig();
      if (this.state.isModelPickerOpen) {
        this.renderModelPickerModal();
      }
    } catch (err) {
      this.toast("error", err.message);
    }
  },

  // ============================================================
  // Dock Box Controls (Position, Transparency, Minimized)
  // ============================================================
  initDockDrag() {
    const wrap = document.getElementById("gemini-dock-wrap");
    const handle = document.getElementById("dock-drag-handle");
    if (!wrap || this._dockDragInit) return;
    this._dockDragInit = true;

    if (handle) {
      handle.addEventListener("dblclick", (e) => {
        e.stopPropagation();
        this.setDockPosition("center");
        this.toast("info", "Док сброшен в нижний центр");
      });
    }

    let isDragging = false;
    let startX = 0, startY = 0, origX = 0, origY = 0;

    wrap.addEventListener("pointerdown", (e) => {
      const isFromHandle = handle && (e.target === handle || handle.contains(e.target));
      if (!isFromHandle && !this.state.dockDragEnabled) return;
      if (e.target.closest("button") || e.target.closest("input") || e.target.closest("textarea") || e.target.closest("select")) {
        return;
      }
      this.state.dockDragEnabled = true;
      isDragging = true;
      startX = e.clientX;
      startY = e.clientY;
      const rect = wrap.getBoundingClientRect();
      origX = rect.left;
      origY = rect.top;
      try { wrap.setPointerCapture(e.pointerId); } catch (_) {}
      wrap.classList.add("dock-dragging");
      wrap.classList.add("dock-free-drag");
    });

    wrap.addEventListener("pointermove", (e) => {
      if (!isDragging || !this.state.dockDragEnabled) return;
      const dx = e.clientX - startX;
      const dy = e.clientY - startY;
      const maxX = window.innerWidth - wrap.offsetWidth;
      const maxY = window.innerHeight - wrap.offsetHeight;
      const nx = Math.max(8, Math.min(maxX - 8, origX + dx));
      const ny = Math.max(8, Math.min(maxY - 8, origY + dy));
      this.state.dockCustomX = Math.round(nx);
      this.state.dockCustomY = Math.round(ny);
      wrap.style.left = `${this.state.dockCustomX}px`;
      wrap.style.top = `${this.state.dockCustomY}px`;
      wrap.style.transform = "none";
      wrap.style.bottom = "auto";
      wrap.style.right = "auto";
    });

    const stopDrag = (e) => {
      if (!isDragging) return;
      isDragging = false;
      wrap.classList.remove("dock-dragging");
      try { wrap.releasePointerCapture(e.pointerId); } catch (_) {}
      this.saveDockSettings();
    };

    wrap.addEventListener("pointerup", stopDrag);
    wrap.addEventListener("pointercancel", stopDrag);
  },

  initDockSettings() {
    try {
      const raw = localStorage.getItem("lazysd_dock");
      const cfg = raw ? JSON.parse(raw) : {};
      if (cfg.position) this.state.dockPosition = cfg.position;
      if (cfg.opacity !== undefined) this.state.dockOpacity = cfg.opacity;
      if (cfg.blur !== undefined) this.state.dockBlur = cfg.blur;
      if (cfg.minimized !== undefined) this.state.dockMinimized = cfg.minimized;
    } catch (_) {}

    this.applyDockSettings();
    this.initDockDrag();
  },

  applyDockSettings() {
    const wrap = document.getElementById("gemini-dock-wrap");
    if (!wrap) return;

    wrap.classList.remove("dock-pos-left", "dock-pos-right", "dock-pos-bottom", "dock-pos-center", "dock-minimized", "dock-free-drag");

    if (this.state.dockDragEnabled && this.state.dockCustomX !== undefined && this.state.dockCustomY !== undefined) {
      wrap.classList.add("dock-free-drag");
      wrap.style.left = `${this.state.dockCustomX}px`;
      wrap.style.top = `${this.state.dockCustomY}px`;
      wrap.style.transform = "none";
      wrap.style.bottom = "auto";
      wrap.style.right = "auto";
    } else {
      wrap.style.left = "";
      wrap.style.top = "";
      wrap.style.transform = "";
      wrap.style.bottom = "";
      wrap.style.right = "";
      if (this.state.dockPosition && this.state.dockPosition !== "center") {
        wrap.classList.add(`dock-pos-${this.state.dockPosition}`);
      }
    }
    if (this.state.dockMinimized) {
      wrap.classList.add("dock-minimized");
    }

    const root = document.documentElement;
    root.style.setProperty("--dock-opacity", this.state.dockOpacity);
    root.style.setProperty("--dock-blur", `${this.state.dockBlur}px`);

    const posSelect = document.getElementById("setting-dock-pos");
    if (posSelect) posSelect.value = this.state.dockPosition || "center";

    const opSlider = document.getElementById("setting-dock-opacity");
    const opLabel = document.getElementById("setting-dock-opacity-val");
    if (opSlider) opSlider.value = this.state.dockOpacity;
    if (opLabel) opLabel.textContent = `${Math.round(this.state.dockOpacity * 100)}%`;

    const blurSlider = document.getElementById("setting-dock-blur");
    const blurLabel = document.getElementById("setting-dock-blur-val");
    if (blurSlider) blurSlider.value = this.state.dockBlur;
    if (blurLabel) blurLabel.textContent = `${this.state.dockBlur}px`;
  },

  saveDockSettings() {
    try {
      localStorage.setItem("lazysd_dock", JSON.stringify({
        position: this.state.dockPosition,
        opacity: this.state.dockOpacity,
        blur: this.state.dockBlur,
        minimized: this.state.dockMinimized,
      }));
    } catch (_) {}
  },

  setDockPosition(pos) {
    this.state.dockPosition = pos;
    this.applyDockSettings();
    this.saveDockSettings();
  },

  cycleDockPosition() {
    const order = ["center", "left", "right", "bottom"];
    const curIdx = order.indexOf(this.state.dockPosition || "center");
    const next = order[(curIdx + 1) % order.length];
    this.setDockPosition(next);
    this.toast("info", `Позиция коробки: ${next}`);
  },

  toggleDockMinimized() {
    this.state.dockMinimized = !this.state.dockMinimized;
    this.applyDockSettings();
    this.saveDockSettings();
  },

  setDockOpacity(val) {
    const num = parseFloat(val);
    if (isNaN(num)) return;
    this.state.dockOpacity = Math.max(0.2, Math.min(1.0, num));
    this.applyDockSettings();
    this.saveDockSettings();
  },

  setDockBlur(val) {
    const num = parseInt(val, 10);
    if (isNaN(num)) return;
    this.state.dockBlur = Math.max(0, Math.min(40, num));
    this.applyDockSettings();
    this.saveDockSettings();
  },

  // ============================================================
  // Compare Curtain / Slider (Шторка)
  // ============================================================
  toggleCompare() {
    const container = document.getElementById("compare-container");
    const mainImg = document.getElementById("preview-img");
    const btn = document.getElementById("btn-compare-toggle");
    if (!container || !mainImg) return;

    if (this.state.compareActive) {
      this.state.compareActive = false;
      container.style.display = "none";
      mainImg.style.display = "block";
      if (btn) btn.classList.remove("active");
      return;
    }

    const hasRef = Boolean(this.state.img2imgRef?.url);
    const fgSrc = this.state.currentPreviewUrl || (mainImg.src && !mainImg.src.endsWith("#") ? mainImg.src : null);
    // Prioritize imported document/reference as the baseline (left/bg)
    const bgSrc = hasRef ? this.state.img2imgRef.url : (this.state.previousPreviewUrl || null);

    if (!fgSrc) {
      this.toast("warning", "Нет изображения для сравнения. Сгенерируйте кадр.");
      return;
    }
    if (!bgSrc) {
      this.toast("warning", "Нет второго кадра для сравнения (загрузите документ/референс или сгенерируйте ещё один кадр).");
      return;
    }

    this.state.compareActive = true;
    container.style.display = "flex";
    mainImg.style.display = "none";
    if (btn) btn.classList.add("active");

    const bgImg = document.getElementById("compare-bg-img");
    const fgImg = document.getElementById("compare-fg-img");
    if (bgImg) bgImg.src = bgSrc;
    if (fgImg) fgImg.src = fgSrc;

    const leftLabel = document.getElementById("compare-label-left");
    const rightLabel = document.getElementById("compare-label-right");
    if (leftLabel) {
      leftLabel.textContent = hasRef ? "Документ / Исходник" : "Пред. кадр";
    }
    if (rightLabel) {
      rightLabel.textContent = "Результат";
    }

    this.updateCompareSlider(this.state.comparePercentage || 50);
  },

  updateCompareImages() {
    const bgImg = document.getElementById("compare-bg-img");
    const fgImg = document.getElementById("compare-fg-img");
    const leftLabel = document.getElementById("compare-label-left");
    const rightLabel = document.getElementById("compare-label-right");
    const hasRef = Boolean(this.state.img2imgRef?.url);

    if (fgImg && this.state.currentPreviewUrl) fgImg.src = this.state.currentPreviewUrl;
    if (bgImg) {
      if (hasRef) {
        bgImg.src = this.state.img2imgRef.url;
      } else if (this.state.previousPreviewUrl) {
        bgImg.src = this.state.previousPreviewUrl;
      }
    }
    if (leftLabel) {
      leftLabel.textContent = hasRef ? "Документ / Исходник" : "Пред. кадр";
    }
    if (rightLabel) {
      rightLabel.textContent = "Результат";
    }
  },

  updateCompareSlider(pct) {
    pct = Math.max(0, Math.min(100, pct));
    this.state.comparePercentage = pct;
    const clipEl = document.getElementById("compare-fg-clip");
    const barEl = document.getElementById("compare-slider-bar");
    if (clipEl) {
      clipEl.style.clipPath = `inset(0 0 0 ${pct}%)`;
    }
    if (barEl) {
      barEl.style.left = `${pct}%`;
    }
  },

  initCompareEvents() {
    const container = document.getElementById("compare-container");
    if (!container) return;

    let isDragging = false;
    const onMove = (e) => {
      if (!isDragging) return;
      const rect = container.getBoundingClientRect();
      if (!rect.width) return;
      const clientX = e.touches ? e.touches[0].clientX : e.clientX;
      const x = clientX - rect.left;
      const pct = (x / rect.width) * 100;
      this.updateCompareSlider(pct);
    };

    container.addEventListener("pointerdown", (e) => {
      isDragging = true;
      try { container.setPointerCapture(e.pointerId); } catch (_) {}
      const rect = container.getBoundingClientRect();
      if (rect.width) {
        this.updateCompareSlider(((e.clientX - rect.left) / rect.width) * 100);
      }
    });

    container.addEventListener("pointermove", onMove);
    const endDrag = (e) => {
      if (isDragging) {
        isDragging = false;
        try { container.releasePointerCapture(e.pointerId); } catch (_) {}
      }
    };
    container.addEventListener("pointerup", endDrag);
    container.addEventListener("pointercancel", endDrag);
  },

  // ============================================================
  // LoRA Management & Stacking
  // ============================================================
  async loadLoras() {
    try {
      const res = await this.apiGet("/api/loras");
      this.state.loraLibrary = Array.isArray(res) ? res : (res.loras || []);
      this.renderLoraUI();
    } catch (e) {
      console.warn("[lora] Failed to load loras:", e);
    }
  },

  renderLoraUI() {
    const count = this.state.activeLoras.length;
    const pillText = document.getElementById("dock-lora-pill-text");
    if (pillText) {
      pillText.textContent = `LoRA (${count})`;
    }
    const dockCount = document.getElementById("dock-lora-count");
    if (dockCount) {
      dockCount.textContent = `${count} активных`;
    }
    const inspectorCount = document.getElementById("inspector-lora-count");
    if (inspectorCount) {
      inspectorCount.textContent = String(count);
    }

    const dockList = document.getElementById("dock-lora-list");
    const dockEmpty = document.getElementById("dock-lora-empty");
    if (dockList && dockEmpty) {
      if (this.state.loraLibrary.length === 0) {
        dockEmpty.style.display = "block";
        dockList.innerHTML = "";
      } else {
        dockEmpty.style.display = "none";
        dockList.innerHTML = this.state.loraLibrary.map(item => {
          const activeItem = this.state.activeLoras.find(l => l.name === item.name);
          const isActive = !!activeItem;
          const strength = activeItem ? activeItem.strength : 1.0;
          const family = item.family ? `<span class="lora-meta-tag">${escapeHtml(item.family)}</span>` : "";
          const size = item.size_mb ? `<span class="lora-meta-tag">${item.size_mb} MB</span>` : "";
          return `
            <div class="lora-card-item ${isActive ? 'active' : ''}">
              <div class="lora-card-header">
                <div class="lora-title-group" onclick="App.toggleLora('${escapeHtml(item.name)}')" style="cursor: pointer;">
                  <input type="checkbox" ${isActive ? 'checked' : ''} onclick="event.stopPropagation(); App.toggleLora('${escapeHtml(item.name)}')">
                  <span class="lora-name-text" title="${escapeHtml(item.name)}">${escapeHtml(item.name)}</span>
                  ${family}
                  ${size}
                </div>
                <button type="button" class="dock-ctrl-btn" onclick="App.deleteLora('${escapeHtml(item.name)}')" title="Удалить файл LoRA">✕</button>
              </div>
              ${isActive ? `
                <div class="lora-slider-row">
                  <input type="range" min="-2.0" max="2.0" step="0.05" value="${strength}" oninput="App.setLoraStrength('${escapeHtml(item.name)}', this.value)">
                  <span class="lora-weight-val" id="lora-weight-val-${escapeHtml(item.name)}">${strength.toFixed(2)}</span>
                </div>
              ` : ''}
            </div>
          `;
        }).join("");
      }
    }

    const inspectorList = document.getElementById("lora-inspector-list");
    if (inspectorList) {
      if (this.state.activeLoras.length === 0) {
        inspectorList.innerHTML = '<div style="font-size: 11px; color: var(--text-dim); text-align: center; padding: 6px 0;">Нет подключенных LoRA</div>';
      } else {
        inspectorList.innerHTML = this.state.activeLoras.map(item => {
          return `
            <div class="lora-card-item active">
              <div class="lora-card-header">
                <div class="lora-title-group">
                  <span class="lora-name-text" title="${escapeHtml(item.name)}">${escapeHtml(item.name)}</span>
                </div>
                <button type="button" class="dock-ctrl-btn" onclick="App.toggleLora('${escapeHtml(item.name)}')" title="Отключить">✕</button>
              </div>
              <div class="lora-slider-row">
                <input type="range" min="-2.0" max="2.0" step="0.05" value="${item.strength}" oninput="App.setLoraStrength('${escapeHtml(item.name)}', this.value)">
                <span class="lora-weight-val" id="lora-weight-insp-${escapeHtml(item.name)}">${item.strength.toFixed(2)}</span>
              </div>
            </div>
          `;
        }).join("");
      }
    }

    const modalTotal = document.getElementById("lora-modal-total");
    if (modalTotal) modalTotal.textContent = String(this.state.loraLibrary.length);
    const hubCount = document.getElementById("models-hub-lora-count");
    if (hubCount) hubCount.textContent = String(this.state.loraLibrary.length);

    this.renderLoraHubGrid();
    this.renderLoraLibrary();
  },

  renderLoraHubGrid() {
    const grid = document.getElementById("lora-hub-grid");
    if (!grid) return;

    const allLoras = this.state.loraLibrary || [];
    const q = (document.getElementById("lora-hub-search")?.value || "").trim().toLowerCase();
    let list = allLoras;
    if (q) {
      list = list.filter(l =>
        (l.name || "").toLowerCase().includes(q) ||
        (l.label || "").toLowerCase().includes(q) ||
        (l.trigger || "").toLowerCase().includes(q) ||
        (l.family || "").toLowerCase().includes(q)
      );
    }

    if (list.length === 0) {
      const message = allLoras.length === 0
        ? "В папке LoRA пока нет файлов. Добавьте .safetensors или .gguf через кнопку «Добавить / Скачать LoRA»."
        : "LoRA не найдены по заданному фильтру.";
      grid.innerHTML = `<div style="grid-column: 1 / -1; padding: 40px; text-align: center; color: var(--text-dim);">${message}</div>`;
      return;
    }

    grid.innerHTML = list.map(item => {
      const activeItem = this.state.activeLoras.find(l => l.name === item.name);
      const isActive = !!activeItem;
      const family = item.family
        ? `<span class="model-badge" style="background: rgba(59, 130, 246, 0.15); color: #93c5fd; border: 1px solid rgba(59, 130, 246, 0.3);">${escapeHtml(item.family)}</span>`
        : "";
      const size = item.size_mb
        ? `<span class="model-badge" style="background: rgba(255, 255, 255, 0.05); color: var(--text-muted);">${item.size_mb} MB</span>`
        : "";
      return `
        <div class="model-card ${isActive ? 'active' : ''}" style="display: flex; flex-direction: column; justify-content: space-between; min-height: 140px;">
          <div>
            <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 6px;">
              <div style="font-weight: 600; font-size: 13.5px; color: var(--text); word-break: break-all;" title="${escapeHtml(item.name)}">${escapeHtml(item.name)}</div>
              <div style="display: flex; gap: 4px;">
                <button class="dock-ctrl-btn" onclick="App.editLoraMetadata('${escapeHtml(item.name)}', '${escapeHtml(item.family || '')}', '${escapeHtml(item.trigger || '')}')" title="Настроить архитектуру и триггеры">⚙️</button>
                <button class="dock-ctrl-btn" onclick="App.deleteLora('${escapeHtml(item.name)}')" title="Удалить">✕</button>
              </div>
            </div>
            <div style="display: flex; gap: 5px; flex-wrap: wrap; margin-bottom: 8px;">
              ${family}
              ${size}
            </div>
            ${item.trigger ? `<div style="font-size: 11px; color: #93c5fd; background: rgba(59, 130, 246, 0.1); border-radius: 4px; padding: 3px 6px; margin-bottom: 8px; cursor: pointer;" onclick="App.insertLoraTrigger('${escapeHtml(item.trigger)}')" title="Нажмите, чтобы вставить в промпт">💬 ${escapeHtml(item.trigger)}</div>` : ''}
          </div>
          <div style="display: flex; gap: 8px; align-items: center; margin-top: 8px;">
            <button class="card-action-btn ${isActive ? 'danger' : 'primary'}" style="flex: 1; height: 30px; font-size: 12px;" onclick="App.toggleLora('${escapeHtml(item.name)}')">
              ${isActive ? 'Отключить' : 'Подключить'}
            </button>
          </div>
        </div>
      `;
    }).join("");
  },

  renderLoraLibrary() {
    const listEl = document.getElementById("lora-modal-list-wrap");
    if (!listEl) return;

    const allLoras = this.state.loraLibrary || [];
    const q = (document.getElementById("lora-library-search")?.value || "").trim().toLowerCase();
    let list = allLoras;
    if (q) {
      list = list.filter(l =>
        (l.name || "").toLowerCase().includes(q) ||
        (l.label || "").toLowerCase().includes(q) ||
        (l.trigger || "").toLowerCase().includes(q)
      );
    }

    if (list.length === 0) {
      const message = allLoras.length === 0 ? "В библиотеке пока нет LoRA." : "Ничего не найдено.";
      listEl.innerHTML = `<div style="padding: 20px; text-align: center; color: var(--text-dim);">${message}</div>`;
      return;
    }

    listEl.innerHTML = list.map(item => {
      const isActive = this.state.activeLoras.some(l => l.name === item.name);
      const meta = [
        item.size_mb ? `${item.size_mb} MB` : "",
        item.family || "",
        item.trigger ? `💬 ${item.trigger}` : "",
      ].filter(Boolean).join(" · ");
      return `
        <div class="lora-card-item ${isActive ? 'active' : ''}" style="margin-bottom: 6px;">
          <div class="lora-card-header">
            <div class="lora-title-group" onclick="App.toggleLora('${escapeHtml(item.name)}')" style="cursor: pointer; min-width: 0;">
              <input type="checkbox" ${isActive ? 'checked' : ''} onclick="event.stopPropagation(); App.toggleLora('${escapeHtml(item.name)}')">
              <span class="lora-name-text" title="${escapeHtml(item.name)}">${escapeHtml(item.name)}</span>
            </div>
            <div style="display: flex; gap: 4px; align-items: center; flex-shrink: 0;">
              <span style="font-size: 10px; color: var(--text-dim);">${escapeHtml(meta)}</span>
              <button type="button" class="dock-ctrl-btn" onclick="App.deleteLora('${escapeHtml(item.name)}')" title="Удалить">✕</button>
            </div>
          </div>
        </div>
      `;
    }).join("");
  },

  toggleLora(name) {
    const idx = this.state.activeLoras.findIndex(l => l.name === name);
    if (idx >= 0) {
      this.state.activeLoras.splice(idx, 1);
    } else {
      const item = this.state.loraLibrary.find(l => l.name === name);
      this.state.activeLoras.push({
        name: name,
        strength: 1.0,
        info: item || {},
      });
    }
    this.renderLoraUI();
  },

  setLoraStrength(name, val) {
    const num = parseFloat(val);
    if (isNaN(num)) return;
    const item = this.state.activeLoras.find(l => l.name === name);
    if (item) {
      item.strength = Math.round(num * 100) / 100;
      const el1 = document.getElementById(`lora-weight-val-${name}`);
      const el2 = document.getElementById(`lora-weight-insp-${name}`);
      if (el1) el1.textContent = item.strength.toFixed(2);
      if (el2) el2.textContent = item.strength.toFixed(2);
    }
  },

  openLoraModal() {
    this.openLoraManageModal("hub");
  },

  closeLoraModal() {
    const modal = document.getElementById("lora-manage-modal");
    if (modal) modal.style.display = "none";
  },

  async onLoraFileSelected(input) {
    if (!input.files || !input.files[0]) return;
    const file = input.files[0];
    const fd = new FormData();
    fd.append("file", file);

    this.toast("info", `Загрузка LoRA «${file.name}»...`);
    try {
      const res = await fetch("/api/loras/upload", {
        method: "POST",
        body: fd,
      });
      if (!res.ok) {
        const data = await res.json();
        throw new Error(data.detail || "Upload failed");
      }
      this.toast("success", `✓ LoRA «${file.name}» загружена!`);
      input.value = "";
      await this.loadLoras();
    } catch (e) {
      this.toast("error", e.message);
    }
  },

  async importLocalLora() {
    const input = document.getElementById("lora-local-path-input");
    const path = input ? input.value.trim() : "";
    if (!path) {
      this.toast("warning", "Укажите путь к файлу LoRA на диске");
      return;
    }
    try {
      this.toast("info", "Импорт файла...");
      const res = await this.apiPost("/api/loras/import", { path: path });
      this.toast("success", `✓ LoRA импортирована: ${res.lora?.name || "OK"}`);
      if (input) input.value = "";
      await this.loadLoras();
    } catch (e) {
      this.toast("error", e.message);
    }
  },

  async downloadHfLora() {
    const repoInput = document.getElementById("lora-hf-repo-input");
    const fileSelect = document.getElementById("lora-hf-file-select");
    const renameInput = document.getElementById("lora-hf-rename-input");
    const familySelect = document.getElementById("lora-hf-family-select");
    const triggerInput = document.getElementById("lora-hf-trigger-input");
    const repo = repoInput ? repoInput.value.trim() : "";
    const filename = fileSelect ? fileSelect.value : "";
    if (!repo) {
      this.toast("warning", "Укажите HuggingFace repo (author/repo)");
      return;
    }

    try {
      this.toast("info", `Скачивание LoRA из ${repo}...`);
      const res = await this.apiPost("/api/loras/download", {
        repo_id: repo,
        filename: filename || null,
        rename: renameInput?.value.trim() || null,
        family: familySelect?.value && familySelect.value !== "any" ? familySelect.value : null,
        trigger: triggerInput?.value.trim() || null,
      });
      this.toast("success", `✓ LoRA успешно скачана: ${res.lora?.name}`);
      if (renameInput) renameInput.value = "";
      if (triggerInput) triggerInput.value = "";
      const inspector = document.getElementById("lora-hf-inspector");
      if (inspector) inspector.style.display = "none";
      await this.loadLoras();
    } catch (e) {
      this.toast("error", e.message);
    }
  },

  async editLoraMetadata(name, family, trigger) {
    const nextFamily = prompt(`Архитектура LoRA «${name}» (flux / qwen / sdxl / z_image / any):`, family || "");
    if (nextFamily === null) return;
    const nextTrigger = prompt("Триггер-слова (через запятую):", trigger || "");
    if (nextTrigger === null) return;

    try {
      await this.apiPatch(`/api/loras/${encodeURIComponent(name)}/metadata`, {
        family: nextFamily.trim() || null,
        trigger: nextTrigger.trim() || null,
      });
      this.toast("success", "✓ Метаданные LoRA обновлены");
      await this.loadLoras();
    } catch (e) {
      this.toast("error", e.message);
    }
  },

  insertLoraTrigger(trigger) {
    const promptInput = document.getElementById("prompt-input");
    if (!promptInput || !trigger) return;
    const current = promptInput.value.trim();
    promptInput.value = current ? `${current}, ${trigger}` : trigger;
    promptInput.dispatchEvent(new Event("input", { bubbles: true }));
    this.toast("info", `Триггер добавлен: ${trigger}`);
  },

  async deleteLora(name) {
    if (!confirm(`Удалить LoRA «${name}» с диска?`)) return;
    try {
      await fetch(`/api/loras/${encodeURIComponent(name)}`, { method: "DELETE" });
      this.toast("info", `LoRA «${name}» удалена`);
      const idx = this.state.activeLoras.findIndex(l => l.name === name);
      if (idx >= 0) this.state.activeLoras.splice(idx, 1);
      await this.loadLoras();
    } catch (e) {
      this.toast("error", e.message);
    }
  },

  // ============================================================
  // Remote Sharing (Gradio-like without Gradio)
  // ============================================================
  async loadShareStatus() {
    try {
      const data = await this.apiGet("/api/share/status");
      this.state.shareStatus = data;
      this.renderShareUI();
    } catch (_) {}
  },

  renderShareUI() {
    const status = this.state.shareStatus;
    if (!status) return;

    const badge = document.getElementById("setting-share-status-badge");
    const urlWrap = document.getElementById("setting-share-url-wrap");
    const urlInput = document.getElementById("setting-share-url");
    const toggleBtn = document.getElementById("btn-share-toggle");
    const copyBtn = document.getElementById("btn-share-copy");
    const tokenInput = document.getElementById("setting-share-token");
    const autoCheck = document.getElementById("setting-share-autostart");

    if (tokenInput && status.token) tokenInput.value = status.token;
    if (autoCheck) autoCheck.checked = !!status.auto_start;

    if (status.running && status.public_url) {
      if (badge) {
        badge.className = "state-badge installed";
        badge.textContent = `✓ Активен (${status.provider})`;
      }
      if (urlWrap) urlWrap.style.display = "block";
      if (urlInput) urlInput.value = status.share_url || status.public_url;
      if (toggleBtn) toggleBtn.textContent = "Остановить туннель";
      if (copyBtn) copyBtn.style.display = "inline-block";
    } else {
      if (badge) {
        badge.className = "state-badge missing";
        badge.textContent = "Остановлен";
      }
      if (urlWrap) urlWrap.style.display = "none";
      if (toggleBtn) toggleBtn.textContent = "Запустить туннель";
      if (copyBtn) copyBtn.style.display = "none";
    }
  },

  async toggleShareTunnel() {
    const isRunning = this.state.shareStatus?.running;
    const btn = document.getElementById("btn-share-toggle");
    try {
      if (isRunning) {
        if (btn) btn.textContent = "Остановка...";
        this.toast("info", "Остановка туннеля...");
        await this.apiPost("/api/share/stop", {});
        this.toast("info", "Туннель остановлен");
        await this.loadShareStatus();
      } else {
        if (btn) btn.textContent = "Запуск Cloudflare...";
        this.toast("info", "Запуск Cloudflare quick tunnel...");
        await this.apiPost("/api/share/start", { provider: "cloudflared" });

        // Poll for public trycloudflare URL (takes ~1-3s to establish)
        for (let i = 0; i < 8; i++) {
          await new Promise((r) => setTimeout(r, 1200));
          await this.loadShareStatus();
          const st = this.state.shareStatus;
          if (st && st.running && st.public_url) {
            this.toast("success", "✓ Туннель активен: " + st.public_url);
            return;
          }
          if (st && st.error) {
            throw new Error(st.error);
          }
        }
        this.toast("info", "Туннель запущен, ссылка формируется...");
      }
    } catch (e) {
      this.toast("error", e.message);
      await this.loadShareStatus();
    }
  },

  copyShareUrl() {
    const input = document.getElementById("setting-share-url");
    if (!input || !input.value) return;
    navigator.clipboard.writeText(input.value);
    this.toast("success", "✓ Публичная ссылка скопирована в буфер обмена!");
  },

  generateShareToken() {
    const token = Array.from(crypto.getRandomValues(new Uint8Array(16)))
      .map(b => b.toString(16).padStart(2, "0")).join("");
    const input = document.getElementById("setting-share-token");
    if (input) input.value = token;
  },

  async saveShareToken() {
    const input = document.getElementById("setting-share-token");
    const val = input ? input.value.trim() : "";
    try {
      await this.apiPost("/api/share/token", { token: val || null });
      this.toast("success", "✓ Токен гостя сохранён");
      await this.loadShareStatus();
    } catch (e) {
      this.toast("error", e.message);
    }
  },

  async saveShareAutoStart(checked) {
    try {
      await this.apiPost("/api/settings", { share_auto_start: checked });
      this.toast("info", `Автозапуск туннеля: ${checked ? "ВКЛ" : "ВЫКЛ"}`);
    } catch (e) {
      this.toast("error", e.message);
    }
  },

  // ============================================================
  // Extras & Engines (OrbitQuant & Nunchaku)
  // ============================================================
  async refreshExtrasStatus(showToast = false) {
    try {
      const data = await this.apiGet("/api/extras/status?refresh=true");
      this.state.extrasStatus = data;
      this.renderExtrasUI();
      if (showToast) {
        const isReady = data.env?.ready;
        const msg = isReady
          ? `✓ Окружение Extras готово: Python ${data.env?.python || ''}, PyTorch ${data.env?.torch || ''}`
          : `Окружение Extras: ${data.env?.summary || 'требуется установка компонентов'}`;
        this.toast(isReady ? "success" : "info", msg);
      }
    } catch (e) {
      if (showToast) {
        this.toast("error", "Не удалось опросить статус Extras: " + e.message);
      }
    }
  },

  renderExtrasUI() {
    const data = this.state.extrasStatus;
    if (!data) return;

    const nunchakuBadge = document.getElementById("setting-nunchaku-status");
    const orbitBadge = document.getElementById("setting-orbitquant-status");
    const envDesc = document.getElementById("setting-extras-env-desc");
    const customPyInput = document.getElementById("setting-extras-custom-py");

    if (customPyInput && data.env?.python_path) {
      if (!customPyInput.value && !data.env.managed) {
        customPyInput.value = data.env.python_path;
      }
    }

    if (envDesc && data.env) {
      const e = data.env;
      if (e.ready) {
        envDesc.innerHTML = `<span style="color: #4ade80; font-weight: 600;">✓ Готово к работе:</span> Python ${e.python || '3.12'} · Torch ${e.torch || 'OK'} · CUDA ${e.cuda || 'OK'} (${e.device || 'GPU'}) · Diffusers ${e.diffusers || 'OK'}`;
      } else if (e.python_path || e.venv_exists) {
        envDesc.innerHTML = `<span style="color: #fbbf24; font-weight: 600;">⚠️ Настраивается:</span> ${e.error || e.summary || 'Окружение создано, ожидается установка PyTorch и Diffusers'}`;
      } else {
        envDesc.innerHTML = `<span style="color: var(--text-dim);">ℹ️ ${e.summary || 'Изолированное окружение (envs/extras) пока не создано. Создаётся автоматически при установке Nunchaku или OrbitQuant.'}</span>`;
      }
    }

    if (nunchakuBadge) {
      if (data.nunchaku_ready) {
        nunchakuBadge.innerHTML = `<span class="state-badge installed">✓ Установлен (${data.env?.nunchaku || 'ready'})</span>`;
      } else {
        nunchakuBadge.innerHTML = '<span class="state-badge missing">Не установлен</span>';
      }
    }

    if (orbitBadge) {
      if (data.orbitquant_ready) {
        orbitBadge.innerHTML = `<span class="state-badge installed">✓ Установлен (${data.env?.orbitquant || 'ready'})</span>`;
      } else {
        orbitBadge.innerHTML = '<span class="state-badge missing">Не установлен</span>';
      }
    }
  },

  async saveExtrasPythonPath() {
    const input = document.getElementById("setting-extras-custom-py");
    const val = input ? input.value.trim() : "";
    try {
      this.toast("info", "Сохранение пути к Python...");
      const res = await this.apiPost("/api/extras/python", { path: val });
      this.toast("success", "✓ Путь к Extras Python обновлён");
      await this.refreshExtrasStatus(false);
    } catch (e) {
      this.toast("error", "Ошибка: " + e.message);
    }
  },

  async installExtrasPackage(target) {
    const btn = document.getElementById(`btn-install-${target}`);
    const origText = btn ? btn.textContent : `Установить ${target}`;
    if (btn) {
      btn.disabled = true;
      btn.textContent = "Установка...";
    }
    const labelTarget = target === "runtime" ? "базового окружения" : target;
    this.toast("info", `Запущена установка ${labelTarget}...`);

    try {
      const res = await this.apiPost("/api/extras/install", { component: target, target: target });
      const jobId = res.job?.id || res.id;
      if (!jobId) throw new Error(res.error || "Не удалось получить ID задачи установки");

      const poll = setInterval(async () => {
        try {
          const job = await this.apiGet(`/api/extras/jobs/${jobId}`);
          const state = job.state || job.status;
          const label = job.step_label || job.stage || 'Установка';
          if (btn) btn.textContent = `${label}...`;

          if (state === "done" || state === "succeeded") {
            clearInterval(poll);
            if (btn) {
              btn.disabled = false;
              btn.textContent = origText;
            }
            this.toast("success", `✓ Установка ${labelTarget} завершена!`);
            await this.refreshExtrasStatus(false);
          } else if (state === "error" || state === "failed" || state === "cancelled") {
            clearInterval(poll);
            if (btn) {
              btn.disabled = false;
              btn.textContent = origText;
            }
            this.toast("error", `Ошибка установки ${target}: ${job.error || "failed"}`);
          }
        } catch (_) {
          clearInterval(poll);
        }
      }, 2000);
    } catch (e) {
      if (btn) {
        btn.disabled = false;
        btn.textContent = origText;
      }
      this.toast("error", e.message);
    }
  },
};

// Global handlers for inline HTML onclick attributes
window.loadHistory = () => App.loadHistory();
window.closeHistoryModal = () => App.closeHistoryModal();
window.clearImg2ImgRef = () => App.clearImg2ImgRef();
window.updateStrengthLabel = (val) => App.updateStrengthLabel(val);
window.sendCurrentPreviewToImg2Img = () => App.sendCurrentPreviewToImg2Img();



// ============================================================
// --- INTEGRATED ZOOM & PAN ENGINE ---
// ============================================================
(function() {
/* ============================================================
   QuantFunc WebUI — Viewport Zoom & Pan Engine
   Smooth cursor-centered zoom, pan, floating HUD, and scrollbars.
   Shared across Generation, Halftone Studio GT, and Manga Studio.
   ============================================================ */


const viewports = {};
let viewportsInitialized = false;
let spacePressed = false;

function initAllViewportZoomPan(appRef = null) {
  if (viewportsInitialized) return;
  viewportsInitialized = true;

  initViewportZoomPan("generate", {
    areaSelector: "#tab-generate .preview-area",
    wrapId: "preview-image-wrap",
    hudId: "gen-zoom-hud",
    labelId: "gen-zoom-label",
    scrollXId: "gen-scrollbar-x",
    thumbXId: "gen-scrollbar-thumb-x",
    scrollYId: "gen-scrollbar-y",
    thumbYId: "gen-scrollbar-thumb-y",
  }, appRef);

  initViewportZoomPan("halftone", {
    areaId: "ht-dropzone",
    wrapId: "ht-preview-wrap",
    hudId: "ht-zoom-hud",
    labelId: "ht-zoom-label",
    scrollXId: "ht-scrollbar-x",
    thumbXId: "ht-scrollbar-thumb-x",
    scrollYId: "ht-scrollbar-y",
    thumbYId: "ht-scrollbar-thumb-y",
  }, appRef);

  initViewportZoomPan("manga", {
    areaId: "manga-dropzone",
    wrapId: "manga-preview-wrap",
    hudId: "manga-zoom-hud",
    labelId: "manga-zoom-label",
    scrollXId: "manga-scrollbar-x",
    thumbXId: "manga-scrollbar-thumb-x",
    scrollYId: "manga-scrollbar-y",
    thumbYId: "manga-scrollbar-thumb-y",
  }, appRef);

  window.addEventListener("keydown", (e) => {
    if (e.code === "Space" && !["INPUT", "TEXTAREA", "SELECT"].includes(document.activeElement?.tagName)) {
      spacePressed = true;
    }
  });

  window.addEventListener("keyup", (e) => {
    if (e.code === "Space") {
      spacePressed = false;
    }
  });
}

function initViewportZoomPan(mode, cfg, appRef = null) {
  const area = cfg.areaId ? document.getElementById(cfg.areaId) : document.querySelector(cfg.areaSelector);
  const wrap = document.getElementById(cfg.wrapId);
  if (!area || !wrap) return;

  const vp = {
    mode,
    area,
    wrap,
    hud: document.getElementById(cfg.hudId),
    label: document.getElementById(cfg.labelId),
    scrollX: document.getElementById(cfg.scrollXId),
    thumbX: document.getElementById(cfg.thumbXId),
    scrollY: document.getElementById(cfg.scrollYId),
    thumbY: document.getElementById(cfg.thumbYId),
    zoom: 1.0,
    minZoom: 0.5,
    maxZoom: 8.0,
    panX: 0,
    panY: 0,
    hideTimer: null,
  };

  viewports[mode] = vp;

  // 1. Mouse Wheel Zoom centered around cursor
  area.addEventListener("wheel", (e) => {
    e.preventDefault();
    const factor = e.deltaY < 0 ? 1.16 : 1 / 1.16;
    const nextZoom = Math.max(vp.minZoom, Math.min(vp.maxZoom, vp.zoom * factor));
    if (Math.abs(nextZoom - vp.zoom) < 0.001) {
      showViewportHudAndScrollbars(vp);
      return;
    }

    const areaRect = area.getBoundingClientRect();
    const cx = e.clientX - (areaRect.left + areaRect.width * 0.5);
    const cy = e.clientY - (areaRect.top + areaRect.height * 0.5);
    const ratio = nextZoom / vp.zoom;

    vp.panX = cx - (cx - vp.panX) * ratio;
    vp.panY = cy - (cy - vp.panY) * ratio;
    vp.zoom = nextZoom;

    applyViewportTransform(vp, true);
  }, { passive: false });

  // 2. Pointer Pan & Zoom-Tool Click/Drag
  let isPanning = false;
  let startX = 0;
  let startY = 0;
  let startPanX = 0;
  let startPanY = 0;
  let movedDist = 0;
  let isZoomToolClick = false;

  area.addEventListener("pointerdown", (e) => {
    if (
      e.target.closest(".viewport-zoom-hud") ||
      e.target.closest(".viewport-scrollbar-x") ||
      e.target.closest(".viewport-scrollbar-y") ||
      e.target.closest(".preview-toolbar") ||
      e.target.closest(".compare-slider-bar")
    ) {
      return;
    }

    const htTool = (appRef?.state?.halftone?.tool || window.App?.state?.halftone?.tool) || "view";
    const isMiddle = e.button === 1;
    const isRight = e.button === 2;
    const isLeft = e.button === 0;

    if (mode === "halftone" && htTool === "zoom" && (isLeft || isRight)) {
      isPanning = true;
      isZoomToolClick = true;
    } else if (isMiddle || (isLeft && spacePressed)) {
      isPanning = true;
      isZoomToolClick = false;
    } else if (isLeft && mode === "halftone" && htTool === "view") {
      isPanning = true;
      isZoomToolClick = false;
    } else if (isLeft && (mode === "manga" || mode === "generate")) {
      isPanning = true;
      isZoomToolClick = false;
    } else {
      return;
    }

    e.preventDefault();
    startX = e.clientX;
    startY = e.clientY;
    startPanX = vp.panX;
    startPanY = vp.panY;
    movedDist = 0;

    area.classList.add("panning-active");
    try { area.setPointerCapture(e.pointerId); } catch (_) {}
    showViewportHudAndScrollbars(vp);
  });

  area.addEventListener("pointermove", (e) => {
    if (!isPanning) return;
    const dx = e.clientX - startX;
    const dy = e.clientY - startY;
    movedDist = Math.hypot(dx, dy);

    vp.panX = startPanX + dx;
    vp.panY = startPanY + dy;
    applyViewportTransform(vp, true);
  });

  const endPan = (e) => {
    if (!isPanning) return;
    isPanning = false;
    area.classList.remove("panning-active");
    try { area.releasePointerCapture(e.pointerId); } catch (_) {}

    if (isZoomToolClick && movedDist < 6) {
      const zoomOut = e.altKey || e.button === 2;
      const factor = zoomOut ? 1 / 1.35 : 1.35;
      const nextZoom = Math.max(vp.minZoom, Math.min(vp.maxZoom, vp.zoom * factor));
      const areaRect = area.getBoundingClientRect();
      const cx = e.clientX - (areaRect.left + areaRect.width * 0.5);
      const cy = e.clientY - (areaRect.top + areaRect.height * 0.5);
      const ratio = nextZoom / vp.zoom;

      vp.panX = cx - (cx - vp.panX) * ratio;
      vp.panY = cy - (cy - vp.panY) * ratio;
      vp.zoom = nextZoom;

      applyViewportTransform(vp, true);
    }
    isZoomToolClick = false;
  };

  area.addEventListener("pointerup", endPan);
  area.addEventListener("pointercancel", endPan);
  area.addEventListener("contextmenu", (e) => {
    const htTool = (appRef?.state?.halftone?.tool || window.App?.state?.halftone?.tool);
    if (mode === "halftone" && htTool === "zoom") {
      e.preventDefault();
    }
  });

  // Double-click resets zoom to 100% Fit
  area.addEventListener("dblclick", (e) => {
    if (
      e.target.closest(".viewport-zoom-hud") ||
      e.target.closest(".preview-toolbar")
    ) {
      return;
    }
    const htTool = (appRef?.state?.halftone?.tool || window.App?.state?.halftone?.tool);
    if (mode !== "halftone" || htTool === "view") {
      resetViewportZoom(mode);
    }
  });

  // 3. Interactive Draggable Floating Scrollbars
  bindScrollbarDrag(vp, "x");
  bindScrollbarDrag(vp, "y");
}

function bindScrollbarDrag(vp, axis) {
  const track = axis === "x" ? vp.scrollX : vp.scrollY;
  const thumb = axis === "x" ? vp.thumbX : vp.thumbY;
  if (!track || !thumb) return;

  let dragging = false;
  let startPos = 0;
  let startPan = 0;

  thumb.addEventListener("pointerdown", (e) => {
    e.stopPropagation();
    e.preventDefault();
    dragging = true;
    startPos = axis === "x" ? e.clientX : e.clientY;
    startPan = axis === "x" ? vp.panX : vp.panY;

    try { thumb.setPointerCapture(e.pointerId); } catch (_) {}
    showViewportHudAndScrollbars(vp);
  });

  thumb.addEventListener("pointermove", (e) => {
    if (!dragging) return;
    const delta = (axis === "x" ? e.clientX : e.clientY) - startPos;
    const bounds = getViewportPanBounds(vp);
    const maxPan = axis === "x" ? bounds.maxPanX : bounds.maxPanY;
    const trackLen = axis === "x" ? track.clientWidth : track.clientHeight;
    const thumbLen = axis === "x" ? thumb.offsetWidth : thumb.offsetHeight;
    const avail = Math.max(1, trackLen - thumbLen);

    if (maxPan > 0) {
      const panDelta = -(delta / avail) * (maxPan * 2);
      if (axis === "x") vp.panX = startPan + panDelta;
      else vp.panY = startPan + panDelta;
      applyViewportTransform(vp, true);
    }
  });

  const stopThumb = (e) => {
    if (!dragging) return;
    dragging = false;
    try { thumb.releasePointerCapture(e.pointerId); } catch (_) {}
  };

  thumb.addEventListener("pointerup", stopThumb);
  thumb.addEventListener("pointercancel", stopThumb);
}

function getViewportPanBounds(vp) {
  const vw = vp.area.clientWidth || 1;
  const vh = vp.area.clientHeight || 1;
  const baseW = vp.wrap.offsetWidth || vw * 0.8;
  const baseH = vp.wrap.offsetHeight || vh * 0.8;
  const cw = baseW * vp.zoom;
  const ch = baseH * vp.zoom;

  const marginX = Math.min(vw, baseW) * 0.28;
  const marginY = Math.min(vh, baseH) * 0.28;
  const maxPanX = vp.zoom <= 1.0
    ? marginX * 0.5
    : Math.max(marginX * 0.5, (cw - vw) * 0.5 + marginX);
  const maxPanY = vp.zoom <= 1.0
    ? marginY * 0.5
    : Math.max(marginY * 0.5, (ch - vh) * 0.5 + marginY);

  return { vw, vh, baseW, baseH, cw, ch, maxPanX, maxPanY };
}

function applyViewportTransform(vp, showHud = true) {
  if (!vp || !vp.wrap) return;
  const bounds = getViewportPanBounds(vp);

  vp.panX = Math.max(-bounds.maxPanX, Math.min(bounds.maxPanX, vp.panX));
  vp.panY = Math.max(-bounds.maxPanY, Math.min(bounds.maxPanY, vp.panY));

  if (Math.abs(vp.zoom - 1.0) < 0.01 && Math.hypot(vp.panX, vp.panY) < 4) {
    vp.zoom = 1.0;
    vp.panX = 0;
    vp.panY = 0;
  }

  vp.wrap.style.transform = `translate3d(${vp.panX.toFixed(1)}px, ${vp.panY.toFixed(1)}px, 0) scale(${vp.zoom.toFixed(4)})`;

  if (vp.label) {
    vp.label.textContent = `🔍 ${Math.round(vp.zoom * 100)}%`;
  }

  const showScroll = vp.zoom > 1.03;
  if (vp.scrollX && vp.thumbX) {
    if (showScroll && bounds.maxPanX > 1) {
      const trackW = vp.scrollX.clientWidth || 200;
      const ratioW = Math.max(0.15, Math.min(0.85, bounds.vw / Math.max(bounds.cw, bounds.vw)));
      const thumbW = Math.max(28, Math.round(trackW * ratioW));
      const normX = Math.max(0, Math.min(1, (bounds.maxPanX - vp.panX) / (2 * bounds.maxPanX)));
      const leftPx = Math.round(normX * (trackW - thumbW));
      vp.thumbX.style.width = `${thumbW}px`;
      vp.thumbX.style.left = `${leftPx}px`;
    }
  }

  if (vp.scrollY && vp.thumbY) {
    if (showScroll && bounds.maxPanY > 1) {
      const trackH = vp.scrollY.clientHeight || 200;
      const ratioH = Math.max(0.15, Math.min(0.85, bounds.vh / Math.max(bounds.ch, bounds.vh)));
      const thumbH = Math.max(28, Math.round(trackH * ratioH));
      const normY = Math.max(0, Math.min(1, (bounds.maxPanY - vp.panY) / (2 * bounds.maxPanY)));
      const topPx = Math.round(normY * (trackH - thumbH));
      vp.thumbY.style.height = `${thumbH}px`;
      vp.thumbY.style.top = `${topPx}px`;
    }
  }

  if (showHud) {
    showViewportHudAndScrollbars(vp);
  }
}

function showViewportHudAndScrollbars(vp) {
  if (!vp) return;
  if (vp.hud) vp.hud.classList.add("visible");

  const showScroll = vp.zoom > 1.03;
  if (vp.scrollX) vp.scrollX.classList.toggle("visible", showScroll);
  if (vp.scrollY) vp.scrollY.classList.toggle("visible", showScroll);

  if (vp.hideTimer) clearTimeout(vp.hideTimer);
  vp.hideTimer = setTimeout(() => {
    if (vp.hud) vp.hud.classList.remove("visible");
    if (vp.scrollX) vp.scrollX.classList.remove("visible");
    if (vp.scrollY) vp.scrollY.classList.remove("visible");
  }, 1450);
}

function initViewportHudOpacity() {
  const saved = localStorage.getItem("lazysd_zoom_hud_opacity");
  if (saved) {
    document.documentElement.style.setProperty("--zoom-hud-opacity", saved);
  }
}

function cycleViewportHudOpacity() {
  const cur = parseFloat(getComputedStyle(document.documentElement).getPropertyValue("--zoom-hud-opacity")) || 0.92;
  let next = 0.92;
  if (cur > 0.85) next = 0.65;
  else if (cur > 0.55) next = 0.40;
  else if (cur > 0.30) next = 0.18;
  else next = 0.95;

  document.documentElement.style.setProperty("--zoom-hud-opacity", String(next));
  try {
    localStorage.setItem("lazysd_zoom_hud_opacity", String(next));
  } catch (_) {}
  toast("info", `Прозрачность панели зума: ${Math.round(next * 100)}%`);
}

function stepViewportZoom(mode, dir) {
  initAllViewportZoomPan();
  const vp = viewports[mode];
  if (!vp) return;
  const factor = dir > 0 ? 1.25 : 1 / 1.25;
  vp.zoom = Math.max(vp.minZoom, Math.min(vp.maxZoom, vp.zoom * factor));
  applyViewportTransform(vp, true);
}

function setViewportZoom(mode, targetZoom, showHud = true) {
  initAllViewportZoomPan();
  const vp = viewports[mode];
  if (!vp) return;
  vp.zoom = Math.max(vp.minZoom, Math.min(vp.maxZoom, Number(targetZoom) || 1.0));
  applyViewportTransform(vp, showHud);
}

function resetViewportZoom(mode) {
  initAllViewportZoomPan();
  const vp = viewports[mode];
  if (!vp) return;
  vp.zoom = 1.0;
  vp.panX = 0;
  vp.panY = 0;
  applyViewportTransform(vp, true);
}


  Object.assign(App, {
    initAllViewportZoomPan,
    initViewportZoomPan,
    getViewportPanBounds,
    applyViewportTransform,
    showViewportHudAndScrollbars,
    initViewportHudOpacity,
    cycleViewportHudOpacity,
    stepViewportZoom,
    setViewportZoom,
    resetViewportZoom,
  });
})();

// ============================================================
// --- INTEGRATED HALFTONE STUDIO GT ENGINE (WebGL2 + 100% Full-Res) ---
// ============================================================
(function() {
  const apiPost = (url, body) => App.apiPost(url, body);
  const apiGet = (url) => App.apiGet(url);
  const toast = (type, msg) => App.toast(type, msg);

/* ============================================================
   QuantFunc WebUI — Halftone Studio GT Engine
   Multi-layer screentone system, polygon lasso/brush masking,
   and Real-Time WebGL2 GPU Shader Engine (60–144 FPS, <0.3ms latency).
   100% WYSIWYG parity with Python backend CPU engine.
   ============================================================ */


const halftoneBaseExtraParams = {
  satellite_enabled: false,
  satellite_count: 4,
  satellite_size_ratio: 0.45,
  satellite_distance: 0.42,
  satellite_angle: 0.0,
  satellite_shape: "circle",
  satellite_stretch: 0.0,
  merge_factor: 0.0,
  aspect_x: 1.0,
  aspect_y: 1.0,
  row_offset: 0.0,
  jitter_pos: 0.0,
  jitter_size: 0.0,
  grad_type: "none",
  grad_angle: 45.0,
  grad_strength: 0.0,
  qualityMode: "full",
};

const halftonePresets = {
  manga: {
    mode: "manga", cell_size: 8, dot_shape: "circle", angle: 45, max_radius_ratio: 0.9,
    contrast: 1.25, gamma: 1.05, brightness: 0.0, black_threshold: 0.18, white_threshold: 0.86,
    invert: false, intensity: 1.0, dot_color: "#121212", bg_color: "#ffffff", antialias: 2,
    dither_type: "none", seed: 42, pre_blur: 0.0, preserve_lines: true, line_threshold: 0.24,
    paper_texture: 0.0, invert_mask: false, ...halftoneBaseExtraParams
  },
  classic: {
    mode: "classic", cell_size: 10, dot_shape: "circle", angle: 45, max_radius_ratio: 0.82,
    contrast: 1.05, gamma: 1.0, brightness: 0.0, black_threshold: 0.06, white_threshold: 0.94,
    invert: false, intensity: 1.0, dot_color: "#111111", bg_color: "#ffffff", antialias: 2,
    dither_type: "none", seed: 42, pre_blur: 0.3, preserve_lines: false, line_threshold: 0.22,
    paper_texture: 0.0, invert_mask: false, ...halftoneBaseExtraParams
  },
  newspaper: {
    mode: "newspaper", cell_size: 12, dot_shape: "circle", angle: 45, max_radius_ratio: 0.88,
    contrast: 1.15, gamma: 0.95, brightness: 0.0, black_threshold: 0.10, white_threshold: 0.90,
    invert: false, intensity: 1.0, dot_color: "#1f1e1c", bg_color: "#f4efe4", antialias: 2,
    dither_type: "noise", seed: 42, pre_blur: 0.4, preserve_lines: false, line_threshold: 0.25,
    paper_texture: 0.35, invert_mask: false, ...halftoneBaseExtraParams
  },
  color: {
    mode: "color", cell_size: 9, dot_shape: "circle", angle: 45, max_radius_ratio: 0.85,
    contrast: 1.10, gamma: 1.0, brightness: 0.0, black_threshold: 0.05, white_threshold: 0.95,
    invert: false, intensity: 1.0, dot_color: "#111111", bg_color: "#ffffff", antialias: 2,
    dither_type: "none", seed: 42, pre_blur: 0.2, preserve_lines: false, line_threshold: 0.22,
    paper_texture: 0.0, invert_mask: false, ...halftoneBaseExtraParams
  },
  custom: {
    mode: "custom", cell_size: 8, dot_shape: "circle", angle: 45, max_radius_ratio: 0.85,
    contrast: 1.0, gamma: 1.0, brightness: 0.0, black_threshold: 0.10, white_threshold: 0.90,
    invert: false, intensity: 1.0, dot_color: "#111111", bg_color: "#ffffff", antialias: 2,
    dither_type: "none", seed: 42, pre_blur: 0.0, preserve_lines: true, line_threshold: 0.22,
    paper_texture: 0.0, invert_mask: false, ...halftoneBaseExtraParams
  }
};

let glContext = null;
let glProgram = null;
let glSourceTexture = null;
let glMaskTexture = null;
let glUniformLocations = {};
// Ping-pong FBOs used by the multi-layer WebGL2 compositor.
let glFboTextures = [null, null];
let glFbos = [null, null];
let glFboSize = { w: 0, h: 0 };

// Zero-allocation reusable buffers for CPU compositor fallback
const _htBuffers = {
  _htAccR: null,
  _htBufN: 0,
};

// ============================================================
// WebGL2 Screentone Shader Sources (Ported from GenTonik & Python)
// ============================================================
const VS_SOURCE = `#version 300 es
in vec2 a_position;
uniform int u_flip_v;
out vec2 v_uv;
void main() {
    v_uv = (a_position + 1.0) * 0.5;
    // Image textures are uploaded top-row-first, while FBO textures are
    // bottom-row-first. Layer passes render into FBOs with u_flip_v=0 and the
    // final blit to the canvas uses u_flip_v=1, keeping both orientations exact.
    if (u_flip_v == 1) v_uv.y = 1.0 - v_uv.y;
    gl_Position = vec4(a_position, 0.0, 1.0);
}`;

const FS_SOURCE = `#version 300 es
precision highp float;

in vec2 v_uv;
out vec4 outColor;

uniform sampler2D u_image;
uniform sampler2D u_base;
uniform sampler2D u_mask;
uniform int u_has_mask;
uniform int u_invert_mask;

uniform vec2 u_resolution;
uniform float u_cell_size;
uniform int u_dot_shape; // 0..12 geometry shapes
uniform float u_angle; // radians
uniform float u_max_radius_ratio;
uniform float u_contrast;
uniform float u_gamma;
uniform float u_brightness;
uniform float u_black_threshold;
uniform float u_white_threshold;
uniform int u_invert;
uniform float u_intensity;
uniform vec3 u_dot_color;
uniform vec3 u_bg_color;
uniform int u_preserve_lines;
uniform float u_line_threshold;
uniform vec2 u_aspect;
uniform float u_row_offset;
uniform float u_jitter_pos;
uniform float u_jitter_size;

// GenTonik Planets & Satellites
uniform int u_satEnabled;
uniform int u_satCount;
uniform float u_satSizeRatio;
uniform float u_satDist;
uniform float u_satAngleBase;
uniform float u_satStretch;
uniform int u_satShape;
uniform float u_mergeFactor;

// Spatial Gradients
uniform int u_gradType; // 0=none, 1=linear, 2=radial, 3=vignette
uniform float u_gradAngle; // radians
uniform float u_gradStrength;

// Layer Opacity, Blend Mode & AA
uniform float u_opacity;
uniform int u_blend_mode; // 0=normal, 1=multiply, 2=screen, 3=overlay, 4=darken, 5=lighten
uniform int u_antialias;  // 1=1x (hard 1-bit), 2..4=AA
uniform int u_mode;       // 0=classic, 1=manga, 2=color (CMYK)
uniform int u_pass_mode;  // 0=halftone layer, 1=blit composite to canvas

// 13 Geometry Shapes distance field (exact 1:1 math match with backend/halftone.py)
float evalShapeDist(vec2 p, float halfR, int shape, out bool isLinear) {
    isLinear = false;
    vec2 ap = abs(p);

    // 0: circle
    if (shape == 0) {
        return length(p) / (halfR * 1.128379);
    }
    // 1: square
    if (shape == 1) {
        return max(ap.x, ap.y) / halfR;
    }
    // 2: diamond
    if (shape == 2) {
        return (ap.x + ap.y) / (halfR * 1.41421356);
    }
    // 3: line
    if (shape == 3) {
        isLinear = true;
        return ap.y / halfR;
    }
    // 4: crosshatch
    if (shape == 4) {
        isLinear = true;
        return min(ap.x, ap.y) / halfR;
    }
    // 5: hexagon
    if (shape == 5) {
        float hexD = max(ap.x * 0.8660254 + ap.y * 0.5, ap.y);
        return hexD / (halfR * 0.95);
    }
    // 6: cross
    if (shape == 6) {
        float d = min(max(ap.x * 2.6, ap.y * 0.85), max(ap.x * 0.85, ap.y * 2.6));
        return d / (halfR * 0.9);
    }
    // 7: ellipse
    if (shape == 7) {
        return length(vec2(p.x * 0.82, p.y * 1.22)) / (halfR * 1.05);
    }
    // 8: triangle
    if (shape == 8) {
        float ty = p.y + 0.08 * (halfR * 2.0);
        float triD = max(ap.x * 1.73205 + ty, -ty * 2.0) / 1.5;
        return max(0.0, triD) / (halfR * 0.95);
    }
    // 9: star
    if (shape == 9) {
        float r = length(p);
        float theta = atan(p.y, p.x);
        float starMod = 0.68 + 0.32 * cos(5.0 * theta - 1.570796327);
        return r / (halfR * 1.22 * max(starMod, 0.2));
    }
    // 10: heart
    if (shape == 10) {
        float hx = p.x / (halfR * 1.15);
        float hy = -p.y / (halfR * 1.15) + 0.15;
        float r = length(vec2(hx, hy - 0.35 * sqrt(abs(hx) + 0.0001)));
        return r / 0.85;
    }
    // 11: concentric
    if (shape == 11) {
        isLinear = true;
        float r = length(p) / (halfR * 1.2);
        return abs((r * 2.0) - floor(r * 2.0) - 0.5) * 2.0;
    }
    // 12: ring
    if (shape == 12) {
        isLinear = true;
        float r = length(p) / (halfR * 1.1);
        return abs(r - 0.52) * 2.3;
    }
    // Default circle
    return length(p) / (halfR * 1.128379);
}

float coverageAA(float d, float targetR, int aa, float fallbackFw) {
    if (aa <= 1) {
        return (d < targetR) ? 1.0 : 0.0;
    }
    float fw = max(fwidth(d), fallbackFw) * (float(aa) * 0.45);
    fw = max(fw, 0.0001);
    return clamp(0.5 - (d - targetR) / fw, 0.0, 1.0);
}

float hash21(vec2 p) {
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}

vec3 blendLayerRgb(vec3 b, vec3 t, int mode) {
    if (mode == 1) return b * t;                                  // multiply
    if (mode == 2) return 1.0 - (1.0 - b) * (1.0 - t);            // screen
    if (mode == 3) {                                              // overlay
        vec3 res;
        res.r = (b.r < 0.5) ? (2.0 * b.r * t.r) : (1.0 - 2.0 * (1.0 - b.r) * (1.0 - t.r));
        res.g = (b.g < 0.5) ? (2.0 * b.g * t.g) : (1.0 - 2.0 * (1.0 - b.g) * (1.0 - t.g));
        res.b = (b.b < 0.5) ? (2.0 * b.b * t.b) : (1.0 - 2.0 * (1.0 - b.b) * (1.0 - t.b));
        return clamp(res, 0.0, 1.0);
    }
    if (mode == 4) return min(b, t);                              // darken
    if (mode == 5) return max(b, t);                              // lighten
    return t;                                                     // normal
}

float evalDotInk(vec2 pixelPos, vec2 center, float cov, float angleRad, vec2 pitch, float halfR, int shape, int aa, float rowOff) {
    if (cov <= 0.0001) return 0.0;
    if (cov >= 0.9999) return 1.0;
    vec2 d = pixelPos - center;
    float cosA = cos(angleRad);
    float sinA = sin(angleRad);
    float u = (d.x * cosA + d.y * sinA) / pitch.x;
    float v = (-d.x * sinA + d.y * cosA) / pitch.y;
    float cellV = floor(v);
    if (abs(rowOff) > 0.0001) {
        float oddRow = mod(abs(cellV), 2.0);
        u = u + oddRow * rowOff;
    }
    float cellU = floor(u);
    float du = (u - cellU) - 0.5;
    float dv = (v - cellV) - 0.5;
    bool isLinear = false;
    float dist = evalShapeDist(vec2(du, dv), halfR, shape, isLinear);
    float targetR = isLinear ? clamp(cov, 0.0, 1.0) : sqrt(clamp(cov, 0.0, 1.0));
    float cellPx = 1.0 / min(pitch.x, pitch.y);
    float fallbackFw = cellPx / max(halfR, 0.02);
    return coverageAA(dist, targetR, aa, fallbackFw);
}

void main() {
    vec4 srcColor = texture(u_image, v_uv);

    // Final pass: present the composited FBO to the canvas without re-toning.
    if (u_pass_mode == 1) {
        outColor = vec4(texture(u_base, v_uv).rgb, 1.0);
        return;
    }

    vec3 toneColor = vec3(0.0);

    // If CMYK Mode (u_mode == 2): Multi-channel subtractive halftoning
    if (u_mode == 2) {
        vec3 rgbNorm = srcColor.rgb;
        if (u_invert == 1) rgbNorm = 1.0 - rgbNorm;
        rgbNorm = clamp(rgbNorm + u_brightness, 0.0, 1.0);
        float span = max(0.001, u_white_threshold - u_black_threshold);
        rgbNorm = clamp((rgbNorm - u_black_threshold) / span, 0.0, 1.0);
        rgbNorm = pow(clamp(rgbNorm, 0.0, 1.0), vec3(1.0 / max(0.01, u_gamma)));
        rgbNorm = clamp((rgbNorm - 0.5) * u_contrast + 0.5, 0.0, 1.0);

        float kChan = clamp((1.0 - max(max(rgbNorm.r, rgbNorm.g), rgbNorm.b)) * 0.85, 0.0, 1.0);
        float denom = max(1.0 - kChan, 0.0001);
        float cChan = clamp((1.0 - rgbNorm.r - kChan) / denom, 0.0, 1.0);
        float mChan = clamp((1.0 - rgbNorm.g - kChan) / denom, 0.0, 1.0);
        float yChan = clamp((1.0 - rgbNorm.b - kChan) / denom, 0.0, 1.0);

        vec2 pixelPos = v_uv * u_resolution;
        vec2 center = u_resolution * 0.5;
        vec2 pitch = max(vec2(1.5), vec2(u_cell_size) * u_aspect);
        float halfR = max(0.05, 0.5 * u_max_radius_ratio);
        float effRowOffset = (u_dot_shape == 5 && abs(u_row_offset) < 0.0001) ? 0.5 : u_row_offset;

        // Screen each channel at standard offset angles: C: 15°, M: 75°, Y: 0°, K: 45°
        float cDot = evalDotInk(pixelPos, center, cChan, u_angle + 0.261799, pitch, halfR, u_dot_shape, u_antialias, effRowOffset);
        float mDot = evalDotInk(pixelPos, center, mChan, u_angle + 1.308997, pitch, halfR, u_dot_shape, u_antialias, effRowOffset);
        float yDot = evalDotInk(pixelPos, center, yChan, u_angle + 0.000000, pitch, halfR, u_dot_shape, u_antialias, effRowOffset);
        float kDot = evalDotInk(pixelPos, center, kChan, u_angle + 0.785398, pitch, halfR, u_dot_shape, u_antialias, effRowOffset);

        vec3 cmykRgb = vec3(
            u_bg_color.r * (1.0 - cDot) * (1.0 - kDot),
            u_bg_color.g * (1.0 - mDot) * (1.0 - kDot),
            u_bg_color.b * (1.0 - yDot) * (1.0 - kDot)
        );

        if (u_preserve_lines == 1) {
            vec2 texel = 1.0 / u_resolution;
            float lCenter = dot(srcColor.rgb, vec3(0.299, 0.587, 0.114));
            float lLeft   = dot(texture(u_image, v_uv - vec2(texel.x, 0.0)).rgb, vec3(0.299, 0.587, 0.114));
            float lRight  = dot(texture(u_image, v_uv + vec2(texel.x, 0.0)).rgb, vec3(0.299, 0.587, 0.114));
            float lUp     = dot(texture(u_image, v_uv - vec2(0.0, texel.y)).rgb, vec3(0.299, 0.587, 0.114));
            float lDown   = dot(texture(u_image, v_uv + vec2(0.0, texel.y)).rgb, vec3(0.299, 0.587, 0.114));
            float edgeMag = length(vec2(lRight - lLeft, lDown - lUp));
            if ((edgeMag > 0.08 && lCenter < 0.65) || lCenter < u_line_threshold) {
                cmykRgb = vec3(0.0);
            }
        }

        toneColor = cmykRgb;
    } else {
    // Monochromatic & Manga Modes (u_mode 0 or 1)
    float gray = dot(srcColor.rgb, vec3(0.299, 0.587, 0.114));
    float g = (u_invert == 1) ? (1.0 - gray) : gray;
    g = clamp(g + u_brightness, 0.0, 1.0);
    float span = max(0.001, u_white_threshold - u_black_threshold);
    g = clamp((g - u_black_threshold) / span, 0.0, 1.0);
    g = pow(clamp(g, 0.0, 1.0), 1.0 / max(0.01, u_gamma));
    g = clamp((g - 0.5) * u_contrast + 0.5, 0.0, 1.0);

    float inkAmount = 1.0 - g;
    if (u_mode == 1) {
        if (g >= 0.94) inkAmount = 0.0;
        if (g <= 0.06) inkAmount = 1.0;
    }

    float covMod = inkAmount;
    if (u_gradType > 0 && abs(u_gradStrength) > 0.001) {
        vec2 gCoord = v_uv - 0.5;
        float gMap = 0.5;
        if (u_gradType == 1) {
            gMap = (gCoord.x * cos(u_gradAngle) + gCoord.y * sin(u_gradAngle)) + 0.5;
        } else if (u_gradType == 2) {
            gMap = clamp(1.0 - length(gCoord) * 1.41421356, 0.0, 1.0);
        } else if (u_gradType == 3) {
            gMap = clamp(pow(length(gCoord) * 1.41421356, 1.6) - 0.25, 0.0, 1.0);
        }
        covMod = clamp(mix(covMod, covMod * gMap, u_gradStrength), 0.0, 1.0);
    }

    vec2 pixelPos = v_uv * u_resolution;
    vec2 center = u_resolution * 0.5;
    vec2 d = pixelPos - center;
    float cosA = cos(u_angle);
    float sinA = sin(u_angle);
    vec2 pitch = max(vec2(1.5), vec2(u_cell_size) * u_aspect);

    float u = (d.x * cosA + d.y * sinA) / pitch.x;
    float v = (-d.x * sinA + d.y * cosA) / pitch.y;

    float cellV = floor(v);
    float effRowOffset = (u_dot_shape == 5 && abs(u_row_offset) < 0.0001) ? 0.5 : u_row_offset;
    if (abs(effRowOffset) > 0.0001) {
        float oddRow = mod(abs(cellV), 2.0);
        u = u + oddRow * effRowOffset;
    }
    float cellU = floor(u);
    float du = (u - cellU) - 0.5;
    float dv = (v - cellV) - 0.5;

    vec2 cellCoord = vec2(cellU, cellV);
    if (u_jitter_pos > 0.001) {
        float jx = (hash21(cellCoord + 11.0) - 0.5) * u_jitter_pos * 0.55;
        float jy = (hash21(cellCoord + 29.0) - 0.5) * u_jitter_pos * 0.55;
        du -= jx;
        dv -= jy;
    }

    float halfR = max(0.05, 0.5 * u_max_radius_ratio);
    bool isLinear = false;
    float dist = evalShapeDist(vec2(du, dv), halfR, u_dot_shape, isLinear);
    float targetR = isLinear ? clamp(covMod, 0.0, 1.0) : sqrt(clamp(covMod, 0.0, 1.0));
    if (u_jitter_size > 0.001) {
        targetR *= (1.0 + (hash21(cellCoord) - 0.5) * u_jitter_size);
    }
    float cellPx = 1.0 / min(pitch.x, pitch.y);
    float fallbackFw = cellPx / max(halfR, 0.02);
    float ink = coverageAA(dist, targetR, u_antialias, fallbackFw);

    // GenTonik Planets & Satellites (1..8 slot orbit)
    if (u_satEnabled == 1 && u_satCount > 0) {
        float satDist = max(0.12, u_satDist * (1.0 - u_mergeFactor * 0.5));
        float satMaxR = max(0.08, halfR * u_satSizeRatio * (1.0 + u_mergeFactor * 0.45));
        int effCount = clamp(u_satCount, 1, 8);
        float stepAng = 6.283185307 / float(effCount);
        float satFallbackFw = cellPx / max(satMaxR, 0.02);

        for (int i = 0; i < 8; i++) {
            if (i >= effCount) break;
            float theta = u_satAngleBase + float(i) * stepAng;
            float rsx = satDist * cos(theta);
            float rsy = satDist * sin(theta);
            vec2 sPos = vec2(du - rsx, dv - rsy);

            if (u_satStretch > 0.001 && (abs(rsx) > 0.00001 || abs(rsy) > 0.00001)) {
                float dirAngle = atan(-rsy, -rsx);
                float cd = cos(dirAngle);
                float sd = sin(dirAngle);
                float rotU = (sPos.x * cd + sPos.y * sd) / (1.0 + u_satStretch);
                float rotV = -sPos.x * sd + sPos.y * cd;
                sPos.x = rotU * cd - rotV * sd;
                sPos.y = rotU * sd + rotV * cd;
            }

            bool sLinear = false;
            float sDist = evalShapeDist(sPos, satMaxR, u_satShape, sLinear);
            float sTarget = sLinear ? clamp(covMod, 0.0, 1.0) : sqrt(clamp(covMod, 0.0, 1.0));
            float satInk = coverageAA(sDist, sTarget, u_antialias, satFallbackFw);
            ink = max(ink, satInk);
        }
    }

    if (covMod <= 0.0001) ink = 0.0;
    if (u_mode == 1 && g <= 0.04) ink = 1.0;
    if (u_max_radius_ratio >= 0.92 && covMod >= 0.96) ink = 1.0;

    if (u_preserve_lines == 1) {
        vec2 texel = 1.0 / u_resolution;
        float lCenter = dot(srcColor.rgb, vec3(0.299, 0.587, 0.114));
        float lLeft   = dot(texture(u_image, v_uv - vec2(texel.x, 0.0)).rgb, vec3(0.299, 0.587, 0.114));
        float lRight  = dot(texture(u_image, v_uv + vec2(texel.x, 0.0)).rgb, vec3(0.299, 0.587, 0.114));
        float lUp     = dot(texture(u_image, v_uv - vec2(0.0, texel.y)).rgb, vec3(0.299, 0.587, 0.114));
        float lDown   = dot(texture(u_image, v_uv + vec2(0.0, texel.y)).rgb, vec3(0.299, 0.587, 0.114));
        float edgeMag = length(vec2(lRight - lLeft, lDown - lUp));
        if ((edgeMag > 0.08 && lCenter < 0.65) || lCenter < u_line_threshold) {
            ink = max(ink, 1.0);
        }
    }

    toneColor = mix(u_bg_color, u_dot_color, ink);
    }

    // Multi-pass layer compositor: blend this raster layer over the previous
    // composite (u_base), then apply the layer mask and opacity.
    vec4 baseColor = texture(u_base, v_uv);
    float maskFactor = 1.0;
    if (u_has_mask == 1) {
        vec4 mPix = texture(u_mask, v_uv);
        maskFactor = max(mPix.a, dot(mPix.rgb, vec3(0.299, 0.587, 0.114)));
        if (u_invert_mask == 1) maskFactor = 1.0 - maskFactor;
    }
    vec3 layerRgb = mix(srcColor.rgb, toneColor, clamp(u_intensity, 0.0, 1.0));
    vec3 blended = blendLayerRgb(baseColor.rgb, layerRgb, u_blend_mode);
    float totalAlpha = clamp(u_opacity * maskFactor, 0.0, 1.0);
    outColor = vec4(mix(baseColor.rgb, blended, totalAlpha), 1.0);
}`;

const HT_ORIGINAL_LAYER_ID = "hl_original";

function createHalftoneLayer(name, paramsObj = null, options = {}) {
  const opts = options || {};
  return {
    id: "hl_" + Date.now() + "_" + Math.floor(Math.random() * 10000),
    name: name || "Слой растра",
    visible: true,
    opacity: 1.0,
    blend_mode: "normal",
    params: { ...(paramsObj || halftonePresets.manga) },
    maskCanvas: null,
    maskDirty: false,
    hasMask: Boolean(opts.hasMask),
    isBase: Boolean(opts.isBase),
    isOriginal: false,
    locked: false,
  };
}

function createHalftoneOriginalLayer() {
  return {
    id: HT_ORIGINAL_LAYER_ID,
    name: "Исходное фото",
    visible: true,
    opacity: 1.0,
    blend_mode: "normal",
    params: { ...halftonePresets.manga },
    maskCanvas: null,
    maskDirty: false,
    hasMask: false,
    isBase: false,
    isOriginal: true,
    locked: true,
  };
}

function ensureHalftoneLayerStack(st) {
  if (!st) return null;
  st.layers = Array.isArray(st.layers) ? st.layers.filter(Boolean) : [];
  let original = st.layers.find((l) => l.isOriginal || l.id === HT_ORIGINAL_LAYER_ID);
  if (!original) {
    original = createHalftoneOriginalLayer();
    st.layers.unshift(original);
  } else {
    original.id = HT_ORIGINAL_LAYER_ID;
    original.isOriginal = true;
    original.locked = true;
    original.isBase = false;
  }
  st.layers.forEach((l) => {
    if (l === original) return;
    l.isOriginal = false;
    l.locked = false;
    if (typeof l.hasMask !== "boolean") l.hasMask = Boolean(l.maskCanvas);
  });
  if (!st.layers.some((l) => !l.isOriginal)) {
    const base = createHalftoneLayer("Базовый растр", st.params || halftonePresets.manga, { isBase: true });
    st.layers.push(base);
  }
  if (!st.layers.some((l) => l.isBase)) {
    const firstRaster = st.layers.find((l) => !l.isOriginal);
    if (firstRaster) firstRaster.isBase = true;
  }
  const active = st.layers.find((l) => l.id === st.activeLayerId);
  if (!active || active.isOriginal) {
    const firstRaster = st.layers.find((l) => !l.isOriginal);
    st.activeLayerId = firstRaster ? firstRaster.id : null;
  }
  return st.layers;
}

function resetHalftoneLayerStack(appRef) {
  const app = appRef || window.App;
  const st = app?.state?.halftone;
  if (!st) return;
  const baseParams = { ...(st.params || halftonePresets.manga) };
  const base = createHalftoneLayer("Базовый растр", baseParams, { isBase: true });
  st.layers = [createHalftoneOriginalLayer(), base];
  st.activeLayerId = base.id;
  st.params = { ...baseParams };
  st.comparing = false;
  st.tool = "view";
  const maskCanvas = document.getElementById("ht-mask-canvas");
  if (maskCanvas) {
    const ctx = maskCanvas.getContext("2d");
    ctx.clearRect(0, 0, maskCanvas.width, maskCanvas.height);
  }
  renderHalftoneLayersUI(app);
  return st.layers;
}

function initHalftoneStudio(appRef) {
  const app = appRef || window.App;
  if (!app) return;
  if (!app.state.halftone) {
    app.state.halftone = {
      params: { ...halftonePresets.manga },
      layers: [],
      activeLayerId: null,
      sourcePath: null,
      sourceUrl: null,
      sourceDataUrl: null,
      resultDataUrl: null,
      comparing: false,
      tool: "view",
      brushSize: 24,
      lassoMode: "replace",
      lassoPoints: [],
      isDrawingLasso: false,
      isDrawingBrush: false,
      maskDirty: false,
      liveReady: false,
      liveActive: false,
      qualityMode: "full",
      debounceTimer: null,
      _htAccR: null,
      _htBufN: 0,
    };
  }

  ensureHalftoneLayerStack(app.state.halftone);
  ensureWebGLCanvas(app);
  renderHalftoneLayersUI(app);
  syncHalftoneControlsFromParams(app.state.halftone.params);
}

function ensureWebGLCanvas(appRef) {
  const app = appRef || window.App;
  let canvas = document.getElementById("ht-live-canvas");
  if (!canvas) {
    const wrap = document.getElementById("ht-preview-wrap");
    if (!wrap) return null;
    canvas = document.createElement("canvas");
    canvas.id = "ht-live-canvas";
    canvas.className = "ht-live-canvas";
    canvas.style.position = "absolute";
    canvas.style.top = "0";
    canvas.style.left = "0";
    canvas.style.width = "100%";
    canvas.style.height = "100%";
    canvas.style.pointerEvents = "none";
    canvas.style.display = "none";
    canvas.style.zIndex = "2";
    wrap.appendChild(canvas);
  }

  if (!glContext) {
    try {
      glContext = canvas.getContext("webgl2", { alpha: false, depth: false, antialias: false, preserveDrawingBuffer: true });
      if (glContext) {
        initWebGLProgram(glContext);
      }
    } catch (e) {
      console.warn("WebGL2 initialization failed, falling back to 2D:", e);
    }
  }
  return canvas;
}

function initWebGLProgram(gl) {
  const compileShader = (src, type) => {
    const s = gl.createShader(type);
    gl.shaderSource(s, src);
    gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
      console.error(gl.getShaderInfoLog(s));
      gl.deleteShader(s);
      return null;
    }
    return s;
  };

  const vs = compileShader(VS_SOURCE, gl.VERTEX_SHADER);
  const fs = compileShader(FS_SOURCE, gl.FRAGMENT_SHADER);
  if (!vs || !fs) return;

  glProgram = gl.createProgram();
  gl.attachShader(glProgram, vs);
  gl.attachShader(glProgram, fs);
  gl.linkProgram(glProgram);

  if (!gl.getProgramParameter(glProgram, gl.LINK_STATUS)) {
    console.error(gl.getProgramInfoLog(glProgram));
    return;
  }

  // Setup full-screen quad [-1, 1]
  const quadBuf = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, quadBuf);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([
    -1, -1,
     1, -1,
    -1,  1,
     1,  1,
  ]), gl.STATIC_DRAW);

  const posLoc = gl.getAttribLocation(glProgram, "a_position");
  gl.enableVertexAttribArray(posLoc);
  gl.vertexAttribPointer(posLoc, 2, gl.FLOAT, false, 0, 0);

  // Cache uniform locations
  const uniforms = [
    "u_image", "u_resolution", "u_cell_size", "u_dot_shape", "u_angle",
    "u_max_radius_ratio", "u_contrast", "u_gamma", "u_brightness",
    "u_black_threshold", "u_white_threshold", "u_invert", "u_intensity",
    "u_dot_color", "u_bg_color", "u_preserve_lines", "u_line_threshold",
    "u_aspect", "u_row_offset", "u_jitter_pos", "u_jitter_size",
    "u_satEnabled", "u_satCount", "u_satSizeRatio", "u_satDist",
    "u_satAngleBase", "u_satStretch", "u_satShape", "u_mergeFactor",
    "u_gradType", "u_gradAngle", "u_gradStrength",
    "u_opacity", "u_blend_mode", "u_antialias", "u_mode", "u_mask", "u_has_mask", "u_invert_mask",
    "u_base", "u_flip_v", "u_pass_mode"
  ];
  glUniformLocations = {};
  for (const u of uniforms) {
    glUniformLocations[u] = gl.getUniformLocation(glProgram, u);
  }
}

function setHalftoneQualityMode(mode, appRef) {
  const app = appRef || window.App;
  initHalftoneStudio(app);
  const st = app.state.halftone;
  if (!st) return;

  st.qualityMode = mode || "full";

  const btnFull = document.getElementById("ht-quality-full");
  const btnLive = document.getElementById("ht-quality-live");
  if (btnFull) btnFull.classList.toggle("active", st.qualityMode === "full");
  if (btnLive) btnLive.classList.toggle("active", st.qualityMode === "live");

  const canvas = document.getElementById("ht-live-canvas");
  const imgEl = document.getElementById("ht-preview-img");

  if (st.qualityMode === "full") {
    if (canvas) canvas.style.display = "none";
    if (imgEl) imgEl.style.visibility = "visible";
    runHalftonePreview(true, app);
  } else {
    if (imgEl) imgEl.style.visibility = "hidden";
    if (canvas) canvas.style.display = "block";
    renderHalftoneWebGL(app);
  }
  if (app.toast) {
    app.toast("info", st.qualityMode === "live" ? "⚡ Включён движок WebGL2 GPU (Живой)" : "🎨 Включён движок Точный CPU (Canvas 2D)");
  }
}

function prepareHalftoneLiveSource(srcUrl, appRef) {
  if (!srcUrl) return;
  const app = appRef || window.App;
  const st = app?.state?.halftone;
  if (!st) return;

  const img = new Image();
  img.crossOrigin = "anonymous";
  img.onload = () => {
    st.liveWidth = img.naturalWidth || img.width;
    st.liveHeight = img.naturalHeight || img.height;
    st.liveReady = true;

    const canvas = ensureWebGLCanvas(app);
    if (canvas && glContext && glProgram) {
      // Full natural resolution up to 4096px
      const maxSide = 4096;
      let w = st.liveWidth;
      let h = st.liveHeight;
      if (Math.max(w, h) > maxSide) {
        const sc = maxSide / Math.max(w, h);
        w = Math.round(w * sc);
        h = Math.round(h * sc);
      }
      canvas.width = w;
      canvas.height = h;
      glContext.viewport(0, 0, canvas.width, canvas.height);

      if (!glSourceTexture) {
        glSourceTexture = glContext.createTexture();
      }
      glContext.activeTexture(glContext.TEXTURE0);
      glContext.bindTexture(glContext.TEXTURE_2D, glSourceTexture);
      glContext.texParameteri(glContext.TEXTURE_2D, glContext.TEXTURE_WRAP_S, glContext.CLAMP_TO_EDGE);
      glContext.texParameteri(glContext.TEXTURE_2D, glContext.TEXTURE_WRAP_T, glContext.CLAMP_TO_EDGE);
      glContext.texParameteri(glContext.TEXTURE_2D, glContext.TEXTURE_MIN_FILTER, glContext.NEAREST);
      glContext.texParameteri(glContext.TEXTURE_2D, glContext.TEXTURE_MAG_FILTER, glContext.NEAREST);
      glContext.texImage2D(glContext.TEXTURE_2D, 0, glContext.RGBA, glContext.RGBA, glContext.UNSIGNED_BYTE, img);
      syncHalftoneMaskCanvasSize(st.liveWidth, st.liveHeight, app);

      if (st.qualityMode === "live") {
        renderHalftoneWebGL(app);
      }
    }
  };
  img.src = srcUrl;
}

function triggerHalftoneLiveUpdate(appRef) {
  const app = appRef || window.App;
  const st = app?.state?.halftone;
  if (!st || (!st.sourcePath && !st.sourceDataUrl)) return;

  const canvas = document.getElementById("ht-live-canvas");
  const imgEl = document.getElementById("ht-preview-img");

  if (st.qualityMode === "full") {
    if (canvas) canvas.style.display = "none";
    if (imgEl) imgEl.style.visibility = "visible";
    if (st.debounceTimer) clearTimeout(st.debounceTimer);
    st.debounceTimer = setTimeout(() => {
      runHalftonePreview(false, app);
    }, 120);
    return;
  }

  // Live GPU mode
  if (canvas) canvas.style.display = "block";
  if (imgEl) imgEl.style.visibility = "hidden";

  if (st.liveReady && glContext) {
    if (!st._rafPending) {
      st._rafPending = true;
      requestAnimationFrame(() => {
        st._rafPending = false;
        renderHalftoneWebGL(app);
      });
    }

    if (st.debounceTimer) clearTimeout(st.debounceTimer);
    st.debounceTimer = setTimeout(() => {
      syncHalftoneBackendSilent(app);
    }, 500);
  } else {
    if (st.debounceTimer) clearTimeout(st.debounceTimer);
    st.debounceTimer = setTimeout(() => {
      runHalftonePreview(false, app);
    }, 120);
  }
}

function ensureHalftoneFramebuffers(gl, w, h) {
  const width = Math.max(1, Math.round(Number(w) || 1));
  const height = Math.max(1, Math.round(Number(h) || 1));
  if (
    glFboSize.w === width &&
    glFboSize.h === height &&
    glFbos[0] && glFbos[1] &&
    glFboTextures[0] && glFboTextures[1]
  ) {
    return true;
  }

  for (let i = 0; i < 2; i++) {
    if (glFbos[i]) {
      gl.deleteFramebuffer(glFbos[i]);
      glFbos[i] = null;
    }
    if (glFboTextures[i]) {
      gl.deleteTexture(glFboTextures[i]);
      glFboTextures[i] = null;
    }
  }

  for (let i = 0; i < 2; i++) {
    const tex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, width, height, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

    const fbo = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
    if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE) {
      console.warn("Halftone Studio: incomplete WebGL2 framebuffer");
    }
    glFboTextures[i] = tex;
    glFbos[i] = fbo;
  }

  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  glFboSize = { w: width, h: height };
  return true;
}

function bindHalftoneLayerMask(gl, lyr) {
  const maskUnit = 1;
  gl.activeTexture(gl.TEXTURE0 + maskUnit);
  gl.uniform1i(glUniformLocations.u_mask, maskUnit);

  if (!lyr || !lyr.hasMask || !lyr.maskCanvas) {
    gl.bindTexture(gl.TEXTURE_2D, null);
    gl.uniform1i(glUniformLocations.u_has_mask, 0);
    return;
  }

  if (!lyr._glMaskTexture) {
    lyr._glMaskTexture = gl.createTexture();
  }
  gl.bindTexture(gl.TEXTURE_2D, lyr._glMaskTexture);
  if (lyr.maskDirty || lyr._glMaskCanvas !== lyr.maskCanvas) {
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, lyr.maskCanvas);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    lyr._glMaskCanvas = lyr.maskCanvas;
    lyr.maskDirty = false;
  }
  gl.uniform1i(glUniformLocations.u_has_mask, 1);
}

function applyHalftoneLayerParams(gl, p, st, width, height) {
  const params = p || {};
  gl.uniform2f(glUniformLocations.u_resolution, width, height);

  // Cell size scaled to natural dimensions
  const scaleFactor = st.origWidth ? (width / Math.max(st.origWidth, 1)) : 1.0;
  const cell = Math.max(2.5, (Number(params.cell_size) || 8) * Math.max(0.45, scaleFactor));
  gl.uniform1f(glUniformLocations.u_cell_size, cell);

  // 13 Geometry shapes map
  const shapeMap = {
    circle: 0,
    square: 1,
    diamond: 2,
    line: 3,
    crosshatch: 4,
    hexagon: 5,
    cross: 6,
    ellipse: 7,
    triangle: 8,
    star: 9,
    heart: 10,
    concentric: 11,
    ring: 12,
    stipple: 0,
    stochastic: 0,
  };
  gl.uniform1i(glUniformLocations.u_dot_shape, shapeMap[params.dot_shape] ?? 0);

  const angleRad = ((Number(params.angle) || 45) * Math.PI) / 180.0;
  gl.uniform1f(glUniformLocations.u_angle, angleRad);

  gl.uniform1f(glUniformLocations.u_max_radius_ratio, Number(params.max_radius_ratio) || 0.9);
  gl.uniform1f(glUniformLocations.u_contrast, Number(params.contrast ?? 1.25));
  gl.uniform1f(glUniformLocations.u_gamma, Number(params.gamma ?? 1.05));
  gl.uniform1f(glUniformLocations.u_brightness, Number(params.brightness ?? 0.0));
  gl.uniform1f(glUniformLocations.u_black_threshold, Number(params.black_threshold ?? 0.18));
  gl.uniform1f(glUniformLocations.u_white_threshold, Number(params.white_threshold ?? 0.86));
  gl.uniform1i(glUniformLocations.u_invert, params.invert ? 1 : 0);
  gl.uniform1f(glUniformLocations.u_intensity, Number(params.intensity ?? 1.0));

  const parseRgb = (hex, def) => {
    const s = String(hex || "").replace("#", "");
    if (s.length === 6) {
      return [parseInt(s.slice(0, 2), 16) / 255, parseInt(s.slice(2, 4), 16) / 255, parseInt(s.slice(4, 6), 16) / 255];
    }
    return def;
  };
  const dotRgb = parseRgb(params.dot_color, [0.07, 0.07, 0.07]);
  const bgRgb = parseRgb(params.bg_color, [1.0, 1.0, 1.0]);
  gl.uniform3f(glUniformLocations.u_dot_color, dotRgb[0], dotRgb[1], dotRgb[2]);
  gl.uniform3f(glUniformLocations.u_bg_color, bgRgb[0], bgRgb[1], bgRgb[2]);

  gl.uniform1i(glUniformLocations.u_preserve_lines, params.preserve_lines ? 1 : 0);
  gl.uniform1f(glUniformLocations.u_line_threshold, Number(params.line_threshold ?? 0.22));
  gl.uniform2f(glUniformLocations.u_aspect, Number(params.aspect_x ?? 1.0), Number(params.aspect_y ?? 1.0));
  gl.uniform1f(glUniformLocations.u_row_offset, Number(params.row_offset ?? 0.0));
  gl.uniform1f(glUniformLocations.u_jitter_pos, Number(params.jitter_pos ?? 0.0));
  gl.uniform1f(glUniformLocations.u_jitter_size, Number(params.jitter_size ?? 0.0));

  // GenTonik Planets & Satellites uniforms
  const satEnabled = Boolean(params.satellite_enabled);
  gl.uniform1i(glUniformLocations.u_satEnabled, satEnabled ? 1 : 0);
  gl.uniform1i(glUniformLocations.u_satCount, Number(params.satellite_count ?? 4));
  gl.uniform1f(glUniformLocations.u_satSizeRatio, Number(params.satellite_size_ratio ?? 0.45));
  gl.uniform1f(glUniformLocations.u_satDist, Number(params.satellite_distance ?? 0.42));
  const satAngleRad = ((Number(params.satellite_angle) || 0) * Math.PI) / 180.0;
  gl.uniform1f(glUniformLocations.u_satAngleBase, satAngleRad);
  gl.uniform1f(glUniformLocations.u_satStretch, Number(params.satellite_stretch ?? 0.0));
  gl.uniform1i(glUniformLocations.u_satShape, shapeMap[params.satellite_shape] ?? 0);
  gl.uniform1f(glUniformLocations.u_mergeFactor, Number(params.merge_factor ?? 0.0));

  // Spatial Gradients uniforms
  const gradMap = { none: 0, linear: 1, radial: 2, vignette: 3 };
  gl.uniform1i(glUniformLocations.u_gradType, gradMap[params.grad_type] ?? 0);
  const gradAngleRad = ((Number(params.grad_angle) || 45) * Math.PI) / 180.0;
  gl.uniform1f(glUniformLocations.u_gradAngle, gradAngleRad);
  gl.uniform1f(glUniformLocations.u_gradStrength, Number(params.grad_strength ?? 0.0));

  gl.uniform1i(glUniformLocations.u_antialias, Number(params.antialias) || 2);
  const modeVal = params.mode === "manga" ? 1 : (params.mode === "color" ? 2 : 0);
  gl.uniform1i(glUniformLocations.u_mode, modeVal);
}

function renderHalftoneWebGL(appRef) {
  const app = appRef || window.App;
  const st = app?.state?.halftone;
  if (!st || !st.liveReady || !glContext || !glProgram || !glSourceTexture) return;

  const gl = glContext;
  const canvas = document.getElementById("ht-live-canvas");
  if (!canvas || !canvas.width || !canvas.height) return;

  const width = canvas.width;
  const height = canvas.height;
  if (!ensureHalftoneFramebuffers(gl, width, height) || !glFbos[0] || !glFbos[1]) return;

  gl.useProgram(glProgram);
  gl.viewport(0, 0, width, height);

  // The original source texture is the same base the CPU backend composites over.
  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_2D, glSourceTexture);
  gl.uniform1i(glUniformLocations.u_image, 0);

  const rasterLayers = (Array.isArray(st.layers) ? st.layers : [])
    .filter((l) => l && !l.isOriginal && l.visible !== false);

  const blendModes = { normal: 0, multiply: 1, screen: 2, overlay: 3, darken: 4, lighten: 5 };
  let baseTexture = glSourceTexture;
  let writeIndex = 0;

  for (const lyr of rasterLayers) {
    gl.bindFramebuffer(gl.FRAMEBUFFER, glFbos[writeIndex]);
    gl.viewport(0, 0, width, height);

    gl.activeTexture(gl.TEXTURE2);
    gl.bindTexture(gl.TEXTURE_2D, baseTexture);
    gl.uniform1i(glUniformLocations.u_base, 2);

    const p = lyr.params || st.params || {};
    applyHalftoneLayerParams(gl, p, st, width, height);
    gl.uniform1i(glUniformLocations.u_pass_mode, 0);
    gl.uniform1i(glUniformLocations.u_flip_v, 0);
    gl.uniform1f(glUniformLocations.u_opacity, typeof lyr.opacity === "number" ? Math.max(0, Math.min(1, lyr.opacity)) : 1.0);
    gl.uniform1i(glUniformLocations.u_blend_mode, blendModes[lyr.blend_mode] ?? 0);
    gl.uniform1i(glUniformLocations.u_invert_mask, p.invert_mask ? 1 : 0);
    bindHalftoneLayerMask(gl, lyr);

    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    baseTexture = glFboTextures[writeIndex];
    writeIndex = 1 - writeIndex;
  }

  // Final blit to the default framebuffer. Source/FBO textures are sampled
  // bottom-row-first, so flip V exactly once here to display the image upright.
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  gl.viewport(0, 0, width, height);
  gl.activeTexture(gl.TEXTURE2);
  gl.bindTexture(gl.TEXTURE_2D, baseTexture);
  gl.uniform1i(glUniformLocations.u_base, 2);
  gl.uniform1i(glUniformLocations.u_pass_mode, 1);
  gl.uniform1i(glUniformLocations.u_flip_v, 1);
  gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  gl.activeTexture(gl.TEXTURE0);
}


async function syncHalftoneBackendSilent(appRef) {
  const app = appRef || window.App;
  const st = app?.state?.halftone;
  if (!st || (!st.sourcePath && !st.sourceDataUrl)) return;

  try {
    const layersPayload = buildHalftoneLayersPayload(st);
    const payload = {
      image_path: st.sourcePath,
      image_data: st.sourceDataUrl,
      params: st.params,
      layers: layersPayload,
      save: false,
      preview_max_side: 0,
    };
    const res = await apiPost("/api/halftone/process", payload);
    if (res && res.preview_data_url) {
      st.resultDataUrl = res.preview_data_url;
      const imgEl = document.getElementById("ht-preview-img");
      if (imgEl) imgEl.src = res.preview_data_url;
    }
  } catch (_) {}
}

async function runHalftonePreview(immediate = false, appRef) {
  const app = appRef || window.App;
  initHalftoneStudio(app);
  const st = app.state.halftone;
  if (!st.sourcePath && !st.sourceDataUrl) {
    if (immediate) toast("warning", "Сначала выберите или загрузите изображение");
    return;
  }

  const metaEl = document.getElementById("ht-meta-info");
  const loadOverlay = document.getElementById("ht-loading-overlay");
  if (metaEl) metaEl.textContent = "⏳ Обработка...";
  if (loadOverlay) loadOverlay.style.display = "flex";

  try {
    const layersPayload = buildHalftoneLayersPayload(st);
    const res = await apiPost("/api/halftone/process", {
      image_path: st.sourcePath,
      image_data: st.sourceDataUrl,
      params: st.params,
      layers: layersPayload,
      save: false,
      preview_max_side: 0, // 100% Full-Res WYSIWYG
    });

    st.resultDataUrl = res.preview_data_url;
    st.origWidth = res.orig_width || res.width;
    st.origHeight = res.orig_height || res.height;
    syncHalftoneMaskCanvasSize(st.origWidth, st.origHeight, app);

    const emptyEl = document.getElementById("ht-empty-state");
    const wrapEl = document.getElementById("ht-preview-wrap");
    const imgEl = document.getElementById("ht-preview-img");
    const canvas = document.getElementById("ht-live-canvas");

    if (emptyEl) emptyEl.style.display = "none";
    if (wrapEl) wrapEl.style.display = "inline-flex";

    if (imgEl) {
      imgEl.src = res.preview_data_url;
    }

    if (st.qualityMode === "full") {
      if (canvas) canvas.style.display = "none";
      if (imgEl) imgEl.style.visibility = "visible";
    } else {
      if (imgEl) imgEl.style.visibility = "hidden";
      if (canvas) canvas.style.display = "block";
    }

    prepareHalftoneLiveSource(st.sourceUrl || res.preview_data_url, app);
  } catch (e) {
    toast("error", "Halftone error: " + e.message);
  } finally {
    if (loadOverlay) loadOverlay.style.display = "none";
  }
}

function selectHalftoneMode(mode, appRef) {
  const app = appRef || window.App;
  initHalftoneStudio(app);
  const preset = halftonePresets[mode] || halftonePresets.custom;
  app.state.halftone.params = { ...app.state.halftone.params, ...preset, mode };
  const activeLyr = getActiveHalftoneLayer(app);
  if (activeLyr) activeLyr.params = app.state.halftone.params;
  syncHalftoneControlsFromParams(app.state.halftone.params);
  triggerHalftoneLiveUpdate(app);
}

function syncHalftoneControlsFromParams(p) {
  if (!p) return;
  document.querySelectorAll("#ht-mode-pills .studio-mode-pill").forEach(btn => {
    btn.classList.toggle("active", btn.dataset.mode === p.mode);
  });
  const setPair = (rId, nId, val) => {
    const r = document.getElementById(rId);
    const n = document.getElementById(nId);
    if (r && val !== undefined) r.value = val;
    if (n && val !== undefined) n.value = val;
  };
  setPair("ht-cell-size", "ht-cell-size-num", p.cell_size);
  setPair("ht-angle", "ht-angle-num", p.angle);
  setPair("ht-contrast", "ht-contrast-num", p.contrast);
  setPair("ht-gamma", "ht-gamma-num", p.gamma);
  setPair("ht-brightness", "ht-brightness-num", p.brightness);
  setPair("ht-max-radius", "ht-max-radius-num", p.max_radius_ratio);
  setPair("ht-black-point", "ht-black-point-num", p.black_threshold);
  setPair("ht-white-cut", "ht-white-cut-num", p.white_threshold);
  setPair("ht-intensity", "ht-intensity-num", p.intensity);

  const dotShapeSel = document.getElementById("ht-dot-shape");
  if (dotShapeSel && p.dot_shape) dotShapeSel.value = p.dot_shape;
  const invertChk = document.getElementById("ht-invert");
  if (invertChk) invertChk.checked = Boolean(p.invert);
  const linesChk = document.getElementById("ht-preserve-lines");
  if (linesChk) linesChk.checked = Boolean(p.preserve_lines);

  // Satellite & Planets controls sync
  const satBox = document.getElementById("ht-sat-controls");
  if (satBox) satBox.style.display = p.satellite_enabled ? "flex" : "none";
  const satChk = document.getElementById("ht-sat-enabled");
  if (satChk) satChk.checked = Boolean(p.satellite_enabled);
  const satCountSel = document.getElementById("ht-sat-count");
  if (satCountSel && p.satellite_count !== undefined) satCountSel.value = String(p.satellite_count);
  const satShapeSel = document.getElementById("ht-sat-shape");
  if (satShapeSel && p.satellite_shape) satShapeSel.value = p.satellite_shape;
  setPair("ht-sat-dist", "ht-sat-dist-num", p.satellite_distance);
  setPair("ht-sat-size", "ht-sat-size-num", p.satellite_size_ratio);
  setPair("ht-sat-angle", "ht-sat-angle-num", p.satellite_angle);
  setPair("ht-sat-stretch", "ht-sat-stretch-num", p.satellite_stretch);
  setPair("ht-merge-factor", "ht-merge-factor-num", p.merge_factor);

  // Grid Geometry & Gradient sync
  const aspX = document.getElementById("ht-aspect-x");
  if (aspX && p.aspect_x !== undefined) aspX.value = p.aspect_x;
  const aspY = document.getElementById("ht-aspect-y");
  if (aspY && p.aspect_y !== undefined) aspY.value = p.aspect_y;
  setPair("ht-row-offset", "ht-row-offset-num", p.row_offset);
  const jitPos = document.getElementById("ht-jitter-pos");
  if (jitPos && p.jitter_pos !== undefined) jitPos.value = p.jitter_pos;
  const jitSize = document.getElementById("ht-jitter-size");
  if (jitSize && p.jitter_size !== undefined) jitSize.value = p.jitter_size;
  const gradTypeSel = document.getElementById("ht-grad-type");
  if (gradTypeSel && p.grad_type) gradTypeSel.value = p.grad_type;
  setPair("ht-grad-angle", "ht-grad-angle-num", p.grad_angle);
  setPair("ht-grad-strength", "ht-grad-strength-num", p.grad_strength);

  // Antialias and Dither sync
  const aaSel = document.getElementById("ht-antialias");
  if (aaSel && p.antialias !== undefined) aaSel.value = String(p.antialias);
  const ditherSel = document.getElementById("ht-dither-type");
  if (ditherSel && p.dither_type) ditherSel.value = p.dither_type;
}

function onHalftoneParamInput(key, rawVal, appRef) {
  const app = appRef || window.App;
  initHalftoneStudio(app);
  let val = rawVal;
  if (["cell_size", "angle", "seed", "satellite_count", "antialias"].includes(key)) {
    val = parseInt(rawVal, 10);
    if (isNaN(val)) return;
  } else if (["contrast", "gamma", "brightness", "max_radius_ratio", "black_threshold", "white_threshold", "intensity", "satellite_distance", "satellite_size_ratio", "satellite_angle", "satellite_stretch", "aspect_x", "aspect_y", "row_offset", "jitter_pos", "jitter_size", "grad_angle", "grad_strength", "merge_factor"].includes(key)) {
    val = parseFloat(rawVal);
    if (isNaN(val)) return;
  }

  if (key === "satellite_enabled") {
    const satBox = document.getElementById("ht-sat-controls");
    if (satBox) satBox.style.display = val ? "flex" : "none";
  }

  app.state.halftone.params[key] = val;
  const activeLyr = getActiveHalftoneLayer(app);
  if (activeLyr) activeLyr.params[key] = val;

  triggerHalftoneLiveUpdate(app);
}

function buildHalftoneLayersPayload(st) {
  const layers = Array.isArray(st?.layers) ? st.layers : [];
  return layers
    .filter((l) => l && !l.isOriginal)
    .map((l) => ({
      id: l.id,
      name: l.name,
      visible: l.visible !== false,
      opacity: typeof l.opacity === "number" ? Math.max(0, Math.min(1, l.opacity)) : 1.0,
      blend_mode: l.blend_mode || "normal",
      params: { ...(l.params || st.params || {}) },
      hasMask: Boolean(l.hasMask && l.maskCanvas),
      isBase: Boolean(l.isBase),
      mask: (l.hasMask && l.maskCanvas) ? l.maskCanvas.toDataURL("image/png") : null,
    }));
}

function getActiveHalftoneLayer(appRef) {
  const app = appRef || window.App;
  const st = app?.state?.halftone;
  if (!st || !Array.isArray(st.layers)) return null;
  const active = st.layers.find((l) => l.id === st.activeLayerId && !l.isOriginal);
  if (active) return active;
  return st.layers.find((l) => !l.isOriginal) || null;
}

function normalizeHalftoneAngle(angle) {
  let a = Number(angle);
  if (!Number.isFinite(a)) a = 45;
  a = ((a % 180) + 180) % 180;
  return Math.round(a * 10) / 10;
}

function halftoneAngleDistance(a, b) {
  const d = Math.abs(normalizeHalftoneAngle(a) - normalizeHalftoneAngle(b));
  return Math.min(d, 180 - d);
}

function applyHalftoneMoireProtection(lyr, st, appRef) {
  if (!lyr || lyr.isOriginal || lyr.blend_mode !== "multiply" || !lyr.params) return false;
  const selfAngle = normalizeHalftoneAngle(lyr.params.angle ?? st.params?.angle ?? 45);
  const otherAngles = (st.layers || [])
    .filter((o) => o && o !== lyr && !o.isOriginal && o.visible !== false)
    .map((o) => normalizeHalftoneAngle(o.params?.angle ?? st.params?.angle ?? 45));

  let angle = selfAngle;
  for (let i = 0; i < 12; i++) {
    if (!otherAngles.some((a) => halftoneAngleDistance(a, angle) < 7.5)) break;
    angle = normalizeHalftoneAngle(angle + 15);
  }

  if (halftoneAngleDistance(angle, selfAngle) < 0.01) return false;

  lyr.params = { ...lyr.params, angle };
  if (lyr.id === st.activeLayerId) {
    st.params = { ...st.params, angle };
    syncHalftoneControlsFromParams(st.params);
  }
  const app = appRef || window.App;
  if (app?.toast) {
    app.toast("info", `🛡 Защита от муара: угол слоя «${lyr.name}» изменён на ${Math.round(angle)}°`);
  }
  return true;
}

function syncHalftoneLayerControls(appRef) {
  const app = appRef || window.App;
  const lyr = getActiveHalftoneLayer(app);
  const opacityR = document.getElementById("ht-layer-opacity");
  const opacityN = document.getElementById("ht-layer-opacity-num");
  const blendSel = document.getElementById("ht-layer-blend");
  const disabled = !lyr;

  [opacityR, opacityN, blendSel].forEach((el) => {
    if (el) el.disabled = disabled;
  });
  if (!lyr) return;

  const pct = Math.round((typeof lyr.opacity === "number" ? lyr.opacity : 1.0) * 100);
  if (opacityR) opacityR.value = pct;
  if (opacityN) opacityN.value = pct;
  if (blendSel) blendSel.value = lyr.blend_mode || "normal";
}

function renderHalftoneLayersUI(appRef) {
  const app = appRef || window.App;
  const listEl = document.getElementById("ht-layers-list");
  const st = app?.state?.halftone;
  if (!listEl || !st) return;

  const layers = Array.isArray(st.layers) ? st.layers : [];
  const blendLabels = {
    normal: "Normal",
    multiply: "Multiply",
    screen: "Screen",
    overlay: "Overlay",
    darken: "Darken",
    lighten: "Lighten",
  };

  listEl.innerHTML = layers.slice().reverse().map((layer) => {
    const isOriginal = Boolean(layer.isOriginal);
    const isAct = layer.id === st.activeLayerId && !isOriginal;
    const index = layers.indexOf(layer);
    const canUp = !isOriginal && index < layers.length - 1;
    const canDown = !isOriginal && index > 1;
    const badges = [];
    if (isOriginal) badges.push('<span class="ht-layer-tag ht-layer-tag-lock">🔒 оригинал</span>');
    if (layer.isBase) badges.push('<span class="ht-layer-tag ht-layer-tag-base">база</span>');
    if (layer.hasMask) badges.push('<span class="ht-layer-tag ht-layer-tag-mask">маска</span>');
    const meta = isOriginal
      ? "Фон / база сравнения"
      : `${blendLabels[layer.blend_mode] || "Normal"} · ${Math.round((typeof layer.opacity === "number" ? layer.opacity : 1.0) * 100)}%`;

    return `
      <div class="ht-layer-item ${isAct ? "active" : ""} ${isOriginal ? "locked" : ""}" onclick="App.selectHalftoneLayer('${layer.id}')" title="${escapeHtml(meta)}">
        <div class="ht-layer-left">
          ${isOriginal
            ? '<span class="ht-layer-lock">🔒</span>'
            : `<input type="checkbox" ${layer.visible !== false ? "checked" : ""} onclick="event.stopPropagation(); App.toggleHalftoneLayerVisible('${layer.id}')">`}
          <span class="ht-layer-name">${escapeHtml(layer.name)}</span>
          ${badges.join("")}
        </div>
        <div class="ht-layer-actions">
          ${isOriginal ? "" : `
            <button class="dock-ctrl-btn" onclick="event.stopPropagation(); App.moveHalftoneLayer('up', '${layer.id}')" title="Переместить слой вверх" ${canUp ? "" : "disabled"}>⬆</button>
            <button class="dock-ctrl-btn" onclick="event.stopPropagation(); App.moveHalftoneLayer('down', '${layer.id}')" title="Переместить слой вниз" ${canDown ? "" : "disabled"}>⬇</button>
            <button class="dock-ctrl-btn" onclick="event.stopPropagation(); App.duplicateHalftoneLayer('${layer.id}')" title="Дублировать слой">📋</button>
            <button class="dock-ctrl-btn" onclick="event.stopPropagation(); App.removeHalftoneLayer('${layer.id}')" title="Удалить слой">✕</button>
          `}
        </div>
      </div>
    `;
  }).join("");

  syncHalftoneLayerControls(app);
}

function selectHalftoneLayer(id, appRef) {
  const app = appRef || window.App;
  const st = app?.state?.halftone;
  if (!st) return;
  const lyr = (st.layers || []).find((l) => l.id === id);
  if (!lyr || lyr.isOriginal) {
    renderHalftoneLayersUI(app);
    return;
  }

  st.activeLayerId = id;
  st.params = { ...(lyr.params || st.params || {}) };
  syncHalftoneControlsFromParams(st.params);
  renderHalftoneLayersUI(app);
  triggerHalftoneLiveUpdate(app);
}

function addHalftoneLayer(appRef) {
  const app = appRef || window.App;
  const st = app?.state?.halftone;
  if (!st) return;
  const rasterCount = (st.layers || []).filter((l) => l && !l.isOriginal).length;
  const newLyr = createHalftoneLayer(`Слой ${rasterCount + 1}`, { ...(st.params || {}) }, { hasMask: true });
  ensureLayerMaskCanvas(newLyr, false, app);
  newLyr.hasMask = true;
  newLyr.maskDirty = true;
  st.layers.push(newLyr);
  selectHalftoneLayer(newLyr.id, app);
}

function duplicateHalftoneLayer(id, appRef) {
  const app = appRef || window.App;
  const st = app?.state?.halftone;
  if (!st) return;
  const source = id
    ? (st.layers || []).find((l) => l.id === id)
    : getActiveHalftoneLayer(app);
  if (!source || source.isOriginal) {
    toast("warning", "Исходное фото нельзя дублировать");
    return;
  }

  const copy = createHalftoneLayer(`${source.name} (копия)`, source.params, {
    hasMask: Boolean(source.hasMask || source.maskCanvas),
  });
  copy.visible = source.visible !== false;
  copy.opacity = typeof source.opacity === "number" ? source.opacity : 1.0;
  copy.blend_mode = source.blend_mode || "normal";

  if (source.maskCanvas) {
    const srcCanvas = ensureLayerMaskCanvas(source, false, app);
    const dstCanvas = ensureLayerMaskCanvas(copy, false, app);
    dstCanvas.width = srcCanvas.width;
    dstCanvas.height = srcCanvas.height;
    const ctx = dstCanvas.getContext("2d");
    ctx.clearRect(0, 0, dstCanvas.width, dstCanvas.height);
    ctx.drawImage(srcCanvas, 0, 0);
    copy.hasMask = true;
    copy.maskDirty = true;
  } else {
    copy.hasMask = false;
  }

  const srcIndex = st.layers.indexOf(source);
  st.layers.splice(srcIndex + 1, 0, copy);
  selectHalftoneLayer(copy.id, app);
}

function removeHalftoneLayer(id, appRef) {
  const app = appRef || window.App;
  const st = app?.state?.halftone;
  if (!st) return;
  const target = id
    ? (st.layers || []).find((l) => l.id === id)
    : getActiveHalftoneLayer(app);
  if (!target) return;
  if (target.isOriginal) {
    toast("warning", "Исходное фото заблокировано и не может быть удалено");
    return;
  }
  if ((st.layers || []).filter((l) => l && !l.isOriginal).length <= 1) {
    toast("warning", "Нельзя удалить единственный растровый слой");
    return;
  }

  if (target._glMaskTexture && glContext) {
    glContext.deleteTexture(target._glMaskTexture);
    target._glMaskTexture = null;
  }

  const index = st.layers.indexOf(target);
  st.layers = st.layers.filter((l) => l !== target);
  if (!st.layers.some((l) => !l.isOriginal && l.isBase)) {
    const firstRaster = st.layers.find((l) => !l.isOriginal);
    if (firstRaster) firstRaster.isBase = true;
  }

  const fallback = st.layers[Math.min(index, st.layers.length - 1)];
  const next = fallback && !fallback.isOriginal ? fallback : st.layers.find((l) => !l.isOriginal);
  st.activeLayerId = next ? next.id : null;
  if (next) st.params = { ...(next.params || st.params) };
  syncHalftoneControlsFromParams(st.params);
  renderHalftoneLayersUI(app);
  triggerHalftoneLiveUpdate(app);
}

function moveHalftoneLayer(direction, id, appRef) {
  const app = appRef || window.App;
  const st = app?.state?.halftone;
  if (!st) return;
  const layer = id
    ? (st.layers || []).find((l) => l.id === id)
    : getActiveHalftoneLayer(app);
  if (!layer || layer.isOriginal) return;

  const index = st.layers.indexOf(layer);
  const delta = direction === "up" ? 1 : (direction === "down" ? -1 : 0);
  const target = index + delta;
  if (!delta || target < 1 || target >= st.layers.length) return;

  st.layers[index] = st.layers[target];
  st.layers[target] = layer;
  renderHalftoneLayersUI(app);
  triggerHalftoneLiveUpdate(app);
}

function toggleHalftoneLayerVisible(id, appRef) {
  const app = appRef || window.App;
  const st = app?.state?.halftone;
  if (!st) return;
  const lyr = (st.layers || []).find((l) => l.id === id);
  if (!lyr || lyr.isOriginal) return;
  lyr.visible = !lyr.visible;
  renderHalftoneLayersUI(app);
  triggerHalftoneLiveUpdate(app);
}

function toggleHalftoneCompare(appRef) {
  const app = appRef || window.App;
  const st = app?.state?.halftone;
  if (!st || !st.sourceUrl) return;
  st.comparing = !st.comparing;
  const canvas = document.getElementById("ht-live-canvas");
  const img = document.getElementById("ht-preview-img");
  const badge = document.getElementById("ht-compare-badge");

  const btn = document.getElementById("ht-compare-btn");
  if (btn) btn.classList.toggle("active", st.comparing);

  if (st.comparing) {
    if (canvas) canvas.style.display = "none";
    if (img) {
      img.style.visibility = "visible";
      img.src = st.sourceUrl;
    }
    if (badge) badge.style.display = "block";
  } else {
    if (badge) badge.style.display = "none";
    if (st.qualityMode === "live") {
      if (img) img.style.visibility = "hidden";
      if (canvas) canvas.style.display = "block";
      renderHalftoneWebGL(app);
    } else {
      if (canvas) canvas.style.display = "none";
      if (img) {
        img.style.visibility = "visible";
        img.src = st.resultDataUrl || st.sourceUrl;
      }
    }
  }
}

function revertHalftoneToOriginal(appRef) {
  const app = appRef || window.App;
  const st = app?.state?.halftone;
  if (!st || !st.sourceUrl) return;
  st.comparing = true;
  const canvas = document.getElementById("ht-live-canvas");
  const img = document.getElementById("ht-preview-img");
  const badge = document.getElementById("ht-compare-badge");

  if (canvas) canvas.style.display = "none";
  if (img) {
    img.style.visibility = "visible";
    img.src = st.sourceUrl;
  }
  if (badge) badge.style.display = "block";
}

async function saveHalftoneToGallery(appRef) {
  const app = appRef || window.App;
  const st = app?.state?.halftone;
  if (!st || (!st.sourcePath && !st.sourceDataUrl)) {
    toast("warning", "Нет изображения для сохранения");
    return;
  }

  const loadOverlay = document.getElementById("ht-loading-overlay");
  if (loadOverlay) loadOverlay.style.display = "flex";

  // If WebGL2 GPU Live mode is active, export directly from canvas for 100% WYSIWYG
  if (st.qualityMode === "live") {
    const canvas = document.getElementById("ht-live-canvas");
    if (canvas && canvas.width > 0) {
      try {
        canvas.toBlob((blob) => {
          if (!blob) {
            toast("error", "Ошибка создания PNG");
            return;
          }
          const url = URL.createObjectURL(blob);
          const a = document.createElement("a");
          a.href = url;
          a.download = `halftone_gt_gpu_${Date.now()}.png`;
          document.body.appendChild(a);
          a.click();
          document.body.removeChild(a);
          setTimeout(() => URL.revokeObjectURL(url), 1000);
          toast("success", "PNG сохранён без рассинхрона (100% WYSIWYG WebGL2)");
        }, "image/png");
        if (loadOverlay) loadOverlay.style.display = "none";
        return;
      } catch (e) {
        console.warn("Direct canvas export fallback:", e);
      }
    }
  }

  // If Full-Res CPU mode has rendered dataUrl, download immediately
  if (st.qualityMode === "full" && st.resultDataUrl) {
    try {
      const a = document.createElement("a");
      a.href = st.resultDataUrl;
      a.download = `halftone_gt_cpu_${Date.now()}.png`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      toast("success", "PNG сохранён (100% WYSIWYG CPU)");
      if (loadOverlay) loadOverlay.style.display = "none";
      return;
    } catch (_) {}
  }

  // If WebGL2 GPU Live mode is active, pass the exact canvas dataUrl to gallery
  let liveDataUrl = null;
  if (st.qualityMode === "live") {
    const canvas = document.getElementById("ht-live-canvas");
    if (canvas && canvas.width > 0) {
      try { liveDataUrl = canvas.toDataURL("image/png"); } catch (_) {}
    }
  }

  try {
    const layersPayload = buildHalftoneLayersPayload(st);
    const res = await apiPost("/api/halftone/process", {
      image_path: liveDataUrl ? null : st.sourcePath,
      image_data: liveDataUrl || st.resultDataUrl || st.sourceDataUrl,
      params: st.params,
      layers: layersPayload,
      save: true,
      preview_max_side: 0,
    });
    if (res.saved) {
      toast("success", `Сохранено в галерею: ${res.saved.name} ✓`);
    }
  } catch (e) {
    toast("error", "Ошибка сохранения: " + e.message);
  } finally {
    if (loadOverlay) loadOverlay.style.display = "none";
  }
}

async function downloadHalftonePng(appRef) {
  const app = appRef || window.App;
  const st = app?.state?.halftone;
  if (!st || (!st.sourcePath && !st.sourceDataUrl)) {
    toast("warning", "Сначала выберите или загрузите изображение");
    return;
  }

  const loadOverlay = document.getElementById("ht-loading-overlay");
  if (loadOverlay) loadOverlay.style.display = "flex";
  toast("info", "Рендеринг в полном разрешении для скачивания...");

  try {
    const layersPayload = buildHalftoneLayersPayload(st);
    const res = await apiPost("/api/halftone/process", {
      image_path: st.sourcePath,
      image_data: st.sourceDataUrl,
      params: st.params,
      layers: layersPayload,
      save: false,
      preview_max_side: 0,
    });

    const src = res?.preview_data_url || st.resultDataUrl;
    if (!src) throw new Error("Не удалось получить результат рендеринга");

    const a = document.createElement("a");
    a.href = src;
    a.download = `halftone_${res.width || "full"}x${res.height || "res"}_${Date.now()}.png`;
    a.click();
    toast("success", `Скачано в оригинальном разрешении ${res.width}×${res.height} px ✓`);
  } catch (e) {
    toast("error", "Ошибка скачивания: " + e.message);
  } finally {
    if (loadOverlay) loadOverlay.style.display = "none";
  }
}

function syncHalftoneMaskCanvasSize(w, h, appRef = null) {
  const app = appRef || window.App;
  const st = app?.state?.halftone;
  if (!st || !w || !h) return;
  const maskCanvas = document.getElementById("ht-mask-canvas");
  const lassoOverlay = document.getElementById("ht-lasso-overlay");
  if (maskCanvas && (maskCanvas.width !== w || maskCanvas.height !== h)) {
    maskCanvas.width = w;
    maskCanvas.height = h;
  }
  if (lassoOverlay && (lassoOverlay.width !== w || lassoOverlay.height !== h)) {
    lassoOverlay.width = w;
    lassoOverlay.height = h;
  }
}

function ensureLayerMaskCanvas(lyr, fillWhiteIfNew = false, appRef = null) {
  const app = appRef || window.App;
  const st = app?.state?.halftone;
  const w = st?.origWidth || st?.liveWidth || 800;
  const h = st?.origHeight || st?.liveHeight || 800;
  if (!lyr.maskCanvas) {
    lyr.maskCanvas = document.createElement("canvas");
    lyr.maskCanvas.width = w;
    lyr.maskCanvas.height = h;
    const ctx = lyr.maskCanvas.getContext("2d");
    if (fillWhiteIfNew) {
      ctx.fillStyle = "#ffffff";
      ctx.fillRect(0, 0, w, h);
    } else {
      ctx.clearRect(0, 0, w, h);
    }
  } else if (lyr.maskCanvas.width !== w || lyr.maskCanvas.height !== h) {
    const old = lyr.maskCanvas;
    const next = document.createElement("canvas");
    next.width = w;
    next.height = h;
    const nCtx = next.getContext("2d");
    if (fillWhiteIfNew && !lyr.hasMask) {
      nCtx.fillStyle = "#ffffff";
      nCtx.fillRect(0, 0, w, h);
    } else {
      nCtx.drawImage(old, 0, 0, w, h);
    }
    lyr.maskCanvas = next;
  }
  return lyr.maskCanvas;
}

function renderMaskOverlayVisual(lyr, appRef = null) {
  const app = appRef || window.App;
  const st = app?.state?.halftone;
  const maskCanvas = document.getElementById("ht-mask-canvas");
  const badge = document.getElementById("ht-selection-badge");
  if (!maskCanvas) return;
  const ctx = maskCanvas.getContext("2d");
  ctx.clearRect(0, 0, maskCanvas.width, maskCanvas.height);

  const hasMask = Boolean(lyr && lyr.hasMask && lyr.maskCanvas);
  if (badge) {
    badge.textContent = hasMask ? "Маска активна" : "Весь холст";
    badge.style.color = hasMask ? "var(--accent, #a855f7)" : "var(--text-dim)";
  }

  const showOverlay = st?.showMaskOverlay !== false;
  if (!showOverlay || !hasMask) {
    return;
  }

  ctx.save();
  ctx.drawImage(lyr.maskCanvas, 0, 0, maskCanvas.width, maskCanvas.height);
  ctx.globalCompositeOperation = "source-in";
  ctx.fillStyle = "rgba(239, 68, 68, 0.45)";
  ctx.fillRect(0, 0, maskCanvas.width, maskCanvas.height);
  ctx.restore();
}

function initHalftoneMaskEvents(appRef = null) {
  const app = appRef || window.App;
  const maskCanvas = document.getElementById("ht-mask-canvas");
  const lassoOverlay = document.getElementById("ht-lasso-overlay");
  if (!maskCanvas || !lassoOverlay || maskCanvas._maskBound) return;
  maskCanvas._maskBound = true;

  const getPos = (e, canvas) => {
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / Math.max(1, rect.width);
    const scaleY = canvas.height / Math.max(1, rect.height);
    return [
      (e.clientX - rect.left) * scaleX,
      (e.clientY - rect.top) * scaleY
    ];
  };

  let isPainting = false;
  let lastPos = null;

  maskCanvas.addEventListener("pointerdown", (e) => {
    const st = app?.state?.halftone;
    if (!st || (st.tool !== "brush" && st.tool !== "eraser")) return;
    const lyr = getActiveHalftoneLayer(app);
    if (!lyr) return;

    const fillWhite = (st.tool === "eraser" && !lyr.hasMask);
    const mCanvas = ensureLayerMaskCanvas(lyr, fillWhite, app);

    isPainting = true;
    lastPos = getPos(e, maskCanvas);
    try { maskCanvas.setPointerCapture(e.pointerId); } catch (_) {}

    const ctx = mCanvas.getContext("2d");
    const brushRadius = Math.max(2, ((st.brushSize || 24) * (mCanvas.width / 1000.0)) * 0.5);
    ctx.lineWidth = brushRadius * 2.0;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";

    if (st.tool === "brush") {
      ctx.globalCompositeOperation = "source-over";
      ctx.strokeStyle = "#ffffff";
      ctx.fillStyle = "#ffffff";
    } else {
      ctx.globalCompositeOperation = "destination-out";
      ctx.strokeStyle = "rgba(0,0,0,1)";
      ctx.fillStyle = "rgba(0,0,0,1)";
    }

    ctx.beginPath();
    ctx.arc(lastPos[0], lastPos[1], brushRadius, 0, Math.PI * 2);
    ctx.fill();

    lyr.hasMask = true;
    lyr.maskDirty = true;
    renderMaskOverlayVisual(lyr, app);
    triggerHalftoneLiveUpdate(app);
  });

  maskCanvas.addEventListener("pointermove", (e) => {
    if (!isPainting) return;
    const st = app?.state?.halftone;
    const lyr = getActiveHalftoneLayer(app);
    if (!lyr || !lyr.maskCanvas) return;
    const pos = getPos(e, maskCanvas);
    const ctx = lyr.maskCanvas.getContext("2d");
    ctx.beginPath();
    ctx.moveTo(lastPos[0], lastPos[1]);
    ctx.lineTo(pos[0], pos[1]);
    ctx.stroke();
    lastPos = pos;
    lyr.maskDirty = true;
    renderMaskOverlayVisual(lyr, app);
    triggerHalftoneLiveUpdate(app);
  });

  const stopPaint = (e) => {
    if (!isPainting) return;
    isPainting = false;
    try { maskCanvas.releasePointerCapture(e.pointerId); } catch (_) {}
    const lyr = getActiveHalftoneLayer(app);
    if (lyr) {
      lyr.maskDirty = true;
      renderMaskOverlayVisual(lyr, app);
      triggerHalftoneLiveUpdate(app);
    }
  };
  maskCanvas.addEventListener("pointerup", stopPaint);
  maskCanvas.addEventListener("pointercancel", stopPaint);

  let isLassoing = false;
  let lassoPts = [];

  lassoOverlay.addEventListener("pointerdown", (e) => {
    const st = app?.state?.halftone;
    if (!st || st.tool !== "lasso") return;
    const lyr = getActiveHalftoneLayer(app);
    if (!lyr) return;

    isLassoing = true;
    lassoPts = [getPos(e, lassoOverlay)];
    try { lassoOverlay.setPointerCapture(e.pointerId); } catch (_) {}
  });

  lassoOverlay.addEventListener("pointermove", (e) => {
    if (!isLassoing) return;
    lassoPts.push(getPos(e, lassoOverlay));
    const ctx = lassoOverlay.getContext("2d");
    ctx.clearRect(0, 0, lassoOverlay.width, lassoOverlay.height);
    ctx.strokeStyle = "#a855f7";
    ctx.lineWidth = 2.5;
    ctx.setLineDash([6, 6]);
    ctx.beginPath();
    ctx.moveTo(lassoPts[0][0], lassoPts[0][1]);
    for (let i = 1; i < lassoPts.length; i++) {
      ctx.lineTo(lassoPts[i][0], lassoPts[i][1]);
    }
    ctx.stroke();
    ctx.setLineDash([]);
  });

  const stopLasso = (e) => {
    if (!isLassoing) return;
    isLassoing = false;
    try { lassoOverlay.releasePointerCapture(e.pointerId); } catch (_) {}
    const ctx = lassoOverlay.getContext("2d");
    ctx.clearRect(0, 0, lassoOverlay.width, lassoOverlay.height);

    if (lassoPts.length > 2) {
      const lyr = getActiveHalftoneLayer(app);
      if (lyr) {
        const mode = app?.state?.halftone?.lassoMode || "replace";
        const fillWhite = (mode === "subtract" && !lyr.hasMask);
        const mCanvas = ensureLayerMaskCanvas(lyr, fillWhite, app);
        const mCtx = mCanvas.getContext("2d");

        if (mode === "replace") {
          mCtx.clearRect(0, 0, mCanvas.width, mCanvas.height);
          mCtx.globalCompositeOperation = "source-over";
          mCtx.fillStyle = "#ffffff";
          mCtx.beginPath();
          mCtx.moveTo(lassoPts[0][0], lassoPts[0][1]);
          for (let i = 1; i < lassoPts.length; i++) mCtx.lineTo(lassoPts[i][0], lassoPts[i][1]);
          mCtx.closePath();
          mCtx.fill();
        } else if (mode === "add") {
          mCtx.globalCompositeOperation = "source-over";
          mCtx.fillStyle = "#ffffff";
          mCtx.beginPath();
          mCtx.moveTo(lassoPts[0][0], lassoPts[0][1]);
          for (let i = 1; i < lassoPts.length; i++) mCtx.lineTo(lassoPts[i][0], lassoPts[i][1]);
          mCtx.closePath();
          mCtx.fill();
        } else if (mode === "subtract") {
          mCtx.globalCompositeOperation = "destination-out";
          mCtx.fillStyle = "rgba(0,0,0,1)";
          mCtx.beginPath();
          mCtx.moveTo(lassoPts[0][0], lassoPts[0][1]);
          for (let i = 1; i < lassoPts.length; i++) mCtx.lineTo(lassoPts[i][0], lassoPts[i][1]);
          mCtx.closePath();
          mCtx.fill();
        } else if (mode === "intersect") {
          if (!lyr.hasMask) {
            mCtx.clearRect(0, 0, mCanvas.width, mCanvas.height);
            mCtx.globalCompositeOperation = "source-over";
            mCtx.fillStyle = "#ffffff";
            mCtx.beginPath();
            mCtx.moveTo(lassoPts[0][0], lassoPts[0][1]);
            for (let i = 1; i < lassoPts.length; i++) mCtx.lineTo(lassoPts[i][0], lassoPts[i][1]);
            mCtx.closePath();
            mCtx.fill();
          } else {
            const tC = document.createElement("canvas");
            tC.width = mCanvas.width;
            tC.height = mCanvas.height;
            const tCtx = tC.getContext("2d");
            tCtx.fillStyle = "#ffffff";
            tCtx.beginPath();
            tCtx.moveTo(lassoPts[0][0], lassoPts[0][1]);
            for (let i = 1; i < lassoPts.length; i++) tCtx.lineTo(lassoPts[i][0], lassoPts[i][1]);
            tCtx.closePath();
            tCtx.fill();

            mCtx.globalCompositeOperation = "destination-in";
            mCtx.drawImage(tC, 0, 0);
            mCtx.globalCompositeOperation = "source-over";
          }
        }
        lyr.hasMask = true;
        lyr.maskDirty = true;
        renderMaskOverlayVisual(lyr, app);
        triggerHalftoneLiveUpdate(app);
      }
    }
    lassoPts = [];
  };
  lassoOverlay.addEventListener("pointerup", stopLasso);
  lassoOverlay.addEventListener("pointercancel", stopLasso);
}

function setHalftoneTool(tool, appRef = null) {
  const app = appRef || window.App;
  initHalftoneStudio(app);
  const st = app?.state?.halftone;
  if (!st) return;
  st.tool = tool || "view";

  const toolIds = ["ht-tool-view", "ht-tool-lasso", "ht-tool-brush", "ht-tool-eraser"];
  toolIds.forEach(id => {
    const btn = document.getElementById(id);
    if (btn) btn.classList.toggle("active", id === `ht-tool-${st.tool}`);
  });

  const wrap = document.getElementById("ht-preview-wrap");
  const maskCanvas = document.getElementById("ht-mask-canvas");
  const lassoOverlay = document.getElementById("ht-lasso-overlay");

  if (wrap) {
    wrap.style.cursor = st.tool === "view" ? "grab" : "crosshair";
  }

  const isMaskPaint = (st.tool === "brush" || st.tool === "eraser");
  const isLasso = (st.tool === "lasso");
  const showMask = (st.showMaskOverlay !== false);

  if (maskCanvas) {
    maskCanvas.style.display = (showMask || isMaskPaint) ? "block" : "none";
    maskCanvas.style.cursor = isMaskPaint ? "crosshair" : "default";
    maskCanvas.style.pointerEvents = isMaskPaint ? "auto" : "none";
  }

  if (lassoOverlay) {
    lassoOverlay.style.display = isLasso ? "block" : "none";
    lassoOverlay.style.cursor = isLasso ? "crosshair" : "default";
    lassoOverlay.style.pointerEvents = isLasso ? "auto" : "none";
  }

  initHalftoneMaskEvents(app);
  renderMaskOverlayVisual(getActiveHalftoneLayer(app), app);
  toast("info", `Инструмент: ${st.tool}`);
}

function setHalftoneBrushSize(size, appRef = null) {
  const app = appRef || window.App;
  initHalftoneStudio(app);
  const val = parseInt(size, 10) || 24;
  if (app?.state?.halftone) app.state.halftone.brushSize = val;
  const sR = document.getElementById("ht-mask-brush");
  const sN = document.getElementById("ht-mask-brush-num");
  if (sR && sR.value != val) sR.value = val;
  if (sN && sN.value != val) sN.value = val;
}

function setHalftoneLassoMode(mode, appRef = null) {
  const app = appRef || window.App;
  initHalftoneStudio(app);
  const m = mode || "replace";
  if (app?.state?.halftone) app.state.halftone.lassoMode = m;

  ["replace", "add", "subtract", "intersect"].forEach(id => {
    const btn = document.getElementById(`ht-sel-${id}`);
    if (btn) btn.classList.toggle("active", id === m);
  });

  if (app?.state?.halftone?.tool === "view") {
    setHalftoneTool("lasso", app);
  }
}

function clearHalftoneMask(appRef = null) {
  const app = appRef || window.App;
  initHalftoneStudio(app);
  const lyr = getActiveHalftoneLayer(app);
  if (!lyr) return;

  if (lyr.maskCanvas) {
    const ctx = lyr.maskCanvas.getContext("2d");
    ctx.clearRect(0, 0, lyr.maskCanvas.width, lyr.maskCanvas.height);
  }
  if (lyr.isBase) {
    // Base raster returns to full-canvas coverage.
    lyr.hasMask = false;
  } else {
    // Local layers stay masked; an empty mask means "not painted yet".
    ensureLayerMaskCanvas(lyr, false, app);
    lyr.hasMask = true;
  }
  lyr.maskDirty = true;

  const maskCanvas = document.getElementById("ht-mask-canvas");
  if (maskCanvas) {
    const ctx = maskCanvas.getContext("2d");
    ctx.clearRect(0, 0, maskCanvas.width, maskCanvas.height);
  }
  renderMaskOverlayVisual(lyr, app);
  renderHalftoneLayersUI(app);
  triggerHalftoneLiveUpdate(app);
  toast("info", lyr.isBase ? "Маска снята (скринтон на весь холст)" : "Маска очищена (локальный слой скрыт)");
}

function invertHalftoneMaskSelection(appRef = null) {
  const app = appRef || window.App;
  initHalftoneStudio(app);
  const lyr = getActiveHalftoneLayer(app);
  if (lyr && lyr.maskCanvas && lyr.hasMask) {
    const ctx = lyr.maskCanvas.getContext("2d");
    const imgData = ctx.getImageData(0, 0, lyr.maskCanvas.width, lyr.maskCanvas.height);
    const d = imgData.data;
    for (let i = 0; i < d.length; i += 4) {
      const newA = 255 - d[i + 3];
      d[i] = newA;
      d[i + 1] = newA;
      d[i + 2] = newA;
      d[i + 3] = newA;
    }
    ctx.putImageData(imgData, 0, 0);
    lyr.maskDirty = true;
    renderMaskOverlayVisual(lyr, app);
    triggerHalftoneLiveUpdate(app);
    toast("info", "Выделение маски инвертировано");
  } else if (app?.state?.halftone?.params) {
    const p = app.state.halftone.params;
    p.invert_mask = !p.invert_mask;
    const chk = document.getElementById("ht-invert-mask");
    if (chk) chk.checked = p.invert_mask;
    triggerHalftoneLiveUpdate(app);
    toast("info", "Инверсия маски слоя переключена");
  }
}

function toggleHalftoneMaskOverlay(checked, appRef = null) {
  const app = (typeof checked === "object" && checked && checked.state) ? checked : (appRef || window.App);
  initHalftoneStudio(app);
  const st = app?.state?.halftone;
  if (!st) return;
  st.showMaskOverlay = (typeof checked === "boolean") ? checked : !st.showMaskOverlay;
  const chk = document.getElementById("ht-show-mask-overlay");
  if (chk) chk.checked = st.showMaskOverlay;
  setHalftoneTool(st.tool, app);
}

function toggleHalftoneMaskMode(checked, appRef = null) {
  const app = (typeof checked === "object" && checked && checked.state) ? checked : (appRef || window.App);
  initHalftoneStudio(app);
  const isChecked = (typeof checked === "boolean") ? checked : true;
  setHalftoneTool(isChecked ? "brush" : "view", app);
}

function toggleHalftoneMaskEraser(appRef = null) {
  setHalftoneTool("eraser", appRef);
}

function resetHalftoneParams(appRef) {
  const app = appRef || window.App;
  initHalftoneStudio(app);
  const st = app?.state?.halftone;
  if (!st) return;
  const mode = st.params.mode || "manga";
  const def = halftonePresets[mode] || halftonePresets.manga;
  st.params = { ...def };
  const lyr = getActiveHalftoneLayer(app);
  if (lyr) lyr.params = { ...st.params };
  syncHalftoneControlsFromParams(st.params);
  triggerHalftoneLiveUpdate(app);
  toast("info", "Параметры сброшены до значений пресета");
}

function onHalftoneFileUpload(input, appRef) {
  const app = appRef || window.App;
  const file = input?.files?.[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = (e) => {
    loadHalftoneSource(e.target.result, file.name, app);
  };
  reader.readAsDataURL(file);
  input.value = "";
}

function loadHalftoneSource(dataUrl, filename = "image.png", appRef) {
  const app = appRef || window.App;
  initHalftoneStudio(app);
  const st = app.state.halftone;
  st.sourceDataUrl = dataUrl;
  st.sourceUrl = dataUrl;
  st.sourcePath = null;
  resetHalftoneLayerStack(app);

  const emptyEl = document.getElementById("ht-empty-state");
  const wrapEl = document.getElementById("ht-preview-wrap");
  const imgEl = document.getElementById("ht-preview-img");
  if (emptyEl) emptyEl.style.display = "none";
  if (wrapEl) wrapEl.style.display = "inline-flex";
  if (imgEl) imgEl.src = dataUrl;

  const metaEl = document.getElementById("ht-meta-info");
  if (metaEl) metaEl.textContent = `Файл: ${filename}`;

  prepareHalftoneLiveSource(dataUrl, app);
  runHalftonePreview(true, app);
}

function onStudioDrop(e, target, appRef) {
  e.preventDefault();
  e.stopPropagation();
  const app = appRef || window.App;
  const files = e.dataTransfer?.files;
  if (!files || !files.length) return;
  const file = files[0];
  if (!file.type.startsWith("image/")) {
    toast("warning", "Пожалуйста, перетащите файл изображения (PNG, JPG, WEBP)");
    return;
  }
  const reader = new FileReader();
  reader.onload = (ev) => {
    const dataUrl = ev.target.result;
    if (target === "halftone") {
      loadHalftoneSource(dataUrl, file.name, app);
    } else if (target === "manga") {
      if (app.loadMangaSource) app.loadMangaSource(dataUrl, file.name);
    } else {
      if (app.uploadImg2ImgFile) app.uploadImg2ImgFile(file);
    }
  };
  reader.readAsDataURL(file);
}

function sendHalftoneToManga(appRef) {
  const app = appRef || window.App;
  const st = app?.state?.halftone;
  const src = st?.resultDataUrl || st?.sourceDataUrl;
  if (!src) {
    toast("warning", "Нет результата Halftone Studio для передачи в Manga Screen AI");
    return;
  }
  app.switchTab("manga");
  if (app.loadMangaSource) {
    app.loadMangaSource(src, "halftone_export.png");
  }
  toast("success", "Результат Halftone Studio передан в Manga Screen AI ✓");
}

let _galleryPickerTarget = "halftone";

function onHalftoneLayerMetaChange(prop, val, appRef) {
  const app = appRef || window.App;
  initHalftoneStudio(app);
  const st = app?.state?.halftone;
  const lyr = getActiveHalftoneLayer(app);
  if (!st || !lyr) return;
  if (prop === "opacity") {
    const v = parseFloat(val) / 100.0;
    lyr.opacity = isNaN(v) ? 1.0 : Math.max(0, Math.min(1, v));
  } else if (prop === "blend_mode") {
    lyr.blend_mode = val || "normal";
    if (lyr.blend_mode === "multiply") {
      applyHalftoneMoireProtection(lyr, st, app);
    }
  }
  syncHalftoneLayerControls(app);
  renderHalftoneLayersUI(app);
  triggerHalftoneLiveUpdate(app);
}

function openGalleryPicker(targetTool = "halftone", appRef) {
  const app = appRef || window.App;
  _galleryPickerTarget = targetTool;
  const modal = document.getElementById("gallery-picker-modal");
  const grid = document.getElementById("gallery-picker-grid");
  const title = document.getElementById("gallery-picker-title");
  if (!modal || !grid) return;

  if (title) {
    title.textContent = targetTool === "halftone"
      ? "Выберите изображение для Halftone Studio GT"
      : "Выберите изображение для Manga Screen AI";
  }

  const items = app?.state?.historyItems || [];
  if (!items.length) {
    grid.innerHTML = '<div style="grid-column: 1/-1; padding: 24px; text-align: center; color: var(--text-dim);">Галерея генераций пуста</div>';
  } else {
    grid.innerHTML = items.map(it => {
      const thumbUrl = it.thumb_url || it.image_url || it.url;
      const fullUrl = it.image_url || it.url || thumbUrl;
      return `
        <div class="gallery-picker-item" onclick="App.selectGalleryPickerItem('${escapeHtml(fullUrl)}', '${escapeHtml(it.filename || '')}')" style="cursor: pointer; border-radius: 8px; overflow: hidden; border: 1px solid var(--border); aspect-ratio: 1; background: var(--bg-surface); transition: transform 0.15s, border-color 0.15s;">
          <img src="${thumbUrl}" alt="History item" style="width: 100%; height: 100%; object-fit: cover; display: block;" loading="lazy">
        </div>
      `;
    }).join("");
  }

  modal.style.display = "flex";
}

function closeGalleryPicker() {
  const modal = document.getElementById("gallery-picker-modal");
  if (modal) modal.style.display = "none";
}

function selectGalleryPickerItem(url, filename, appRef) {
  closeGalleryPicker();
  const app = appRef || window.App;
  const target = _galleryPickerTarget || "halftone";
  if (target === "halftone") {
    loadHalftoneSource(url, filename || "gallery_item.png", app);
  } else if (target === "manga") {
    if (app.loadMangaSource) app.loadMangaSource(url, filename || "gallery_item.png");
  }
}

function openHistoryItemInTool(tool, appRef) {
  const app = appRef || window.App;
  const it = app?.state?.selectedHistoryItem;
  if (!it) return;
  const url = it.image_url || it.url || it.thumb_url;
  if (!url) return;
  if (app.closeHistoryModal) app.closeHistoryModal();
  app.switchTab(tool);
  if (tool === "halftone") {
    loadHalftoneSource(url, it.filename || "history_image.png", app);
  } else if (tool === "manga") {
    if (app.loadMangaSource) app.loadMangaSource(url, it.filename || "history_image.png");
  }
}

function useCanvasImageInTool(tool, appRef) {
  const app = appRef || window.App;
  const curImg = app?.state?.lastPreviewUrl || document.getElementById("preview-img")?.src;
  if (!curImg) {
    toast("warning", "Нет сгенерированного изображения для передачи");
    return;
  }
  app.switchTab(tool);
  if (tool === "halftone") {
    loadHalftoneSource(curImg, "generated_preview.png", app);
  } else if (tool === "manga") {
    if (app.loadMangaSource) app.loadMangaSource(curImg, "generated_preview.png");
  }
}


  Object.assign(App, {
    halftoneBaseExtraParams,
    halftonePresets,
    _htBuffers,
    createHalftoneLayer,
    initHalftoneStudio,
    setHalftoneQualityMode,
    prepareHalftoneLiveSource,
    triggerHalftoneLiveUpdate,
    renderHalftoneWebGL,
    runHalftonePreview,
    selectHalftoneMode,
    syncHalftoneControlsFromParams,
    onHalftoneParamInput,
    getActiveHalftoneLayer,
    renderHalftoneLayersUI,
    selectHalftoneLayer,
    addHalftoneLayer,
    duplicateHalftoneLayer,
    removeHalftoneLayer,
    moveHalftoneLayer,
    syncHalftoneLayerControls,
    toggleHalftoneLayerVisible,
    toggleHalftoneCompare,
    revertHalftoneToOriginal,
    saveHalftoneToGallery,
    downloadHalftonePng,
    syncHalftoneMaskCanvasSize,
    ensureLayerMaskCanvas,
    renderMaskOverlayVisual,
    initHalftoneMaskEvents,
    setHalftoneTool,
    setHalftoneBrushSize,
    setHalftoneLassoMode,
    clearHalftoneMask,
    invertHalftoneMaskSelection,
    toggleHalftoneMaskOverlay,
    toggleHalftoneMaskMode,
    toggleHalftoneMaskEraser,
    resetHalftoneParams,
    onHalftoneLayerMetaChange,
    onHalftoneFileUpload,
    loadHalftoneSource,
    onStudioDrop,
    sendHalftoneToManga,
    openGalleryPicker,
    closeGalleryPicker,
    selectGalleryPickerItem,
    openHistoryItemInTool,
    useCanvasImageInTool,
  });
})();

// ============================================================
// --- INTEGRATED MANGA SCREEN AI ENGINE ---
// ============================================================
(function() {
  const apiPost = (url, body) => App.apiPost(url, body);
  const apiGet = (url) => App.apiGet(url);
  const toast = (type, msg) => App.toast(type, msg);

/* ============================================================
   QuantFunc WebUI — Manga Studio AI Controller
   B&W stylization, G-pen lineart, AI reference pipeline, compare mode.
   ============================================================ */


function defaultMangaParams() {
  return {
    mode: "manga_lite",
    ai_enabled: false,
    model_id: "",
    variant_id: "",
    lora_name: "",
    lora_strength: 0.85,
    ai_prompt: "black and white Japanese manga illustration, clean G-pen ink lineart, high contrast screentone shading, monochrome comic art",
    ai_negative_prompt: "color, blurry, photo, 3d render, watermark, muddy shading",
    ai_denoise_strength: 0.65,
    ai_steps: 0,
    ai_cfg: 0.0,
    ai_ref_mode: "control_ref",
    levels_enabled: true,
    levels_contrast: 1.25,
    levels_gamma: 1.05,
    levels_brightness: 0.0,
    levels_descreen: 0.20,
    lines_enabled: true,
    line_strength: 1.15,
    line_threshold: 0.28,
    line_thickness: 1,
    line_invert: false,
    finish_enabled: true,
    finish_preset: "clean",
    finish_dot_size: 6,
    finish_dot_shape: "circle",
    finish_angle: 45.0,
    finish_max_radius_ratio: 0.90,
    finish_ink_threshold: 0.28,
    finish_tone_gamma: 1.05,
    finish_light_tone: 0.80,
    finish_mid_tone: 0.50,
    finish_black_point: 0.18,
    finish_region_size: 6,
    finish_gradation: 0.25,
    finish_white_cut: 0.35,
    finish_invert: false,
    panel_enabled: false,
    panel_aspect: "original",
    panel_crop_x: 0.0,
    panel_crop_y: 0.0,
    panel_border: 4,
    panel_margin: 18,
    panel_paper_tone: "off-white",
    panel_print_bleed: 0.15,
    seed: 42,
  };
}

function initMangaStudio(appRef) {
  const app = appRef || window.App;
  if (!app) return;
  if (!app.state.manga) {
    app.state.manga = {
      params: defaultMangaParams(),
      sourcePath: null,
      sourceUrl: null,
      sourceDataUrl: null,
      resultDataUrl: null,
      comparing: false,
      debounceTimer: null,
    };
  }
  populateMangaModelsAndLoras(app);
  syncMangaControlsFromParams(app.state.manga.params);
}

function populateMangaModelsAndLoras(appRef) {
  const app = appRef || window.App;
  if (!app) return;

  const modelSel = document.getElementById("manga-ai-model");
  if (modelSel) {
    const models = (app.state.models || []).filter(m => !m.disabled);
    const curVal = app.state.manga?.params?.model_id || modelSel.value || app.state.selectedModel || "";
    modelSel.innerHTML = models.map(m => {
      const eng = (m.engine || "qf").toUpperCase();
      const inst = m.installed ? "✓ " : "⬇ ";
      return `<option value="${escapeHtml(m.id)}">${inst}${escapeHtml(m.name)} [${eng}]</option>`;
    }).join("");

    if (curVal && models.some(m => m.id === curVal)) {
      modelSel.value = curVal;
      if (app.state.manga) app.state.manga.params.model_id = curVal;
    } else if (models.length) {
      modelSel.value = models[0].id;
      if (app.state.manga) app.state.manga.params.model_id = models[0].id;
    }
  }

  const loraSel = document.getElementById("manga-ai-lora");
  if (loraSel) {
    const loras = app.state.loraLibrary || [];
    const curLora = app.state.manga?.params?.lora_name || "";
    const noneText = window.t ? (window.t("manga.ai_lora_none") || "Без LoRA (базовая модель)") : "Без LoRA (базовая модель)";
    loraSel.innerHTML = `<option value="">${noneText}</option>` +
      loras.map(l => `<option value="${escapeHtml(l.name)}">${escapeHtml(l.name)} (${escapeHtml(l.family || "flux")})</option>`).join("");
    loraSel.value = curLora;
  }

  checkMangaLoraCompatibility(app);
}

async function onMangaModelChange(modelId, appRef) {
  const app = appRef || window.App;
  initMangaStudio(app);
  if (app.state.manga) app.state.manga.params.model_id = modelId;
  await checkMangaLoraCompatibility(app);
}

async function onMangaLoraChange(loraName, appRef) {
  const app = appRef || window.App;
  initMangaStudio(app);
  if (app.state.manga) app.state.manga.params.lora_name = loraName || "";
  await checkMangaLoraCompatibility(app);
}

async function checkMangaLoraCompatibility(appRef) {
  const app = appRef || window.App;
  const badge = document.getElementById("manga-lora-compat-badge");
  if (!badge || !app?.state?.manga) return;

  const modelId = app.state.manga.params.model_id;
  const loraName = app.state.manga.params.lora_name;
  if (!modelId) {
    badge.textContent = "";
    return;
  }

  const modelObj = (app.state.models || []).find(m => m.id === modelId);
  if (modelObj && modelObj.engine === "sdnq" && !loraName) {
    badge.style.color = "var(--text-dim)";
    badge.textContent = "ℹ️ Движок SDNQ использует мультимодальный Qwen-Image референс без внешних LoRA.";
    return;
  }
  if (!loraName) {
    badge.style.color = "var(--text-dim)";
    badge.textContent = "✓ Работа без LoRA (семантическая стилизация базовой моделью)";
    return;
  }

  try {
    const res = await apiGet(`/api/loras/check-compatibility?model_id=${encodeURIComponent(modelId)}&lora_name=${encodeURIComponent(loraName)}`);
    if (res.compatible) {
      badge.style.color = "#34d399";
      badge.textContent = `✓ Совместимо (${res.model_family || "Flux/Qwen"})`;
    } else {
      badge.style.color = "#f87171";
      badge.textContent = `⚠️ Несовместимо: модель ${res.model_family}, а LoRA для ${res.lora_family}`;
    }
  } catch (_) {
    badge.textContent = "";
  }
}

function selectMangaMode(mode, appRef) {
  const app = appRef || window.App;
  initMangaStudio(app);
  const p = app.state.manga.params;
  p.mode = mode;

  if (mode === "manga_lite") {
    p.ai_enabled = false;
    p.levels_enabled = true;
    p.lines_enabled = true;
    p.finish_enabled = true;
  } else if (mode === "finish_only") {
    p.ai_enabled = false;
    p.levels_enabled = false;
    p.lines_enabled = false;
    p.finish_enabled = true;
  } else if (mode === "ai_manga") {
    p.ai_enabled = true;
    p.levels_enabled = false;
    p.lines_enabled = false;
    p.finish_enabled = false;
  } else if (mode === "ai_manga_finish") {
    p.ai_enabled = true;
    p.levels_enabled = true;
    p.lines_enabled = true;
    p.finish_enabled = true;
  }

  const banner = document.getElementById("manga-fallback-banner");
  if (banner && (mode === "manga_lite" || mode === "finish_only")) {
    banner.style.display = "none";
  }

  syncMangaControlsFromParams(p);
  if (mode === "manga_lite" || mode === "finish_only") {
    runMangaPipeline(false, app);
  }
}

function syncMangaControlsFromParams(p) {
  if (!p) return;
  document.querySelectorAll("#manga-mode-pills .studio-mode-pill").forEach(btn => {
    btn.classList.toggle("active", btn.dataset.mode === p.mode);
  });

  const stageACard = document.getElementById("manga-stage-a-card");
  if (stageACard) {
    stageACard.style.display = (p.mode === "ai_manga" || p.mode === "ai_manga_finish" || p.ai_enabled) ? "flex" : "none";
  }

  const setChk = (id, v) => {
    const el = document.getElementById(id);
    if (el) el.checked = Boolean(v);
  };
  setChk("manga-ai-enabled", p.ai_enabled);
  setChk("manga-levels-enabled", p.levels_enabled);
  setChk("manga-lines-enabled", p.lines_enabled);
  setChk("manga-finish-enabled", p.finish_enabled);
  setChk("manga-panel-enabled", p.panel_enabled);
  setChk("manga-line-invert", p.line_invert);

  const setPair = (rId, nId, val) => {
    const r = document.getElementById(rId);
    const n = document.getElementById(nId);
    if (r && val !== undefined) r.value = val;
    if (n && val !== undefined) n.value = val;
  };
  setPair("manga-ai-lora-weight", "manga-ai-lora-weight-num", p.lora_strength);
  setPair("manga-ai-denoise", "manga-ai-denoise-num", p.ai_denoise_strength);
  setPair("manga-levels-contrast", "manga-levels-contrast-num", p.levels_contrast);
  setPair("manga-levels-gamma", "manga-levels-gamma-num", p.levels_gamma);
  setPair("manga-levels-descreen", "manga-levels-descreen-num", p.levels_descreen);
  setPair("manga-line-strength", "manga-line-strength-num", p.line_strength);
  setPair("manga-line-threshold", "manga-line-threshold-num", p.line_threshold);
  setPair("manga-line-thickness", "manga-line-thickness-num", p.line_thickness);
  setPair("manga-finish-dot-size", "manga-finish-dot-size-num", p.finish_dot_size);
  setPair("manga-finish-ink", "manga-finish-ink-num", p.finish_ink_threshold);
  setPair("manga-finish-light-tone", "manga-finish-light-tone-num", p.finish_light_tone);
  setPair("manga-finish-mid-tone", "manga-finish-mid-tone-num", p.finish_mid_tone);
  setPair("manga-finish-black-point", "manga-finish-black-point-num", p.finish_black_point);
  setPair("manga-finish-white-cut", "manga-finish-white-cut-num", p.finish_white_cut);
  setPair("manga-panel-border", "manga-panel-border-num", p.panel_border);
  setPair("manga-panel-margin", "manga-panel-margin-num", p.panel_margin);
  setPair("manga-panel-bleed", "manga-panel-bleed-num", p.panel_print_bleed);

  const setVal = (id, val) => {
    const el = document.getElementById(id);
    if (el && val !== undefined) el.value = val;
  };
  setVal("manga-ai-ref-mode", p.ai_ref_mode);
  setVal("manga-ai-prompt", p.ai_prompt);
  setVal("manga-finish-preset", p.finish_preset);
  setVal("manga-finish-dot-shape", p.finish_dot_shape);
  setVal("manga-panel-aspect", p.panel_aspect);
  setVal("manga-panel-paper", p.panel_paper_tone);
}

function onMangaParamInput(key, rawVal, appRef) {
  const app = appRef || window.App;
  initMangaStudio(app);
  let val = rawVal;
  if (["line_thickness", "finish_dot_size", "panel_border", "panel_margin", "seed"].includes(key)) {
    val = parseInt(rawVal, 10);
    if (isNaN(val)) return;
  } else if ([
    "lora_strength", "ai_denoise_strength", "levels_contrast", "levels_gamma", "levels_descreen",
    "line_strength", "line_threshold", "finish_ink_threshold", "finish_light_tone",
    "finish_mid_tone", "finish_black_point", "finish_white_cut", "panel_print_bleed"
  ].includes(key)) {
    val = parseFloat(rawVal);
    if (isNaN(val)) return;
  }

  app.state.manga.params[key] = val;
  syncMangaControlsFromParams(app.state.manga.params);

  if (app.state.manga.params.mode === "manga_lite" || app.state.manga.params.mode === "finish_only") {
    if (app.state.manga.debounceTimer) clearTimeout(app.state.manga.debounceTimer);
    app.state.manga.debounceTimer = setTimeout(() => {
      runMangaPipeline(false, app);
    }, 220);
  }
}

function resetMangaParams(appRef) {
  const app = appRef || window.App;
  initMangaStudio(app);
  const curMode = app.state.manga.params?.mode || "manga_lite";
  app.state.manga.params = defaultMangaParams();
  selectMangaMode(curMode, app);
}

async function runMangaPipeline(immediate = false, appRef) {
  const app = appRef || window.App;
  initMangaStudio(app);
  const st = app.state.manga;
  if (!st.sourcePath && !st.sourceDataUrl) {
    if (immediate) toast("warning", "Сначала загрузите изображение или выберите его из галереи");
    return;
  }

  const p = st.params;
  const isAi = (p.mode === "ai_manga" || p.mode === "ai_manga_finish") && p.ai_enabled;
  const metaEl = document.getElementById("manga-meta-info");
  const banner = document.getElementById("manga-fallback-banner");
  const bannerMsg = document.getElementById("manga-fallback-msg");
  const loadOverlay = document.getElementById("manga-loading-overlay");
  const loadText = document.getElementById("manga-loading-text");

  if (!isAi) {
    if (metaEl) metaEl.textContent = "⏳ Обработка (CPU)...";
    if (loadText) loadText.textContent = "Обработка манги (CPU)...";
    if (loadOverlay) loadOverlay.style.display = "flex";

    try {
      const res = await apiPost("/api/manga/process", {
        image_path: st.sourcePath,
        image_data: st.sourceDataUrl,
        params: p,
        save: false,
        preview_max_side: 0, // Full resolution: preserves fine manga lines
      });

      st.resultDataUrl = res.preview_data_url;
      st.comparing = false;

      const emptyEl = document.getElementById("manga-empty-state");
      const wrapEl = document.getElementById("manga-preview-wrap");
      const imgEl = document.getElementById("manga-preview-img");
      const badgeEl = document.getElementById("manga-compare-badge");

      if (emptyEl) emptyEl.style.display = "none";
      if (wrapEl) wrapEl.style.display = "inline-flex";
      if (imgEl) imgEl.src = res.preview_data_url;
      if (badgeEl) badgeEl.style.display = "none";

      if (metaEl) {
        metaEl.textContent = `${res.width}×${res.height} px · ${(res.elapsed_sec * 1000).toFixed(0)} ms (CPU)`;
      }
    } catch (e) {
      if (metaEl) metaEl.textContent = "Ошибка";
      toast("error", "Manga pipeline error: " + e.message);
    } finally {
      if (loadOverlay) loadOverlay.style.display = "none";
    }
    return;
  }

  // AI GPU pipeline
  if (metaEl) metaEl.textContent = "⏳ Запуск AI Manga (GPU)...";
  if (loadText) loadText.textContent = "Запуск AI Manga (GPU)...";
  if (loadOverlay) loadOverlay.style.display = "flex";
  if (banner) banner.style.display = "none";

  try {
    const res = await apiPost("/api/manga/ai-job", {
      image_path: st.sourcePath,
      image_data: st.sourceDataUrl,
      params: p,
    });

    const jobId = res.job_id;
    const poll = setInterval(async () => {
      try {
        const status = await apiGet(`/api/jobs/${jobId}`);
        const msgText = status.message || `⏳ ${Math.round((status.progress || 0) * 100)}%`;
        if (metaEl) metaEl.textContent = msgText;
        if (loadText) loadText.textContent = msgText;

        if (status.status === "complete") {
          clearInterval(poll);
          if (loadOverlay) loadOverlay.style.display = "none";
          const filename = (status.image_path || "").split(/[\/\\]/).pop();
          const url = `/outputs/${encodeURIComponent(filename)}?t=${Date.now()}`;
          st.resultDataUrl = url;
          st.comparing = false;

          const emptyEl = document.getElementById("manga-empty-state");
          const wrapEl = document.getElementById("manga-preview-wrap");
          const imgEl = document.getElementById("manga-preview-img");
          if (emptyEl) emptyEl.style.display = "none";
          if (wrapEl) wrapEl.style.display = "inline-flex";
          if (imgEl) imgEl.src = url;

          if (metaEl) metaEl.textContent = status.message || "✓ Готово";
          const savedMsg = window.t ? (window.t("manga.saved_toast") || "Результат сохранён в галерею ✓") : "Результат сохранён в галерею ✓";
          toast("success", savedMsg);
        } else if (status.status === "error") {
          clearInterval(poll);
          if (loadOverlay) loadOverlay.style.display = "none";
          if (metaEl) metaEl.textContent = "Ошибка AI";
          if (banner && bannerMsg) {
            bannerMsg.textContent = `Ошибка AI-этапа: ${status.error || status.message}. Режим Manga Lite доступен без GPU.`;
            banner.style.display = "flex";
          }
          toast("error", status.error || status.message || "AI Manga failed");
        }
      } catch (_) {
        clearInterval(poll);
        if (loadOverlay) loadOverlay.style.display = "none";
      }
    }, 400);
  } catch (e) {
    if (loadOverlay) loadOverlay.style.display = "none";
    if (metaEl) metaEl.textContent = "Ошибка AI";
    if (banner && bannerMsg) {
      bannerMsg.textContent = `${e.message} — Вы можете переключиться в режим Manga Lite (CPU) без загрузки моделей.`;
      banner.style.display = "flex";
    }
    toast("error", e.message);
  }
}

function toggleMangaCompare(appRef) {
  const app = appRef || window.App;
  initMangaStudio(app);
  const st = app.state.manga;
  if (!st.sourceUrl || !st.resultDataUrl) return;

  st.comparing = !st.comparing;
  const imgEl = document.getElementById("manga-preview-img");
  const badgeEl = document.getElementById("manga-compare-badge");
  const btnEl = document.getElementById("manga-compare-btn");

  if (imgEl) imgEl.src = st.comparing ? st.sourceUrl : st.resultDataUrl;
  if (badgeEl) badgeEl.style.display = st.comparing ? "block" : "none";
  if (btnEl) btnEl.classList.toggle("active", st.comparing);
}

function revertMangaToOriginal(appRef) {
  const app = appRef || window.App;
  initMangaStudio(app);
  const st = app.state.manga;
  if (!st.sourceUrl) return;

  st.comparing = true;
  const imgEl = document.getElementById("manga-preview-img");
  const badgeEl = document.getElementById("manga-compare-badge");
  if (imgEl) imgEl.src = st.sourceUrl;
  if (badgeEl) badgeEl.style.display = "block";
}

async function saveMangaToGallery(appRef) {
  const app = appRef || window.App;
  initMangaStudio(app);
  const st = app.state.manga;
  if (!st.sourcePath && !st.sourceDataUrl) {
    toast("warning", "Нет изображения для сохранения");
    return;
  }

  if (st.params.mode === "ai_manga" || st.params.mode === "ai_manga_finish") {
    await runMangaPipeline(true, app);
    return;
  }

  try {
    const res = await apiPost("/api/manga/process", {
      image_path: st.sourcePath,
      image_data: st.sourceDataUrl,
      params: st.params,
      save: true,
    });
    if (res.saved) {
      st.resultDataUrl = res.preview_data_url;
      const imgEl = document.getElementById("manga-preview-img");
      if (imgEl) imgEl.src = res.preview_data_url;
      const savedMsg = window.t ? (window.t("manga.saved_toast") || "Сохранено в галерею") : "Сохранено в галерею";
      toast("success", `${savedMsg}: ${res.saved.name} ✓`);
    }
  } catch (e) {
    toast("error", "Ошибка сохранения: " + e.message);
  }
}

async function downloadMangaPng(appRef) {
  const app = appRef || window.App;
  initMangaStudio(app);
  const st = app.state.manga;
  if (!st || (!st.sourcePath && !st.sourceDataUrl)) {
    toast("warning", "Сначала выберите или загрузите изображение");
    return;
  }

  const loadOverlay = document.getElementById("manga-loading-overlay");
  if (loadOverlay) loadOverlay.style.display = "flex";
  toast("info", "Рендеринг в полном разрешении для скачивания...");

  try {
    const p = { ...st.params, mode: "manga_screen", finish_enabled: false };
    const res = await apiPost("/api/manga/process", {
      image_path: st.sourcePath,
      image_data: st.sourceDataUrl,
      params: p,
      save: false,
      preview_max_side: 0,
    });

    const src = res?.preview_data_url || st.resultDataUrl;
    if (!src) throw new Error("Не удалось получить результат рендеринга");

    const a = document.createElement("a");
    a.href = src;
    a.download = `manga_${res.width || "full"}x${res.height || "res"}_${Date.now()}.png`;
    a.click();
    toast("success", `Скачано в оригинальном разрешении ${res.width}×${res.height} px ✓`);
  } catch (e) {
    toast("error", "Ошибка скачивания: " + e.message);
  } finally {
    if (loadOverlay) loadOverlay.style.display = "none";
  }
}

function onMangaFileUpload(input, appRef) {
  const app = appRef || window.App;
  const file = input?.files?.[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = (e) => {
    loadMangaSource(e.target.result, file.name, app);
  };
  reader.readAsDataURL(file);
  input.value = "";
}

function loadMangaSource(dataUrl, filename = "image.png", appRef) {
  const app = appRef || window.App;
  initMangaStudio(app);
  const st = app.state.manga;
  st.sourceDataUrl = dataUrl;
  st.sourceUrl = dataUrl;
  st.sourcePath = null;

  const emptyEl = document.getElementById("manga-empty-state");
  const wrapEl = document.getElementById("manga-preview-wrap");
  const imgEl = document.getElementById("manga-preview-img");
  if (emptyEl) emptyEl.style.display = "none";
  if (wrapEl) wrapEl.style.display = "inline-flex";
  if (imgEl) imgEl.src = dataUrl;

  const metaEl = document.getElementById("manga-meta-info");
  if (metaEl) metaEl.textContent = `Файл: ${filename}`;

  runMangaPipeline(true, app);
}


  Object.assign(App, {
    defaultMangaParams,
    initMangaStudio,
    populateMangaModelsAndLoras,
    onMangaModelChange,
    onMangaLoraChange,
    checkMangaLoraCompatibility,
    selectMangaMode,
    syncMangaControlsFromParams,
    onMangaParamInput,
    resetMangaParams,
    runMangaPipeline,
    toggleMangaCompare,
    revertMangaToOriginal,
    saveMangaToGallery,
    downloadMangaPng,
    onMangaFileUpload,
    loadMangaSource,
  });
})();

// ============================================================
// --- UI HELPERS FOR INDEX.HTML INLINE CALLS ---
// ============================================================
Object.assign(App, {
  onModelsTabSearch(query) {
    this.state.modelsTabSearch = query || "";
    this.renderModelsTab();
  },

  setModelsTabFilter(filter) {
    this.state.modelsTabFilter = filter || "all";
    document.querySelectorAll("#models-hub-filters .filter-pill").forEach(p => {
      p.classList.toggle("active", p.dataset.filter === filter);
    });
    this.renderModelsTab();
  },

  switchModelsHubSubtab(subtab) {
    const normalized = subtab === "loras" ? "loras" : "models";
    this.state.modelsHubSubtab = normalized;
    const isLoras = normalized === "loras";

    document.getElementById("models-subtab-btn-models")?.classList.toggle("active", !isLoras);
    document.getElementById("models-subtab-btn-loras")?.classList.toggle("active", isLoras);

    const modelsPanel = document.getElementById("models-subpanel-models");
    const lorasPanel = document.getElementById("models-subpanel-loras");
    if (modelsPanel) modelsPanel.style.display = isLoras ? "none" : "block";
    if (lorasPanel) lorasPanel.style.display = isLoras ? "block" : "none";

    const addModelBtn = document.getElementById("models-hub-add-model-btn");
    const addLoraBtn = document.getElementById("models-hub-add-lora-btn");
    if (addModelBtn) addModelBtn.style.display = isLoras ? "none" : "inline-flex";
    if (addLoraBtn) addLoraBtn.style.display = isLoras ? "inline-flex" : "none";

    if (isLoras) {
      this.loadLoras();
    } else {
      this.renderModelsTab();
    }
  },

  goToModelInHub(modelId) {
    this.switchTab("models");
    this.state.modelsTabSearch = modelId || "";
    const searchInput = document.getElementById("models-tab-search");
    if (searchInput) searchInput.value = modelId || "";
    this.renderModelsTab();
  },

  async scanLocalModelPath() {
    const input = document.getElementById("custom-model-local-path");
    const path = (input?.value || "").trim();
    if (!path) {
      this.toast("warning", "Укажите путь к файлу или папке модели");
      return;
    }
    try {
      this.toast("info", "Сканирование пути...");
      const res = await this.apiPost("/api/custom-models/sniff", { path });
      this.populateCustomFileSelects(res);
      this.toast("success", `Обнаружено: ${res.detected_architecture || "модель"} ✓`);
    } catch (e) {
      this.toast("error", "Ошибка сканирования: " + e.message);
    }
  },

  filterLoraLibrary(query) {
    this.state.loraLibrarySearch = (query || "").toLowerCase();
    this.renderLoraLibrary();
  },

  switchLoraModalTab(tab) {
    const normalized = (tab === "hf" || tab === "active" || tab === "browse" || tab === "import")
      ? (tab === "import" ? "local" : "hub")
      : (tab || "hub");
    const tabs = ["hub", "local", "library"];

    tabs.forEach(t => {
      document.getElementById(`lora-tab-btn-${t}`)?.classList.toggle("active", t === normalized);
      const pane = document.getElementById(`lora-pane-${t}`);
      if (pane) pane.style.display = (t === normalized) ? "flex" : "none";
    });

    if (normalized === "library") this.loadLoras();
  },

  openLoraManageModal(tab = "hub") {
    const modal = document.getElementById("lora-manage-modal");
    if (modal) modal.style.display = "flex";
    this.switchLoraModalTab(tab);
    this.loadLoras();
  },

  async inspectHfLora() {
    const repoInput = document.getElementById("lora-hf-repo-input");
    const repo = (repoInput?.value || "").trim();
    if (!repo) {
      this.toast("warning", "Укажите репозиторий HuggingFace");
      return;
    }
    try {
      this.toast("info", "Проверка репозитория...");
      const res = await this.apiGet(`/api/loras/hf-inspect?repo_id=${encodeURIComponent(repo)}`);
      const files = res.files || [];
      if (!files.length) {
        this.toast("warning", "В репозитории не найдены файлы LoRA (.safetensors / .gguf)");
        return;
      }

      if (repoInput) repoInput.value = res.repo_id || repo;
      const inspector = document.getElementById("lora-hf-inspector");
      if (inspector) inspector.style.display = "flex";
      const title = document.getElementById("lora-hf-inspected-title");
      if (title) title.textContent = res.repo_id || repo;
      const badge = document.getElementById("lora-hf-base-badge");
      if (badge) badge.textContent = res.base_model || res.family || "LoRA";

      const fileSelect = document.getElementById("lora-hf-file-select");
      if (fileSelect) {
        fileSelect.innerHTML = files.map(f =>
          `<option value="${escapeHtml(f.path)}">${escapeHtml(f.name)}${f.size_mb ? ` · ${f.size_mb} MB` : ""}</option>`
        ).join("");
      }

      const familySelect = document.getElementById("lora-hf-family-select");
      if (familySelect) {
        const fam = res.suggested_family || res.family || "any";
        const hasOption = Array.from(familySelect.options).some(o => o.value === fam);
        familySelect.value = hasOption ? fam : "any";
      }

      const triggerInput = document.getElementById("lora-hf-trigger-input");
      if (triggerInput) triggerInput.value = res.suggested_trigger || res.trigger || "";
      const renameInput = document.getElementById("lora-hf-rename-input");
      if (renameInput) renameInput.value = "";

      this.toast("success", `Найдено файлов LoRA: ${files.length}`);
    } catch (e) {
      this.toast("error", e.message);
    }
  },

  async searchHfLoras(query) {
    const q = (query || document.getElementById("lora-hf-search-input")?.value || "").trim();
    if (!q) {
      this.toast("warning", "Введите запрос для поиска на HuggingFace");
      return;
    }
    try {
      this.toast("info", "Поиск на HuggingFace...");
      const res = await this.apiGet(`/api/loras/hf-search?q=${encodeURIComponent(q)}`);
      this.renderHfLoraSearchResults(res.results || []);
    } catch (e) {
      this.toast("error", e.message);
    }
  },

  renderHfLoraSearchResults(results) {
    const box = document.getElementById("lora-hf-search-results");
    if (!box) return;

    if (!results.length) {
      box.style.display = "block";
      box.innerHTML = '<div style="padding: 14px; text-align: center; color: var(--text-dim);">Ничего не найдено</div>';
      return;
    }

    box.style.display = "block";
    box.innerHTML = results.map(item => `
      <div style="display: flex; align-items: center; justify-content: space-between; gap: 10px; padding: 6px 8px; border-bottom: 1px solid var(--border);">
        <div style="min-width: 0;">
          <div style="font-size: 12px; font-weight: 600; color: var(--text-primary); overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">${escapeHtml(item.repo_id)}</div>
          <div style="font-size: 10px; color: var(--text-dim);">⬇ ${item.downloads || 0} · ❤ ${item.likes || 0}${item.base_model ? ` · ${escapeHtml(item.base_model)}` : ""}</div>
        </div>
        <button type="button" class="dock-mini-btn" onclick="App.selectHfLoraResult('${escapeHtml(item.repo_id)}')">Выбрать</button>
      </div>
    `).join("");
  },

  selectHfLoraResult(repoId) {
    const repoInput = document.getElementById("lora-hf-repo-input");
    if (repoInput) repoInput.value = repoId;
    this.inspectHfLora();
  },

  dismissUninstalledNotice() {
    const notice = document.getElementById("gen-uninstalled-notice");
    if (notice) notice.style.display = "none";
    const banner = document.getElementById("uninstalled-models-notice");
    if (banner) banner.style.display = "none";
  },

  async onSettingHideUninstalledChange(chk) {
    const checked = (typeof chk === "boolean") ? chk : Boolean(chk?.checked ?? chk);
    try {
      await this.apiPatch("/api/settings", { hide_uninstalled_in_generator: checked });
      if (this.state.settings) this.state.settings.hide_uninstalled_in_generator = checked;
      this.renderModels();
      this.updateDockPills();
      this.toast("info", checked ? "Нескачанные модели скрыты из Дока" : "Отображаются все модели");
    } catch (e) {
      this.toast("error", e.message || "Ошибка сохранения настроек");
    }
  },

  setHistoryPageSize(sz) {
    this.state.historyPageSize = parseInt(sz, 10) || 50;
    this.renderHistory();
  },
});


// ---------- Resilient Boot ----------
if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", () => App.init());
} else {
  App.init();
}

// Export for inline scripts
window.App = App;

