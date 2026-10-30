/* ============================================================
   QuantFunc WebUI — Alternate LazySD Studio Dock
   ============================================================ */

function getApp(appRef) {
  return appRef || window.App;
}

function syncSelect(sourceId, targetId) {
  const source = document.getElementById(sourceId);
  const target = document.getElementById(targetId);
  if (!source || !target) return;
  if (target.innerHTML !== source.innerHTML) target.innerHTML = source.innerHTML;
  target.value = source.value;
  target.disabled = source.disabled;
}

function getStage() {
  return document.getElementById("canvas-stage");
}

function tr(key, fallback) {
  return (window.t && window.t(key)) || fallback;
}

let activeNewDockPopover = null;
let newDockShutterOpen = false;
let newDockModelDetailsOpen = false;
const NEW_DOCK_RESOLUTION_PRESETS = [
  { ratio: "1:1", width: 1024, height: 1024, orientation: "square" },
  { ratio: "16:9", width: 1344, height: 768, orientation: "landscape" },
  { ratio: "9:16", width: 768, height: 1344, orientation: "portrait" },
  { ratio: "4:3", width: 1152, height: 896, orientation: "landscape" },
  { ratio: "3:4", width: 896, height: 1152, orientation: "portrait" },
];

function getSelectedText(id) {
  const select = document.getElementById(id);
  return select?.selectedOptions?.[0]?.textContent?.trim() || select?.value || "—";
}

function getAspectRatio(value) {
  const match = String(value || "").match(/^(\d+)\s*x\s*(\d+)$/i);
  if (!match) return "";
  const width = Number(match[1]);
  const height = Number(match[2]);
  if (!width || !height) return "";
  const gcd = (a, b) => (b ? gcd(b, a % b) : a);
  const divisor = gcd(width, height);
  return `${width / divisor}:${height / divisor}`;
}

function closeNewDockPopover() {
  activeNewDockPopover = null;
  newDockModelDetailsOpen = false;
  document.querySelectorAll(".new-dock-popover").forEach((popover) => {
    popover.style.display = "none";
    popover.classList.remove("details-open");
  });
  document.querySelectorAll(".new-dock-setting-pill").forEach((button) => {
    button.classList.remove("active");
    button.setAttribute("aria-expanded", "false");
  });
}

function addButton(container, text, className, title, onClick) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = className;
  button.textContent = text;
  if (title) button.title = title;
  button.addEventListener("click", onClick);
  container.appendChild(button);
  return button;
}

function renderModelPopover(app) {
  const modelList = document.getElementById("new-dock-model-list");
  const modelSelect = document.getElementById("model-select");
  const variants = document.getElementById("new-dock-quant-options");
  const encoders = document.getElementById("new-dock-encoder-options");
  if (!modelList || !modelSelect || !variants || !encoders) return;

  const modelsById = new Map((app.state.models || []).map((model) => [model.id, model]));
  const options = Array.from(modelSelect.options).filter((option) => option.value);
  const listSignature = options.map((option) => {
    const item = modelsById.get(option.value);
    return [option.value, item?.name || option.textContent.trim(), item?.badge || "", option.disabled ? "1" : "0"].join("\u001f");
  }).join("\u001e");
  if (modelList.dataset.signature !== listSignature) {
    modelList.replaceChildren();
    options.forEach((option) => {
      const item = modelsById.get(option.value);
      const row = document.createElement("div");
      row.className = "new-dock-model-row";
      row.dataset.modelId = option.value;
      const button = document.createElement("button");
      button.type = "button";
      button.className = "new-dock-model-select";
      button.disabled = option.disabled;
      button.title = item?.name || option.textContent.trim();
      const name = document.createElement("span");
      name.className = "new-dock-model-option-name";
      name.textContent = item?.name || option.textContent.replace(/^[✓⬇⛔]\s*/, "").split(" (")[0];
      button.appendChild(name);
      const badgeText = item?.badge || (item?.name?.toLowerCase().includes("schnell") ? "Fast" : "");
      if (badgeText) {
        const badge = document.createElement("span");
        badge.className = "new-dock-model-badge";
        badge.textContent = badgeText;
        button.appendChild(badge);
      }
      button.dataset.modelAction = "select";
      const check = document.createElement("span");
      check.className = "new-dock-model-check";
      check.textContent = "✓";
      button.appendChild(check);

      const expand = document.createElement("button");
      expand.type = "button";
      expand.className = "new-dock-model-expand";
      expand.textContent = "›";
      expand.title = tr("dock.model_details", "Model details");
      expand.setAttribute("aria-label", expand.title);
      expand.disabled = option.disabled;
      expand.dataset.modelAction = "details";
      row.append(button, expand);
      modelList.appendChild(row);
    });
    modelList.dataset.signature = listSignature;
  }
  if (!modelList.dataset.clickBound) {
    modelList.dataset.clickBound = "true";
    modelList.addEventListener("click", (event) => {
      const action = event.target.closest("[data-model-action]");
      const row = action?.closest(".new-dock-model-row");
      if (!action || !row || !modelList.contains(action)) return;
      const modelId = row.dataset.modelId;
      if (!modelId) return;
      if (action.dataset.modelAction === "details") {
        const isSelected = modelId === getApp()?.state.selectedModel;
        newDockModelDetailsOpen = !isSelected || !newDockModelDetailsOpen;
        if (!isSelected) void onNewDockModelSelect(modelId);
        updateNewDock();
      } else if (modelId !== getApp()?.state.selectedModel) {
        void onNewDockModelSelect(modelId);
      }
    });
  }
  modelList.querySelectorAll(".new-dock-model-row").forEach((row) => {
    const selected = row.dataset.modelId === app.state.selectedModel;
    row.classList.toggle("selected", selected);
    const selectButton = row.querySelector(".new-dock-model-select");
    const expandButton = row.querySelector(".new-dock-model-expand");
    const check = row.querySelector(".new-dock-model-check");
    if (check) check.hidden = !selected;
    if (expandButton) expandButton.setAttribute("aria-expanded", String(selected && newDockModelDetailsOpen));
  });
  const popover = document.getElementById("new-dock-popover-model");
  popover?.classList.toggle("details-open", newDockModelDetailsOpen);
  if (!newDockModelDetailsOpen) return;

  const activeModel = modelsById.get(app.state.selectedModel);
  const modelName = document.getElementById("new-dock-model-name");
  const architecture = document.getElementById("new-dock-model-architecture");
  if (modelName) modelName.textContent = activeModel?.name || "—";
  if (architecture) {
    const variant = app.state.modelState?.transformers?.find((item) => item.id === app.state.selectedVariant);
    architecture.textContent = [
      activeModel?.architecture || activeModel?.family || activeModel?.engine || "",
      variant?.backend || "",
    ].filter(Boolean).join("  ");
  }

  variants.replaceChildren();
  const modelVariants = app.state.modelState?.transformers || [];
  const quantCurrent = document.getElementById("new-dock-quant-current");
  const selectedVariant = modelVariants.find((item) => item.id === app.state.selectedVariant);
  if (quantCurrent) quantCurrent.textContent = selectedVariant?.id || app.state.selectedVariant || "";
  modelVariants.forEach((variant) => {
    const button = addButton(
      variants,
      variant.id,
      "new-dock-quant-option",
      [variant.backend, variant.steps ? `${variant.steps} steps` : "", variant.state_label].filter(Boolean).join(" · "),
      () => onNewDockVariantSelect(variant.id, app),
    );
    button.classList.toggle("selected", variant.id === app.state.selectedVariant);
    if (variant.id === app.state.selectedVariant) {
      const check = document.createElement("span");
      check.textContent = "✓";
      button.appendChild(check);
    }
  });
  if (!modelVariants.length) {
    const empty = document.createElement("div");
    empty.className = "new-dock-model-empty";
    empty.textContent = tr("dock.no_variants", "No variants available");
    variants.appendChild(empty);
  }

  encoders.replaceChildren();
  const teSelect = document.getElementById("text-encoder-select");
  const encoderSection = document.getElementById("new-dock-encoder-section");
  const encoderOptions = Array.from(teSelect?.options || []).filter((option) => option.value);
  if (encoderSection) {
    encoderSection.style.display = document.getElementById("dock-te-quick-field")?.style.display === "none" || !encoderOptions.length ? "none" : "block";
  }
  encoderOptions.forEach((option) => {
    const encoderLabel = option.textContent.trim().replace(/\s+\([✓⬇]\)\s*$/, "");
    const button = addButton(
      encoders,
      encoderLabel,
      "new-dock-encoder-option",
      option.textContent.trim(),
      () => {
        if (!teSelect) return;
        teSelect.value = option.value;
        if (app.onTextEncoderChange) app.onTextEncoderChange();
        updateNewDock(app);
      },
    );
    button.classList.toggle("selected", option.value === teSelect.value);
    if (option.value === teSelect.value) {
      const check = document.createElement("span");
      check.textContent = "✓";
      button.appendChild(check);
    }
  });

  const vaeSelect = document.getElementById("new-dock-vae");
  const vaeSource = document.getElementById("dock-vae-quick-select");
  const vaeControl = document.getElementById("new-dock-vae-control");
  if (vaeSelect && vaeSource) {
    syncSelect("dock-vae-quick-select", "new-dock-vae");
    if (vaeControl) vaeControl.style.display = document.getElementById("dock-vae-quick-field")?.style.display === "none" ? "none" : "flex";
  }

  const vramValue = document.getElementById("new-dock-vram-value");
  const vramCaption = document.getElementById("new-dock-vram-caption");
  const vramMeter = document.querySelector(".new-dock-vram-meter");
  const minimumVram = Number(activeModel?.requirements?.vram_min_gb);
  const totalVram = Number(app.state.gpuInfo?.vram_mb) / 1024;
  if (vramValue) vramValue.textContent = minimumVram > 0
    ? `${tr("dock.vram_min", "Min. VRAM")}: ${minimumVram.toFixed(1)} GB${totalVram > 0 ? ` / ${totalVram.toFixed(0)} GB` : ""}`
    : tr("dock.vram_unavailable", "VRAM requirement unavailable");
  if (vramCaption) vramCaption.textContent = minimumVram > 0
    ? tr("dock.vram_min_caption", "Minimum requirement in model data")
    : "";
  if (vramMeter) {
    const percent = minimumVram > 0 && totalVram > 0 ? Math.min(100, minimumVram / totalVram * 100) : 0;
    vramMeter.style.setProperty("--vram-percent", `${percent}%`);
    vramMeter.dataset.value = percent ? `${Math.round(percent)}%` : "—";
  }
}

function renderResolutionPresets(app) {
  const container = document.getElementById("new-dock-resolution-presets");
  if (!container) return;
  container.replaceChildren();
  NEW_DOCK_RESOLUTION_PRESETS.forEach((preset) => {
    const button = addButton(
      container,
      "",
      "new-dock-resolution-option",
      `${preset.ratio} · ${preset.width} × ${preset.height}`,
      () => setNewDockResolution(preset.width, preset.height, app),
    );
    const icon = document.createElement("span");
    icon.className = `new-dock-orientation-icon ${preset.orientation}`;
    icon.setAttribute("aria-hidden", "true");
    const ratio = document.createElement("strong");
    ratio.textContent = preset.ratio;
    const size = document.createElement("span");
    size.className = "new-dock-resolution-size";
    size.textContent = `${preset.width}×${preset.height}`;
    button.append(icon, ratio, size);
  });
}

function setNewDockResolution(width, height, appRef) {
  const app = getApp(appRef);
  const source = document.getElementById("resolution-select");
  if (!app || !source) return;
  const snappedWidth = app.snapDimension ? app.snapDimension(width) : width;
  const snappedHeight = app.snapDimension ? app.snapDimension(height) : height;
  const value = `${snappedWidth}x${snappedHeight}`;
  if (!Array.from(source.options).some((option) => option.value === value)) {
    source.add(new Option(`${snappedWidth} × ${snappedHeight}`, value));
  }
  source.value = value;
  source.dispatchEvent(new Event("change", { bubbles: true }));
  updateNewDock(app);
}

function getNewDockBounds() {
  const stage = getStage();
  const wrap = document.getElementById("new-dock-wrap");
  const width = wrap?.offsetWidth || 680;
  const height = wrap?.offsetHeight || 180;
  const stageRect = stage?.getBoundingClientRect();
  const availableWidth = Math.max(0, Math.min(
    stage?.clientWidth || window.innerWidth,
    window.innerWidth - (stageRect?.left || 0),
  ));
  const availableHeight = Math.max(0, Math.min(
    stage?.clientHeight || window.innerHeight,
    window.innerHeight - (stageRect?.top || 0),
  ));
  return {
    maxX: Math.max(0, availableWidth - width),
    maxY: Math.max(0, availableHeight - height),
  };
}

function clampNewDockPosition(app) {
  if (!Number.isFinite(app.state.newDockCustomX) || !Number.isFinite(app.state.newDockCustomY)) return;
  const bounds = getNewDockBounds();
  app.state.newDockCustomX = Math.round(Math.max(0, Math.min(bounds.maxX, app.state.newDockCustomX)));
  app.state.newDockCustomY = Math.round(Math.max(0, Math.min(bounds.maxY, app.state.newDockCustomY)));
}

export function updateNewDock(appRef) {
  const app = getApp(appRef);
  if (!app) return;

  syncSelect("model-select", "new-dock-model");
  syncSelect("variant-select", "new-dock-variant");
  syncSelect("resolution-select", "new-dock-resolution");
  syncSelect("sampler-select", "new-dock-sampler");

  const valuePairs = [
    ["steps-input", "new-dock-steps"],
    ["cfg-input", "new-dock-cfg"],
    ["noise-input", "new-dock-noise"],
    ["seed-input", "new-dock-seed"],
    ["text-encoder-select", "new-dock-te-sync"],
    ["dock-vae-quick-select", "new-dock-vae"],
  ];
  valuePairs.forEach(([sourceId, targetId]) => {
    const source = document.getElementById(sourceId);
    const target = document.getElementById(targetId);
    if (source && target) {
      if (target.tagName === "SELECT") syncSelect(sourceId, targetId);
      else target.value = source.value;
    }
  });

  const prompt = document.getElementById("prompt-input");
  const newPrompt = document.getElementById("new-dock-prompt");
  if (prompt && newPrompt && newPrompt.value !== prompt.value) newPrompt.value = prompt.value;

  const negative = document.getElementById("negative-prompt-input");
  const newNegative = document.getElementById("new-dock-negative-prompt");
  if (negative && newNegative && newNegative.value !== negative.value) newNegative.value = negative.value;

  const generateButton = document.getElementById("generate-btn");
  const newGenerateButton = document.getElementById("new-dock-generate");
  if (generateButton && newGenerateButton) newGenerateButton.disabled = generateButton.disabled;

  const modelSummary = document.getElementById("new-dock-model-summary");
  if (modelSummary) {
    const model = app.state.models?.find((item) => item.id === app.state.selectedModel)?.name
      || getSelectedText("new-dock-model").replace(/^[✓⬇⛔]\s*/, "").split(" (")[0];
    const variant = app.state.selectedVariant || getSelectedText("new-dock-variant");
    modelSummary.textContent = `${tr("dock.model_prefix", "Model")}: ${model}${variant !== "—" ? ` (${variant})` : ""}`;
    modelSummary.title = modelSummary.textContent;
  }

  const resolutionSummary = document.getElementById("new-dock-resolution-summary");
  if (resolutionSummary) {
    const resolution = document.getElementById("new-dock-resolution")?.value || "";
    const ratio = getAspectRatio(resolution);
    resolutionSummary.textContent = `${ratio ? `${ratio} · ` : ""}${resolution.replace("x", "×") || "—"}`;
  }

  const paramsSummary = document.getElementById("new-dock-params-summary");
  if (paramsSummary) {
    const sampler = getSelectedText("new-dock-sampler").replace(/\s*\((?:Default|Noise)\)\s*/i, "");
    const steps = document.getElementById("new-dock-steps")?.value || "—";
    const cfg = Number(document.getElementById("new-dock-cfg")?.value);
    paramsSummary.textContent = `${sampler} · ${steps} st · CFG ${Number.isFinite(cfg) ? cfg.toFixed(1) : "—"}`;
  }

  const stepsControl = document.getElementById("new-dock-steps");
  const stepsValue = document.getElementById("new-dock-steps-value");
  if (stepsControl && stepsValue) stepsValue.textContent = stepsControl.value;
  const cfgControl = document.getElementById("new-dock-cfg");
  const cfgValue = document.getElementById("new-dock-cfg-value");
  if (cfgControl && cfgValue) cfgValue.textContent = Number(cfgControl.value).toFixed(1);

  const resolution = document.getElementById("resolution-select")?.value || "1024x1024";
  const match = resolution.match(/^(\d+)x(\d+)$/i);
  if (match) {
    const width = Number(match[1]);
    const height = Number(match[2]);
    const widthControl = document.getElementById("new-dock-width");
    const heightControl = document.getElementById("new-dock-height");
    if (widthControl && widthControl !== document.activeElement) widthControl.value = String(width);
    if (heightControl && heightControl !== document.activeElement) heightControl.value = String(height);
    const meta = document.getElementById("new-dock-resolution-meta");
    if (meta) meta.textContent = `${getAspectRatio(resolution)} · ${((width * height) / 1_000_000).toFixed(2)} MP`;
    document.querySelectorAll(".new-dock-resolution-option").forEach((button, index) => {
      const preset = NEW_DOCK_RESOLUTION_PRESETS[index];
      button.classList.toggle("selected", preset.width === width && preset.height === height);
    });
  }

  if (activeNewDockPopover === "model") renderModelPopover(app);

  syncNewDockRef(app);

  const negativeSummary = document.getElementById("new-dock-negative-summary");
  const negativePill = document.getElementById("new-dock-negative-pill");
  if (negativeSummary && negativePill) {
    const hasNegative = Boolean(document.getElementById("new-dock-negative-prompt")?.value.trim());
    negativeSummary.textContent = `${tr("dock.negative_label_short", "Neg")} (${hasNegative ? tr("dock.on_short", "On") : tr("dock.off_short", "Off")})`;
    negativePill.classList.toggle("active", hasNegative || newDockShutterOpen && document.getElementById("new-dock-negative")?.style.display !== "none");
  }
}

// ---------- Reference / ControlNet bar ----------
// The classic dock owns the real controls (#img2img-mode, #img2img-strength,
// #controlnet-custom-path) that generation reads; the new dock mirrors and drives them.
export function syncNewDockRef(appRef) {
  const app = getApp(appRef);
  const bar = document.getElementById("new-dock-ref-bar");
  if (!app || !bar) return;
  const ref = app.state?.img2imgRef;
  const visible = Boolean(ref && ref.url);
  bar.style.display = visible ? "flex" : "none";
  if (!visible) return;

  const thumb = document.getElementById("new-dock-ref-thumb");
  if (thumb && thumb.getAttribute("src") !== ref.url) thumb.src = ref.url;

  syncSelect("img2img-mode", "new-dock-ref-mode");

  const srcStrength = document.getElementById("img2img-strength");
  const strength = document.getElementById("new-dock-ref-strength");
  const strengthVal = document.getElementById("new-dock-ref-strength-val");
  if (srcStrength && strength) {
    strength.max = srcStrength.max;
    if (strength !== document.activeElement) strength.value = srcStrength.value;
    if (strengthVal) strengthVal.textContent = Number(srcStrength.value).toFixed(2);
  }

  const srcCn = document.getElementById("controlnet-custom-path");
  const cn = document.getElementById("new-dock-ref-cn-path");
  const cnBtn = document.getElementById("new-dock-ref-cn-btn");
  if (srcCn && cn) {
    const active = srcCn.style.display !== "none";
    if (cn !== document.activeElement) cn.value = srcCn.value;
    cn.style.display = active ? "block" : "none";
    if (cnBtn) cnBtn.classList.toggle("active", active && Boolean(srcCn.value.trim()));
  }
}

export function onNewDockRefModeChange(value, appRef) {
  const app = getApp(appRef);
  const mode = document.getElementById("img2img-mode");
  if (!mode) return;
  mode.value = value;
  app?.onRefModeChange?.();
  syncNewDockRef(app);
}

export function onNewDockRefStrengthChange(value, appRef) {
  const app = getApp(appRef);
  const slider = document.getElementById("img2img-strength");
  const label = document.getElementById("img2img-strength-val");
  if (slider) slider.value = value;
  if (label) label.textContent = Number(value).toFixed(2);
  const ownLabel = document.getElementById("new-dock-ref-strength-val");
  if (ownLabel) ownLabel.textContent = Number(value).toFixed(2);
  if (app?.state) app.state.newDockRefDirty = true;
}

export function toggleNewDockControlNet(appRef) {
  const app = getApp(appRef);
  const srcCn = document.getElementById("controlnet-custom-path");
  const cn = document.getElementById("new-dock-ref-cn-path");
  if (!srcCn || !cn) return;
  const show = srcCn.style.display === "none";
  // Generation reads the classic input only while it is not display:none.
  srcCn.style.display = show ? "block" : "none";
  syncNewDockRef(app);
  if (show) cn.focus();
}

export function onNewDockControlNetPath(value, appRef) {
  const app = getApp(appRef);
  const srcCn = document.getElementById("controlnet-custom-path");
  if (srcCn) srcCn.value = value;
  const cnBtn = document.getElementById("new-dock-ref-cn-btn");
  if (cnBtn) cnBtn.classList.toggle("active", Boolean(String(value).trim()));
  void app;
}

export function applyNewDockSettings(appRef) {
  const app = getApp(appRef);
  const stage = getStage();
  const wrap = document.getElementById("new-dock-wrap");
  if (!app || !stage || !wrap) return;

  stage.classList.toggle("dock-variant-new", app.state.dockVariant === "new");
  const floatEnabled = app.state.newDockFloat === true;
  const minimized = app.state.dockMinimized === true;
  wrap.classList.toggle("new-dock-minimized", minimized);
  if (minimized) setNewDockShutter(false);
  if (floatEnabled && wrap.offsetWidth > 0
    && (!Number.isFinite(app.state.newDockCustomX) || !Number.isFinite(app.state.newDockCustomY))) {
    const rect = wrap.getBoundingClientRect();
    const stageRect = stage.getBoundingClientRect();
    app.state.newDockCustomX = Math.round(rect.left - stageRect.left);
    app.state.newDockCustomY = Math.round(rect.top - stageRect.top);
  }
  wrap.classList.toggle("new-dock-floating", floatEnabled);
  if (floatEnabled) {
    clampNewDockPosition(app);
    if (Number.isFinite(app.state.newDockCustomX) && Number.isFinite(app.state.newDockCustomY)) {
      wrap.style.left = `${app.state.newDockCustomX}px`;
      wrap.style.top = `${app.state.newDockCustomY}px`;
    }
  } else {
    wrap.style.left = "";
    wrap.style.top = "";
  }

  const floatToggle = document.getElementById("new-dock-float-toggle");
  if (floatToggle) {
    floatToggle.classList.toggle("active", floatEnabled);
    floatToggle.setAttribute("aria-pressed", String(floatEnabled));
    floatToggle.title = floatEnabled
      ? tr("dock.pin_new", "Закрепить новый док снизу")
      : tr("dock.float_new", "Переключить новый док в режим Float");
  }

  const pinToggle = document.getElementById("new-dock-pin-toggle");
  if (pinToggle) {
    pinToggle.classList.toggle("active", !floatEnabled);
    pinToggle.setAttribute("aria-pressed", String(!floatEnabled));
    pinToggle.title = floatEnabled
      ? tr("dock.pin_new", "Закрепить новый док снизу")
      : tr("dock.unpin_new", "Открепить новый док");
    const pinLabel = document.getElementById("new-dock-pin-label");
    if (pinLabel) pinLabel.textContent = floatEnabled
      ? tr("dock.pin_new_short", "Закрепить")
      : tr("dock.unpin_new_short", "Открепить");
  }

  const variantSelect = document.getElementById("setting-dock-variant");
  if (variantSelect) variantSelect.value = app.state.dockVariant || "legacy";
  updateNewDock(app);
}

export function initNewDock(appRef) {
  const app = getApp(appRef);
  const prompt = document.getElementById("prompt-input");
  const newPrompt = document.getElementById("new-dock-prompt");
  const negative = document.getElementById("negative-prompt-input");
  const newNegative = document.getElementById("new-dock-negative-prompt");
  if (!app || !prompt || !newPrompt || !negative || !newNegative) return;
  renderResolutionPresets(app);

  if (!newPrompt.dataset.syncBound) {
    newPrompt.dataset.syncBound = "true";
    newPrompt.addEventListener("input", () => {
      if (prompt.value === newPrompt.value) return;
      prompt.value = newPrompt.value;
      prompt.dispatchEvent(new Event("input", { bubbles: true }));
      newPrompt.style.height = "auto";
      newPrompt.style.height = `${Math.max(36, Math.min(220, newPrompt.scrollHeight))}px`;
    });
    prompt.addEventListener("input", () => {
      if (newPrompt.value !== prompt.value) newPrompt.value = prompt.value;
    });
    newNegative.addEventListener("input", () => {
      if (negative.value === newNegative.value) return;
      negative.value = newNegative.value;
      negative.dispatchEvent(new Event("input", { bubbles: true }));
    });
    negative.addEventListener("input", () => {
      if (newNegative.value !== negative.value) newNegative.value = negative.value;
      updateNewDock(app);
    });
  }

  ["model-select", "variant-select", "resolution-select", "sampler-select", "text-encoder-select", "dock-vae-quick-select"].forEach((id) => {
    const source = document.getElementById(id);
    if (source && !source.dataset.newDockSummaryBound) {
      source.dataset.newDockSummaryBound = "true";
      source.addEventListener("change", () => updateNewDock(app));
    }
  });

  ["steps-input", "cfg-input", "noise-input", "seed-input"].forEach((id) => {
    const source = document.getElementById(id);
    if (source && !source.dataset.newDockSyncBound) {
      source.dataset.newDockSyncBound = "true";
      source.addEventListener("input", () => updateNewDock(app));
      source.addEventListener("change", () => updateNewDock(app));
    }
  });

  if (!document.documentElement.dataset.newDockPopoverBound) {
    document.documentElement.dataset.newDockPopoverBound = "true";
    document.addEventListener("pointerdown", (event) => {
      if (!activeNewDockPopover) return;
      const wrap = document.getElementById("new-dock-wrap");
      if (wrap && !wrap.contains(event.target)) closeNewDockPopover();
    });
    document.addEventListener("keydown", (event) => {
      if (event.key === "Escape" && activeNewDockPopover) closeNewDockPopover();
    });
  }

  const handle = document.getElementById("new-dock-drag-handle");
  const wrap = document.getElementById("new-dock-wrap");
  const stage = getStage();
  if (handle && wrap && stage && !handle.dataset.dragBound) {
    handle.dataset.dragBound = "true";
    let dragStart = null;
    handle.addEventListener("pointerdown", (event) => {
      if (!app.state.newDockFloat || event.button !== 0) return;
      event.preventDefault();
      const rect = wrap.getBoundingClientRect();
      dragStart = {
        pointerId: event.pointerId,
        x: event.clientX,
        y: event.clientY,
        left: rect.left - stage.getBoundingClientRect().left,
        top: rect.top - stage.getBoundingClientRect().top,
      };
      handle.setPointerCapture(event.pointerId);
      wrap.classList.add("new-dock-dragging");
    });

    handle.addEventListener("pointermove", (event) => {
      if (!dragStart || event.pointerId !== dragStart.pointerId) return;
      const bounds = getNewDockBounds();
      app.state.newDockCustomX = Math.round(Math.max(0, Math.min(bounds.maxX, dragStart.left + event.clientX - dragStart.x)));
      app.state.newDockCustomY = Math.round(Math.max(0, Math.min(bounds.maxY, dragStart.top + event.clientY - dragStart.y)));
      wrap.style.left = `${app.state.newDockCustomX}px`;
      wrap.style.top = `${app.state.newDockCustomY}px`;
    });

    const stopDrag = (event) => {
      if (!dragStart || event.pointerId !== dragStart.pointerId) return;
      dragStart = null;
      wrap.classList.remove("new-dock-dragging");
      if (app.saveDockSettings) app.saveDockSettings();
    };
    handle.addEventListener("pointerup", stopDrag);
    handle.addEventListener("pointercancel", stopDrag);
  }

  window.addEventListener("resize", () => {
    if (!app.state.newDockFloat) return;
    clampNewDockPosition(app);
    applyNewDockSettings(app);
    if (app.saveDockSettings) app.saveDockSettings();
  });

  applyNewDockSettings(app);
}

export function setDockVariant(variant, appRef) {
  const app = getApp(appRef);
  if (!app || !["legacy", "new"].includes(variant)) return;
  if (app.state.activeDockPopover && app.closeAllDockPopovers) app.closeAllDockPopovers();
  app.state.dockVariant = variant;
  if (app.applyDockSettings) app.applyDockSettings();
  else applyNewDockSettings(app);
  if (app.saveDockSettings) app.saveDockSettings();
}

export function toggleNewDockFloat(appRef) {
  const app = getApp(appRef);
  if (!app) return;
  if (app.state.dockVariant !== "new") setDockVariant("new", app);
  setNewDockFloat(!app.state.newDockFloat, app);
}

export function setNewDockFloat(enabled, appRef) {
  const app = getApp(appRef);
  const wrap = document.getElementById("new-dock-wrap");
  const stage = getStage();
  if (!app || !wrap || !stage) return;

  const shouldFloat = Boolean(enabled);
  if (shouldFloat && app.state.dockVariant !== "new") setDockVariant("new", app);
  if (shouldFloat && !app.state.newDockFloat) {
    const rect = wrap.getBoundingClientRect();
    const stageRect = stage.getBoundingClientRect();
    app.state.newDockCustomX = Math.round(rect.left - stageRect.left);
    app.state.newDockCustomY = Math.round(rect.top - stageRect.top);
  }
  app.state.newDockFloat = shouldFloat;
  if (!shouldFloat) {
    app.state.newDockCustomX = null;
    app.state.newDockCustomY = null;
  }
  if (app.applyDockSettings) app.applyDockSettings();
  else applyNewDockSettings(app);
  if (app.saveDockSettings) app.saveDockSettings();
}

export function toggleNewDockNegative() {
  const panel = document.getElementById("new-dock-negative");
  if (!panel) return;
  if (!newDockShutterOpen) setNewDockShutter(true);
  const open = panel.style.display === "none";
  panel.style.display = open ? "flex" : "none";
  const pill = document.getElementById("new-dock-negative-pill");
  if (pill) pill.setAttribute("aria-expanded", String(open));
  updateNewDock();
}

export function toggleNewDockShutter() {
  setNewDockShutter(!newDockShutterOpen);
}

function setNewDockShutter(open) {
  newDockShutterOpen = Boolean(open);
  const row = document.getElementById("new-dock-settings-row");
  const button = document.getElementById("new-dock-shutter-toggle");
  const caret = button?.querySelector(".new-dock-shutter-caret");
  const negative = document.getElementById("new-dock-negative");
  if (row) row.style.display = newDockShutterOpen ? "flex" : "none";
  if (button) {
    button.setAttribute("aria-expanded", String(newDockShutterOpen));
    button.title = newDockShutterOpen
      ? tr("dock.close_parameters", "Hide parameters")
      : tr("dock.open_parameters", "Open parameters");
  }
  if (caret) caret.textContent = newDockShutterOpen ? "⌄" : "⌃";
  if (!newDockShutterOpen) {
    closeNewDockPopover();
    if (negative) negative.style.display = "none";
    document.getElementById("new-dock-negative-pill")?.setAttribute("aria-expanded", "false");
  }
}

export function toggleNewDockPopover(name) {
  const row = document.getElementById("new-dock-settings-row");
  const target = document.getElementById(`new-dock-popover-${name}`);
  if (!row || !target) return;
  if (!newDockShutterOpen) setNewDockShutter(true);
  if (activeNewDockPopover === name) {
    closeNewDockPopover();
    return;
  }
  closeNewDockPopover();
  activeNewDockPopover = name;
  if (name === "resolution") updateNewDock(getApp());
  target.style.display = name === "model" ? "grid" : "block";
  const button = document.getElementById(`new-dock-${name}-pill`);
  button?.classList.add("active");
  button?.setAttribute("aria-expanded", "true");
  if (name === "model") {
    window.requestAnimationFrame(() => {
      if (activeNewDockPopover === "model") {
        renderModelPopover(getApp());
        getApp()?.renderLoraUI?.();
      }
    });
  }
}

export async function onNewDockModelSelect(modelId, appRef) {
  const app = getApp(appRef);
  const selector = document.getElementById("model-select");
  if (!app || !selector) return;
  if (selector.value === modelId && app.state.selectedModel === modelId) return;
  selector.value = modelId;
  app.state.selectedModel = modelId;
  updateNewDock(app);
  if (app.onModelChange) await app.onModelChange();
  if (app.state.selectedModel === modelId) updateNewDock(app);
}

export function onNewDockVariantSelect(variantId, appRef) {
  const app = getApp(appRef);
  const selector = document.getElementById("variant-select");
  if (!app || !selector) return;
  selector.value = variantId;
  app.state.selectedVariant = variantId;
  selector.dispatchEvent(new Event("change", { bubbles: true }));
  updateNewDock(app);
}

export function onNewDockParamChange(field, value, appRef) {
  const app = getApp(appRef);
  if (!app) return;
  if (field === "resolution") {
    const selector = document.getElementById("resolution-select");
    if (selector) {
      selector.value = value;
      selector.dispatchEvent(new Event("change", { bubbles: true }));
    }
    return;
  }
  if (app.syncFromDockParams) app.syncFromDockParams(field, value);
  updateNewDock(app);
}

export function swapNewDockResolution(appRef) {
  const width = Number(document.getElementById("new-dock-width")?.value);
  const height = Number(document.getElementById("new-dock-height")?.value);
  if (Number.isFinite(width) && Number.isFinite(height)) setNewDockResolution(height, width, appRef);
}

export function updateNewDockResolutionFromInputs(appRef) {
  const width = Number(document.getElementById("new-dock-width")?.value);
  const height = Number(document.getElementById("new-dock-height")?.value);
  if (!Number.isFinite(width) || !Number.isFinite(height)) return;
  setNewDockResolution(Math.max(256, Math.min(2048, width)), Math.max(256, Math.min(2048, height)), appRef);
}

export function randomizeNewDockSeed(appRef) {
  const seed = String(Math.floor(Math.random() * 2_147_483_646) + 1);
  const input = document.getElementById("new-dock-seed");
  if (input) input.value = seed;
  onNewDockParamChange("seed", seed, appRef);
}
