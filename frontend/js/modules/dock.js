/* ============================================================
   QuantFunc WebUI — Floating Dock & Control Popovers
   Gemini-style floating bar, quick pills, popovers, and free drag mode.
   ============================================================ */

import { toast } from "../api.js";

export function autoResizePrompt(textarea) {
  if (!textarea) return;
  textarea.style.height = "auto";
  const nextH = Math.max(46, Math.min(150, textarea.scrollHeight));
  textarea.style.height = `${nextH}px`;
}

export function updateDockPills(appRef) {
  const app = appRef || window.App;
  if (!app) return;

  // 1. Model Pill Text
  const modelPillText = document.getElementById("dock-model-pill-text");
  const activeModel = app.state.models.find(m => m.id === app.state.selectedModel);
  if (modelPillText) {
    if (activeModel) {
      const shortVar = (app.state.selectedVariant || "")
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
}

export function toggleDockPopover(name, event, appRef) {
  if (event) event.stopPropagation();
  const app = appRef || window.App;
  if (!app) return;

  const map = {
    model: "dock-popover-model",
    res: "dock-popover-res",
    params: "dock-popover-params",
    presets: "dock-popover-presets",
    lora: "dock-popover-lora",
  };

  const targetId = map[name];
  if (!targetId) return;

  if (app.state.activeDockPopover === name) {
    closeAllDockPopovers(app);
    return;
  }

  closeAllDockPopovers(app);
  const el = document.getElementById(targetId);
  if (el) {
    updateDockPills(app);
    if (name === "presets" && app.renderPresetsUI) app.renderPresetsUI();
    if (name === "lora" && app.renderLoraUI) app.renderLoraUI();
    el.style.display = "block";
    app.state.activeDockPopover = name;
  }
}

export function closeAllDockPopovers(appRef) {
  const app = appRef || window.App;
  ["dock-popover-model", "dock-popover-res", "dock-popover-params", "dock-popover-presets", "dock-popover-lora"].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.style.display = "none";
  });
  if (app) app.state.activeDockPopover = null;
}

export async function onDockModelSelect(modelId, appRef) {
  const app = appRef || window.App;
  const mainModelSel = document.getElementById("model-select");
  if (mainModelSel) {
    mainModelSel.value = modelId;
    if (app && app.onModelChange) await app.onModelChange();
  }
}

export function onDockVariantSelect(variantId) {
  const mainVarSel = document.getElementById("variant-select");
  if (mainVarSel) {
    mainVarSel.value = variantId;
    mainVarSel.dispatchEvent(new Event("change"));
  }
}

export function onDockTeSelect(teId, appRef) {
  const app = appRef || window.App;
  const mainTeSel = document.getElementById("text-encoder-select");
  if (mainTeSel) {
    mainTeSel.value = teId;
    if (app && app.onTextEncoderChange) app.onTextEncoderChange();
  }
}

export function selectResolutionChip(resValue, appRef) {
  const app = appRef || window.App;
  const resSel = document.getElementById("resolution-select");
  if (resSel) {
    resSel.value = resValue;
    updateDockPills(app);
  }
  closeAllDockPopovers(app);
}

export function syncFromDockParams(field, val, appRef) {
  const app = appRef || window.App;
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
  if (app && app.updateModelTrigger) app.updateModelTrigger();
}

export function initDockSettings(appRef) {
  const app = appRef || window.App;
  try {
    const raw = localStorage.getItem("lazysd_dock");
    const cfg = raw ? JSON.parse(raw) : {};
    if (cfg.position) app.state.dockPosition = cfg.position;
    if (cfg.opacity !== undefined) app.state.dockOpacity = cfg.opacity;
    if (cfg.blur !== undefined) app.state.dockBlur = cfg.blur;
    if (cfg.minimized !== undefined) app.state.dockMinimized = cfg.minimized;
    if (cfg.dragEnabled !== undefined) app.state.dockDragEnabled = cfg.dragEnabled;
    if (cfg.customX !== undefined) app.state.dockCustomX = cfg.customX;
    if (cfg.customY !== undefined) app.state.dockCustomY = cfg.customY;
  } catch (_) {}

  applyDockSettings(app);
  initDockDrag(app);
}

export function applyDockSettings(appRef) {
  const app = appRef || window.App;
  const wrap = document.getElementById("gemini-dock-wrap");
  if (!wrap) return;

  wrap.classList.remove("dock-pos-left", "dock-pos-right", "dock-pos-bottom", "dock-pos-center", "dock-minimized", "dock-drag-active");

  if (app.state.dockDragEnabled && app.state.dockCustomX !== undefined && app.state.dockCustomY !== undefined) {
    wrap.classList.add("dock-free-drag");
    wrap.style.left = `${app.state.dockCustomX}px`;
    wrap.style.top = `${app.state.dockCustomY}px`;
    wrap.style.transform = "none";
    wrap.style.bottom = "auto";
    wrap.style.right = "auto";
  } else {
    wrap.classList.remove("dock-free-drag");
    wrap.style.left = "";
    wrap.style.top = "";
    wrap.style.transform = "";
    wrap.style.bottom = "";
    wrap.style.right = "";
    if (app.state.dockPosition && app.state.dockPosition !== "center") {
      wrap.classList.add(`dock-pos-${app.state.dockPosition}`);
    }
  }

  if (app.state.dockMinimized) {
    wrap.classList.add("dock-minimized");
  }

  const root = document.documentElement;
  root.style.setProperty("--dock-opacity", app.state.dockOpacity);
  root.style.setProperty("--dock-blur", `${app.state.dockBlur}px`);

  const posSelect = document.getElementById("setting-dock-pos");
  if (posSelect) posSelect.value = app.state.dockPosition || "center";

  const opSlider = document.getElementById("setting-dock-opacity");
  const opLabel = document.getElementById("setting-dock-opacity-val");
  if (opSlider) opSlider.value = app.state.dockOpacity;
  if (opLabel) opLabel.textContent = `${Math.round(app.state.dockOpacity * 100)}%`;

  const blurSlider = document.getElementById("setting-dock-blur");
  const blurLabel = document.getElementById("setting-dock-blur-val");
  if (blurSlider) blurSlider.value = app.state.dockBlur;
  if (blurLabel) blurLabel.textContent = `${app.state.dockBlur}px`;
}

export function saveDockSettings(appRef) {
  const app = appRef || window.App;
  try {
    localStorage.setItem("lazysd_dock", JSON.stringify({
      position: app.state.dockPosition,
      opacity: app.state.dockOpacity,
      blur: app.state.dockBlur,
      minimized: app.state.dockMinimized,
      dragEnabled: app.state.dockDragEnabled,
      customX: app.state.dockCustomX,
      customY: app.state.dockCustomY,
    }));
  } catch (_) {}
}

export function setDockPosition(pos, appRef) {
  const app = appRef || window.App;
  app.state.dockPosition = pos;
  app.state.dockDragEnabled = false;
  delete app.state.dockCustomX;
  delete app.state.dockCustomY;
  applyDockSettings(app);
  saveDockSettings(app);
}

export function cycleDockPosition(appRef) {
  const app = appRef || window.App;
  const order = ["center", "left", "right", "bottom"];
  const curIdx = order.indexOf(app.state.dockPosition || "center");
  const next = order[(curIdx + 1) % order.length];
  setDockPosition(next, app);
  toast("info", `Позиция дока: ${next}`);
}

export function toggleDockMinimized(appRef) {
  const app = appRef || window.App;
  app.state.dockMinimized = !app.state.dockMinimized;
  applyDockSettings(app);
  saveDockSettings(app);
}

export function setDockOpacity(val, appRef) {
  const app = appRef || window.App;
  const num = parseFloat(val);
  if (isNaN(num)) return;
  app.state.dockOpacity = Math.max(0.2, Math.min(1.0, num));
  applyDockSettings(app);
  saveDockSettings(app);
}

export function setDockBlur(val, appRef) {
  const app = appRef || window.App;
  const num = parseInt(val, 10);
  if (isNaN(num)) return;
  app.state.dockBlur = Math.max(0, Math.min(40, num));
  applyDockSettings(app);
  saveDockSettings(app);
}

export function toggleDockDragMode(appRef) {
  const app = appRef || window.App;
  app.state.dockDragEnabled = !app.state.dockDragEnabled;
  if (!app.state.dockDragEnabled) {
    delete app.state.dockCustomX;
    delete app.state.dockCustomY;
  }
  applyDockSettings(app);
  saveDockSettings(app);
  toast("info", app.state.dockDragEnabled ? "Режим свободного перетаскивания Дока ВКЛ" : "Режим привязки Дока (Dock Presets) ВКЛ");
}

let dragInitialized = false;
export function initDockDrag(appRef) {
  if (dragInitialized) return;
  dragInitialized = true;
  const app = appRef || window.App;
  const wrap = document.getElementById("gemini-dock-wrap");
  const handle = document.getElementById("dock-drag-handle");
  if (!wrap) return;

  let isDragging = false;
  let startX = 0;
  let startY = 0;
  let origX = 0;
  let origY = 0;

  if (handle) {
    handle.addEventListener("dblclick", (e) => {
      e.stopPropagation();
      setDockPosition("center", app);
      toast("info", "Док сброшен в нижний центр");
    });
  }

  const startDrag = (e) => {
    const isFromHandle = handle && (e.target === handle || handle.contains(e.target));
    if (!isFromHandle && !app.state.dockDragEnabled) return;
    if (e.target.closest("button") || e.target.closest("input") || e.target.closest("textarea") || e.target.closest("select")) {
      return;
    }
    app.state.dockDragEnabled = true;
    isDragging = true;
    startX = e.clientX;
    startY = e.clientY;
    const rect = wrap.getBoundingClientRect();
    origX = rect.left;
    origY = rect.top;
    try { wrap.setPointerCapture(e.pointerId); } catch (_) {}
    wrap.classList.add("dock-dragging");
    wrap.classList.add("dock-free-drag");
  };

  wrap.addEventListener("pointerdown", startDrag);

  wrap.addEventListener("pointermove", (e) => {
    if (!isDragging || !app.state.dockDragEnabled) return;
    const dx = e.clientX - startX;
    const dy = e.clientY - startY;
    const maxX = window.innerWidth - wrap.offsetWidth;
    const maxY = window.innerHeight - wrap.offsetHeight;
    const nx = Math.max(8, Math.min(maxX - 8, origX + dx));
    const ny = Math.max(8, Math.min(maxY - 8, origY + dy));
    app.state.dockCustomX = Math.round(nx);
    app.state.dockCustomY = Math.round(ny);
    wrap.style.left = `${app.state.dockCustomX}px`;
    wrap.style.top = `${app.state.dockCustomY}px`;
    wrap.style.transform = "none";
    wrap.style.bottom = "auto";
    wrap.style.right = "auto";
  });

  const stopDrag = (e) => {
    if (!isDragging) return;
    isDragging = false;
    wrap.classList.remove("dock-dragging");
    try { wrap.releasePointerCapture(e.pointerId); } catch (_) {}
    saveDockSettings(app);
  };

  wrap.addEventListener("pointerup", stopDrag);
  wrap.addEventListener("pointercancel", stopDrag);
}
