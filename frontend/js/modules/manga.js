/* ============================================================
   QuantFunc WebUI — Manga Studio AI Controller
   B&W stylization, G-pen lineart, AI reference pipeline, compare mode.
   ============================================================ */

import { apiGet, apiPost, toast, escapeHtml } from "../api.js";

export function defaultMangaParams() {
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

export function initMangaStudio(appRef) {
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

export function populateMangaModelsAndLoras(appRef) {
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

export async function onMangaModelChange(modelId, appRef) {
  const app = appRef || window.App;
  initMangaStudio(app);
  if (app.state.manga) app.state.manga.params.model_id = modelId;
  await checkMangaLoraCompatibility(app);
}

export async function onMangaLoraChange(loraName, appRef) {
  const app = appRef || window.App;
  initMangaStudio(app);
  if (app.state.manga) app.state.manga.params.lora_name = loraName || "";
  await checkMangaLoraCompatibility(app);
}

export async function checkMangaLoraCompatibility(appRef) {
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

export function selectMangaMode(mode, appRef) {
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

export function syncMangaControlsFromParams(p) {
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

export function onMangaParamInput(key, rawVal, appRef) {
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

export function resetMangaParams(appRef) {
  const app = appRef || window.App;
  initMangaStudio(app);
  const curMode = app.state.manga.params?.mode || "manga_lite";
  app.state.manga.params = defaultMangaParams();
  selectMangaMode(curMode, app);
}

export async function runMangaPipeline(immediate = false, appRef) {
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

export function toggleMangaCompare(appRef) {
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

export function revertMangaToOriginal(appRef) {
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

export async function saveMangaToGallery(appRef) {
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

export async function downloadMangaPng(appRef) {
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

export function onMangaFileUpload(input, appRef) {
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

export function loadMangaSource(dataUrl, filename = "image.png", appRef) {
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
