/* ============================================================
   QuantFunc WebUI — Model Catalog, Hub, Importer & Encoders
   Model grid, modal picker, SVDQ/Nunchaku filter, download/delete.
   ============================================================ */

import { apiGet, apiPost, apiDelete, toast, escapeHtml } from "../api.js";

export function sortModelsList(models) {
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
}

export function openModelPickerModal(appRef) {
  const app = appRef || window.App;
  const modal = document.getElementById("model-picker-modal");
  if (!modal) return;
  if (app) app.state.isModelPickerOpen = true;
  modal.style.display = "flex";
  renderModelPickerModal(app);
  const searchInput = document.getElementById("model-picker-search");
  if (searchInput) {
    searchInput.value = app?.state?.modelPickerSearch || "";
    searchInput.focus();
  }
}

export function closeModelPickerModal(appRef) {
  const app = appRef || window.App;
  const modal = document.getElementById("model-picker-modal");
  if (!modal) return;
  if (app) app.state.isModelPickerOpen = false;
  modal.style.display = "none";
}

export function onModelPickerSearch(query, appRef) {
  const app = appRef || window.App;
  if (app) app.state.modelPickerSearch = (query || "").toLowerCase().trim();
  renderModelPickerModal(app);
}

export function setModelPickerFilter(filter, btnEl, appRef) {
  const app = appRef || window.App;
  if (app) app.state.modelPickerFilter = filter;
  document.querySelectorAll("#model-picker-filters .filter-chip").forEach(el => {
    el.classList.toggle("active", el === btnEl);
  });
  renderModelPickerModal(app);
}

export function renderModelPickerModal(appRef) {
  const app = appRef || window.App;
  const grid = document.getElementById("model-picker-grid");
  if (!grid || !app) return;

  if (!app.state.models || !app.state.models.length) {
    grid.innerHTML = `<div class="placeholder-text">Загрузка...</div>`;
    return;
  }

  const query = app.state.modelPickerSearch || "";
  const filter = app.state.modelPickerFilter || "all";

  const filtered = sortModelsList(app.state.models).filter(m => {
    if (query) {
      const matchesName = m.name?.toLowerCase().includes(query);
      const matchesFamily = m.family?.toLowerCase().includes(query);
      const matchesDesc = m.description?.toLowerCase().includes(query);
      if (!matchesName && !matchesFamily && !matchesDesc) return false;
    }

    if (filter === "hidden") {
      return m.hidden_from_host === true || m.hidden_for_guests === true;
    } else if (m.hidden_from_host === true) {
      return false;
    }

    if (filter === "installed") return m.installed === true;
    if (filter === "custom") return m.is_custom === true;
    if (filter === "klein") return m.family === "klein" || m.id.includes("klein");
    if (filter === "qwen") return m.family === "qwen" || m.id.includes("qwen");
    if (filter === "z-image") return m.family === "z-image" || m.id.includes("z_image");
    if (filter === "svdq") {
      return m.engine === "nunchaku"
        || (m.tags && m.tags.includes("svdq"))
        || (m.family && m.family.includes("svdq"))
        || (m.transformers && m.transformers.some(t => t.backend === "nunchaku" || t.id.includes("svdq")));
    }
    if (filter === "sd_cpp") return m.engine === "sd_cpp" || m.id.includes("gguf");
    if (filter === "sdnq") return m.engine === "sdnq" || m.id.includes("sdnq");
    return true;
  });

  if (!filtered.length) {
    grid.innerHTML = `<div class="placeholder-text" style="grid-column: 1 / -1; padding: 40px 0;">Модели не найдены</div>`;
    return;
  }

  grid.innerHTML = "";
  for (const model of filtered) {
    const isSelected = model.id === app.state.selectedModel;
    const isDisabled = model.disabled === true;
    const card = document.createElement("div");
    card.className = `model-picker-card ${isSelected ? "active" : ""} ${isDisabled ? "model-card-qf-disabled" : ""}`;

    let engineBadge = "";
    if (model.is_custom) {
      engineBadge = `<span class="badge-custom">🎨 Custom</span>`;
    } else if (model.engine === "sd_cpp") {
      engineBadge = `<span class="badge-engine sd-cpp" style="background: rgba(16, 185, 129, 0.15); color: #34d399; border-color: rgba(16, 185, 129, 0.3);">⚡ SD.CPP</span>`;
    } else if (model.engine === "sdnq") {
      engineBadge = `<span class="badge-engine sdnq" style="background: rgba(139, 92, 246, 0.18); color: #c4b5fd; border-color: rgba(139, 92, 246, 0.35);">⚡ SDNQ</span>`;
    } else if (model.engine === "nunchaku") {
      engineBadge = `<span class="badge-engine nunchaku" style="background: rgba(59, 130, 246, 0.18); color: #93c5fd; border-color: rgba(59, 130, 246, 0.35);">⚡ Nunchaku</span>`;
    } else if (model.engine === "orbitquant") {
      engineBadge = `<span class="badge-engine orbitquant" style="background: rgba(236, 72, 153, 0.18); color: #f472b6; border-color: rgba(236, 72, 153, 0.35);">⚡ OrbitQuant</span>`;
    }

    let installedBadge = "";
    if (isDisabled) {
      installedBadge = `<span class="badge-unsupported">⚠️ Не поддерживается</span>`;
    } else if (model.installed) {
      installedBadge = `<span class="badge-installed">✓ Готова</span>`;
    } else {
      installedBadge = `<span class="badge-missing">⬇ Требует скачивания</span>`;
    }

    card.innerHTML = `
      <div class="model-picker-card-header">
        <span class="model-picker-card-title">${escapeHtml(model.name)}</span>
        <div style="display: flex; gap: 6px; align-items: center;">
          ${engineBadge}
          ${installedBadge}
        </div>
      </div>
      <div class="model-picker-card-desc">${escapeHtml(model.description || "")}</div>
      <div class="model-picker-card-footer">
        <span style="font-size: 11px; color: var(--text-dim);">${escapeHtml(model.family)} · ${model.pipeline || ""}</span>
        <div style="display: flex; gap: 6px;">
          ${model.installed
            ? `<button class="card-action-btn secondary" onclick="event.stopPropagation(); App.openModelFolder('${escapeHtml(model.id)}')" title="Открыть папку">📁</button>
               <button class="card-action-btn danger" onclick="event.stopPropagation(); App.deleteModelFiles('${escapeHtml(model.id)}')" title="Удалить с ПК">🗑️</button>`
            : `<button class="card-action-btn download" onclick="event.stopPropagation(); App.downloadAllModelVariants('${escapeHtml(model.id)}')">⬇ Скачать</button>`}
        </div>
      </div>
    `;

    card.addEventListener("click", () => {
      selectModelFromModal(model.id, app);
    });
    grid.appendChild(card);
  }
}

export async function selectModelFromModal(modelId, appRef) {
  const app = appRef || window.App;
  closeModelPickerModal(app);
  const mainModelSel = document.getElementById("model-select");
  if (mainModelSel) {
    mainModelSel.value = modelId;
    if (app && app.onModelChange) await app.onModelChange();
  }
}

export function renderModelsTab(appRef) {
  const app = appRef || window.App;
  const listEl = document.getElementById("models-list");
  if (!listEl || !app) return;

  if (!app.state.models.length) {
    listEl.innerHTML = `<div class="placeholder-text">Загрузка...</div>`;
    return;
  }

  const filter = app.state.modelsTabFilter || "all";
  if (filter === "encoders") {
    renderEncodersTab(listEl, app);
    return;
  }

  const query = app.state.modelsTabSearch || "";
  const filtered = sortModelsList(app.state.models).filter(m => {
    if (query) {
      const q = query.toLowerCase();
      const matches = (m.name && m.name.toLowerCase().includes(q))
        || (m.family && m.family.toLowerCase().includes(q))
        || (m.description && m.description.toLowerCase().includes(q))
        || (m.engine && m.engine.toLowerCase().includes(q));
      if (!matches) return false;
    }
    if (filter === "installed") return m.installed === true;
    if (filter === "custom") return m.is_custom === true;
    if (filter === "svdq") {
      return m.engine === "nunchaku"
        || (m.tags && m.tags.includes("svdq"))
        || (m.family && m.family.includes("svdq"))
        || (m.transformers && m.transformers.some(t => t.backend === "nunchaku" || t.id.includes("svdq")));
    }
    if (filter === "sd_cpp") return m.engine === "sd_cpp" || m.id.includes("gguf");
    if (filter === "sdnq") return m.engine === "sdnq" || m.id.includes("sdnq");
    if (filter === "klein") return m.family === "klein" || m.family === "flux" || m.id.includes("flux") || m.id.includes("klein");
    if (filter === "qwen") return m.family === "qwen" || m.id.includes("qwen");
    if (filter === "z-image") return m.family === "z-image" || m.id.includes("z_image");
    return true;
  });

  if (!filtered.length) {
    listEl.innerHTML = `<div class="placeholder-text" style="padding: 40px 0;">Модели не найдены</div>`;
    return;
  }

  listEl.innerHTML = "";
  const grid = document.createElement("div");
  grid.className = "models-grid";

  for (const model of filtered) {
    const card = document.createElement("div");
    card.className = "model-card";
    card.dataset.modelId = model.id;

    let engineBadge = "";
    if (model.is_custom) {
      engineBadge = `<span class="badge-custom">🎨 Custom</span>`;
    } else if (model.engine === "sd_cpp") {
      engineBadge = `<span class="badge-engine sd-cpp" style="background: rgba(16, 185, 129, 0.15); color: #34d399;">⚡ SD.CPP</span>`;
    } else if (model.engine === "sdnq") {
      engineBadge = `<span class="badge-engine sdnq" style="background: rgba(139, 92, 246, 0.18); color: #c4b5fd;">⚡ SDNQ</span>`;
    } else if (model.engine === "nunchaku") {
      engineBadge = `<span class="badge-engine nunchaku" style="background: rgba(59, 130, 246, 0.18); color: #93c5fd;">⚡ Nunchaku</span>`;
    }

    let installedBadge = model.installed
      ? `<span class="badge-installed">✓ Готова</span>`
      : `<span class="badge-missing">⬇ Не скачана</span>`;

    let actions = "";
    if (model.installed) {
      actions = `
        <button class="card-action-btn secondary" onclick="App.openModelFolder('${escapeHtml(model.id)}')" title="Открыть папку с файлами">📁 Папка</button>
        <button class="card-action-btn danger" onclick="App.deleteModelFiles('${escapeHtml(model.id)}')" title="Удалить скачанные веса">🗑️ Удалить с ПК</button>
      `;
    } else {
      actions = `
        <button class="card-action-btn download" style="flex: 1;" onclick="App.downloadAllModelVariants('${escapeHtml(model.id)}')">⬇ Скачать модель</button>
      `;
    }

    card.innerHTML = `
      <div>
        <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 6px;">
          <h3 class="model-title">${escapeHtml(model.name)}</h3>
          <div style="display: flex; gap: 4px;">
            ${engineBadge}
            ${installedBadge}
          </div>
        </div>
        <div class="model-desc">${escapeHtml(model.description || "")}</div>
        <div style="margin-top: 8px; font-size: 11px; color: var(--text-dim);">${escapeHtml(model.family)} · ${model.pipeline || ""}</div>
      </div>
      <div style="display: flex; gap: 8px; margin-top: 14px;">
        ${actions}
      </div>
    `;
    grid.appendChild(card);
  }

  listEl.appendChild(grid);
}

export function onModelsTabSearch(query, appRef) {
  const app = appRef || window.App;
  if (app) app.state.modelsTabSearch = (query || "").toLowerCase().trim();
  renderModelsTab(app);
}

export function setModelsTabFilter(filter, btnEl, appRef) {
  const app = appRef || window.App;
  if (app) app.state.modelsTabFilter = filter;
  document.querySelectorAll("#models-filters .filter-chip").forEach(el => {
    el.classList.toggle("active", el === btnEl);
  });
  renderModelsTab(app);
}

export async function renderEncodersTab(listEl, appRef) {
  const app = appRef || window.App;
  listEl.innerHTML = `<div class="placeholder-text">Загрузка энкодеров...</div>`;
  try {
    const encoders = await apiGet("/api/encoders");
    if (!encoders || !encoders.length) {
      listEl.innerHTML = `<div class="placeholder-text" style="padding: 40px 0;">Энкодеры не найдены</div>`;
      return;
    }

    const query = (app?.state?.modelsTabSearch || "").toLowerCase().trim();
    const filtered = encoders.filter(e => {
      if (!query) return true;
      return (e.name && e.name.toLowerCase().includes(query))
        || (e.file && e.file.toLowerCase().includes(query))
        || (e.repo_id && e.repo_id.toLowerCase().includes(query));
    });

    if (!filtered.length) {
      listEl.innerHTML = `<div class="placeholder-text" style="padding: 40px 0;">Ничего не найдено</div>`;
      return;
    }

    listEl.innerHTML = "";
    const grid = document.createElement("div");
    grid.className = "encoders-grid";

    for (const enc of filtered) {
      const card = document.createElement("div");
      card.className = "encoder-card";
      const statusBadge = enc.installed
        ? `<span class="badge-installed">✓ Установлен</span>`
        : `<span class="badge-missing">⬇ Не скачан</span>`;
      const sizeStr = enc.size_gb ? `${enc.size_gb.toFixed(1)} GB` : "";

      let actionBtn = enc.installed
        ? `<button class="card-action-btn active" style="width: 100%;" disabled>✓ Установлен в кэш</button>`
        : `<button class="card-action-btn download" style="width: 100%;" onclick="App.downloadSharedEncoder('${escapeHtml(enc.repo_id)}', '${escapeHtml(enc.file_path || enc.file)}', '${escapeHtml(enc.id)}', this)">⬇ Скачать энкодер (${sizeStr})</button>`;

      card.innerHTML = `
        <div>
          <div class="encoder-card-header">
            <span class="encoder-title">${escapeHtml(enc.name || enc.file)}</span>
            ${statusBadge}
          </div>
          <div class="encoder-meta">
            <span>📁 Файл: <code>${escapeHtml(enc.file)}</code></span>
            <span>🌐 Репозиторий: <code>${escapeHtml(enc.repo_id)}</code></span>
            <span>⚙ Формат: <code>${escapeHtml(enc.precision || 'int4')}</code> · ${sizeStr}</span>
          </div>
        </div>
        <div style="margin-top: 12px;">
          ${actionBtn}
        </div>
      `;
      grid.appendChild(card);
    }
    listEl.appendChild(grid);
  } catch (err) {
    listEl.innerHTML = `<div class="placeholder-text" style="color: #f87171;">Ошибка: ${escapeHtml(err.message)}</div>`;
  }
}

export async function openModelFolder(modelId) {
  try {
    const res = await apiPost(`/api/models/${modelId}/open-folder`);
    if (res.status === "remote") {
      toast("info", res.message);
    } else {
      toast("success", `Папка открыта: ${res.path}`);
    }
  } catch (e) {
    toast("error", "Не удалось открыть папку: " + e.message);
  }
}

export async function deleteModelFiles(modelId, appRef) {
  const app = appRef || window.App;
  const model = app?.state?.models.find(m => m.id === modelId);
  const name = model?.name || modelId;
  if (!confirm(`Удалить скачанные веса модели "${name}" с диска?\nЭто освободит место на диске. Модель останется в списке, но её нужно будет скачать заново.`)) {
    return;
  }
  try {
    const res = await apiDelete(`/api/models/${modelId}/files`);
    toast("success", `Файлы модели "${name}" удалены. Освобождено: ${res.freed_mb || 0} МБ ✓`);
    if (app && app.refreshConfig) await app.refreshConfig();
    if (app && app.onModelChange) await app.onModelChange();
    renderModelsTab(app);
  } catch (e) {
    toast("error", "Ошибка при удалении: " + e.message);
  }
}

export async function downloadAllModelVariants(modelId, appRef) {
  const app = appRef || window.App;
  const model = app?.state?.models.find(m => m.id === modelId);
  if (!model) return;
  toast("info", `Запущено скачивание модели ${model.name}...`);
  try {
    const res = await apiPost(`/api/models/${modelId}/download`, { include_base: true, include_transformer: true });
    if (res.job_id && app && app.startDownloadPolling) {
      app.startDownloadPolling(res.job_id);
    }
  } catch (e) {
    toast("error", "Ошибка запуска скачивания: " + e.message);
  }
}

export async function downloadSharedEncoder(repoId, filePath, teId, btnEl, appRef) {
  const app = appRef || window.App;
  if (btnEl) {
    btnEl.disabled = true;
    btnEl.textContent = "Downloading...";
  }
  try {
    const result = await apiPost("/api/encoders/download", {
      repo_id: repoId,
      file_path: filePath,
    });
    toast("info", "Скачивание энкодера начато...");
    if (result.job_id && app && app.startDownloadPolling) {
      app.startDownloadPolling(result.job_id);
    }
  } catch (e) {
    toast("error", "Ошибка скачивания: " + e.message);
    if (btnEl) {
      btnEl.disabled = false;
      btnEl.textContent = "⬇ Скачать";
    }
  }
}

export function openCustomModelModal(tab = "local") {
  const modal = document.getElementById("custom-model-modal");
  if (!modal) return;
  switchCustomModelTab(tab);
  modal.style.display = "flex";
}

export function closeCustomModelModal() {
  const modal = document.getElementById("custom-model-modal");
  if (modal) modal.style.display = "none";
}

export function switchCustomModelTab(tab) {
  const isHf = tab === "hf";
  const tabLocal = document.getElementById("importer-tab-btn-local");
  const tabHf = document.getElementById("importer-tab-btn-hf");
  const paneLocal = document.getElementById("custom-model-pane-local");
  const paneHf = document.getElementById("custom-model-pane-hf");

  if (tabLocal) tabLocal.classList.toggle("active", !isHf);
  if (tabHf) tabHf.classList.toggle("active", isHf);
  if (paneLocal) paneLocal.style.display = !isHf ? "block" : "none";
  if (paneHf) paneHf.style.display = isHf ? "block" : "none";
}
