/* ============================================================
   QuantFunc WebUI — LoRA Hub & Preset Adapters
   Management of local LoRAs, trigger words, metadata, downloading.
   ============================================================ */

import { apiGet, apiPost, apiDelete, toast, escapeHtml } from "../api.js";

export async function loadLoras(appRef) {
  const app = appRef || window.App;
  try {
    const list = await apiGet("/api/loras");
    if (app) {
      app.state.loraLibrary = list || [];
      renderLoraUI(app);
    }
  } catch (e) {
    console.warn("Failed to load LoRAs:", e.message);
  }
}

export function renderLoraUI(appRef) {
  const app = appRef || window.App;
  if (!app) return;

  const count = app.state.activeLoras.length;
  const pillText = document.getElementById("dock-lora-pill-text");
  if (pillText) pillText.textContent = `LoRA (${count})`;

  const dockCount = document.getElementById("dock-lora-count");
  if (dockCount) dockCount.textContent = `${count} активных`;

  const inspectorCount = document.getElementById("inspector-lora-count");
  if (inspectorCount) inspectorCount.textContent = String(count);

  const dockList = document.getElementById("dock-lora-list");
  const dockEmpty = document.getElementById("dock-lora-empty");
  if (dockList && dockEmpty) {
    if (app.state.loraLibrary.length === 0) {
      dockEmpty.style.display = "block";
      dockList.innerHTML = "";
    } else {
      dockEmpty.style.display = "none";
      dockList.innerHTML = app.state.loraLibrary.map(item => {
        const activeItem = app.state.activeLoras.find(l => l.name === item.name);
        const isActive = !!activeItem;
        const strength = activeItem ? activeItem.strength : 1.0;
        const family = item.family ? `<span class="lora-meta-tag">${escapeHtml(item.family)}</span>` : "";
        const size = item.size_mb ? `<span class="lora-meta-tag">${item.size_mb} MB</span>` : "";
        const trigger = item.trigger ? `<button type="button" class="dock-mini-btn" onclick="event.stopPropagation(); App.insertLoraTrigger('${escapeHtml(item.trigger)}')" title="Вставить триггер-слова в промпт" style="font-size: 10px; padding: 1px 5px; height: auto; color: #93c5fd; background: rgba(59, 130, 246, 0.1);">💬 ${escapeHtml(item.trigger)}</button>` : "";

        return `
          <div class="lora-card-item ${isActive ? 'active' : ''}">
            <div class="lora-card-header">
              <div class="lora-title-group" onclick="App.toggleLora('${escapeHtml(item.name)}')" style="cursor: pointer;">
                <input type="checkbox" ${isActive ? 'checked' : ''} onclick="event.stopPropagation(); App.toggleLora('${escapeHtml(item.name)}')">
                <span class="lora-name-text" title="${escapeHtml(item.name)}">${escapeHtml(item.name)}</span>
                ${family}
                ${size}
              </div>
              <div style="display: flex; gap: 4px; align-items: center;">
                <button type="button" class="dock-ctrl-btn" onclick="App.editLoraMetadata('${escapeHtml(item.name)}', '${escapeHtml(item.family || '')}', '${escapeHtml(item.trigger || '')}')" title="Настроить архитектуру и триггеры">⚙️</button>
                <button type="button" class="dock-ctrl-btn" onclick="App.deleteLora('${escapeHtml(item.name)}')" title="Удалить файл LoRA">✕</button>
              </div>
            </div>
            ${trigger ? `<div style="margin: 3px 0 2px 20px;">${trigger}</div>` : ''}
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
    if (app.state.activeLoras.length === 0) {
      inspectorList.innerHTML = '<div style="font-size: 11px; color: var(--text-dim); text-align: center; padding: 6px 0;">Нет подключенных LoRA</div>';
    } else {
      inspectorList.innerHTML = app.state.activeLoras.map(item => {
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
  if (modalTotal) modalTotal.textContent = String(app.state.loraLibrary.length);

  const hubCount = document.getElementById("models-hub-lora-count");
  if (hubCount) hubCount.textContent = String(app.state.loraLibrary.length);

  filterLoraLibrary(app);
  renderLoraHubGrid(app);
}

export function switchModelsHubSubtab(subtab, appRef) {
  const app = appRef || window.App;
  const isLoras = subtab === "loras";
  const btnModels = document.getElementById("models-subtab-btn-models");
  const btnLoras = document.getElementById("models-subtab-btn-loras");
  const panelModels = document.getElementById("models-subpanel-models");
  const panelLoras = document.getElementById("models-subpanel-loras");
  const addModelBtn = document.getElementById("models-hub-add-model-btn");
  const addLoraBtn = document.getElementById("models-hub-add-lora-btn");

  if (btnModels) btnModels.classList.toggle("active", !isLoras);
  if (btnLoras) btnLoras.classList.toggle("active", isLoras);
  if (panelModels) panelModels.style.display = isLoras ? "none" : "block";
  if (panelLoras) panelLoras.style.display = isLoras ? "block" : "none";
  if (addModelBtn) addModelBtn.style.display = isLoras ? "none" : "inline-flex";
  if (addLoraBtn) addLoraBtn.style.display = isLoras ? "inline-flex" : "none";

  if (isLoras) {
    loadLoras(app);
  } else if (app && app.renderModelsTab) {
    app.renderModelsTab();
  }
}

export function renderLoraHubGrid(appRef) {
  const app = appRef || window.App;
  if (!app) return;
  const grid = document.getElementById("lora-hub-grid");
  if (!grid) return;

  const allLoras = app.state.loraLibrary || [];
  const q = (document.getElementById("lora-hub-search")?.value || "").trim().toLowerCase();
  let list = allLoras;
  if (q) {
    list = list.filter(l => l.name.toLowerCase().includes(q) || (l.trigger || "").toLowerCase().includes(q) || (l.family || "").toLowerCase().includes(q));
  }

  if (list.length === 0) {
    grid.innerHTML = '<div style="grid-column: 1 / -1; padding: 40px; text-align: center; color: var(--text-dim);">LoRA не найдены</div>';
    return;
  }

  grid.innerHTML = list.map(item => {
    const activeItem = app.state.activeLoras.find(l => l.name === item.name);
    const isActive = !!activeItem;
    const strength = activeItem ? activeItem.strength : 1.0;
    const family = item.family ? `<span class="model-badge" style="background: rgba(59, 130, 246, 0.15); color: #93c5fd; border: 1px solid rgba(59, 130, 246, 0.3);">${escapeHtml(item.family)}</span>` : "";
    const size = item.size_mb ? `<span class="model-badge" style="background: rgba(255, 255, 255, 0.05); color: var(--text-muted);">${item.size_mb} MB</span>` : "";

    return `
      <div class="model-card ${isActive ? 'active' : ''}" style="display: flex; flex-direction: column; justify-content: space-between; min-height: 140px;">
        <div>
          <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 6px;">
            <div style="font-weight: 600; font-size: 13.5px; color: var(--text); word-break: break-all;" title="${escapeHtml(item.name)}">${escapeHtml(item.name)}</div>
            <div style="display: flex; gap: 4px;">
              <button class="dock-ctrl-btn" onclick="App.editLoraMetadata('${escapeHtml(item.name)}', '${escapeHtml(item.family || '')}', '${escapeHtml(item.trigger || '')}')" title="Настройки">⚙️</button>
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
}

export function filterLoraLibrary(appRef) {
  const app = appRef || window.App;
  if (!app) return;
  const listEl = document.getElementById("lora-modal-list");
  if (!listEl) return;
  const q = (document.getElementById("lora-modal-search")?.value || "").trim().toLowerCase();
  let list = app.state.loraLibrary || [];
  if (q) {
    list = list.filter(l => l.name.toLowerCase().includes(q) || (l.trigger || "").toLowerCase().includes(q));
  }
  if (list.length === 0) {
    listEl.innerHTML = '<div style="padding: 20px; text-align: center; color: var(--text-dim);">Ничего не найдено</div>';
    return;
  }
  listEl.innerHTML = list.map(item => {
    const activeItem = app.state.activeLoras.find(l => l.name === item.name);
    const isActive = !!activeItem;
    return `
      <div class="lora-card-item ${isActive ? 'active' : ''}" style="margin-bottom: 6px;">
        <div class="lora-card-header">
          <div class="lora-title-group" onclick="App.toggleLora('${escapeHtml(item.name)}')" style="cursor: pointer;">
            <input type="checkbox" ${isActive ? 'checked' : ''} onclick="event.stopPropagation(); App.toggleLora('${escapeHtml(item.name)}')">
            <span class="lora-name-text">${escapeHtml(item.name)}</span>
          </div>
          <button type="button" class="dock-ctrl-btn" onclick="App.deleteLora('${escapeHtml(item.name)}')">✕</button>
        </div>
      </div>
    `;
  }).join("");
}

export function insertLoraTrigger(trigger) {
  const promptInput = document.getElementById("prompt-input");
  if (!promptInput || !trigger) return;
  const cur = promptInput.value.trim();
  promptInput.value = cur ? `${cur}, ${trigger}` : trigger;
  toast("info", `Триггер добавлен: ${trigger}`);
}

export function toggleLora(name, appRef) {
  const app = appRef || window.App;
  if (!app) return;
  const idx = app.state.activeLoras.findIndex(l => l.name === name);
  if (idx >= 0) {
    app.state.activeLoras.splice(idx, 1);
  } else {
    app.state.activeLoras.push({ name, strength: 1.0 });
  }
  renderLoraUI(app);
}

export function setLoraStrength(name, val, appRef) {
  const app = appRef || window.App;
  if (!app) return;
  const num = parseFloat(val);
  if (isNaN(num)) return;
  const item = app.state.activeLoras.find(l => l.name === name);
  if (item) {
    item.strength = num;
    const lbl1 = document.getElementById(`lora-weight-val-${name}`);
    if (lbl1) lbl1.textContent = num.toFixed(2);
    const lbl2 = document.getElementById(`lora-weight-insp-${name}`);
    if (lbl2) lbl2.textContent = num.toFixed(2);
  }
}

export async function deleteLora(name, appRef) {
  const app = appRef || window.App;
  if (!confirm(`Удалить LoRA файл ${name}?`)) return;
  try {
    await apiDelete(`/api/loras/${encodeURIComponent(name)}`);
    toast("success", `LoRA ${name} удалена`);
    const idx = app?.state?.activeLoras.findIndex(l => l.name === name);
    if (idx !== undefined && idx >= 0) app.state.activeLoras.splice(idx, 1);
    await loadLoras(app);
  } catch (e) {
    toast("error", "Ошибка удаления LoRA: " + e.message);
  }
}

export function openLoraManageModal(tab = "hub") {
  const modal = document.getElementById("lora-modal");
  if (modal) {
    modal.style.display = "flex";
    switchLoraModalTab(tab);
  }
}

export function openLoraModal() {
  openLoraManageModal("hub");
}

export function closeLoraModal() {
  const modal = document.getElementById("lora-modal");
  if (modal) modal.style.display = "none";
}

export function switchLoraModalTab(tab) {
  const tabs = ["hub", "import", "custom"];
  tabs.forEach(tName => {
    const btn = document.getElementById(`lora-tab-btn-${tName}`);
    const panel = document.getElementById(`lora-tab-panel-${tName}`);
    if (btn) btn.classList.toggle("active", tName === tab);
    if (panel) panel.style.display = tName === tab ? "block" : "none";
  });
}
