/* ============================================================
   QuantFunc WebUI — Settings, Accounts, Engines & System Stats
   Backend config, GPU indicators, HuggingFace tokens, Share tunnel.
   ============================================================ */

import { apiGet, apiPost, apiPatch, apiDelete, toast, escapeHtml } from "../api.js";

export function toggleTokenVisibility(inputId) {
  const input = document.getElementById(inputId);
  if (input) input.type = input.type === "password" ? "text" : "password";
}

export async function saveHfToken(appRef) {
  const app = appRef || window.App;
  const token = document.getElementById("setting-hf-token")?.value?.trim() || "";
  try {
    await apiPatch("/api/settings", { huggingface_token: token });
    if (app?.state?.settings) app.state.settings.huggingface_token = token;
    toast("success", "Токен Hugging Face сохранён ✓");
  } catch (e) {
    toast("error", "Ошибка сохранения токена: " + e.message);
  }
}

export async function testHfToken() {
  const token = document.getElementById("setting-hf-token")?.value?.trim();
  const btn = document.getElementById("btn-test-hf-token");
  if (btn) btn.textContent = "⏳ Проверка...";
  try {
    const res = await apiPost("/api/hf/test-token", { token: token || undefined });
    if (res.valid) {
      toast("success", res.message || "Токен действителен ✓");
    } else {
      toast("error", res.message || "Неверный токен Hugging Face");
    }
  } catch (e) {
    toast("error", "Ошибка проверки: " + e.message);
  } finally {
    if (btn) btn.textContent = "🔍 Проверить";
  }
}

export async function loadGuestAccounts(appRef) {
  const app = appRef || window.App;
  if (app?.state?.account?.is_guest) return;
  try {
    const res = await apiGet("/api/accounts");
    if (res && res.accounts && app) {
      app.state.guestAccounts = res.accounts;
      renderGuestAccounts(res.accounts);
    }
  } catch (e) {
    console.warn("[accounts] Could not load guest accounts:", e);
  }
}

export function renderGuestAccounts(accounts) {
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
}

export function generateRandomGuestPassword() {
  const chars = "abcdefghjkmnpqrstuvwxyz23456789";
  let pass = "guest_";
  for (let i = 0; i < 6; i++) {
    pass += chars.charAt(Math.floor(Math.random() * chars.length));
  }
  const pwdInput = document.getElementById("guest-account-password-input");
  if (pwdInput) pwdInput.value = pass;
  return pass;
}

export function showCreateGuestModal(accountId = null, currentName = "", currentPassword = "") {
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
    generateRandomGuestPassword();
  }
  modal.style.display = "flex";
  if (nameInput) nameInput.focus();
}

export function closeCreateGuestModal() {
  const modal = document.getElementById("guest-account-modal");
  if (modal) modal.style.display = "none";
}

export async function confirmCreateGuestAccount(appRef) {
  const app = appRef || window.App;
  const idInput = document.getElementById("guest-account-edit-id");
  const nameInput = document.getElementById("guest-account-username-input");
  const pwdInput = document.getElementById("guest-account-password-input");
  const username = (nameInput?.value || "").trim();
  const password = (pwdInput?.value || "").trim();

  if (!username) {
    toast("error", "Укажите имя гостя");
    return;
  }

  const editId = idInput?.value;
  try {
    if (editId) {
      await apiPatch(`/api/accounts/${encodeURIComponent(editId)}`, { username, password });
      toast("success", "Аккаунт обновлён ✓");
    } else {
      const res = await apiPost("/api/accounts", { username, password });
      toast("success", `Аккаунт '${username}' создан ✓`);
      if (res && res.account) {
        copyGuestCredentials(username, password || res.account.initial_password);
      }
    }
    closeCreateGuestModal();
    await loadGuestAccounts(app);
  } catch (e) {
    toast("error", "Ошибка: " + e.message);
  }
}

export async function deleteGuestAccount(accountId, appRef) {
  const app = appRef || window.App;
  if (!confirm("Удалить этот гостевой аккаунт? Его ссылка и пароль перестанут работать.")) return;
  try {
    await apiDelete(`/api/accounts/${encodeURIComponent(accountId)}`);
    toast("success", "Аккаунт удалён ✓");
    await loadGuestAccounts(app);
  } catch (e) {
    toast("error", "Не удалось удалить аккаунт: " + e.message);
  }
}

export function copyGuestCredentials(username, password) {
  const tunnelInput = document.getElementById("setting-share-url");
  let tunnelUrl = tunnelInput && tunnelInput.value && !tunnelInput.value.includes("...")
    ? tunnelInput.value.trim()
    : window.location.origin;

  try {
    const parsed = new URL(tunnelUrl);
    tunnelUrl = `${parsed.protocol}//${parsed.host}`;
  } catch (_) {
    tunnelUrl = tunnelUrl.split("?")[0].replace(/\/+$/, "");
  }

  const pwdText = password ? password : "(сохранённый вами пароль)";
  const text = `⚡ LazySD UI — Доступ к генератору\nСсылка: ${tunnelUrl}\nЛогин: ${username}\nПароль: ${pwdText}`;

  navigator.clipboard.writeText(text)
    .then(() => toast("success", "Данные доступа (ссылка, логин, пароль) скопированы! 📋"))
    .catch(() => prompt("Скопируйте данные для гостя:", text));
}

export function copyGuestLink() {
  const url = window.location.origin;
  navigator.clipboard.writeText(url)
    .then(() => toast("success", "Ссылка на вход скопирована! 📋"))
    .catch(() => toast("info", `Ссылка: ${url}`));
}

export function renderShareUI(appRef) {
  const app = appRef || window.App;
  const status = app?.state?.shareStatus;
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
}

export async function toggleShareTunnel(appRef) {
  const app = appRef || window.App;
  const isRunning = app?.state?.shareStatus?.running;
  const btn = document.getElementById("btn-share-toggle");
  try {
    if (isRunning) {
      if (btn) btn.textContent = "Остановка...";
      toast("info", "Остановка туннеля...");
      await apiPost("/api/share/stop", {});
      toast("info", "Туннель остановлен");
      await loadShareStatus(app);
    } else {
      if (btn) btn.textContent = "Запуск Cloudflare...";
      toast("info", "Запуск Cloudflare quick tunnel...");
      await apiPost("/api/share/start", { provider: "cloudflared" });
      for (let i = 0; i < 8; i++) {
        await new Promise((r) => setTimeout(r, 1200));
        await loadShareStatus(app);
        const st = app?.state?.shareStatus;
        if (st && st.running && st.public_url) {
          toast("success", "✓ Туннель активен: " + st.public_url);
          return;
        }
        if (st && st.error) throw new Error(st.error);
      }
      toast("info", "Туннель запущен, ссылка формируется...");
    }
  } catch (e) {
    toast("error", e.message);
    await loadShareStatus(app);
  }
}

export async function loadShareStatus(appRef) {
  const app = appRef || window.App;
  try {
    const data = await apiGet("/api/share/status");
    if (app) {
      app.state.shareStatus = data;
      renderShareUI(app);
    }
  } catch (e) {
    console.warn("Could not load share status:", e);
  }
}

export function copyShareUrl() {
  const input = document.getElementById("setting-share-url");
  if (!input || !input.value) return;
  navigator.clipboard.writeText(input.value);
  toast("success", "✓ Публичная ссылка скопирована в буфер обмена!");
}

export function generateShareToken() {
  const token = Array.from(crypto.getRandomValues(new Uint8Array(16)))
    .map(b => b.toString(16).padStart(2, "0")).join("");
  const input = document.getElementById("setting-share-token");
  if (input) input.value = token;
}

export async function saveShareToken(appRef) {
  const app = appRef || window.App;
  const input = document.getElementById("setting-share-token");
  const val = input ? input.value.trim() : "";
  try {
    await apiPost("/api/share/token", { token: val || null });
    toast("success", "✓ Токен гостя сохранён");
    await loadShareStatus(app);
  } catch (e) {
    toast("error", e.message);
  }
}

export async function saveShareAutoStart(checked) {
  try {
    await apiPost("/api/settings", { share_auto_start: checked });
    toast("info", `Автозапуск туннеля: ${checked ? "ВКЛ" : "ВЫКЛ"}`);
  } catch (e) {
    toast("error", e.message);
  }
}

export async function refreshExtrasStatus(showToast = false, appRef) {
  const app = appRef || window.App;
  try {
    const data = await apiGet("/api/extras/status?refresh=true");
    if (app) {
      app.state.extrasStatus = data;
      renderExtrasUI(app);
    }
    if (showToast) {
      const isReady = data.env?.ready;
      const msg = isReady
        ? `✓ Окружение Extras готово: Python ${data.env?.python || ''}, PyTorch ${data.env?.torch || ''}`
        : `Окружение Extras: ${data.env?.summary || 'требуется установка компонентов'}`;
      toast(isReady ? "success" : "info", msg);
    }
  } catch (e) {
    if (showToast) toast("error", "Не удалось опросить статус Extras: " + e.message);
  }
}

export function renderExtrasUI(appRef) {
  const app = appRef || window.App;
  const data = app?.state?.extrasStatus;
  if (!data) return;

  const nunchakuBadge = document.getElementById("setting-nunchaku-status");
  const orbitBadge = document.getElementById("setting-orbitquant-status");
  const envDesc = document.getElementById("setting-extras-env-desc");
  const customPyInput = document.getElementById("setting-extras-custom-py");

  if (customPyInput && data.env?.python_path) {
    if (!customPyInput.value && !data.env.managed) customPyInput.value = data.env.python_path;
  }

  if (envDesc && data.env) {
    const e = data.env;
    if (e.ready) {
      envDesc.innerHTML = `<span style="color: #4ade80; font-weight: 600;">✓ Готово к работе:</span> Python ${e.python || '3.12'} · Torch ${e.torch || 'OK'} · CUDA ${e.cuda || 'OK'} (${e.device || 'GPU'}) · Diffusers ${e.diffusers || 'OK'}`;
    } else if (e.python_path || e.venv_exists) {
      envDesc.innerHTML = `<span style="color: #fbbf24; font-weight: 600;">⚠️ Настраивается:</span> ${e.error || e.summary || 'Окружение создано, ожидается установка PyTorch и Diffusers'}`;
    } else {
      envDesc.innerHTML = `<span style="color: var(--text-dim);">ℹ️ ${e.summary || 'Изолированное окружение (envs/extras) пока не создано.'}</span>`;
    }
  }

  if (nunchakuBadge) {
    nunchakuBadge.innerHTML = data.nunchaku_ready
      ? `<span class="state-badge installed">✓ Установлен (${data.env?.nunchaku || 'ready'})</span>`
      : '<span class="state-badge missing">Не установлен</span>';
  }

  if (orbitBadge) {
    orbitBadge.innerHTML = data.orbitquant_ready
      ? `<span class="state-badge installed">✓ Установлен (${data.env?.orbitquant || 'ready'})</span>`
      : '<span class="state-badge missing">Не установлен</span>';
  }
}

export async function saveExtrasPythonPath(appRef) {
  const app = appRef || window.App;
  const input = document.getElementById("setting-extras-custom-py");
  const val = input ? input.value.trim() : "";
  try {
    toast("info", "Сохранение пути к Python...");
    await apiPost("/api/extras/python", { path: val });
    toast("success", "✓ Путь к Extras Python обновлён");
    await refreshExtrasStatus(false, app);
  } catch (e) {
    toast("error", "Ошибка: " + e.message);
  }
}

export async function installExtrasPackage(target, appRef) {
  const app = appRef || window.App;
  const btn = document.getElementById(`btn-install-${target}`);
  const origText = btn ? btn.textContent : `Установить ${target}`;
  if (btn) {
    btn.disabled = true;
    btn.textContent = "Установка...";
  }

  const labelTarget = target === "runtime" ? "базового окружения" : target;
  toast("info", `Запущена установка ${labelTarget}...`);

  try {
    const res = await apiPost("/api/extras/install", { component: target, target: target });
    const jobId = res.job?.id || res.id;
    if (!jobId) throw new Error(res.error || "Не удалось получить ID задачи установки");

    const poll = setInterval(async () => {
      try {
        const job = await apiGet(`/api/extras/jobs/${jobId}`);
        const state = job.state || job.status;
        const label = job.step_label || job.stage || 'Установка';
        if (btn) btn.textContent = `${label}...`;

        if (state === "done" || state === "succeeded") {
          clearInterval(poll);
          if (btn) {
            btn.disabled = false;
            btn.textContent = origText;
          }
          toast("success", `✓ Установка ${labelTarget} завершена!`);
          await refreshExtrasStatus(false, app);
        } else if (state === "error" || state === "failed" || state === "cancelled") {
          clearInterval(poll);
          if (btn) {
            btn.disabled = false;
            btn.textContent = origText;
          }
          toast("error", `Ошибка установки ${target}: ${job.error || "failed"}`);
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
    toast("error", e.message);
  }
}

export function renderGpuIndicator(appRef) {
  const app = appRef || window.App;
  const dot = document.getElementById("gpu-indicator-dot");
  const text = document.getElementById("gpu-indicator-text");
  const badge = document.getElementById("gpu-tier-badge");
  const info = app?.state?.gpuInfo;

  if (!dot || !text) return;
  if (!info || !info.name) {
    dot.className = "status-dot unknown";
    text.textContent = "No GPU";
    return;
  }

  dot.className = "status-dot ok";
  text.textContent = `${info.name} · ${info.vram_gb} GB`;
  if (badge && info.tier) {
    badge.textContent = `Tier: ${info.tier.toUpperCase()}`;
    badge.className = `tier-badge tier-${info.tier}`;
  }
}

export function renderSystemStats(stats) {
  const vramEl = document.getElementById("stat-vram");
  const ramEl = document.getElementById("stat-ram");
  const cpuEl = document.getElementById("stat-cpu");
  if (!stats) return;

  if (vramEl && stats.vram_used_gb !== undefined && stats.vram_total_gb !== undefined) {
    const pct = stats.vram_total_gb > 0 ? (stats.vram_used_gb / stats.vram_total_gb * 100).toFixed(0) : 0;
    vramEl.textContent = `${stats.vram_used_gb.toFixed(1)} / ${stats.vram_total_gb.toFixed(1)} GB (${pct}%)`;
  }
  if (ramEl && stats.ram_used_gb !== undefined && stats.ram_total_gb !== undefined) {
    const pct = stats.ram_total_gb > 0 ? (stats.ram_used_gb / stats.ram_total_gb * 100).toFixed(0) : 0;
    ramEl.textContent = `${stats.ram_used_gb.toFixed(1)} / ${stats.ram_total_gb.toFixed(1)} GB (${pct}%)`;
  }
  if (cpuEl && stats.cpu_percent !== undefined) {
    cpuEl.textContent = `${stats.cpu_percent.toFixed(0)}%`;
  }
}

export async function verifyEngine(version) {
  const ver = version || '0.0.17';
  try {
    toast("info", `Verifying engine v${ver}...`);
    const res = await apiPost("/api/engine/verify", { version: ver });
    toast(res.status === "ok" ? "success" : "warning", res.message);
  } catch (e) {
    toast("error", e.message);
  }
}

export async function downloadEngine(version, event, appRef) {
  const app = appRef || window.App;
  const ver = version || '0.0.17';
  const btn = event?.target;
  if (btn) { btn.disabled = true; btn.textContent = "Downloading..."; }
  try {
    toast("info", `Downloading QuantFunc v${ver} (~120 MB)...`);
    const res = await apiPost("/api/engine/download", { engine: "quantfunc", version: ver });
    toast("success", res.message);
    if (app && app.refreshConfig) await app.refreshConfig();
  } catch (e) {
    toast("error", "Engine download failed: " + e.message);
  }
}
