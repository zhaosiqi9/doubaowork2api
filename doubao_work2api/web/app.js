const state = {
  apiKey: sessionStorage.getItem("doubao_api_key") || "",
  config: null,
  accounts: [],
  logs: [],
  toastTimer: null,
  refreshTimer: null,
};

const $ = (selector) => document.querySelector(selector);
const loginView = $("#loginView");
const dashboardView = $("#dashboardView");
const connectionState = $("#connectionState");

async function api(path, options = {}) {
  const headers = new Headers(options.headers || {});
  if (state.apiKey) {
    headers.set("Authorization", `Bearer ${state.apiKey}`);
  }
  if (options.body && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }
  const response = await fetch(path, { ...options, headers });
  const text = await response.text();
  let body = null;
  try {
    body = text ? JSON.parse(text) : null;
  } catch {
    body = { raw: text };
  }
  if (!response.ok) {
    const message =
      body?.error?.message ||
      body?.detail ||
      `请求失败（HTTP ${response.status}）`;
    const error = new Error(message);
    error.status = response.status;
    throw error;
  }
  return body;
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function maskKey(key) {
  if (!key) return "未配置";
  if (key.length <= 8) return "••••••••";
  return `${key.slice(0, 4)}••••${key.slice(-4)}`;
}

function formatTime(timestamp) {
  if (!timestamp) return "—";
  return new Intl.DateTimeFormat("zh-CN", {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  }).format(new Date(timestamp * 1000));
}

function showToast(message) {
  const toast = $("#toast");
  toast.textContent = message;
  toast.classList.add("show");
  clearTimeout(state.toastTimer);
  state.toastTimer = setTimeout(() => toast.classList.remove("show"), 2800);
}

function setConnection(label, className) {
  connectionState.textContent = label;
  connectionState.className = `status-pill ${className}`;
}

function showLogin(message = "") {
  dashboardView.hidden = true;
  loginView.hidden = false;
  $("#logoutButton").hidden = true;
  $("#autoRefreshButton").hidden = true;
  setConnection("未连接", "status-idle");
  $("#loginError").hidden = !message;
  $("#loginError").textContent = message;
}

function showDashboard() {
  loginView.hidden = true;
  dashboardView.hidden = false;
  $("#logoutButton").hidden = false;
  $("#autoRefreshButton").hidden = false;
  setConnection("已连接", "status-online");
}

async function login(key) {
  state.apiKey = key.trim();
  const result = await api("/admin/login", {
    method: "POST",
    body: JSON.stringify({ api_key: state.apiKey }),
  });
  if (!result.ok) throw new Error("API Key 校验失败");
  sessionStorage.setItem("doubao_api_key", state.apiKey);
  await loadDashboard();
}

async function logout() {
  clearInterval(state.refreshTimer);
  state.refreshTimer = null;
  sessionStorage.removeItem("doubao_api_key");
  state.apiKey = "";
  showLogin();
}

async function loadDashboard() {
  const [config, accounts, logs] = await Promise.all([
    api("/admin/config"),
    api("/admin/accounts"),
    api("/admin/logs?limit=100"),
  ]);
  state.config = config;
  state.accounts = accounts.data || [];
  state.logs = logs.data || [];
  render();
}

async function loadLogs() {
  const logs = await api("/admin/logs?limit=100");
  state.logs = logs.data || [];
  renderMetrics();
  renderLogs();
}

function render() {
  showDashboard();
  renderMetrics();
  $("#defaultModel").textContent = state.config?.default_model || "—";
  $("#baseUrl").textContent = `${location.origin}/v1`;
  $("#maskedKey").textContent = maskKey(state.apiKey);
  $("#lastUpdated").textContent = `最后同步 ${new Date().toLocaleTimeString("zh-CN")}`;
  renderAccounts();
  renderLogs();
}

function renderMetrics() {
  const available = state.accounts.filter((account) => account.available);
  const cooling = state.accounts.filter(
    (account) => account.state === "cooling",
  );
  const success = state.logs.filter((log) => log.status === "success");
  $("#metricAccounts").textContent = state.accounts.length;
  $("#metricAvailable").textContent = available.length;
  $("#metricCooling").textContent = cooling.length;
  $("#metricSuccess").textContent = success.length;
}

function renderAccounts() {
  const list = $("#accountList");
  const empty = $("#accountEmpty");
  if (!state.accounts.length) {
    list.innerHTML = "";
    empty.hidden = false;
    return;
  }
  empty.hidden = true;
  list.innerHTML = state.accounts
    .map((account) => {
      const stateLabel = {
        ready: "可用",
        disabled: "已停用",
        cooling: "冷却中",
        busy: "请求中",
      }[account.state] || account.state;
      const cooldown = account.cooldown_remaining_seconds
        ? `冷却剩余 ${account.cooldown_remaining_seconds}s`
        : "无冷却";
      return `
        <article class="account-card">
          <div class="account-main">
            <div class="account-title">
              <strong title="${escapeHtml(account.id)}">${escapeHtml(account.id)}</strong>
              <span class="account-state ${escapeHtml(account.state)}">${escapeHtml(stateLabel)}</span>
            </div>
            <div class="account-meta">
              <span>device ${escapeHtml(account.device_id || "—")}</span>
              <span>web ${escapeHtml(account.web_id || "—")}</span>
              <span>${escapeHtml(cooldown)}</span>
              <span>在途 ${account.in_flight}</span>
            </div>
            ${
              account.last_error
                ? `<div class="account-error">${escapeHtml(account.last_error)}</div>`
                : ""
            }
          </div>
          <div class="account-actions">
            ${
              account.cooldown_remaining_seconds
                ? `<button class="button button-quiet button-small" data-action="clear-cooldown" data-id="${escapeHtml(account.id)}">解除冷却</button>`
                : ""
            }
            <button class="button ${account.enabled ? "button-quiet" : "button-primary"} button-small" data-action="toggle-account" data-id="${escapeHtml(account.id)}" data-enabled="${account.enabled}">
              ${account.enabled ? "停用" : "启用"}
            </button>
          </div>
        </article>
      `;
    })
    .join("");
}

function renderLogs() {
  const body = $("#logRows");
  const empty = $("#logEmpty");
  if (!state.logs.length) {
    body.innerHTML = "";
    empty.hidden = false;
    return;
  }
  empty.hidden = true;
  body.innerHTML = state.logs
    .map((log) => {
      const statusLabel = {
        success: "成功",
        error: "失败",
        running: "进行中",
      }[log.status] || log.status;
      return `
        <tr>
          <td>${formatTime(log.started_at)}</td>
          <td><span class="log-status ${escapeHtml(log.status)}">${escapeHtml(statusLabel)}</span></td>
          <td>${escapeHtml(log.model)}</td>
          <td>${escapeHtml(log.account_id || "—")}</td>
          <td>${log.stream ? "流式" : "普通"}</td>
          <td>${log.duration_ms} ms</td>
          <td title="${escapeHtml(log.error)}">${escapeHtml(log.error || "—")}</td>
        </tr>
      `;
    })
    .join("");
}

async function refresh({ silent = false } = {}) {
  const button = $("#refreshButton");
  button.disabled = true;
  try {
    await loadDashboard();
    if (!silent) showToast("数据已刷新");
  } catch (error) {
    if (error.status === 401) {
      sessionStorage.removeItem("doubao_api_key");
      state.apiKey = "";
      showLogin("API Key 已失效，请重新连接。");
    } else {
      showToast(error.message);
    }
  } finally {
    button.disabled = false;
  }
}

async function toggleAccount(id, enabled) {
  try {
    await api(`/admin/accounts/${encodeURIComponent(id)}`, {
      method: "PATCH",
      body: JSON.stringify({ enabled: !enabled }),
    });
    await refresh({ silent: true });
    showToast(enabled ? "账号已停用" : "账号已启用");
  } catch (error) {
    showToast(error.message);
  }
}

async function clearCooldown(id) {
  try {
    await api(`/admin/accounts/${encodeURIComponent(id)}/clear-cooldown`, {
      method: "POST",
    });
    await refresh({ silent: true });
    showToast("冷却已解除");
  } catch (error) {
    showToast(error.message);
  }
}

async function importAccounts(document) {
  const result = await api("/admin/accounts/import", {
    method: "POST",
    body: JSON.stringify({ account: document }),
  });
  if (result.errors?.length) {
    throw new Error(result.errors.join("；"));
  }
  return result;
}

async function runChat(event) {
  event.preventDefault();
  const submit = $("#chatSubmit");
  const prompt = $("#chatPrompt").value.trim();
  const stream = $("#streamToggle").checked;
  const output = $("#chatOutput");
  const text = $("#chatText");
  if (!prompt) {
    showToast("请输入提示词");
    return;
  }
  submit.disabled = true;
  output.hidden = false;
  text.textContent = "";
  $("#chatStatus").textContent = "请求中";
  $("#chatLatency").textContent = "";
  const started = performance.now();
  try {
    const response = await fetch("/v1/chat/completions", {
      method: "POST",
      headers: {
        Authorization: `Bearer ${state.apiKey}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model: state.config?.default_model || "deepseek-v4.1-flash",
        messages: [{ role: "user", content: prompt }],
        stream,
      }),
    });
    if (!response.ok) {
      const body = await response.json().catch(() => ({}));
      throw new Error(body?.error?.message || `HTTP ${response.status}`);
    }
    if (!stream) {
      const body = await response.json();
      text.textContent = body.choices?.[0]?.message?.content || "";
      $("#chatStatus").textContent = "完成";
    } else {
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split(/\r?\n/);
        buffer = lines.pop() || "";
        for (const line of lines) {
          if (!line.startsWith("data:")) continue;
          const raw = line.slice(5).trim();
          if (!raw || raw === "[DONE]") continue;
          try {
            const chunk = JSON.parse(raw);
            const delta = chunk.choices?.[0]?.delta || {};
            if (delta.content) text.textContent += delta.content;
            if (delta.reasoning_content) {
              text.textContent += delta.reasoning_content;
            }
          } catch {
            // Ignore malformed or non-JSON keepalive frames.
          }
        }
      }
      $("#chatStatus").textContent = "完成";
    }
    $("#chatLatency").textContent = `${Math.round(performance.now() - started)} ms`;
    await loadLogs();
  } catch (error) {
    $("#chatStatus").textContent = "失败";
    text.textContent = error.message;
    $("#chatLatency").textContent = `${Math.round(performance.now() - started)} ms`;
  } finally {
    submit.disabled = false;
  }
}

function openImport() {
  $("#importJson").value = "";
  $("#importFiles").value = "";
  $("#importError").hidden = true;
  $("#importDialog").showModal();
}

function closeImport() {
  $("#importDialog").close();
}

$("#loginForm").addEventListener("submit", async (event) => {
  event.preventDefault();
  const button = event.submitter;
  button.disabled = true;
  $("#loginError").hidden = true;
  try {
    await login($("#loginKey").value);
  } catch (error) {
    $("#loginError").hidden = false;
    $("#loginError").textContent = error.message;
  } finally {
    button.disabled = false;
  }
});

$("#logoutButton").addEventListener("click", () => {
  logout();
});

$("#refreshButton").addEventListener("click", () => refresh());

$("#autoRefreshButton").addEventListener("click", () => {
  const button = $("#autoRefreshButton");
  const enabled = button.getAttribute("aria-pressed") !== "true";
  button.setAttribute("aria-pressed", String(enabled));
  button.textContent = enabled ? "自动刷新 开" : "自动刷新 关";
  clearInterval(state.refreshTimer);
  state.refreshTimer = enabled
    ? setInterval(() => {
        if (!dashboardView.hidden) loadDashboard().catch(() => {});
      }, 5000)
    : null;
});

$("#openImportButton").addEventListener("click", openImport);

document.addEventListener("click", (event) => {
  const target = event.target.closest("[data-action]");
  if (!target) return;
  const action = target.dataset.action;
  if (action === "open-import") openImport();
  if (action === "close-import") closeImport();
  if (action === "toggle-account") {
    toggleAccount(target.dataset.id, target.dataset.enabled === "true");
  }
  if (action === "clear-cooldown") clearCooldown(target.dataset.id);
});

$("#importForm").addEventListener("submit", async (event) => {
  event.preventDefault();
  const button = event.submitter;
  button.disabled = true;
  $("#importError").hidden = true;
  try {
    const document = JSON.parse($("#importJson").value);
    const result = await importAccounts(document);
    closeImport();
    await refresh({ silent: true });
    showToast(`已导入 ${result.imported.length} 个账号`);
  } catch (error) {
    $("#importError").hidden = false;
    $("#importError").textContent = error.message;
  } finally {
    button.disabled = false;
  }
});

$("#chatForm").addEventListener("submit", runChat);

$("#importFiles").addEventListener("change", async (event) => {
  const files = [...event.target.files];
  if (!files.length) return;
  try {
    const documents = [];
    for (const file of files) {
      const parsed = JSON.parse(await file.text());
      if (Array.isArray(parsed)) documents.push(...parsed);
      else documents.push(parsed);
    }
    $("#importJson").value = JSON.stringify(
      documents.length === 1 ? documents[0] : documents,
      null,
      2,
    );
    $("#importError").hidden = true;
  } catch (error) {
    $("#importError").hidden = false;
    $("#importError").textContent = `文件解析失败：${error.message}`;
  }
});

async function boot() {
  if (!state.apiKey) {
    showLogin();
    return;
  }
  try {
    await loadDashboard();
  } catch (error) {
    if (error.status === 401) {
      await logout();
      showLogin("API Key 已失效，请重新连接。");
    } else {
      showLogin(error.message);
    }
  }
}

boot();
