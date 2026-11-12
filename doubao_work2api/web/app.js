const state = {
  apiKey: sessionStorage.getItem("doubao_api_key") || "",
  config: null,
  accounts: [],
  logs: [],
  activeTab: "overview",
  analyticsRange: "all",
  toastTimer: null,
  refreshTimer: null,
};

const PAGE_META = {
  overview: ["总览", "账号、请求和网关状态一屏看完。"],
  gateway: ["网关", "模型目录、上游映射与接入方式。"],
  accounts: ["账号", "导入、启停与冷却状态。"],
  analytics: ["用量", "请求成功率、耗时和模型分布。"],
  settings: ["设置", "运行配置与安全边界。"],
  logs: ["日志", "请求元数据与错误记录。"],
};

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];
const loginView = $("#loginView");
const appView = $("#appView");
const connectionState = $("#connectionState");

async function api(path, options = {}) {
  const headers = new Headers(options.headers || {});
  if (state.apiKey) headers.set("Authorization", `Bearer ${state.apiKey}`);
  if (options.body && !headers.has("Content-Type")) headers.set("Content-Type", "application/json");
  const response = await fetch(path, { ...options, headers });
  const text = await response.text();
  let body = null;
  try {
    body = text ? JSON.parse(text) : null;
  } catch {
    body = { raw: text };
  }
  if (!response.ok) {
    const message = body?.error?.message || body?.detail || `请求失败（HTTP ${response.status}）`;
    const error = new Error(typeof message === "string" ? message : JSON.stringify(message));
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

function formatDuration(value) {
  const ms = Number(value || 0);
  return ms >= 1000 ? `${(ms / 1000).toFixed(1)} s` : `${Math.round(ms)} ms`;
}

function showToast(message) {
  const toast = $("#toast");
  toast.textContent = message;
  toast.classList.add("show");
  clearTimeout(state.toastTimer);
  state.toastTimer = setTimeout(() => toast.classList.remove("show"), 2800);
}

function showLogin(message = "") {
  appView.hidden = true;
  loginView.hidden = false;
  $("#loginError").hidden = !message;
  $("#loginError").textContent = message;
}

function showApp() {
  loginView.hidden = true;
  appView.hidden = false;
  setConnection(true);
}

function setConnection(online) {
  connectionState.innerHTML = `<span class="status-dot"></span>${online ? "已连接" : "连接中断"}`;
  connectionState.classList.toggle("offline", !online);
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

function logout() {
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
    api("/admin/logs?limit=200"),
  ]);
  state.config = config;
  state.accounts = accounts.data || [];
  state.logs = logs.data || [];
  showApp();
  renderAll();
}

async function refresh({ silent = false } = {}) {
  try {
    await loadDashboard();
    if (!silent) showToast("数据已刷新");
  } catch (error) {
    if (error.status === 401) {
      logout();
      showLogin("API Key 已失效，请重新连接。");
      return;
    }
    setConnection(false);
    showToast(error.message);
  }
}

function switchTab(tab) {
  if (!PAGE_META[tab]) tab = "overview";
  state.activeTab = tab;
  $$(".page").forEach((page) => page.classList.toggle("active", page.id === `page-${tab}`));
  $$("[data-tab]").forEach((button) => button.classList.toggle("active", button.dataset.tab === tab));
  $("#pageTitle").textContent = PAGE_META[tab][0];
  $("#pageDescription").textContent = PAGE_META[tab][1];
  document.body.classList.remove("sidebar-open");
  if (tab === "logs") renderLogs();
  if (tab === "analytics") renderAnalytics();
  window.scrollTo({ top: 0, behavior: "smooth" });
}

function renderAll() {
  const config = state.config || {};
  $("#versionPill").textContent = `v${config.version || "0.1.0"}`;
  $("#baseUrl").textContent = `${location.origin}/v1`;
  $("#defaultModel").textContent = config.default_model || "—";
  $("#maskedKey").textContent = maskKey(state.apiKey);
  $("#lastUpdated").textContent = `最后同步 ${new Date().toLocaleTimeString("zh-CN")}`;
  renderMetrics();
  renderAccounts();
  renderCatalog();
  renderOverviewLogs();
  renderLogs();
  renderAnalytics();
  renderSettings();
}

function renderMetrics() {
  const available = state.accounts.filter((account) => account.available);
  const cooling = state.accounts.filter((account) => account.state === "cooling");
  const success = state.logs.filter((log) => log.status === "success");
  $("#metricAccounts").textContent = state.accounts.length;
  $("#metricAvailable").textContent = available.length;
  $("#metricCooling").textContent = cooling.length;
  $("#metricSuccess").textContent = success.length;
}

function renderCatalog() {
  const models = state.config?.models || [];
  $("#catalogCount").textContent = models.length;
  $("#catalogDefault").textContent = state.config?.default_model || "—";
  const select = $("#chatModel");
  const previous = select.value;
  select.innerHTML = models.map((model) => `<option value="${escapeHtml(model.id)}">${escapeHtml(model.name)} · ${escapeHtml(model.id)}</option>`).join("");
  select.value = models.some((item) => item.id === previous) ? previous : (state.config?.default_model || models[0]?.id || "");
  $("#catalogRows").innerHTML = models.map((model) => `
    <tr>
      <td><span class="badge ok">可用</span></td>
      <td><code>${escapeHtml(model.id)}</code></td>
      <td>${escapeHtml(model.name)}</td>
      <td>${model.price ? `<span class="rate-pill">${escapeHtml(model.price)}</span>` : "—"}</td>
      <td><code>${escapeHtml(model.upstream_key)}</code></td>
    </tr>
  `).join("");
}

function renderAccounts() {
  const list = $("#accountList");
  const empty = $("#accountEmpty");
  $("#accountMetricTotal").textContent = state.accounts.length;
  $("#accountMetricReady").textContent = state.accounts.filter((a) => a.available).length;
  $("#accountMetricCooling").textContent = state.accounts.filter((a) => a.state === "cooling").length;
  $("#accountMetricDisabled").textContent = state.accounts.filter((a) => a.state === "disabled").length;
  if (!state.accounts.length) {
    list.innerHTML = "";
    empty.hidden = false;
    return;
  }
  empty.hidden = true;
  list.innerHTML = state.accounts.map((account) => {
    const label = { ready: "可用", disabled: "已停用", cooling: "冷却中", busy: "请求中" }[account.state] || account.state;
    const cooldown = account.cooldown_remaining_seconds ? `冷却剩余 ${account.cooldown_remaining_seconds}s` : "无冷却";
    return `
      <article class="account-card">
        <div class="account-identity">
          <div class="account-title"><strong title="${escapeHtml(account.id)}">${escapeHtml(account.id)}</strong><span class="account-state ${escapeHtml(account.state)}">${escapeHtml(label)}</span></div>
          <div class="account-meta"><span>device ${escapeHtml(account.device_id || "—")}</span><span>web ${escapeHtml(account.web_id || "—")}</span><span>${escapeHtml(cooldown)}</span><span>在途 ${account.in_flight}</span></div>
          ${account.last_error ? `<div class="account-error">${escapeHtml(account.last_error)}</div>` : ""}
        </div>
        <div class="account-actions">
          ${account.cooldown_remaining_seconds ? `<button class="button ghost small" data-action="clear-cooldown" data-id="${escapeHtml(account.id)}">解除冷却</button>` : ""}
          <button class="button ${account.enabled ? "ghost" : "primary"} small" data-action="toggle-account" data-id="${escapeHtml(account.id)}" data-enabled="${account.enabled}">${account.enabled ? "停用" : "启用"}</button>
        </div>
      </article>`;
  }).join("");
}

function statusLabel(status) {
  return { success: "成功", error: "失败", running: "进行中" }[status] || status;
}

function renderOverviewLogs() {
  const logs = state.logs.slice(0, 6);
  $("#overviewLogsEmpty").hidden = logs.length > 0;
  $("#overviewLogs").innerHTML = logs.map((log) => `
    <div class="mini-log">
      <span class="log-status ${escapeHtml(log.status)}">${escapeHtml(statusLabel(log.status))}</span>
      <div><b>${escapeHtml(log.model)}</b><small>${escapeHtml(log.account_id || "未分配账号")} · ${log.stream ? "流式" : "普通"}</small></div>
      <span class="mini-log-time">${formatTime(log.started_at)}<small>${formatDuration(log.duration_ms)}</small></span>
    </div>
  `).join("");
}

function filteredLogs() {
  const query = ($("#logSearch")?.value || "").trim().toLowerCase();
  if (!query) return state.logs;
  return state.logs.filter((log) => [log.model, log.account_id, log.error, log.status].some((value) => String(value || "").toLowerCase().includes(query)));
}

function renderLogs() {
  const logs = filteredLogs();
  $("#logCountBadge").textContent = `${logs.length} 条`;
  $("#logEmpty").hidden = logs.length > 0;
  $("#logRows").innerHTML = logs.map((log) => `
    <tr>
      <td>${formatTime(log.started_at)}</td>
      <td><span class="log-status ${escapeHtml(log.status)}">${escapeHtml(statusLabel(log.status))}</span></td>
      <td><code>${escapeHtml(log.model)}</code></td>
      <td>${escapeHtml(log.account_id || "—")}</td>
      <td>${log.stream ? "流式" : "普通"}</td>
      <td>${formatDuration(log.duration_ms)}</td>
      <td title="${escapeHtml(log.error)}">${escapeHtml(log.error || "—")}</td>
    </tr>
  `).join("");
}

function analyticsLogs() {
  if (state.analyticsRange === "all") return state.logs;
  return state.logs.filter((log) => log.status === state.analyticsRange);
}

function renderAnalytics() {
  const logs = analyticsLogs();
  const finished = logs.filter((log) => log.status !== "running");
  const success = logs.filter((log) => log.status === "success").length;
  const streams = logs.filter((log) => log.stream).length;
  const average = finished.length ? finished.reduce((sum, log) => sum + Number(log.duration_ms || 0), 0) / finished.length : 0;
  $("#analyticsRequests").textContent = logs.length;
  $("#analyticsSuccessRate").textContent = finished.length ? `${((success / finished.length) * 100).toFixed(1)}%` : "—";
  $("#analyticsLatency").textContent = finished.length ? formatDuration(average) : "—";
  $("#analyticsStreamShare").textContent = logs.length ? `${((streams / logs.length) * 100).toFixed(0)}%` : "—";

  const byModel = new Map();
  for (const log of logs) {
    const item = byModel.get(log.model) || { model: log.model, total: 0, success: 0, error: 0, duration: 0 };
    item.total += 1;
    item[log.status] = (item[log.status] || 0) + 1;
    item.duration += Number(log.duration_ms || 0);
    byModel.set(log.model, item);
  }
  const rows = [...byModel.values()].sort((a, b) => b.total - a.total);
  const max = Math.max(1, ...rows.map((item) => item.total));
  $("#modelDistribution").innerHTML = rows.length ? rows.map((item) => `
    <div class="distribution-row"><div><code>${escapeHtml(item.model)}</code><span>${item.total}</span></div><div class="bar-track"><span style="width:${(item.total / max) * 100}%"></span></div></div>
  `).join("") : `<div class="empty-state compact"><strong>暂无数据</strong><p>发起请求后显示模型分布。</p></div>`;
  $("#analyticsRows").innerHTML = rows.length ? rows.map((item) => `
    <tr><td><code>${escapeHtml(item.model)}</code></td><td>${item.total}</td><td>${item.success || 0}</td><td>${item.error || 0}</td><td>${formatDuration(item.duration / item.total)}</td></tr>
  `).join("") : `<tr><td colspan="5" class="table-empty">暂无请求</td></tr>`;

  const recent = logs.slice(0, 24).reverse();
  const peak = Math.max(1, ...recent.map((log) => Number(log.duration_ms || 0)));
  $("#latencyChart").innerHTML = recent.length ? recent.map((log) => `
    <div class="latency-bar ${escapeHtml(log.status)}" style="height:${Math.max(4, (Number(log.duration_ms || 0) / peak) * 100)}%" title="${escapeHtml(log.model)} · ${formatDuration(log.duration_ms)}"></div>
  `).join("") : `<div class="empty-state compact"><strong>暂无数据</strong><p>趋势图会在有请求后出现。</p></div>`;
}

function renderSettings() {
  const config = state.config || {};
  $("#settingListen").textContent = config.listen || "—";
  $("#settingUpstream").textContent = config.upstream || "—";
  $("#settingDefaultModel").textContent = config.default_model || "—";
  $("#settingTimeout").textContent = config.timeout_seconds ? `${config.timeout_seconds} 秒` : "—";
  $("#settingCooldown").textContent = config.cooldown_seconds ? `${config.cooldown_seconds} 秒` : "—";
  $("#settingAccountsDir").textContent = config.accounts_dir || "—";
}

async function toggleAccount(id, enabled) {
  try {
    await api(`/admin/accounts/${encodeURIComponent(id)}`, { method: "PATCH", body: JSON.stringify({ enabled: !enabled }) });
    await refresh({ silent: true });
    showToast(enabled ? "账号已停用" : "账号已启用");
  } catch (error) {
    showToast(error.message);
  }
}

async function clearCooldown(id) {
  try {
    await api(`/admin/accounts/${encodeURIComponent(id)}/clear-cooldown`, { method: "POST" });
    await refresh({ silent: true });
    showToast("冷却已解除");
  } catch (error) {
    showToast(error.message);
  }
}

async function importAccounts(document) {
  const result = await api("/admin/accounts/import", { method: "POST", body: JSON.stringify({ account: document }) });
  if (result.errors?.length) throw new Error(result.errors.join("；"));
  return result;
}

async function scanDesktop(button) {
  const original = button.textContent;
  button.disabled = true;
  button.textContent = "扫描中…";
  try {
    const result = await api("/admin/accounts/scan-desktop", { method: "POST", body: JSON.stringify({ cdp_url: "http://127.0.0.1:9222" }) });
    await refresh({ silent: true });
    showToast(`${result.created ? "已导入" : "已更新"} ${result.account.id}（${result.cookie_count} 个 Cookie）`);
  } catch (error) {
    showToast(error.message);
  } finally {
    button.disabled = false;
    button.textContent = original;
  }
}

async function runChat(event) {
  event.preventDefault();
  const submit = $("#chatSubmit");
  const prompt = $("#chatPrompt").value.trim();
  const stream = $("#streamToggle").checked;
  const model = $("#chatModel").value;
  const output = $("#chatOutput");
  const text = $("#chatText");
  if (!prompt) return showToast("请输入提示词");
  submit.disabled = true;
  output.hidden = false;
  text.textContent = "";
  $("#chatStatus").textContent = "请求中";
  $("#chatLatency").textContent = "";
  const started = performance.now();
  try {
    const response = await fetch("/v1/chat/completions", {
      method: "POST",
      headers: { Authorization: `Bearer ${state.apiKey}`, "Content-Type": "application/json" },
      body: JSON.stringify({ model, messages: [{ role: "user", content: prompt }], stream }),
    });
    if (!response.ok) {
      const body = await response.json().catch(() => ({}));
      throw new Error(body?.error?.message || `HTTP ${response.status}`);
    }
    if (!stream) {
      const body = await response.json();
      text.textContent = body.choices?.[0]?.message?.content || "";
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
            const delta = JSON.parse(raw).choices?.[0]?.delta || {};
            if (delta.content) text.textContent += delta.content;
            if (delta.reasoning_content) text.textContent += delta.reasoning_content;
          } catch {}
        }
      }
    }
    $("#chatStatus").textContent = "完成";
    $("#chatLatency").textContent = formatDuration(performance.now() - started);
    await refresh({ silent: true });
  } catch (error) {
    $("#chatStatus").textContent = "失败";
    text.textContent = error.message;
    $("#chatLatency").textContent = formatDuration(performance.now() - started);
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

$("#logoutButton").addEventListener("click", logout);
$("#menuButton").addEventListener("click", () => document.body.classList.toggle("sidebar-open"));
$("#sidebarBackdrop").addEventListener("click", () => document.body.classList.remove("sidebar-open"));
$("#themeButton").addEventListener("click", () => {
  const dark = document.documentElement.dataset.theme !== "dark";
  document.documentElement.dataset.theme = dark ? "dark" : "light";
  localStorage.setItem("doubao_theme", dark ? "dark" : "light");
});
$("#autoRefreshButton").addEventListener("click", () => {
  const button = $("#autoRefreshButton");
  const enabled = button.getAttribute("aria-pressed") !== "true";
  button.setAttribute("aria-pressed", String(enabled));
  button.textContent = enabled ? "自动刷新 开" : "自动刷新 关";
  clearInterval(state.refreshTimer);
  state.refreshTimer = enabled ? setInterval(() => { if (!appView.hidden) refresh({ silent: true }); }, 5000) : null;
});
$("#logSearch").addEventListener("input", renderLogs);
$("#chatForm").addEventListener("submit", runChat);

document.addEventListener("click", (event) => {
  const tabButton = event.target.closest("[data-tab]");
  if (tabButton) return switchTab(tabButton.dataset.tab);
  const jump = event.target.closest("[data-tab-jump]");
  if (jump) return switchTab(jump.dataset.tabJump);
  const range = event.target.closest("[data-range]");
  if (range) {
    state.analyticsRange = range.dataset.range;
    $$("#analyticsRange button").forEach((button) => button.classList.toggle("active", button === range));
    return renderAnalytics();
  }
  const target = event.target.closest("[data-action]");
  if (!target) return;
  const action = target.dataset.action;
  if (action === "refresh") refresh();
  if (action === "logout") logout();
  if (action === "open-import") openImport();
  if (action === "scan-desktop") scanDesktop(target);
  if (action === "close-import") closeImport();
  if (action === "toggle-account") toggleAccount(target.dataset.id, target.dataset.enabled === "true");
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
    $("#importJson").value = JSON.stringify(documents.length === 1 ? documents[0] : documents, null, 2);
    $("#importError").hidden = true;
  } catch (error) {
    $("#importError").hidden = false;
    $("#importError").textContent = `文件解析失败：${error.message}`;
  }
});

async function boot() {
  const theme = localStorage.getItem("doubao_theme");
  if (theme) document.documentElement.dataset.theme = theme;
  if (!state.apiKey) return showLogin();
  try {
    await loadDashboard();
  } catch (error) {
    if (error.status === 401) {
      logout();
      showLogin("API Key 已失效，请重新连接。");
    } else {
      showLogin(error.message);
    }
  }
}

boot();
