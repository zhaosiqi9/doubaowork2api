const state = {
  keys: [],
  selected: new Set(),
  accountPage: 0,
  pageSize: 20,
  editAccountId: null,
  editKeyId: null,
  pendingBatch: null,
  sessionEpoch: 0,
  refreshPromise: null,
  runtimeDirty: false,
  config: null,
  accounts: [],
  logs: [],
  activeTab: "overview",
  analyticsRange: "all",
  toastTimer: null,
  refreshTimer: null,
  chatImages: [],
  chatImageEpoch: 0,
  chatImageQueue: Promise.resolve(),
  chatAbortController: null,
};

const CHAT_IMAGE_LIMITS = Object.freeze({
  count: 8,
  fileBytes: 10 * 1024 * 1024,
  totalBytes: 20 * 1024 * 1024,
});
const CHAT_IMAGE_TYPES = new Set(["image/png", "image/jpeg", "image/webp", "image/gif"]);

const PAGE_META = {
  overview: ["总览", "账号、请求和网关状态一屏看完。"],
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
  if (options.body && !headers.has("Content-Type")) headers.set("Content-Type", "application/json");
  const response = await fetch(path, { ...options, headers, credentials: "same-origin", cache: "no-store" });
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
    if (response.status === 401 && path !== "/admin/login") resetSession("管理会话已失效，请重新登录。");
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

function timestampDate(timestamp) {
  if (timestamp === null || timestamp === undefined || timestamp === "") return null;
  if (typeof timestamp === "number") {
    if (!Number.isFinite(timestamp)) return null;
    return new Date(Math.abs(timestamp) < 100000000000 ? timestamp * 1000 : timestamp);
  }
  if (typeof timestamp !== "string") return null;
  const value = timestamp.trim();
  if (!value) return null;
  if (/^\d+(?:\.\d+)?$/.test(value)) return timestampDate(Number(value));
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? null : date;
}

function formatTime(timestamp) {
  const date = timestampDate(timestamp);
  if (!date) return "—";
  return new Intl.DateTimeFormat("zh-CN", {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
    timeZone: "Asia/Shanghai",
  }).format(date);
}

function formatDuration(value) {
  const ms = Number(value || 0);
  return ms >= 1000 ? `${(ms / 1000).toFixed(1)} s` : `${Math.round(ms)} ms`;
}

function formatFileSize(bytes) {
  const value = Number(bytes);
  if (!Number.isFinite(value) || value < 0) return "—";
  if (value < 1024) return `${value} B`;
  if (value < 1024 * 1024) return `${(value / 1024).toFixed(1)} KiB`;
  return `${(value / (1024 * 1024)).toFixed(2)} MiB`;
}

function chatImageMime(file) {
  const type = String(file?.type || "").toLowerCase();
  return CHAT_IMAGE_TYPES.has(type) ? type : "";
}

function chatImageStats(images) {
  return (images || []).reduce((total, image) => ({
    count: total.count + 1,
    bytes: total.bytes + Number(image?.size || 0),
  }), { count: 0, bytes: 0 });
}

function validateChatImageFile(file, images = []) {
  const name = String(file?.name || "图片");
  const size = Number(file?.size);
  const mime = chatImageMime(file);
  if (!mime) throw new Error(`${name}：仅支持 PNG、JPEG、WebP、GIF`);
  if (!Number.isFinite(size) || size < 0) throw new Error(`${name}：无法读取文件大小`);
  if (size > CHAT_IMAGE_LIMITS.fileBytes) throw new Error(`${name}：单张不能超过 10 MiB`);
  const stats = chatImageStats(images);
  if (stats.count >= CHAT_IMAGE_LIMITS.count) throw new Error("最多选择 8 张图片");
  if (stats.bytes + size > CHAT_IMAGE_LIMITS.totalBytes) throw new Error("图片总大小不能超过 20 MiB");
  return { name, size, type: mime };
}

function chatContentParts(prompt, images = []) {
  const text = String(prompt || "").trim();
  if (!images.length) return text;
  const parts = text ? [{ type: "text", text }] : [];
  for (const image of images) {
    if (!image?.dataUrl) throw new Error(`图片 ${image?.name || ""} 尚未读取完成`);
    parts.push({ type: "image_url", image_url: { url: image.dataUrl } });
  }
  return parts;
}

function chatImageCapabilityLabel(model) {
  if (model?.image_input === true) return "已验证识图";
  return "";
}

function chatImageKey(file) {
  return `${file?.name || ""}\u0000${Number(file?.size || 0)}\u0000${file?.lastModified || 0}`;
}

function chatImageReadError(error) {
  return error?.message ? `读取失败：${error.message}` : "读取失败，请重试";
}

function readChatImage(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.addEventListener("load", () => {
      if (typeof reader.result !== "string" || !reader.result.startsWith("data:image/")) {
        reject(new Error("文件不是可识别的图片"));
        return;
      }
      resolve(reader.result);
    }, { once: true });
    reader.addEventListener("error", () => reject(reader.error || new Error("文件读取失败")), { once: true });
    reader.addEventListener("abort", () => reject(new Error("文件读取已取消")), { once: true });
    reader.readAsDataURL(file);
  });
}

function renderChatImages() {
  const list = $("#chatImageList");
  const images = state.chatImages;
  const stats = chatImageStats(images);
  list.replaceChildren();
  list.hidden = images.length === 0;
  for (const [index, image] of images.entries()) {
    const item = document.createElement("li");
    item.className = `chat-image-item${image.pending ? " pending" : ""}`;
    const thumb = document.createElement("img");
    thumb.alt = "";
    if (image.previewUrl) thumb.src = image.previewUrl;
    else thumb.hidden = true;
    const copy = document.createElement("div");
    copy.className = "chat-image-copy";
    const name = document.createElement("strong");
    name.textContent = image.name;
    name.title = image.name;
    const size = document.createElement("span");
    size.textContent = `${formatFileSize(image.size)} · 第 ${index + 1} 张${image.pending ? " · 读取中" : ""}`;
    copy.append(name, size);
    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "icon-button chat-image-remove";
    remove.dataset.chatImageRemove = image.id;
    remove.setAttribute("aria-label", `移除 ${image.name}`);
    remove.textContent = "×";
    item.append(thumb, copy, remove);
    list.append(item);
  }
  $("#chatImageSummary").textContent = images.length
    ? `${stats.count} / ${CHAT_IMAGE_LIMITS.count} 张 · ${formatFileSize(stats.bytes)} / 20 MiB`
    : "未选择图片";
  $("#chatImageClear").disabled = images.length === 0;
}

function clearChatImages() {
  state.chatImageEpoch += 1;
  state.chatImages = [];
  state.chatImageQueue = Promise.resolve();
  if ($("#chatImageInput")) $("#chatImageInput").value = "";
  if ($("#chatImageList")) {
    $("#chatImageList").replaceChildren();
    $("#chatImageList").hidden = true;
    $("#chatImageSummary").textContent = "未选择图片";
    $("#chatImageClear").disabled = true;
  }
}

function removeChatImage(id) {
  const next = state.chatImages.filter((image) => image.id !== id);
  if (next.length === state.chatImages.length) return;
  state.chatImages = next;
  renderChatImages();
}

async function addChatImageFiles(files, epoch, sessionEpoch) {
  if (epoch !== state.chatImageEpoch || sessionEpoch !== state.sessionEpoch) return;
  const selected = [...files];
  const rejected = [];
  const seen = new Set(state.chatImages.map((image) => image.key));
  const accepted = [];
  for (const file of selected) {
    if (epoch !== state.chatImageEpoch || sessionEpoch !== state.sessionEpoch) return;
    try {
      const metadata = validateChatImageFile(file, [...state.chatImages, ...accepted]);
      const key = chatImageKey(file);
      if (seen.has(key)) {
        rejected.push(`${metadata.name}：已选择`);
        continue;
      }
      seen.add(key);
      accepted.push({ ...metadata, file, key });
    } catch (error) {
      rejected.push(error.message);
    }
  }
  const feedbackElement = $("#chatImageFeedback");
  if (feedbackElement) {
    feedbackElement.textContent = rejected.join("；");
    feedbackElement.hidden = rejected.length === 0;
  }
  if (!accepted.length) return;
  for (const image of accepted) {
    if (epoch !== state.chatImageEpoch || sessionEpoch !== state.sessionEpoch) return;
    const id = `${Date.now()}-${Math.random().toString(36).slice(2)}`;
    const pending = { ...image, id, dataUrl: "", previewUrl: "", pending: true };
    state.chatImages.push(pending);
    renderChatImages();
    try {
      const dataUrl = await readChatImage(image.file);
      if (epoch !== state.chatImageEpoch || sessionEpoch !== state.sessionEpoch) return;
      const current = state.chatImages.find((item) => item.id === id);
      if (!current) continue;
      current.dataUrl = dataUrl;
      current.previewUrl = dataUrl;
      current.pending = false;
      renderChatImages();
    } catch (error) {
      if (epoch !== state.chatImageEpoch || sessionEpoch !== state.sessionEpoch) return;
      const current = state.chatImages.find((item) => item.id === id);
      if (!current) continue;
      state.chatImages = state.chatImages.filter((item) => item.id !== id);
      renderChatImages();
      showToast(chatImageReadError(error));
    }
  }
}

function enqueueChatImageFiles(files) {
  const batch = [...files];
  if (!batch.length) return;
  const epoch = state.chatImageEpoch;
  const sessionEpoch = state.sessionEpoch;
  state.chatImageQueue = state.chatImageQueue
    .catch(() => {})
    .then(() => {
      if (epoch !== state.chatImageEpoch || sessionEpoch !== state.sessionEpoch) return;
      return addChatImageFiles(batch, epoch, sessionEpoch);
    })
    .catch(() => {});
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
  $("#mobileNav").hidden = true;
  $("#loginError").hidden = !message;
  $("#loginError").textContent = message;
}

function showApp() {
  loginView.hidden = true;
  appView.hidden = false;
  $("#mobileNav").hidden = false;
  setConnection(true);
}

function setConnection(online) {
  connectionState.innerHTML = `<span class="status-dot"></span>${online ? "已连接" : "连接中断"}`;
  connectionState.classList.toggle("offline", !online);
}

async function login(password) {
  const result = await api("/admin/login", {
    method: "POST",
    body: JSON.stringify({ password }),
  });
  if (!result.ok) throw new Error("管理员密码校验失败");
  $("#loginPassword").value = "";
  await loadDashboard();
}

function resetSession(message = "") {
  state.sessionEpoch += 1;
  state.chatAbortController?.abort();
  state.chatAbortController = null;
  clearChatImages();
  clearInterval(state.refreshTimer);
  state.refreshTimer = null;
  sessionStorage.removeItem("doubao_api_key");
  state.config = null;
  state.accounts = [];
  state.keys = [];
  state.logs = [];
  state.selected.clear();
  state.pendingBatch = null;
  state.runtimeDirty = false;
  document.body.classList.remove("sidebar-open");
  $$("dialog[open]").forEach((dialog) => dialog.close());
  $("#newKeySecret").value = "";
  ["#passwordForm", "#keyForm", "#accountForm", "#importForm", "#chatForm", "#runtimeForm"].forEach((id) => $(id).reset());
  $("#chatText").textContent = "";
  $("#chatOutput").hidden = true;
  $("#chatSubmit").disabled = false;
  $("#accountList").replaceChildren();
  $("#keyList").replaceChildren();
  $("#importJson").value = "";
  $("#autoRefreshButton").setAttribute("aria-pressed", "false");
  $("#autoRefreshButton").textContent = "自动刷新 关";
  showLogin(message);
}

async function logout() {
  try {
    await api("/admin/logout", { method: "POST" });
    resetSession();
  } catch (error) {
    if (error.status !== 401) showToast(`退出失败：${error.message}，请重试`);
  }
}

async function loadDashboard() {
  const epoch = state.sessionEpoch;
  const [config, accounts, logs, keys] = await Promise.all([
    api("/admin/config"),
    api("/admin/accounts"),
    api("/admin/logs?limit=200"),
    api("/admin/keys"),
  ]);
  if (epoch !== state.sessionEpoch) return;
  state.config = config;
  state.accounts = accounts.data || [];
  state.logs = logs.data || [];
  state.keys = keys.data || [];
  state.selected = new Set([...state.selected].filter((id) => state.accounts.some((a) => a.id === id)));
  showApp();
  renderAll();
}

async function refresh({ silent = false } = {}) {
  if (state.refreshPromise) return state.refreshPromise;
  state.refreshPromise = (async () => {
  try {
    await loadDashboard();
    if (!silent) showToast("数据已刷新");
  } catch (error) {
    if (error.status === 401) {
      throw error;
    }
    setConnection(false);
    showToast(error.message);
    throw error;
  }
  })();
  try { return await state.refreshPromise; } finally { state.refreshPromise = null; }
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
  $("#lastUpdated").textContent = `最后同步 ${new Date().toLocaleTimeString("zh-CN")}`;
  renderMetrics();
  renderAccounts();
  renderCatalog();
  renderOverviewLogs();
  renderLogs();
  renderAnalytics();
  renderSettings();
  renderKeys();
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
  const imageCount = models.filter((model) => model.image_input === true).length;
  $("#catalogImageCount").textContent = imageCount;
  const select = $("#chatModel");
  const previous = select.value;
  select.innerHTML = models.map((model) => `<option value="${escapeHtml(model.id)}">${escapeHtml(model.name)} · ${escapeHtml(model.id)}</option>`).join("");
  select.value = models.some((item) => item.id === previous) ? previous : (state.config?.default_model || models[0]?.id || "");
  renderReasoningOptions();
  $("#catalogRows").innerHTML = models.map((model) => `
    <tr>
      <td><span class="badge ok">可用</span></td>
      <td><code>${escapeHtml(model.id)}</code></td>
      <td>${escapeHtml(model.name)}</td>
      <td>${model.price ? `<span class="rate-pill">${escapeHtml(model.price)}</span>` : "—"}</td>
      <td>${escapeHtml([
        model.context_window ? `${Number(model.context_window).toLocaleString()} tokens 上下文` : "",
        model.reasoning_efforts?.length ? `${model.reasoning_efforts.length} 档思考` : "",
        chatImageCapabilityLabel(model),
      ].filter(Boolean).join(" · ") || "待核实")}</td>
      <td><code>${escapeHtml(model.upstream_key)}</code></td>
    </tr>
  `).join("");
}

function renderReasoningOptions() {
  const model = state.config?.models?.find((item) => item.id === $("#chatModel").value);
  const select = $("#chatReasoning");
  const previous = select.value;
  const levels = model?.reasoning_efforts || [];
  const labels = { low: "低", medium: "中", high: "高", xhigh: "极高", max: "最高" };
  select.innerHTML = '<option value="">上游默认</option>' + levels.map((level) => `<option value="${escapeHtml(level)}">${escapeHtml(labels[level] || level)}</option>`).join("");
  select.value = levels.includes(previous) ? previous : "";
  select.disabled = !levels.length;
}

function renderAccounts() {
  const list = $("#accountList");
  const empty = $("#accountEmpty");
  $("#accountMetricTotal").textContent = state.accounts.length;
  $("#accountMetricReady").textContent = state.accounts.filter((a) => a.available).length;
  $("#accountMetricCooling").textContent = state.accounts.filter((a) => a.state === "cooling").length;
  $("#accountMetricDisabled").textContent = state.accounts.filter((a) => a.state === "disabled").length;
  updateAccountFilters();
  const results = filteredAccounts();
  state.accountPage = Math.max(0, Math.min(state.accountPage, Math.ceil(results.length / state.pageSize) - 1));
  const visible = visibleAccounts(results);
  empty.hidden = results.length > 0;
  empty.querySelector("strong").textContent = state.accounts.length ? "没有匹配的账号" : "还没有账号";
  empty.querySelector("p").textContent = state.accounts.length ? "请调整搜索或筛选条件。" : "离线扫描客户端或导入账号 JSON。";
  $("#accountsPrevious").disabled = state.accountPage === 0;
  $("#accountsNext").disabled = (state.accountPage + 1) * state.pageSize >= results.length;
  $("#accountsPage").textContent = `第 ${state.accountPage + 1} / ${Math.max(1, Math.ceil(results.length / state.pageSize))} 页`;
  list.innerHTML = visible.map((account) => {
    const label = { ready: "可用", disabled: "已停用", cooling: "冷却中", busy: "请求中" }[account.state] || account.state;
    const cooldown = account.cooldown_remaining_seconds ? `冷却剩余 ${account.cooldown_remaining_seconds}s` : "无冷却";
    return `
      <article class="account-card">
        <div class="account-identity">
          <div class="account-title"><label class="check-label"><input type="checkbox" data-account-select="${escapeHtml(account.id)}" ${state.selected.has(account.id) ? "checked" : ""} aria-label="选择账号 ${escapeHtml(account.name || account.id)}"><strong title="${escapeHtml(account.id)}">${escapeHtml(account.name || account.id)}</strong></label><span class="account-state ${escapeHtml(account.state)}">${escapeHtml(label)}</span></div>
          <div class="account-meta"><span>ID ${escapeHtml(account.id)}</span><span>分组 ${escapeHtml(account.group || "未分组")}</span><span>错误码 ${escapeHtml(account.error_code ?? "无")}</span></div>
          ${account.remark ? `<p class="account-remark">${escapeHtml(account.remark)}</p>` : ""}
          <div class="account-meta"><span>device ${escapeHtml(account.device_id || "—")}</span><span>web ${escapeHtml(account.web_id || "—")}</span><span>${escapeHtml(cooldown)}</span><span>在途 ${account.in_flight}</span></div>
          ${account.last_error ? `<div class="account-error">${escapeHtml(account.last_error)}</div>` : ""}
        </div>
        <div class="account-actions">
          <button class="button ghost small" data-action="edit-account" data-id="${escapeHtml(account.id)}">编辑信息</button>
          ${account.cooldown_remaining_seconds ? `<button class="button ghost small" data-action="clear-cooldown" data-id="${escapeHtml(account.id)}">解除冷却</button>` : ""}
          <button class="button ${account.enabled ? "ghost" : "primary"} small" data-action="toggle-account" data-id="${escapeHtml(account.id)}" data-enabled="${account.enabled}">${account.enabled ? "停用" : "启用"}</button>
        </div>
      </article>`;
  }).join("");
  renderSelection();
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
  if (!state.runtimeDirty) {
    const runtime = config.runtime || config;
    $("#settingDefaultModel").innerHTML = (config.models || []).map((model) => `<option value="${escapeHtml(model.id)}">${escapeHtml(model.name || model.id)} · ${escapeHtml(model.id)}</option>`).join("");
    $("#settingDefaultModel").value = runtime.default_model || config.default_model || "";
    $("#settingTimeout").value = runtime.timeout_seconds ?? "";
    $("#settingCooldown").value = runtime.cooldown_seconds ?? "";
  }
  $("#settingAccountsDir").textContent = config.accounts_dir || "—";
}

async function toggleAccount(id, enabled, button) {
  if (enabled && !window.confirm("确认停用 1 个账号？该账号将停止接收新请求。")) return;
  button.disabled = true;
  try {
    await api(`/admin/accounts/${encodeURIComponent(id)}`, { method: "PATCH", body: JSON.stringify({ enabled: !enabled }) });
    await refresh({ silent: true });
    showToast(enabled ? "账号已停用" : "账号已启用");
  } catch (error) {
    showToast(error.message);
  } finally {
    button.disabled = false;
  }
}

async function clearCooldown(id, button) {
  button.disabled = true;
  try {
    const result = await api("/admin/accounts/batch", { method: "POST", body: JSON.stringify({ action: "clear-cooldown", ids: [id] }) });
    if (result.errors?.length) throw new Error(batchErrorText(result.errors));
    await refresh({ silent: true });
    showToast("冷却已解除");
  } catch (error) {
    showToast(error.message);
  } finally {
    button.disabled = false;
  }
}

async function importAccounts(document) {
  const accounts = accountImportDocuments(document);
  const result = await api("/admin/accounts/import", { method: "POST", body: JSON.stringify({ accounts }) });
  if (result.errors?.length) throw new Error(result.errors.join("；"));
  return result;
}

function parseAccountImportFile(text) {
  let parsed;
  try { parsed = JSON.parse(text); }
  catch { throw new Error("文件不是有效的 JSON。"); }
  if (Array.isArray(parsed)) return parsed;
  if (parsed && typeof parsed === "object" && Array.isArray(parsed.accounts)) return parsed.accounts;
  if (parsed && typeof parsed === "object") return [parsed];
  throw new Error("文件必须包含账号对象、账号数组或 accounts 数组。");
}

function accountImportDocuments(document) {
  let value = document;
  if (typeof value === "string") {
    try { value = JSON.parse(value); }
    catch { throw new Error("账号 JSON 格式无效，请检查引号和逗号。"); }
  }
  if (value && !Array.isArray(value) && typeof value === "object" && "accounts" in value) {
    if (!Array.isArray(value.accounts)) {
      throw new Error("导出文件的 accounts 字段必须是账号数组。");
    }
    value = value.accounts;
  }
  const accounts = Array.isArray(value) ? value : [value];
  if (!accounts.length) throw new Error("没有提供账号。");
  if (accounts.some((account) => !account || typeof account !== "object" || Array.isArray(account))) {
    throw new Error("账号 JSON 必须是账号对象、账号数组，或包含 accounts 数组的导出文件。");
  }
  return accounts;
}

async function scanOffline(button) {
  const original = button.textContent;
  button.disabled = true;
  button.textContent = "扫描中…";
  try {
    const result = await api("/admin/accounts/scan-offline", { method: "POST" });
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
  const sessionEpoch = state.sessionEpoch;
  const submit = $("#chatSubmit");
  const prompt = $("#chatPrompt").value.trim();
  const stream = $("#streamToggle").checked;
  const model = $("#chatModel").value;
  const apiKey = $("#chatApiKey").value.trim();
  const output = $("#chatOutput");
  const text = $("#chatText");
  const images = state.chatImages.map((image) => ({ ...image }));
  if (!prompt && !images.length) return showToast("请输入提示词或选择图片");
  if (images.some((image) => image.pending || !image.dataUrl)) return showToast("图片仍在读取，请稍候");
  if (!apiKey) return showToast("请输入用于模型测试的 API Key");
  let content;
  try {
    content = chatContentParts(prompt, images);
  } catch (error) {
    return showToast(error.message);
  }
  submit.disabled = true;
  output.hidden = false;
  text.textContent = "";
  $("#chatStatus").textContent = "请求中";
  $("#chatLatency").textContent = "";
  const started = performance.now();
  const payload = { model, messages: [{ role: "user", content }], stream };
  if (!$("#chatReasoning").disabled && $("#chatReasoning").value) payload.reasoning_effort = $("#chatReasoning").value;
  const controller = new AbortController();
  state.chatAbortController?.abort();
  state.chatAbortController = controller;
  try {
    const response = await fetch("/v1/chat/completions", {
      method: "POST",
      credentials: "omit",
      headers: { Authorization: `Bearer ${apiKey}`, "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      signal: controller.signal,
    });
    if (sessionEpoch !== state.sessionEpoch) return;
    if (!response.ok) {
      const body = await response.json().catch(() => ({}));
      if (sessionEpoch !== state.sessionEpoch) return;
      throw new Error(body?.error?.message || `HTTP ${response.status}`);
    }
    if (!stream) {
      const body = await response.json();
      if (sessionEpoch !== state.sessionEpoch) return;
      text.textContent = body.choices?.[0]?.message?.content || "";
    } else {
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      while (true) {
        const { done, value } = await reader.read();
        if (sessionEpoch !== state.sessionEpoch) return;
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split(/\r?\n/);
        buffer = lines.pop() || "";
        for (const line of lines) {
          if (!line.startsWith("data:")) continue;
          const raw = line.slice(5).trim();
          if (!raw || raw === "[DONE]") continue;
          let chunk;
          try { chunk = JSON.parse(raw); } catch { continue; }
          if (chunk.error) throw new Error(chunk.error.message || "上游请求失败");
          const delta = chunk.choices?.[0]?.delta || {};
          if (delta.content) text.textContent += delta.content;
          if (delta.reasoning_content) text.textContent += delta.reasoning_content;
        }
      }
    }
    if (sessionEpoch !== state.sessionEpoch) return;
    $("#chatStatus").textContent = "完成";
    $("#chatLatency").textContent = formatDuration(performance.now() - started);
    await refresh({ silent: true });
  } catch (error) {
    if (sessionEpoch !== state.sessionEpoch) return;
    $("#chatStatus").textContent = "失败";
    text.textContent = error.message;
    $("#chatLatency").textContent = formatDuration(performance.now() - started);
  } finally {
    if (state.chatAbortController === controller) state.chatAbortController = null;
    if (sessionEpoch === state.sessionEpoch) submit.disabled = false;
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
    button.textContent = "登录中…";
    await login($("#loginPassword").value);
  } catch (error) {
    $("#loginError").hidden = false;
    $("#loginError").textContent = error.message;
  } finally {
    button.disabled = false;
    button.textContent = "进入控制台 →";
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
$("#chatImageInput").addEventListener("change", (event) => {
  feedback("#chatImageFeedback");
  enqueueChatImageFiles(event.target.files);
  event.target.value = "";
});
$("#chatImageClear").addEventListener("click", () => {
  clearChatImages();
  feedback("#chatImageFeedback");
});
$("#chatImageList").addEventListener("click", (event) => {
  const button = event.target.closest("[data-chat-image-remove]");
  if (!button) return;
  removeChatImage(button.dataset.chatImageRemove);
});
$("#chatImageList").addEventListener("error", (event) => {
  if (event.target.matches(".chat-image-item img")) event.target.hidden = true;
}, true);

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
  if (action === "refresh") withButton(target, "刷新中…", () => refresh());
  if (action === "logout") logout();
  if (action === "open-import") openImport();
  if (action === "scan-offline") scanOffline(target);
  if (action === "close-import") closeImport();
  if (action === "toggle-account") toggleAccount(target.dataset.id, target.dataset.enabled === "true", target);
  if (action === "clear-cooldown") clearCooldown(target.dataset.id, target);
  if (action === "edit-account") openAccount(target.dataset.id);
  if (action === "create-key") openKey();
  if (action === "edit-key") openKey(target.dataset.id);
  if (action === "toggle-key") toggleKey(target);
  if (action === "delete-key") deleteKey(target);
  if (action === "select-visible") { visibleAccounts().forEach((a) => state.selected.add(a.id)); renderAccounts(); }
  if (action === "select-results") { filteredAccounts().forEach((a) => state.selected.add(a.id)); renderAccounts(); }
  if (action === "clear-selection") { state.selected.clear(); renderAccounts(); }
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
      documents.push(...parseAccountImportFile(await file.text()));
    }
    $("#importJson").value = JSON.stringify(documents.length === 1 ? documents[0] : documents, null, 2);
    $("#importError").hidden = true;
  } catch (error) {
    $("#importError").hidden = false;
    $("#importError").textContent = `文件解析失败：${error.message}`;
  }
});

async function boot() {
  sessionStorage.removeItem("doubao_api_key");
  const theme = localStorage.getItem("doubao_theme");
  if (theme) document.documentElement.dataset.theme = theme;
  try {
    await loadDashboard();
  } catch (error) {
    if (error.status === 401) {
      showLogin();
    } else {
      showLogin(error.message);
    }
  }
}

function feedback(selector, message = "", error = false) {
  const element = $(selector);
  element.textContent = message;
  element.hidden = !message;
  element.classList.toggle("is-error", error);
}

async function withButton(button, label, operation) {
  if (button.disabled) return;
  const original = button.textContent;
  button.disabled = true;
  button.textContent = label;
  button.setAttribute("aria-busy", "true");
  try { await operation(); }
  finally {
    button.disabled = false;
    button.textContent = original;
    button.removeAttribute("aria-busy");
  }
}

function bindForm(selector, feedbackSelector, operation) {
  $(selector).addEventListener("submit", (event) => {
    event.preventDefault();
    const button = event.submitter || $(selector).querySelector('[type="submit"]');
    withButton(button, "保存中…", async () => {
      feedback(feedbackSelector);
      try { await operation(); }
      catch (error) {
        if (error.status === 401) return;
        feedback(feedbackSelector, error.message, true);
      }
    });
  });
}

function updateAccountFilters() {
  const groups = [...new Set(state.accounts.map((a) => a.group || ""))].sort();
  const codes = [...new Set(state.accounts.map((a) => String(a.error_code ?? "")))].sort();
  for (const [selector, values, all, empty] of [
    ["#accountGroup", groups, "全部分组", "未分组"],
    ["#accountErrorCode", codes, "全部错误码", "无错误码"],
  ]) {
    const select = $(selector);
    const previous = select.value;
    // Keep the active filter even if its last matching account was edited/deleted.
    const options = [...new Set([...values, ...(previous ? [previous.slice(2)] : [])])];
    const markup = `<option value="">${all}</option>` + options.map((value) => `<option value="v:${escapeHtml(value)}">${escapeHtml(value || empty)}</option>`).join("");
    if (select.innerHTML !== markup) { select.innerHTML = markup; select.value = previous; }
  }
  $("#groupSuggestions").innerHTML = groups.filter(Boolean).map((g) => `<option value="${escapeHtml(g)}"></option>`).join("");
}

function filteredAccounts() {
  const query = $("#accountSearch").value.trim().toLocaleLowerCase();
  const status = $("#accountStatus").value;
  const group = $("#accountGroup").value;
  const code = $("#accountErrorCode").value;
  return state.accounts.filter((a) =>
    (!query || [a.id, a.name, a.remark, a.group, a.error_code, a.last_error].some((v) => String(v ?? "").toLocaleLowerCase().includes(query))) &&
    (!status || a.state === status) && (!group || (a.group || "") === group.slice(2)) &&
    (!code || String(a.error_code ?? "") === code.slice(2)));
}

function visibleAccounts(results = filteredAccounts()) {
  return results.slice(state.accountPage * state.pageSize, (state.accountPage + 1) * state.pageSize);
}

function renderSelection() {
  const results = filteredAccounts();
  const matchingSelected = results.filter((a) => state.selected.has(a.id)).length;
  $("#selectionSummary").textContent = `筛选结果 ${results.length} 个 · 当前页 ${visibleAccounts(results).length} 个 · 已选 ${state.selected.size} 个（筛选外 ${state.selected.size - matchingSelected} 个）。批量操作作用于全部已选账号。`;
  $$("[data-batch]").forEach((button) => { button.disabled = state.selected.size === 0; });
}

for (const id of ["#accountSearch", "#accountStatus", "#accountGroup", "#accountErrorCode"]) {
  $(id).addEventListener(id === "#accountSearch" ? "input" : "change", () => { state.accountPage = 0; renderAccounts(); });
}
$("#accountList").addEventListener("change", (event) => {
  const id = event.target.dataset.accountSelect;
  if (!id) return;
  if (event.target.checked) state.selected.add(id); else state.selected.delete(id);
  renderSelection();
});
$("#accountsPrevious").addEventListener("click", () => { state.accountPage -= 1; renderAccounts(); });
$("#accountsNext").addEventListener("click", () => { state.accountPage += 1; renderAccounts(); });

function openAccount(id) {
  const account = state.accounts.find((a) => a.id === id);
  if (!account) return;
  state.editAccountId = id;
  $("#accountEditId").textContent = id;
  $("#accountName").value = account.name || "";
  $("#accountRemark").value = account.remark || "";
  $("#accountEditGroup").value = account.group || "";
  feedback("#accountFeedback");
  $("#accountDialog").showModal();
}
bindForm("#accountForm", "#accountFeedback", async () => {
  await api(`/admin/accounts/${encodeURIComponent(state.editAccountId)}`, { method: "PATCH", body: JSON.stringify({
    name: $("#accountName").value.trim(), remark: $("#accountRemark").value.trim(), group: $("#accountEditGroup").value.trim(),
  }) });
  $("#accountDialog").close();
  await refresh({ silent: true });
  showToast("账号信息已保存");
});

const BATCH_LABELS = { enable: "启用", disable: "停用", delete: "删除", "clear-cooldown": "清除冷却", group: "修改分组", export: "导出" };
function beginBatch(action) {
  const ids = [...state.selected];
  if (!ids.length) return;
  state.pendingBatch = { action, ids };
  $("#batchTitle").textContent = `确认${BATCH_LABELS[action]} ${ids.length} 个账号`;
  $("#batchDescription").textContent = action === "delete" ? `将永久删除所选 ${ids.length} 个账号，无法撤销。` : action === "export" ? `将下载所选 ${ids.length} 个账号的 JSON 文件，其中包含 Cookie 等凭证。请将文件保存到安全位置。` : `将对所选 ${ids.length} 个账号执行「${BATCH_LABELS[action]}」。包含筛选范围外已选账号。`;
  $("#batchGroupField").hidden = action !== "group";
  $("#batchGroup").value = "";
  $("#batchConfirm").textContent = `确认${BATCH_LABELS[action]} ${ids.length} 个账号`;
  $("#batchConfirm").classList.toggle("danger", action === "delete");
  feedback("#batchDialogFeedback");
  $("#batchDialog").showModal();
}
function batchErrorText(errors) {
  return errors.map((error) => typeof error === "string" ? error : `${error.id || "账号"}：${error.error?.message || error.error || error.message || "操作失败"}`).join("；");
}
bindForm("#batchForm", "#batchDialogFeedback", async () => {
  const pending = state.pendingBatch;
  if (!pending) return;
  const { action, ids } = pending;
  if (action === "export") {
    const data = await api("/admin/accounts/export", { method: "POST", body: JSON.stringify({ ids }) });
    if (!Array.isArray(data.accounts)) throw new Error("导出响应缺少账号列表，请重试。");
    const url = URL.createObjectURL(new Blob([JSON.stringify(data, null, 2)], { type: "application/json" }));
    const link = document.createElement("a");
    link.href = url;
    link.download = `doubao-accounts-${new Date().toISOString().replace(/[:.]/g, "-")}.json`;
    document.body.append(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
    feedback("#batchFeedback", `已导出 ${data.accounts.length} 个账号。文件包含凭证，请妥善保管。`);
  } else {
    const payload = { action, ids };
    if (action === "group") payload.group = $("#batchGroup").value.trim();
    const result = await api("/admin/accounts/batch", { method: "POST", body: JSON.stringify(payload) });
    const succeeded = result.succeeded || [];
    const errors = result.errors || [];
    succeeded.forEach((id) => state.selected.delete(id));
    feedback("#batchFeedback", `${BATCH_LABELS[action]}完成：成功 ${succeeded.length} / ${ids.length} 个，失败 ${errors.length} 个。${errors.length ? `失败项保留选择。${batchErrorText(errors)}` : ""}`, errors.length > 0);
    await refresh({ silent: true });
  }
  state.pendingBatch = null;
  $("#batchDialog").close();
});

function renderKeys() {
  $("#keysEmpty").hidden = state.keys.length > 0;
  $("#keyList").innerHTML = state.keys.map((key) => {
    const missing = missingKeyAccounts(key);
    const accountText = key.accounts?.length
      ? key.accounts.map((id) => missing.includes(id) ? `${id}（已删除，当前不可用）` : id).join("、")
      : "全部";
    return `
    <article class="account-card">
      <div class="account-identity"><div class="account-title"><strong>${escapeHtml(key.name || key.id)}</strong><span class="badge ${key.enabled ? "ok" : ""}">${key.enabled ? "已启用" : "已停用"}</span></div>
      <div class="account-meta"><span>ID ${escapeHtml(key.id)}</span><span>并发 ${escapeHtml(key.concurrency_limit || "不限")}</span><span>创建 ${escapeHtml(formatTime(key.created_at))}</span><span>最近使用 ${escapeHtml(formatTime(key.last_used_at))}</span></div>
      <p class="account-remark">模型：${escapeHtml(key.models?.length ? key.models.join("、") : "全部")}<br>账号：${escapeHtml(accountText)}${missing.length ? `<br><span class="form-error">绑定账号已删除；编辑时必须显式清除或改绑。</span>` : ""}</p></div>
      <div class="account-actions"><button type="button" class="button ghost small" data-action="edit-key" data-id="${escapeHtml(key.id)}">编辑</button><button type="button" class="button ghost small" data-action="toggle-key" data-id="${escapeHtml(key.id)}">${key.enabled ? "停用" : "启用"}</button><button type="button" class="button danger small" data-action="delete-key" data-id="${escapeHtml(key.id)}">删除</button></div>
    </article>`;
  }).join("");
}

function missingKeyAccounts(key) {
  const existing = new Set(state.accounts.map((account) => account.id));
  return (key?.accounts || []).filter((id) => !existing.has(id));
}

function selectorOptions(selector, values, selected, unavailableLabel = "当前不可用，保留配置") {
  const entries = new Map(values);
  selected.forEach((value) => { if (!entries.has(value)) entries.set(value, `${value}（${unavailableLabel}）`); });
  $(selector).innerHTML = [...entries].map(([value, label]) => `<label class="check-label"><input type="checkbox" value="${escapeHtml(value)}" ${selected.includes(value) ? "checked" : ""}><span>${escapeHtml(label)}</span></label>`).join("") || '<p class="muted small">当前没有可选项。</p>';
}
function selectedAccountsForSave(existing, checked) {
  const accounts = [...checked];
  const missing = existing
    ? missingKeyAccounts(existing).filter((accountId) => accounts.includes(accountId))
    : [];
  if (missing.length) {
    throw new Error(`此 Key 绑定的账号 ${missing.join("、")} 已删除。请显式取消该勾选以清除绑定，或改绑到现有账号；不会自动转为允许全部账号。`);
  }
  return accounts;
}
function openKey(id = null) {
  const key = state.keys.find((item) => item.id === id);
  if (id && !key) return;
  state.editKeyId = id;
  $("#keyForm").reset();
  $("#keyTitle").textContent = id ? "编辑 API Key" : "创建 API Key";
  $("#keyName").value = key?.name || "";
  $("#keyConcurrency").value = key?.concurrency_limit ?? 0;
  $("#keyEnabled").checked = key?.enabled ?? true;
  $("#keyEnabledField").hidden = !id;
  selectorOptions("#keyModels", (state.config?.models || []).map((m) => [m.id, `${m.name || m.id} · ${m.id}`]), key?.models || []);
  selectorOptions("#keyAccounts", state.accounts.map((a) => [a.id, `${a.name || a.id} · ${a.id}${a.group ? ` · ${a.group}` : ""}`]), key?.accounts || [], "已删除，当前不可用");
  feedback("#keyFeedback");
  $("#keyDialog").showModal();
}
bindForm("#keyForm", "#keyFeedback", async () => {
  const id = state.editKeyId;
  const existing = state.keys.find((key) => key.id === id);
  if (existing?.enabled && !$("#keyEnabled").checked && !window.confirm("确认停用 1 个 API Key？使用此密钥的请求将无法通过鉴权。")) return;
  const accounts = selectedAccountsForSave(existing, $$("#keyAccounts input:checked").map((input) => input.value));
  const payload = {
    name: $("#keyName").value.trim(), concurrency_limit: Number($("#keyConcurrency").value),
    models: $$("#keyModels input:checked").map((input) => input.value),
    accounts,
  };
  if (!payload.name) throw new Error("请输入密钥名称");
  if (id) payload.enabled = $("#keyEnabled").checked;
  const result = await api(id ? `/admin/keys/${encodeURIComponent(id)}` : "/admin/keys", { method: id ? "PATCH" : "POST", body: JSON.stringify(payload) });
  $("#keyDialog").close();
  if (!id) {
    if (!result.key) throw new Error("创建成功但未返回完整密钥，请重新读取密钥列表。");
    $("#newKeySecret").value = result.key;
    feedback("#copyFeedback");
    $("#keySecretDialog").showModal();
    $("#newKeySecret").select();
  }
  await refresh({ silent: true });
  showToast(id ? "API Key 已更新" : "API Key 已创建，请保存密钥");
});
$("#keySecretDialog").addEventListener("close", () => { $("#newKeySecret").value = ""; feedback("#copyFeedback"); });
$("#copyKey").addEventListener("click", () => withButton($("#copyKey"), "复制中…", async () => {
  try { await navigator.clipboard.writeText($("#newKeySecret").value); feedback("#copyFeedback", "密钥已复制，请保存到安全位置。"); }
  catch { $("#newKeySecret").focus(); $("#newKeySecret").select(); feedback("#copyFeedback", "无法自动复制，请手动复制上方已选中的密钥。", true); }
}));
async function toggleKey(button) {
  const key = state.keys.find((item) => item.id === button.dataset.id);
  if (!key || (key.enabled && !window.confirm("确认停用 1 个 API Key？使用此密钥的请求将无法通过鉴权。"))) return;
  await withButton(button, "保存中…", async () => {
    try {
      await api(`/admin/keys/${encodeURIComponent(key.id)}`, { method: "PATCH", body: JSON.stringify({ enabled: !key.enabled }) });
      await refresh({ silent: true });
      showToast(key.enabled ? "API Key 已停用" : "API Key 已启用");
    } catch (error) { showToast(error.message); }
  });
}
async function deleteKey(button) {
  const key = state.keys.find((item) => item.id === button.dataset.id);
  if (!key || !window.confirm(`确认永久删除 1 个 API Key「${key.name || key.id}」？此操作无法撤销。`)) return;
  await withButton(button, "删除中…", async () => {
    try {
      await api(`/admin/keys/${encodeURIComponent(key.id)}`, { method: "DELETE" });
      await refresh({ silent: true });
      showToast("已删除 1 个 API Key");
    } catch (error) { showToast(error.message); }
  });
}

$("#runtimeForm").addEventListener("input", () => { state.runtimeDirty = true; });
$("#runtimeForm").addEventListener("change", () => { state.runtimeDirty = true; });
bindForm("#runtimeForm", "#runtimeFeedback", async () => {
  const payload = { default_model: $("#settingDefaultModel").value, timeout_seconds: Number($("#settingTimeout").value), cooldown_seconds: Number($("#settingCooldown").value) };
  if (!payload.default_model || !Number.isFinite(payload.timeout_seconds) || payload.timeout_seconds <= 0 || !Number.isFinite(payload.cooldown_seconds) || payload.cooldown_seconds < 0) throw new Error("请选择默认模型并输入有效的超时、冷却秒数。");
  const result = await api("/admin/config", { method: "PUT", body: JSON.stringify(payload) });
  state.config = { ...state.config, ...result };
  state.runtimeDirty = false;
  renderAll();
  feedback("#runtimeFeedback", "运行配置已保存。");
});
bindForm("#passwordForm", "#passwordFeedback", async () => {
  const current = $("#currentPassword").value;
  const password = $("#newPassword").value;
  if (password !== $("#confirmPassword").value) throw new Error("两次输入的新密码不一致。");
  if (!current || !password) throw new Error("请输入当前密码和新密码。");
  if (!window.confirm("确认修改管理员密码？所有管理会话将失效，需要重新登录。")) return;
  await api("/admin/password", { method: "POST", body: JSON.stringify({ current_password: current, new_password: password }) });
  resetSession("管理员密码已修改，请使用新密码重新登录。");
  $("#loginPassword").focus();
});

document.addEventListener("click", (event) => {
  const close = event.target.closest("[data-close]");
  if (close) {
    const dialog = document.getElementById(close.dataset.close);
    if (!dialog.querySelector('[aria-busy="true"]')) dialog.close();
  }
  const batch = event.target.closest("[data-batch]");
  if (batch) beginBatch(batch.dataset.batch);
});
$$("dialog").forEach((dialog) => dialog.addEventListener("cancel", (event) => {
  if (dialog.querySelector('[aria-busy="true"]')) event.preventDefault();
}));

$("#chatModel").addEventListener("change", renderReasoningOptions);
boot();
