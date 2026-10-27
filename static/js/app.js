"use strict";

/**
 * Fire & Smoke Vision dashboard.
 * Toàn bộ số liệu hiển thị lấy từ /api/status, /api/history và /api/analytics.
 * Không có detection hoặc KPI giả trong file này.
 */

const ACTIVE_STATES = new Set(["starting", "running", "paused", "stopping"]);
const IMAGE_EXTENSIONS = new Set(["jpg", "jpeg", "png", "bmp", "webp"]);
const VIDEO_EXTENSIONS = new Set(["mp4", "avi", "mov", "mkv", "webm"]);
const MAX_IMAGE_BYTES = 25 * 1024 * 1024;
const MAX_VIDEO_BYTES = 300 * 1024 * 1024;

const PANELS = {
  media: {
    eyebrow: "Nguồn dữ liệu",
    title: "Image / Video",
    subtitle: "Xử lý ảnh tĩnh hoặc video bằng model đã train.",
  },
  history: {
    eyebrow: "Nhật ký",
    title: "Detection History",
    subtitle: "Các sự kiện cảnh báo thật trong phiên hiện tại.",
  },
  analytics: {
    eyebrow: "Thống kê",
    title: "Analytics",
    subtitle: "Số liệu đo từ pipeline inference đang chạy.",
  },
  alerts: {
    eyebrow: "Giám sát",
    title: "Alerts",
    subtitle: "Cảnh báo cháy/khói và âm thanh thông báo.",
  },
  yolo: {
    eyebrow: "Cấu hình",
    title: "YOLO Settings",
    subtitle: "Tinh chỉnh inference mà không nạp lại model.",
  },
  system: {
    eyebrow: "Cấu hình",
    title: "System Settings",
    subtitle: "Camera, luồng hiển thị và chính sách cảnh báo.",
  },
};

const state = {
  status: null,
  model: null,
  settings: null,
  health: null,
  activeSection: "live",
  activePanel: null,
  drawerOpen: false,
  alertsInitialized: false,
  lastAlertId: 0,
  history: [],
  alertEvents: [],
  soundEnabled: false,
  pollFailures: 0,
  polling: false,
  modelRefreshing: false,
  selectedFile: null,
  lastAnalyticsAt: 0,
  streamReconnectTimer: null,
};

const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => Array.from(root.querySelectorAll(selector));

document.addEventListener("DOMContentLoaded", init);

/* ------------------------------------------------------------------ utils */
async function api(path, options = {}) {
  const config = { method: options.method || "GET", headers: {} };
  if (options.body instanceof FormData) {
    config.body = options.body;
  } else if (options.body !== undefined) {
    config.headers["Content-Type"] = "application/json";
    config.body = JSON.stringify(options.body);
  }

  const response = await fetch(path, config);
  const contentType = response.headers.get("content-type") || "";
  let payload = null;
  if (contentType.includes("application/json")) {
    payload = await response.json().catch(() => null);
  } else {
    payload = await response.text().catch(() => "");
  }

  if (!response.ok) {
    const message =
      (payload && typeof payload === "object" && payload.error) ||
      (typeof payload === "string" && payload.trim()) ||
      `Yêu cầu thất bại (HTTP ${response.status})`;
    throw new Error(message);
  }
  return payload;
}

function kindOf(name) {
  const lowered = String(name || "").toLowerCase();
  if (lowered.includes("fire")) return "fire";
  if (lowered.includes("smoke")) return "smoke";
  return "other";
}

function formatBytes(bytes) {
  if (!Number.isFinite(bytes) || bytes <= 0) return "0 B";
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function formatPercent(value) {
  return typeof value === "number" ? `${Math.round(value * 100)}%` : "—";
}

function formatTime(iso) {
  if (!iso) return "—";
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return "—";
  return date.toLocaleTimeString("vi-VN", { hour12: false });
}

function formatUptime(seconds) {
  if (typeof seconds !== "number" || seconds < 0) return "—";
  const total = Math.floor(seconds);
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const secs = total % 60;
  if (hours > 0) return `${hours}h ${String(minutes).padStart(2, "0")}m`;
  return `${minutes}m ${String(secs).padStart(2, "0")}s`;
}

function classEntries(names) {
  return Object.entries(names || {})
    .map(([id, name]) => [Number(id), String(name)])
    .sort((a, b) => a[0] - b[0]);
}

/* ------------------------------------------------------------------ toast */
function toast(message, type = "info", timeout = 5000) {
  const container = $("#toasts");
  if (!container) return;

  const item = document.createElement("div");
  item.className = `toast toast--${type}`;

  const iconMap = {
    success: "#i-check",
    error: "#i-alert",
    warning: "#i-alert",
    info: "#i-info",
  };
  const ns = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(ns, "svg");
  svg.setAttribute("class", "icon toast__icon");
  svg.setAttribute("aria-hidden", "true");
  const use = document.createElementNS(ns, "use");
  use.setAttribute("href", iconMap[type] || "#i-info");
  svg.appendChild(use);

  const text = document.createElement("p");
  text.textContent = message;

  const close = document.createElement("button");
  close.type = "button";
  close.className = "toast__close";
  close.setAttribute("aria-label", "Đóng thông báo");
  close.textContent = "×";

  const remove = () => {
    if (!item.isConnected) return;
    item.classList.add("is-out");
    window.setTimeout(() => item.remove(), 240);
  };
  close.addEventListener("click", remove);

  item.append(svg, text, close);
  container.appendChild(item);

  if (timeout > 0) window.setTimeout(remove, timeout);
  while (container.children.length > 4) container.firstElementChild.remove();
}

/* --------------------------------------------------------------- confirm */
function confirmDialog({ title, text, confirmLabel = "Xác nhận", danger = true }) {
  return new Promise((resolve) => {
    const dialog = $("#confirmDialog");
    $("#confirmTitle").textContent = title;
    $("#confirmText").textContent = text;
    const okButton = $("#confirmOk");
    okButton.textContent = confirmLabel;
    okButton.className = `btn ${danger ? "btn--danger" : "btn--primary"}`;

    const onClose = () => {
      dialog.removeEventListener("close", onClose);
      resolve(dialog.returnValue === "ok");
    };
    if (dialog.open) {
      resolve(false);
      return;
    }
    dialog.addEventListener("close", onClose, { once: true });
    dialog.showModal();
  });
}

function setButtonBusy(button, busy, busyLabel = "Đang xử lý…") {
  if (!button) return;
  if (!button.dataset.idleLabel) button.dataset.idleLabel = button.textContent.trim();
  const label = button.querySelector("span:last-child") || button;
  if (busy) {
    button.disabled = true;
    button.classList.add("btn--busy");
    label.textContent = busyLabel;
  } else {
    button.disabled = false;
    button.classList.remove("btn--busy");
    label.textContent = button.dataset.idleLabel;
  }
}

/* --------------------------------------------------------------- boot */
async function init() {
  setupStream();
  setupDrawer();
  setupSheet();
  setupControls();
  setupMediaPanel();
  setupHistoryPanel();
  setupAnalyticsPanel();
  setupAlertsPanel();
  setupForms();
  setupFullscreen();
  setupKeyboard();

  state.soundEnabled = readPreference("fv:sound") === "1";
  const soundToggle = $("#soundToggle");
  if (soundToggle) soundToggle.checked = state.soundEnabled;

  await loadHealth();
  await loadModel();
  await loadSettings();
  await pollStatus();
  window.setInterval(pollStatus, 1000);
}

function readPreference(key) {
  try {
    return window.localStorage.getItem(key);
  } catch (error) {
    console.warn("Không đọc được localStorage", error);
    return null;
  }
}

function writePreference(key, value) {
  try {
    window.localStorage.setItem(key, value);
  } catch (error) {
    console.warn("Không ghi được localStorage", error);
  }
}

/* --------------------------------------------------------------- data */
async function loadHealth() {
  try {
    state.health = await api("/api/health");
    renderAbout();
  } catch (error) {
    console.warn("Health check lỗi", error);
  }
}

async function loadModel() {
  try {
    state.model = await api("/api/model");
  } catch (error) {
    state.model = {
      loaded: false,
      weights: "best.pt",
      loaded_error: true,
      load_error: error.message,
      names: {},
      device: null,
      preferred_device: null,
      task: null,
      nc: null,
    };
    toast(`Không tải được model: ${error.message}`, "error", 8000);
  }
  renderModel(state.model);
}

async function loadSettings() {
  try {
    state.settings = await api("/api/settings");
    renderSettings(state.settings);
  } catch (error) {
    toast(`Không tải được cài đặt: ${error.message}`, "error", 7000);
  }
}

async function loadHistory() {
  try {
    const data = await api("/api/history?limit=200");
    state.history = Array.isArray(data.events) ? data.events : [];
    renderHistory();
  } catch (error) {
    toast(`Không tải được lịch sử: ${error.message}`, "error");
  }
}

async function loadAlerts() {
  try {
    const data = await api("/api/alerts?limit=50");
    state.alertEvents = Array.isArray(data.latest) ? data.latest : [];
    renderAlertList();
    if (state.status) renderActiveAlerts(state.status.alerts || data);
  } catch (error) {
    toast(`Không tải được cảnh báo: ${error.message}`, "error");
  }
}

async function loadAnalytics() {
  try {
    const data = await api("/api/analytics");
    state.lastAnalyticsAt = Date.now();
    renderAnalytics(data);
  } catch (error) {
    toast(`Không tải được analytics: ${error.message}`, "error");
  }
}

/* --------------------------------------------------------------- polling */
async function pollStatus() {
  if (state.polling) return;
  state.polling = true;
  try {
    const status = await api("/api/status");
    const wasOffline = state.pollFailures >= 3;
    state.pollFailures = 0;
    state.status = status;
    renderStatus(status);

    if (state.status.alerts) handleAlerts(state.status.alerts);

    if (state.activePanel === "analytics" && Date.now() - state.lastAnalyticsAt > 5000) {
      loadAnalytics();
    }
    if (wasOffline) toast("Đã kết nối lại máy chủ.", "success");
  } catch (error) {
    state.pollFailures += 1;
    if (state.pollFailures === 3) {
      renderOffline(error.message);
      toast("Mất kết nối máy chủ. Dashboard sẽ tự thử lại.", "error", 6000);
    }
  } finally {
    state.polling = false;
  }
}

const STATE_LABELS = {
  idle: "Sẵn sàng",
  starting: "Đang khởi động",
  running: "Đang nhận diện",
  paused: "Tạm dừng",
  stopping: "Đang dừng",
  error: "Lỗi nguồn",
  finished: "Đã kết thúc",
};

function renderStatus(status) {
  const stateName = status.state || "idle";
  document.body.dataset.state = stateName;
  const active = ACTIVE_STATES.has(stateName);
  const hasFrame = Boolean(status.frame && status.frame.seq > 0);
  const showMedia = hasFrame && ["running", "paused", "stopping"].includes(stateName);

  $("#chipStateText").textContent = STATE_LABELS[stateName] || stateName;
  $("#chipSource").textContent = status.source && status.source.label ? status.source.label : "Chưa có nguồn";
  $("#chipFps").textContent =
    typeof status.fps === "number" ? status.fps.toFixed(1) : "—";
  $("#chipLatency").textContent =
    typeof status.latency_ms === "number" ? `${Math.round(status.latency_ms)} ms` : "—";
  $("#chipObjects").textContent = String(status.objects || 0);

  const modelInfo = status.model || {};
  $("#chipModel").textContent = modelInfo.loaded
    ? `${modelInfo.weights || "model"}${modelInfo.device ? ` · ${modelInfo.device}` : ""}`
    : modelInfo.load_error
      ? "Lỗi model"
      : "Chưa nạp model";

  const stream = $("#stream");
  stream.hidden = !showMedia;
  $("#stageEmpty").hidden = showMedia;
  if (!showMedia) renderEmptyState(stateName, status);

  $("#startBtn").disabled = active;
  $("#pauseBtn").disabled = !(stateName === "running" || stateName === "paused");
  $("#stopBtn").disabled = !active;
  $("#shotBtn").disabled = !hasFrame;

  const paused = stateName === "paused";
  $("#pauseLabel").textContent = paused ? "Tiếp tục" : "Tạm dừng";
  $("#pauseIcon").setAttribute("href", paused ? "#i-play" : "#i-pause");

  renderDetectionBadge(status, showMedia);

  const alerts = status.alerts || { total: 0 };
  const navCount = $("#alertsNavCount");
  const totalAlerts = alerts.total || 0;
  navCount.textContent = totalAlerts > 99 ? "99+" : String(totalAlerts);
  navCount.hidden = totalAlerts === 0;

  if (status.model && status.model.loaded && state.model && !state.model.loaded && !state.modelRefreshing) {
    state.modelRefreshing = true;
    loadModel().finally(() => {
      state.modelRefreshing = false;
    });
  }
}

function renderEmptyState(stateName, status) {
  const eyebrow = $("#emptyEyebrow");
  const title = $("#emptyTitle");
  const text = $("#emptyText");
  const iconUse = $("#emptyIcon use");
  const actions = $("#emptyActions");
  const spinner = $("#emptySpinner");
  const startButton = $("#emptyStart");

  const defaultText =
    "Bật webcam máy chủ hoặc tải ảnh/video để bắt đầu nhận diện bằng model đã train.";

  actions.hidden = false;
  spinner.hidden = true;
  startButton.hidden = false;

  if (stateName === "starting") {
    eyebrow.textContent = "Đang khởi động";
    title.textContent = "Đang chuẩn bị nguồn…";
    text.textContent = status.message || "Đang mở nguồn và nạp model.";
    iconUse.setAttribute("href", "#i-camera");
    actions.hidden = true;
    spinner.hidden = false;
    return;
  }

  if (stateName === "error") {
    eyebrow.textContent = "Lỗi nguồn";
    title.textContent = "Không thể tiếp tục";
    text.textContent = status.message || "Nguồn không khả dụng.";
    iconUse.setAttribute("href", "#i-alert");
    return;
  }

  if (stateName === "finished") {
    eyebrow.textContent = "Đã kết thúc";
    title.textContent = "Phiên phát đã xong";
    text.textContent = status.message || "Video đã phát xong. Bạn có thể chọn nguồn khác.";
    iconUse.setAttribute("href", "#i-check");
    return;
  }

  if (stateName === "idle" && state.model && !state.model.loaded && state.model.load_error) {
    eyebrow.textContent = "Model chưa sẵn sàng";
    title.textContent = "Cần nạp model";
    text.textContent = state.model.load_error;
    iconUse.setAttribute("href", "#i-alert");
    return;
  }

  eyebrow.textContent = "Sẵn sàng";
  title.textContent = "Fire & Smoke Vision";
  text.textContent = defaultText;
  iconUse.setAttribute("href", "#i-camera");
}

function renderDetectionBadge(status, showMedia) {
  const badge = $("#detectionBadge");
  const pills = $("#detectionPills");
  const counts = status.counts || {};
  const entries = Object.entries(counts);

  pills.replaceChildren();
  if (!showMedia || entries.length === 0) {
    badge.hidden = true;
    return;
  }

  entries
    .sort((a, b) => b[1] - a[1])
    .forEach(([name, count]) => {
      const pill = document.createElement("span");
      pill.className = "mini-pill";
      pill.dataset.kind = kindOf(name);
      const dot = document.createElement("span");
      dot.className = "mini-pill__dot";
      const label = document.createElement("span");
      label.textContent = `${name} × ${count}`;
      pill.append(dot, label);
      pills.append(pill);
    });
  badge.hidden = false;
}

function renderOffline(message) {
  document.body.dataset.state = "error";
  $("#chipStateText").textContent = "Mất kết nối";
  $("#chipFps").textContent = "—";
  $("#chipLatency").textContent = "—";
  $("#chipObjects").textContent = "—";
  $("#stream").hidden = true;
  $("#stageEmpty").hidden = false;
  $("#emptyEyebrow").textContent = "Mất kết nối";
  $("#emptyTitle").textContent = "Không kết nối được máy chủ";
  $("#emptyText").textContent =
    message || "Kiểm tra tiến trình run.py và kết nối tới cổng dashboard.";
  $("#emptyIcon use").setAttribute("href", "#i-alert");
  $("#emptyActions").hidden = true;
  $("#emptySpinner").hidden = true;
  $("#detectionBadge").hidden = true;
  $("#startBtn").disabled = true;
  $("#pauseBtn").disabled = true;
  $("#stopBtn").disabled = true;
  $("#shotBtn").disabled = true;
}

/* --------------------------------------------------------------- alerts */
function handleAlerts(alerts) {
  const latest = Array.isArray(alerts.latest) ? alerts.latest : [];
  const maxId = latest.reduce((max, event) => Math.max(max, event.id || 0), 0);

  if (!state.alertsInitialized) {
    state.lastAlertId = maxId;
    state.alertsInitialized = true;
    return;
  }

  const fresh = latest
    .filter((event) => (event.id || 0) > state.lastAlertId)
    .sort((a, b) => a.id - b.id);

  if (fresh.length === 0) return;
  state.lastAlertId = Math.max(state.lastAlertId, maxId);

  fresh.slice(-3).forEach((event) => {
    toast(
      `${event.name} ${formatPercent(event.confidence)} · ${event.source || "nguồn hiện tại"}`,
      "warning",
      6000
    );
  });
  playAlertSound();

  if (state.activePanel === "history") loadHistory();
  if (state.activePanel === "alerts") loadAlerts();
}

/* --------------------------------------------------------------- panels */
function setDrawer(open) {
  state.drawerOpen = open;
  document.body.classList.toggle("drawer-open", open);
  const drawer = $("#drawer");
  drawer.setAttribute("aria-hidden", String(!open));
  const menuButton = $("#menuBtn");
  menuButton.setAttribute("aria-expanded", String(open));
  menuButton.setAttribute("aria-label", open ? "Đóng menu" : "Mở menu");
  $("#menuIcon").setAttribute("href", open ? "#i-close" : "#i-menu");

  if (open) {
    const target = drawer.querySelector(".nav-item.is-active") || drawer.querySelector(".nav-item");
    if (target) target.focus({ preventScroll: true });
  } else if (!state.activePanel) {
    menuButton.focus({ preventScroll: true });
  }
}

function setActiveNav(name) {
  state.activeSection = name;
  $$(".nav-item").forEach((button) => {
    const active = button.dataset.panel === name;
    button.classList.toggle("is-active", active);
    if (active) button.setAttribute("aria-current", "page");
    else button.removeAttribute("aria-current");
  });
}

function openPanel(name) {
  if (name === "live") {
    setActiveNav("live");
    closeSheet({ focusMenu: false });
    setDrawer(false);
    return;
  }
  const meta = PANELS[name];
  if (!meta) return;

  state.activePanel = name;
  setActiveNav(name);

  $("#sheetEyebrow").textContent = meta.eyebrow;
  $("#sheetTitle").textContent = meta.title;
  $("#sheetSubtitle").textContent = meta.subtitle;
  $$(".sheet-section").forEach((section) => {
    section.hidden = section.dataset.panel !== name;
  });

  document.body.classList.add("sheet-open");
  $("#sheet").setAttribute("aria-hidden", "false");
  setDrawer(false);
  $("#sheetClose").focus({ preventScroll: true });

  if (name === "history") loadHistory();
  if (name === "alerts") loadAlerts();
  if (name === "analytics") loadAnalytics();
}

function closeSheet({ focusMenu = true } = {}) {
  if (!state.activePanel) return;
  state.activePanel = null;
  document.body.classList.remove("sheet-open");
  $("#sheet").setAttribute("aria-hidden", "true");
  if (focusMenu && !state.drawerOpen) $("#menuBtn").focus({ preventScroll: true });
}

function setupDrawer() {
  $("#menuBtn").addEventListener("click", () => {
    const next = !state.drawerOpen;
    // Drawer và sheet cùng che vùng nhận diện; mở drawer thì đóng sheet
    // để tránh hai lớp overlay chồng lên nhau trên màn hình hẹp.
    if (next) closeSheet({ focusMenu: false });
    setDrawer(next);
  });
  $("#drawerClose").addEventListener("click", () => setDrawer(false));
  $$(".nav-item").forEach((button) => {
    button.addEventListener("click", () => openPanel(button.dataset.panel));
  });
  $("#backdrop").addEventListener("click", () => {
    setDrawer(false);
    closeSheet({ focusMenu: false });
    $("#menuBtn").focus({ preventScroll: true });
  });
}

function setupSheet() {
  $("#sheetClose").addEventListener("click", () => closeSheet());
  $$("[data-goto]").forEach((button) => {
    button.addEventListener("click", () => openPanel(button.dataset.goto));
  });
}

/* --------------------------------------------------------------- controls */
function setupControls() {
  $("#startBtn").addEventListener("click", startWebcam);
  $("#emptyStart").addEventListener("click", startWebcam);
  $("#emptyMedia").addEventListener("click", () => openPanel("media"));
  $("#pauseBtn").addEventListener("click", togglePause);
  $("#stopBtn").addEventListener("click", () => stopPipeline(true));
  $("#shotBtn").addEventListener("click", downloadScreenshot);
  $("#mediaStartCam").addEventListener("click", startWebcam);
}

function isPipelineActive() {
  return Boolean(state.status && ACTIVE_STATES.has(state.status.state));
}

async function startWebcam() {
  try {
    if (isPipelineActive()) {
      const confirmed = await confirmDialog({
        title: "Chuyển sang webcam?",
        text: "Nguồn hiện tại sẽ dừng trước khi mở webcam máy chủ. Cấu hình camera được lấy từ System Settings.",
        confirmLabel: "Chuyển nguồn",
        danger: false,
      });
      if (!confirmed) return;
      await api("/api/stop", { method: "POST" });
    }

    await api("/api/source/webcam", { method: "POST", body: {} });
    setActiveNav("live");
    closeSheet({ focusMenu: false });
    setDrawer(false);
    toast("Đang mở webcam máy chủ…", "info", 3500);
    await pollStatus();
  } catch (error) {
    toast(error.message, "error", 7000);
  }
}

async function togglePause() {
  try {
    const current = state.status ? state.status.state : null;
    if (current === "paused") {
      await api("/api/resume", { method: "POST" });
    } else if (current === "running" || current === "starting") {
      await api("/api/pause", { method: "POST" });
    } else {
      return;
    }
    await pollStatus();
  } catch (error) {
    toast(error.message, "error");
  }
}

async function stopPipeline(withConfirm = true) {
  if (!isPipelineActive()) return;
  try {
    if (withConfirm) {
      const confirmed = await confirmDialog({
        title: "Dừng phiên nhận diện?",
        text: "Nguồn hiện tại sẽ được giải phóng. Bạn có thể bắt đầu lại bằng webcam hoặc tệp khác.",
        confirmLabel: "Dừng phiên",
        danger: true,
      });
      if (!confirmed) return;
    }
    await api("/api/stop", { method: "POST" });
    toast("Đã dừng nguồn nhận diện.", "success", 3000);
    await pollStatus();
  } catch (error) {
    toast(error.message, "error");
  }
}

async function downloadScreenshot() {
  try {
    const response = await fetch("/api/screenshot");
    if (!response.ok) {
      let message = `Không chụp được (HTTP ${response.status})`;
      try {
        const payload = await response.json();
        if (payload && payload.error) message = payload.error;
      } catch (error) {
        console.warn("Không đọc được phản hồi lỗi screenshot", error);
      }
      throw new Error(message);
    }
    const blob = await response.blob();
    const url = URL.createObjectURL(blob);
    const stamp = new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);
    const link = document.createElement("a");
    link.href = url;
    link.download = `detection-${stamp}.jpg`;
    document.body.appendChild(link);
    link.click();
    link.remove();
    window.setTimeout(() => URL.revokeObjectURL(url), 4000);
    toast("Đã tải ảnh chụp khung hình.", "success");
  } catch (error) {
    toast(error.message, "error");
  }
}

/* --------------------------------------------------------------- stream */
function setupStream() {
  const stream = $("#stream");
  stream.addEventListener("error", () => {
    if (state.streamReconnectTimer) return;
    state.streamReconnectTimer = window.setTimeout(() => {
      state.streamReconnectTimer = null;
      connectStream();
    }, 2000);
  });
  connectStream();
}

function connectStream() {
  const stream = $("#stream");
  stream.src = `/api/stream?t=${Date.now()}`;
}

/* --------------------------------------------------------------- media */
function setupMediaPanel() {
  const dropzone = $("#dropzone");
  const input = $("#fileInput");

  const openPicker = () => input.click();
  dropzone.addEventListener("click", openPicker);
  dropzone.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      openPicker();
    }
  });

  ["dragenter", "dragover"].forEach((name) => {
    dropzone.addEventListener(name, (event) => {
      event.preventDefault();
      dropzone.classList.add("is-dragover");
    });
  });
  ["dragleave", "drop"].forEach((name) => {
    dropzone.addEventListener(name, (event) => {
      event.preventDefault();
      dropzone.classList.remove("is-dragover");
    });
  });
  dropzone.addEventListener("drop", (event) => {
    const file = event.dataTransfer && event.dataTransfer.files ? event.dataTransfer.files[0] : null;
    if (file) selectFile(file);
  });

  input.addEventListener("change", () => {
    const file = input.files && input.files[0];
    if (file) selectFile(file);
    input.value = "";
  });

  $("#clearFile").addEventListener("click", clearSelectedFile);
  $("#processBtn").addEventListener("click", processSelectedFile);
}

function extensionOf(filename) {
  const parts = String(filename || "").toLowerCase().split(".");
  return parts.length > 1 ? parts.pop() : "";
}

function selectFile(file) {
  const extension = extensionOf(file.name);
  const isImage = IMAGE_EXTENSIONS.has(extension);
  const isVideo = VIDEO_EXTENSIONS.has(extension);

  if (!isImage && !isVideo) {
    toast("Chỉ hỗ trợ ảnh JPG/PNG/BMP/WEBP và video MP4/AVI/MOV/MKV/WEBM.", "error", 6000);
    return;
  }
  const limit = isImage ? MAX_IMAGE_BYTES : MAX_VIDEO_BYTES;
  if (file.size > limit) {
    toast(`Tệp vượt quá giới hạn ${formatBytes(limit)}.`, "error", 6000);
    return;
  }
  if (file.size === 0) {
    toast("Tệp rỗng.", "error");
    return;
  }

  state.selectedFile = file;
  $("#fileRow").hidden = false;
  $("#fileName").textContent = file.name;
  $("#fileMeta").textContent = `${isImage ? "Ảnh" : "Video"} · ${formatBytes(file.size)}`;
  $("#fileIcon").setAttribute("href", isImage ? "#i-image" : "#i-film");
  $("#processBtn").disabled = false;
}

function clearSelectedFile() {
  state.selectedFile = null;
  $("#fileRow").hidden = true;
  $("#fileName").textContent = "";
  $("#fileMeta").textContent = "";
  $("#processBtn").disabled = true;
  $("#fileInput").value = "";
}

async function processSelectedFile() {
  if (!state.selectedFile) return;

  if (isPipelineActive()) {
    const confirmed = await confirmDialog({
      title: "Thay nguồn hiện tại?",
      text: "Nguồn đang chạy sẽ dừng và tệp mới sẽ được xử lý. Tệp cũ (nếu có) sẽ bị xoá khỏi thư mục tạm.",
      confirmLabel: "Xử lý tệp",
      danger: false,
    });
    if (!confirmed) return;
  }

  const button = $("#processBtn");
  setButtonBusy(button, true, "Đang tải và khởi động…");
  try {
    const form = new FormData();
    form.append("file", state.selectedFile);
    const result = await api("/api/source/upload", { method: "POST", body: form });
    toast(`Đang xử lý: ${result.name}`, "success", 4000);
    clearSelectedFile();
    setActiveNav("live");
    closeSheet({ focusMenu: false });
    setDrawer(false);
    connectStream();
    await pollStatus();
  } catch (error) {
    toast(error.message, "error", 8000);
  } finally {
    setButtonBusy(button, false);
    if (!state.selectedFile) button.disabled = true;
  }
}

/* --------------------------------------------------------------- history */
function setupHistoryPanel() {
  $("#historyFilter").addEventListener("change", renderHistory);
  $("#clearHistoryBtn").addEventListener("click", async () => {
    const confirmed = await confirmDialog({
      title: "Xoá lịch sử cảnh báo?",
      text: "Toàn bộ sự kiện cảnh báo trong phiên hiện tại sẽ bị xoá khỏi bộ nhớ. Số liệu analytics vẫn giữ nguyên.",
      confirmLabel: "Xoá lịch sử",
      danger: true,
    });
    if (!confirmed) return;
    try {
      await api("/api/history", { method: "DELETE" });
      state.history = [];
      state.alertEvents = [];
      state.alertsInitialized = true;
      state.lastAlertId = 0;
      renderHistory();
      renderAlertList();
      toast("Đã xoá lịch sử cảnh báo.", "success");
    } catch (error) {
      toast(error.message, "error");
    }
  });
}

function renderHistory() {
  const names = state.settings ? state.settings.class_names : {};
  const select = $("#historyFilter");
  const previous = select.value || "all";
  const options = classEntries(names);
  select.replaceChildren();
  const allOption = document.createElement("option");
  allOption.value = "all";
  allOption.textContent = "Tất cả";
  select.append(allOption);
  options.forEach(([id, name]) => {
    const option = document.createElement("option");
    option.value = String(id);
    option.textContent = name;
    select.append(option);
  });
  select.value = previous === "all" || options.some(([id]) => String(id) === previous) ? previous : "all";

  const events = state.history.filter((event) => {
    if (select.value === "all") return true;
    return String(event.class_id) === select.value;
  });
  renderEventList($("#historyList"), events, "Chưa có sự kiện cảnh báo.");
}

function renderEventList(container, events, emptyMessage) {
  container.replaceChildren();
  if (!events || events.length === 0) {
    const row = document.createElement("li");
    row.className = "empty-row";
    row.textContent = emptyMessage;
    container.append(row);
    return;
  }

  events.forEach((event) => {
    const row = document.createElement("li");
    row.className = "event";

    const dot = document.createElement("span");
    dot.className = "event__dot";
    dot.dataset.kind = kindOf(event.name);

    const body = document.createElement("div");
    body.className = "event__body";
    const title = document.createElement("strong");
    title.append(document.createTextNode(event.name || "unknown"));
    const conf = document.createElement("span");
    conf.className = "det-item__conf";
    conf.textContent = formatPercent(event.confidence);
    title.append(conf);

    const meta = document.createElement("small");
    const track = event.track_id !== null && event.track_id !== undefined ? ` · #${event.track_id}` : "";
    meta.textContent = `${event.source || "—"}${track}`;

    body.append(title, meta);

    const time = document.createElement("time");
    time.dateTime = event.at || "";
    time.textContent = formatTime(event.at);

    row.append(dot, body, time);
    container.append(row);
  });
}

/* --------------------------------------------------------------- analytics */
function setupAnalyticsPanel() {
  $("#analyticsRefresh").addEventListener("click", loadAnalytics);
}

function renderAnalytics(data) {
  $("#anFrames").textContent = String(data.frames_processed ?? 0);
  $("#anDetections").textContent = String(data.total_detections ?? 0);
  $("#anConfidence").textContent = formatPercent(data.average_confidence);
  $("#anAlerts").textContent = String(data.alerts_total ?? 0);
  $("#anUptime").textContent = formatUptime(data.uptime_seconds);
  $("#anRate").textContent =
    typeof data.detections_per_minute === "number" ? data.detections_per_minute.toFixed(2) : "—";

  const container = $("#classBars");
  container.replaceChildren();
  const perClass = Array.isArray(data.per_class) ? data.per_class : [];
  if (perClass.length === 0) {
    const empty = document.createElement("p");
    empty.className = "empty-row";
    empty.textContent = "Chưa có detection nào trong phiên này.";
    container.append(empty);
    return;
  }

  const max = Math.max(...perClass.map((item) => item.count || 0), 1);
  perClass.forEach((item) => {
    const row = document.createElement("div");
    row.className = "bar-row";

    const head = document.createElement("div");
    head.className = "bar-row__head";
    const label = document.createElement("strong");
    label.textContent = item.name;
    const value = document.createElement("span");
    value.textContent = `${item.count} lượt`;
    head.append(label, value);

    const track = document.createElement("div");
    track.className = "bar-track";
    const fill = document.createElement("div");
    fill.className = "bar-fill";
    fill.dataset.kind = kindOf(item.name);
    fill.style.width = `${Math.max(3, Math.round(((item.count || 0) / max) * 100))}%`;
    track.append(fill);

    row.append(head, track);
    container.append(row);
  });
}

/* --------------------------------------------------------------- alerts UI */
function setupAlertsPanel() {
  $("#soundToggle").addEventListener("change", (event) => {
    state.soundEnabled = event.target.checked;
    writePreference("fv:sound", state.soundEnabled ? "1" : "0");
    if (state.soundEnabled) {
      unlockAudio();
      playAlertSound();
      toast("Đã bật âm thanh cảnh báo.", "success", 3000);
    } else {
      toast("Đã tắt âm thanh cảnh báo.", "info", 3000);
    }
  });
}

function renderActiveAlerts(alerts) {
  const card = $("#activeAlertCard");
  const pills = $("#activeAlertPills");
  const active = Array.isArray(alerts && alerts.active) ? alerts.active : [];
  pills.replaceChildren();

  if (active.length === 0) {
    card.hidden = true;
    return;
  }

  active.forEach((name) => {
    const pill = document.createElement("span");
    pill.className = "class-pill";
    pill.dataset.kind = kindOf(name);
    pill.textContent = name;
    pills.append(pill);
  });
  card.hidden = false;
}

function renderAlertList() {
  renderEventList($("#alertList"), state.alertEvents, "Chưa có cảnh báo.");
}

/* --------------------------------------------------------------- audio */
let audioContext = null;

function unlockAudio() {
  try {
    const Ctx = window.AudioContext || window.webkitAudioContext;
    if (!Ctx) return;
    if (!audioContext) audioContext = new Ctx();
    if (audioContext.state === "suspended") audioContext.resume();
  } catch (error) {
    console.warn("Không khởi tạo được WebAudio", error);
  }
}

function playAlertSound() {
  if (!state.soundEnabled) return;
  try {
    unlockAudio();
    if (!audioContext) return;
    const now = audioContext.currentTime;
    const oscillator = audioContext.createOscillator();
    const gain = audioContext.createGain();
    oscillator.type = "sine";
    oscillator.frequency.setValueAtTime(880, now);
    oscillator.frequency.exponentialRampToValueAtTime(620, now + 0.18);
    gain.gain.setValueAtTime(0.0001, now);
    gain.gain.exponentialRampToValueAtTime(0.07, now + 0.02);
    gain.gain.exponentialRampToValueAtTime(0.0001, now + 0.3);
    oscillator.connect(gain).connect(audioContext.destination);
    oscillator.start(now);
    oscillator.stop(now + 0.32);
  } catch (error) {
    console.warn("Không phát được âm thanh cảnh báo", error);
  }
}

/* --------------------------------------------------------------- model UI */
function renderModel(info) {
  const names = info.names || {};
  const entries = classEntries(names);
  const loaded = Boolean(info.loaded);
  const pill = $("#drawerModelState");

  pill.textContent = loaded ? "Đã nạp" : info.load_error ? "Lỗi" : "Chưa nạp";
  pill.className = `pill ${loaded ? "pill--success" : info.load_error ? "pill--danger" : "pill--muted"}`;

  $("#drawerWeights").textContent = info.weights || "—";
  $("#drawerModelMeta").textContent = loaded
    ? `${info.task || "detect"} · ${entries.length} lớp · ${info.device || info.preferred_device || "cpu"}`
    : info.load_error
      ? "Không nạp được weights"
      : "Chưa nạp model";

  const statePill = $("#modelStatePill");
  statePill.textContent = loaded ? "Đã nạp" : info.load_error ? "Lỗi nạp" : "Chưa nạp";
  statePill.className = `pill ${loaded ? "pill--success" : info.load_error ? "pill--danger" : "pill--muted"}`;

  $("#modelWeightsName").textContent = info.weights || "—";
  $("#modelTask").textContent = info.task || "—";
  $("#modelClassesCount").textContent = entries.length ? String(entries.length) : "—";
  $("#modelDevice").textContent = info.device || info.device_error || "—";

  const pills = $("#modelClassPills");
  pills.replaceChildren();
  entries.forEach(([, name]) => {
    const item = document.createElement("span");
    item.className = "class-pill";
    item.dataset.kind = kindOf(name);
    item.textContent = name;
    pills.append(item);
  });

  const errorBox = $("#modelError");
  if (info.load_error) {
    errorBox.hidden = false;
    errorBox.textContent = info.load_error;
  } else {
    errorBox.hidden = true;
    errorBox.textContent = "";
  }

  renderAbout();

  if (state.settings && $("#inferenceClasses").childElementCount === 0) {
    renderSettings(state.settings);
  }
}

function renderAbout() {
  const health = state.health || {};
  const model = state.model || {};
  $("#aboutVersion").textContent = health.version || "—";
  $("#aboutWeights").textContent = health.weights || model.weights || "—";
  $("#aboutWeightsState").textContent =
    typeof health.weights_exists === "boolean"
      ? health.weights_exists
        ? "Sẵn sàng"
        : "Không tìm thấy"
      : "—";
  $("#aboutModelState").textContent = model.loaded ? "Đã nạp" : model.load_error ? "Lỗi nạp" : "Chưa nạp";
}

/* --------------------------------------------------------------- settings */
function setupForms() {
  bindRange("confRange", "confValue", 2);
  bindRange("iouRange", "iouValue", 2);
  bindRange("qualityRange", "qualityValue", 0);

  $("#yoloForm").addEventListener("submit", saveInferenceSettings);
  $("#systemForm").addEventListener("submit", saveSystemSettings);
  $("#resetYolo").addEventListener("click", () => resetSettings("inference"));
  $("#resetSystem").addEventListener("click", () => resetSettings("system"));
}

function bindRange(inputId, outputId, digits) {
  const input = $(`#${inputId}`);
  const output = $(`#${outputId}`);
  if (!input || !output) return;
  const sync = () => {
    output.textContent = Number(input.value).toFixed(digits);
  };
  input.addEventListener("input", sync);
  sync();
}

function ensureOption(select, value, label) {
  const stringValue = String(value);
  if (!Array.from(select.options).some((option) => option.value === stringValue)) {
    const option = document.createElement("option");
    option.value = stringValue;
    option.textContent = label || stringValue;
    select.append(option);
  }
  select.value = stringValue;
}

function renderSettings(data) {
  state.settings = data;
  const inference = data.inference || {};
  const system = data.system || {};
  const classNames = data.class_names || {};

  $("#confRange").value = inference.conf ?? 0.35;
  $("#iouRange").value = inference.iou ?? 0.45;
  $("#maxDetInput").value = inference.max_det ?? 100;
  ensureOption($("#imgszSelect"), inference.imgsz ?? 640);
  ensureOption(
    $("#deviceSelect"),
    inference.device ?? "auto",
    inference.device && inference.device !== "auto" && inference.device !== "cpu"
      ? `GPU ${inference.device}`
      : inference.device
  );
  $("#trackingToggle").checked = Boolean(inference.tracking);
  renderClassChecks("#inferenceClasses", classNames, inference.classes);
  $("#confValue").textContent = Number($("#confRange").value).toFixed(2);
  $("#iouValue").textContent = Number($("#iouRange").value).toFixed(2);

  $("#cameraIndex").value = system.camera_index ?? 0;
  $("#cameraWidth").value = system.width ?? 1280;
  $("#cameraHeight").value = system.height ?? 720;
  $("#targetFps").value = system.target_fps ?? 24;
  $("#mirrorToggle").checked = Boolean(system.mirror);
  $("#qualityRange").value = system.jpeg_quality ?? 82;
  $("#alertCooldown").value = system.alert_cooldown ?? 10;
  $("#maxHistory").value = system.max_history ?? 200;
  renderClassChecks("#alertClasses", classNames, system.alert_classes);
  $("#qualityValue").textContent = String(system.jpeg_quality ?? 82);

  renderHistory();
}

function renderClassChecks(selector, classNames, selected) {
  const container = $(selector);
  if (!container) return;
  container.replaceChildren();
  const entries = classEntries(classNames);

  if (entries.length === 0) {
    const note = document.createElement("span");
    note.className = "hint";
    note.textContent = "Chưa nạp được danh sách lớp từ model.";
    container.append(note);
    return;
  }

  const selectedIds = selected === null || selected === undefined ? null : selected.map(Number);
  entries.forEach(([id, name]) => {
    const label = document.createElement("label");
    label.className = "check-pill";
    const input = document.createElement("input");
    input.type = "checkbox";
    input.value = String(id);
    input.checked = selectedIds === null || selectedIds.includes(id);
    const span = document.createElement("span");
    span.textContent = name;
    label.append(input, span);
    container.append(label);
  });
}

function collectClassChecks(selector) {
  const inputs = $$(`${selector} input[type="checkbox"]`);
  const checked = inputs.filter((input) => input.checked).map((input) => Number(input.value));
  return { checked, total: inputs.length };
}

function readNumber(selector, label, { min, max, integer = false }) {
  const value = Number($(selector).value);
  if (!Number.isFinite(value)) throw new Error(`${label} phải là số`);
  if (value < min || value > max) throw new Error(`${label} phải nằm trong ${min}–${max}`);
  if (integer && !Number.isInteger(value)) throw new Error(`${label} phải là số nguyên`);
  return value;
}

async function saveInferenceSettings(event) {
  event.preventDefault();
  const button = $("#saveYolo");
  try {
    const classes = collectClassChecks("#inferenceClasses");
    if (classes.total > 0 && classes.checked.length === 0) {
      throw new Error("Chọn ít nhất một lớp nhận diện.");
    }
    const payload = {
      inference: {
        conf: readNumber("#confRange", "Confidence", { min: 0.01, max: 0.99 }),
        iou: readNumber("#iouRange", "IoU", { min: 0.01, max: 0.99 }),
        imgsz: readNumber("#imgszSelect", "Kích thước inference", {
          min: 320,
          max: 1280,
          integer: true,
        }),
        max_det: readNumber("#maxDetInput", "Max detections", {
          min: 1,
          max: 300,
          integer: true,
        }),
        device: $("#deviceSelect").value,
        tracking: $("#trackingToggle").checked,
        classes:
          classes.total > 0 && classes.checked.length === classes.total ? null : classes.checked,
      },
    };
    setButtonBusy(button, true, "Đang lưu…");
    const data = await api("/api/settings", { method: "PATCH", body: payload });
    renderSettings(data);
    toast("Đã lưu cài đặt YOLO. Áp dụng cho frame tiếp theo.", "success", 4000);
  } catch (error) {
    toast(error.message, "error", 7000);
  } finally {
    setButtonBusy(button, false);
  }
}

async function saveSystemSettings(event) {
  event.preventDefault();
  const button = $("#saveSystem");
  try {
    const alertClasses = collectClassChecks("#alertClasses");
    if (alertClasses.total > 0 && alertClasses.checked.length === 0) {
      throw new Error("Chọn ít nhất một lớp phát cảnh báo.");
    }
    const payload = {
      system: {
        camera_index: readNumber("#cameraIndex", "Chỉ số camera", {
          min: 0,
          max: 16,
          integer: true,
        }),
        width: readNumber("#cameraWidth", "Chiều rộng", { min: 160, max: 3840, integer: true }),
        height: readNumber("#cameraHeight", "Chiều cao", { min: 120, max: 2160, integer: true }),
        target_fps: readNumber("#targetFps", "FPS mục tiêu", { min: 1, max: 60, integer: true }),
        mirror: $("#mirrorToggle").checked,
        jpeg_quality: readNumber("#qualityRange", "Chất lượng JPEG", {
          min: 40,
          max: 95,
          integer: true,
        }),
        alert_cooldown: readNumber("#alertCooldown", "Cooldown", { min: 0, max: 600 }),
        max_history: readNumber("#maxHistory", "Số sự kiện lưu", {
          min: 20,
          max: 1000,
          integer: true,
        }),
        alert_classes:
          alertClasses.total > 0 && alertClasses.checked.length === alertClasses.total
            ? null
            : alertClasses.checked,
      },
    };
    setButtonBusy(button, true, "Đang lưu…");
    const data = await api("/api/settings", { method: "PATCH", body: payload });
    renderSettings(data);
    toast("Đã lưu cài đặt hệ thống.", "success", 4000);
  } catch (error) {
    toast(error.message, "error", 7000);
  } finally {
    setButtonBusy(button, false);
  }
}

async function resetSettings(scope) {
  const confirmed = await confirmDialog({
    title: scope === "inference" ? "Khôi phục YOLO mặc định?" : "Khôi phục hệ thống mặc định?",
    text: "Cài đặt trong nhóm này sẽ trở về giá trị mặc định. Nguồn đang chạy không bị dừng.",
    confirmLabel: "Khôi phục",
    danger: false,
  });
  if (!confirmed) return;
  try {
    const data = await api("/api/settings/reset", {
      method: "POST",
      body: { scope },
    });
    renderSettings(data);
    toast("Đã khôi phục cài đặt mặc định.", "success", 3500);
  } catch (error) {
    toast(error.message, "error");
  }
}

/* --------------------------------------------------------------- fullscreen */
function setupFullscreen() {
  $("#fsBtn").addEventListener("click", async () => {
    try {
      if (!document.fullscreenElement) {
        await $("#mainStage").requestFullscreen();
      } else {
        await document.exitFullscreen();
      }
    } catch (error) {
      toast("Trình duyệt không cho phép bật toàn màn hình.", "error", 5000);
    }
  });

  document.addEventListener("fullscreenchange", () => {
    const active = Boolean(document.fullscreenElement);
    $("#fsLabel").textContent = active ? "Thoát toàn màn hình" : "Toàn màn hình";
    $("#fsBtn").title = active ? "Thoát toàn màn hình" : "Bật toàn màn hình";
  });
}

/* --------------------------------------------------------------- keyboard */
function setupKeyboard() {
  document.addEventListener("keydown", (event) => {
    if (event.key !== "Escape") return;
    const dialog = $("#confirmDialog");
    if (dialog.open) return;
    if (state.activePanel) {
      closeSheet();
      return;
    }
    if (state.drawerOpen) {
      setDrawer(false);
    }
  });
}
