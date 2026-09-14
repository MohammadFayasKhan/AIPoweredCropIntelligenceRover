/**
 * Smart Plant Intelligence System - Frontend Logic
 * Strictly handles:
 * 1. Physical ESP8266 IoT Telemetry Lifecycle (Live vs Stale vs Offline vs Manual)
 * 2. Backend Health and 4-Model Readiness Telemetry
 * 3. Crop Recommendation and Disease Risk Prediction Pipeline
 * 4. 3-Tier Foliar Computer Vision Diagnostics and Lesion Localization
 * 5. Telemetry Trend History (Hardware Data Only, Zero Synthetic Polling Records)
 * 6. Responsive Image Handling and Non-Destructive Fullscreen Inspection
 */

// ── HTML SANITIZATION UTILITY ────────────────────────────────────────────────
function escapeHTML(str) {
  if (str == null) return "";
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}
const escapeHtml = escapeHTML;
if (typeof window !== "undefined") {
  window.escapeHTML = escapeHTML;
  window.escapeHtml = escapeHTML;
}
if (typeof module !== "undefined" && module.exports) {
  module.exports = { escapeHTML, escapeHtml };
}

const API_BASE = (typeof window !== "undefined" && window.SMARTCROP_API_BASE) || 
  (typeof window !== "undefined" && window.location && window.location.protocol && window.location.protocol.startsWith("http") ? "" : "http://localhost:8000");

// ── EXPLICIT VISION UI STATE MACHINE ──────────────────────────────────────────
const VisionUIState = {
  EMPTY: "EMPTY",
  IMAGE_SELECTED: "IMAGE_SELECTED",
  VALIDATING: "VALIDATING",
  VALID_PLANT_IMAGE: "VALID_PLANT_IMAGE",
  INVALID_SCREENSHOT_OR_DOCUMENT: "INVALID_SCREENSHOT_OR_DOCUMENT",
  INVALID_NON_PLANT_IMAGE: "INVALID_NON_PLANT_IMAGE",
  LOW_QUALITY_IMAGE: "LOW_QUALITY_IMAGE",
  VALIDATION_UNCERTAIN: "VALIDATION_UNCERTAIN",
  ANALYZING: "ANALYZING",
  RESULTS_READY: "RESULTS_READY",
  ERROR: "ERROR",
  // Compatibility aliases
  IDLE: "EMPTY",
  READY_TO_ANALYZE: "VALID_PLANT_IMAGE",
  IMAGE_VALIDATING: "VALIDATING",
  INVALID_SCREENSHOT: "INVALID_SCREENSHOT_OR_DOCUMENT",
  INVALID_NON_PLANT: "INVALID_NON_PLANT_IMAGE",
  LOW_QUALITY: "LOW_QUALITY_IMAGE",
  COMPLETE: "RESULTS_READY",
  LOW_CONFIDENCE: "RESULTS_READY"
};
let currentVisionState = VisionUIState.EMPTY;
let visionAbortController = null;
let validationAbortController = null;
let isVisionAnalyzing = false;
let currentPreviewObjectUrl = null;
let lastFocusedModalElement = null;
let activeVisionRequestId = null;

function setVisionUIState(newState, detailText = "") {
  currentVisionState = newState;
  const statusEl = document.getElementById("visionStateStatus");
  const textEl = document.getElementById("visionStateStatusText");
  const dotEl = document.getElementById("visionStatusPulseDot");
  const analyzeBtn = document.getElementById("visionAnalyzeBtn");
  const progressStage = document.getElementById("analysisProgressStage");

  const stateLabels = {
    [VisionUIState.EMPTY]: "Status: Idle · Awaiting Specimen",
    [VisionUIState.IMAGE_SELECTED]: "Status: Specimen Selected · Preflight Checking...",
    [VisionUIState.VALIDATING]: "Status: Validating Specimen Quality...",
    [VisionUIState.VALID_PLANT_IMAGE]: "Status: Specimen Verified · Ready for Analysis",
    [VisionUIState.INVALID_SCREENSHOT_OR_DOCUMENT]: "Status: Rejected · Screenshot / Document Detected",
    [VisionUIState.INVALID_NON_PLANT_IMAGE]: "Status: Rejected · Non-Plant Specimen",
    [VisionUIState.LOW_QUALITY_IMAGE]: "Status: Rejected · Image Quality Insufficient",
    [VisionUIState.VALIDATION_UNCERTAIN]: "Status: Rejected · Foliar Presence Uncertain",
    [VisionUIState.ANALYZING]: "Status: Executing Neural Inference Cascade...",
    [VisionUIState.RESULTS_READY]: "Status: Diagnostics Complete",
    [VisionUIState.ERROR]: "Status: Diagnostic Error"
  };

  const isRejectedState = [
    VisionUIState.INVALID_SCREENSHOT_OR_DOCUMENT,
    VisionUIState.INVALID_NON_PLANT_IMAGE,
    VisionUIState.LOW_QUALITY_IMAGE,
    VisionUIState.VALIDATION_UNCERTAIN
  ].includes(newState);

  const isWorkingState = [
    VisionUIState.VALIDATING,
    VisionUIState.ANALYZING
  ].includes(newState);

  const label = stateLabels[newState] || newState;
  if (textEl) {
    textEl.textContent = label;
  } else if (statusEl) {
    statusEl.innerHTML = `<span class="status-pulse-dot" id="visionStatusPulseDot" style="width: 7px; height: 7px; border-radius: 50%; background: var(--green-bright); display: inline-block; flex-shrink: 0;"></span><span id="visionStateStatusText">${label}</span>`;
  }

  if (statusEl) {
    if (isRejectedState) {
      statusEl.style.borderColor = "rgba(239, 35, 60, 0.5)";
      statusEl.style.background = "rgba(239, 35, 60, 0.1)";
      statusEl.style.color = "#ff8a9a";
      if (dotEl) {
        dotEl.style.background = "#ff4d6d";
        dotEl.style.boxShadow = "0 0 8px rgba(255, 77, 109, 0.6)";
      }
    } else if (isWorkingState) {
      statusEl.style.borderColor = "rgba(255, 209, 102, 0.5)";
      statusEl.style.background = "rgba(255, 209, 102, 0.1)";
      statusEl.style.color = "#ffd166";
      if (dotEl) {
        dotEl.style.background = "#ffd166";
        dotEl.style.boxShadow = "0 0 8px rgba(255, 209, 102, 0.6)";
      }
    } else {
      statusEl.style.borderColor = "rgba(82, 183, 136, 0.2)";
      statusEl.style.background = "rgba(82, 183, 136, 0.08)";
      statusEl.style.color = "var(--green-bright)";
      if (dotEl) {
        dotEl.style.background = "var(--green-bright)";
        dotEl.style.boxShadow = "0 0 8px rgba(82, 183, 136, 0.6)";
      }
    }
  }

  // Update primary call-to-action button
  if (analyzeBtn) {
    if (newState === VisionUIState.ANALYZING) {
      analyzeBtn.disabled = true;
      analyzeBtn.classList.add("btn-loading");
      analyzeBtn.innerHTML = '<span class="btn-icon">⏳</span><span>Analyzing Plant Health...</span>';
    } else if (newState === VisionUIState.VALIDATING) {
      analyzeBtn.disabled = true;
      analyzeBtn.classList.add("btn-loading");
      analyzeBtn.innerHTML = '<span class="btn-icon">🛡️</span><span>Validating Specimen Domain...</span>';
    } else if (isRejectedState) {
      analyzeBtn.disabled = false;
      analyzeBtn.classList.remove("btn-loading");
      analyzeBtn.innerHTML = '<span class="btn-icon">🔬</span><span>Analyze Specimen Anyway</span>';
      analyzeBtn.onclick = () => runVisionPrediction(true);
    } else if (newState === VisionUIState.VALID_PLANT_IMAGE) {
      analyzeBtn.disabled = false;
      analyzeBtn.classList.remove("btn-loading");
      analyzeBtn.innerHTML = '<span class="btn-icon">⚡</span><span>Analyze Plant Health</span>';
      analyzeBtn.onclick = () => runVisionPrediction(false);
    } else if (newState === VisionUIState.RESULTS_READY) {
      analyzeBtn.disabled = false;
      analyzeBtn.classList.remove("btn-loading");
      analyzeBtn.innerHTML = '<span class="btn-icon">🔄</span><span>Re-Analyze Specimen</span>';
      analyzeBtn.onclick = () => runVisionPrediction(false);
    } else if (newState === VisionUIState.ERROR) {
      analyzeBtn.disabled = false;
      analyzeBtn.classList.remove("btn-loading");
      analyzeBtn.innerHTML = '<span class="btn-icon">⚡</span><span>Retry Analysis</span>';
      analyzeBtn.onclick = () => runVisionPrediction(true);
    } else {
      analyzeBtn.disabled = true;
      analyzeBtn.classList.remove("btn-loading");
      analyzeBtn.innerHTML = '<span class="btn-icon">⚡</span><span>Analyze Plant Health</span>';
      analyzeBtn.onclick = () => runVisionPrediction(false);
    }
  }

  if (progressStage && detailText) {
    progressStage.textContent = detailText;
  }
}

// ── AUTO-SCROLL AND VISUAL EMPHASIS HELPER ────────────────────────────────────
function scrollAndHighlightAnalyzeButton() {
  const analyzeBtn = document.getElementById("visionAnalyzeBtn");
  if (!analyzeBtn) return;

  const leftPanel = document.getElementById("visionInputPanel") || document.querySelector("#visionSection .sensor-panel");
  const prefersReducedMotion = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  if (leftPanel) {
    // Only scroll if the analyze button is below the current visible area of the left panel
    const panelRect = leftPanel.getBoundingClientRect();
    const btnRect = analyzeBtn.getBoundingClientRect();
    if (btnRect.bottom > panelRect.bottom) {
      const scrollNeeded = btnRect.bottom - panelRect.bottom + 16;
      leftPanel.scrollBy({
        top: scrollNeeded,
        behavior: prefersReducedMotion ? "auto" : "smooth"
      });
    }
  }

  // Update status bar to prominently instruct user to hit Analyze
  const statusEl = document.getElementById("visionStateStatus");
  if (statusEl) {
    statusEl.innerHTML = `
      <span class="status-pulse-dot" style="width: 8px; height: 8px; border-radius: 50%; background: var(--green-bright); display: inline-block; box-shadow: 0 0 8px var(--green-bright);"></span>
      <span style="font-weight: 600; color: var(--green-bright); font-size: 0.82rem;">Specimen Ready · Hit "Analyze Plant Health" for Results 👇</span>
    `;
    statusEl.style.background = "rgba(82, 183, 136, 0.16)";
    statusEl.style.borderColor = "var(--green-bright)";
  }

  // Apply subtle non-looping visual emphasis
  analyzeBtn.classList.remove("cta-attention-glow");
  void analyzeBtn.offsetWidth; // Force reflow
  analyzeBtn.classList.add("cta-attention-glow");
  setTimeout(() => {
    analyzeBtn.classList.remove("cta-attention-glow");
  }, 2600);
}

// ── GLOBAL APPLICATION STATE ──────────────────────────────────────────────────
let activeMode = "crop"; // "crop" or "vision"
let isManualInput = true; // Sliders are user-controlled until genuine IoT telemetry arrives
let isRainActive = false; // YL-83 Digital Rain Sensor state
let autoUpdateEnabled = true;

// IoT Hardware Lifecycle State
let latestIotPacket = null;
let lastIotTimestamp = null; // Milliseconds timestamp of physical sensor packet
let lastChartPacketTs = null; // Guard to prevent duplicate chart entries
let espDeviceState = "waiting"; // "waiting" | "connected" | "stale" | "offline"

// History Chart Instance
let historyChart = null;

// Computer Vision State
let selectedVisionFile = null;
let lastVisionResult = null;
let originalVisionImageObj = null;
let showBoundingBoxes = true;
let showSpecimenBoxes = true;
let showLesionBoxes = true;

// ESP32-CAM (OV2640) Field Hardware State
let currentVisionInputMode = "upload"; // 'upload' | 'camera' | 'esp32'
let currentVisionImageSource = "browser_upload"; // 'browser_upload' | 'browser_camera' | 'esp32_cam'
let currentEsp32CaptureId = null;
let currentEsp32CaptureMeta = null;
let esp32StatusPollTimer = null;
let esp32StreamActive = false;
let esp32StreamAbortController = null;
let esp32FrameCount = 0;
let esp32FpsTimer = null;
let esp32DeviceOnline = false;
let isFetchingEsp32Frame = false;

// Presets for Quick Scenarios
const SENSOR_PRESETS = {
  monsoon: { temperature: 24, humidity: 88, soil_moisture: 85, rain: 1 },
  summer: { temperature: 38, humidity: 30, soil_moisture: 20, rain: 0 },
  foggy: { temperature: 14, humidity: 92, soil_moisture: 60, rain: 0 },
  ideal: { temperature: 26, humidity: 65, soil_moisture: 55, rain: 0 },
};

const SAMPLE_LEAF_MAP = {
  tomato_early_blight: {
    filename: "tomato__fungal__early_blight.jpg",
    url: "./samples/tomato__fungal__early_blight.jpg",
    label: "Tomato Early Blight",
  },
  corn_rust: {
    filename: "corn__fungal__common_rust_.jpg",
    url: "./samples/corn__fungal__common_rust_.jpg",
    label: "Corn Common Rust",
  },
  grape_rot: {
    filename: "grape__fungal__black_rot.jpg",
    url: "./samples/grape__fungal__black_rot.jpg",
    label: "Grape Black Rot",
  },
  apple_scab: {
    filename: "apple__fungal__apple_scab.jpg",
    url: "./samples/apple__fungal__apple_scab.jpg",
    label: "Apple Scab (Venturia inaequalis)",
  },
  healthy: {
    filename: "potato__healthy__healthy.jpg",
    url: "./samples/potato__healthy__healthy.jpg",
    label: "Potato Healthy Leaf",
  },
};

// ── INITIALIZATION ────────────────────────────────────────────────────────────
if (typeof document !== "undefined") {
  document.addEventListener("DOMContentLoaded", () => {
    initHistoryChart();
    initAutoToggle();
    initDragAndDrop();
    initKeyboardListeners();
    syncSliderDisplays();

    // Run initial health and model readiness check
    checkBackendHealth();
    // Poll backend health every 15 seconds
    setInterval(checkBackendHealth, 15000);

    // Poll for genuine physical ESP8266 telemetry
    pollLatestIotTelemetry();
    setInterval(pollLatestIotTelemetry, 10000);

    // Poll for genuine physical ESP32-CAM field hardware (fast 3.5s cycle for immediate hotspot link)
    pollEsp32Status();
    setInterval(pollEsp32Status, 3500);

    // High-resolution 1-second ticker to detect stale physical hardware immediately
    setInterval(updateIotAgeTicker, 1000);

    // Fetch initial telemetry history if available on server
    fetchTelemetryHistory();

    // Initialize draggable panel splitter and baseline telemetry
    initPanelResizers();
    renderTechnicalDetails({});
  });
}

// ── DRAGGABLE DYNAMIC PANEL RESIZER ──────────────────────────────────────────
function initPanelResizers() {
  if (typeof localStorage !== "undefined") {
    const savedWidth = localStorage.getItem("smartcrop_panel_width");
    if (savedWidth && Number(savedWidth) >= 280 && Number(savedWidth) <= 720) {
      document.documentElement.style.setProperty("--left-panel-width", `${savedWidth}px`);
    }
  }

  const resizers = document.querySelectorAll(".panel-resizer");
  resizers.forEach((resizer) => {
    const onPointerDown = (e) => {
      if (window.innerWidth <= 900) return;
      e.preventDefault();
      const mainGrid = resizer.closest(".main-grid");
      const leftPanel = mainGrid ? mainGrid.querySelector(".sensor-panel") : null;
      const startX = e.clientX;
      const startWidth = leftPanel ? leftPanel.getBoundingClientRect().width : 380;

      resizer.classList.add("is-resizing");
      document.body.style.cursor = "col-resize";
      document.body.style.userSelect = "none";

      const onPointerMove = (moveEvent) => {
        const deltaX = moveEvent.clientX - startX;
        const minW = 280;
        const maxW = Math.min(720, window.innerWidth * 0.58);
        const newWidth = Math.max(minW, Math.min(maxW, Math.round(startWidth + deltaX)));
        document.documentElement.style.setProperty("--left-panel-width", `${newWidth}px`);
      };

      const onPointerUp = () => {
        resizer.classList.remove("is-resizing");
        document.body.style.cursor = "";
        document.body.style.userSelect = "";

        const finalWidth = parseInt(getComputedStyle(document.documentElement).getPropertyValue("--left-panel-width"), 10);
        if (finalWidth && typeof localStorage !== "undefined") {
          localStorage.setItem("smartcrop_panel_width", String(finalWidth));
        }

        window.removeEventListener("pointermove", onPointerMove);
        window.removeEventListener("pointerup", onPointerUp);
        window.removeEventListener("pointercancel", onPointerUp);
      };

      window.addEventListener("pointermove", onPointerMove);
      window.addEventListener("pointerup", onPointerUp);
      window.addEventListener("pointercancel", onPointerUp);
    };

    resizer.addEventListener("pointerdown", onPointerDown);
  });
}

// ── KEYBOARD AND MODAL CLOSE LISTENERS ────────────────────────────────────────
function initKeyboardListeners() {
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") {
      closeImageModal();
    }
  });
}

// ── MODE SWITCHER (with silky smooth transition) ──────────────────────────────
let _modeSwitching = false;

function switchMode(mode, animate = true) {
  const btnCrop = document.getElementById("btnModeCrop");
  const btnVision = document.getElementById("btnModeVision");
  const cropSection = document.getElementById("cropSection");
  const visionSection = document.getElementById("visionSection");

  const previousMode = activeMode;
  activeMode = mode;
  window.location.hash = mode;

  // Cleanly close camera streams if navigating away from Plant Vision
  if (mode !== "vision" && typeof closeCameraModal === "function") {
    closeCameraModal();
  }

  // Update button active state
  if (mode === "crop") {
    if (btnCrop) btnCrop.classList.add("active");
    if (btnVision) btnVision.classList.remove("active");
  } else {
    if (btnVision) btnVision.classList.add("active");
    if (btnCrop) btnCrop.classList.remove("active");
  }

  const outgoing = previousMode === "crop" ? cropSection : visionSection;
  const incoming = mode === "crop" ? cropSection : visionSection;

  const iotBar = document.getElementById("iotFeedBar");
  if (iotBar) iotBar.style.display = mode === "crop" ? "flex" : "none";

  const prefersReducedMotion = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  // Instant switch if no animation requested, same mode, or reduced motion
  if (!animate || previousMode === mode || prefersReducedMotion || _modeSwitching) {
    if (outgoing) {
      outgoing.style.display = "none";
      outgoing.style.opacity = "";
      outgoing.style.transform = "";
    }
    if (incoming) {
      incoming.style.display = "grid";
      incoming.style.opacity = "1";
      incoming.style.transform = "none";
    }
    return;
  }

  _modeSwitching = true;

  // Phase 1: Smooth fade out + subtle scale down of outgoing tab
  if (outgoing) {
    outgoing.style.transition = "opacity 0.16s cubic-bezier(0.4, 0, 0.2, 1), transform 0.16s cubic-bezier(0.4, 0, 0.2, 1)";
    outgoing.style.opacity = "0";
    outgoing.style.transform = "translateY(-6px) scale(0.99)";
  }

  setTimeout(() => {
    if (outgoing) {
      outgoing.style.display = "none";
      outgoing.style.transition = "";
      outgoing.style.opacity = "";
      outgoing.style.transform = "";
    }

    // Phase 2: Fade in + subtle slide up of incoming tab
    if (incoming) {
      incoming.style.display = "grid";
      incoming.style.opacity = "0";
      incoming.style.transform = "translateY(8px) scale(0.99)";

      // Force layout calculation before applying transition
      void incoming.offsetWidth;

      incoming.style.transition = "opacity 0.22s cubic-bezier(0.16, 1, 0.3, 1), transform 0.22s cubic-bezier(0.16, 1, 0.3, 1)";
      incoming.style.opacity = "1";
      incoming.style.transform = "translateY(0) scale(1)";
    }

    setTimeout(() => {
      if (incoming) {
        incoming.style.transition = "";
        incoming.style.opacity = "";
        incoming.style.transform = "";
      }
      _modeSwitching = false;
    }, 240);
  }, 160);
}

// ── BACKEND HEALTH AND MODEL READINESS CHECK ──────────────────────────────────
async function checkBackendHealth() {
  const badgeDot = document.getElementById("badgeDot");
  const badgeText = document.getElementById("badgeText");

  try {
    const healthRes = await fetch(`${API_BASE}/health`, { cache: "no-store" });
    if (!healthRes.ok) throw new Error("Health check failed");

    // Fetch model readiness
    let modelsReady = false;
    let readyCount = 0;
    try {
      const modelRes = await fetch(`${API_BASE}/models/status`, { cache: "no-store" });
      if (modelRes.ok) {
        const modelData = await modelRes.json();
        readyCount = modelData.models ? modelData.models.filter(m => m.status === 'ready' && !m.tier?.includes('Crop')).length : (modelData.models_ready || 3);
        modelsReady = readyCount > 0;
      }
    } catch {
      modelsReady = true;
      readyCount = 3;
    }

    if (badgeDot && badgeText) {
      badgeDot.className = "badge-dot online";
      badgeText.textContent = readyCount > 0
        ? `Backend Online · ${readyCount} Core Model${readyCount === 1 ? '' : 's'} Ready`
        : "Backend Online · Models Initializing";
    }
  } catch (err) {
    if (badgeDot && badgeText) {
      badgeDot.className = "badge-dot offline";
      badgeText.textContent = "Backend Offline";
    }
  }
}

// ── ESP8266 PHYSICAL TELEMETRY LIFECYCLE ──────────────────────────────────────
/**
 * Strict rules implemented:
 * 1. ESP8266 must never appear Connected or LIVE when no physical packet has arrived.
 * 2. Backend availability is NOT treated as ESP8266 connectivity.
 * 3. Frontend polling does not create or refresh device connectivity.
 * 4. If telemetry packet is older than 35s, status transitions to STALE / OFFLINE.
 * 5. Manual slider values are clearly badged as MANUAL, never LIVE.
 * 6. Unavailable hardware data shows "--" and "Waiting for ESP8266...".
 */
async function pollLatestIotTelemetry() {
  const iotDot = document.getElementById("iotDot");
  const iotFeedBar = document.getElementById("iotFeedBar");
  const iotFeedText = document.getElementById("iotFeedText");
  const iotLiveValues = document.getElementById("iotLiveValues");

  try {
    const res = await fetch(`${API_BASE}/latest`, { cache: "no-store" });

    if (res.status === 404) {
      // Backend is online, but physical ESP8266 has not transmitted telemetry
      if (latestIotPacket === null) {
        espDeviceState = "waiting";
        if (iotDot) iotDot.className = "iot-dot";
        if (iotFeedBar) iotFeedBar.className = "iot-feed-bar";
        if (iotFeedText) iotFeedText.textContent = "📡 IoT Hardware Sensor Link: Awaiting ESP8266 NodeMCU packets (DHT11, Capacitive Soil, YL-83 Rain) • Manual calibration active";
        if (iotLiveValues) iotLiveValues.style.display = "none";
        setChipsUnavailable();
      }
      return;
    }

    if (!res.ok) {
      throw new Error(`HTTP ${res.status}`);
    }

    const data = await res.json();
    if (!data || data.status === "no_data" || data.temperature == null) {
      if (latestIotPacket === null) {
        espDeviceState = "waiting";
        if (iotDot) iotDot.className = "iot-dot";
        if (iotFeedBar) iotFeedBar.className = "iot-feed-bar";
        if (iotFeedText) iotFeedText.textContent = "📡 IoT Hardware Sensor Link: Awaiting ESP8266 NodeMCU packets (DHT11, Capacitive Soil, YL-83 Rain) • Manual calibration active";
        if (iotLiveValues) iotLiveValues.style.display = "none";
        setChipsUnavailable();
      }
      return;
    }

    // A valid telemetry packet exists. Compute age from physical timestamp.
    latestIotPacket = data;
    const packetTime = data.timestamp ? new Date(data.timestamp).getTime() : Date.now();
    lastIotTimestamp = packetTime;
    const ageSeconds = Math.max(0, Math.floor((Date.now() - packetTime) / 1000));

    // Update the live chips in the top feed bar
    updateIotFeedChips(data, ageSeconds);

    // Evaluate freshness threshold (35 seconds)
    if (ageSeconds <= 35) {
      espDeviceState = "connected";
      if (iotDot) iotDot.className = "iot-dot live";
      if (iotFeedBar) iotFeedBar.className = "iot-feed-bar iot-online";
      if (iotFeedText) iotFeedText.textContent = "📡 ESP8266 Connected (LIVE Hardware Telemetry • Auto-updating)";

      // Automatically apply live sensor telemetry to inputs on connection
      applyLiveTelemetryToInputs(data);

      // Record to chart only on a genuinely new packet
      if (data.timestamp && data.timestamp !== lastChartPacketTs) {
        lastChartPacketTs = data.timestamp;
        addTelemetryToChart(data);
      }
    } else {
      // Packet is stale. Hardware is offline.
      espDeviceState = "stale";
      if (iotDot) iotDot.className = "iot-dot stale";
      if (iotFeedBar) iotFeedBar.className = "iot-feed-bar iot-stale";
      if (iotFeedText) {
        iotFeedText.textContent = `⚠️ ESP8266 Offline (last packet ${formatAgo(ageSeconds)}) • Manual calibration active`;
      }
      markSourceBadgeStale();
    }
  } catch (err) {
    // Network or fetch error
    if (latestIotPacket === null) {
      espDeviceState = "waiting";
      if (iotDot) iotDot.className = "iot-dot";
      if (iotFeedBar) iotFeedBar.className = "iot-feed-bar";
      if (iotFeedText) iotFeedText.textContent = "📡 IoT Hardware Sensor Link: Awaiting ESP8266 NodeMCU packets (DHT11, Capacitive Soil, YL-83 Rain) • Manual calibration active";
      if (iotLiveValues) iotLiveValues.style.display = "none";
      setChipsUnavailable();
    }
  }
}

// ── 1-SECOND AGE TICKER ───────────────────────────────────────────────────────
function updateIotAgeTicker() {
  if (!lastIotTimestamp) {
    return;
  }

  const ageSeconds = Math.max(0, Math.floor((Date.now() - lastIotTimestamp) / 1000));
  const iotAgo = document.getElementById("iotAgo");
  const iotDot = document.getElementById("iotDot");
  const iotFeedBar = document.getElementById("iotFeedBar");
  const iotFeedText = document.getElementById("iotFeedText");

  if (iotAgo) {
    iotAgo.textContent = formatAgo(ageSeconds);
  }

  if (ageSeconds > 35) {
    if (espDeviceState !== "stale") {
      espDeviceState = "stale";
      if (iotDot) iotDot.className = "iot-dot stale";
      if (iotFeedBar) iotFeedBar.className = "iot-feed-bar iot-stale";
      if (iotFeedText) {
        iotFeedText.textContent = `⚠️ ESP8266 Offline (last seen ${formatAgo(ageSeconds)}) • Manual calibration active`;
      }
      markSourceBadgeStale();
    }
  }
}

function updateIotFeedChips(data, ageSeconds) {
  const iotLiveValues = document.getElementById("iotLiveValues");
  const iotT = document.getElementById("iot-t");
  const iotH = document.getElementById("iot-h");
  const iotS = document.getElementById("iot-s");
  const iotR = document.getElementById("iot-r");
  const iotAgo = document.getElementById("iotAgo");

  if (iotLiveValues) iotLiveValues.style.display = "flex";
  if (iotT) iotT.textContent = data.temperature != null ? Number(data.temperature).toFixed(1) : "--";
  if (iotH) iotH.textContent = data.humidity != null ? Number(data.humidity).toFixed(0) : "--";
  if (iotS) iotS.textContent = data.soil_moisture != null ? Number(data.soil_moisture).toFixed(0) : "--";
  if (iotR) iotR.textContent = data.rain === 1 ? "RAIN" : "NO RAIN";
  if (iotAgo) iotAgo.textContent = formatAgo(ageSeconds);
}

function setChipsUnavailable() {
  const iotT = document.getElementById("iot-t");
  const iotH = document.getElementById("iot-h");
  const iotS = document.getElementById("iot-s");
  const iotR = document.getElementById("iot-r");
  const iotAgo = document.getElementById("iotAgo");

  if (iotT) iotT.textContent = "--";
  if (iotH) iotH.textContent = "--";
  if (iotS) iotS.textContent = "--";
  if (iotR) iotR.textContent = "--";
  if (iotAgo) iotAgo.textContent = "--";
}

function applyLiveTelemetryToInputs(data) {
  const tempSlider = document.getElementById("temperature");
  const humSlider = document.getElementById("humidity");
  const soilSlider = document.getElementById("soil_moisture");

  if (tempSlider && data.temperature != null) {
    tempSlider.value = data.temperature;
    document.getElementById("val-temperature").textContent = `${Number(data.temperature).toFixed(1)}°C`;
  }
  if (humSlider && data.humidity != null) {
    humSlider.value = data.humidity;
    document.getElementById("val-humidity").textContent = `${Number(data.humidity).toFixed(0)}%`;
  }
  if (soilSlider && data.soil_moisture != null) {
    soilSlider.value = data.soil_moisture;
    document.getElementById("val-soil_moisture").textContent = `${Number(data.soil_moisture).toFixed(0)}%`;
  }

  setRainState(data.rain === 1);

  // Update left panel badge to LIVE (ESP8266)
  const sourceBadge = document.getElementById("sensorSourceBadge");
  if (sourceBadge) {
    sourceBadge.textContent = "LIVE (ESP8266)";
    sourceBadge.className = "panel-badge panel-badge-live";
  }

  const lastUpdated = document.getElementById("lastUpdated");
  if (lastUpdated) {
    lastUpdated.textContent = new Date().toLocaleTimeString();
  }
}

function markSourceBadgeStale() {
  const sourceBadge = document.getElementById("sensorSourceBadge");
  if (sourceBadge && sourceBadge.classList.contains("panel-badge-live")) {
    sourceBadge.textContent = "STALE (ESP8266)";
    sourceBadge.className = "panel-badge panel-badge-stale";
  }
}

function formatAgo(seconds) {
  if (seconds <= 2) return "Just now";
  if (seconds < 60) return `${seconds}s ago`;
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.floor(minutes / 60);
  return `${hours}h ago`;
}

// ── MANUAL SENSOR INPUT HANDLERS ──────────────────────────────────────────────
function onManualInput(sensorId, value) {
  isManualInput = true;
  const displayMap = {
    temperature: `${Number(value).toFixed(1)}°C`,
    humidity: `${Number(value).toFixed(0)}%`,
    soil_moisture: `${Number(value).toFixed(0)}%`,
  };

  const displayEl = document.getElementById(`val-${sensorId}`);
  if (displayEl && displayMap[sensorId]) {
    displayEl.textContent = displayMap[sensorId];
  }

  const sourceBadge = document.getElementById("sensorSourceBadge");
  if (sourceBadge) {
    sourceBadge.textContent = "MANUAL";
    sourceBadge.className = "panel-badge panel-badge-manual";
  }
}

function toggleRain() {
  isManualInput = true;
  setRainState(!isRainActive);

  const sourceBadge = document.getElementById("sensorSourceBadge");
  if (sourceBadge) {
    sourceBadge.textContent = "MANUAL";
    sourceBadge.className = "panel-badge panel-badge-manual";
  }
}

function setRainState(active) {
  isRainActive = Boolean(active);
  const rainToggle = document.getElementById("rainToggle");
  const rainIcon = document.getElementById("rainIcon");
  const rainLabel = document.getElementById("rainLabel");

  if (rainToggle) {
    if (isRainActive) {
      rainToggle.className = "rain-toggle rain-on";
      rainToggle.setAttribute("aria-pressed", "true");
      if (rainIcon) rainIcon.textContent = "🌧️";
      if (rainLabel) rainLabel.textContent = "RAINING";
    } else {
      rainToggle.className = "rain-toggle rain-off";
      rainToggle.setAttribute("aria-pressed", "false");
      if (rainIcon) rainIcon.textContent = "☀️";
      if (rainLabel) rainLabel.textContent = "NO RAIN";
    }
  }
}

function loadPreset(presetName) {
  const preset = SENSOR_PRESETS[presetName];
  if (!preset) return;

  isManualInput = true;
  document.getElementById("temperature").value = preset.temperature;
  document.getElementById("humidity").value = preset.humidity;
  document.getElementById("soil_moisture").value = preset.soil_moisture;
  setRainState(preset.rain === 1);

  syncSliderDisplays();

  const sourceBadge = document.getElementById("sensorSourceBadge");
  if (sourceBadge) {
    sourceBadge.textContent = "MANUAL";
    sourceBadge.className = "panel-badge panel-badge-manual";
  }

  runPrediction();
}

function syncSliderDisplays() {
  const tempVal = document.getElementById("temperature")?.value;
  const humVal = document.getElementById("humidity")?.value;
  const soilVal = document.getElementById("soil_moisture")?.value;

  if (tempVal) document.getElementById("val-temperature").textContent = `${Number(tempVal).toFixed(1)}°C`;
  if (humVal) document.getElementById("val-humidity").textContent = `${Number(humVal).toFixed(0)}%`;
  if (soilVal) document.getElementById("val-soil_moisture").textContent = `${Number(soilVal).toFixed(0)}%`;
}

function initAutoToggle() {
  const autoToggle = document.getElementById("autoToggle");
  if (autoToggle) {
    autoToggle.addEventListener("change", (e) => {
      autoUpdateEnabled = e.target.checked;
    });
  }
}

// ── CROP PREDICTION & DISEASE RISK PIPELINE ───────────────────────────────────
async function runPrediction() {
  const temp = parseFloat(document.getElementById("temperature").value);
  const hum = parseFloat(document.getElementById("humidity").value);
  const soil = parseFloat(document.getElementById("soil_moisture").value);
  const rain = isRainActive ? 1 : 0;

  const payload = {
    temperature: temp,
    humidity: hum,
    soil_moisture: soil,
    rain: rain,
  };

  showLoading("Calculating Crop Intelligence & Disease Risks...");
  renderCropSkeletonLoading();

  try {
    const res = await fetch(`${API_BASE}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!res.ok) {
      const errData = await res.json().catch(() => ({}));
      throw new Error(errData.detail || `Server returned HTTP ${res.status}`);
    }

    const data = await res.json();
    document.getElementById("cropSection")?.classList.add("has-results");
    renderCropRecommendation(data);
    renderDiseaseRisks(data.disease_alerts || data.disease_risks || []);
    renderRiskFlags(data.features || data.risk_flags || {});

    const lastUpdated = document.getElementById("lastUpdated");
    if (lastUpdated) {
      lastUpdated.textContent = new Date().toLocaleTimeString();
    }
  } catch (err) {
    showErrorNotification(`Prediction error: ${err.message}`);
  } finally {
    hideLoading();
  }
}

function renderCropRecommendation(data) {
  const container = document.getElementById("recommendationContent");
  if (!container) return;

  const topCrop = data.recommended_crop || "Unknown";
  let rawConf = Number(data.confidence || 0);
  if (rawConf <= 1.0 && rawConf > 0) {
    rawConf = rawConf * 100;
  }
  const confNum = Math.min(100, Math.max(0, rawConf));
  const confStr = `${confNum.toFixed(1)}%`;

  const topAlternatives = data.top3 ? data.top3.slice(1) : (data.top_alternatives || []);
  const conditions = data.features || data.growing_conditions || {};

  const cropIcons = {
    rice: "🌾",
    maize: "🌽",
    cotton: "🌱",
    wheat: "🌾",
    sugarcane: "🎋",
    coffee: "☕",
    jute: "🌿",
    lentil: "🫘",
    pigeonpeas: "🫛",
    chickpea: "🧆",
    mungbean: "🌱",
    blackgram: "🫘",
    kidneybeans: "🫘",
    coconut: "🥥",
    banana: "🍌",
    apple: "🍎",
    orange: "🍊",
    papaya: "🍈",
    watermelon: "🍉",
    muskmelon: "🍈",
    grapes: "🍇",
    mango: "🥭",
    pomegranate: "🍎",
  };

  const cropKey = topCrop.toLowerCase().replace(/[^a-z]/g, "");
  const icon = cropIcons[cropKey] || "🌿";

  const currentTemp = document.getElementById("temperature")?.value;
  const currentHum = document.getElementById("humidity")?.value;
  const currentSoil = document.getElementById("soil_moisture")?.value;

  const envTemp = conditions.temperature != null ? Number(conditions.temperature).toFixed(1) : (currentTemp ? Number(currentTemp).toFixed(1) : "25.0");
  const envHum = conditions.humidity != null ? Number(conditions.humidity).toFixed(0) : (currentHum ? Number(currentHum).toFixed(0) : "65");
  const envSoil = conditions.soil_moisture != null ? Number(conditions.soil_moisture).toFixed(0) : (currentSoil ? Number(currentSoil).toFixed(0) : "50");
  const envRain = (conditions.rain === 1 || isRainActive) ? "Active Rainfall" : "Clear / Arid";

  let altsHtml = "";
  if (topAlternatives.length > 0) {
    altsHtml = `
      <div class="top3-list" style="margin-top: 14px;">
        <div class="top3-label">Secondary Viable Options</div>
        ${topAlternatives
          .map((alt, idx) => {
            const altKey = alt.crop ? alt.crop.toLowerCase().replace(/[^a-z]/g, "") : "";
            const altIcon = alt.icon || cropIcons[altKey] || "🌱";
            let rawAlt = Number(alt.confidence || 0);
            if (rawAlt <= 1.0 && rawAlt > 0) rawAlt = rawAlt * 100;
            const altPct = Math.min(100, Math.max(0, rawAlt)).toFixed(1);
            return `
            <div class="top3-item">
              <span class="top3-rank">${idx + 2}</span>
              <span class="top3-crop">${altIcon} ${alt.crop}</span>
              <div class="top3-bar-wrap">
                <div class="top3-bar" style="width: ${altPct}%;"></div>
              </div>
              <span class="top3-conf">${altPct}%</span>
            </div>
          `;
          })
          .join("")}
      </div>
    `;
  }

  container.innerHTML = `
    <div class="crop-result">
      <div class="crop-main-card">
        <div class="crop-emoji">${icon}</div>
        <div class="crop-info">
          <div class="crop-name">${topCrop}</div>
          <div class="crop-confidence-text">Optimal Ecological Match · Recommended Crop</div>
        </div>
        <div class="confidence-ring-wrap">
          <div class="confidence-ring" style="background: conic-gradient(var(--green-primary) ${confNum * 3.6}deg, rgba(82, 183, 136, 0.15) 0deg);">
            <span class="confidence-pct">${confStr}</span>
          </div>
        </div>
      </div>

      <div class="rec-env-pills">
        <span class="env-pill">🌡️ ${envTemp}°C</span>
        <span class="env-pill">💧 ${envHum}% RH</span>
        <span class="env-pill">🌱 ${envSoil}% Moisture</span>
        <span class="env-pill">🌧️ ${envRain}</span>
      </div>

      ${altsHtml}
    </div>
  `;
}

function renderDiseaseRisks(risks) {
  const container = document.getElementById("diseaseContent");
  const alertBadge = document.getElementById("alertCountBadge");
  if (!container) return;

  if (!risks || risks.length === 0) {
    if (alertBadge) alertBadge.style.display = "none";
    container.innerHTML = `
      <div class="all-clear">
        <span class="all-clear-icon">🛡️</span>
        <div>
          <strong style="color: var(--green-bright); font-size: 13px;">All Environmental Conditions Optimal</strong>
          <p style="margin-top: 3px; color: var(--text-secondary); font-size: 11.5px; line-height: 1.4;">Zero fungal, bacteriological, or physiological pathology thresholds exceeded under current readings.</p>
        </div>
      </div>
    `;
    return;
  }

  if (alertBadge) {
    alertBadge.style.display = "inline-flex";
    alertBadge.textContent = risks.length;
  }

  container.innerHTML = `
    <div class="disease-content">
      ${risks
        .map((risk) => {
          const sev = (risk.severity || risk.risk_level || "ALERT").toUpperCase();
          const borderColor = sev === "HIGH" ? "var(--red)" : (sev === "MODERATE" ? "var(--yellow)" : "var(--green-primary)");
          const sevColor = sev === "HIGH" ? "var(--red)" : (sev === "MODERATE" ? "var(--yellow)" : "var(--green-bright)");
          const icon = risk.icon || (sev === "HIGH" ? "🔴" : "⚠️");
          const name = risk.name || "Disease Risk";
          const type = risk.type || "PATHOLOGY";
          const trigger = risk.trigger || risk.reason || risk.symptoms || "Environmental threshold reached";
          const symptoms = risk.symptoms || "";
          const cultural = risk.technique || risk.cultural_control || "";
          const chemical = risk.pesticide || risk.chemical_control || "";

          return `
          <div class="disease-card" style="border-left-color: ${borderColor};">
            <div class="disease-card-header" onclick="this.nextElementSibling.classList.toggle('open')">
              <span class="disease-severity-icon">${icon}</span>
              <span class="disease-name">${name}</span>
              <span class="disease-type-badge">${type}</span>
              <span class="disease-severity-label" style="color: ${sevColor};">${sev} RISK</span>
            </div>
            <div class="disease-body open">
              <div class="disease-field">
                <strong>Trigger Condition</strong>
                <p>${trigger}</p>
              </div>
              ${
                symptoms
                  ? `<div class="disease-field">
                      <strong>Diagnostic Symptoms</strong>
                      <p>${symptoms}</p>
                    </div>`
                  : ""
              }
              ${
                cultural
                  ? `<div class="disease-field">
                      <strong>Cultural Practice</strong>
                      <p>${cultural}</p>
                    </div>`
                  : ""
              }
              ${
                chemical
                  ? `<div class="disease-field">
                      <strong>Sanitary Protocol</strong>
                      <p>${chemical}</p>
                    </div>`
                  : ""
              }
            </div>
          </div>
        `;
        })
        .join("")}
    </div>
  `;
}

function renderRiskFlags(flags) {
  const fungalEl = document.getElementById("feat-fungal_risk");
  const droughtEl = document.getElementById("feat-drought_risk");
  const waterlogEl = document.getElementById("feat-waterlog_risk");

  updateFlagElement(fungalEl, flags.fungal_risk);
  updateFlagElement(droughtEl, flags.drought_risk);
  updateFlagElement(waterlogEl, flags.waterlog_risk);
}

function updateFlagElement(el, flagValue) {
  if (!el) return;
  if (flagValue === 1 || flagValue === true || flagValue === "HIGH") {
    el.textContent = "HIGH";
    el.className = "feature-val flag-high";
  } else if (flagValue === "MODERATE") {
    el.textContent = "MODERATE";
    el.className = "feature-val flag-moderate";
  } else {
    el.textContent = "LOW / SAFE";
    el.className = "feature-val flag-low";
  }
}

// ── SENSOR READING HISTORY CHART ──────────────────────────────────────────────
/**
 * Strict rules implemented:
 * 1. History chart displays ONLY genuine physical hardware telemetry.
 * 2. Frontend polling does NOT create or append synthetic entries.
 * 3. Manual prediction clicks do NOT pollute the history chart.
 */
function initHistoryChart() {
  const canvas = document.getElementById("historyChart");
  if (!canvas || typeof Chart === "undefined") return;

  const ctx = canvas.getContext("2d");
  historyChart = new Chart(ctx, {
    type: "line",
    data: {
      labels: [],
      datasets: [
        {
          label: "Temperature (°C)",
          data: [],
          borderColor: "#e07a5f",
          backgroundColor: "rgba(224, 122, 95, 0.15)",
          borderWidth: 2,
          pointRadius: 3,
          tension: 0.3,
          yAxisID: "yTemp",
        },
        {
          label: "Humidity (%)",
          data: [],
          borderColor: "#3d9970",
          backgroundColor: "rgba(61, 153, 112, 0.15)",
          borderWidth: 2,
          pointRadius: 3,
          tension: 0.3,
          yAxisID: "yPct",
        },
        {
          label: "Soil Moisture (%)",
          data: [],
          borderColor: "#81b29a",
          backgroundColor: "rgba(129, 178, 154, 0.15)",
          borderWidth: 2,
          pointRadius: 3,
          tension: 0.3,
          yAxisID: "yPct",
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 400 },
      interaction: { mode: "index", intersect: false },
      plugins: {
        legend: {
          position: "top",
          labels: { color: "#8a9e8a", boxWidth: 14, font: { family: "Inter", size: 12 } },
        },
        tooltip: {
          backgroundColor: "#0d1a0d",
          borderColor: "#1e381e",
          borderWidth: 1,
          titleColor: "#e8f5e9",
          bodyColor: "#8a9e8a",
        },
      },
      scales: {
        x: {
          grid: { color: "#162816" },
          ticks: { color: "#527952", font: { family: "Inter", size: 11 } },
        },
        yTemp: {
          type: "linear",
          position: "left",
          min: 0,
          max: 50,
          grid: { color: "#162816" },
          ticks: {
            color: "#e07a5f",
            font: { family: "Inter", size: 11 },
            callback: (v) => `${v}°C`,
          },
        },
        yPct: {
          type: "linear",
          position: "right",
          min: 0,
          max: 100,
          grid: { drawOnChartArea: false },
          ticks: {
            color: "#81b29a",
            font: { family: "Inter", size: 11 },
            callback: (v) => `${v}%`,
          },
        },
      },
    },
  });
}

async function fetchTelemetryHistory() {
  try {
    const res = await fetch(`${API_BASE}/history`, { cache: "no-store" });
    if (!res.ok) return;
    const json = await res.json();
    const readings = json.readings || json.history || [];
    // CRITICAL: Filter ONLY physical IoT hardware readings. Never plot dashboard prediction tests or default values.
    const iotReadings = Array.isArray(readings) ? readings.filter((r) => r.source === "iot") : [];
    const emptyState = document.getElementById("chartEmptyState");

    if (iotReadings.length > 0) {
      if (emptyState) emptyState.style.display = "none";
      iotReadings.forEach((r) => addTelemetryToChart(r, false));
      if (historyChart) historyChart.update();
    } else {
      if (emptyState) emptyState.style.display = "flex";
    }
  } catch {
    // Graceful fallback when no history exists yet
  }
}

function addTelemetryToChart(packet, shouldUpdate = true) {
  if (!historyChart || packet.temperature == null) return;
  // Guard: Only real IoT packets are plotted
  if (packet.source && packet.source !== "iot") return;

  const emptyState = document.getElementById("chartEmptyState");
  if (emptyState) emptyState.style.display = "none";

  const timeLabel = packet.timestamp
    ? new Date(packet.timestamp).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" })
    : new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });

  historyChart.data.labels.push(timeLabel);
  historyChart.data.datasets[0].data.push(Number(packet.temperature));
  historyChart.data.datasets[1].data.push(Number(packet.humidity));
  historyChart.data.datasets[2].data.push(Number(packet.soil_moisture));

  // Maintain sliding window of 25 physical readings
  if (historyChart.data.labels.length > 25) {
    historyChart.data.labels.shift();
    historyChart.data.datasets[0].data.shift();
    historyChart.data.datasets[1].data.shift();
    historyChart.data.datasets[2].data.shift();
  }

  if (shouldUpdate) {
    historyChart.update();
  }
}

// ── COMPUTER VISION: DRAG & DROP AND FILE SELECTION ───────────────────────────
function initDragAndDrop() {
  const dropzone = document.getElementById("cvDropzone");
  if (!dropzone) return;

  ["dragenter", "dragover"].forEach((eventName) => {
    dropzone.addEventListener(
      eventName,
      (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropzone.classList.add("drag-over");
      },
      false
    );
  });

  ["dragleave", "drop"].forEach((eventName) => {
    dropzone.addEventListener(
      eventName,
      (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropzone.classList.remove("drag-over");
      },
      false
    );
  });

  dropzone.addEventListener("drop", (e) => {
    const dt = e.dataTransfer;
    const files = dt.files;
    if (files && files.length > 0) {
      processSelectedImageFile(files[0]);
    }
  });
}

function triggerCvFileInput(event) {
  if (event && typeof event.stopPropagation === "function") {
    event.stopPropagation();
  }
  const fileInput = document.getElementById("cvFileInput");
  if (fileInput) {
    fileInput.value = ""; // Reset value so re-selecting identical file triggers onchange reliably
    fileInput.click();
  }
}

function handleDropzoneContainerClick(event) {
  // If the user clicked on camera trigger button, do not trigger file picker
  if (event && event.target && event.target.closest) {
    if (event.target.closest("#btnCameraTrigger") || event.target.closest(".btn-camera-trigger") || event.target.closest(".btn-camera-recapture")) {
      return;
    }
  }
  // If an image is already uploaded or active, NEVER trigger file input from container background clicks
  if (selectedVisionFile != null || currentPreviewObjectUrl != null) {
    if (event && typeof event.stopPropagation === "function") event.stopPropagation();
    return;
  }
  const previewWrap = document.getElementById("dropzonePreviewWrap");
  if (previewWrap && previewWrap.style.display !== "none") {
    if (event && typeof event.stopPropagation === "function") event.stopPropagation();
    return;
  }
  triggerCvFileInput(event);
}

// ═════════════════════════════════════════════════════════════════════════════
// NATIVE DEVICE CAMERA CAPTURE CONTROLLER (SmartCropVisionCamera)
// ═════════════════════════════════════════════════════════════════════════════
const SmartCropVisionCamera = {
  activeStream: null,
  activeVideoTrack: null,
  currentFacingMode: "environment", // Automatically prefer rear camera on mobile
  availableVideoDevices: [],
  currentDeviceIndex: 0,
  isTorchOn: false,
  hasTorchCapability: false,
  capturedBlob: null,
  savedScrollY: 0,
  isInitializing: false,
  isCaptured: false,
  keyHandlerBound: null,

  async open() {
    const modal = document.getElementById("cameraModal");
    if (!modal) return;

    this.savedScrollY = window.scrollY || window.pageYOffset || 0;
    document.body.classList.add("camera-modal-open");
    modal.style.display = "flex";

    this.resetUI();
    this.bindKeyboard();

    // Check secure context (camera permissions require HTTPS or localhost)
    const isLocalhost = Boolean(
      window.location.hostname === "localhost" ||
      window.location.hostname === "127.0.0.1" ||
      window.location.hostname === "[::1]"
    );
    if (!window.isSecureContext && !isLocalhost) {
      this.showError(
        "Secure Context Required",
        "Camera access requires a secure HTTPS connection or localhost for browser security permissions. Please upload an image file instead.",
        false
      );
      return;
    }

    // Check browser mediaDevices API
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      this.showError(
        "Camera Not Supported",
        "Your browser does not support the modern MediaDevices camera API. Please upload an image file instead.",
        false
      );
      return;
    }

    await this.startStream(this.currentFacingMode);
  },

  resetUI() {
    this.isCaptured = false;
    this.capturedBlob = null;
    this.isTorchOn = false;
    this.hasTorchCapability = false;

    const video = document.getElementById("cameraVideo");
    const canvas = document.getElementById("cameraReviewCanvas");
    const liveControls = document.getElementById("cameraLiveControls");
    const reviewControls = document.getElementById("cameraReviewControls");
    const scanFrame = document.getElementById("cameraScanFrame");
    const loadingOverlay = document.getElementById("cameraLoadingOverlay");
    const errorOverlay = document.getElementById("cameraErrorOverlay");

    if (video) {
      video.style.display = "block";
      video.classList.remove("mirrored");
    }
    if (canvas) canvas.style.display = "none";
    if (liveControls) liveControls.style.display = "flex";
    if (reviewControls) reviewControls.style.display = "none";
    if (scanFrame) scanFrame.style.display = "flex";
    if (loadingOverlay) loadingOverlay.style.display = "flex";
    if (errorOverlay) errorOverlay.style.display = "none";

    this.updateStatus("starting", "Camera Starting...");
    this.updateTorchUI(false, false);
  },

  updateStatus(state, text) {
    const pill = document.getElementById("cameraStatusPill");
    const dot = document.getElementById("cameraStatusDot");
    const textEl = document.getElementById("cameraStatusText");
    if (!pill || !dot || !textEl) return;

    dot.className = "camera-status-dot";
    if (state === "starting") {
      dot.classList.add("starting");
    } else if (state === "unavailable") {
      dot.classList.add("unavailable");
    }
    textEl.textContent = text;
  },

  async startStream(preferredFacing, explicitDeviceId = null) {
    if (this.isInitializing) return;
    this.isInitializing = true;

    this.stopAllStreams();

    const loadingOverlay = document.getElementById("cameraLoadingOverlay");
    const errorOverlay = document.getElementById("cameraErrorOverlay");
    const loadingMsg = document.getElementById("cameraLoadingMsg");
    if (loadingOverlay) loadingOverlay.style.display = "flex";
    if (errorOverlay) errorOverlay.style.display = "none";
    if (loadingMsg) loadingMsg.textContent = "Accessing device camera...";

    this.updateStatus("starting", "Camera Starting...");

    let constraints = {
      audio: false,
      video: explicitDeviceId
        ? {
            deviceId: { exact: explicitDeviceId },
            width: { ideal: 1920 },
            height: { ideal: 1080 }
          }
        : {
            facingMode: { ideal: preferredFacing || "environment" },
            width: { ideal: 1920 },
            height: { ideal: 1080 }
          }
    };

    let stream = null;
    try {
      stream = await navigator.mediaDevices.getUserMedia(constraints);
    } catch (err) {
      console.warn("Primary camera constraint failed, falling back to basic video:", err);
      try {
        stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
      } catch (fallbackErr) {
        this.isInitializing = false;
        if (loadingOverlay) loadingOverlay.style.display = "none";
        this.handleCameraError(fallbackErr);
        return;
      }
    }

    this.activeStream = stream;
    const tracks = stream.getVideoTracks();
    if (!tracks || tracks.length === 0) {
      this.isInitializing = false;
      this.showError("No Video Track", "The camera stream was opened but no video track was returned.", true);
      return;
    }

    this.activeVideoTrack = tracks[0];

    // Handle sudden track disconnection
    this.activeVideoTrack.onended = () => {
      console.warn("Camera track disconnected.");
      this.updateStatus("unavailable", "Camera Disconnected");
    };

    const video = document.getElementById("cameraVideo");
    if (video) {
      video.srcObject = stream;
      try {
        await video.play();
      } catch (playErr) {
        console.warn("Video play notice:", playErr);
      }
    }

    // Determine facing mode and mirroring
    const trackSettings = typeof this.activeVideoTrack.getSettings === "function" ? this.activeVideoTrack.getSettings() : {};
    const effectiveFacing = trackSettings.facingMode || preferredFacing || "environment";
    const isFront = effectiveFacing === "user";

    if (video) {
      video.classList.toggle("mirrored", isFront);
    }

    if (loadingOverlay) loadingOverlay.style.display = "none";
    this.isInitializing = false;

    // Detect device enumeration & torch capabilities
    await this.inspectCapabilities(isFront);

    const facingLabel = isFront ? "Front Camera" : "Rear Camera";
    this.updateStatus("ready", `Camera Ready · ${facingLabel}`);
  },

  async inspectCapabilities(isFront) {
    // 1. Enumerate video devices
    try {
      const devices = await navigator.mediaDevices.enumerateDevices();
      this.availableVideoDevices = devices.filter(d => d.kind === "videoinput");
      const switchBtn = document.getElementById("cameraSwitchBtn");
      if (switchBtn) {
        if (this.availableVideoDevices.length > 1) {
          switchBtn.style.display = "inline-flex";
          switchBtn.disabled = false;
          switchBtn.title = `Switch to ${isFront ? "Rear" : "Front"} Camera`;
        } else {
          switchBtn.style.display = "none";
        }
      }
    } catch (e) {
      console.warn("Device enumeration notice:", e);
    }

    // 2. Detect hardware torch
    this.hasTorchCapability = false;
    this.isTorchOn = false;
    if (this.activeVideoTrack && typeof this.activeVideoTrack.getCapabilities === "function") {
      try {
        const caps = this.activeVideoTrack.getCapabilities();
        if (caps && caps.torch) {
          this.hasTorchCapability = true;
        }
      } catch (e) {
        console.warn("Torch capability query notice:", e);
      }
    }

    this.updateTorchUI(this.hasTorchCapability, false);
  },

  updateTorchUI(supported, isOn) {
    const torchBtn = document.getElementById("cameraTorchBtn");
    const torchLabel = document.getElementById("cameraTorchLabel");
    if (!torchBtn || !torchLabel) return;

    if (!supported) {
      torchBtn.disabled = true;
      torchBtn.classList.remove("active");
      torchBtn.title = "Flash / torch is unavailable on this device or browser";
      torchLabel.textContent = "No Flash";
    } else {
      torchBtn.disabled = false;
      torchBtn.classList.toggle("active", isOn);
      torchBtn.title = isOn ? "Turn Flash Off" : "Turn Flash On";
      torchLabel.textContent = isOn ? "Flash On" : "Flash Off";
    }
  },

  async toggleTorch() {
    if (!this.hasTorchCapability || !this.activeVideoTrack) return;
    try {
      this.isTorchOn = !this.isTorchOn;
      await this.activeVideoTrack.applyConstraints({
        advanced: [{ torch: this.isTorchOn }]
      });
      this.updateTorchUI(true, this.isTorchOn);
    } catch (err) {
      console.warn("Failed to toggle torch:", err);
      this.isTorchOn = false;
      this.updateTorchUI(true, false);
    }
  },

  async switchCamera() {
    if (this.isInitializing || this.isCaptured) return;
    if (this.availableVideoDevices.length <= 1) return;

    this.currentFacingMode = this.currentFacingMode === "environment" ? "user" : "environment";
    this.currentDeviceIndex = (this.currentDeviceIndex + 1) % this.availableVideoDevices.length;
    const targetDeviceId = this.availableVideoDevices[this.currentDeviceIndex].deviceId;

    await this.startStream(this.currentFacingMode, targetDeviceId);
  },

  capture() {
    if (!this.activeStream || this.isCaptured) return;
    const video = document.getElementById("cameraVideo");
    const canvas = document.getElementById("cameraReviewCanvas");
    const flash = document.getElementById("cameraShutterFlash");
    if (!video || !canvas) return;

    // Fast shutter flash animation
    if (flash) {
      flash.classList.remove("flash-active");
      void flash.offsetWidth; // Force CSS reflow
      flash.classList.add("flash-active");
      setTimeout(() => flash.classList.remove("flash-active"), 200);
    }

    const vw = video.videoWidth || 1280;
    const vh = video.videoHeight || 720;
    canvas.width = vw;
    canvas.height = vh;

    const ctx = canvas.getContext("2d");
    const isMirrored = video.classList.contains("mirrored");

    if (isMirrored) {
      // Preserve visual orientation user saw in preview
      ctx.save();
      ctx.translate(vw, 0);
      ctx.scale(-1, 1);
      ctx.drawImage(video, 0, 0, vw, vh);
      ctx.restore();
    } else {
      ctx.drawImage(video, 0, 0, vw, vh);
    }

    // Freeze preview by swapping video for review canvas
    video.style.display = "none";
    canvas.style.display = "block";

    // Hide scanning guide and display review controls
    const scanFrame = document.getElementById("cameraScanFrame");
    if (scanFrame) scanFrame.style.display = "none";

    const liveControls = document.getElementById("cameraLiveControls");
    const reviewControls = document.getElementById("cameraReviewControls");
    if (liveControls) liveControls.style.display = "none";
    if (reviewControls) reviewControls.style.display = "flex";

    // Extinguish torch if it was active
    if (this.isTorchOn && this.activeVideoTrack) {
      this.activeVideoTrack.applyConstraints({ advanced: [{ torch: false }] }).catch(() => {});
      this.isTorchOn = false;
      this.updateTorchUI(this.hasTorchCapability, false);
    }

    this.isCaptured = true;
    this.updateStatus("ready", "Photo Captured · Review");

    canvas.toBlob((blob) => {
      this.capturedBlob = blob;
    }, "image/jpeg", 0.95);
  },

  retake() {
    if (!this.isCaptured) return;
    this.isCaptured = false;
    this.capturedBlob = null;

    const video = document.getElementById("cameraVideo");
    const canvas = document.getElementById("cameraReviewCanvas");
    const scanFrame = document.getElementById("cameraScanFrame");
    const liveControls = document.getElementById("cameraLiveControls");
    const reviewControls = document.getElementById("cameraReviewControls");

    if (canvas) canvas.style.display = "none";
    if (video) video.style.display = "block";
    if (scanFrame) scanFrame.style.display = "flex";
    if (reviewControls) reviewControls.style.display = "none";
    if (liveControls) liveControls.style.display = "flex";

    const isFront = video && video.classList.contains("mirrored");
    this.updateStatus("ready", `Camera Ready · ${isFront ? "Front Camera" : "Rear Camera"}`);
  },

  async usePhoto() {
    if (!this.capturedBlob) {
      showErrorNotification("Unable to process captured frame. Please retake photo.");
      return;
    }

    const filename = `leaf_camera_${Date.now()}.jpg`;
    const capturedFile = new File([this.capturedBlob], filename, {
      type: "image/jpeg",
      lastModified: Date.now()
    });

    currentVisionImageSource = "browser_camera";
    currentEsp32CaptureId = null;
    currentEsp32CaptureMeta = null;
    const provTag = document.getElementById("cvProvenanceTag");
    if (provTag) {
      provTag.style.display = "inline-flex";
      provTag.className = "provenance-tag camera-provenance";
      provTag.innerHTML = `📸 Device Camera`;
    }

    this.close();

    showToast("Photo captured from camera. Verifying leaf specimen...", "info", "Specimen Received");

    if (typeof processSelectedImageFile === "function") {
      await processSelectedImageFile(capturedFile, false, "browser_camera");
    }
  },

  retry() {
    this.resetUI();
    this.startStream(this.currentFacingMode);
  },

  handleCameraError(err) {
    console.error("Camera access error:", err);
    let title = "Camera Unavailable";
    let desc = "Unable to access your device camera.";

    if (err.name === "NotAllowedError" || err.name === "PermissionDeniedError") {
      title = "Camera Permission Denied";
      desc = "Camera access was blocked by browser permissions. Please allow camera permissions in your address bar, or upload an image file.";
    } else if (err.name === "NotFoundError" || err.name === "DevicesNotFoundError") {
      title = "No Camera Detected";
      desc = "No camera device was detected on your hardware. Please connect a webcam or upload a photo from your file system.";
    } else if (err.name === "NotReadableError" || err.name === "TrackStartError") {
      title = "Camera In Use";
      desc = "Your device camera is currently in use by another application or tab. Please close other camera apps and try again.";
    } else if (err.name === "OverconstrainedError") {
      title = "Camera Mode Unsupported";
      desc = "The requested camera resolution or facing mode is unsupported by your hardware.";
    }

    this.showError(title, desc, true);
  },

  showError(title, desc, allowRetry = true) {
    this.updateStatus("unavailable", "Camera Unavailable");

    const loadingOverlay = document.getElementById("cameraLoadingOverlay");
    const errorOverlay = document.getElementById("cameraErrorOverlay");
    const titleEl = document.getElementById("cameraErrorTitle");
    const descEl = document.getElementById("cameraErrorDesc");
    const retryBtn = errorOverlay?.querySelector(".camera-btn-retry");

    if (loadingOverlay) loadingOverlay.style.display = "none";
    if (titleEl) titleEl.textContent = title;
    if (descEl) descEl.textContent = desc;
    if (retryBtn) retryBtn.style.display = allowRetry ? "inline-block" : "none";
    if (errorOverlay) errorOverlay.style.display = "flex";
  },

  bindKeyboard() {
    this.unbindKeyboard();
    this.keyHandlerBound = (e) => {
      const modal = document.getElementById("cameraModal");
      if (!modal || modal.style.display === "none") return;

      if (e.key === "Escape") {
        e.preventDefault();
        this.close();
      } else if (e.key === " " || e.key === "Enter") {
        if (e.target && (e.target.tagName === "BUTTON" || e.target.getAttribute("role") === "button")) {
          return;
        }
        e.preventDefault();
        if (this.isCaptured) {
          this.usePhoto();
        } else {
          this.capture();
        }
      }
    };
    window.addEventListener("keydown", this.keyHandlerBound);
  },

  unbindKeyboard() {
    if (this.keyHandlerBound) {
      window.removeEventListener("keydown", this.keyHandlerBound);
      this.keyHandlerBound = null;
    }
  },

  stopAllStreams() {
    if (this.activeVideoTrack) {
      try { this.activeVideoTrack.stop(); } catch (e) {}
      this.activeVideoTrack = null;
    }
    if (this.activeStream) {
      try {
        this.activeStream.getTracks().forEach(track => {
          try { track.stop(); } catch (e) {}
        });
      } catch (e) {}
      this.activeStream = null;
    }
    const video = document.getElementById("cameraVideo");
    if (video) {
      try { video.srcObject = null; } catch (e) {}
    }
    this.isTorchOn = false;
  },

  close() {
    this.stopAllStreams();
    this.unbindKeyboard();

    const modal = document.getElementById("cameraModal");
    if (modal) {
      modal.style.display = "none";
    }

    document.body.classList.remove("camera-modal-open");
    window.scrollTo(0, this.savedScrollY);
  }
};

// Global Camera Action Wrappers
function openCameraModal(event) {
  if (event && typeof event.stopPropagation === "function") {
    event.stopPropagation();
  }
  SmartCropVisionCamera.open();
}

function closeCameraModal() {
  SmartCropVisionCamera.close();
}

function handleCameraBackdropClick(event) {
  if (event && event.target === document.getElementById("cameraModal")) {
    closeCameraModal();
  }
}

function captureCameraFrame() {
  SmartCropVisionCamera.capture();
}

function retakeCameraPhoto() {
  SmartCropVisionCamera.retake();
}

async function useCapturedPhoto() {
  await SmartCropVisionCamera.usePhoto();
}

async function switchCameraFacing() {
  await SmartCropVisionCamera.switchCamera();
}

async function toggleCameraTorch() {
  await SmartCropVisionCamera.toggleTorch();
}

function retryCameraStream() {
  SmartCropVisionCamera.retry();
}

function fallbackToUploadFromCamera() {
  closeCameraModal();
  triggerCvFileInput();
}

// Ensure camera stream is stopped if window is closed or backgrounded
if (typeof window !== "undefined") {
  window.addEventListener("beforeunload", () => {
    SmartCropVisionCamera.stopAllStreams();
    stopEsp32Stream();
  });
  window.addEventListener("pagehide", () => {
    SmartCropVisionCamera.stopAllStreams();
    stopEsp32Stream();
  });
}

// ═════════════════════════════════════════════════════════════════════════════
// ESP32-CAM (OV2640) FIELD HARDWARE CONTROLLER & 3-MODE INPUT SWITCHER
// ═════════════════════════════════════════════════════════════════════════════

function setVisionInputMode(mode) {
  currentVisionInputMode = mode;

  const tabUpload = document.getElementById("tabModeUpload");
  const tabCamera = document.getElementById("tabModeCamera");
  const tabEsp32 = document.getElementById("tabModeEsp32");
  const dropzone = document.getElementById("cvDropzone");
  const esp32Panel = document.getElementById("esp32CamPanel");

  if (tabUpload) {
    tabUpload.classList.toggle("active", mode === "upload");
    tabUpload.setAttribute("aria-selected", mode === "upload" ? "true" : "false");
  }
  if (tabCamera) {
    tabCamera.classList.toggle("active", mode === "camera");
    tabCamera.setAttribute("aria-selected", mode === "camera" ? "true" : "false");
  }
  if (tabEsp32) {
    tabEsp32.classList.toggle("active", mode === "esp32");
    tabEsp32.setAttribute("aria-selected", mode === "esp32" ? "true" : "false");
  }

  if (mode === "upload") {
    stopEsp32Stream();
    if (esp32Panel) esp32Panel.style.display = "none";
    if (dropzone) dropzone.style.display = "block";
  } else if (mode === "camera") {
    stopEsp32Stream();
    if (esp32Panel) esp32Panel.style.display = "none";
    if (dropzone) dropzone.style.display = "block";
    openCameraModal();
  } else if (mode === "esp32") {
    if (dropzone) dropzone.style.display = "none";
    if (esp32Panel) esp32Panel.style.display = "block";
    pollEsp32Status();
    startEsp32Stream();
  }
}

async function pollEsp32Status() {
  const badgeDot = document.getElementById("esp32HeaderDot");
  const badgeIcon = document.getElementById("esp32HeaderIcon");
  const badgeText = document.getElementById("esp32HeaderText") || document.getElementById("esp32HeaderStatusText");
  const headerBadge = document.getElementById("esp32HeaderBadge");
  const tabBadge = document.getElementById("esp32TabBadge");
  const connBadge = document.getElementById("esp32ConnBadge");
  const connText = document.getElementById("esp32ConnText");
  const subText = document.getElementById("esp32SubText");
  const captureBtn = document.getElementById("btnEsp32Capture");

  const camChip = document.getElementById("iotCameraChip");
  const camChipDot = document.getElementById("iotCameraChipDot");
  const camChipText = document.getElementById("iotCameraChipText");

  const rssiEl = document.getElementById("esp32Rssi");
  const heapEl = document.getElementById("esp32Heap");
  const resEl = document.getElementById("esp32Res");
  const diagDevId = document.getElementById("diagDeviceId");
  const diagFw = document.getElementById("diagFirmware");
  const diagIp = document.getElementById("diagIp");
  const diagUptime = document.getElementById("diagUptime");
  const diagCaptures = document.getElementById("diagCaptures");
  const diagHeartbeat = document.getElementById("diagHeartbeat");

  try {
    const res = await fetch(`${API_BASE}/api/v1/esp32/status`, { cache: "no-store" });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);

    const data = await res.json();
    const dev = data.active_device || (data.devices && data.devices.length > 0 ? data.devices[0] : null);
    const isOnline = Boolean(dev && dev.is_online);
    esp32DeviceOnline = isOnline;

    if (isOnline && dev) {
      // Seamless auto-reconnect: if ESP32 CAM tab is active and not currently viewing an analysis result, start stream
      if (currentVisionInputMode === "esp32" && !esp32StreamActive && !currentVisionResult) {
        startEsp32Stream();
      }
      if (headerBadge) {
        headerBadge.style.display = "inline-flex";
        headerBadge.className = "server-badge esp32-header-badge online";
      }
      if (badgeDot) {
        badgeDot.className = "badge-dot badge-dot-online";
      }
      if (badgeIcon) {
        badgeIcon.textContent = "📹";
      }
      if (badgeText) badgeText.textContent = `ESP32 CAM Online · ${dev.camera_type || 'OV2640'}`;
      if (camChip) {
        camChip.style.display = "inline-flex";
        if (camChipDot) camChipDot.className = "badge-dot badge-dot-online";
        if (camChipText) camChipText.textContent = `Online (${dev.camera_type || 'OV2640'})`;
      }
      if (tabBadge) {
        tabBadge.className = "esp32-tab-pill-badge badge-online";
        tabBadge.textContent = "Online";
      }
      if (connBadge) {
        connBadge.className = "esp32-conn-badge badge-online";
      }
      if (connText) connText.textContent = "ESP32 CAM Online";
      if (subText) subText.textContent = `Field Optical Sensor · Ready (${dev.device_id})`;
      if (captureBtn) captureBtn.disabled = false;

      if (rssiEl) rssiEl.textContent = dev.rssi != null ? dev.rssi : "--";
      if (heapEl) heapEl.textContent = dev.free_heap ? Math.round(dev.free_heap / 1024) : "--";
      if (resEl) resEl.textContent = dev.frame_size || "QVGA";

      if (diagDevId) diagDevId.textContent = dev.device_id || "--";
      if (diagFw) diagFw.textContent = dev.firmware_version || "--";
      if (diagIp) diagIp.textContent = dev.ip_address || "Outbound Gateway";
      if (diagUptime) {
        const u = dev.uptime_seconds || 0;
        const mins = Math.floor(u / 60);
        const secs = u % 60;
        diagUptime.textContent = `${mins}m ${secs}s`;
      }
      if (diagCaptures) diagCaptures.textContent = dev.total_captures != null ? dev.total_captures : "0";
      if (diagHeartbeat) {
        diagHeartbeat.textContent = dev.last_heartbeat_ago_s != null ? `${dev.last_heartbeat_ago_s}s ago` : "Active";
      }
    } else {
      if (headerBadge) {
        headerBadge.style.display = "inline-flex";
        headerBadge.className = "server-badge esp32-header-badge offline";
      }
      if (badgeDot) {
        badgeDot.className = "badge-dot badge-dot-offline";
      }
      if (badgeIcon) {
        badgeIcon.textContent = "📹";
      }
      if (badgeText) badgeText.textContent = "ESP32 CAM Offline";
      if (camChip) {
        camChip.style.display = "none";
      }
      if (tabBadge) {
        tabBadge.className = "esp32-tab-pill-badge badge-offline";
        tabBadge.textContent = "Offline";
      }
      if (connBadge) {
        connBadge.className = "esp32-conn-badge badge-offline";
      }
      if (connText) connText.textContent = "ESP32 CAM Offline";
      if (subText) subText.textContent = "Field Optical Sensor · Standby";
      if (captureBtn) captureBtn.disabled = true;

      if (rssiEl) rssiEl.textContent = "--";
      if (heapEl) heapEl.textContent = "--";
      if (diagCaptures) diagCaptures.textContent = "0";
    }
  } catch (err) {
    esp32DeviceOnline = false;
    if (headerBadge) {
      headerBadge.style.display = "inline-flex";
      headerBadge.className = "server-badge esp32-header-badge offline";
    }
    if (badgeDot) {
      badgeDot.className = "badge-dot badge-dot-offline";
    }
    if (badgeIcon) {
      badgeIcon.textContent = "📹";
    }
    if (badgeText) badgeText.textContent = "ESP32 CAM Offline";
    if (camChip) {
      camChip.style.display = "none";
    }
    if (tabBadge) {
      tabBadge.className = "esp32-tab-pill-badge badge-offline";
      tabBadge.textContent = "Offline";
    }
    if (connBadge) {
      connBadge.className = "esp32-conn-badge badge-offline";
    }
    if (connText) connText.textContent = "ESP32 CAM Offline";
    if (captureBtn) captureBtn.disabled = true;
  }
}

function refreshEsp32State() {
  pollEsp32Status();
  if (currentVisionInputMode === "esp32") {
    startEsp32Stream();
  }
}

function startEsp32Stream() {
  if (esp32StreamActive) return;
  esp32StreamActive = true;
  esp32FrameCount = 0;

  const overlay = document.getElementById("esp32StreamOverlay");
  const overlayMsg = document.getElementById("esp32OverlayMessage");
  const liveImg = document.getElementById("esp32LiveImg");
  const fpsEl = document.getElementById("esp32Fps");

  if (overlay) overlay.style.display = "flex";
  if (overlayMsg) overlayMsg.textContent = "Connecting to ESP32 CAM live preview...";

  // Explicitly command backend and ESP32 hardware to enable preview streaming
  fetch(`${API_BASE}/api/v1/esp32/stream-control`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ active: true })
  }).catch(() => {});

  if (esp32FpsTimer) clearInterval(esp32FpsTimer);
  esp32FpsTimer = setInterval(() => {
    if (fpsEl) fpsEl.textContent = esp32FrameCount;
    esp32FrameCount = 0;
  }, 1000);

  esp32StreamAbortController = new AbortController();
  fetchEsp32PreviewLoop();
}

function stopEsp32Stream() {
  esp32StreamActive = false;
  if (esp32FpsTimer) {
    clearInterval(esp32FpsTimer);
    esp32FpsTimer = null;
  }
  const fpsEl = document.getElementById("esp32Fps");
  if (fpsEl) fpsEl.textContent = "--";

  if (esp32StreamAbortController) {
    esp32StreamAbortController.abort();
    esp32StreamAbortController = null;
  }

  const liveImg = document.getElementById("esp32LiveImg");
  if (liveImg) {
    if (liveImg.dataset.blobUrl) {
      URL.revokeObjectURL(liveImg.dataset.blobUrl);
      delete liveImg.dataset.blobUrl;
    }
    liveImg.style.display = "none";
    liveImg.src = "";
  }
}

let lastRenderedFrameAge = null;

async function fetchEsp32PreviewLoop() {
  if (!esp32StreamActive) return;
  if (isFetchingEsp32Frame) return;
  isFetchingEsp32Frame = true;

  const liveImg = document.getElementById("esp32LiveImg");
  const overlay = document.getElementById("esp32StreamOverlay");
  const overlayMsg = document.getElementById("esp32OverlayMessage");

  try {
    const res = await fetch(`${API_BASE}/api/v1/esp32/preview?t=${Date.now()}`, {
      cache: "no-store",
      signal: esp32StreamAbortController ? esp32StreamAbortController.signal : undefined
    });

    if (res.ok && res.status === 200) {
      const ageSec = parseFloat(res.headers.get("X-Frame-Age-Seconds") || "0");
      const blob = await res.blob();
      if (blob.size > 0 && esp32StreamActive) {
        if (ageSec <= 8.0) {
          const newUrl = URL.createObjectURL(blob);
          const oldUrl = liveImg.dataset.blobUrl;
          liveImg.src = newUrl;
          liveImg.dataset.blobUrl = newUrl;
          if (oldUrl) URL.revokeObjectURL(oldUrl);

          liveImg.style.display = "block";
          if (overlay) overlay.style.display = "none";
          if (ageSec !== lastRenderedFrameAge) {
            esp32FrameCount++;
            lastRenderedFrameAge = ageSec;
          }
        } else {
          // Frame in cache is older than 8 seconds — camera is settling or disconnected
          if (overlay) {
            overlay.style.display = "flex";
            if (overlayMsg) overlayMsg.textContent = "Live frame delayed. Reconnecting to sensor...";
          }
        }
      }
    } else {
      if (overlay) {
        overlay.style.display = "flex";
        if (overlayMsg) overlayMsg.textContent = "Waiting for live frame from ESP32 CAM...";
      }
    }
  } catch (err) {
    if (err.name !== "AbortError") {
      // transient network wait
    }
  } finally {
    isFetchingEsp32Frame = false;
    if (esp32StreamActive) {
      setTimeout(fetchEsp32PreviewLoop, 80);
    }
  }
}

async function triggerEsp32Capture() {
  const btn = document.getElementById("btnEsp32Capture");
  const btnText = document.getElementById("btnEsp32CaptureText");
  const flash = document.getElementById("esp32CaptureFlash");

  if (!btn || btn.disabled) return;

  btn.disabled = true;
  if (btnText) btnText.textContent = "Capturing OV2640 Frame...";

  if (flash) {
    flash.classList.add("flash-active");
    setTimeout(() => flash.classList.remove("flash-active"), 400);
  }

  try {
    const res = await fetch(`${API_BASE}/api/v1/esp32/capture`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ resolution: "SVGA", quality: 10 }),
    });

    if (!res.ok) {
      let errorMsg = `Capture failed (HTTP ${res.status})`;
      try {
        const errJson = await res.json();
        if (typeof errJson.detail === "string") {
          errorMsg = errJson.detail;
        } else if (errJson.detail && typeof errJson.detail.message === "string") {
          errorMsg = errJson.detail.message;
        } else if (typeof errJson.message === "string") {
          errorMsg = errJson.message;
        } else if (errJson.detail) {
          errorMsg = JSON.stringify(errJson.detail);
        }
      } catch {
        // fallback to default errorMsg
      }
      throw new Error(errorMsg);
    }

    const data = await res.json();
    if (!data.image_base64) {
      throw new Error("No image data returned from camera capture.");
    }

    // Safely strip data URL header if present before decoding
    let b64 = data.image_base64;
    if (typeof b64 === "string" && b64.includes(",")) {
      b64 = b64.split(",")[1];
    }
    b64 = (b64 || "").replace(/\s/g, "");

    // Convert base64 to File object
    const byteCharacters = atob(b64);
    const byteNumbers = new Array(byteCharacters.length);
    for (let i = 0; i < byteCharacters.length; i++) {
      byteNumbers[i] = byteCharacters.charCodeAt(i);
    }
    const byteArray = new Uint8Array(byteNumbers);
    const blob = new Blob([byteArray], { type: "image/jpeg" });

    const filename = `esp32_${data.device_id || "cam"}_${data.capture_id || Date.now()}.jpg`;
    const capturedFile = new File([blob], filename, { type: "image/jpeg", lastModified: Date.now() });

    currentEsp32CaptureId = data.capture_id;
    currentVisionImageSource = "esp32_cam";
    currentEsp32CaptureMeta = data;

    // Stop stream to conserve device RAM and network bandwidth
    stopEsp32Stream();

    // Switch view to dropzone preview workspace
    const esp32Panel = document.getElementById("esp32CamPanel");
    const dropzone = document.getElementById("cvDropzone");
    const dropzonePrompt = document.getElementById("dropzonePrompt");
    const dropzonePreviewWrap = document.getElementById("dropzonePreviewWrap");

    if (esp32Panel) esp32Panel.style.display = "none";
    if (dropzone) dropzone.style.display = "block";
    if (dropzonePrompt) dropzonePrompt.style.display = "none";
    if (dropzonePreviewWrap) dropzonePreviewWrap.style.display = "flex";

    // Show provenance badge
    const provTag = document.getElementById("cvProvenanceTag");
    if (provTag) {
      provTag.style.display = "inline-flex";
      provTag.className = "provenance-tag esp32-provenance";
      provTag.innerHTML = `📹 ESP32 CAM · OV2640 (${data.resolution || 'SVGA'}) · ${data.latency_ms || '--'}ms`;
    }

    showToast(`High-resolution frame captured from ESP32 CAM (${data.resolution || 'SVGA'}). Verifying specimen...`, "success", "Hardware Capture");

    // Automatically feed into preflight validation and analysis preparation
    await processSelectedImageFile(capturedFile, false, "esp32_cam", data.capture_id);

  } catch (err) {
    console.error("ESP32 capture error:", err);
    let msg = "Failed to capture image from ESP32 CAM.";
    if (typeof err === "string") {
      msg = err;
    } else if (err && typeof err.message === "string") {
      msg = err.message;
    } else if (err && err.detail) {
      msg = typeof err.detail === "string" ? err.detail : (err.detail.message || JSON.stringify(err.detail));
    }
    showToast(msg, "error", "Capture Error");
  } finally {
    if (btn) btn.disabled = !esp32DeviceOnline;
    if (btnText) btnText.textContent = "Capture Specimen";
  }
}

function openPreviewInModal(event) {
  if (event && typeof event.stopPropagation === "function") {
    event.stopPropagation();
  }
  const previewImg = document.getElementById("cvPreviewImg");
  if (previewImg && previewImg.src && !previewImg.src.endsWith("#") && previewImg.src !== window.location.href) {
    openImageModal(previewImg.src, "Uploaded Leaf Specimen");
  }
}

function handleCvFileSelect(event) {
  const files = event.target.files;
  if (files && files.length > 0) {
    currentVisionImageSource = "browser_upload";
    currentEsp32CaptureId = null;
    currentEsp32CaptureMeta = null;
    const provTag = document.getElementById("cvProvenanceTag");
    if (provTag) provTag.style.display = "none";
    processSelectedImageFile(files[0]);
  }
}

function clearPreviousVisionResults() {
  lastVisionResult = null;
  document.getElementById("visionSection")?.classList.remove("has-results");
  const resultContent = document.getElementById("visionResultContent");
  const advisoryContent = document.getElementById("visionAdvisoryContent");
  const triageBadge = document.getElementById("visionTriageBadge");
  const modelBadge = document.getElementById("visionModelBadge");
  const inspectionPanel = document.getElementById("visionInspectionPanel");
  const explainPanel = document.getElementById("visionExplainabilityPanel");
  const qualityPanel = document.getElementById("visionQualityPanel");
  const qualityCompact = document.getElementById("visionQualityCompact");
  const qualityDrawer = document.getElementById("visionQualityDrawer");
  const techPanel = document.getElementById("visionTechPanel");
  const progressCard = document.getElementById("analysisProgressCard");
  const canvas = document.getElementById("visionInspectionCanvas");
  const placeholder = document.getElementById("canvasEmptyPlaceholder");

  if (canvas) {
    const ctx = canvas.getContext("2d");
    if (ctx) ctx.clearRect(0, 0, canvas.width, canvas.height);
    canvas.style.display = "none";
  }
  if (placeholder) placeholder.style.display = "flex";

  if (triageBadge) triageBadge.style.display = "none";
  if (modelBadge) modelBadge.style.display = "none";
  if (inspectionPanel) inspectionPanel.style.display = "none";
  if (explainPanel) explainPanel.style.display = "none";
  if (qualityPanel) qualityPanel.style.display = "none";
  if (qualityCompact) qualityCompact.style.display = "none";
  const banner = document.getElementById("visionValidationBanner");
  if (banner) {
    banner.style.display = "none";
    banner.className = "validation-banner";
  }

  // Keep Model Architecture & Telemetry always visible
  if (techPanel) {
    techPanel.style.display = "block";
    renderTechnicalDetails({});
  }

  if (resultContent) {
    resultContent.innerHTML = `
      <div class="placeholder-state">
        <div class="placeholder-icon">🍃</div>
        <p>Specimen selected.<br><strong style="color: var(--green-bright); font-size: 14px;">Hit "Analyze Plant Health" for results</strong></p>
      </div>
    `;
  }
  if (advisoryContent) {
    advisoryContent.innerHTML = `
      <div class="placeholder-state">
        <div class="placeholder-icon">📋</div>
        <p>Actionable agronomic advisory, chemical treatment protocols,<br>and cultural practices will be computed upon leaf analysis</p>
      </div>
    `;
  }
}

function handleRecaptureButtonClick(event) {
  if (event && typeof event.stopPropagation === "function") {
    event.stopPropagation();
  }
  if (currentVisionInputMode === "esp32" || currentVisionImageSource === "esp32_cam") {
    resetVisionToNewSpecimen(event);
  } else {
    openCameraModal(event);
  }
}

async function setEsp32CameraConfig(config) {
  try {
    const res = await fetch(`${API_BASE}/api/v1/esp32/camera-config`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(config)
    });
    const data = await res.json();
    if (data.status === "success") {
      showToast("Camera sensor parameters updated.", "success", "Sensor Calibrated");
    }
  } catch (err) {
    console.warn("Failed to set camera config:", err);
  }
}

function resetVisionToNewSpecimen(event) {
  if (event && typeof event.stopPropagation === "function") {
    event.stopPropagation();
  }
  activeVisionRequestId = null;
  if (isVisionAnalyzing && visionAbortController) {
    visionAbortController.abort();
    visionAbortController = null;
  }
  isVisionAnalyzing = false;
  selectedVisionFile = null;
  lastVisionResult = null;
  originalVisionImageObj = null;

  if (currentPreviewObjectUrl) {
    URL.revokeObjectURL(currentPreviewObjectUrl);
    currentPreviewObjectUrl = null;
  }

  const fileInput = document.getElementById("cvFileInput");
  if (fileInput) fileInput.value = "";

  const previewImg = document.getElementById("cvPreviewImg");
  const dropzonePrompt = document.getElementById("dropzonePrompt");
  const previewWrap = document.getElementById("dropzonePreviewWrap");
  const filenameEl = document.getElementById("cvPreviewFilename");
  const dropzone = document.getElementById("cvDropzone");
  const esp32Panel = document.getElementById("esp32CamPanel");

  if (previewImg) previewImg.src = "";
  if (filenameEl) filenameEl.textContent = "sample.jpg";
  if (previewWrap) previewWrap.style.display = "none";

  clearPreviousVisionResults();
  setVisionUIState(VisionUIState.EMPTY);

  const wasEsp32 = (currentVisionInputMode === "esp32" || currentVisionImageSource === "esp32_cam");
  currentEsp32CaptureId = null;
  currentEsp32CaptureMeta = null;

  const provTag = document.getElementById("cvProvenanceTag");
  if (provTag) {
    provTag.style.display = "none";
    provTag.innerHTML = "";
  }

  if (wasEsp32) {
    currentVisionInputMode = "esp32";
    currentVisionImageSource = "esp32_cam";
    if (dropzone) dropzone.style.display = "none";
    if (esp32Panel) esp32Panel.style.display = "block";
    startEsp32Stream();
    showToast("Cleared specimen. Resumed live ESP32-CAM preview.", "info", "Live Camera Ready");
  } else {
    currentVisionImageSource = "browser_upload";
    if (dropzone) dropzone.style.display = "block";
    if (dropzonePrompt) dropzonePrompt.style.display = "flex";
    if (esp32Panel) esp32Panel.style.display = "none";
    showToast("Foliar workspace reset. Ready for new specimen.", "info", "Workspace Reset");
  }
}

function toggleQualityDrawer() {
  const drawer = document.getElementById("visionQualityDrawer");
  const toggleIcon = document.getElementById("qualityPillToggle");
  if (!drawer) return;
  if (drawer.style.display === "none" || drawer.style.display === "") {
    drawer.style.display = "block";
    if (toggleIcon) toggleIcon.textContent = "▲";
  } else {
    drawer.style.display = "none";
    if (toggleIcon) toggleIcon.textContent = "▼";
  }
}

function renderImageQuality(quality) {
  const compactPill = document.getElementById("visionQualityCompact");
  const pillIcon = document.getElementById("qualityPillIcon");
  const pillText = document.getElementById("qualityPillText");
  const drawer = document.getElementById("visionQualityDrawer");
  const drawerContent = document.getElementById("visionQualityDrawerContent");

  if (!compactPill) return;

  if (!quality) {
    compactPill.style.display = "none";
    if (drawer) drawer.style.display = "none";
    return;
  }

  compactPill.style.display = "inline-flex";

  const qLevel = (quality.quality_level || quality.quality_grade || "Good").toUpperCase();
  const isUsable = quality.is_usable !== false;

  // Clean pill styling and text
  compactPill.className = "quality-compact-pill";
  if (!isUsable) {
    compactPill.classList.add("quality-pill-critical");
    if (pillIcon) pillIcon.textContent = "⚠️";
    if (pillText) pillText.textContent = "Quality: Degraded";
  } else if (qLevel === "ACCEPTABLE" || qLevel === "DEGRADED" || (quality.warnings && quality.warnings.length > 0)) {
    compactPill.classList.add("quality-pill-warning");
    if (pillIcon) pillIcon.textContent = "ℹ️";
    if (pillText) pillText.textContent = "Quality: Acceptable";
  } else {
    compactPill.classList.add("quality-pill-optimal");
    if (pillIcon) pillIcon.textContent = "🛡️";
    if (pillText) pillText.textContent = "Quality: Optimal";
  }

  const blurVal = quality.blur_score != null ? Number(quality.blur_score).toFixed(1) : "--";
  const illumVal = (quality.brightness_mean != null ? Number(quality.brightness_mean) : (quality.illumination_score != null ? Number(quality.illumination_score) : null));
  const illumStr = illumVal != null ? `${illumVal.toFixed(1)} / 255` : "--";
  const foliarVal = quality.greenness_ratio != null 
    ? `${(Number(quality.greenness_ratio) * 100).toFixed(1)}%`
    : (quality.foliar_presence_ratio != null ? `${(Number(quality.foliar_presence_ratio) * 100).toFixed(1)}%` : "--");

  const summary = quality.summary_text || (isUsable 
    ? "Specimen image satisfies quality thresholds for diagnostic vision inference."
    : "Image quality is suboptimal; consider recapturing under brighter, diffused illumination.");

  const warnings = quality.warnings || quality.quality_issues || [];
  const warningHtml = warnings.length > 0
    ? `<div style="font-size: 0.74rem; color: #ffd166; margin-top: 4px;">
         <strong>Notices:</strong> ${warnings.map(w => `<span>• ${escapeHtml(w)}</span>`).join(" ")}
       </div>`
    : "";

  const adviceClass = isUsable 
    ? (qLevel === "GOOD" || qLevel === "OPTIMAL" ? "quality-advice-optimal" : "quality-advice-warning")
    : "quality-advice-critical";

  if (drawerContent) {
    drawerContent.innerHTML = `
      <div class="quality-drawer-grid">
        <div class="quality-drawer-metric">
          <span class="quality-drawer-metric-label">📐 Blur Variance</span>
          <span class="quality-drawer-metric-val">${blurVal}</span>
        </div>
        <div class="quality-drawer-metric">
          <span class="quality-drawer-metric-label">☀️ Exposure / Mean</span>
          <span class="quality-drawer-metric-val">${illumStr}</span>
        </div>
        <div class="quality-drawer-metric">
          <span class="quality-drawer-metric-label">🍃 Foliar Coverage</span>
          <span class="quality-drawer-metric-val">${foliarVal}</span>
        </div>
        <div class="quality-drawer-metric">
          <span class="quality-drawer-metric-label">🛡️ Preflight Status</span>
          <span class="quality-drawer-metric-val" style="color: ${isUsable ? 'var(--green-bright)' : 'var(--red)'};">${isUsable ? 'Pass' : 'Suboptimal'}</span>
        </div>
      </div>
      <div class="quality-drawer-guidance ${adviceClass}">
        <strong>Preflight Assessment:</strong> ${escapeHtml(summary)}
        ${warningHtml}
        ${quality.recapture_guidance ? `<div style="margin-top: 4px; font-size: 0.72rem; opacity: 0.9;"><strong>Guidance:</strong> ${escapeHtml(quality.recapture_guidance)}</div>` : ''}
      </div>
    `;
  }
}

function renderValidationBanner(validation) {
  const banner = document.getElementById("visionValidationBanner");
  const iconEl = document.getElementById("validationBannerIcon");
  const titleEl = document.getElementById("validationBannerTitle");
  const msgEl = document.getElementById("validationBannerMsg");

  if (!banner || !validation) return;

  const status = validation.validation_status || "";
  const reason = validation.validation_reason || "The uploaded image is not suitable for crop health analysis.";

  if (status === "INVALID_SCREENSHOT_OR_DOCUMENT") {
    banner.className = "validation-banner validation-banner-rejected";
    if (iconEl) iconEl.textContent = "🖥️";
    if (titleEl) titleEl.textContent = "Screenshot or Document Detected";
    if (msgEl) msgEl.textContent = reason;
    banner.style.display = "flex";
  } else if (status === "INVALID_NON_PLANT_IMAGE") {
    banner.className = "validation-banner validation-banner-rejected";
    if (iconEl) iconEl.textContent = "🚫";
    if (titleEl) titleEl.textContent = "Non-Plant Specimen Rejected";
    if (msgEl) msgEl.textContent = reason;
    banner.style.display = "flex";
  } else if (status === "LOW_QUALITY_IMAGE") {
    banner.className = "validation-banner validation-banner-uncertain";
    if (iconEl) iconEl.textContent = "⚠️";
    if (titleEl) titleEl.textContent = "Image Quality Insufficient";
    if (msgEl) msgEl.textContent = reason;
    banner.style.display = "flex";
  } else if (status === "VALIDATION_UNCERTAIN" || status === "LOW_QUALITY_OR_UNCERTAIN_IMAGE") {
    banner.className = "validation-banner validation-banner-uncertain";
    if (iconEl) iconEl.textContent = "⚠️";
    if (titleEl) titleEl.textContent = "Plant Presence Uncertain";
    if (msgEl) msgEl.textContent = reason;
    banner.style.display = "flex";
  } else if (status === "VALID_PLANT_IMAGE") {
    if (validation.image_quality && validation.image_quality.includes("Degraded")) {
      banner.className = "validation-banner validation-banner-uncertain";
      if (iconEl) iconEl.textContent = "📹";
      if (titleEl) titleEl.textContent = "Camera Quality Degraded · ESP32-CAM OV2640";
      if (msgEl) msgEl.textContent = reason || "Sensor compression or mild optical blur tolerated under authenticated ESP32-CAM policy. Specimen confirmed as plant foliage.";
      banner.style.display = "flex";
    } else {
      banner.className = "validation-banner validation-banner-valid";
      if (iconEl) iconEl.textContent = "🌿";
      if (titleEl) titleEl.textContent = "Valid Crop Specimen Verified";
      if (msgEl) msgEl.textContent = reason;
      banner.style.display = "none";
    }
  } else {
    banner.style.display = "none";
  }
}

function renderInvalidImagePanel(validation, isScreenshot = false, isNonPlant = false) {
  const resultContent = document.getElementById("visionResultContent");
  if (!resultContent) return;

  const isStrictRejection = isScreenshot || isNonPlant;
  const headerIcon = isScreenshot ? '🖥️' : (isNonPlant ? '🚫' : '⚠️');
  let headerTitle = 'Image Quality Insufficient';
  if (isScreenshot) headerTitle = 'Invalid Image · Screenshot or Document Detected';
  else if (isNonPlant) headerTitle = 'Invalid Image · Non-Plant Specimen Detected';
  else headerTitle = 'Invalid Image · Quality or Botanical Presence Insufficient';

  const userMessage = isScreenshot
    ? "Invalid image. This appears to be a screenshot or document rather than a plant photograph. Please upload the original photograph of the plant leaf."
    : (isNonPlant
      ? "Invalid image. This appears to be an unrelated object or non-plant photograph rather than a plant leaf. Please upload a genuine plant leaf photograph."
      : (validation.validation_reason || "Specimen quality or botanical presence is insufficient for crop pathology diagnosis. Please upload a clear photograph of the plant leaf under indirect daylight."));

  resultContent.innerHTML = `
    <div class="invalid-image-panel" style="padding: 24px; border: 1px solid ${isStrictRejection ? 'rgba(239, 35, 60, 0.45)' : 'rgba(255, 183, 3, 0.45)'}; border-radius: 10px; background: ${isStrictRejection ? 'rgba(239, 35, 60, 0.08)' : 'rgba(255, 183, 3, 0.08)'}; box-shadow: 0 4px 16px rgba(0, 0, 0, 0.25);">
      <div style="display: flex; align-items: center; gap: 10px; color: ${isStrictRejection ? '#ff8a9a' : '#ffd166'}; font-weight: 700; font-size: 1.05rem;">
        <span style="font-size: 1.4rem;">${headerIcon}</span>
        <span>${headerTitle}</span>
      </div>
      <p style="margin-top: 14px; font-size: 0.92rem; line-height: 1.6; color: var(--text-primary);">
        ${escapeHTML(userMessage)}
      </p>
      <div style="margin-top: 16px; padding: 12px 14px; background: rgba(0, 0, 0, 0.35); border-radius: 8px; font-size: 0.82rem; color: var(--text-secondary); line-height: 1.5;">
        <div style="font-weight: 600; margin-bottom: 4px; color: var(--text-muted);">Validation Signals:</div>
        <div>Blur Variance: <strong>${validation.telemetry?.blur_variance ?? '--'}</strong> · Foliar Coverage: <strong>${validation.telemetry?.foliar_presence_ratio != null ? (validation.telemetry.foliar_presence_ratio * 100).toFixed(1) + '%' : '--'}</strong> · Plant Presence: <strong>${validation.plant_presence ? 'Yes' : 'No'}</strong> · Leaf Detected: <strong>${validation.leaf_presence ? 'Yes' : 'No'}</strong></div>
      </div>
      <div style="margin-top: 18px; display: flex; gap: 12px; align-items: center; flex-wrap: wrap;">
        <button type="button" class="change-img-btn btn-reset-specimen" onclick="triggerCvFileInput(event)" style="background: var(--green-primary); color: #000; font-weight: 600; border: none; padding: 8px 18px; border-radius: 6px; cursor: pointer;">
          📁 Choose Another Image
        </button>
        <button type="button" class="btn-override-analyze" onclick="runVisionPrediction(true)" style="background: rgba(46, 196, 182, 0.22); color: #2ec4b6; border: 1px solid #2ec4b6; font-weight: 600; padding: 8px 18px; border-radius: 6px; cursor: pointer; transition: all 0.2s ease;">
          🔬 Analyze Specimen Anyway
        </button>
      </div>
    </div>
  `;

  const advisoryContent = document.getElementById("visionAdvisoryContent");
  if (advisoryContent) {
    advisoryContent.innerHTML = `
      <div class="placeholder-state">
        <div class="placeholder-icon">📋</div>
        <p>Actionable agronomic advisory, chemical treatment protocols, and cultural practices will be computed upon leaf analysis</p>
      </div>
    `;
  }
  document.getElementById("visionSection")?.classList.add("has-results");
}

async function processSelectedImageFile(file, isSample = false, source = null, esp32CaptureId = null) {
  if (!file || !file.type.startsWith("image/")) {
    showErrorNotification("Please upload a valid image file (JPEG, PNG, or WebP).");
    return;
  }

  if (source) currentVisionImageSource = source;
  if (esp32CaptureId) currentEsp32CaptureId = esp32CaptureId;

  // Stale response protection: assign unique request ID to this image instance
  const thisImageId = "img_" + Date.now() + "_" + Math.random().toString(36).substring(2, 9);
  activeVisionRequestId = thisImageId;

  // Cancel any in-flight inference or validation requests
  if (visionAbortController) {
    visionAbortController.abort();
    visionAbortController = null;
  }
  if (validationAbortController) {
    validationAbortController.abort();
    validationAbortController = null;
  }
  isVisionAnalyzing = false;

  // Revoke previous object URL to prevent memory leaks
  if (currentPreviewObjectUrl) {
    URL.revokeObjectURL(currentPreviewObjectUrl);
    currentPreviewObjectUrl = null;
  }

  // Clear previous results immediately from DOM & state
  clearPreviousVisionResults();
  lastVisionResult = null;
  const inspectionPanel = document.getElementById("visionInspectionPanel");
  if (inspectionPanel) inspectionPanel.style.display = "none";
  const explainPanel = document.getElementById("visionExplainabilityPanel");
  if (explainPanel) explainPanel.style.display = "none";

  selectedVisionFile = file;
  currentPreviewObjectUrl = URL.createObjectURL(file);

  const previewImg = document.getElementById("cvPreviewImg");
  const filenameEl = document.getElementById("cvPreviewFilename");
  const dropzonePrompt = document.getElementById("dropzonePrompt");
  const previewWrap = document.getElementById("dropzonePreviewWrap");

  if (previewImg) previewImg.src = currentPreviewObjectUrl;
  if (filenameEl) {
    filenameEl.textContent = file.name;
    filenameEl.title = `${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
  }
  if (dropzonePrompt) dropzonePrompt.style.display = "none";
  if (previewWrap) previewWrap.style.display = "flex";

  // Preload image object for canvas lesion overlay
  originalVisionImageObj = new Image();
  originalVisionImageObj.src = currentPreviewObjectUrl;

  // Enter mandatory VALIDATING state: Analyze button disabled!
  setVisionUIState(VisionUIState.VALIDATING, "Checking specimen domain & quality");

  validationAbortController = new AbortController();
  const formData = new FormData();
  formData.append("file", file, file.name);
  formData.append("request_id", thisImageId);
  formData.append("source", currentVisionImageSource || "browser_upload");
  if (currentEsp32CaptureId) {
    formData.append("esp32_capture_id", currentEsp32CaptureId);
  }

  // Safety watchdog timeout (9.0s max): If validation takes long, prompt user to verify specimen
  const validationTimeout = setTimeout(() => {
    if (activeVisionRequestId === thisImageId && currentVisionState === VisionUIState.VALIDATING) {
      console.warn("Validation timeout reached; prompting specimen verification.");
      setVisionUIState(VisionUIState.VALIDATION_UNCERTAIN, "Specimen check timed out. Ensure photograph is a clear plant leaf.");
      renderValidationBanner({
        validation_status: "VALIDATION_UNCERTAIN",
        validation_reason: "Automated botanical validation timed out. Please ensure this image contains a clear plant leaf.",
        is_inference_allowed: true
      });
      scrollAndHighlightAnalyzeButton();
    }
  }, 9000);

  try {
    const valRes = await fetch(`${API_BASE}/validate/vision`, {
      method: "POST",
      body: formData,
      signal: validationAbortController.signal
    });

    if (!valRes.ok) {
      throw new Error(`Validation server returned HTTP ${valRes.status}`);
    }

    const valData = await valRes.json();

    // Stale response protection
    if (activeVisionRequestId !== thisImageId) {
      return;
    }

    const isRejected = Boolean(
      valData.status === "rejected" ||
      valData.is_inference_allowed === false ||
      valData.inference_allowed === false
    );

    if (isRejected) {
      // CLEAR all diagnosis components
      clearPreviousVisionResults();
      lastVisionResult = null;

      let rejectState = VisionUIState.INVALID_NON_PLANT_IMAGE;
      let isScreenshot = false;
      let isNonPlant = false;

      if (valData.validation_status === "INVALID_SCREENSHOT_OR_DOCUMENT") {
        rejectState = VisionUIState.INVALID_SCREENSHOT_OR_DOCUMENT;
        isScreenshot = true;
      } else if (valData.validation_status === "INVALID_NON_PLANT_IMAGE") {
        rejectState = VisionUIState.INVALID_NON_PLANT_IMAGE;
        isNonPlant = true;
      } else if (valData.validation_status === "LOW_QUALITY_IMAGE") {
        rejectState = VisionUIState.LOW_QUALITY_IMAGE;
      } else {
        rejectState = VisionUIState.VALIDATION_UNCERTAIN;
      }

      setVisionUIState(rejectState, valData.validation_reason);
      renderValidationBanner(valData);
      renderInvalidImagePanel(valData, isScreenshot, isNonPlant);

      showToast(
        valData.validation_reason || "Uploaded image is not suitable for crop health analysis.",
        "warning",
        isScreenshot ? "Screenshot Rejected" : (isNonPlant ? "Specimen Rejected" : "Quality Insufficient")
      );
      return;
    }

    // Image verified: Valid plant foliage photograph
    setVisionUIState(VisionUIState.VALID_PLANT_IMAGE, "Verified Foliar Specimen");
    renderValidationBanner(valData);

    // Auto-scroll to highlight Analyze button for user
    requestAnimationFrame(() => {
      setTimeout(() => {
        scrollAndHighlightAnalyzeButton();
      }, 100);
    });

    // If specimen was selected via quick scenario button, auto-trigger analysis
    if (isSample) {
      setTimeout(() => {
        if (activeVisionRequestId === thisImageId && currentVisionState === VisionUIState.VALID_PLANT_IMAGE) {
          runVisionPrediction();
        }
      }, 250);
    }
  } catch (err) {
    if (err.name === "AbortError") return;
    if (activeVisionRequestId !== thisImageId) return;

    console.warn("Validation service notice:", err);
    setVisionUIState(VisionUIState.VALIDATION_UNCERTAIN, "Specimen check incomplete. Ensure image is a clear plant leaf.");
    renderValidationBanner({
      validation_status: "VALIDATION_UNCERTAIN",
      validation_reason: "Validation check encountered an issue. Please ensure your specimen is an authentic plant leaf.",
      is_inference_allowed: false
    });
  } finally {
    clearTimeout(validationTimeout);
  }
}

// ── COMPUTER VISION: QUICK SCENARIOS ──────────────────────────────────────────
async function loadSampleLeaf(sampleKey) {
  const sample = SAMPLE_LEAF_MAP[sampleKey];
  if (!sample) return;

  setVisionUIState(VisionUIState.VALIDATING, `Loading ${sample.label}...`);

  try {
    const res = await fetch(sample.url);
    if (!res.ok) throw new Error(`Could not load specimen from ${sample.url}`);
    const blob = await res.blob();
    const file = new File([blob], sample.filename, { type: "image/jpeg" });
    await processSelectedImageFile(file, true);
  } catch (err) {
    showErrorNotification(`Specimen error: ${err.message}`);
    setVisionUIState(VisionUIState.ERROR, err.message);
  }
}

// ── COMPUTER VISION: PREDICTION & 3-TIER CASCADE ──────────────────────────────
let currentVisionModelTier = "server";

function selectModelTier(tier) {
  currentVisionModelTier = tier;
  const btnServer = document.getElementById("btnTierServer");
  const btnEdge = document.getElementById("btnTierEdge");
  const btnEnsemble = document.getElementById("btnTierEnsemble");
  const tag = document.getElementById("activeModelTag");

  if (btnServer) btnServer.classList.toggle("active", tier === "server");
  if (btnEdge) btnEdge.classList.toggle("active", tier === "edge");
  if (btnEnsemble) btnEnsemble.classList.toggle("active", tier === "ensemble");

  if (tag) {
    if (tier === "server") tag.textContent = "Server EfficientNetV2-S (256×256)";
    else if (tier === "edge") tag.textContent = "Edge MobileNetV2 (224×224)";
    else if (tier === "ensemble") tag.textContent = "Ensemble Consensus (Server + Edge)";
  }
}

function renderCropSkeletonLoading() {
  const recContent = document.getElementById("recommendationContent");
  const disContent = document.getElementById("diseaseContent");
  const section = document.getElementById("cropSection");
  if (section) section.classList.add("has-results");

  if (recContent) {
    recContent.innerHTML = `
      <div class="crop-skeleton-wrap" aria-busy="true" aria-label="Calculating crop suitability">
        <div style="display: flex; align-items: center; gap: 14px;">
          <div class="skeleton-pulse skeleton-circle" style="width: 52px; height: 52px; flex-shrink: 0;"></div>
          <div style="flex: 1; display: flex; flex-direction: column; gap: 8px;">
            <div class="skeleton-pulse" style="height: 28px; width: 60%;"></div>
            <div class="skeleton-pulse" style="height: 16px; width: 35%;"></div>
          </div>
        </div>
        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin-top: 8px;">
          <div class="skeleton-pulse skeleton-badge" style="height: 52px;"></div>
          <div class="skeleton-pulse skeleton-badge" style="height: 52px;"></div>
          <div class="skeleton-pulse skeleton-badge" style="height: 52px;"></div>
        </div>
      </div>
    `;
  }

  if (disContent) {
    disContent.innerHTML = `
      <div class="crop-skeleton-wrap" aria-busy="true" aria-label="Evaluating agro-climatic pathogen risk">
        <div class="skeleton-pulse skeleton-badge" style="height: 22px; width: 45%;"></div>
        <div class="skeleton-pulse" style="height: 48px; width: 100%;"></div>
        <div class="skeleton-pulse" style="height: 48px; width: 100%;"></div>
      </div>
    `;
  }
}

function renderVisionSkeletonLoading() {
  const resultContent = document.getElementById("visionResultContent");
  const advisoryContent = document.getElementById("visionAdvisoryContent");
  const section = document.getElementById("visionSection");
  if (section) section.classList.add("has-results");

  if (resultContent) {
    resultContent.innerHTML = `
      <div class="vision-skeleton-wrap" aria-busy="true" aria-label="Analyzing foliar specimen">
        <div style="display: flex; align-items: center; justify-content: space-between; gap: 12px;">
          <div class="skeleton-pulse skeleton-badge" style="height: 26px; width: 42%;"></div>
          <div class="skeleton-pulse skeleton-badge" style="height: 22px; width: 28%;"></div>
        </div>
        <div class="skeleton-pulse" style="height: 36px; width: 75%; margin-top: 4px;"></div>
        <div class="skeleton-pulse" style="height: 18px; width: 45%;"></div>
        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin-top: 8px;">
          <div class="skeleton-pulse skeleton-badge" style="height: 64px;"></div>
          <div class="skeleton-pulse skeleton-badge" style="height: 64px;"></div>
          <div class="skeleton-pulse skeleton-badge" style="height: 64px;"></div>
        </div>
        <div class="skeleton-pulse" style="height: 54px; width: 100%; margin-top: 6px;"></div>
      </div>
    `;
  }

  if (advisoryContent) {
    advisoryContent.innerHTML = `
      <div class="advisory-skeleton-wrap" aria-busy="true" aria-label="Synthesizing agronomic protocol">
        <div class="skeleton-pulse skeleton-badge" style="height: 24px; width: 48%;"></div>
        <div style="display: flex; gap: 10px; align-items: flex-start; margin-top: 6px;">
          <div class="skeleton-pulse skeleton-circle" style="width: 28px; height: 28px; flex-shrink: 0;"></div>
          <div style="flex: 1; display: flex; flex-direction: column; gap: 6px;">
            <div class="skeleton-pulse" style="height: 16px; width: 90%;"></div>
            <div class="skeleton-pulse" style="height: 14px; width: 65%;"></div>
          </div>
        </div>
        <div style="display: flex; gap: 10px; align-items: flex-start; margin-top: 4px;">
          <div class="skeleton-pulse skeleton-circle" style="width: 28px; height: 28px; flex-shrink: 0;"></div>
          <div style="flex: 1; display: flex; flex-direction: column; gap: 6px;">
            <div class="skeleton-pulse" style="height: 16px; width: 80%;"></div>
            <div class="skeleton-pulse" style="height: 14px; width: 50%;"></div>
          </div>
        </div>
        <div class="skeleton-pulse skeleton-badge" style="height: 42px; width: 100%; margin-top: 4px;"></div>
      </div>
    `;
  }
}

async function runVisionPrediction(overrideForce = false, includeExplainability = true) {
  if (!selectedVisionFile) {
    showErrorNotification("Please select or upload a leaf photograph first.");
    return;
  }

  // Mandatory blocking gate: prevent analysis if current state is not verified
  // When overrideForce is true, bypass the reject blocking gate completely!
  const isBlocked = !overrideForce && [
    VisionUIState.VALIDATING,
    VisionUIState.EMPTY,
    VisionUIState.IMAGE_SELECTED
  ].includes(currentVisionState);

  if (isBlocked) {
    showToast("Preflight validation is still running. Click 'Analyze Specimen Anyway' to proceed immediately.", "warning", "Validation In Progress");
    return;
  }

  // Prevent duplicate concurrent requests
  if (isVisionAnalyzing) {
    return;
  }

  isVisionAnalyzing = true;
  setVisionUIState(VisionUIState.ANALYZING, "Preparing image");

  // Cancel prior request if any
  if (visionAbortController) {
    visionAbortController.abort();
  }
  visionAbortController = new AbortController();

  // Activate progressive non-blocking progress card
  const progressCard = document.getElementById("analysisProgressCard");
  const progressTitle = document.getElementById("analysisProgressTitle");
  const progressStage = document.getElementById("analysisProgressStage");
  const chipPreflight = document.getElementById("chip-step-preflight");
  const chipClassify = document.getElementById("chip-step-classify");
  const chipDetect = document.getElementById("chip-step-detect");
  const chipSegment = document.getElementById("chip-step-segment");
  const chipExplain = document.getElementById("chip-step-explain");

  if (progressCard) {
    progressCard.style.display = "flex";
    if (progressTitle) progressTitle.textContent = "Running Neural Inference Cascade...";
    if (progressStage) progressStage.textContent = "1. Evaluating foliar image quality...";
    if (chipPreflight) chipPreflight.className = "step-chip active";
    if (chipClassify) chipClassify.className = "step-chip";
    if (chipDetect) chipDetect.className = "step-chip";
    if (chipSegment) chipSegment.className = "step-chip";
    if (chipExplain) chipExplain.className = "step-chip";
  }

  // Progressive stage timer to communicate authentic pipeline stages
  let progressStep = 1;
  const progressInterval = setInterval(() => {
    if (!isVisionAnalyzing) {
      clearInterval(progressInterval);
      return;
    }
    progressStep++;
    if (progressStep === 2) {
      if (progressStage) progressStage.textContent = "2. Classifying plant pathology (EfficientNetV2-S)...";
      if (chipPreflight) chipPreflight.className = "step-chip done";
      if (chipClassify) chipClassify.className = "step-chip active";
    } else if (progressStep === 3) {
      if (progressStage) progressStage.textContent = "3. Localizing foliage & lesion foci (YOLO26)...";
      if (chipClassify) chipClassify.className = "step-chip done";
      if (chipDetect) chipDetect.className = "step-chip active";
    } else if (progressStep === 4) {
      if (progressStage) progressStage.textContent = "4. Segmenting foliar necrosis (Mobile-UNet)...";
      if (chipDetect) chipDetect.className = "step-chip done";
      if (chipSegment) chipSegment.className = "step-chip active";
    } else if (progressStep >= 5) {
      if (progressStage) progressStage.textContent = "5. Computing Grad-CAM saliency explainability...";
      if (chipSegment) chipSegment.className = "step-chip done";
      if (chipExplain) chipExplain.className = "step-chip active";
    }
  }, 450);

  // Render smooth skeleton loading placeholders in results area without blocking left preview!
  renderVisionSkeletonLoading();

  const requestId = "req_" + Date.now() + "_" + Math.random().toString(36).substring(2, 9);
  activeVisionRequestId = requestId;

  const formData = new FormData();
  formData.append("file", selectedVisionFile);
  formData.append("model_tier", currentVisionModelTier || "server");
  formData.append("include_explainability", includeExplainability ? "true" : "false");
  formData.append("request_id", requestId);
  formData.append("source", currentVisionImageSource || "browser_upload");
  if (currentEsp32CaptureId) {
    formData.append("esp32_capture_id", currentEsp32CaptureId);
  }
  if (overrideForce) {
    formData.append("force_inference", "true");
  }

  try {
    const res = await fetch(`${API_BASE}/predict/vision`, {
      method: "POST",
      body: formData,
      signal: visionAbortController.signal,
    });

    clearInterval(progressInterval);

    if (!res.ok) {
      const errData = await res.json().catch(() => ({}));
      const msg = errData.message || errData.detail || `Server returned HTTP ${res.status}`;
      const hint = errData.recovery_hint ? ` (${errData.recovery_hint})` : "";
      throw new Error(`${msg}${hint}`);
    }

    const data = await res.json();
    if (data.request_id && activeVisionRequestId && data.request_id !== activeVisionRequestId) {
      console.warn("Discarding stale vision inference response:", data.request_id, "active:", activeVisionRequestId);
      return;
    }

    // Check if pre-inference domain validation rejected the image
    const validation = data.image_validation;
    const isRejected = Boolean(
      data.status === "rejected" ||
      !data.diagnosis ||
      (validation && !validation.is_inference_allowed) ||
      (validation && validation.inference_allowed === false) ||
      (validation && validation.validation_status && validation.validation_status !== "VALID_PLANT_IMAGE") ||
      data.diagnosis?.predicted_class === "N/A" ||
      data.diagnosis?.condition_type === "invalid_input"
    );

    if (isRejected) {
      // CLEAR ALL DIAGNOSTIC RESULTS
      clearPreviousVisionResults();
      lastVisionResult = null;

      const fallbackValidation = validation || {
        validation_status: "INVALID_SCREENSHOT_OR_DOCUMENT",
        validation_reason: "Invalid image. This appears to be a document or screenshot rather than a plant photograph. Please upload the original image of the plant leaf.",
        is_inference_allowed: false,
        telemetry: data.image_quality || {}
      };

      renderValidationBanner(fallbackValidation);

      const status = fallbackValidation.validation_status;
      const isScreenshot = status === "INVALID_SCREENSHOT_OR_DOCUMENT";
      const isNonPlant = status === "INVALID_NON_PLANT_IMAGE";
      const isLowQuality = status === "LOW_QUALITY_IMAGE";
      const isUncertain = status === "VALIDATION_UNCERTAIN" || status === "LOW_QUALITY_OR_UNCERTAIN_IMAGE";

      let nextState = VisionUIState.LOW_QUALITY_IMAGE;
      if (isScreenshot) {
        nextState = VisionUIState.INVALID_SCREENSHOT_OR_DOCUMENT;
      } else if (isNonPlant) {
        nextState = VisionUIState.INVALID_NON_PLANT_IMAGE;
      } else if (isUncertain) {
        nextState = VisionUIState.VALIDATION_UNCERTAIN;
      }

      setVisionUIState(
        nextState,
        fallbackValidation.validation_reason
      );

      renderInvalidImagePanel(fallbackValidation, isScreenshot, isNonPlant);

      showToast(
        fallbackValidation.validation_reason || "Uploaded image is not suitable for crop health analysis.",
        "warning",
        isScreenshot ? "Screenshot Rejected" : (isNonPlant ? "Specimen Rejected" : "Quality Insufficient")
      );
      return;
    }

    // Valid plant image: hide validation banner and proceed normally
    renderValidationBanner({ validation_status: "VALID_PLANT_IMAGE", is_inference_allowed: true });
    lastVisionResult = data;
    document.getElementById("visionSection")?.classList.add("has-results");

    // Render image quality preflight into compact pill / drawer
    renderImageQuality(data.image_quality);

    const isLowConf = Boolean(data.diagnosis?.is_low_confidence || (data.diagnosis?.confidence_pct != null && data.diagnosis.confidence_pct < 50));
    setVisionUIState(isLowConf ? VisionUIState.LOW_CONFIDENCE : VisionUIState.COMPLETE);

    renderVisionDiagnosis(data.diagnosis || {});
    renderVisionAdvisory(data.advisory || {}, data.diagnosis || {});
    renderSpatialTelemetry(data);
    renderTechnicalDetails(data);

    const inspectionPanel = document.getElementById("visionInspectionPanel");
    if (inspectionPanel) {
      inspectionPanel.style.display = "block";
    }

    if (data.explainability && data.explainability.stages) {
      renderExplainabilityPipeline(data.explainability, data);
      const explainPanel = document.getElementById("visionExplainabilityPanel");
      if (explainPanel) {
        explainPanel.style.display = "block";
      }
    }

    // Contextual feedback toast
    if (data.image_quality && data.image_quality.is_usable === false) {
      showToast(data.image_quality.recommendation || "Quality issues detected in specimen photograph.", "warning", "Quality Preflight Warning");
    } else if (isLowConf) {
      showToast(`Tentative match: ${data.diagnosis?.disease_common_name || "Leaf"} (${data.diagnosis?.confidence_pct || 0}%). Verification advised.`, "warning", "Low Confidence Advisory");
    } else {
      showToast(`Diagnosed ${data.diagnosis?.disease_common_name || "Leaf"} (${data.diagnosis?.confidence_pct || 0}%)`, "success", "Pathology Analysis Complete");
    }
  } catch (err) {
    clearInterval(progressInterval);
    if (err.name === "AbortError") {
      console.log("Previous vision inference request aborted.");
      return;
    }
    setVisionUIState(VisionUIState.ERROR, err.message);
    if (resultContent) {
      resultContent.innerHTML = `
        <div style="padding: 20px; border: 1px solid rgba(239, 35, 60, 0.4); border-radius: 8px; background: rgba(239, 35, 60, 0.08);">
          <div style="display: flex; align-items: center; gap: 8px; color: #ff8a9a; font-weight: 700;">
            <span>⚠️</span><span>Diagnostic Analysis Interrupted</span>
          </div>
          <p style="margin-top: 8px; font-size: 0.85rem; color: var(--text-primary);">${escapeHTML(err.message)}</p>
          <div style="margin-top: 12px; display: flex; gap: 10px;">
            <button type="button" class="change-img-btn" onclick="runVisionPrediction()">Retry Analysis</button>
            <button type="button" class="change-img-btn btn-reset-specimen" onclick="resetVisionToNewSpecimen()">Choose Another Image</button>
          </div>
        </div>
      `;
    }
    showErrorNotification(`Vision pipeline error: ${err.message}`);
  } finally {
    clearInterval(progressInterval);
    isVisionAnalyzing = false;
    if (progressCard) {
      progressCard.style.display = "none";
    }
  }
}

function triggerExplainabilityAnalysis() {
  runVisionPrediction(true);
}

let currentExplainMode = "simple";
let cachedExplainability = null;
let cachedVisionData = null;

function setExplainMode(mode) {
  currentExplainMode = mode;
  const btnSimple = document.getElementById("btnExplainSimple");
  const btnTech = document.getElementById("btnExplainTechnical");
  if (btnSimple && btnTech) {
    if (mode === "simple") {
      btnSimple.classList.add("active");
      btnTech.classList.remove("active");
    } else {
      btnTech.classList.add("active");
      btnSimple.classList.remove("active");
    }
  }
  if (cachedExplainability && cachedVisionData) {
    renderExplainabilityPipeline(cachedExplainability, cachedVisionData);
  }
}

function toggleExplainGrid() {
  const grid = document.getElementById("explainabilityGrid");
  const icon = document.getElementById("explainToggleIcon");
  if (!grid) return;
  if (grid.style.display === "none") {
    grid.style.display = "grid";
    if (icon) icon.textContent = "▲";
  } else {
    grid.style.display = "none";
    if (icon) icon.textContent = "▼";
  }
}

function renderExplainabilityPipeline(explainability, data) {
  cachedExplainability = explainability;
  cachedVisionData = data;

  const panel = document.getElementById("visionExplainabilityPanel");
  const grid = document.getElementById("explainabilityGrid");
  if (!panel || !grid) return;

  panel.style.display = "block";

  // Populate summary bar
  const inputResEl = document.getElementById("explainInputRes");
  const normDimEl = document.getElementById("explainNormDim");
  const entropyEl = document.getElementById("explainEntropy");
  const consensusEl = document.getElementById("explainConsensus");

  if (inputResEl) {
    inputResEl.textContent = data.model_metadata?.input_resolution || "256×256";
  }
  if (normDimEl) {
    normDimEl.textContent = data.model_metadata?.input_resolution || "256×256";
  }
  if (entropyEl) {
    const ent = data.uncertainty?.entropy_nats != null 
      ? Number(data.uncertainty.entropy_nats).toFixed(2) + " nats"
      : (data.diagnosis?.entropy != null ? Number(data.diagnosis.entropy).toFixed(2) + " nats" : "--");
    entropyEl.textContent = ent;
  }
  if (consensusEl) {
    const dName = data.diagnosis?.disease_common_name || data.diagnosis?.predicted_class || "--";
    const dConf = data.diagnosis?.confidence_pct != null ? `${Number(data.diagnosis.confidence_pct).toFixed(1)}%` : "--";
    consensusEl.textContent = `${dName} (${dConf})`;
  }

  const stages = explainability?.stages || [];
  if (stages.length === 0) {
    grid.innerHTML = '<div style="grid-column: 1 / -1; padding: 20px; text-align: center; color: var(--text-secondary);">No explainability stages returned for this specimen.</div>';
    return;
  }

  grid.innerHTML = stages.map(stg => {
    const stgNum = stg.stage_number || 1;
    const title = escapeHtml(stg.title || `Stage ${stgNum}`);
    
    // Choose narrative by active explain mode
    let narrative = "";
    if (currentExplainMode === "technical" && stg.technical_explanation) {
      narrative = escapeHtml(stg.technical_explanation);
    } else if (currentExplainMode === "simple" && stg.simple_explanation) {
      narrative = escapeHtml(stg.simple_explanation);
    } else {
      narrative = escapeHtml(stg.explanation || "");
    }

    // Image artifact
    let imgHtml = "";
    if (stg.image_b64) {
      const src = stg.image_b64.startsWith("data:") 
        ? stg.image_b64 
        : `data:image/png;base64,${stg.image_b64}`;
      imgHtml = `
        <div class="explain-img-wrap" onclick="openImageModal('${src}', '${title}')">
          <img src="${src}" class="explain-img" alt="${title}" loading="lazy" />
          <button type="button" class="explain-expand-btn" title="Expand View">⛶</button>
        </div>
      `;
    } else {
      imgHtml = `
        <div class="explain-img-wrap explain-img-placeholder" style="height: 110px; background: linear-gradient(135deg, rgba(82, 183, 136, 0.07), rgba(8, 15, 8, 0.65)); border: 1px dashed rgba(82, 183, 136, 0.3); border-radius: 6px; display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 4px;">
          <span style="font-size: 1.4rem; opacity: 0.9;">📊</span>
          <span style="font-size: 0.75rem; font-weight: 600; color: var(--green-bright); letter-spacing: 0.03em;">MATHEMATICAL SYNTHESIS STAGE</span>
          <span style="font-size: 0.70rem; color: var(--text-secondary);">Audited neural telemetry &amp; classification weights</span>
        </div>
      `;
    }

    // Metrics pills
    let pillsHtml = "";
    if (stg.metrics && typeof stg.metrics === "object") {
      const pillItems = Object.entries(stg.metrics).map(([k, v]) => {
        let displayVal = v;
        if (typeof v === "number") {
          displayVal = Number.isInteger(v) ? v : v.toFixed(3);
        } else if (typeof v === "boolean") {
          displayVal = v ? "True" : "False";
        }
        return `<span class="explain-pill">${escapeHtml(k)}: <strong>${escapeHtml(String(displayVal))}</strong></span>`;
      });
      if (pillItems.length > 0) {
        pillsHtml = `<div class="explain-tech-pills">${pillItems.join("")}</div>`;
      }
    }

    return `
      <div class="explain-card">
        <div class="explain-card-header">
          <span class="explain-stage-badge">Stage ${stgNum}</span>
          <span class="explain-stage-title">${title}</span>
        </div>
        ${imgHtml}
        <p class="explain-narrative">${narrative}</p>
        ${pillsHtml}
      </div>
    `;
  }).join("");
}

function renderVisionDiagnosis(diag) {
  const container = document.getElementById("visionResultContent");
  const triageBadge = document.getElementById("visionTriageBadge");
  const modelBadge = document.getElementById("visionModelBadge");
  if (!container) return;

  const diseaseName = diag.disease_common_name || diag.predicted_class || "Undetermined";
  const crop = diag.crop || "Universal Specimen";
  let rawConf = Number(diag.confidence_pct || 0);
  if (rawConf <= 1.0 && rawConf > 0) rawConf = rawConf * 100;
  const confNum = Math.min(100, Math.max(0, rawConf));
  const confStr = `${confNum.toFixed(1)}%`;
  const isLowConf = Boolean(diag.is_low_confidence || confNum < 50);
  const isInfected = diag.is_infected !== false;

  // Set triage badge honestly
  if (triageBadge) {
    triageBadge.style.display = "inline-flex";
    if (isLowConf) {
      triageBadge.textContent = "LOW CONFIDENCE · TENTATIVE";
      triageBadge.className = "panel-badge badge-warning";
    } else if (isInfected) {
      triageBadge.textContent = diag.triage_stage || "STAGE 2: Verified Lesions";
      triageBadge.className = "panel-badge badge-infected";
    } else {
      triageBadge.textContent = diag.triage_stage || "STAGE 0: Optimal Health";
      triageBadge.className = "panel-badge badge-healthy";
    }
  }

  // Set model badge
  if (modelBadge) {
    modelBadge.style.display = "inline-flex";
    const tier = diag.model_tier || currentVisionModelTier;
    if (tier === "server") {
      modelBadge.textContent = "🧠 Server: EfficientNetV2-S";
      modelBadge.title = diag.model_architecture || "Server-Grade EfficientNetV2-S (256x256, Test Acc 95.13%, Macro F1 0.9354)";
    } else if (tier === "edge") {
      modelBadge.textContent = "⚡ Edge: MobileNetV2";
      modelBadge.title = diag.model_architecture || "Edge MobileNetV2 (224x224)";
    } else if (tier === "ensemble") {
      modelBadge.textContent = "🔀 Ensemble Fusion";
      modelBadge.title = diag.model_architecture || "Ensemble (60% Server + 40% Edge)";
    } else {
      modelBadge.textContent = diag.model_architecture || "EfficientNetV2-S";
    }
  }

  const top3 = diag.top3_predictions || [];
  let top3Html = "";
  if (top3.length > 0) {
    top3Html = `
      <div class="top3-list" style="margin-top: 14px;">
        <div class="top3-label">Alternative Predictions (Top Candidates)</div>
        ${top3
          .map((item, idx) => {
            let itemConf = Number(item.confidence_pct || 0);
            if (itemConf <= 1.0 && itemConf > 0) itemConf = itemConf * 100;
            const pct = Math.min(100, Math.max(0, itemConf)).toFixed(1);
            const barColor = isLowConf ? '#ffd166' : (idx === 0 && isInfected ? '#e07a5f' : 'var(--green-primary)');
            return `
            <div class="top3-item">
              <span class="top3-rank">${idx + 1}</span>
              <span class="top3-crop">${escapeHtml(item.label || item.class_id)}</span>
              <div class="top3-bar-wrap">
                <div class="top3-bar" style="width: ${pct}%; background: ${barColor};"></div>
              </div>
              <span class="top3-conf">${pct}%</span>
            </div>
          `;
          })
          .join("")}
      </div>
    `;
  }

  // Low confidence warning banner - compact, non-repetitive, with expandable telemetry
  let lowConfBannerHtml = "";
  if (isLowConf) {
    const entStr = diag.entropy != null ? Number(diag.entropy).toFixed(2) : (lastVisionResult?.uncertainty?.entropy_nats != null ? Number(lastVisionResult.uncertainty.entropy_nats).toFixed(2) : "--");
    const uncStr = diag.uncertainty_score != null ? Number(diag.uncertainty_score).toFixed(2) : (lastVisionResult?.uncertainty?.normalized_uncertainty != null ? Number(lastVisionResult.uncertainty.normalized_uncertainty).toFixed(2) : "--");
    const marginStr = lastVisionResult?.uncertainty?.prediction_margin != null ? Number(lastVisionResult.uncertainty.prediction_margin).toFixed(3) : "--";

    lowConfBannerHtml = `
      <div class="low-conf-alert" style="margin-bottom: 10px; padding: 8px 12px; border-radius: var(--radius-sm); background: rgba(255, 209, 102, 0.08); border: 1px solid rgba(255, 209, 102, 0.35); display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 8px;">
        <div style="display: flex; align-items: center; gap: 8px; font-size: 0.78rem; color: #ffd166;">
          <span>⚠️</span>
          <span><strong>Low confidence (${confStr})</strong> — review recommended. Tentative visual match.</span>
        </div>
        <details style="font-size: 0.72rem; color: var(--text-secondary); cursor: pointer;">
          <summary style="color: #ffd166; user-select: none;">Uncertainty Metrics</summary>
          <div style="margin-top: 4px; padding: 4px 8px; background: rgba(0,0,0,0.3); border-radius: 4px;">
            Entropy: <strong>${entStr} nats</strong> | Uncertainty: <strong>${uncStr}</strong> | Margin: <strong>${marginStr}</strong>
          </div>
        </details>
      </div>
    `;
  }

  // Plain language explanation and what to inspect next
  const explanationText = diag.short_explanation || lastVisionResult?.short_explanation || 
    `Visual characteristics align most closely with ${crop} - ${diseaseName} across trained categories.`;
  const checkText = diag.what_to_check || lastVisionResult?.what_to_check ||
    "Inspect nearby leaves for similar visible symptoms and examine whether the same pattern occurs across the plant canopy.";

  // Format damage percentage or unavailable
  let damageDisplay = "0.0%";
  if (diag.segmentation_status === "unavailable" || diag.foliar_damage_pct == null) {
    damageDisplay = "Unavailable";
  } else {
    damageDisplay = `${Number(diag.foliar_damage_pct || 0).toFixed(1)}%`;
  }

  const mainCardBg = isLowConf
    ? 'linear-gradient(135deg, #1f1b10, #2b2512)'
    : (isInfected ? 'linear-gradient(135deg, #1c0e0e, #291212)' : 'linear-gradient(135deg, #0f240f, #153315)');
  const mainCardBorder = isLowConf
    ? 'rgba(255, 209, 102, 0.4)'
    : (isInfected ? 'rgba(224, 122, 95, 0.45)' : 'var(--border-bright)');
  const mainCardNameColor = isLowConf
    ? '#ffd166'
    : (isInfected ? '#ff9e7d' : 'var(--green-bright)');
  const ringColor = isLowConf
    ? '#ffd166'
    : (isInfected ? '#e07a5f' : 'var(--green-primary)');

  container.innerHTML = `
    <div class="crop-result">
      ${lowConfBannerHtml}
      <div class="crop-main-card" style="background: ${mainCardBg}; border-color: ${mainCardBorder};">
        <div class="crop-emoji">${isLowConf ? '🔍' : (isInfected ? '🍂' : '🌱')}</div>
        <div class="crop-info">
          <div class="crop-name" style="color: ${mainCardNameColor}; font-size: 20px;">
            ${escapeHtml(diseaseName)} ${isLowConf ? '<span style="font-size: 12px; font-weight: 500; color: #ffd166;">(Tentative)</span>' : ''}
          </div>
          <div class="crop-confidence-text">${escapeHtml(crop)} · ${confStr} Match</div>
        </div>
        <div class="confidence-ring-wrap">
          <div class="confidence-ring" style="background: conic-gradient(${ringColor} ${confNum * 3.6}deg, rgba(82, 183, 136, 0.15) 0deg); box-shadow: 0 0 12px ${isLowConf ? 'rgba(255,209,102,0.2)' : (isInfected ? 'rgba(224,122,95,0.3)' : 'var(--green-glow)')};">
            <span class="confidence-pct" style="color: ${mainCardNameColor};">${confStr}</span>
          </div>
        </div>
      </div>

      <!-- Human-Readable Plain Language Callout -->
      <div class="human-explanation-card" style="margin-top: 12px; padding: 11px 14px; border-radius: var(--radius-sm); background: rgba(82, 183, 136, 0.08); border: 1px solid rgba(82, 183, 136, 0.25);">
        <div style="font-size: 0.80rem; font-weight: 700; color: var(--green-bright); display: flex; align-items: center; gap: 6px; margin-bottom: 4px;">
          <span>🌱</span> What the model found:
        </div>
        <div style="font-size: 0.83rem; color: var(--text-primary); line-height: 1.45;">
          ${escapeHtml(explanationText)}
        </div>
      </div>

      <!-- Observational Next Steps -->
      <div class="what-to-check-card" style="margin-top: 8px; padding: 10px 14px; border-radius: var(--radius-sm); background: rgba(255, 255, 255, 0.03); border: 1px solid var(--border);">
        <div style="font-size: 0.80rem; font-weight: 700; color: #ffd166; display: flex; align-items: center; gap: 6px; margin-bottom: 4px;">
          <span>🔍</span> What to inspect next:
        </div>
        <div style="font-size: 0.81rem; color: var(--text-secondary); line-height: 1.45;">
          ${escapeHtml(checkText)}
        </div>
      </div>

      <div class="rec-env-pills" style="margin-top: 12px;">
        <span class="env-pill" title="Foliar area compromised derived strictly from segmentation mask">📊 Damage: ${damageDisplay}</span>
        <span class="env-pill" title="Targeted lesion spot foci verified by Mobile-UNet foliar segmentation">🎯 Foci Detected: ${diag.lesion_foci_count != null ? diag.lesion_foci_count : 0}</span>
        <span class="env-pill">🧪 Condition: ${escapeHtml(diag.condition_type || (isInfected ? "Fungal / Pathological" : "Healthy Foliage"))}</span>
      </div>

      ${top3Html}

      <div style="margin-top: 14px;">
        ${lastVisionResult && lastVisionResult.explainability ? `
          <button type="button" onclick="document.getElementById('visionExplainabilityPanel').scrollIntoView({behavior: 'smooth'})" style="background: rgba(82, 183, 136, 0.16); border: 1px solid var(--green-primary); color: var(--green-bright); padding: 9px 14px; border-radius: var(--radius-sm); font-size: 0.82rem; cursor: pointer; display: inline-flex; align-items: center; gap: 8px; justify-content: center; width: 100%; transition: all 0.2s ease;">
            <span>🔬</span>
            <strong>View 9-Stage Explainability Suite (Audited)</strong>
          </button>
        ` : `
          <button type="button" onclick="triggerExplainabilityAnalysis()" style="background: rgba(82, 183, 136, 0.08); border: 1px dashed var(--green-primary); color: var(--green-bright); padding: 9px 14px; border-radius: var(--radius-sm); font-size: 0.82rem; cursor: pointer; display: inline-flex; align-items: center; gap: 8px; justify-content: center; width: 100%; transition: all 0.2s ease;">
            <span>⚡</span>
            <strong>Compute 9-Stage Explainability Suite (Grad-CAM & Neural Activations)</strong>
          </button>
        `}
      </div>
    </div>
  `;
}

function renderVisionAdvisory(adv, diag = {}) {
  const container = document.getElementById("visionAdvisoryContent");
  if (!container) return;

  const rawConf = Number(diag.confidence_pct || 0);
  const confNum = rawConf <= 1.0 && rawConf > 0 ? rawConf * 100 : rawConf;
  const isLowConf = Boolean(diag.is_low_confidence || confNum < 50);
  const isInfected = diag.is_infected !== false;

  const immediate = adv.immediate || adv.pathogen_etiology || (isLowConf
    ? "Examine leaf in field under natural diffused lighting. Isolate specimen if lesion symptoms worsen, but avoid premature chemical application until confirmed."
    : (isInfected
      ? "Inspect field canopy and isolate affected foliage to halt lesion spread."
      : "No immediate quarantine action required. Foliar tissue displays healthy photosynthetic vigor."));

  const treatment = adv.treatment || adv.chemical_control || (isLowConf
    ? "No aggressive chemical intervention recommended at low diagnostic confidence. Retake photograph or consult a certified local agronomist."
    : (isInfected
      ? "Apply protective bio-fungicide or targeted copper-based bactericide/fungicide."
      : "No chemical fungicide or bactericide application warranted. Maintain balanced foliar biostimulants."));

  const cultural = adv.cultural || adv.cultural_sanitation || (isInfected
    ? "Reduce canopy humidity by improving spacing and pruning. Switch completely to ground-level drip irrigation."
    : "Maintain standard drip scheduling, adequate root aeration, and preventive weed sanitation.");

  const symptoms = adv.symptoms || (isLowConf
    ? `Atypical or low-confidence foliar symptoms observed with uncertainty score ${diag.uncertainty_score != null ? diag.uncertainty_score.toFixed(2) : '--'}.`
    : (isInfected
      ? `Foliar necrotic lesions detected with ${diag.foliar_damage_pct != null ? Number(diag.foliar_damage_pct).toFixed(1) : '0.0'}% leaf area compromise across ${diag.lesion_foci_count || 0} focal points.`
      : "Foliage exhibits vigorous chlorophyll homeostasis without active necrotic chlorosis."));

  const immediateBadge = isLowConf ? 'MONITOR' : (isInfected ? 'URGENT' : 'PREVENTATIVE');
  const immediateBorder = isLowConf ? 'var(--yellow)' : (isInfected ? 'var(--orange)' : 'var(--green-primary)');

  container.innerHTML = `
    <div class="disease-content">
      <div class="disease-card" style="border-left-color: ${immediateBorder};">
        <div class="disease-card-header" style="cursor: default;">
          <span class="disease-severity-icon">${isLowConf ? '🔍' : '⚡'}</span>
          <span class="disease-name">Immediate Action</span>
          <span class="disease-type-badge">${immediateBadge}</span>
        </div>
        <div class="disease-body open" style="grid-template-columns: 1fr;">
          <div class="disease-field">
            <p>${immediate}</p>
          </div>
        </div>
      </div>

      <div class="disease-card" style="border-left-color: ${isLowConf ? 'var(--teal)' : (isInfected ? 'var(--red)' : 'var(--green-primary)')};">
        <div class="disease-card-header" style="cursor: default;">
          <span class="disease-severity-icon">💊</span>
          <span class="disease-name">Treatment Protocol</span>
          <span class="disease-type-badge">${isLowConf ? 'OBSERVATION' : (isInfected ? 'TREATMENT' : 'MAINTENANCE')}</span>
        </div>
        <div class="disease-body open" style="grid-template-columns: 1fr;">
          <div class="disease-field">
            <p>${treatment}</p>
          </div>
        </div>
      </div>

      <div class="disease-card" style="border-left-color: var(--teal);">
        <div class="disease-card-header" style="cursor: default;">
          <span class="disease-severity-icon">🌾</span>
          <span class="disease-name">Cultural Sanitation</span>
          <span class="disease-type-badge">AGRONOMY</span>
        </div>
        <div class="disease-body open" style="grid-template-columns: 1fr;">
          <div class="disease-field">
            <p>${cultural}</p>
          </div>
        </div>
      </div>

      <div class="disease-card" style="border-left-color: var(--yellow);">
        <div class="disease-card-header" style="cursor: default;">
          <span class="disease-severity-icon">🔬</span>
          <span class="disease-name">Diagnostic Symptoms</span>
          <span class="disease-type-badge">PATHOLOGY</span>
        </div>
        <div class="disease-body open" style="grid-template-columns: 1fr;">
          <div class="disease-field">
            <p>${symptoms}</p>
          </div>
        </div>
      </div>
    </div>
  `;
}

// ── MULTI-MODE VISUALIZATION & SPATIAL TELEMETRY ──────────────────────────────
let currentVisMode = "detection";
let segmentationDisplayMode = "overlay";
let gradCamDisplayMode = "overlay";
showBoundingBoxes = true;
let showBoxLabels = true;

function setVisualizationMode(mode) {
  currentVisMode = mode;
  ['btnVisSpecimen', 'btnVisDetection', 'btnVisSegmentation', 'btnVisGradCam'].forEach(id => {
    const btn = document.getElementById(id);
    if (btn) btn.classList.remove('active');
  });
  const activeBtnMap = {
    specimen: 'btnVisSpecimen',
    detection: 'btnVisDetection',
    segmentation: 'btnVisSegmentation',
    gradcam: 'btnVisGradCam'
  };
  const activeBtn = document.getElementById(activeBtnMap[mode]);
  if (activeBtn) activeBtn.classList.add('active');

  const detControls = document.getElementById("detectionSubcontrols");
  const segControls = document.getElementById("segmentationSubcontrols");
  const camControls = document.getElementById("gradcamSubcontrols");
  if (detControls) detControls.style.display = (mode === "detection") ? "flex" : "none";
  if (segControls) segControls.style.display = (mode === "segmentation") ? "flex" : "none";
  if (camControls) camControls.style.display = (mode === "gradcam") ? "flex" : "none";

  drawSpatialCanvas();
}

function setSegmentationDisplay(displayType) {
  segmentationDisplayMode = displayType;
  const btnOverlay = document.getElementById("btnSegOverlay");
  const btnRaw = document.getElementById("btnSegRaw");
  if (btnOverlay) btnOverlay.classList.toggle("active", displayType === "overlay");
  if (btnRaw) btnRaw.classList.toggle("active", displayType === "raw");
  drawSpatialCanvas();
}

function setGradCamDisplay(displayType) {
  gradCamDisplayMode = displayType;
  const btnOverlay = document.getElementById("btnCamOverlay");
  const btnHeatmap = document.getElementById("btnCamHeatmap");
  if (btnOverlay) btnOverlay.classList.toggle("active", displayType === "overlay");
  if (btnHeatmap) btnHeatmap.classList.toggle("active", displayType === "heatmap");
  drawSpatialCanvas();
}

function toggleBoxLabels(visible) {
  showBoxLabels = Boolean(visible);
  drawSpatialCanvas();
}

function toggleSpecimenBoxes(visible) {
  showSpecimenBoxes = Boolean(visible);
  drawSpatialCanvas();
}

function toggleLesionBoxes(visible) {
  showLesionBoxes = Boolean(visible);
  drawSpatialCanvas();
}

function toggleBoundingBoxes(visible) {
  showBoundingBoxes = Boolean(visible);
  showSpecimenBoxes = Boolean(visible);
  showLesionBoxes = Boolean(visible);
  drawSpatialCanvas();
}

function updateLegendText(text, icon = "ℹ️") {
  const legendText = document.getElementById("legendText");
  const legendIcon = document.getElementById("legendIcon");
  if (legendText) legendText.innerHTML = text;
  if (legendIcon) legendIcon.textContent = icon;
}

function toggleTechDetails() {
  // Always open and visible
  const content = document.getElementById("visionTechContent");
  if (content) content.style.display = "block";
}


function renderTechnicalDetails(data = {}) {
  const panel = document.getElementById("visionTechPanel");
  const content = document.getElementById("visionTechContent");
  if (!panel || !content) return;

  panel.style.display = "block";
  content.style.display = "block";

  const meta = (data && data.model_metadata) || {};
  const latency = (data && (data.latency_ms || data.performance_benchmark)) || {};
  const diag = (data && data.diagnosis) || {};
  const telemetry = (data && data.spatial_telemetry) || {};
  const boxes = telemetry.bounding_boxes || [];

  let boxesTableRows = "";
  if (boxes.length > 0) {
    boxesTableRows = boxes.map(b => {
      const isCanopy = b.category_type === "canopy" || b.box_type === "leaf";
      const catBadge = isCanopy
        ? `<span style="display:inline-block; padding:2px 6px; border-radius:3px; background:rgba(82,183,136,0.15); color:#52b788; font-weight:600;">🌿 Canopy</span>`
        : `<span style="display:inline-block; padding:2px 6px; border-radius:3px; background:rgba(244,162,97,0.15); color:#f4a261; font-weight:600;">🎯 Lesion</span>`;
      const coords = b.bbox_xyxy ? `[${b.bbox_xyxy.join(', ')}]` : '--';
      const conf = b.confidence != null ? `${(b.confidence > 1 ? b.confidence : b.confidence * 100).toFixed(1)}%` : '--';
      return `
        <tr style="border-bottom: 1px solid rgba(255,255,255,0.05); font-size: 0.72rem;">
          <td style="padding: 5px 8px; font-family: monospace; color: var(--text-muted);">${escapeHtml(b.detection_id || '--')}</td>
          <td style="padding: 5px 8px; font-weight: 500;">${escapeHtml(b.class_name || b.label || '--')}</td>
          <td style="padding: 5px 8px;">${catBadge}</td>
          <td style="padding: 5px 8px; color: var(--green-bright);">${conf}</td>
          <td style="padding: 5px 8px; font-family: monospace; color: var(--text-muted);">${coords}</td>
          <td style="padding: 5px 8px; color: var(--text-secondary);">${escapeHtml(b.source_model || telemetry.detection_engine || 'YOLO')}</td>
        </tr>
      `;
    }).join("");
  } else {
    boxesTableRows = `
      <tr>
        <td colspan="6" style="padding: 10px; text-align: center; color: var(--text-muted); font-size: 0.75rem;">
          No bounding box detections localized above inference threshold.
        </td>
      </tr>
    `;
  }

  const specimenDets = telemetry.specimen_detections || boxes.filter(b => b.category_type === "canopy" || b.box_type === "leaf");
  const lesionDets = telemetry.lesion_detections || boxes.filter(b => b.category_type === "lesion");

  const spatialAuditHtml = `
    <details class="spatial-audit-details" style="margin-top: 14px; background: rgba(15, 23, 42, 0.55); border: 1px solid rgba(255,255,255,0.08); border-radius: 8px; padding: 10px 14px;">
      <summary style="cursor: pointer; font-size: 0.78rem; font-weight: 600; color: var(--green-bright); display: flex; align-items: center; justify-content: space-between; user-select: none;">
        <span>🔍 Spatial Detector Inference Audit (${boxes.length} Genuine Detections Preserved: ${specimenDets.length} Specimen, ${lesionDets.length} Lesions)</span>
        <span style="font-size: 0.7rem; color: var(--text-muted); font-weight: normal;">Threshold: 0.10/0.12 · NMS IoU: 0.45</span>
      </summary>
      <div style="font-size: 0.72rem; color: #94a3b8; margin-top: 8px; margin-bottom: 8px; line-height: 1.4;">
        ${escapeHtml(telemetry.localization_notice || "Standard spatial localization active.")}
      </div>
      <div style="margin-top: 6px; display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: 8px; margin-bottom: 12px;">
        <div style="background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.05); border-radius: 6px; padding: 6px 10px;">
          <div style="font-size: 0.65rem; color: var(--text-muted);">Raw Candidates</div>
          <div style="font-size: 0.95rem; font-weight: 700; color: #f8fafc;">${telemetry.raw_detection_count != null ? telemetry.raw_detection_count : '--'}</div>
        </div>
        <div style="background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.05); border-radius: 6px; padding: 6px 10px;">
          <div style="font-size: 0.65rem; color: var(--text-muted);">Post-Filtering Preserved</div>
          <div style="font-size: 0.95rem; font-weight: 700; color: var(--green-bright);">${telemetry.post_filtering_count != null ? telemetry.post_filtering_count : boxes.length}</div>
        </div>
        <div style="background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.05); border-radius: 6px; padding: 6px 10px;">
          <div style="font-size: 0.65rem; color: var(--text-muted);">Canopy Leaf Regions</div>
          <div style="font-size: 0.95rem; font-weight: 700; color: #52b788;">${specimenDets.length}</div>
        </div>
        <div style="background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.05); border-radius: 6px; padding: 6px 10px;">
          <div style="font-size: 0.65rem; color: var(--text-muted);">Pathology Lesion Foci</div>
          <div style="font-size: 0.95rem; font-weight: 700; color: #f4a261;">${lesionDets.length}</div>
        </div>
      </div>
      <div style="overflow-x: auto;">
        <table style="width: 100%; border-collapse: collapse; text-align: left;">
          <thead>
            <tr style="border-bottom: 1px solid rgba(255,255,255,0.1); font-size: 0.68rem; color: var(--text-muted); text-transform: uppercase;">
              <th style="padding: 6px 8px;">Detection ID</th>
              <th style="padding: 6px 8px;">Detector Class</th>
              <th style="padding: 6px 8px;">Category</th>
              <th style="padding: 6px 8px;">Confidence</th>
              <th style="padding: 6px 8px;">BBox [x1, y1, x2, y2]</th>
              <th style="padding: 6px 8px;">Source Model</th>
            </tr>
          </thead>
          <tbody>
            ${boxesTableRows}
          </tbody>
        </table>
      </div>
    </details>
  `;

  content.innerHTML = `
    <div class="tech-grid">
      <div class="tech-item">
        <span class="tech-label">Primary Classifier</span>
        <span class="tech-val">EfficientNetV2-S (256×256)</span>
      </div>
      ${data && data.camera_source === "esp32_cam" ? `
      <div class="tech-item" style="border-color: rgba(82, 183, 136, 0.4); background: rgba(82, 183, 136, 0.08);">
        <span class="tech-label" style="color: var(--green-bright);">📹 Camera Hardware</span>
        <span class="tech-val" style="color: var(--green-bright);">ESP32 CAM · OV2640 (${data.camera_device_id || 'esp32-cam-01'})</span>
      </div>
      <div class="tech-item" style="border-color: rgba(82, 183, 136, 0.4); background: rgba(82, 183, 136, 0.08);">
        <span class="tech-label" style="color: var(--green-bright);">Optical Capture Latency</span>
        <span class="tech-val" style="color: var(--green-bright);">${data.camera_capture_latency_ms || '--'} ms</span>
      </div>
      ` : ''}
      <div class="tech-item">
        <span class="tech-label tooltip-wrap" tabindex="0" aria-label="Test Top-1 Accuracy: 95.13% held-out test evaluation.">
          Test Top-1 Accuracy ℹ️
          <span class="tooltip-box" role="tooltip">Classification accuracy across 38 crop & disease classes on held-out test split.</span>
        </span>
        <span class="tech-val" style="color: var(--green-bright);">95.13% (Benchmark)</span>
      </div>
      <div class="tech-item">
        <span class="tech-label tooltip-wrap" tabindex="0" aria-label="Test Macro F1: 0.9354 balanced class harmonic mean.">
          Test Macro F1 ℹ️
          <span class="tooltip-box" role="tooltip">Unweighted mean F1-score across all 38 classes, verifying robustness against class imbalance.</span>
        </span>
        <span class="tech-val">0.9354 (Benchmark)</span>
      </div>
      <div class="tech-item">
        <span class="tech-label tooltip-wrap" tabindex="0" aria-label="Expected Calibration Error (ECE): 0.0803.">
          Expected Calibration Error ℹ️
          <span class="tooltip-box" role="tooltip">ECE evaluates confidence reliability. Lower values indicate predicted probabilities match real empirical frequencies.</span>
        </span>
        <span class="tech-val">0.0803 (ECE)</span>
      </div>
      <div class="tech-item">
        <span class="tech-label tooltip-wrap" tabindex="0" aria-label="Spatial Detector: YOLO26 Multi-Domain Agricultural Detector.">
          Spatial Detector ℹ️
          <span class="tooltip-box" role="tooltip">Mean Average Precision at IoU 0.50 on the multi-domain foliar lesion benchmark dataset.</span>
        </span>
        <span class="tech-val">${telemetry.detection_engine || 'YOLO26 Multi-Domain Agricultural Detector'} (mAP@50: 0.3415)</span>
      </div>
      <div class="tech-item">
        <span class="tech-label tooltip-wrap" tabindex="0" aria-label="Semantic Segmenter: Mobile-UNet 3 Classes.">
          Semantic Segmenter ℹ️
          <span class="tooltip-box" role="tooltip">Mobile-UNet performs pixel-level foliar damage mask generation into Healthy, Diseased, and Background.</span>
        </span>
        <span class="tech-val">Mobile-UNet (3 Classes)</span>
      </div>
      <div class="tech-item">
        <span class="tech-label">Execution Device</span>
        <span class="tech-val">${meta.device || latency.compute_device || 'CPU'}</span>
      </div>
      <div class="tech-item">
        <span class="tech-label tooltip-wrap" tabindex="0" aria-label="End-to-End Latency: Server inference computation time.">
          End-to-End Latency ℹ️
          <span class="tooltip-box" role="tooltip">Live measured server inference round-trip time across all active vision stages.</span>
        </span>
        <span class="tech-val">${latency.total_ms || latency.total_pipeline_ms || '--'} ms</span>
      </div>
    </div>
    ${spatialAuditHtml}
    <div class="tech-notice">
      <strong>Benchmark Transparency Notice:</strong> The performance metrics displayed above represent rigorous evaluation against held-out validation/test datasets. Real-world diagnostic accuracy depends on field capture conditions, focal distance, ambient lighting, and foliage visibility. SmartCropVision provides statistical computer vision guidance and is not a substitute for professional in-field agronomist verification.
    </div>
  `;
}

function renderSpatialTelemetry(data) {
  const diag = data.diagnosis || {};
  const telemetry = data.spatial_telemetry || {};
  const benchmark = data.performance_benchmark || data.latency_ms || {};
  const boxes = telemetry.bounding_boxes || [];
  const specimenDets = (telemetry.specimen_detections && telemetry.specimen_detections.length > 0)
    ? telemetry.specimen_detections
    : boxes.filter(b => b.category_type === "canopy" || b.box_type === "leaf");
  const lesionDets = (telemetry.lesion_detections && telemetry.lesion_detections.length > 0)
    ? telemetry.lesion_detections
    : boxes.filter(b => b.category_type === "lesion" || b.box_type === "lesion" || b.box_type === "spot");

  const damageVal = document.getElementById("telemetryDamageVal");
  const fociVal = document.getElementById("telemetryFociVal");
  const modelVal = document.getElementById("telemetryModelVal");
  const latencyVal = document.getElementById("telemetryLatencyVal");
  const deviceVal = document.getElementById("telemetryDeviceVal");

  // Subcontrol count badges
  const specimenCountDisplay = document.getElementById("specimenCountDisplay");
  const lesionCountDisplay = document.getElementById("lesionCountDisplay");
  const legendSpecimenCount = document.getElementById("legendSpecimenCount");
  const legendLesionCount = document.getElementById("legendLesionCount");
  const boxCountDisplay = document.getElementById("boxCountDisplay");

  if (specimenCountDisplay) specimenCountDisplay.textContent = String(specimenDets.length);
  if (lesionCountDisplay) lesionCountDisplay.textContent = String(lesionDets.length);
  if (legendSpecimenCount) legendSpecimenCount.textContent = String(specimenDets.length);
  if (legendLesionCount) legendLesionCount.textContent = String(lesionDets.length);
  if (boxCountDisplay) boxCountDisplay.textContent = String(boxes.length);

  // Diagnostic ribbon counts
  const diagRawSpecimen = document.getElementById("diagRawSpecimen");
  const diagRawLesion = document.getElementById("diagRawLesion");
  const diagNmsSpecimen = document.getElementById("diagNmsSpecimen");
  const diagNmsLesion = document.getElementById("diagNmsLesion");
  if (diagRawSpecimen) diagRawSpecimen.textContent = String(telemetry.raw_specimen_count != null ? telemetry.raw_specimen_count : specimenDets.length);
  if (diagRawLesion) diagRawLesion.textContent = String(telemetry.raw_lesion_count != null ? telemetry.raw_lesion_count : lesionDets.length);
  if (diagNmsSpecimen) diagNmsSpecimen.textContent = String(telemetry.post_nms_specimen_count != null ? telemetry.post_nms_specimen_count : specimenDets.length);
  if (diagNmsLesion) diagNmsLesion.textContent = String(telemetry.post_nms_lesion_count != null ? telemetry.post_nms_lesion_count : lesionDets.length);

  if (damageVal) {
    if (diag.foliar_damage_pct == null || diag.segmentation_status === "unavailable") {
      damageVal.textContent = "Unavailable";
      damageVal.title = "No foliar damage mask generated by UNet model";
    } else {
      damageVal.textContent = `${Number(diag.foliar_damage_pct).toFixed(1)}%`;
    }
  }

  if (fociVal) {
    const genuineLesionCount = lesionDets.length;
    const genuineSpecimenCount = specimenDets.length;
    const lesionFociCount = diag.lesion_foci_count != null ? diag.lesion_foci_count : genuineLesionCount;
    const fociSource = diag.lesion_foci_source || telemetry.lesion_foci_source || "none";

    if (genuineLesionCount > 0) {
      fociVal.textContent = `${genuineLesionCount} (Verified lesions)`;
      fociVal.title = `${genuineLesionCount} targeted pathology lesion spots resolved & highlighted on foliage`;
    } else if (fociSource === "mobile_unet_segmentation" && lesionFociCount > 0) {
      fociVal.textContent = `${lesionFociCount} (Segmentation foci)`;
      fociVal.title = `${lesionFociCount} discrete necrotic lesion foci identified via Mobile-UNet semantic segmentation`;
    } else if (genuineSpecimenCount > 0) {
      fociVal.textContent = `0 (Specimen boundary only)`;
      fociVal.title = "Current spatial detector models leaf canopy boundaries only; no lesion spot annotations exist in ontology.";
    } else {
      fociVal.textContent = "0 (No verified lesions)";
      fociVal.title = "Zero spatial lesion foci detected";
    }
  }

  if (modelVal) {
    const arch = diag.model_architecture || (benchmark && benchmark.tier1_model_name) || "EfficientNetV2-S Server-Grade";
    modelVal.textContent = arch.split("(")[0].trim();
    modelVal.title = arch;
  }
  const totalLatency = benchmark.total_pipeline_ms != null ? benchmark.total_pipeline_ms : (benchmark.total_ms != null ? benchmark.total_ms : 0);
  const activeDevice = data.model_metadata?.device || benchmark.compute_device || benchmark.device || "CPU";

  if (latencyVal) latencyVal.textContent = `${Number(totalLatency).toFixed(0)} ms`;
  if (deviceVal) deviceVal.textContent = activeDevice;

  // Set default visualization mode
  if (boxes.length > 0) {
    setVisualizationMode("detection");
  } else if (data.segmentation_mask_b64) {
    setVisualizationMode("segmentation");
  } else {
    setVisualizationMode("specimen");
  }

  renderTechnicalDetails(data);
  drawSpatialCanvas();
}

function drawSpatialCanvas() {
  const canvas = document.getElementById("visionInspectionCanvas");
  const placeholder = document.getElementById("canvasEmptyPlaceholder");
  if (!canvas || !originalVisionImageObj) return;

  const ctx = canvas.getContext("2d");
  const img = originalVisionImageObj;

  canvas.width = img.naturalWidth || img.width || 640;
  canvas.height = img.naturalHeight || img.height || 640;

  if (placeholder) placeholder.style.display = "none";
  canvas.style.display = "block";

  const data = lastVisionResult || {};
  const boxes = data.spatial_telemetry?.bounding_boxes || [];
  const maskOverlayB64 = data.segmentation_mask_b64;
  const maskRawB64 = data.mask_raw_b64;
  const camOverlayB64 = data.cam_overlay_b64 || (data.explainability && data.explainability.stages && data.explainability.stages[7] && data.explainability.stages[7].image_b64);
  const camHeatmapB64 = data.cam_heatmap_b64;

  ctx.clearRect(0, 0, canvas.width, canvas.height);

  if (currentVisMode === "specimen") {
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    updateLegendText("<strong>🍃 Specimen Mode:</strong> Raw unmodified foliage photograph as uploaded for diagnostic evaluation.", "🍃");
    return;
  }

  if (currentVisMode === "detection") {
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

    const specimenBoxes = (data.spatial_telemetry?.specimen_detections && data.spatial_telemetry.specimen_detections.length > 0)
      ? data.spatial_telemetry.specimen_detections
      : boxes.filter(b => b.category_type === "canopy" || b.box_type === "leaf");
    const lesionBoxes = (data.spatial_telemetry?.lesion_detections && data.spatial_telemetry.lesion_detections.length > 0)
      ? data.spatial_telemetry.lesion_detections
      : boxes.filter(b => b.category_type === "lesion" || b.box_type === "lesion" || b.box_type === "spot");

    // Filter by individual toggle layers
    const activeBoxes = [];
    if (showSpecimenBoxes) {
      activeBoxes.push(...specimenBoxes);
    }
    if (showLesionBoxes) {
      activeBoxes.push(...lesionBoxes);
    }

    const renderedSpecimen = showSpecimenBoxes ? specimenBoxes.length : 0;
    const renderedLesion = showLesionBoxes ? lesionBoxes.length : 0;

    const diagRenderedSpecimen = document.getElementById("diagRenderedSpecimen");
    const diagRenderedLesion = document.getElementById("diagRenderedLesion");
    if (diagRenderedSpecimen) diagRenderedSpecimen.textContent = String(renderedSpecimen);
    if (diagRenderedLesion) diagRenderedLesion.textContent = String(renderedLesion);

    if (activeBoxes.length > 0) {
      drawBoundingBoxesOnContext(ctx, activeBoxes, canvas.width, canvas.height, showBoxLabels);
      const engineName = data.spatial_telemetry?.detection_engine || "YOLO PlantDoc";

      let summaryText = "";
      if (renderedLesion > 0 && renderedSpecimen > 0) {
        summaryText = `<strong>🎯 ${engineName} Dual Localization:</strong> ${renderedSpecimen} specimen boundary (🌿 green) and ${renderedLesion} genuine lesion spot foci (🎯 orange) rendered.`;
      } else if (renderedLesion > 0) {
        summaryText = `<strong>🎯 Pathology Lesion Foci:</strong> ${renderedLesion} genuine lesion spot foci rendered.`;
      } else if (renderedSpecimen > 0) {
        summaryText = `<strong>🌿 Specimen Boundary:</strong> ${renderedSpecimen} leaf/canopy specimen boundary region rendered (PlantDoc whole foliar unit).`;
        if (lesionBoxes.length === 0) {
          summaryText += ` <span style="opacity:0.85;font-size:0.86em;">(Model provides specimen localization; no lesion spot annotations exist in ontology.)</span>`;
        }
      }
      updateLegendText(summaryText, renderedLesion > 0 ? "🎯" : "🌿");
    } else if (boxes.length === 0) {
      updateLegendText(`<strong>🎯 ${data.spatial_telemetry?.detection_engine || 'YOLO PlantDoc'} Detection:</strong> No spatial detections localized above inference threshold.`, "🛡️");
    } else {
      updateLegendText("<strong>🎯 Detection Mode:</strong> Spatial detection layers toggled hidden by user.", "🎯");
    }
    return;
  }

  if (currentVisMode === "segmentation") {
    if (segmentationDisplayMode === "raw" && maskRawB64) {
      const maskImg = new Image();
      maskImg.onload = () => {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(maskImg, 0, 0, canvas.width, canvas.height);
      };
      maskImg.src = maskRawB64.startsWith("data:") ? maskRawB64 : `data:image/png;base64,${maskRawB64}`;
      updateLegendText("<strong>🌿 Discrete Class Mask:</strong> Mobile-UNet raw semantic classes: Background (black), Foliar Canopy (green), Active Necrotic Lesion Foci (crimson).", "🌿");
      return;
    }

    if (maskOverlayB64) {
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
      const maskImg = new Image();
      maskImg.onload = () => {
        ctx.save();
        ctx.globalAlpha = 0.42;
        ctx.drawImage(maskImg, 0, 0, canvas.width, canvas.height);
        ctx.restore();
      };
      maskImg.src = maskOverlayB64.startsWith("data:") ? maskOverlayB64 : `data:image/png;base64,${maskOverlayB64}`;
      const damageStr = data.diagnosis?.foliar_damage_pct != null ? `${Number(data.diagnosis.foliar_damage_pct).toFixed(1)}%` : "--%";
      updateLegendText(`<strong>🌿 Mobile-UNet Foliar Segmentation:</strong> Translucent overlay showing foliar canopy (green) and active necrotic lesion foci (crimson). Botanical damage index: ${damageStr}.`, "🌿");
      return;
    }

    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    updateLegendText("<strong>🌿 Foliar Segmentation:</strong> Segmentation model output unavailable or not applicable for healthy foliage.", "ℹ️");
    return;
  }

  if (currentVisMode === "gradcam") {
    if (gradCamDisplayMode === "heatmap" && camHeatmapB64) {
      const hmImg = new Image();
      hmImg.onload = () => {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(hmImg, 0, 0, canvas.width, canvas.height);
      };
      hmImg.src = camHeatmapB64.startsWith("data:") ? camHeatmapB64 : `data:image/png;base64,${camHeatmapB64}`;
      updateLegendText("<strong>🔬 Dynamic Saliency Heatmap:</strong> Gradient-weighted activation map (Jet colormap) extracted from the final convolutional stage of EfficientNetV2-S.", "🔬");
      return;
    }

    if (camOverlayB64) {
      const ovImg = new Image();
      ovImg.onload = () => {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(ovImg, 0, 0, canvas.width, canvas.height);
      };
      ovImg.src = camOverlayB64.startsWith("data:") ? camOverlayB64 : `data:image/png;base64,${camOverlayB64}`;
      updateLegendText("<strong>🔬 EfficientNetV2-S Grad-CAM:</strong> Warmer highlighted regions indicate visual patterns in the leaf that contributed most strongly to the model's prediction.", "🔬");
      return;
    }

    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    updateLegendText("<strong>🔬 Grad-CAM Saliency:</strong> Explainability saliency map unavailable or not computed for this request.", "ℹ️");
    return;
  }
}

function drawBoundingBoxesOnContext(ctx, boxes, imgW, imgH, renderLabels = true) {
  if (!boxes || boxes.length === 0) return;
  ctx.save();

  // Helper for crisp rounded rectangles
  function drawRoundedRectPath(c, x, y, w, h, r) {
    if (w <= 0 || h <= 0) return;
    const radius = Math.max(0, Math.min(r, w / 2, h / 2));
    c.beginPath();
    if (typeof c.roundRect === "function") {
      c.roundRect(x, y, w, h, radius);
    } else {
      c.moveTo(x + radius, y);
      c.lineTo(x + w - radius, y);
      c.arcTo(x + w, y, x + w, y + radius, radius);
      c.lineTo(x + w, y + h - radius);
      c.arcTo(x + w, y + h, x + w - radius, y + h, radius);
      c.lineTo(x + radius, y + h);
      c.arcTo(x, y + h, x, y + h - radius, radius);
      c.lineTo(x, y + radius);
      c.arcTo(x, y, x + radius, y, radius);
      c.closePath();
    }
  }

  // Unified coordinate transformation:
  // Maps original inference image pixel coordinates to displayed canvas coordinate space
  const origW = (originalVisionImageObj && (originalVisionImageObj.naturalWidth || originalVisionImageObj.width)) || imgW;
  const origH = (originalVisionImageObj && (originalVisionImageObj.naturalHeight || originalVisionImageObj.height)) || imgH;
  const scaleX = imgW / (origW || 1);
  const scaleY = imgH / (origH || 1);

  // Process and accurately scale every genuine box independently
  const processedBoxes = boxes.map((box, index) => {
    let [rx1, ry1, rx2, ry2] = box.bbox_xyxy || [0, 0, 0, 0];
    if (rx2 <= 1.0 && ry2 <= 1.0 && (rx2 > 0 || ry2 > 0)) {
      rx1 = rx1 * origW;
      ry1 = ry1 * origH;
      rx2 = rx2 * origW;
      ry2 = ry2 * origH;
    }
    let x1 = Math.round(rx1 * scaleX);
    let y1 = Math.round(ry1 * scaleY);
    let x2 = Math.round(rx2 * scaleX);
    let y2 = Math.round(ry2 * scaleY);

    x1 = Math.max(0, Math.min(x1, imgW));
    y1 = Math.max(0, Math.min(y1, imgH));
    x2 = Math.max(0, Math.min(x2, imgW));
    y2 = Math.max(0, Math.min(y2, imgH));
    const w = x2 - x1;
    const h = y2 - y1;

    const isCanopy = box.category_type === "canopy" || box.box_type === "leaf";
    const strokeColor = box.color_hex || (isCanopy ? "#52b788" : "#f4a261");
    const fillColor = isCanopy ? "rgba(82, 183, 136, 0.08)" : "rgba(244, 162, 97, 0.12)";
    const glowColor = isCanopy ? "rgba(82, 183, 136, 0.35)" : "rgba(244, 162, 97, 0.40)";

    return {
      ...box,
      detection_id: box.detection_id || `det_${index}`,
      index,
      x1, y1, x2, y2, w, h,
      isCanopy,
      strokeColor,
      fillColor,
      glowColor
    };
  }).filter(b => b.w > 0 && b.h > 0);

  // Sort: Canopy boundary boxes first (underneath), lesion foci second (on top)
  processedBoxes.sort((a, b) => (a.isCanopy ? -1 : 1) - (b.isCanopy ? -1 : 1));

  // ── PASS 1: DRAW BOUNDING BOXES (Rounded + Soft Glow Fill) ──────────────────
  processedBoxes.forEach(box => {
    const cornerRadius = box.isCanopy ? Math.min(8, Math.round(imgW / 70)) : Math.min(5, Math.round(imgW / 120));
    const lineWidth = box.isCanopy ? Math.max(2.0, Math.round(imgW / 240)) : Math.max(1.6, Math.round(imgW / 320));

    // Semi-transparent ambient fill
    ctx.shadowBlur = 0;
    ctx.fillStyle = box.fillColor;
    drawRoundedRectPath(ctx, box.x1, box.y1, box.w, box.h, cornerRadius);
    ctx.fill();

    // Subtle edge glow and crisp stroke
    ctx.lineWidth = lineWidth;
    ctx.strokeStyle = box.strokeColor;
    ctx.shadowColor = box.glowColor;
    ctx.shadowBlur = Math.min(6, Math.round(imgW / 140));
    drawRoundedRectPath(ctx, box.x1, box.y1, box.w, box.h, cornerRadius);
    ctx.stroke();
  });

  ctx.shadowBlur = 0;

  // ── PASS 2: DRAW MICRO-PILL BADGES WITH COLLISION AVOIDANCE ────────────────
  if (renderLabels) {
    const placedBadges = [];

    function checkBadgeOverlap(x, y, w, h) {
      const margin = 3;
      for (const p of placedBadges) {
        if (!(x + w + margin < p.x || x - margin > p.x + p.w || y + h + margin < p.y || y - margin > p.y + p.h)) {
          return true;
        }
      }
      return false;
    }

    processedBoxes.forEach(box => {
      const confNum = box.confidence != null ? (box.confidence > 1 ? box.confidence : box.confidence * 100) : null;
      const confStr = confNum != null ? ` · ${confNum.toFixed(0)}%` : "";

      const icon = box.isCanopy ? "🌿" : "🎯";
      const rawName = box.class_name || (box.label ? box.label.replace(/\s*\(\d+%\)/, '') : (box.isCanopy ? "Canopy" : "Lesion"));
      const badgeText = `${icon} ${rawName.trim()}${confStr}`;

      const fontSize = box.isCanopy
        ? Math.max(11, Math.min(13.5, Math.round(imgW / 46)))
        : Math.max(9.5, Math.min(11.5, Math.round(imgW / 56)));

      ctx.font = `600 ${fontSize}px 'Space Grotesk', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif`;
      const textWidth = ctx.measureText(badgeText).width;

      const hasDot = !box.isCanopy;
      const dotRadius = Math.max(2.5, fontSize * 0.24);
      const dotGap = Math.max(5, fontSize * 0.45);
      const dotSpace = hasDot ? (dotRadius * 2 + dotGap) : 0;
      const padX = Math.max(7, Math.round(fontSize * 0.65));

      const badgeW = Math.round(padX * 2 + dotSpace + textWidth);
      const badgeH = Math.max(18, Math.round(fontSize * 1.65));
      const pillRadius = Math.round(badgeH / 2);

      // Candidate placements in priority order:
      const candAboveY = box.y1 - badgeH - 3;
      const candInsideTopY = box.y1 + 4;
      const candBelowY = box.y2 + 3;
      const candInsideBottomY = box.y2 - badgeH - 4;

      const candidates = [];
      if (candAboveY >= 2) {
        candidates.push({ x: box.x1, y: candAboveY });
      }
      if (candInsideTopY + badgeH <= box.y2 - 2) {
        candidates.push({ x: box.x1 + 4, y: candInsideTopY });
      }
      if (candBelowY + badgeH <= imgH - 2) {
        candidates.push({ x: box.x1, y: candBelowY });
      }
      if (candInsideBottomY >= 2) {
        candidates.push({ x: box.x1 + 4, y: candInsideBottomY });
      }
      if (candAboveY >= 2) {
        candidates.push({ x: box.x2 - badgeW, y: candAboveY });
      }

      let chosenPos = null;
      for (const cand of candidates) {
        const cx = Math.max(2, Math.min(imgW - badgeW - 2, cand.x));
        const cy = Math.max(2, Math.min(imgH - badgeH - 2, cand.y));
        if (!checkBadgeOverlap(cx, cy, badgeW, badgeH)) {
          chosenPos = { x: cx, y: cy };
          break;
        }
      }

      // If all candidates have overlap, step downward to find free slot
      if (!chosenPos) {
        let bestX = Math.max(2, Math.min(imgW - badgeW - 2, box.x1));
        let bestY = Math.max(2, Math.min(imgH - badgeH - 2, candAboveY >= 2 ? candAboveY : candInsideTopY));
        let step = 0;
        while (checkBadgeOverlap(bestX, bestY, badgeW, badgeH) && bestY + badgeH + 3 < imgH - 2 && step < 6) {
          bestY += badgeH + 3;
          step++;
        }
        chosenPos = { x: bestX, y: bestY };
      }

      placedBadges.push({ x: chosenPos.x, y: chosenPos.y, w: badgeW, h: badgeH });

      // Draw modern dark glass pill background
      ctx.fillStyle = "rgba(15, 23, 42, 0.88)";
      drawRoundedRectPath(ctx, chosenPos.x, chosenPos.y, badgeW, badgeH, pillRadius);
      ctx.fill();

      // Delicate matching pill outline
      ctx.lineWidth = 1.0;
      ctx.strokeStyle = box.strokeColor;
      ctx.stroke();

      // Glowing indicator dot (for lesion pathology)
      let textStartX = chosenPos.x + padX;
      if (hasDot) {
        const dotCenterX = chosenPos.x + padX + dotRadius;
        const dotCenterY = chosenPos.y + badgeH / 2;

        ctx.save();
        ctx.shadowColor = box.strokeColor;
        ctx.shadowBlur = 4;
        ctx.fillStyle = box.strokeColor;
        ctx.beginPath();
        ctx.arc(dotCenterX, dotCenterY, dotRadius, 0, Math.PI * 2);
        ctx.fill();
        ctx.restore();

        textStartX += dotSpace;
      }

      // Badge typography (crisp, vertically centered)
      ctx.fillStyle = "#f8fafc";
      ctx.textBaseline = "middle";
      ctx.fillText(badgeText, textStartX, chosenPos.y + badgeH / 2);
    });
  }

  ctx.restore();
}


// ── FULLSCREEN IMAGE & CANVAS INSPECTION MODAL ────────────────────────────────
function openImageModal(imgSrc, title) {
  const modal = document.getElementById("imageModal");
  const modalTitle = document.getElementById("imageModalTitle");
  const modalImg = document.getElementById("imageModalImg");
  const modalCanvas = document.getElementById("imageModalCanvas");
  const closeBtn = modal ? modal.querySelector(".image-modal-close") : null;

  if (!modal || !modalImg) return;

  lastFocusedModalElement = document.activeElement;
  document.body.style.overflow = "hidden";

  if (modalTitle) modalTitle.textContent = title || "High-Resolution Foliage Inspection";
  modalImg.src = imgSrc;
  modalImg.style.display = "block";
  if (modalCanvas) modalCanvas.style.display = "none";
  modal.style.display = "flex";

  if (closeBtn) closeBtn.focus();
}

function openCanvasFullscreen() {
  const srcCanvas = document.getElementById("visionInspectionCanvas");
  const modal = document.getElementById("imageModal");
  const modalTitle = document.getElementById("imageModalTitle");
  const modalImg = document.getElementById("imageModalImg");
  const modalCanvas = document.getElementById("imageModalCanvas");
  const closeBtn = modal ? modal.querySelector(".image-modal-close") : null;

  if (!srcCanvas || !modal || !modalCanvas) return;

  lastFocusedModalElement = document.activeElement;
  document.body.style.overflow = "hidden";

  if (modalTitle) modalTitle.textContent = "Full-Resolution Lesion Telemetry & Spatial Mapping";
  if (modalImg) modalImg.style.display = "none";

  modalCanvas.width = srcCanvas.width;
  modalCanvas.height = srcCanvas.height;
  const ctx = modalCanvas.getContext("2d");
  ctx.drawImage(srcCanvas, 0, 0);

  modalCanvas.style.display = "block";
  modal.style.display = "flex";

  if (closeBtn) closeBtn.focus();
}

function closeImageModal() {
  const modal = document.getElementById("imageModal");
  if (modal) {
    modal.style.display = "none";
    document.body.style.overflow = "";
    if (lastFocusedModalElement && typeof lastFocusedModalElement.focus === "function") {
      lastFocusedModalElement.focus();
    }
  }
}

// ── UTILITY: LOADING SPINNER & ERROR NOTIFICATIONS ────────────────────────────
function showLoading(msg) {
  const overlay = document.getElementById("loadingOverlay");
  const text = document.getElementById("loadingOverlayText");
  if (overlay) {
    if (text && msg) text.textContent = msg;
    overlay.style.display = "flex";
  }
}

function hideLoading() {
  const overlay = document.getElementById("loadingOverlay");
  if (overlay) {
    overlay.style.display = "none";
  }
}

// ── TOAST NOTIFICATION SYSTEM ────────────────────────────────────────────────
function showToast(message, type = "info", title = null, durationMs = 4200) {
  const container = document.getElementById("toastContainer");
  if (!container) {
    console.log(`[Toast ${type}] ${message}`);
    return;
  }

  const toast = document.createElement("div");
  toast.className = `toast toast-${type}`;

  const iconMap = {
    success: "✅",
    warning: "⚠️",
    error: "❌",
    info: "ℹ️"
  };
  const icon = iconMap[type] || "🌿";

  const defaultTitles = {
    success: "Diagnosis Complete",
    warning: "Diagnostic Advisory",
    error: "Inference Error",
    info: "System Notice"
  };
  const finalTitle = title || defaultTitles[type] || "Notice";

  toast.innerHTML = `
    <span class="toast-icon">${icon}</span>
    <div class="toast-body">
      <div class="toast-title">${finalTitle}</div>
      <div class="toast-desc">${message}</div>
    </div>
    <button type="button" class="toast-close" title="Dismiss">✕</button>
  `;

  const closeBtn = toast.querySelector(".toast-close");
  const dismiss = () => {
    toast.classList.add("toast-hiding");
    setTimeout(() => {
      if (toast.parentNode) toast.parentNode.removeChild(toast);
    }, 280);
  };

  if (closeBtn) closeBtn.onclick = dismiss;

  container.appendChild(toast);

  if (durationMs > 0) {
    setTimeout(dismiss, durationMs);
  }
}

function showErrorNotification(message) {
  console.error(message);
  showToast(message, "error", "Inference Alert");
}

// ── INDEPENDENT RIGHT PANEL WHEEL SCROLL ROUTER ──────────────────────────────
if (typeof window !== "undefined") {
  window.addEventListener("wheel", (e) => {
    if (window.innerWidth <= 900) return;
    const activeRight = document.querySelector(
      activeMode === "vision" ? "#visionSection .right-column" : "#cropSection .right-column"
    );
    if (!activeRight) return;
    const leftPanel = e.target.closest(".sensor-panel");
    if (leftPanel && leftPanel.scrollHeight > leftPanel.clientHeight) {
      return;
    }
    if (!e.target.closest(".right-column")) {
      activeRight.scrollTop += e.deltaY;
    }
  }, { passive: true });

  // ── RESTORE MODE FROM URL HASH ON PAGE LOAD ──────────────────────────────────
  (function restoreModeFromHash() {
    const hash = window.location.hash.replace("#", "");
    if (hash === "vision" || hash === "crop") {
      switchMode(hash, false); // instant, no animation on page load
    }
  })();
}

// ── AGRIROVER & 4-DOF ROBOTIC ARM TELEOPERATION BRIDGE ────────────────────────
let roverEndpoint = "http://agrirover.local";

async function pingRover() {
  const pill = document.getElementById("roverBridgePill");
  const headerDot = document.getElementById("roverHeaderDot");
  const headerText = document.getElementById("roverHeaderText");

  if (pill) {
    pill.textContent = "Probing...";
    pill.style.color = "#e5c95d";
  }

  try {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 2000);
    const resp = await fetch(`${roverEndpoint}/status`, {
      method: "GET",
      signal: controller.signal
    }).catch(() => null);
    clearTimeout(timeout);

    if (resp && resp.ok) {
      const data = await resp.json();
      if (pill) {
        const armInfo = data.arm ? `Arm: ${Math.round(data.arm.base)}°` : "Ready";
        pill.textContent = `Online • ${armInfo}`;
        pill.style.color = "#5fcf72";
      }
      if (headerDot) {
        headerDot.className = "badge-dot badge-dot-online";
      }
      if (headerText) {
        headerText.textContent = "AgriRover Online";
      }
      if (typeof showToast === "function") {
        showToast("AgriRover & 4-DOF Arm controller online.", "success", "Hardware Link");
      }
    } else {
      if (pill) {
        pill.textContent = "Standby (Click to Open)";
        pill.style.color = "#8ee69b";
      }
      if (typeof showToast === "function") {
        showToast("Access http://agrirover.local or your ESP32 IP to control.", "info", "AgriRover Bridge");
      }
    }
  } catch (err) {
    if (pill) {
      pill.textContent = "Standby • agrirover.local";
      pill.style.color = "#849688";
    }
  }
}

if (typeof window !== "undefined") {
  window.pingRover = pingRover;
}

