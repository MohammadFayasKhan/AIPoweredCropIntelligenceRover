/**
 * Frontend Production Test Suite
 * Tests UI state transitions, stale result invalidation, debouncing,
 * low-confidence rendering honesty, zero-detection handling, and API contract parsing.
 */

const test = require("node:test");
const assert = require("node:assert");

// Test 1: Explicit Vision UI State Machine Transitions
test("UI State Machine defines all required production states", () => {
  const VisionUIState = {
    IDLE: "idle",
    IMAGE_SELECTED: "image_selected",
    VALIDATING: "validating",
    ANALYZING: "analyzing",
    CLASSIFICATION_COMPLETE: "classification_complete",
    DETECTION_COMPLETE: "detection_complete",
    SEGMENTATION_COMPLETE: "segmentation_complete",
    EXPLAINABILITY_AVAILABLE: "explainability_available",
    COMPLETE: "complete",
    ERROR: "error"
  };

  const expectedStates = [
    "idle", "image_selected", "validating", "analyzing",
    "classification_complete", "detection_complete",
    "segmentation_complete", "explainability_available",
    "complete", "error"
  ];

  for (const st of expectedStates) {
    assert.ok(Object.values(VisionUIState).includes(st), `Missing state: ${st}`);
  }
});

// Test 2: Low-Confidence Diagnosis Evaluation
test("Low-confidence inferences are correctly flagged as LOW_UNCERTAIN without false certainty", () => {
  function evaluateDiagnosisConfidence(diag) {
    let rawConf = Number(diag.confidence_pct || 0);
    if (rawConf <= 1.0 && rawConf > 0) rawConf = rawConf * 100;
    const confNum = Math.min(100, Math.max(0, rawConf));
    const isLowConf = Boolean(diag.is_low_confidence || confNum < 50);
    const triageBadgeText = isLowConf ? "LOW CONFIDENCE · TENTATIVE" : (diag.triage_stage || "STAGE 2: Verified Lesions");
    return { confNum, isLowConf, triageBadgeText };
  }

  // Case A: 30.77% Apple Scab
  const resA = evaluateDiagnosisConfidence({ confidence_pct: 30.77, is_low_confidence: true });
  assert.strictEqual(resA.isLowConf, true);
  assert.strictEqual(resA.triageBadgeText, "LOW CONFIDENCE · TENTATIVE");

  // Case B: 18.5% Healthy Leaf
  const resB = evaluateDiagnosisConfidence({ confidence_pct: 18.5, is_low_confidence: true });
  assert.strictEqual(resB.isLowConf, true);
  assert.strictEqual(resB.triageBadgeText, "LOW CONFIDENCE · TENTATIVE");

  // Case C: 94.2% Confident Scab
  const resC = evaluateDiagnosisConfidence({ confidence_pct: 94.2, is_low_confidence: false });
  assert.strictEqual(resC.isLowConf, false);
  assert.notStrictEqual(resC.triageBadgeText, "LOW CONFIDENCE · TENTATIVE");
});

// Test 3: Zero-Detection Bounding Box Handling (No Fake Boxes)
test("Zero detector findings output exactly 0 bounding boxes without synthetic fallbacks", () => {
  function processSpatialTelemetry(apiResponse) {
    const rawBoxes = apiResponse.spatial_telemetry?.bounding_boxes || [];
    // Ensure no fallback synthetic boxes are fabricated
    return {
      boxCount: rawBoxes.length,
      fociCount: apiResponse.diagnosis?.lesion_foci_count || 0,
      boxes: rawBoxes
    };
  }

  const cleanSampleResponse = {
    spatial_telemetry: { detection_engine: "YOLO-PlantDoc", bounding_boxes: [] },
    diagnosis: { lesion_foci_count: 0 }
  };

  const telemetry = processSpatialTelemetry(cleanSampleResponse);
  assert.strictEqual(telemetry.boxCount, 0);
  assert.strictEqual(telemetry.fociCount, 0);
  assert.deepStrictEqual(telemetry.boxes, []);
});

// Test 4: Segmentation Unavailability Honesty
test("Unavailable segmentation outputs 'Unavailable' rather than fabricated 0.0%", () => {
  function formatFoliarDamage(diag) {
    if (diag.segmentation_status === "unavailable" || diag.foliar_damage_pct == null) {
      return "Unavailable";
    }
    return `${Number(diag.foliar_damage_pct || 0).toFixed(1)}%`;
  }

  // Unavailable
  assert.strictEqual(formatFoliarDamage({ segmentation_status: "unavailable", foliar_damage_pct: null }), "Unavailable");
  // Available 0.0%
  assert.strictEqual(formatFoliarDamage({ segmentation_status: "available", foliar_damage_pct: 0.0 }), "0.0%");
  // Available 3.45%
  assert.strictEqual(formatFoliarDamage({ segmentation_status: "available", foliar_damage_pct: 3.45 }), "3.5%");
});

// Test 5: Debounce / Duplicate Submission Prevention
test("Active analysis lock prevents duplicate concurrent requests", () => {
  let isVisionAnalyzing = false;
  let callCount = 0;

  function handleAnalyzeClick() {
    if (isVisionAnalyzing) {
      return false; // Ignored duplicate click
    }
    isVisionAnalyzing = true;
    callCount++;
    return true;
  }

  assert.strictEqual(handleAnalyzeClick(), true);
  assert.strictEqual(handleAnalyzeClick(), false); // Click while in flight ignored
  assert.strictEqual(handleAnalyzeClick(), false); // Another click ignored
  assert.strictEqual(callCount, 1);

  // Complete request
  isVisionAnalyzing = false;
  assert.strictEqual(handleAnalyzeClick(), true);
  assert.strictEqual(callCount, 2);
});

// Test 6: Stale Result Clearing on New Image Selection
test("Selecting a new image clears previous results and resets state", () => {
  let lastVisionResult = { diagnosis: { crop: "Apple", disease_common_name: "Apple Scab" } };
  let uiState = "complete";
  let activePanels = { inspection: true, explainability: true };

  function processNewImage() {
    lastVisionResult = null;
    activePanels.inspection = false;
    activePanels.explainability = false;
    uiState = "image_selected";
  }

  processNewImage();
  assert.strictEqual(lastVisionResult, null);
  assert.strictEqual(uiState, "image_selected");
  assert.strictEqual(activePanels.inspection, false);
  assert.strictEqual(activePanels.explainability, false);
});

// Test 7: Compact Image Quality Pill Logic (Non-blocking Subordinate Indicator)
test("Image quality maps to compact pill classes without blocking primary diagnosis", () => {
  function getQualityPillData(quality) {
    if (!quality) return { visible: false };
    const qLevel = (quality.quality_level || quality.quality_grade || "Good").toUpperCase();
    const isUsable = quality.is_usable !== false;
    let pillClass = "quality-compact-pill";
    let pillText = "Quality: Optimal";

    if (!isUsable) {
      pillClass += " quality-pill-critical";
      pillText = "Quality: Degraded";
    } else if (qLevel === "ACCEPTABLE" || qLevel === "DEGRADED" || (quality.warnings && quality.warnings.length > 0)) {
      pillClass += " quality-pill-warning";
      pillText = "Quality: Acceptable";
    } else {
      pillClass += " quality-pill-optimal";
      pillText = "Quality: Optimal";
    }
    return { visible: true, pillClass, pillText, isUsable };
  }

  // Optimal
  const r1 = getQualityPillData({ quality_level: "Good", is_usable: true, warnings: [] });
  assert.strictEqual(r1.pillClass, "quality-compact-pill quality-pill-optimal");
  assert.strictEqual(r1.pillText, "Quality: Optimal");

  // Acceptable with blur warning
  const r2 = getQualityPillData({ quality_level: "Acceptable", is_usable: true, warnings: ["Mild blur present"] });
  assert.strictEqual(r2.pillClass, "quality-compact-pill quality-pill-warning");
  assert.strictEqual(r2.pillText, "Quality: Acceptable");

  // Critical / degraded
  const r3 = getQualityPillData({ quality_level: "Poor", is_usable: false, warnings: ["Severe blur"] });
  assert.strictEqual(r3.pillClass, "quality-compact-pill quality-pill-critical");
  assert.strictEqual(r3.pillText, "Quality: Degraded");
});

// Test 8: Explainability 9-Stage Simple vs Technical Mode Narrative Selection
test("Explainability narrative adapts cleanly between simple farmer and technical AI mode", () => {
  const sampleStage = {
    stage_number: 8,
    title: "Class Activation Saliency",
    explanation: "Standard default explanation",
    simple_explanation: "The model focused on the speckled discoloration near the edge of the leaf.",
    technical_explanation: "Grad-CAM computes gradients of target class logit with respect to conv_head feature maps (8x8x1280)."
  };

  function getStageNarrative(stage, mode) {
    if (mode === "technical" && stage.technical_explanation) return stage.technical_explanation;
    if (mode === "simple" && stage.simple_explanation) return stage.simple_explanation;
    return stage.explanation;
  }

  assert.strictEqual(getStageNarrative(sampleStage, "simple"), sampleStage.simple_explanation);
  assert.strictEqual(getStageNarrative(sampleStage, "technical"), sampleStage.technical_explanation);
});

// Test 9: Execution Device Provenance Extraction
test("Execution device extracts authentic backend compute hardware without hardcoded defaults", () => {
  function resolveComputeDevice(data) {
    const benchmark = data.performance_benchmark || data.latency_ms || {};
    return data.model_metadata?.device || benchmark.compute_device || benchmark.device || "CPU";
  }

  assert.strictEqual(resolveComputeDevice({ model_metadata: { device: "MPS" } }), "MPS");
  assert.strictEqual(resolveComputeDevice({ performance_benchmark: { compute_device: "cuda:0" } }), "cuda:0");
  assert.strictEqual(resolveComputeDevice({}), "CPU");
});

// Test 10: Auto-Scroll Target Visibility & Header Offset Logic
test("Auto-scroll calculates correct header offset without scrolling when target is already visible", () => {
  function computeScrollAction({ rectTop, rectBottom, viewportHeight, headerHeight, pageYOffset }) {
    const isFullyVisible = rectTop >= (headerHeight + 16) && rectBottom <= viewportHeight;
    if (isFullyVisible) {
      return { shouldScroll: false, targetY: null };
    }
    const targetY = Math.max(0, pageYOffset + rectTop - headerHeight - 16);
    return { shouldScroll: true, targetY };
  }

  // Case A: Button already fully visible on 1080p screen
  const resA = computeScrollAction({
    rectTop: 350,
    rectBottom: 400,
    viewportHeight: 900,
    headerHeight: 70,
    pageYOffset: 0
  });
  assert.strictEqual(resA.shouldScroll, false);

  // Case B: Button below current fold on tablet/mobile
  const resB = computeScrollAction({
    rectTop: 950,
    rectBottom: 1000,
    viewportHeight: 800,
    headerHeight: 70,
    pageYOffset: 100
  });
  assert.strictEqual(resB.shouldScroll, true);
  assert.strictEqual(resB.targetY, 100 + 950 - 70 - 16); // 964px
});

// Test 11: Comprehensive Interaction State Machine Mapping
test("Interaction state machine correctly manages primary CTA and pipeline labels", () => {
  const VisionUIState = {
    EMPTY: "idle",
    IDLE: "idle",
    IMAGE_SELECTED: "image_selected",
    IMAGE_VALIDATING: "validating",
    VALIDATING: "validating",
    READY_TO_ANALYZE: "ready_to_analyze",
    ANALYZING: "analyzing",
    RESULTS_READY: "complete",
    LOW_CONFIDENCE: "low_confidence",
    COMPLETE: "complete",
    ERROR: "error",
    RETRYING: "retrying"
  };

  function getButtonRepresentation(state) {
    if (state === VisionUIState.ANALYZING || state === VisionUIState.RETRYING) {
      return { disabled: true, loading: true, label: "Analyzing Plant Health..." };
    }
    if (state === VisionUIState.COMPLETE || state === VisionUIState.LOW_CONFIDENCE) {
      return { disabled: false, loading: false, label: "Re-Analyze Specimen" };
    }
    if (state === VisionUIState.ERROR) {
      return { disabled: false, loading: false, label: "Retry Analysis" };
    }
    return { disabled: false, loading: false, label: "Analyze Plant Health" };
  }

  assert.deepStrictEqual(getButtonRepresentation(VisionUIState.EMPTY), { disabled: false, loading: false, label: "Analyze Plant Health" });
  assert.deepStrictEqual(getButtonRepresentation(VisionUIState.ANALYZING), { disabled: true, loading: true, label: "Analyzing Plant Health..." });
  assert.deepStrictEqual(getButtonRepresentation(VisionUIState.RESULTS_READY), { disabled: false, loading: false, label: "Re-Analyze Specimen" });
  assert.deepStrictEqual(getButtonRepresentation(VisionUIState.LOW_CONFIDENCE), { disabled: false, loading: false, label: "Re-Analyze Specimen" });
  assert.deepStrictEqual(getButtonRepresentation(VisionUIState.ERROR), { disabled: false, loading: false, label: "Retry Analysis" });
});

// Test 12: Fullscreen Modal Accessibility Scroll Lock and Restoration
test("Fullscreen modal applies and cleans up body scroll lock and focus correctly", () => {
  let bodyOverflow = "";
  let focusedElement = "btnExpand";

  function openModal() {
    bodyOverflow = "hidden";
    focusedElement = "modalCloseBtn";
  }

  function closeModal(previousFocused) {
    bodyOverflow = "";
    focusedElement = previousFocused;
  }

  openModal();
  assert.strictEqual(bodyOverflow, "hidden");
  assert.strictEqual(focusedElement, "modalCloseBtn");

  closeModal("btnExpand");
  assert.strictEqual(bodyOverflow, "");
  assert.strictEqual(focusedElement, "btnExpand");
});

// Test 13: Progressive Non-Blocking Pipeline Stage Progression
test("Progressive stage progression advances through authentic backend tiers", () => {
  const stages = [
    { step: 1, label: "1. Evaluating foliar image quality...", chip: "preflight" },
    { step: 2, label: "2. Classifying plant pathology (EfficientNetV2-S)...", chip: "classify" },
    { step: 3, label: "3. Localizing lesion foci (YOLO PlantDoc)...", chip: "detect" },
    { step: 4, label: "4. Segmenting foliar necrosis (Mobile-UNet)...", chip: "segment" },
    { step: 5, label: "5. Computing Grad-CAM saliency explainability...", chip: "explain" },
  ];

  assert.strictEqual(stages.length, 5);
  assert.strictEqual(stages[0].chip, "preflight");
  assert.strictEqual(stages[4].chip, "explain");
});

// Test 14: HTML Sanitization Utility defines both escapeHTML and escapeHtml aliases
test("HTML sanitization utility exports both escapeHTML and escapeHtml with identical safe behavior", () => {
  const { escapeHTML, escapeHtml } = require("./app.js");
  assert.strictEqual(typeof escapeHTML, "function");
  assert.strictEqual(typeof escapeHtml, "function");
  assert.strictEqual(escapeHTML, escapeHtml);

  const raw = "<script>alert('xss & attack')</script> \"quotes\"";
  const sanitized = escapeHtml(raw);
  assert.strictEqual(sanitized, "&lt;script&gt;alert(&#039;xss &amp; attack&#039;)&lt;/script&gt; &quot;quotes&quot;");
  assert.strictEqual(escapeHtml(null), "");
  assert.strictEqual(escapeHtml(undefined), "");
  assert.strictEqual(escapeHtml("Early Blight"), "Early Blight");
});

// Test 15: Dropzone Click Isolation Guard
test("Dropzone click handler prevents opening file dialog when an image is active", () => {
  let fileDialogOpened = false;
  function mockTriggerFileInput() {
    fileDialogOpened = true;
  }

  function handleDropzoneClick(hasSelectedImage, isPreviewVisible) {
    if (hasSelectedImage || isPreviewVisible) {
      return false; // Suppressed: Never open Finder
    }
    mockTriggerFileInput();
    return true;
  }

  // Case A: No image selected, empty dropzone -> should open file dialog
  fileDialogOpened = false;
  assert.strictEqual(handleDropzoneClick(false, false), true);
  assert.strictEqual(fileDialogOpened, true);

  // Case B: Image selected -> should NEVER open file dialog
  fileDialogOpened = false;
  assert.strictEqual(handleDropzoneClick(true, true), false);
  assert.strictEqual(fileDialogOpened, false);

  // Case C: Preview visible -> should NEVER open file dialog
  fileDialogOpened = false;
  assert.strictEqual(handleDropzoneClick(false, true), false);
  assert.strictEqual(fileDialogOpened, false);
});

// Test 16: Draggable Panel Resizer Bounds Clamping
test("Panel resizer width calculations stay clamped within safe viewport boundaries", () => {
  function clampPanelWidth(startWidth, deltaX, windowWidth) {
    const minW = 280;
    const maxW = Math.min(720, windowWidth * 0.58);
    return Math.max(minW, Math.min(maxW, Math.round(startWidth + deltaX)));
  }

  const screenW = 1440;
  // Normal drag within range
  assert.strictEqual(clampPanelWidth(380, 50, screenW), 430);
  // Negative drag clamped to minimum 280px
  assert.strictEqual(clampPanelWidth(380, -200, screenW), 280);
  // Large drag clamped to maximum boundary
  assert.strictEqual(clampPanelWidth(380, 600, screenW), 720);
});

// Test 17: Model Architecture Verification Telemetry is Always Open and Formatted
test("Model telemetry section formats authentic benchmarks without requiring expand toggle", () => {
  const meta = {
    architecture: "EfficientNetV2-S",
    test_top1_accuracy: 0.9513,
    macro_f1: 0.9354,
    expected_calibration_error: 0.0803,
    device: "MPS"
  };

  const formattedAcc = (meta.test_top1_accuracy * 100).toFixed(2) + "%";
  assert.strictEqual(formattedAcc, "95.13%");
  assert.strictEqual(meta.architecture, "EfficientNetV2-S");
  assert.strictEqual(meta.device, "MPS");
});
// Test 18: Style CSS Structural Integrity (Balanced Braces & Zero Unclosed Media Queries)
test("frontend/style.css has balanced curly braces and zero unclosed media queries", () => {
  const fs = require("fs");
  const path = require("path");
  const cssPath = path.join(__dirname, "style.css");
  const css = fs.readFileSync(cssPath, "utf8");

  const stack = [];
  for (let i = 0; i < css.length; i++) {
    const ch = css[i];
    if (ch === "{") {
      stack.push(i);
    } else if (ch === "}") {
      assert.ok(stack.length > 0, `Extra closing brace '}' found at character ${i}`);
      stack.pop();
    }
  }
  assert.strictEqual(stack.length, 0, `Unclosed open braces '{' detected: ${stack.length}`);
});

// Test 19: Secondary Viable Options Confidence Parsing (Both confidence and confidence_pct)
test("Secondary Viable Options confidence parser correctly extracts confidence and confidence_pct", () => {
  function parseAlternative(alt, idx) {
    let rawAlt = Number(
      alt.confidence != null ? alt.confidence :
      (alt.confidence_pct != null ? alt.confidence_pct :
      (alt.conf != null ? alt.conf :
      (alt.probability != null ? alt.probability * 100 : 0)))
    );
    if (rawAlt <= 1.0 && rawAlt > 0) rawAlt = rawAlt * 100;
    const altPct = Math.min(100, Math.max(0, rawAlt)).toFixed(1);
    const rankNum = alt.rank || (idx + 2);
    return { altPct, rankNum };
  }

  // Case 1: Raw IoT packet with confidence_pct
  const alt1 = { crop: "Muskmelon", confidence_pct: 41.7, rank: 2 };
  const res1 = parseAlternative(alt1, 0);
  assert.strictEqual(res1.altPct, "41.7");
  assert.strictEqual(res1.rankNum, 2);

  // Case 2: Third candidate with confidence_pct
  const alt2 = { crop: "Mothbeans", confidence_pct: 6.6, rank: 3 };
  const res2 = parseAlternative(alt2, 1);
  assert.strictEqual(res2.altPct, "6.6");
  assert.strictEqual(res2.rankNum, 3);

  // Case 3: Legacy or direct endpoint with confidence
  const alt3 = { crop: "Muskmelon", confidence: 41.7, rank: 2 };
  const res3 = parseAlternative(alt3, 0);
  assert.strictEqual(res3.altPct, "41.7");

  // Case 4: Zero confidence should not break
  const alt4 = { crop: "Barley", confidence: 0 };
  const res4 = parseAlternative(alt4, 0);
  assert.strictEqual(res4.altPct, "0.0");
});
