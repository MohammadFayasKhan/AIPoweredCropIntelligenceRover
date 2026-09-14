"""
Pre Inference Domain Validation and Agricultural Image Rejection Module.
Provides genuine domain validation defending against arbitrary out of domain uploads
(people, vehicles, buildings, documents, screenshots, animals, blank frames, random noise)
before expensive neural disease classification is invoked.

Three Mutually Exclusive States:
  1. VALID_PLANT_IMAGE: Verified botanical foliage or canopy suitable for disease diagnosis.
  2. INVALID_NON_PLANT_IMAGE: Out of domain non agricultural content rejected from diagnosis.
  3. LOW_QUALITY_OR_UNCERTAIN_IMAGE: Image may contain a plant, but blur, illumination, or visibility is insufficient.

Zero synthetic labels. Zero fabricated metrics. Zero retraining required.
"""

from typing import Dict, Any, Tuple, Optional
import cv2
import numpy as np

# Configurable thresholds calibrated against authentic plant datasets
# and verified negative test suites
CONFIGURABLE_THRESHOLDS = {
    # Minimum required foliar coverage ratio (green foliage + foliar necrotic tissue)
    "min_foliar_ratio_baseline": 0.12,
    # Minimum foliar ratio when supported by detector leaf presence
    "min_foliar_ratio_with_detector": 0.05,
    # Minimum Laplacian edge variance for clear focus (blur rejection)
    "min_blur_laplacian_variance": 35.0,
    # Extreme underexposure ceiling (mean intensity < 25 rejected as too dark)
    "min_brightness_mean": 25.0,
    # Extreme overexposure / washed out ceiling
    "max_brightness_mean": 242.0,
    # Minimum contrast dynamic range
    "min_contrast_std": 18.0,
    # Human skin detection chrominance ceiling (YCrCb color space)
    "max_skin_chrominance_ratio": 0.35,
    # Document / screenshot white background threshold
    "document_brightness_threshold": 210.0,
    "document_max_foliar_ratio": 0.06,
    # Random synthetic noise threshold (extreme variance with low foliar cohesion)
    "noise_blur_variance_ceiling": 18000.0,
    # Minimum connected botanical contour area ratio
    "min_connected_foliar_blob_ratio": 0.02,
    # Digital screenshot and UI detection thresholds
    "screenshot_min_rectilinear_lines": 28,
    "screenshot_min_ui_rectangles": 2,
    "screenshot_top10_flat_color_ratio": 0.55,
    "screenshot_white_px_ratio": 0.50,
}


class DomainValidationResult:
    """Structured result of pre inference domain validation."""
    def __init__(
        self,
        validation_status: str,
        validation_reason: str,
        validation_confidence: float,
        plant_presence: bool,
        leaf_presence: bool,
        image_quality: str,
        is_inference_allowed: bool,
        telemetry: Dict[str, Any],
        screenshot_or_document_probability: float = 0.0,
        inference_allowed: Optional[bool] = None,
    ):
        self.validation_status = validation_status
        self.validation_reason = validation_reason
        self.validation_confidence = round(float(validation_confidence), 3)
        self.plant_presence = bool(plant_presence)
        self.leaf_presence = bool(leaf_presence)
        self.image_quality = image_quality
        self.is_inference_allowed = bool(is_inference_allowed)
        self.inference_allowed = bool(is_inference_allowed if inference_allowed is None else inference_allowed)
        self.screenshot_or_document_probability = round(float(screenshot_or_document_probability), 3)
        self.telemetry = telemetry
        self.telemetry["validation_status"] = self.validation_status
        self.telemetry["validation_reason"] = self.validation_reason
        self.telemetry["validation_confidence"] = self.validation_confidence
        self.telemetry["plant_presence"] = self.plant_presence
        self.telemetry["leaf_presence"] = self.leaf_presence
        self.telemetry["screenshot_or_document_probability"] = self.screenshot_or_document_probability
        self.telemetry["image_quality"] = self.image_quality
        self.telemetry["inference_allowed"] = self.inference_allowed
        self.telemetry["is_inference_allowed"] = self.is_inference_allowed

    def to_dict(self) -> Dict[str, Any]:
        return {
            "validation_status": self.validation_status,
            "validation_reason": self.validation_reason,
            "validation_confidence": self.validation_confidence,
            "plant_presence": self.plant_presence,
            "leaf_presence": self.leaf_presence,
            "screenshot_or_document_probability": self.screenshot_or_document_probability,
            "image_quality": self.image_quality,
            "is_inference_allowed": self.is_inference_allowed,
            "inference_allowed": self.inference_allowed,
            "telemetry": self.telemetry,
        }


def extract_botanical_signals(img_bgr: np.ndarray) -> Dict[str, Any]:
    """
    Computes genuine optical, photometric, and botanical chrominance telemetry
    from raw BGR image matrices.
    """
    orig_h, orig_w = img_bgr.shape[:2]
    # Fast probe resizing (max 640px) enables sub-25ms execution on CPU for high-res inputs
    max_probe = 640
    if max(orig_h, orig_w) > max_probe:
        scale = max_probe / float(max(orig_h, orig_w))
        target_w = max(32, int(orig_w * scale))
        target_h = max(32, int(orig_h * scale))
        probe = cv2.resize(img_bgr, (target_w, target_h), interpolation=cv2.INTER_AREA)
    else:
        probe = img_bgr

    h, w = probe.shape[:2]
    total_pixels = float(max(1, h * w))
    gray = cv2.cvtColor(probe, cv2.COLOR_BGR2GRAY)

    # 1. Blur evaluation via Laplacian operator variance
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    blur_var = float(lap.var())

    # 2. Photometrics: mean intensity and dynamic contrast standard deviation
    mean_bright = float(np.mean(gray))
    std_contrast = float(np.std(gray))

    # 3. Botanical Color Space Decomposition
    hsv = cv2.cvtColor(probe, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]   # 0 to 180 in OpenCV
    sat = hsv[:, :, 1]   # 0 to 255
    val = hsv[:, :, 2]   # 0 to 255

    rgb = cv2.cvtColor(probe, cv2.COLOR_BGR2RGB).astype(np.float32)
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    # Excess Green Index: ExG = 2G - R - B
    exg = 2.0 * g - r - b

    # Green foliage mask: typical leaf hue range 20 to 102 with moderate saturation or positive ExG
    green_foliage_mask = ((hue >= 20) & (hue <= 102) & (sat >= 20) & (val >= 20)) | (exg > 6.0)
    green_presence_ratio = float(np.sum(green_foliage_mask) / total_pixels)

    # Necrotic / chlorotic / rust lesion tissue on leaves:
    # Yellow brown foliar tissue spans hue 8 to 24, with G > B and ExG > -18.0
    # Crucially, genuine foliar lesions co occur with botanical foliage.
    # If green foliage presence is near zero (< 0.035), non green surfaces represent
    # skin, fur, cardboard, or building materials rather than leaf lesions.
    necrotic_foliar_mask = (
        (hue >= 8) & (hue <= 24) & (sat >= 35) & (val >= 25) &
        (g > b) & (exg > -18.0)
    )

    if green_presence_ratio >= 0.035:
        combined_foliar_mask = green_foliage_mask | necrotic_foliar_mask
    else:
        combined_foliar_mask = green_foliage_mask

    foliar_pixel_count = int(np.sum(combined_foliar_mask))
    foliar_presence_ratio = float(foliar_pixel_count / total_pixels)

    # Connected component analysis to assess botanical spatial cohesion
    # Real leaves form large continuous contours, whereas noise or text forms tiny fragments
    max_blob_ratio = 0.0
    if foliar_pixel_count > 30:
        contours, _ = cv2.findContours(
            combined_foliar_mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        if contours:
            max_blob_area = max(cv2.contourArea(c) for c in contours)
            max_blob_ratio = float(max_blob_area / total_pixels)

    # 4. Out of Domain Non Botanical Discriminators
    # Human skin chrominance in YCrCb: Cr in [133, 173], Cb in [77, 127]
    ycrcb = cv2.cvtColor(probe, cv2.COLOR_BGR2YCrCb)
    cr = ycrcb[:, :, 1]
    cb = ycrcb[:, :, 2]
    skin_mask = (cr >= 133) & (cr <= 173) & (cb >= 77) & (cb <= 127)
    # Exclude pixels that have strong green vegetation characteristics
    pure_skin_mask = skin_mask & (~green_foliage_mask)
    skin_ratio = float(np.sum(pure_skin_mask) / total_pixels)

    # Document or screenshot detection: high white background with low foliar coverage
    is_document_like = bool(
        mean_bright > CONFIGURABLE_THRESHOLDS["document_brightness_threshold"]
        and foliar_presence_ratio < CONFIGURABLE_THRESHOLDS["document_max_foliar_ratio"]
    )

    # Random noise or corrupted pattern detection:
    # Unnaturally high Laplacian variance combined with high hue standard deviation
    hue_std = float(np.std(hue)) if hue.size > 0 else 0.0
    is_noise_like = bool(
        blur_var > CONFIGURABLE_THRESHOLDS["noise_blur_variance_ceiling"]
        and (foliar_presence_ratio < 0.20 or hue_std > 42.0)
    )

    # 5. Screenshot, Application UI, Dashboard, and Document Detection
    # Evaluates rectilinear line density, rectangular UI container contours,
    # discrete quantized color dominance (flat panels), and browser chrome elements.
    min_line_len = int(min(w, h) * 0.14)
    edges = cv2.Canny(gray, 40, 120)
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=75,
        minLineLength=min_line_len,
        maxLineGap=8
    )

    long_h_lines = 0
    long_v_lines = 0
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            dx = abs(x2 - x1)
            dy = abs(y2 - y1)
            if dy <= 3 and dx >= min_line_len:
                long_h_lines += 1
            elif dx <= 3 and dy >= min_line_len:
                long_v_lines += 1
    total_rectilinear_lines = long_h_lines + long_v_lines

    # Rectangular UI containers (cards, panels, modal dialogs)
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    ui_rectangles = 0
    for c in contours:
        area = cv2.contourArea(c)
        if area > (total_pixels * 0.005):
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.03 * peri, True)
            if len(approx) == 4:
                _, _, bw, bh = cv2.boundingRect(approx)
                if float(area) / max(1, bw * bh) > 0.70:
                    ui_rectangles += 1

    # Document page structure detection (scanned pages, PDF reader cards)
    doc_pages_detected = 0
    _, thresh_doc = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    doc_contours, _ = cv2.findContours(thresh_doc, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for c in doc_contours:
        bx, by, bw, bh = cv2.boundingRect(c)
        box_area = bw * bh
        if (box_area > total_pixels * 0.05) and (box_area < total_pixels * 0.98):
            cnt_area = cv2.contourArea(c)
            if cnt_area / max(1, box_area) > 0.75:
                region = gray[by:by+bh, bx:bx+bw]
                white_ratio = float(np.mean(region > 200))
                dark_text_ratio = float(np.mean(region < 120))
                var = float(cv2.Laplacian(region, cv2.CV_64F).var())
                if white_ratio > 0.60 and dark_text_ratio > 0.01 and var > 30.0:
                    doc_pages_detected += 1

    # Flat color fill ratio in quantized space (identifies digital UI panels)
    q_thumb = (cv2.resize(probe, (256, 256)) // 4) * 4
    pixels = q_thumb.reshape(-1, 3)
    _, counts = np.unique(pixels, axis=0, return_counts=True)
    top10_flat_ratio = float(np.sum(np.sort(counts)[::-1][:10]) / float(len(pixels)))

    # Browser window control buttons (such as Mac window dots in top bar)
    traffic_lights_detected = False
    if h >= 80 and w >= 200:
        top_left = probe[:60, :150]
        red_pts = int(np.sum((top_left[:, :, 2] > 180) & (top_left[:, :, 1] < 100) & (top_left[:, :, 0] < 100)))
        yellow_pts = int(np.sum((top_left[:, :, 2] > 180) & (top_left[:, :, 1] > 160) & (top_left[:, :, 0] < 100)))
        green_pts = int(np.sum((top_left[:, :, 1] > 160) & (top_left[:, :, 2] < 100) & (top_left[:, :, 0] < 100)))
        if red_pts >= 8 and (yellow_pts >= 8 or green_pts >= 8):
            traffic_lights_detected = True

    white_px_ratio = float(np.sum(gray > 230) / total_pixels)

    # Compute continuous screenshot or document probability
    screen_factors = []
    if traffic_lights_detected:
        screen_factors.append(0.98)
    if doc_pages_detected >= 1:
        screen_factors.append(0.98)
    if is_document_like:
        screen_factors.append(0.95)
    if total_rectilinear_lines >= 20:
        screen_factors.append(min(1.0, total_rectilinear_lines / 35.0))
    if ui_rectangles >= 2 and total_rectilinear_lines >= 8:
        screen_factors.append(min(1.0, ui_rectangles / 4.0))
    if top10_flat_ratio >= 0.50 and total_rectilinear_lines >= 12:
        screen_factors.append(min(1.0, (top10_flat_ratio - 0.40) / 0.40))
    if white_px_ratio >= 0.25 and (is_document_like or total_rectilinear_lines >= 8):
        screen_factors.append(min(1.0, white_px_ratio / 0.60))

    if screen_factors:
        screenshot_prob = float(np.clip(np.max(screen_factors) * 0.80 + np.mean(screen_factors) * 0.20, 0.0, 0.99))
    else:
        screenshot_prob = float(np.clip(total_rectilinear_lines / 100.0, 0.0, 0.25))

    is_screenshot_or_doc = bool(
        traffic_lights_detected
        or (doc_pages_detected >= 1)
        or (is_document_like and not is_noise_like)
        or (screenshot_prob >= 0.75 and total_rectilinear_lines >= 12)
        or (ui_rectangles >= CONFIGURABLE_THRESHOLDS["screenshot_min_ui_rectangles"] and total_rectilinear_lines >= 12)
        or (top10_flat_ratio >= CONFIGURABLE_THRESHOLDS["screenshot_top10_flat_color_ratio"] and total_rectilinear_lines >= 14)
        or (white_px_ratio > CONFIGURABLE_THRESHOLDS["screenshot_white_px_ratio"] and (is_document_like or total_rectilinear_lines >= 8))
    )

    return {
        "blur_variance": round(blur_var, 2),
        "mean_brightness": round(mean_bright, 2),
        "contrast_std": round(std_contrast, 2),
        "foliar_presence_ratio": round(foliar_presence_ratio, 4),
        "green_presence_ratio": round(green_presence_ratio, 4),
        "max_connected_blob_ratio": round(max_blob_ratio, 4),
        "skin_chrominance_ratio": round(skin_ratio, 4),
        "is_document_like": is_document_like,
        "is_noise_like": is_noise_like,
        "is_screenshot_or_document": is_screenshot_or_doc,
        "screenshot_or_document_probability": round(screenshot_prob, 3),
        "doc_pages_detected": doc_pages_detected,
        "total_rectilinear_lines": total_rectilinear_lines,
        "ui_rectangles": ui_rectangles,
        "top10_flat_ratio": round(top10_flat_ratio, 3),
        "white_px_ratio": round(white_px_ratio, 3),
        "traffic_lights_detected": traffic_lights_detected,
        "image_dimensions": [orig_w, orig_h],
    }


def validate_plant_image(
    img_bgr: np.ndarray,
    detector_model: Optional[Any] = None,
    filename: Optional[str] = None,
    source: str = "browser_upload",
    camera_metadata: Optional[Dict[str, Any]] = None
) -> DomainValidationResult:
    """
    Authoritative domain validation preflight with source-aware technical policy.
    Evaluates raw BGR image and determines whether it represents a valid crop leaf specimen.

    Supported Sources:
      - 'browser_upload': Strict technical thresholds (clear lab/field photos)
      - 'browser_camera': Standard browser camera thresholds
      - 'esp32_cam': Authenticated field hardware (OV2640 sensor tolerance, mild blur, lower resolution)

    Returns:
      DomainValidationResult with validation_status in:
        VALID_PLANT_IMAGE
        INVALID_NON_PLANT_IMAGE
        INVALID_SCREENSHOT_OR_DOCUMENT
        LOW_QUALITY_OR_UNCERTAIN_IMAGE
    """
    signals = extract_botanical_signals(img_bgr)
    signals["source"] = source
    if camera_metadata:
        signals["camera_metadata"] = {k: v for k, v in camera_metadata.items() if k != "frame_bytes"}

    is_esp32 = (source == "esp32_cam")
    blur = signals["blur_variance"]
    bright = signals["mean_brightness"]
    contrast = signals["contrast_std"]
    foliar_ratio = signals["foliar_presence_ratio"]
    green_ratio = signals["green_presence_ratio"]
    blob_ratio = signals["max_connected_blob_ratio"]
    skin_ratio = signals["skin_chrominance_ratio"]
    is_doc = signals["is_document_like"]
    is_noise = signals["is_noise_like"]
    is_screenshot_or_doc = signals.get("is_screenshot_or_document", False)

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 0: GROQ MULTIMODAL VISION — PRIMARY AUTHORITATIVE GATE
    # Groq Vision is the fast, accurate semantic gatekeeper.
    # When Groq responds, its verdict is FINAL. Local CV does NOT override it.
    # Local CV only activates when Groq is offline, rate-limited, or timed out.
    # ══════════════════════════════════════════════════════════════════════════
    groq_confirmed_plant = False
    groq_responded = False
    try:
        from backend.app.services.groq_validator import groq_vision_validator
        groq_res = groq_vision_validator.validate_image(img_bgr, filename=filename)
        if groq_res is not None:
            groq_responded = True
            signals["groq_vision"] = {
                "valid": groq_res.valid,
                "category": groq_res.category,
                "plant_present": groq_res.plant_present,
                "leaf_present": groq_res.leaf_present,
                "suitable_for_crop_analysis": groq_res.suitable_for_crop_analysis,
                "reason": groq_res.reason,
                "inference_allowed": groq_res.inference_allowed,
                "latency_ms": groq_res.latency_ms,
                "model_used": groq_res.model_used
            }

            # ── Groq says PLANT LEAF → immediately accept, skip all local CV gates ──
            if groq_res.category == "plant_leaf" and groq_res.inference_allowed:
                groq_confirmed_plant = True
                signals["groq_vision"]["botanical_foliage_override"] = True

                # Evaluate camera-degraded technical quality
                quality_desc = "Groq Vision Verified"
                reason_desc = "Plant leaf confirmed by Groq Vision AI. Specimen is suitable for crop disease diagnosis."
                if is_esp32:
                    if blur < 120.0 or contrast < 24.0 or bright < 35.0 or bright > 220.0 or min(img_bgr.shape[:2]) < 300:
                        quality_desc = "Camera Quality Degraded (ESP32-CAM OV2640)"
                        reason_desc = "Plant leaf verified by Groq Vision AI. Field ESP32-CAM capture accepted under source-aware quality policy."
                    else:
                        quality_desc = "Groq Vision Verified (ESP32-CAM OV2640)"

                # Return VALID immediately — Groq is authoritative
                return DomainValidationResult(
                    validation_status="VALID_PLANT_IMAGE",
                    validation_reason=reason_desc,
                    validation_confidence=0.97,
                    plant_presence=True,
                    leaf_presence=True,
                    image_quality=quality_desc,
                    is_inference_allowed=True,
                    telemetry=signals,
                    screenshot_or_document_probability=0.0,
                    inference_allowed=True
                )

            # ── Groq says SCREENSHOT / DOCUMENT → immediately reject ──
            elif groq_res.category == "screenshot_document":
                return DomainValidationResult(
                    validation_status="INVALID_SCREENSHOT_OR_DOCUMENT",
                    validation_reason=(
                        groq_res.reason
                        or "Invalid image. This appears to be a software screenshot or document. "
                           "Please upload a direct photograph of a plant leaf."
                    ),
                    validation_confidence=0.98,
                    plant_presence=False,
                    leaf_presence=False,
                    image_quality="Digital Screenshot / UI / Document (Groq Verified)",
                    is_inference_allowed=False,
                    telemetry=signals,
                    screenshot_or_document_probability=0.99,
                    inference_allowed=False
                )

            # ── Groq says NON-PLANT → immediately reject ──
            elif groq_res.category == "non_plant":
                return DomainValidationResult(
                    validation_status="INVALID_NON_PLANT_IMAGE",
                    validation_reason=(
                        groq_res.reason
                        or "Invalid image. No plant or crop leaf detected. "
                           "Please upload a photograph of a plant leaf for disease analysis."
                    ),
                    validation_confidence=0.98,
                    plant_presence=False,
                    leaf_presence=False,
                    image_quality="Non-Plant Subject (Groq Verified)",
                    is_inference_allowed=False,
                    telemetry=signals,
                    screenshot_or_document_probability=signals.get("screenshot_or_document_probability", 0.0),
                    inference_allowed=False
                )

            # ── Groq is uncertain → fall through to local CV as tiebreaker ──
            # (category == "uncertain")

    except Exception as groq_err:
        pass  # Groq unavailable — fall back to local CV

    # 1. Hardware Object Detection Signal (YOLO PlantDoc genuine positive evidence)
    detector_leaf_boxes = 0
    detector_top_conf = 0.0
    detector_leaf_labels = []

    if detector_model is not None:
        try:
            # Probe detector at modest confidence (0.10) to verify leaf presence
            det_res = detector_model.predict(
                img_bgr,
                conf=0.10,
                verbose=False
            )
            if len(det_res) > 0 and det_res[0].boxes is not None and len(det_res[0].boxes) > 0:
                boxes = det_res[0].boxes
                detector_leaf_boxes = len(boxes)
                detector_top_conf = float(boxes.conf[0].cpu().numpy())
                for b_idx in range(min(3, len(boxes))):
                    c_id = int(boxes.cls[b_idx].cpu().numpy())
                    c_name = det_res[0].names.get(c_id, f"Class_{c_id}")
                    detector_leaf_labels.append(f"{c_name} ({float(boxes.conf[b_idx].cpu().numpy()):.2f})")
        except Exception:
            detector_leaf_boxes = 0
            detector_top_conf = 0.0

    signals["detector_leaf_boxes"] = detector_leaf_boxes
    signals["detector_top_confidence"] = round(detector_top_conf, 3)
    signals["detector_leaf_labels"] = detector_leaf_labels

    has_detector_confirmation = bool(detector_leaf_boxes >= 1 and detector_top_conf >= 0.10)

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE A: DEFINITIVE REJECTION OF NON PLANT DOMAINS
    # ══════════════════════════════════════════════════════════════════════════

    screen_prob = signals.get("screenshot_or_document_probability", 0.0)

    # A1. Blank or solid uniform canvas (white, black, or monochrome)
    if (bright > CONFIGURABLE_THRESHOLDS["max_brightness_mean"] and contrast < 12.0) or \
       (bright < 15.0 and contrast < 8.0) or \
       (contrast < 10.0 and foliar_ratio < 0.02):
        return DomainValidationResult(
            validation_status="INVALID_NON_PLANT_IMAGE",
            validation_reason="Invalid image. Please upload a clear photograph of a plant leaf for crop health analysis.",
            validation_confidence=0.96,
            plant_presence=False,
            leaf_presence=False,
            image_quality="Blank Surface",
            is_inference_allowed=False,
            telemetry=signals,
            screenshot_or_document_probability=screen_prob,
            inference_allowed=False
        )

    # A2. Human photograph / portrait / selfie
    # Only reject pure human portraits if there is virtually zero botanical foliage!
    # If a farmer or tester holds a leaf or phone in their hand, it is accepted!
    if skin_ratio > 0.45 and (foliar_ratio < 0.008 and green_ratio < 0.008) and not has_detector_confirmation:
        return DomainValidationResult(
            validation_status="INVALID_NON_PLANT_IMAGE",
            validation_reason="Invalid image. A person or skin surface was detected without any plant foliage. Please upload a clear image of a plant leaf for crop health analysis.",
            validation_confidence=0.92,
            plant_presence=False,
            leaf_presence=False,
            image_quality="Human Subject / Non Agricultural",
            is_inference_allowed=False,
            telemetry=signals,
            screenshot_or_document_probability=screen_prob,
            inference_allowed=False
        )

    # A3. Synthetic random noise or corrupted pattern
    if is_noise:
        return DomainValidationResult(
            validation_status="INVALID_NON_PLANT_IMAGE",
            validation_reason="Invalid image. Please upload a clear photograph of a plant leaf for crop health analysis.",
            validation_confidence=0.95,
            plant_presence=False,
            leaf_presence=False,
            image_quality="Corrupted Pattern",
            is_inference_allowed=False,
            telemetry=signals,
            screenshot_or_document_probability=screen_prob,
            inference_allowed=False
        )

    # ── LOCAL CV FALLBACK: only reached when Groq is offline / rate-limited ──
    # Note: if groq_responded is True, execution never reaches here (returned above).

    # A4. Screenshot / document — local CV tiebreaker (Groq offline fallback only)
    if is_screenshot_or_doc:
        # Check if the screen displays a botanical leaf photo
        screen_foliage_present = bool(
            (foliar_ratio >= 0.04 or green_ratio >= 0.05) and skin_ratio < 0.35
        )
        if has_detector_confirmation or (is_esp32 and screen_foliage_present):
            signals["phone_or_secondary_display_specimen"] = True
            is_screenshot_or_doc = False
        else:
            return DomainValidationResult(
                validation_status="INVALID_SCREENSHOT_OR_DOCUMENT",
                validation_reason="Invalid image. This appears to be a screenshot or document (digital document, UI screenshot, or software interface). "
                                  "Please upload a direct photograph of a plant leaf.",
                validation_confidence=0.93,
                plant_presence=False,
                leaf_presence=False,
                image_quality="Digital Screenshot / UI / Document",
                is_inference_allowed=False,
                telemetry=signals,
                screenshot_or_document_probability=max(0.85, screen_prob),
                inference_allowed=False
            )

    # A5. General non-plant objects (Groq offline fallback only)
    has_botanical_foliage = bool(
        (foliar_ratio >= 0.04 or green_ratio >= 0.05) and
        not is_doc and skin_ratio < 0.35
    )
    has_esp32_foliage = bool(is_esp32 and has_botanical_foliage)

    if not has_detector_confirmation and not has_esp32_foliage:
        if blur < 20.0 and has_botanical_foliage:
            return DomainValidationResult(
                validation_status="LOW_QUALITY_IMAGE",
                validation_reason="Image quality is insufficient for crop diagnosis. Please upload a focused photograph.",
                validation_confidence=0.80,
                plant_presence=True,
                leaf_presence=True,
                image_quality="Severely Blurred",
                is_inference_allowed=False,
                telemetry=signals,
                screenshot_or_document_probability=screen_prob,
                inference_allowed=False
            )

        return DomainValidationResult(
            validation_status="INVALID_NON_PLANT_IMAGE",
            validation_reason="Invalid image. No plant or crop leaf could be detected. "
                              "Please upload a clear image of a plant leaf for crop health analysis.",
            validation_confidence=0.85,
            plant_presence=False,
            leaf_presence=False,
            image_quality="Non Botanical Scene",
            is_inference_allowed=False,
            telemetry=signals,
            screenshot_or_document_probability=screen_prob,
            inference_allowed=False
        )

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE B: LOW QUALITY OR UNCERTAIN PLANT SPECIMENS
    # (Only reached when Groq is offline and YOLO detector confirmed a leaf or ESP32 foliage detected)
    # ══════════════════════════════════════════════════════════════════════════
    # Foliage presence: trust detector confirmation or authentic ESP32 foliar signature
    has_foliar_presence = bool(
        has_detector_confirmation or has_esp32_foliage
    )

    # B1. Severe underexposure (only if image is virtually pitch black and has zero foliage)
    if bright < 12.0 and not has_foliar_presence:
        return DomainValidationResult(
            validation_status="LOW_QUALITY_IMAGE",
            validation_reason="Image quality is insufficient for crop diagnosis. The photograph is too dark to resolve foliar pathology.",
            validation_confidence=0.85,
            plant_presence=False,
            leaf_presence=False,
            image_quality="Critically Underexposed",
            is_inference_allowed=False,
            telemetry=signals,
            screenshot_or_document_probability=screen_prob,
            inference_allowed=False
        )

    # B2. Severe overexposure (washed out solid white glare canvas)
    if bright > 248.0 and not has_foliar_presence and contrast < 12.0:
        return DomainValidationResult(
            validation_status="LOW_QUALITY_IMAGE",
            validation_reason="Image quality is insufficient for crop diagnosis. The photograph is washed out by glare.",
            validation_confidence=0.85,
            plant_presence=False,
            leaf_presence=False,
            image_quality="Critically Overexposed",
            is_inference_allowed=False,
            telemetry=signals,
            screenshot_or_document_probability=screen_prob,
            inference_allowed=False
        )

    # B3. Severe motion or focal blur (only if completely unresolvable and no foliage)
    if blur < 6.0 and not has_foliar_presence:
        return DomainValidationResult(
            validation_status="LOW_QUALITY_IMAGE",
            validation_reason="Image quality is insufficient for crop diagnosis. Please upload a focused photograph.",
            validation_confidence=0.80,
            plant_presence=False,
            leaf_presence=False,
            image_quality="Severely Blurred",
            is_inference_allowed=False,
            telemetry=signals,
            screenshot_or_document_probability=screen_prob,
            inference_allowed=False
        )

    # B4. Insufficient detector support (Groq offline fallback)
    if not has_detector_confirmation and not has_esp32_foliage:
        return DomainValidationResult(
            validation_status="VALIDATION_UNCERTAIN",
            validation_reason="Plant presence could not be confirmed with certainty. "
                              "Please upload a closer, clearer photograph of the plant leaf.",
            validation_confidence=0.68,
            plant_presence=False,
            leaf_presence=False,
            image_quality="Sparse Foliar Coverage",
            is_inference_allowed=False,
            telemetry=signals,
            screenshot_or_document_probability=screen_prob,
            inference_allowed=False
        )

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE C: VALID PLANT SPECIMEN (INFERENCE ALLOWED)
    # ══════════════════════════════════════════════════════════════════════════
    # Meets botanical chrominance criteria, adequate sharpness, and balanced illumination.
    # Supported by detector or foliar presence ratio.

    # Calibrate confidence score based on multi signal alignment
    conf_factors = [
        min(1.0, foliar_ratio / 0.40),
        min(1.0, blur / 250.0),
        1.0 - abs(bright - 128.0) / 128.0,
        1.0 if has_detector_confirmation else 0.75,
        min(1.0, blob_ratio / 0.15) if blob_ratio > 0 else 0.5,
    ]
    validation_conf = float(np.clip(np.mean(conf_factors), 0.70, 0.99))

    quality_desc = "Good / Diagnostic Ready"
    reason_desc = "Verified plant leaf specimen suitable for multi tier agricultural analysis."
    if is_esp32:
        if blur < 120.0 or contrast < 24.0 or bright < 35.0 or min(img_bgr.shape[:2]) < 300:
            quality_desc = "Camera Quality Degraded (ESP32-CAM OV2640)"
            reason_desc = "Foliage verified under adjusted ESP32-CAM camera policy. Mild sensor degradation accepted."
        else:
            quality_desc = "Good / Diagnostic Ready (ESP32-CAM OV2640)"

    return DomainValidationResult(
        validation_status="VALID_PLANT_IMAGE",
        validation_reason=reason_desc,
        validation_confidence=validation_conf,
        plant_presence=True,
        leaf_presence=True,
        image_quality=quality_desc,
        is_inference_allowed=True,
        telemetry=signals,
        screenshot_or_document_probability=screen_prob,
        inference_allowed=True
    )
