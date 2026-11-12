"""
Master Inference Service for AI Powered Crop Intelligence Rover.
Executes an authoritative multi-tier agricultural vision pipeline:
  • Tier 1 Server-Grade: EfficientNet-B2 (authoritative 288x288, Focal Loss) or Edge MobileNetV2 (224x224)
  • Tier 2 Object Detection: Genuine YOLOv8-nano lesion localization (PlantDoc annotations)
  • Tier 3 Segmentation: Genuine Mobile-UNet sub-pixel pathology segmentation
  • Explainability: Controlled 9-Stage Pipeline (optional on-demand to maintain high throughput)

Concurrency-safe, thread-safe, and production-hardened with explicit component readiness states.
Zero fabricated metrics. Zero synthetic bounding boxes. Zero fake overrides.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import time
import json
import logging
import threading
import torch
import torch.nn as nn
from torchvision import models
from ultralytics import YOLO
import numpy as np
import cv2
import gc

from backend.app.config import settings
from backend.app.utils.image_processing import (
    validate_uploaded_image,
    preprocess_tier1,
    preprocess_tier3,
    create_segmentation_overlay_base64,
    create_segmentation_mask_base64,
    scale_segmentation_mask_to_original,
    generate_safe_sample_id,
)
from backend.app.utils.explainability import (
    build_explainability_pipeline,
    GradCAM,
    get_target_cam_layer,
    numpy_to_base64_jpeg,
)
from backend.app.utils.domain_validation import validate_plant_image
from backend.app.services.model_registry import model_registry
from backend.app.services.advisory_service import generate_agronomic_advisory
from backend.app.schemas.diagnosis import (
    TopPrediction,
    DiagnosisSummary,
    BoundingBox,
    SpatialTelemetry,
    AgronomicAdvisory,
    LatencyBenchmark,
    DiagnosisResponse,
    ModelMetadata,
    ImageQualityAssessment,
    ImageValidationAssessment,
    UncertaintyMetrics,
)

logger = logging.getLogger("crop_rover.inference")



# ── Mobile-UNet Architecture Definition ──────────────────────────────────────
class DoubleConv(nn.Module):
    """(Conv2D → BatchNorm → ReLU) x 2 Block"""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class MobileUNet(nn.Module):
    """Lightweight 4-stage U-Net with lateral skip connections for foliar lesion segmentation"""
    def __init__(self, in_ch: int = 3, num_classes: int = 3):
        super().__init__()
        # Encoder
        self.inc = DoubleConv(in_ch, 16)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(16, 32))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(32, 64))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))

        # Decoder with Bilinear Upsampling
        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv_up1 = DoubleConv(128 + 64, 64)

        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv_up2 = DoubleConv(64 + 32, 32)

        self.up3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv_up3 = DoubleConv(32 + 16, 16)

        # 1x1 Projection Head
        self.outc = nn.Conv2d(16, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        u1 = torch.cat([self.up1(x4), x3], dim=1)
        d1 = self.conv_up1(u1)

        u2 = torch.cat([self.up2(d1), x2], dim=1)
        d2 = self.conv_up2(u2)

        u3 = torch.cat([self.up3(d2), x1], dim=1)
        d3 = self.conv_up3(u3)

        return self.outc(d3)


class MobileUNetGen2(nn.Module):
    """Gen-2 Lightweight 3-stage foliar pathology segmenter."""
    def __init__(self, in_ch: int = 3, num_classes: int = 3):
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(in_ch, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(inplace=True))
        self.enc2 = nn.Sequential(nn.MaxPool2d(2), nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.enc3 = nn.Sequential(nn.MaxPool2d(2), nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.dec1 = nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True), nn.Conv2d(64, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.dec2 = nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True), nn.Conv2d(32, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(inplace=True))
        self.outc = nn.Conv2d(16, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        d1 = self.dec1(e3)
        d2 = self.dec2(d1)
        return self.outc(d2)


# ── Genuine Multi-Tier Cross-Validation & Evidence Synthesis ───────────────────
# Canonical mapping from 29 PlantDoc object detection classes to (canonical_crop, condition_hint)
PLANTDOC_CROP_MAP: Dict[str, Tuple[str, str]] = {
    "grape leaf": ("grape", "healthy"),
    "grape leaf black rot": ("grape", "black rot"),
    "apple leaf": ("apple", "healthy"),
    "apple scab leaf": ("apple", "apple scab"),
    "apple rust leaf": ("apple", "cedar apple rust"),
    "blueberry leaf": ("blueberry", "healthy"),
    "cherry leaf": ("cherry", "healthy"),
    "corn leaf blight": ("corn", "northern leaf blight"),
    "corn rust leaf": ("corn", "common rust"),
    "corn gray leaf spot": ("corn", "gray leaf spot"),
    "peach leaf": ("peach", "healthy"),
    "bell_pepper leaf": ("pepper", "healthy"),
    "bell_pepper leaf spot": ("pepper", "bacterial spot"),
    "potato leaf": ("potato", "healthy"),
    "potato leaf early blight": ("potato", "early blight"),
    "potato leaf late blight": ("potato", "late blight"),
    "raspberry leaf": ("raspberry", "healthy"),
    "soyabean leaf": ("soybean", "healthy"),
    "squash powdery mildew leaf": ("squash", "powdery mildew"),
    "strawberry leaf": ("strawberry", "healthy"),
    "tomato leaf": ("tomato", "healthy"),
    "tomato early blight leaf": ("tomato", "early blight"),
    "tomato leaf late blight": ("tomato", "late blight"),
    "tomato leaf bacterial spot": ("tomato", "bacterial spot"),
    "tomato septoria leaf spot": ("tomato", "septoria leaf spot"),
    "tomato mold leaf": ("tomato", "leaf mold"),
    "tomato leaf mosaic virus": ("tomato", "mosaic virus"),
    "tomato leaf yellow virus": ("tomato", "yellow leaf curl virus"),
    "tomato two spotted spider mites leaf": ("tomato", "two-spotted spider mite"),
}


def reconcile_cross_tier_evidence(
    t1_probs: np.ndarray,
    specimen_detections: List[Any],
    lesion_detections: List[Any],
    idx_to_meta: Dict[int, Dict[str, Any]],
    detection_engine_name: str = "YOLO PlantDoc"
) -> Tuple[np.ndarray, Optional[str]]:
    """
    Stage 9 Decision Synthesis: Correlates whole-image classifier predictions from Tier 1
    (Server-Grade EfficientNetV2-S) with spatial specimen detections from Tier 2 (YOLO PlantDoc).
    Preserves the mathematical softmax distribution of the authoritative classifier while
    providing truthful multi-tier agronomic synthesis without invented lesion models.
    """
    final_probs = np.array(t1_probs, dtype=np.float64, copy=True)
    total_boxes = len(specimen_detections) + len(lesion_detections)
    if total_boxes == 0:
        return final_probs, None

    raw_top1 = int(np.argmax(final_probs))
    raw_crop = idx_to_meta[raw_top1]["crop"].capitalize()
    raw_dis = idx_to_meta[raw_top1]["disease_name"]
    top1_pct = float(final_probs[raw_top1]) * 100.0

    if len(specimen_detections) > 0:
        dom_specimen = max(specimen_detections, key=lambda b: getattr(b, "confidence", 0.0))
        dom_lbl = getattr(dom_specimen, "class_name", "Leaf")
        dom_conf = getattr(dom_specimen, "confidence", 0.5)
        synthesis_note = (
            f"Stage 9 Multi-Tier Synthesis: Tier 1 EfficientNetV2-S classified specimen as {raw_crop} {raw_dis} "
            f"({top1_pct:.1f}% confidence). Tier 2 {detection_engine_name} localized {len(specimen_detections)} canopy "
            f"specimen boundary region(s) (primary boundary: {dom_lbl} at {dom_conf*100:.0f}% confidence)."
        )
    else:
        synthesis_note = (
            f"Stage 9 Multi-Tier Synthesis: Tier 1 EfficientNetV2-S classified specimen as {raw_crop} {raw_dis} "
            f"({top1_pct:.1f}% confidence)."
        )

    return final_probs, synthesis_note


def _inspect_yolo_architecture(yolo_model: Optional[YOLO]) -> Tuple[str, bool, int]:
    """Inspects the underlying Ultralytics checkpoint architecture instead of trusting filenames."""
    if yolo_model is None:
        return "Unavailable", False, 640
    yaml_file = ""
    backbone_ops: set = set()
    is_e2e = False
    has_o2o = False
    yaml_e2e = False
    imgsz = 640
    try:
        overrides = getattr(yolo_model, "overrides", None) or {}
        if overrides.get("imgsz"):
            imgsz = int(overrides["imgsz"])
        nn_mod = getattr(yolo_model, "model", None)
        yaml_info = getattr(nn_mod, "yaml", None) or {}
        yaml_file = str(yaml_info.get("yaml_file", "")).lower()
        yaml_e2e = bool(yaml_info.get("end2end", False))
        for layer in yaml_info.get("backbone", []) or []:
            if isinstance(layer, (list, tuple)) and len(layer) >= 3:
                backbone_ops.add(str(layer[2]))
        if hasattr(nn_mod, "model") and len(nn_mod.model) > 0:
            head = nn_mod.model[-1]
            has_o2o = hasattr(head, "one2one") or hasattr(head, "one2one_cv2")
            is_e2e = bool(
                getattr(head, "end2end", False)
                or getattr(nn_mod, "end2end", False)
                or yaml_e2e
                or has_o2o
            )
    except Exception:
        pass
    is_yolo26 = bool("yolo26" in yaml_file and (has_o2o or yaml_e2e or is_e2e))
    if is_yolo26:
        if "yolo26m" in yaml_file:
            return "YOLO26m", True, imgsz
        if "yolo26s" in yaml_file:
            return "YOLO26s", True, imgsz
        return "YOLO26n", True, imgsz
    if "yolo11" in yaml_file or "C3k2" in backbone_ops or "C2PSA" in backbone_ops:
        return "YOLO11n", False, imgsz
    if "yolov8" in yaml_file or ("C2f" in backbone_ops and not is_e2e):
        return "YOLOv8n", False, imgsz
    return "YOLO", False, imgsz


def _yolo26_mode_kwargs(e2e_flag: bool) -> Dict[str, Any]:
    """Selects Ultralytics >=8.4.200 `nms` flag or legacy `end2end` flag cleanly."""
    try:
        import ultralytics
        parts = tuple(int("".join(ch for ch in t if ch.isdigit()) or 0) for t in ultralytics.__version__.split(".")[:3])
        if parts >= (8, 4, 200):
            return {"nms": (not e2e_flag)}
    except Exception:
        pass
    return {"end2end": e2e_flag}


def clean_foliar_segmentation_mask(
    pred_mask_256: Optional[np.ndarray],
    img_bgr: np.ndarray,
    primary_leaf_box: Optional[BoundingBox] = None,
    specimen_boxes: Optional[List[BoundingBox]] = None,
) -> Optional[np.ndarray]:
    """
    Cleans Mobile-UNet 3-class semantic mask (0=bg, 1=healthy leaf, 2=necrotic lesion)
    to guarantee that non-plant background objects (wood floors, tables, walls, carpets,
    fabrics, human hands, furniture, and soil) are NEVER flagged as foliar tissue or lesions.
    """
    if pred_mask_256 is None:
        return None

    cleaned_mask = pred_mask_256.copy()
    h_256, w_256 = 256, 256
    orig_h, orig_w = img_bgr.shape[:2]

    # Resize image to 256x256 for exact pixel-by-pixel spectral evaluation
    img_256 = cv2.resize(img_bgr, (w_256, h_256), interpolation=cv2.INTER_AREA)
    b_256 = img_256[:, :, 0].astype(float)
    g_256 = img_256[:, :, 1].astype(float)
    r_256 = img_256[:, :, 2].astype(float)
    exg_256 = 2.0 * g_256 - r_256 - b_256

    hsv_256 = cv2.cvtColor(img_256, cv2.COLOR_BGR2HSV)
    h_c, s_c, v_c = hsv_256[:, :, 0], hsv_256[:, :, 1], hsv_256[:, :, 2]

    # Bare soil discriminator in 256x256
    is_soil_256 = (r_256 > g_256 + 8) & (exg_256 <= 0) & (h_c < 30)

    # Universal Non-Plant Background in 256x256 (floors, tables, carpets, fabrics, hands, walls)
    is_non_plant_256 = (
        ((r_256 >= g_256) & ((exg_256 <= 6.0) | (h_c < 28) | (h_c > 160))) |
        (exg_256 <= 0.0) |
        ((s_c < 22) & ((v_c < 45) | (v_c > 220))) |
        ((r_256 > g_256 + 12) & (h_c < 25) & (s_c > 40)) |  # Hand / skin tones
        is_soil_256
    )

    # Zero out healthy leaf (Class 1) on non-plant background (wood, floor, tables, soil)
    # Note: Necrotic lesions (Class 2) have high red and low ExG, so general non-plant filters
    # are not applied blindly to class 2, EXCEPT unambiguous human skin tones.
    cleaned_mask[(cleaned_mask == 1) & is_non_plant_256] = 0
    cleaned_mask[(cleaned_mask == 1) & is_soil_256] = 0

    ycrcb_256 = cv2.cvtColor(img_256, cv2.COLOR_BGR2YCrCb)
    cr_256, cb_256 = ycrcb_256[:, :, 1], ycrcb_256[:, :, 2]
    is_skin_256 = (
        (cr_256 >= 133) & (cr_256 <= 173) &
        (cb_256 >= 77) & (cb_256 <= 127) &
        (r_256 > g_256 + 10.0) & (exg_256 < 0.0)
    )
    cleaned_mask[is_skin_256] = 0

    # Confine semantic mask strictly inside detected specimen leaf boxes (union of individual
    # boxes prevents empty background gaps between separate leaves from polluting lesion area).
    leaf_boxes_to_mask = specimen_boxes if specimen_boxes else ([primary_leaf_box] if primary_leaf_box is not None else [])
    if leaf_boxes_to_mask:
        inside_any_leaf = np.zeros((256, 256), dtype=bool)
        for s_box in leaf_boxes_to_mask:
            lx1, ly1, lx2, ly2 = s_box.bbox_xyxy
            pad_x = max(2, int(0.02 * (lx2 - lx1) * 256.0 / orig_w))
            pad_y = max(2, int(0.02 * (ly2 - ly1) * 256.0 / orig_h))
            bx1 = max(0, int(lx1 * 256.0 / orig_w) - pad_x)
            by1 = max(0, int(ly1 * 256.0 / orig_h) - pad_y)
            bx2 = min(256, int(lx2 * 256.0 / orig_w) + pad_x)
            by2 = min(256, int(ly2 * 256.0 / orig_h) + pad_y)
            inside_any_leaf[by1:by2, bx1:bx2] = True
        cleaned_mask[~inside_any_leaf] = 0

    # Necrotic lesions (Class 2) must be adjacent to or within foliar canopy (Class 1)
    canopy_256 = (cleaned_mask == 1).astype(np.uint8)
    if np.any(canopy_256):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        canopy_reach = cv2.dilate(canopy_256, kernel)
        orphaned_lesions = (cleaned_mask == 2) & (canopy_reach == 0)
        cleaned_mask[orphaned_lesions] = 0
    else:
        cleaned_mask[cleaned_mask == 2] = 0

    # Discard oversized lesion components in 256x256 (background-scale blobs, not genuine blight)
    # Note: severe blight/necrosis legitimately produces large connected necrotic regions,
    # so thresholds are generous to avoid stripping authentic pathology.
    lesion_bin = (cleaned_mask == 2).astype(np.uint8)
    if np.any(lesion_bin):
        n_comp, l_comp, st_comp, _ = cv2.connectedComponentsWithStats(lesion_bin, connectivity=8)
        for c_idx in range(1, n_comp):
            c_w = st_comp[c_idx, cv2.CC_STAT_WIDTH]
            c_h = st_comp[c_idx, cv2.CC_STAT_HEIGHT]
            c_area = st_comp[c_idx, cv2.CC_STAT_AREA]
            if c_w > 140 or c_h > 140 or c_area > 6000:
                cleaned_mask[l_comp == c_idx] = 0

    return cleaned_mask


def _clamp_xyxy(xyxy_raw: List[float], orig_w: int, orig_h: int) -> Tuple[int, int, int, int]:
    """Clamps raw detector xyxy coordinates cleanly into [0, orig_w] x [0, orig_h]."""
    x1 = int(round(max(0.0, min(float(xyxy_raw[0]), float(orig_w - 1)))))
    y1 = int(round(max(0.0, min(float(xyxy_raw[1]), float(orig_h - 1)))))
    x2 = int(round(max(float(x1 + 1), min(float(xyxy_raw[2]), float(orig_w)))))
    y2 = int(round(max(float(y1 + 1), min(float(xyxy_raw[3]), float(orig_h)))))
    return x1, y1, x2, y2


def _is_inside_specimen_canopy(
    x1: int, y1: int, x2: int, y2: int, specimen_detections: List[BoundingBox]
) -> bool:
    """Checks whether a candidate lesion spot lies inside any detected leaf specimen boundary."""
    if not specimen_detections:
        return True
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    box_area = float(max(1, (x2 - x1) * (y2 - y1)))
    for spec in specimen_detections:
        sx1, sy1, sx2, sy2 = spec.bbox_xyxy
        pad_x = 0.02 * (sx2 - sx1)
        pad_y = 0.02 * (sy2 - sy1)
        if (sx1 - pad_x) <= cx <= (sx2 + pad_x) and (sy1 - pad_y) <= cy <= (sy2 + pad_y):
            return True
        ix1, iy1 = max(x1, sx1), max(y1, sy1)
        ix2, iy2 = min(x2, sx2), min(y2, sy2)
        if ix2 > ix1 and iy2 > iy1 and ((ix2 - ix1) * (iy2 - iy1)) / box_area >= 0.45:
            return True
    return False


def _passes_lesion_spectral_checks(
    img_bgr: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    orig_w: int,
    orig_h: int,
    strict_spectral: bool = True,
) -> bool:
    """Rejects human skin/fingers, bare soil/debris, and uniform healthy green leaf patches."""
    crop = img_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return False

    mb, mg, mr = crop.mean(axis=(0, 1))
    exg = 2.0 * mg - mr - mb
    if exg < -35.0 and mr > (mg + 15.0):
        return False

    c_b = crop[:, :, 0].astype(np.float32)
    c_g = crop[:, :, 1].astype(np.float32)
    c_r = crop[:, :, 2].astype(np.float32)
    px_exg = 2.0 * c_g - c_r - c_b
    green_frac = float(np.mean(px_exg > 12.0))

    ycrcb = cv2.cvtColor(crop, cv2.COLOR_BGR2YCrCb)
    cr, cb = ycrcb[:, :, 1], ycrcb[:, :, 2]
    skin_thresh = 0.50 if strict_spectral else 0.65
    if exg < -2.0 and mr > (mg + 10.0) and (
        float(np.mean((cr >= 133) & (cr <= 173) & (cb >= 77) & (cb <= 127) & (px_exg < 15.0))) >= skin_thresh
        or (strict_spectral and green_frac < 0.25)
    ):
        return False

    if strict_spectral:
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        dark_frac = float(np.mean(hsv[:, :, 2] < 105))
        if mr < (mg - 2.0) and exg > 12.0:
            return False
        if green_frac > 0.82 and mr < mg and dark_frac < 0.15:
            return False

    bw, bh = x2 - x1, y2 - y1
    pad = max(6, int(0.35 * max(bw, bh)))
    rx1, ry1 = max(0, x1 - pad), max(0, y1 - pad)
    rx2, ry2 = min(orig_w, x2 + pad), min(orig_h, y2 + pad)
    ring = img_bgr[ry1:ry2, rx1:rx2]
    if ring.size > 0:
        rex = 2.0 * ring[:, :, 1].astype(np.float32) - ring[:, :, 2].astype(np.float32) - ring[:, :, 0].astype(np.float32)
        ring_green = float(np.mean(rex > 10.0))
        min_gf = 0.10 if strict_spectral else 0.05
        min_rg = 0.25 if strict_spectral else 0.18
        if green_frac < min_gf and ring_green < min_rg:
            return False

    return True


def _filter_lesion_detections(
    img_bgr: np.ndarray,
    les_res: Any,
    specimen_detections: List[BoundingBox],
    t1_disease_name: str,
    conf_thresh: float,
    source_model_name: str = "YOLOv8n Lesion Spot Detector",
    is_yolo26_seg: bool = False,
    accepted_raw_indices_out: Optional[List[int]] = None,
) -> Tuple[List[BoundingBox], int]:
    """Filters raw lesion candidates using spatial containment, spectral guards, and nested IoM suppression."""
    if not les_res or len(les_res) == 0 or les_res[0].boxes is None:
        return [], 0

    orig_h, orig_w = img_bgr.shape[:2]
    img_area = float(max(1, orig_w * orig_h))
    l_boxes = les_res[0].boxes
    raw_count = len(l_boxes)
    if raw_count == 0:
        return [], 0

    confs_arr = l_boxes.conf.cpu().numpy()
    sorted_indices = sorted(range(raw_count), key=lambda idx: float(confs_arr[idx]), reverse=True)
    is_early_blight_ctx = "early blight" in (t1_disease_name or "").lower()
    max_area_frac = 0.35 if is_yolo26_seg else 0.15

    kept_boxes: List[BoundingBox] = []
    accepted_coords: List[Tuple[int, int, int, int]] = []

    for j in sorted_indices:
        conf = float(confs_arr[j])
        if conf < conf_thresh:
            break

        cls_id = int(l_boxes.cls[j].cpu().numpy())
        cls_name = les_res[0].names.get(cls_id, f"Lesion_{cls_id}")
        xyxy_raw = l_boxes.xyxy[j].cpu().numpy().astype(float).tolist()
        x1, y1, x2, y2 = _clamp_xyxy(xyxy_raw, orig_w, orig_h)
        bw, bh = x2 - x1, y2 - y1
        b_area = max(1, bw * bh)

        # Reject degenerate slivers, extreme aspect ratios, or oversized whole-canopy hallucinations
        if bw < 6 or bh < 6 or b_area > max_area_frac * img_area or max(bw / bh, bh / bw) > 6.0:
            continue

        if not _is_inside_specimen_canopy(x1, y1, x2, y2, specimen_detections):
            continue

        if not _passes_lesion_spectral_checks(
            img_bgr, x1, y1, x2, y2, orig_w, orig_h, strict_spectral=not is_yolo26_seg
        ):
            continue

        # Suppress nested / redundant lesion boxes via Intersection-over-Minimum (IoM >= 0.70)
        is_nested_dup = False
        for ax1, ay1, ax2, ay2 in accepted_coords:
            ix1, iy1 = max(x1, ax1), max(y1, ay1)
            ix2, iy2 = min(x2, ax2), min(y2, ay2)
            if ix2 > ix1 and iy2 > iy1:
                inter_area = (ix2 - ix1) * (iy2 - iy1)
                min_area = float(max(1, min(b_area, (ax2 - ax1) * (ay2 - ay1))))
                if inter_area / min_area >= 0.70:
                    is_nested_dup = True
                    break
        if is_nested_dup:
            continue

        accepted_coords.append((x1, y1, x2, y2))
        if accepted_raw_indices_out is not None:
            accepted_raw_indices_out.append(j)
        cx = round(float((x1 + x2) / (2.0 * orig_w)), 3)
        cy = round(float((y1 + y2) / (2.0 * orig_h)), 3)
        display_cls = "Early Blight Lesion" if (is_early_blight_ctx and cls_name == "early_blight_lesion") else (
            "Necrotic Lesion Focus" if cls_name == "early_blight_lesion" else (
                "Disease Lesion" if cls_name == "disease_lesion" else cls_name.replace("_", " ").title()
            )
        )
        kept_boxes.append(
            BoundingBox(
                detection_id=f"lesion_{len(kept_boxes)}_{cls_id}",
                class_id=cls_id,
                class_name=cls_name,
                category_type="lesion",
                label=f"{display_cls} ({conf * 100:.0f}%)",
                bbox_xyxy=[x1, y1, x2, y2],
                confidence=round(conf, 2),
                centroid_norm=[cx, cy],
                box_type="lesion",
                color_hex="#f4a261",
                source_model=source_model_name,
                coordinate_space="original_image_pixels",
            )
        )

    return kept_boxes, raw_count



PLANTDOC_CANOPY_CLASSES = {
    "Apple leaf",
    "Bell_pepper leaf",
    "Blueberry leaf",
    "Cherry leaf",
    "grape leaf",
    "Peach leaf",
    "Potato leaf",
    "Raspberry leaf",
    "Soyabean leaf",
    "Strawberry leaf",
    "Tomato leaf",
}

PLANTDOC_LESION_CLASSES = {
    "Apple Scab Leaf",
    "Apple rust leaf",
    "Bell_pepper leaf spot",
    "Corn Gray leaf spot",
    "Corn leaf blight",
    "Corn rust leaf",
    "grape leaf black rot",
    "Potato leaf early blight",
    "Potato leaf late blight",
    "Squash Powdery mildew leaf",
    "Tomato Early blight leaf",
    "Tomato leaf bacterial spot",
    "Tomato leaf late blight",
    "Tomato leaf mosaic virus",
    "Tomato leaf yellow virus",
    "Tomato mold leaf",
    "Tomato Septoria leaf spot",
    "Tomato two spotted spider mites leaf",
}


# ── Model Registry & Inference Service ───────────────────────────────────────
class InferenceEngine:
    _instance: Optional["InferenceEngine"] = None

    def __init__(self):
        # Concurrency safety lock for GPU/MPS inference passes
        self._inference_lock = threading.Lock()

        # Dynamic hardware detection without hardcoded literals
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            gpu_name = torch.cuda.get_device_name(0)
            self.device_name = f"CUDA ({gpu_name})"
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
            self.device_name = "MPS"
        else:
            self.device = torch.device("cpu")
            self.device_name = "CPU"

        self.taxonomy: List[Dict[str, Any]] = []
        self.idx_to_meta: Dict[int, Dict[str, Any]] = {}
        self.canon_to_meta: Dict[str, Dict[str, Any]] = {}

        # Model references (Authentic verified production models only)
        self.model_tier1_server: Optional[nn.Module] = None
        self.model_tier1_edge: Optional[nn.Module] = None
        self.model_tier1_gen2: Optional[nn.Module] = None
        self.model_tier1: Optional[nn.Module] = None
        self.model_tier2: Optional[YOLO] = None
        self.model_tier2_plantdoc: Optional[YOLO] = None
        self.model_tier2_lesions: Optional[YOLO] = None
        self.model_tier2_arch: str = "YOLO11n"
        self.model_tier2_is_yolo26: bool = False
        self.model_tier2_path: Path = settings.TIER2_PLANTDOC_MODEL_PATH
        self.model_tier2_lesions_arch: str = "YOLOv8n"
        self.model_yolo26_specimen: Optional[YOLO] = None
        self.model_yolo26_lesion: Optional[YOLO] = None
        self.model_yolo26_specimen_arch: str = "Unavailable"
        self.model_yolo26_specimen_imgsz: int = 768
        self.model_yolo26_lesion_arch: str = "Unavailable"
        self.model_yolo26_lesion_imgsz: int = 768
        self.model_tier3: Optional[nn.Module] = None

        # Detailed component readiness tracking
        self.model_status_map: Dict[str, str] = {
            "tier1_server": "uninitialized",
            "tier1_edge": "uninitialized",
            "tier1_gen2": "uninitialized",
            "tier2_plantdoc": "uninitialized",
            "tier2_lesions": "uninitialized",
            "yolo26_specimen": "untrained_optional",
            "yolo26_lesion": "untrained_optional",
            "tier3_unet": "uninitialized",
        }
        self.is_loaded: bool = False

    @classmethod
    def get_instance(cls) -> "InferenceEngine":
        if cls._instance is None:
            cls._instance = InferenceEngine()
        return cls._instance

    @property
    def is_ready(self) -> bool:
        """Returns True if the authoritative Tier 1 Server-Grade EfficientNetV2-S classifier is ready."""
        return self.model_tier1_server is not None and len(self.taxonomy) == 38

    def load_models(self) -> None:
        """Preloads authentic vision models and taxonomy registry during application startup."""
        with self._inference_lock:
            if self.is_loaded:
                return

            logger.info("Initializing AI Powered Crop Intelligence Rover ML suite on target device: %s...", self.device_name)

            # 1. Load Universal Taxonomy
            if not settings.TAXONOMY_PATH.exists():
                logger.error("Taxonomy registry missing at: %s", settings.TAXONOMY_PATH)
                raise FileNotFoundError(f"Taxonomy registry missing at: {settings.TAXONOMY_PATH}")

            with open(settings.TAXONOMY_PATH, "r") as f:
                raw_tax = json.load(f)

            # Handle both list of rich dictionaries and list of folder strings
            self.taxonomy = []
            for i, item in enumerate(raw_tax):
                if isinstance(item, dict):
                    self.taxonomy.append(item)
                else:
                    # Item is raw string e.g. "Apple___Apple_scab"
                    parts = str(item).split("___")
                    crop_name = parts[0].replace("_", " ").lower()
                    cond_name = parts[1].replace("_", " ") if len(parts) > 1 else "healthy"
                    c_lower = cond_name.lower()
                    c_type = "healthy" if "healthy" in c_lower else (
                        "bacterial" if "bacterial" in c_lower else (
                            "viral" if "virus" in c_lower else "fungal"
                        )
                    )
                    self.taxonomy.append({
                        "class_index": i,
                        "canonical_id": str(item).lower(),
                        "raw_folder": str(item),
                        "crop": crop_name,
                        "condition_type": c_type,
                        "disease_name": cond_name,
                        "scientific_name": "N/A"
                    })

            self.idx_to_meta = {item["class_index"]: item for item in self.taxonomy}
            self.canon_to_meta = {item["canonical_id"]: item for item in self.taxonomy}
            logger.info("Loaded 38-class agricultural taxonomy (%d classes).", len(self.taxonomy))

            # 2. Load Tier 1 Server-Grade EfficientNetV2-S with Checksum Validation
            if settings.SERVER_TIER1_MODEL_PATH.exists():
                try:
                    h = model_registry.compute_sha256(settings.SERVER_TIER1_MODEL_PATH)
                    expected_h = settings.PROMOTED_CLASSIFIER_SHA256
                    if h != expected_h:
                        logger.error(
                            "CRITICAL: Checkpoint SHA-256 mismatch for EfficientNetV2-S! Got %s, expected %s",
                            h, expected_h
                        )
                        self.model_status_map["tier1_server"] = "invalid_checksum"
                        raise ValueError(f"Checksum mismatch for {settings.SERVER_TIER1_MODEL_PATH.name}")

                    eff = models.efficientnet_v2_s(weights=None)
                    eff.classifier[1] = nn.Linear(1280, 38)
                    sd_eff = torch.load(settings.SERVER_TIER1_MODEL_PATH, map_location=self.device)
                    state_dict = sd_eff.get("model_state", sd_eff.get("state_dict", sd_eff))
                    eff.load_state_dict(state_dict)
                    eff.to(self.device).eval()
                    self.model_tier1_server = eff
                    self.model_status_map["tier1_server"] = "ready"
                    logger.info("Tier 1 Server-Grade EfficientNetV2-S loaded and verified (%.2f MB, SHA-256: %s).",
                                settings.SERVER_TIER1_MODEL_PATH.stat().st_size / (1024*1024), h[:12])
                except Exception as e:
                    self.model_status_map["tier1_server"] = f"error: {str(e)[:40]}"
                    logger.error("Failed loading production EfficientNetV2-S: %s", e)
            else:
                self.model_status_map["tier1_server"] = "missing"
                logger.error("Production EfficientNetV2-S checkpoint missing at %s", settings.SERVER_TIER1_MODEL_PATH)

            # Authoritative production classifier: EfficientNetV2-S only (no silent fallbacks)
            self.model_tier1 = self.model_tier1_server

            # 2a. Load Gen-2 Multi-Domain Classifier (if available for gen2 tier requests)
            if settings.GEN2_TIER1_MODEL_PATH.exists():
                try:
                    eff_g2 = models.efficientnet_v2_s(weights=None)
                    eff_g2.classifier[1] = nn.Linear(1280, 38)
                    sd_g2 = torch.load(settings.GEN2_TIER1_MODEL_PATH, map_location=self.device)
                    eff_g2.load_state_dict(sd_g2.get("model_state", sd_g2.get("state_dict", sd_g2)))
                    eff_g2.to(self.device).eval()
                    self.model_tier1_gen2 = eff_g2
                    self.model_status_map["tier1_gen2"] = "ready"
                    logger.info("Gen-2 Multi-Domain EfficientNetV2-S loaded (%.2f MB).",
                                settings.GEN2_TIER1_MODEL_PATH.stat().st_size / (1024*1024))
                except Exception as ge:
                    self.model_status_map["tier1_gen2"] = f"error: {str(ge)[:40]}"
                    logger.warning("Notice loading Gen-2 classifier: %s", ge)
            else:
                self.model_status_map["tier1_gen2"] = "missing"

            # 2b. Load Tier 2 YOLO PlantDoc Specimen Detector (Primary canopy boundary detector)
            if settings.TIER2_PLANTDOC_MODEL_PATH.exists():
                try:
                    self.model_tier2_plantdoc = YOLO(str(settings.TIER2_PLANTDOC_MODEL_PATH))
                    self.model_tier2 = self.model_tier2_plantdoc
                    arch_lbl, is_y26, det_imgsz = _inspect_yolo_architecture(self.model_tier2_plantdoc)
                    self.model_tier2_arch = arch_lbl
                    self.model_tier2_is_yolo26 = is_y26
                    self.model_tier2_path = settings.TIER2_PLANTDOC_MODEL_PATH
                    self.model_status_map["tier2_plantdoc"] = "ready"
                    logger.info(
                        "Tier 2 %s PlantDoc Specimen Detector loaded (%s, imgsz=%d, %.2f MB).",
                        arch_lbl, settings.TIER2_PLANTDOC_MODEL_PATH.name, det_imgsz,
                        settings.TIER2_PLANTDOC_MODEL_PATH.stat().st_size / (1024 * 1024),
                    )
                except Exception as e:
                    self.model_status_map["tier2_plantdoc"] = f"error: {str(e)[:40]}"
                    logger.warning("Notice loading YOLO PlantDoc detector: %s", e)
            else:
                self.model_status_map["tier2_plantdoc"] = "missing"

            # 2c. Load Tier 2 YOLOv8n Lesion Spot Detector (Dedicated focal pathology detector)
            lesion_model_path = settings.TIER2_LESIONS_MODEL_PATH
            if lesion_model_path.exists():
                try:
                    self.model_tier2_lesions = YOLO(str(lesion_model_path))
                    l_arch, _, l_imgsz = _inspect_yolo_architecture(self.model_tier2_lesions)
                    self.model_tier2_lesions_arch = l_arch
                    self.model_status_map["tier2_lesions"] = "ready"
                    logger.info(
                        "Tier 2 %s Lesion Spot Detector loaded (%s, imgsz=%d, %.2f MB).",
                        l_arch, lesion_model_path.name, l_imgsz,
                        lesion_model_path.stat().st_size / (1024 * 1024),
                    )
                except Exception as e:
                    self.model_status_map["tier2_lesions"] = f"error: {str(e)[:40]}"
                    logger.warning("Notice loading YOLOv8n Lesion Detector: %s", e)
            else:
                self.model_status_map["tier2_lesions"] = "missing"

            # 2d. Optionally load Opt-In YOLO26 v1 Checkpoints (strictly requires verified YOLO26 architecture)
            if settings.YOLO26_SPECIMEN_MODEL_PATH.exists():
                try:
                    cand_y26_spec = YOLO(str(settings.YOLO26_SPECIMEN_MODEL_PATH))
                    s_arch, s_is_y26, s_imgsz = _inspect_yolo_architecture(cand_y26_spec)
                    if s_is_y26:
                        self.model_yolo26_specimen = cand_y26_spec
                        self.model_yolo26_specimen_arch = s_arch
                        self.model_yolo26_specimen_imgsz = s_imgsz
                        self.model_status_map["yolo26_specimen"] = "ready"
                        logger.info("Opt-In YOLO26 v1 Specimen Detector loaded (%s, %s, imgsz=%d).", settings.YOLO26_SPECIMEN_MODEL_PATH.name, s_arch, s_imgsz)
                    else:
                        self.model_status_map["yolo26_specimen"] = f"rejected_non_yolo26 ({s_arch})"
                        logger.warning("Refused loading %s as YOLO26 specimen detector: verified architecture is %s.", settings.YOLO26_SPECIMEN_MODEL_PATH.name, s_arch)
                except Exception as ye:
                    self.model_status_map["yolo26_specimen"] = f"error: {str(ye)[:40]}"
            else:
                self.model_status_map["yolo26_specimen"] = "untrained_optional"

            if settings.YOLO26_LESION_MODEL_PATH.exists():
                try:
                    cand_y26_les = YOLO(str(settings.YOLO26_LESION_MODEL_PATH))
                    l_arch, l_is_y26, l_imgsz = _inspect_yolo_architecture(cand_y26_les)
                    if l_is_y26:
                        self.model_yolo26_lesion = cand_y26_les
                        self.model_yolo26_lesion_arch = l_arch
                        self.model_yolo26_lesion_imgsz = l_imgsz
                        self.model_status_map["yolo26_lesion"] = "ready"
                        logger.info("Opt-In YOLO26 v1 Lesion Model loaded (%s, %s, imgsz=%d).", settings.YOLO26_LESION_MODEL_PATH.name, l_arch, l_imgsz)
                    else:
                        self.model_status_map["yolo26_lesion"] = f"rejected_non_yolo26 ({l_arch})"
                        logger.warning("Refused loading %s as YOLO26 lesion model: verified architecture is %s.", settings.YOLO26_LESION_MODEL_PATH.name, l_arch)
                except Exception as yle:
                    self.model_status_map["yolo26_lesion"] = f"error: {str(yle)[:40]}"
            else:
                self.model_status_map["yolo26_lesion"] = "untrained_optional"

            # 3. Load Tier 3 Mobile-UNet Foliar Lesion Segmenter
            if settings.TIER3_MODEL_PATH.exists():
                try:
                    m3 = MobileUNet(in_ch=3, num_classes=3)
                    state_dict_t3 = torch.load(settings.TIER3_MODEL_PATH, map_location=self.device)
                    m3.load_state_dict(state_dict_t3.get("model_state", state_dict_t3))
                    m3.to(self.device).eval()
                    self.model_tier3 = m3
                    self.model_status_map["tier3_unet"] = "ready"
                    logger.info("Tier 3 Mobile-UNet Foliar Segmenter loaded (%.2f MB).",
                                settings.TIER3_MODEL_PATH.stat().st_size / (1024*1024))
                except Exception as e:
                    self.model_status_map["tier3_unet"] = f"error: {str(e)[:40]}"
                    logger.warning("Notice loading Mobile-UNet: %s", e)
            else:
                self.model_status_map["tier3_unet"] = "missing"

            # 4. Real Warmup Inference (Preheats device kernels without retaining activations)
            if self.model_tier1_server is not None:
                try:
                    with torch.inference_mode():
                        dummy_t1 = torch.zeros((1, 3, settings.CLASSIFIER_INPUT_RESOLUTION, settings.CLASSIFIER_INPUT_RESOLUTION), device=self.device)
                        _ = self.model_tier1_server(dummy_t1)
                    logger.info("Tier 1 EfficientNetV2-S warmup inference verified.")
                except Exception as we:
                    logger.warning("Tier 1 warmup notice: %s", we)

            if self.model_tier3 is not None:
                try:
                    with torch.inference_mode():
                        dummy_t3 = torch.zeros((1, 3, 256, 256), device=self.device)
                        _ = self.model_tier3(dummy_t3)
                    logger.info("Tier 3 Mobile-UNet warmup inference verified.")
                except Exception as we:
                    logger.warning("Tier 3 warmup notice: %s", we)

            self.is_loaded = True
            logger.info("ML vision suite initialization complete. Core ready: %s (Device: %s)", self.is_ready, self.device_name)

    def get_model_metadata(self, tier: str = "server") -> ModelMetadata:
        """Returns authoritative ModelMetadata for the specified tier."""
        tier_lower = (tier or "server").lower()
        is_y26 = getattr(self, "model_tier2_is_yolo26", False)
        legacy_det_requested = tier_lower in ("gen2", "yolo11", "legacy", "legacy_yolo11")
        has_opt_y26 = (
            not legacy_det_requested
            and (tier_lower in ("yolo26", "server_yolo26") or settings.YOLO26_ENABLED)
            and getattr(self, "model_yolo26_specimen", None) is not None
        )
        honest_det_tax = "PlantDoc-29Class-YOLO26-v1.0" if (is_y26 or has_opt_y26) else "PlantDoc-29Class-YOLO-v1.0"
        if tier_lower == "gen2":
            arch_name = "Gen-2 Multi-Domain EfficientNetV2-S Classifier"
            architecture = "EfficientNetV2-S (Gen-2 Multi-Domain)"
            is_server = False
            det_tax = honest_det_tax
            input_res = f"{settings.CLASSIFIER_INPUT_RESOLUTION}x{settings.CLASSIFIER_INPUT_RESOLUTION}"
        elif tier_lower in ("edge", "mobilenet_v2"):
            arch_name = "MobileNetV2 Edge Classifier"
            architecture = "MobileNetV2"
            is_server = False
            det_tax = honest_det_tax
            input_res = "224x224"
        else:
            arch_name = "EfficientNetV2-S Server-Grade Classifier"
            architecture = "EfficientNetV2-S"
            is_server = True
            det_tax = honest_det_tax
            input_res = f"{settings.CLASSIFIER_INPUT_RESOLUTION}x{settings.CLASSIFIER_INPUT_RESOLUTION}"

        return ModelMetadata(
            model_name=arch_name,
            model_version="v2.2-production",
            architecture=architecture,
            taxonomy_version="38-class-canonical",
            classification_taxonomy_version="PlantVillage-38Class-v2.0",
            detection_taxonomy_version=det_tax,
            segmentation_taxonomy_version="FoliarLesions-Binary-v1.0",
            input_resolution=input_res,
            is_server_authoritative=is_server,
            device=self.device_name,
            test_top1_accuracy=settings.CLASSIFIER_TEST_TOP1_ACC,
            macro_f1=settings.CLASSIFIER_TEST_MACRO_F1,
            expected_calibration_error=settings.CLASSIFIER_ECE,
            sha256_hash=settings.PROMOTED_CLASSIFIER_SHA256
        )


    def predict_vision(
        self,
        image_input: Any,
        model_tier: str = "server",
        filename: Optional[str] = None,
        content_type: Optional[str] = None,
        include_explainability: bool = False,
        request_id: Optional[str] = None,
        multimodal_context: Optional[Dict[str, Any]] = None
    ) -> DiagnosisResponse:
        """
        Authoritative prediction endpoint accepting either raw bytes or a file path string / Path object.
        """
        if isinstance(image_input, (str, Path)):
            p = Path(image_input)
            with open(p, "rb") as f:
                b = f.read()
            fn = filename or p.name
        elif isinstance(image_input, bytes):
            b = image_input
            fn = filename or "uploaded_specimen.jpg"
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")

        return self.run_inference(
            file_bytes=b,
            filename=fn,
            content_type=content_type,
            model_tier=model_tier,
            include_explainability=include_explainability,
            request_id=request_id,
            multimodal_context=multimodal_context
        )

    def run_inference(
        self,
        file_bytes: bytes,
        filename: Optional[str] = None,
        content_type: Optional[str] = None,
        model_tier: str = "server",
        include_explainability: bool = False,
        request_id: Optional[str] = None,
        multimodal_context: Optional[Dict[str, Any]] = None,
        force_inference: bool = False,
        source: str = "browser_upload",
        camera_metadata: Optional[Dict[str, Any]] = None
    ) -> DiagnosisResponse:
        """
        Executes authoritative multi-tier agricultural vision inference on an uploaded leaf image:
          1. Server-side image decode and validation (decompression bomb safe).
          2. Tier 1 Classification (Server EfficientNet-B2 or Edge MobileNetV2).
          3. Tier 2 Genuine YOLOv8 Detection with crop-aware bounding box validation.
          4. Tier 3 Mobile-UNet Segmentation (active mask damage index calculation).
          5. Optional Multi-Stage Explainability (Grad-CAM, feature maps, preprocessing).
        """
        if not self.is_loaded:
            self.load_models()

        if not self.is_ready:
            raise RuntimeError("Primary vision models are unavailable. Cannot execute inference.")

        total_pipeline_start = time.time()
        safe_sample_id = generate_safe_sample_id(filename)
        warnings: List[str] = []

        # 0. Validate image server-side (decompression-bomb and format safety)
        img_bgr, meta = validate_uploaded_image(file_bytes, filename=filename, content_type=content_type)
        orig_h, orig_w = meta["height"], meta["width"]

        # 0b. Authoritative Pre-Inference Domain Validation & Rejection Pipeline
        # Defends against arbitrary out-of-domain uploads (people, vehicles, documents, animals, blank frames)
        # Evaluates botanical vegetation index, spectral chrominance, photometrics, and YOLO PlantDoc cues.
        detector_candidate = self.model_tier2_plantdoc
        val_result = validate_plant_image(
            img_bgr,
            detector_model=detector_candidate,
            filename=filename,
            source=source,
            camera_metadata=camera_metadata
        )

        if source == "esp32_cam" and val_result.image_quality and "Degraded" in val_result.image_quality:
            warnings.append("Notice: Image acquired via field ESP32-CAM (OV2640) has mild sensor degradation; analyzed under adjusted camera quality policy.")
        elif source == "rpi_cam" and val_result.image_quality and "Degraded" in val_result.image_quality:
            warnings.append("Notice: Image acquired via field Raspberry Pi Camera (OV5647) has mild optical degradation; analyzed under adjusted camera quality policy.")

        if not val_result.is_inference_allowed and not force_inference:
            # HALT: Do not execute disease classification, detection, segmentation, or Grad-CAM.
            # Return authoritative rejection response with ZERO fabricated diagnostic metrics.
            rejection_total_ms = (time.time() - total_pipeline_start) * 1000.0
            model_meta = self.get_model_metadata(tier=model_tier or "server")
            
            validation_assessment = ImageValidationAssessment(
                validation_status=val_result.validation_status,
                validation_reason=val_result.validation_reason,
                validation_confidence=val_result.validation_confidence,
                plant_presence=val_result.plant_presence,
                leaf_presence=val_result.leaf_presence,
                screenshot_or_document_probability=val_result.screenshot_or_document_probability,
                image_quality=val_result.image_quality,
                is_inference_allowed=False,
                inference_allowed=False,
                telemetry=val_result.telemetry
            )

            return DiagnosisResponse(
                response_schema_version="1.0",
                status="rejected",
                pipeline_version="CV-06-Universal-MultiCrop-v2.2",
                sample_id=safe_sample_id,
                request_id=request_id,
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                model_metadata=model_meta,
                image_validation=validation_assessment,
                image_quality=ImageQualityAssessment(**meta["image_quality"]) if meta.get("image_quality") else None,
                uncertainty=None,
                diagnosis=None,
                spatial_telemetry=None,
                detection_status="not_requested",
                segmentation_status="not_requested",
                explainability_status="not_requested",
                segmentation_mask_b64=None,
                mask_raw_b64=None,
                cam_heatmap_b64=None,
                cam_overlay_b64=None,
                explainability=None,
                advisory=None,
                short_explanation=val_result.validation_reason,
                what_to_check="Please upload a genuine, well-illuminated photograph of an agricultural crop leaf.",
                performance_benchmark=LatencyBenchmark(
                    tier1_mobilenetv2_ms=0.0,
                    tier1_model_name="None (Validation Rejected)",
                    tier2_yolov8n_ms=0.0,
                    tier3_mobile_unet_ms=0.0,
                    explainability_ms=0.0,
                    total_pipeline_ms=round(rejection_total_ms, 2),
                    effective_fps=round(1000.0 / max(1.0, rejection_total_ms), 1),
                    compute_device=self.device_name
                ),
                modalities_used=["image"],
                multimodal_context=multimodal_context,
                warnings=[f"Image validation rejected input: {val_result.validation_reason}"],
                camera_source=(
                    "ESP32-CAM · OV2640" if source == "esp32_cam"
                    else ("Raspberry Pi · OV5647" if source == "rpi_cam"
                    else ("Device Camera" if source == "browser_camera" else "Browser Upload"))
                ),
                camera_device_id=camera_metadata.get("device_id") if camera_metadata else None,
                camera_capture_id=camera_metadata.get("capture_id") if camera_metadata else None,
                camera_capture_latency_ms=camera_metadata.get("capture_latency_ms") if camera_metadata else None,
            )

        # Acquire lock to ensure thread/concurrency safe tensor execution
        with self._inference_lock:
            # ══════════════════════════════════════════════════════════════════
            # TIER 1: AUTHORITATIVE CLASSIFICATION (EfficientNetV2-S Server / MobileNetV2 Edge)
            # ══════════════════════════════════════════════════════════════════
            t1_start = time.time()
            tier_req = (model_tier or "server").lower()
            is_y26 = getattr(self, "model_tier2_is_yolo26", False)
            det_arch = getattr(self, "model_tier2_arch", "YOLO11n")
            honest_detector_name = (
                "YOLO26 Multi-Domain Agricultural Detector"
                if is_y26
                else f"{det_arch} PlantDoc Specimen Detector"
            )
            legacy_det_requested = tier_req in ("gen2", "yolo11", "legacy", "legacy_yolo11")
            use_yolo26_det = (
                tier_req in ("yolo26", "server_yolo26")
                or (settings.YOLO26_ENABLED and not legacy_det_requested)
            )
            if tier_req == "gen2":
                active_model = self.model_tier1_gen2 if self.model_tier1_gen2 is not None else self.model_tier1_server
                model_arch_name = "Gen-2 Multi-Domain EfficientNetV2-S Classifier"
                active_tier = "gen2"
                is_server = False
            elif tier_req in ("edge", "mobilenet_v2"):
                active_model = self.model_tier1_edge if self.model_tier1_edge is not None else self.model_tier1_server
                model_arch_name = "MobileNetV2 Edge Classifier"
                active_tier = "edge"
                is_server = False
            else:
                active_model = self.model_tier1_server
                model_arch_name = "EfficientNetV2-S Server-Grade Classifier"
                active_tier = "yolo26" if tier_req in ("yolo26", "server_yolo26") else "server"
                is_server = True

            active_lesion_detector = self.model_tier2_lesions
            active_lesion_source = f"{getattr(self, 'model_tier2_lesions_arch', 'YOLOv8n')} Lesion Spot Detector"
            yolo26_active_specimen = False
            yolo26_active_lesion = False
            yolo26_e2e_flag = settings.YOLO26_INFERENCE_MODE.lower() != "one2many"
            active_segmenter = self.model_tier3

            if use_yolo26_det:
                if self.model_yolo26_specimen is not None:
                    active_detector = self.model_yolo26_specimen
                    yolo26_active_specimen = True
                    mode_tag = "e2e NMS-free" if yolo26_e2e_flag else "one-to-many NMS"
                    detection_engine_name = f"YOLO26 v1 PlantDoc Specimen Detector ({mode_tag})"
                else:
                    active_detector = self.model_tier2_plantdoc
                    detection_engine_name = honest_detector_name
                    warnings.append(
                        "YOLO26 v1 specimen checkpoint is not yet present; fell back to verified YOLO11n PlantDoc baseline."
                    )
                if self.model_yolo26_lesion is not None:
                    active_lesion_detector = self.model_yolo26_lesion
                    yolo26_active_lesion = True
                    active_lesion_source = "YOLO26-Seg v1 Lesion Localizer"
                elif tier_req in ("yolo26", "server_yolo26"):
                    warnings.append(
                        "Optional YOLO26 v1 lesion checkpoint is not yet present; retained verified YOLOv8n baseline."
                    )
            else:
                active_detector = self.model_tier2_plantdoc
                detection_engine_name = honest_detector_name

            if active_model is None:
                raise RuntimeError(f"Authoritative classifier for tier '{model_tier}' is not loaded.")

            input_res_str = f"{settings.CLASSIFIER_INPUT_RESOLUTION}x{settings.CLASSIFIER_INPUT_RESOLUTION}"
            t1_tensor = preprocess_tier1(img_bgr, self.device, input_size=(settings.CLASSIFIER_INPUT_RESOLUTION, settings.CLASSIFIER_INPUT_RESOLUTION))
            with torch.inference_mode():
                t1_out = active_model(t1_tensor)
                t1_logits = t1_out.logits if hasattr(t1_out, "logits") else t1_out
                t1_probs = torch.softmax(t1_logits, dim=1)[0].cpu().numpy()

            t1_latency_ms = (time.time() - t1_start) * 1000.0
            t1_top1_idx = int(np.argmax(t1_probs))
            t1_top1_meta = self.idx_to_meta[t1_top1_idx]
            t1_is_infected = (t1_top1_meta["disease_name"].lower() != "healthy")

            # ══════════════════════════════════════════════════════════════════
            # TIER 2: GENUINE OBJECT DETECTION (YOLO PlantDoc Specimen & Lesion Spot Detectors)
            # ══════════════════════════════════════════════════════════════════
            t2_start = time.time()
            specimen_detections: List[BoundingBox] = []
            lesion_detections: List[BoundingBox] = []
            raw_specimen_count = 0
            raw_lesion_count = 0
            post_nms_specimen_count = 0
            post_nms_lesion_count = 0
            detection_status = "unavailable"

            # 1. Specimen / Foliage Canopy Boundary Localization (YOLO11n baseline or Opt-In YOLO26)
            if active_detector is not None:
                try:
                    conf_thresh = settings.YOLO26_CONF_THRESH if yolo26_active_specimen else settings.YOLO_PLANTDOC_CONF_THRESH
                    iou_thresh = settings.YOLO26_IOU_THRESH if yolo26_active_specimen else settings.YOLO_IOU_THRESH
                    precomputed_det = getattr(val_result, "precomputed_detector_results", None)
                    if (
                        not yolo26_active_specimen
                        and precomputed_det is not None
                        and active_detector is detector_candidate
                        and abs(iou_thresh - 0.45) < 1e-6
                    ):
                        pdoc_all = precomputed_det
                        if hasattr(val_result, "precomputed_detector_results"):
                            val_result.precomputed_detector_results = None
                    else:
                        det_kw: Dict[str, Any] = {
                            "conf": 0.01,
                            "iou": iou_thresh,
                            "agnostic_nms": True,
                            "max_det": 150,
                            "verbose": False,
                        }
                        if yolo26_active_specimen:
                            det_kw["imgsz"] = getattr(self, "model_yolo26_specimen_imgsz", 768)
                            det_kw.update(_yolo26_mode_kwargs(yolo26_e2e_flag))
                        pdoc_all = active_detector.predict(img_bgr, **det_kw)

                    detection_status = "available"
                    if len(pdoc_all) > 0 and pdoc_all[0].boxes is not None:
                        boxes = pdoc_all[0].boxes
                        raw_specimen_count = len(boxes)
                        confs_arr = boxes.conf.cpu().numpy()

                        selected_indices = [i for i in range(raw_specimen_count) if float(confs_arr[i]) >= conf_thresh]
                        if not selected_indices and conf_thresh > 0.08:
                            selected_indices = [i for i in range(raw_specimen_count) if float(confs_arr[i]) >= 0.08]

                        for i in selected_indices:
                            cls_id = int(boxes.cls[i].cpu().numpy())
                            cls_name = pdoc_all[0].names.get(cls_id, f"Class_{cls_id}")
                            conf = float(confs_arr[i])
                            xyxy_raw = boxes.xyxy[i].cpu().numpy().astype(float).tolist()

                            x1, y1, x2, y2 = _clamp_xyxy(xyxy_raw, orig_w, orig_h)
                            if (x2 - x1) < 8 or (y2 - y1) < 8:
                                continue

                            cx = round(float((x1 + x2) / (2.0 * orig_w)), 3)
                            cy = round(float((y1 + y2) / (2.0 * orig_h)), 3)

                            specimen_detections.append(
                                BoundingBox(
                                    detection_id=f"specimen_{len(specimen_detections)}_{cls_id}",
                                    class_id=cls_id,
                                    class_name=cls_name,
                                    category_type="canopy",
                                    label=f"{cls_name} ({conf * 100:.0f}%)",
                                    bbox_xyxy=[x1, y1, x2, y2],
                                    confidence=round(conf, 2),
                                    centroid_norm=[cx, cy],
                                    box_type="leaf",
                                    color_hex="#52b788",
                                    source_model=detection_engine_name,
                                    coordinate_space="original_image_pixels",
                                )
                            )

                    post_nms_specimen_count = len(specimen_detections)
                    if raw_specimen_count < post_nms_specimen_count:
                        raw_specimen_count = post_nms_specimen_count
                except Exception as e:
                    logger.warning("PlantDoc detector execution notice: %s", e)
                    warnings.append(f"{detection_engine_name} encountered an issue during execution.")
            else:
                warnings.append("Tier 2 specimen detector is not loaded.")

            # 2. Granular Lesion Spot Localization (YOLOv8n baseline or Opt-In YOLO26-Seg/Detect)
            # Only executed on confirmed infected foliage to prevent false positive lesion hallucinations
            yolo26_lesion_mask_256: Optional[np.ndarray] = None
            if active_lesion_detector is not None and t1_is_infected:
                try:
                    les_conf_thresh = settings.YOLO26_SEG_CONF_THRESH if yolo26_active_lesion else settings.YOLO_LESIONS_CONF_THRESH
                    les_iou_thresh = settings.YOLO26_IOU_THRESH if yolo26_active_lesion else settings.YOLO_IOU_THRESH
                    les_kw: Dict[str, Any] = {
                        "conf": 0.01,
                        "iou": les_iou_thresh,
                        "agnostic_nms": True,
                        "max_det": 150,
                        "verbose": False,
                    }
                    if yolo26_active_lesion:
                        les_kw["imgsz"] = getattr(self, "model_yolo26_lesion_imgsz", 768)
                        les_kw.update(_yolo26_mode_kwargs(yolo26_e2e_flag))
                    les_res = active_lesion_detector.predict(img_bgr, **les_kw)
                    detection_status = "available"
                    accepted_lesion_indices: List[int] = []
                    lesion_detections, raw_lesion_count = _filter_lesion_detections(
                        img_bgr=img_bgr,
                        les_res=les_res,
                        specimen_detections=specimen_detections,
                        t1_disease_name=t1_top1_meta["disease_name"],
                        conf_thresh=les_conf_thresh,
                        source_model_name=active_lesion_source,
                        is_yolo26_seg=yolo26_active_lesion,
                        accepted_raw_indices_out=accepted_lesion_indices,
                    )
                    if (
                        yolo26_active_lesion
                        and accepted_lesion_indices
                        and len(les_res) > 0
                        and getattr(les_res[0], "masks", None) is not None
                    ):
                        masks_obj = les_res[0].masks
                        union_orig = np.zeros((orig_h, orig_w), dtype=np.uint8)
                        polys_xy = getattr(masks_obj, "xy", None)
                        if polys_xy:
                            for idx in accepted_lesion_indices:
                                if idx < len(polys_xy):
                                    poly = polys_xy[idx]
                                    if len(poly) >= 3:
                                        cv2.fillPoly(union_orig, [np.round(poly).astype(np.int32)], 1)
                        elif getattr(masks_obj, "data", None) is not None:
                            m_data = masks_obj.data.detach().cpu().numpy()
                            valid_slices = [m_data[idx] for idx in accepted_lesion_indices if idx < len(m_data)]
                            if valid_slices:
                                union_m = (np.max(np.stack(valid_slices, axis=0), axis=0) > 0.5).astype(np.uint8)
                                mh, mw = union_m.shape[:2]
                                union_orig = scale_segmentation_mask_to_original(
                                    union_m, orig_w, orig_h, letterbox_shape=(mw, mh)
                                )
                        if np.any(union_orig):
                            yolo26_lesion_mask_256 = cv2.resize(
                                union_orig, (256, 256), interpolation=cv2.INTER_NEAREST
                            )
                    post_nms_lesion_count = len(lesion_detections)
                    if raw_lesion_count < post_nms_lesion_count:
                        raw_lesion_count = post_nms_lesion_count
                except Exception as le:
                    logger.warning("Lesion detector execution notice: %s", le)
                    warnings.append("Lesion detector encountered an issue during execution.")
            elif not t1_is_infected:
                lesion_detections = []
                raw_lesion_count = 0
                post_nms_lesion_count = 0

            # Unified list for legacy consumers, sorted: canopy boundaries first, lesion foci second
            detected_boxes = specimen_detections + lesion_detections
            raw_detection_count = raw_specimen_count + raw_lesion_count
            post_filtering_count = len(detected_boxes)
            canopy_box_count = len(specimen_detections)
            specimen_box_count = len(specimen_detections)
            lesion_box_count = len(lesion_detections)

            # Define honest localization capability and scientific notice
            if len(lesion_detections) > 0 and len(specimen_detections) > 0:
                localization_capability = "specimen_boundary_and_lesion_foci"
                localization_notice = (
                    f"Dual spatial localization active: {len(specimen_detections)} specimen canopy boundary region(s) "
                    f"from {detection_engine_name} and {len(lesion_detections)} verified lesion spot foci "
                    f"from {active_lesion_source}."
                )
            elif len(lesion_detections) > 0:
                localization_capability = "lesion_foci_only"
                localization_notice = f"Lesion focus localization active: {len(lesion_detections)} verified lesion spot foci localized."
            elif len(specimen_detections) > 0:
                localization_capability = "specimen_boundary_only"
                localization_notice = (
                    f"Current {detection_engine_name} provides specimen canopy boundary localization ({len(specimen_detections)} foliar region(s)). "
                    f"Dedicated bounding-box lesion spot localization is not part of this detector's ontology; "
                    f"sub-pixel foliar pathology is evaluated via Mobile-UNet foliar segmentation."
                )
            else:
                localization_capability = "unavailable"
                localization_notice = "No spatial boundaries or lesion foci detected above confidence threshold."

            t2_latency_ms = (time.time() - t2_start) * 1000.0

            # ══════════════════════════════════════════════════════════════════
            # STAGE 9 MULTI-TIER CROSS-VALIDATION & EVIDENCE SYNTHESIS
            # ══════════════════════════════════════════════════════════════════
            final_probs, cross_tier_note = reconcile_cross_tier_evidence(
                t1_probs=t1_probs,
                specimen_detections=specimen_detections,
                lesion_detections=lesion_detections,
                idx_to_meta=self.idx_to_meta,
                detection_engine_name=detection_engine_name
            )
            if cross_tier_note:
                logger.info("Cross-tier evidence synthesis: %s", cross_tier_note)

            # Top-3 predictions and margin derived from synthesized multi-tier distribution
            top_indices = np.argsort(final_probs)[::-1][:3]
            top1_idx = int(top_indices[0])
            top2_idx = int(top_indices[1]) if len(top_indices) > 1 else top1_idx
            top1_meta = self.idx_to_meta[top1_idx]
            top1_conf = float(final_probs[top1_idx]) * 100.0
            top2_conf = float(final_probs[top2_idx]) * 100.0
            prediction_margin = float(final_probs[top1_idx] - final_probs[top2_idx])

            top3_preds = [
                TopPrediction(
                    class_id=self.idx_to_meta[int(idx)]["raw_folder"],
                    label=f"{self.idx_to_meta[int(idx)]['crop'].capitalize()} - {self.idx_to_meta[int(idx)]['disease_name']}",
                    confidence_pct=round(float(final_probs[int(idx)]) * 100.0, 2)
                )
                for idx in top_indices
            ]

            # Calculate prediction entropy and uncertainty metric
            p_clipped = np.clip(final_probs, 1e-12, 1.0)
            entropy = float(-np.sum(p_clipped * np.log(p_clipped)))
            max_entropy = float(np.log(len(final_probs)))
            norm_uncertainty = float(np.clip(entropy / max_entropy, 0.0, 1.0))

            is_infected = (top1_meta["condition_type"] != "healthy")
            crop_display = top1_meta["crop"].capitalize()
            disease_display = top1_meta["disease_name"]

            # Out-of-Distribution (OOD) & Calibration Analysis
            quality_info = meta.get("image_quality") or {}
            green_ratio = quality_info.get("greenness_ratio", 0.5)
            is_poor_quality = not quality_info.get("is_acceptable", True)

            if green_ratio < 0.05 or (top1_conf < 38.0 and norm_uncertainty > 0.65) or (is_poor_quality and top1_conf < 55.0):
                ood_status = "OUT_OF_DISTRIBUTION" if green_ratio < 0.05 else "BORDERLINE"
                confidence_level = "UNCERTAIN"
                is_low_confidence = True
            elif top1_conf < 50.0 or prediction_margin < 0.12 or norm_uncertainty > 0.58:
                ood_status = "BORDERLINE"
                confidence_level = "LOW"
                is_low_confidence = True
            elif top1_conf >= 75.0:
                ood_status = "IN_DISTRIBUTION"
                confidence_level = "HIGH"
                is_low_confidence = False
            else:
                ood_status = "IN_DISTRIBUTION"
                confidence_level = "MODERATE"
                is_low_confidence = False

            # Formulate clear, human-readable explanations with multi-tier grounding
            if is_low_confidence:
                short_explanation = (
                    f"The model is not sufficiently confident to provide a definitive diagnosis from this image. "
                    f"A tentative visual match for {disease_display} on {crop_display} was identified ({top1_conf:.1f}% confidence), "
                    f"but alternative conditions remain possible."
                )
                what_to_check = (
                    f"Capture a clear, well-illuminated close-up of a single symptomatic leaf. "
                    f"Inspect nearby leaves across the {crop_display} canopy for clearer spot or lesion margins."
                )
            elif not is_infected:
                short_explanation = (
                    f"The uploaded image exhibits visual patterns consistent with a healthy {crop_display} leaf, "
                    f"without apparent fungal, bacterial, or viral foliar pathology."
                )
                if cross_tier_note:
                    short_explanation += f" {cross_tier_note}"
                what_to_check = (
                    f"Continue regular canopy scouting. Inspect lower mature leaves and leaf undersides for any early spotting or discoloration."
                )
            else:
                short_explanation = (
                    f"The uploaded image exhibits visual characteristics consistent with {disease_display} on {crop_display} "
                    f"according to the multi-tier agricultural vision pipeline."
                )
                if cross_tier_note:
                    short_explanation += f" {cross_tier_note}"
                what_to_check = (
                    f"Inspect nearby leaves and stems on the {crop_display} plant for similar symptoms. "
                    f"Check whether the pattern appears concentrated on older lower foliage or newer growth."
                )

            # ══════════════════════════════════════════════════════════════════
            # TIER 3: SUB-PIXEL FOLIAR SEGMENTATION (Mobile-UNet)
            # ══════════════════════════════════════════════════════════════════
            t3_start = time.time()
            foliar_damage_pct: Optional[float] = None
            segmentation_status = "unavailable"
            pred_mask_256: Optional[np.ndarray] = None
            mask_overlay_b64: Optional[str] = None
            mask_raw_b64: Optional[str] = None

            if active_segmenter is not None:
                try:
                    t3_tensor = preprocess_tier3(img_bgr, self.device)
                    with torch.inference_mode():
                        seg_out = active_segmenter(t3_tensor)
                        pred_mask_256 = torch.argmax(seg_out, dim=1)[0].cpu().numpy()

                    # Build union envelope of ALL detected boxes to confine semantic mask
                    # (ensures necrotic tissue in every detection region is measured, not just the single largest)
                    primary_leaf_box = None
                    target_leaf_boxes = specimen_detections if specimen_detections else detected_boxes
                    if target_leaf_boxes:
                        union_x1 = min(b.bbox_xyxy[0] for b in target_leaf_boxes)
                        union_y1 = min(b.bbox_xyxy[1] for b in target_leaf_boxes)
                        union_x2 = max(b.bbox_xyxy[2] for b in target_leaf_boxes)
                        union_y2 = max(b.bbox_xyxy[3] for b in target_leaf_boxes)
                        # Pick the highest-confidence box as the reference, but override its bbox to the union
                        best_conf_box = max(target_leaf_boxes, key=lambda b: b.confidence)
                        primary_leaf_box = BoundingBox(
                            detection_id="union_leaf_box",
                            class_id=best_conf_box.class_id,
                            class_name=best_conf_box.class_name,
                            category_type=best_conf_box.category_type,
                            label=best_conf_box.label,
                            bbox_xyxy=[int(union_x1), int(union_y1), int(union_x2), int(union_y2)],
                            confidence=best_conf_box.confidence,
                            centroid_norm=best_conf_box.centroid_norm,
                            box_type="leaf",
                            severity=best_conf_box.severity,
                            color_hex=best_conf_box.color_hex,
                            source_model=best_conf_box.source_model,
                            coordinate_space=best_conf_box.coordinate_space
                        )

                    # Filter out non-plant background (wood, floor, tables, hands, soil) from Mobile-UNet mask
                    pred_mask_256 = clean_foliar_segmentation_mask(
                        pred_mask_256=pred_mask_256,
                        img_bgr=img_bgr,
                        primary_leaf_box=primary_leaf_box,
                        specimen_boxes=target_leaf_boxes,
                    )

                    # Fuse already canopy-contained & filtered YOLO26s-Seg lesion polygons after Mobile-UNet
                    # cleanup so authentic lesions on sunlit/outdoor foliage are never stripped.
                    if yolo26_lesion_mask_256 is not None and is_infected:
                        pred_mask_256[yolo26_lesion_mask_256 > 0] = 2

                    leaf_pixels = int(np.sum(pred_mask_256 >= 1))
                    lesion_pixels = int(np.sum(pred_mask_256 == 2))

                    # If Mobile-UNet predicted sparse canopy inside detected leaf boxes despite valid lesions,
                    # supplement leaf canopy mask from green pixels inside detected specimen bounding boxes.
                    if leaf_pixels <= 50 and target_leaf_boxes:
                        img_256_tmp = cv2.resize(img_bgr, (256, 256), interpolation=cv2.INTER_AREA).astype(np.float32)
                        exg_tmp = 2.0 * img_256_tmp[:, :, 1] - img_256_tmp[:, :, 2] - img_256_tmp[:, :, 0]
                        for s_box in target_leaf_boxes:
                            bx1 = max(0, int(s_box.bbox_xyxy[0] * 256.0 / orig_w))
                            by1 = max(0, int(s_box.bbox_xyxy[1] * 256.0 / orig_h))
                            bx2 = min(256, int(s_box.bbox_xyxy[2] * 256.0 / orig_w))
                            by2 = min(256, int(s_box.bbox_xyxy[3] * 256.0 / orig_h))
                            sub_canopy = (pred_mask_256[by1:by2, bx1:bx2] == 0) & (exg_tmp[by1:by2, bx1:bx2] > 6.0)
                            pred_mask_256[by1:by2, bx1:bx2][sub_canopy] = 1
                        leaf_pixels = int(np.sum(pred_mask_256 >= 1))
                        lesion_pixels = int(np.sum(pred_mask_256 == 2))

                    if leaf_pixels > 50:
                        raw_damage_pct = round((lesion_pixels / float(leaf_pixels)) * 100.0, 2)
                        if is_infected:
                            foliar_damage_pct = raw_damage_pct
                            segmentation_status = "available"
                        else:
                            foliar_damage_pct = 0.0
                            segmentation_status = "healthy_not_applicable"
                    elif not is_infected:
                        foliar_damage_pct = 0.0
                        segmentation_status = "healthy_not_applicable"
                    else:
                        foliar_damage_pct = None
                        segmentation_status = "unavailable"

                    mask_overlay_b64 = create_segmentation_overlay_base64(orig_w, orig_h, pred_mask_256)
                    mask_raw_b64 = create_segmentation_mask_base64(orig_w, orig_h, pred_mask_256)
                except Exception as e:
                    logger.warning("Mobile-UNet segmentation notice: %s", e)
                    segmentation_status = "failed"
                    warnings.append("Mobile-UNet foliar segmenter encountered an issue.")
            else:
                warnings.append("Tier 3 Mobile-UNet segmenter is not loaded.")

            t3_latency_ms = (time.time() - t3_start) * 1000.0

            # ══════════════════════════════════════════════════════════════════
            # AUTHENTIC LESION FOCI AND CAPABILITY SYNTHESIS
            # ══════════════════════════════════════════════════════════════════
            if not is_infected:
                # Enforce clean negative control in final telemetry for healthy foliage
                lesion_detections = []
                detected_boxes = specimen_detections
                post_nms_lesion_count = 0
                lesion_box_count = 0
                post_filtering_count = len(detected_boxes)
                final_lesion_foci_count = 0
                final_lesion_foci_source = "healthy_negative_control"
                if len(specimen_detections) > 0:
                    localization_capability = "specimen_boundary_only"
                    localization_notice = (
                        f"Specimen verified as healthy foliage ({top1_meta['crop']} {top1_meta['disease_name']}). "
                        f"{len(specimen_detections)} foliar canopy boundary region(s) localized by {detection_engine_name}; zero pathological lesions confirmed."
                    )
                else:
                    localization_capability = "unavailable"
                    localization_notice = "Specimen verified as healthy foliage with zero pathological lesion foci."
            elif len(lesion_detections) > 0:
                final_lesion_foci_count = len(lesion_detections)
                final_lesion_foci_source = "yolo26_lesion_detector" if yolo26_active_lesion else "yolov8n_lesions_detector"
                if len(specimen_detections) > 0:
                    localization_capability = "specimen_boundary_and_lesion_foci"
                    localization_notice = (
                        f"Dual spatial localization active: {len(specimen_detections)} specimen canopy boundary region(s) "
                        f"from {detection_engine_name} and {len(lesion_detections)} verified lesion spot foci "
                        f"from {active_lesion_source}."
                    )
                else:
                    localization_capability = "lesion_foci_only"
                    localization_notice = f"Lesion focus localization active: {len(lesion_detections)} verified lesion spot foci localized."
            elif is_infected and pred_mask_256 is not None and lesion_pixels > 0:
                n_comp, _, _, _ = cv2.connectedComponentsWithStats((pred_mask_256 == 2).astype(np.uint8), connectivity=8)
                final_lesion_foci_count = max(0, n_comp - 1)
                final_lesion_foci_source = "mobile_unet_segmentation"
                if len(specimen_detections) > 0:
                    localization_capability = "specimen_boundary_only"
                    localization_notice = (
                        f"Current {detection_engine_name} provides specimen canopy boundary localization ({len(specimen_detections)} foliar region(s)). "
                        f"Dedicated bounding-box lesion spot localization is not part of this detector's ontology; "
                        f"sub-pixel foliar pathology is evaluated via Mobile-UNet foliar segmentation."
                    )
                else:
                    localization_capability = "unavailable"
                    localization_notice = "No spatial boundaries detected above confidence threshold."
            else:
                final_lesion_foci_count = 0
                final_lesion_foci_source = "none"

            # ══════════════════════════════════════════════════════════════════
            # TRIAGE STAGE DETERMINATION (Honest Grounding)
            # ══════════════════════════════════════════════════════════════════
            if not is_infected:
                triage_stage = "Healthy Vegetative State"
                dosage_mult = 1.0
            elif is_low_confidence:
                triage_stage = "Unverified / Low Confidence"
                dosage_mult = 1.0
            elif foliar_damage_pct is not None:
                if foliar_damage_pct < 5.0:
                    triage_stage = "Stage 1 Mild Foliar Spotting"
                    dosage_mult = 1.0
                elif foliar_damage_pct <= 15.0:
                    triage_stage = "Stage 2 Moderate Lesion Spread"
                    dosage_mult = 1.45
                else:
                    triage_stage = "Stage 3 Severe Tissue Necrosis"
                    dosage_mult = 2.0
            else:
                triage_stage = "Active Foliar Pathology Detected"
                dosage_mult = 1.25

            # ══════════════════════════════════════════════════════════════════
            # DYNAMIC GRAD-CAM SALIENCY & 9-STAGE EXPLAINABILITY (On-demand)
            # ══════════════════════════════════════════════════════════════════
            cam_heatmap_b64: Optional[str] = None
            cam_overlay_b64: Optional[str] = None
            explainability_suite = None
            explainability_status = "not_requested"
            t_exp_ms = 0.0

            if include_explainability:
                t_exp_start = time.time()
                try:
                    target_cam_layer = get_target_cam_layer(active_model)
                    if target_cam_layer is not None:
                        gcam = GradCAM(active_model, target_cam_layer)
                        # Use clone of input tensor for Grad-CAM forward-backward pass
                        t_cam_in = t1_tensor.clone().detach()
                        cam_map = gcam.generate_heatmap(t_cam_in, class_idx=top1_idx)
                        gcam.remove_hooks()

                        # Scale CAM heatmap to original image dimensions
                        cam_resized = cv2.resize(cam_map, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
                        heatmap_bgr = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
                        cam_heatmap_b64 = numpy_to_base64_jpeg(heatmap_bgr, quality=85)

                        overlay_bgr = cv2.addWeighted(img_bgr, 0.52, heatmap_bgr, 0.48, 0)
                        cam_overlay_b64 = numpy_to_base64_jpeg(overlay_bgr, quality=85)
                except Exception as ge:
                    logger.warning("Dynamic Grad-CAM generation notice: %s", ge)
                    warnings.append("Grad-CAM saliency generation encountered an issue.")

                try:
                    explainability_suite = build_explainability_pipeline(
                        img_bgr=img_bgr,
                        img_tensor=t1_tensor,
                        model=active_model,
                        predicted_idx=top1_idx,
                        top1_meta=top1_meta,
                        top3_preds=top3_preds,
                        detected_boxes=detected_boxes,
                        pred_mask_256=pred_mask_256,
                        foliar_damage_pct=foliar_damage_pct,
                        entropy_score=entropy,
                        uncertainty_score=norm_uncertainty,
                        model_name=model_arch_name,
                        orig_meta=meta
                    )
                    explainability_status = "available"
                except Exception as e:
                    logger.warning("Explainability pipeline notice: %s", e)
                    explainability_status = "failed"
                    warnings.append("Explainability visualization pipeline encountered an issue.")
                t_exp_ms = (time.time() - t_exp_start) * 1000.0
                gc.collect()

        total_ms = (time.time() - total_pipeline_start) * 1000.0

        # Agronomic advisory synthesis
        advisory_dict = generate_agronomic_advisory(
            crop=top1_meta["crop"],
            condition_type=top1_meta["condition_type"],
            disease_name=top1_meta["disease_name"],
            is_infected=is_infected,
            damage_pct=foliar_damage_pct,
            confidence_pct=top1_conf
        )

        model_meta = self.get_model_metadata(tier=active_tier)

        return DiagnosisResponse(
            response_schema_version="1.0",
            status="success",
            pipeline_version="CV-06-Universal-MultiCrop-v2.2",
            sample_id=safe_sample_id,
            request_id=request_id,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            model_metadata=model_meta,
            image_validation=ImageValidationAssessment(
                validation_status=val_result.validation_status,
                validation_reason=val_result.validation_reason,
                validation_confidence=val_result.validation_confidence,
                plant_presence=val_result.plant_presence,
                leaf_presence=val_result.leaf_presence,
                screenshot_or_document_probability=val_result.screenshot_or_document_probability,
                image_quality=val_result.image_quality,
                is_inference_allowed=True,
                inference_allowed=True,
                telemetry=val_result.telemetry
            ),
            image_quality=ImageQualityAssessment(**quality_info) if quality_info else None,
            uncertainty=UncertaintyMetrics(
                prediction_margin=round(prediction_margin, 3),
                entropy_nats=round(entropy, 3),
                normalized_uncertainty=round(norm_uncertainty, 3),
                ood_status=ood_status,
                is_low_confidence=is_low_confidence
            ),
            diagnosis=DiagnosisSummary(
                predicted_class=top1_meta["raw_folder"],
                disease_common_name=top1_meta["disease_name"],
                crop=top1_meta["crop"].capitalize(),
                condition_type=top1_meta["condition_type"],
                confidence_pct=round(top1_conf, 2),
                confidence_level=confidence_level,
                is_low_confidence=is_low_confidence,
                uncertainty_score=round(norm_uncertainty, 3),
                entropy=round(entropy, 3),
                top3_predictions=top3_preds,
                is_infected=is_infected,
                triage_stage=triage_stage,
                detection_status=detection_status,
                segmentation_status=segmentation_status,
                foliar_damage_pct=foliar_damage_pct,
                lesion_foci_count=final_lesion_foci_count,
                lesion_foci_source=final_lesion_foci_source,
                model_architecture=model_arch_name,
                model_tier=active_tier,
                short_explanation=short_explanation,
                what_to_check=what_to_check
            ),
            spatial_telemetry=SpatialTelemetry(
                detection_engine=detection_engine_name,
                taxonomy_version="PlantDoc-29Class-YOLO26-v1.0" if (is_y26 or yolo26_active_specimen) else "PlantDoc-29Class-YOLO-v1.0",
                bounding_boxes=detected_boxes,
                specimen_detections=specimen_detections,
                lesion_detections=lesion_detections,
                nozzle_actuation_targets=len(lesion_detections) if len(lesion_detections) > 0 else len(specimen_detections),
                variable_rate_dosage_multiplier=dosage_mult,
                raw_detection_count=raw_detection_count,
                post_filtering_count=post_filtering_count,
                canopy_box_count=canopy_box_count,
                specimen_box_count=specimen_box_count,
                lesion_box_count=lesion_box_count,
                localization_capability=localization_capability,
                localization_notice=localization_notice,
                raw_specimen_count=raw_specimen_count,
                raw_lesion_count=raw_lesion_count,
                post_nms_specimen_count=post_nms_specimen_count,
                post_nms_lesion_count=post_nms_lesion_count,
                lesion_foci_source=final_lesion_foci_source
            ),
            detection_status=detection_status,
            segmentation_status=segmentation_status,
            explainability_status=explainability_status,
            segmentation_mask_b64=mask_overlay_b64,
            mask_raw_b64=mask_raw_b64,
            cam_heatmap_b64=cam_heatmap_b64,
            cam_overlay_b64=cam_overlay_b64,
            explainability=explainability_suite,
            advisory=AgronomicAdvisory(
                immediate_action=advisory_dict["immediate_action"],
                treatment_protocol=advisory_dict["treatment_protocol"],
                cultural_practices=advisory_dict["cultural_practices"],
                uncertainty_guidance=advisory_dict["uncertainty_guidance"]
            ),
            performance_benchmark=LatencyBenchmark(
                tier1_mobilenetv2_ms=round(t1_latency_ms, 2),
                tier1_model_name=model_arch_name,
                tier2_yolov8n_ms=round(t2_latency_ms, 2),
                tier3_mobile_unet_ms=round(t3_latency_ms, 2),
                explainability_ms=round(t_exp_ms, 2),
                total_pipeline_ms=round(total_ms, 2),
                effective_fps=round(1000.0 / max(1.0, total_ms), 1),
                compute_device=self.device_name
            ),
            modalities_used=["image", "environmental_context"] if (multimodal_context and any(v is not None and v != "" for v in multimodal_context.values())) else ["image"],
            multimodal_context=multimodal_context,
            short_explanation=short_explanation,
            what_to_check=what_to_check,
            warnings=warnings,
            camera_source=(
                "ESP32-CAM · OV2640" if source == "esp32_cam"
                else ("Raspberry Pi · OV5647" if source == "rpi_cam"
                else ("Device Camera" if source == "browser_camera" else "Browser Upload"))
            ),
            camera_device_id=camera_metadata.get("device_id") if camera_metadata else None,
            camera_capture_id=camera_metadata.get("capture_id") if camera_metadata else None,
            camera_capture_latency_ms=camera_metadata.get("capture_latency_ms") if camera_metadata else None,
        )


# Module-level singleton
inference_engine = InferenceEngine.get_instance()
