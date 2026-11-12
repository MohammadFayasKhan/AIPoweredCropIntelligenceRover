"""
SmartCropVision Multi-Tier Detection Parity & Coordinate Verification Test Suite.
Verifies that:
  1. Authoritative YOLO detector checkpoints are loaded without fallback to generic COCO models.
  2. All genuine detections produced by the detector reach the API response without being collapsed into 1 box.
  3. Original detector classes, confidences, coordinates, and categories (canopy vs lesion) are preserved.
  4. Zero fabricated boxes, synthetic spots, or duplicate boxes are generated.
  5. Coordinate space is strictly bounded to the original image pixel dimensions.
"""

import os
import pytest
from pathlib import Path
from backend.app.services.inference_service import InferenceEngine
from backend.app.config import settings

STRAWBERRY_IMG = "cv/test_images/Strawberry-Leaves.jpg"
TOMATO_VIRUS_IMG = "cv/test_images/tomato_virus_06_zoom.jpg"
TOMATO_SEPTORIA_IMG = "cv/test_images/tomato-badleaves.jpg"


@pytest.fixture(scope="module")
def engine():
    eng = InferenceEngine.get_instance()
    eng.load_models()
    return eng


def test_authoritative_checkpoints_loaded(engine):
    """Ensure production checkpoints exist and are loaded without COCO fallbacks."""
    assert Path("cv/models/yolo_plantdoc_best.pt").exists(), "PlantDoc checkpoint missing"
    assert engine.model_tier2_plantdoc is not None, "Tier 2 PlantDoc model failed to load"
    
    # Verify 29 genuine PlantDoc classes
    names = engine.model_tier2_plantdoc.names
    assert len(names) == 29, f"Expected 29 PlantDoc classes, found {len(names)}"
    assert "Strawberry leaf" in names.values(), "Strawberry leaf missing from detector classes"
    assert "Tomato Septoria leaf spot" in names.values(), "Tomato Septoria leaf spot missing from detector classes"


def test_strawberry_multi_detection_preservation(engine):
    """
    Strawberry-Leaves.jpg validation image contains multiple genuine YOLO detections.
    Verify primary YOLO26s (default server) and preserved YOLO11n multi-box preservation.
    """
    if not os.path.exists(STRAWBERRY_IMG):
        pytest.skip(f"Test image not found: {STRAWBERRY_IMG}")

    resp = engine.predict_vision(STRAWBERRY_IMG, model_tier="server")
    telemetry = resp.spatial_telemetry
    boxes = telemetry.bounding_boxes

    # Verify genuine YOLO26s multi-detection output preserved
    assert "YOLO26" in telemetry.detection_engine
    assert telemetry.raw_detection_count >= 7, f"Expected raw count >= 7, got {telemetry.raw_detection_count}"
    assert telemetry.post_filtering_count == len(boxes), "post_filtering_count mismatch with boxes list"
    assert len(boxes) >= 7, f"Expected >= 7 genuine detections, got {len(boxes)}"

    # Verify every box retains its genuine metadata and identity
    seen_ids = set()
    for b in boxes:
        assert b.detection_id not in seen_ids, f"Duplicate detection ID found: {b.detection_id}"
        seen_ids.add(b.detection_id)
        assert b.class_name == "Strawberry leaf", f"Class identity altered: {b.class_name}"
        assert b.category_type == "canopy", f"Strawberry leaf should be canopy, got {b.category_type}"
        assert b.box_type == "leaf"
        assert b.color_hex == "#52b788"
        assert b.confidence >= min(settings.YOLO26_CONF_THRESH, settings.YOLO_PLANTDOC_CONF_THRESH, 0.08)
        assert b.coordinate_space == "original_image_pixels"
        
        # Check coordinate bounds
        x1, y1, x2, y2 = b.bbox_xyxy
        assert 0 <= x1 < x2 <= 685, f"Invalid x coordinates: {x1}, {x2}"
        assert 0 <= y1 < y2 <= 550, f"Invalid y coordinates: {y1}, {y2}"

    assert telemetry.canopy_box_count == len(boxes)
    assert telemetry.lesion_box_count == 0

    # Preserved legacy YOLO11n baseline remains intact for comparison
    resp_legacy = engine.predict_vision(STRAWBERRY_IMG, model_tier="yolo11")
    assert "YOLO11n" in resp_legacy.spatial_telemetry.detection_engine
    assert len(resp_legacy.spatial_telemetry.bounding_boxes) == 8


def test_tomato_seven_detections_verification(engine):
    """
    Verify tomato foliage field image produces all 7 genuine specimen canopy boundaries
    while suppressing oversized/off-specimen false positives via geometry & canopy containment.
    """
    target_img = TOMATO_SEPTORIA_IMG if os.path.exists(TOMATO_SEPTORIA_IMG) else TOMATO_VIRUS_IMG
    if not os.path.exists(target_img):
        pytest.skip("Tomato test images not found")

    resp = engine.predict_vision(target_img, model_tier="server")
    telemetry = resp.spatial_telemetry
    boxes = telemetry.bounding_boxes

    # Verify specimen and lesion separation with background/oversized false-positive suppression
    assert len(telemetry.specimen_detections) == 7, f"Expected 7 specimen leaf boxes, got {len(telemetry.specimen_detections)}"
    assert len(boxes) == len(telemetry.specimen_detections) + len(telemetry.lesion_detections)
    assert telemetry.post_filtering_count == len(boxes)
    assert telemetry.specimen_box_count == 7
    assert telemetry.lesion_box_count == len(telemetry.lesion_detections)
    assert resp.diagnosis.lesion_foci_count >= 1, "Expected lesion foci via YOLO26-Seg or Mobile-UNet segmentation"

    # Every kept lesion must lie inside a detected specimen leaf boundary (zero background hallucinations)
    for les in telemetry.lesion_detections:
        lx1, ly1, lx2, ly2 = les.bbox_xyxy
        l_cx, l_cy = (lx1 + lx2) / 2.0, (ly1 + ly2) / 2.0
        inside_any = any(
            (s.bbox_xyxy[0] - 0.02 * (s.bbox_xyxy[2] - s.bbox_xyxy[0])) <= l_cx <= (s.bbox_xyxy[2] + 0.02 * (s.bbox_xyxy[2] - s.bbox_xyxy[0]))
            and (s.bbox_xyxy[1] - 0.02 * (s.bbox_xyxy[3] - s.bbox_xyxy[1])) <= l_cy <= (s.bbox_xyxy[3] + 0.02 * (s.bbox_xyxy[3] - s.bbox_xyxy[1]))
            for s in telemetry.specimen_detections
        )
        assert inside_any, f"Lesion {les.bbox_xyxy} leaked outside all specimen boundaries"

    # Preserved legacy baseline retains >=1 canopy-contained lesion focus on YOLOv8n
    resp_legacy = engine.predict_vision(target_img, model_tier="legacy")
    assert len(resp_legacy.spatial_telemetry.lesion_detections) >= 1

    # Verify each detection retains unique ID and correct coordinates
    seen_ids = set()
    for b in boxes:
        assert b.detection_id not in seen_ids
        seen_ids.add(b.detection_id)
        assert b.confidence >= 0.10 or b.confidence >= 0.08
        x1, y1, x2, y2 = b.bbox_xyxy
        assert 0 <= x1 < x2
        assert 0 <= y1 < y2


def test_potato_early_blight_lesion_and_specimen_separation(engine):
    """
    Test Potato Early Blight (images_7.jpg):
    Verifies that:
      1. Primary YOLO26s + YOLO26s-Seg detects 1 leaf canopy boundary and >=15 granular spot lesions.
      2. Preserved legacy YOLOv8n baseline still yields 6 deduplicated spot lesions.
      3. Diagnosis lesion_foci_count matches len(lesion_detections) exactly.
      4. Localization capability is 'specimen_boundary_and_lesion_foci'.
    """
    potato_img = "cv/test_images/images_7.jpg"
    if not os.path.exists(potato_img):
        pytest.skip(f"Potato test image not found: {potato_img}")

    resp = engine.predict_vision(potato_img, model_tier="server")
    telemetry = resp.spatial_telemetry
    diag = resp.diagnosis

    assert len(telemetry.specimen_detections) == 1, f"Expected 1 leaf box, got {len(telemetry.specimen_detections)}"
    assert len(telemetry.lesion_detections) >= 15, f"Expected >=15 YOLO26s-Seg spots, got {len(telemetry.lesion_detections)}"
    assert diag.lesion_foci_count == len(telemetry.lesion_detections)
    assert diag.lesion_foci_source == "yolo26_lesion_detector"
    assert telemetry.localization_capability == "specimen_boundary_and_lesion_foci"

    # Verify specimen box encompasses large foliar region
    specimen = telemetry.specimen_detections[0]
    assert specimen.category_type == "canopy"
    assert specimen.box_type == "leaf"
    assert specimen.color_hex == "#52b788"
    assert specimen.class_name == "Potato leaf early blight"
    s_w = specimen.bbox_xyxy[2] - specimen.bbox_xyxy[0]
    s_h = specimen.bbox_xyxy[3] - specimen.bbox_xyxy[1]
    assert s_w > 300 and s_h > 500, f"Specimen box should cover leaf: w={s_w}, h={s_h}"

    # Verify each lesion box is an internal spot
    for les in telemetry.lesion_detections:
        assert les.category_type == "lesion"
        assert les.box_type == "lesion"
        assert les.color_hex == "#f4a261"
        l_w = les.bbox_xyxy[2] - les.bbox_xyxy[0]
        l_h = les.bbox_xyxy[3] - les.bbox_xyxy[1]
        assert l_w < 120 and l_h < 120, f"Lesion box should be a granular spot: w={l_w}, h={l_h}"

    # Preserved legacy baseline parity check
    resp_leg = engine.predict_vision(potato_img, model_tier="legacy")
    assert len(resp_leg.spatial_telemetry.lesion_detections) == 6
    assert resp_leg.diagnosis.lesion_foci_source == "yolov8n_lesions_detector"


def test_healthy_leaf_negative_control(engine):
    """
    Test healthy foliage (potato__healthy__healthy.jpg):
    Verifies that zero lesion spots are predicted (clean negative control).
    """
    healthy_img = "frontend/samples/potato__healthy__healthy.jpg"
    if not os.path.exists(healthy_img):
        pytest.skip(f"Healthy test image not found: {healthy_img}")

    resp = engine.predict_vision(healthy_img, model_tier="server")
    telemetry = resp.spatial_telemetry
    diag = resp.diagnosis

    assert len(telemetry.lesion_detections) == 0, f"Expected 0 lesions on healthy leaf, got {len(telemetry.lesion_detections)}"
    assert diag.lesion_foci_count == 0
    assert telemetry.localization_capability in ("specimen_boundary_only", "unavailable")


def test_no_fabricated_boxes_or_heuristics(engine):
    """Verify that zero heuristic or synthetic spots are generated."""
    if not os.path.exists(STRAWBERRY_IMG):
        pytest.skip("Strawberry test image not found")

    resp = engine.predict_vision(STRAWBERRY_IMG, model_tier="server")
    valid_spec_names = set(engine.model_tier2_plantdoc.names.values())
    if engine.model_yolo26_specimen is not None:
        valid_spec_names.update(engine.model_yolo26_specimen.names.values())
    for b in resp.spatial_telemetry.bounding_boxes:
        assert not b.label.startswith("Spot #"), f"Found synthetic spot label: {b.label}"
        assert b.class_name in valid_spec_names or (
            engine.model_tier2_lesions and b.class_name in engine.model_tier2_lesions.names.values()
        )


def test_agnostic_nms_eliminates_duplicate_same_leaf_classes(engine):
    """Verify class-agnostic NMS prevents duplicate multi-class boxes on the same leaf."""
    if not os.path.exists(TOMATO_VIRUS_IMG):
        pytest.skip(f"Test image not found: {TOMATO_VIRUS_IMG}")

    resp = engine.predict_vision(TOMATO_VIRUS_IMG, model_tier="server")
    specs = resp.spatial_telemetry.specimen_detections
    assert len(specs) >= 4, f"Expected >= 4 deduplicated leaf boxes on tomato_virus_06_zoom.jpg, got {len(specs)}"

    # Confirm no two specimen boxes overlap heavily (IoU < 0.50 across all pairs regardless of class)
    for i in range(len(specs)):
        ax1, ay1, ax2, ay2 = specs[i].bbox_xyxy
        a_area = max(1, (ax2 - ax1) * (ay2 - ay1))
        for j in range(i + 1, len(specs)):
            bx1, by1, bx2, by2 = specs[j].bbox_xyxy
            ix1, iy1 = max(ax1, bx1), max(ay1, by1)
            ix2, iy2 = min(ax2, bx2), min(ay2, by2)
            if ix2 > ix1 and iy2 > iy1:
                inter = (ix2 - ix1) * (iy2 - iy1)
                union = a_area + max(1, (bx2 - bx1) * (by2 - by1)) - inter
                assert inter / max(1, union) < 0.50, (
                    f"Duplicate overlapping boxes left behind: {specs[i].class_name} vs {specs[j].class_name}"
                )


def test_skin_and_non_foliar_lesion_rejection():
    """Verify spectral and canopy-containment guards reject skin/finger and non-foliar regions."""
    import numpy as np
    from backend.app.services.inference_service import _passes_lesion_spectral_checks, _is_inside_specimen_canopy
    from backend.app.schemas.diagnosis import BoundingBox

    # Create synthetic 200x200 image with skin-toned patch (BGR ~ 125, 145, 195) on dark soil
    img = np.full((200, 200, 3), (40, 50, 75), dtype=np.uint8)
    img[130:180, 60:140] = (125, 145, 195)  # warm skin-like region in BGR (R=195, G=145, B=125)

    assert not _passes_lesion_spectral_checks(img, 70, 140, 130, 175, 200, 200), (
        "Skin/finger region should be rejected by botanical lesion spectral checks"
    )

    spec_box = [
        BoundingBox(
            detection_id="spec_0", class_id=0, class_name="Leaf", category_type="canopy",
            label="Leaf (80%)", bbox_xyxy=[20, 10, 110, 110], confidence=0.80,
            centroid_norm=[0.32, 0.30], box_type="leaf", color_hex="#52b788",
            source_model="test", coordinate_space="original_image_pixels"
        )
    ]
    assert not _is_inside_specimen_canopy(130, 130, 170, 170, spec_box), (
        "Lesion outside specimen box must be rejected when specimen detections exist"
    )
    assert _is_inside_specimen_canopy(40, 40, 70, 70, spec_box)


def test_honest_yolo_architecture_verification(engine):
    """Verify honest architecture reporting across promoted YOLO26s, legacy YOLO11n, and YOLOv8n lesions."""
    assert engine.model_tier2_arch == "YOLO11n"
    assert engine.model_tier2_is_yolo26 is False
    assert engine.model_tier2_lesions_arch == "YOLOv8n"
    assert engine.model_yolo26_specimen is not None
    assert engine.model_yolo26_specimen_arch == "YOLO26s"
    meta_server = engine.get_model_metadata("server")
    assert meta_server.detection_taxonomy_version == "PlantDoc-29Class-YOLO26-v1.0"
    meta_gen2 = engine.get_model_metadata("gen2")
    assert meta_gen2.detection_taxonomy_version == "PlantDoc-29Class-YOLO-v1.0"
