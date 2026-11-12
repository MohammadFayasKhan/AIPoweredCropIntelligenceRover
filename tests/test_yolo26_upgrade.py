"""
Regression, Dataset Preparation, Coordinate, Architecture, and Non-Destructive
Integration Tests for the Official Ultralytics YOLO26 Upgrade (yolo26_v1).
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from ultralytics import YOLO

sys.path.insert(0, str(Path("cv/scripts").resolve()))
from prepare_yolo26_tridataset import (  # noqa: E402
    CANONICAL_PLANTDOC_29_CLASSES,
    DATASET_LICENSES,
    mask_to_yolo_polygons,
    prepare_plantdoc_yolo26,
    prepare_plantseg_lesion_yolo26,
    run_preparation,
)
from evaluate_yolo26_v1 import inspect_yolo_checkpoint  # noqa: E402

from backend.app.config import settings  # noqa: E402
from backend.app.main import app  # noqa: E402
from backend.app.services.inference_service import InferenceEngine, _inspect_yolo_architecture  # noqa: E402
from backend.app.utils.image_processing import reverse_letterbox_xyxy, scale_segmentation_mask_to_original  # noqa: E402


def _write_sample_voc_pair(folder: Path, stem: str, cls_name: str, color_rgb=(50, 140, 50), size=(128, 96), shape_seed: int = 0):
    folder.mkdir(parents=True, exist_ok=True)
    w, h = size
    img_arr = np.full((h, w, 3), color_rgb, dtype=np.uint8)
    # Vary spatial geometry by shape_seed so distinct synthetic samples produce distinct dHashes
    x_off = (shape_seed * 7) % 24
    y_off = (shape_seed * 5) % 16
    cv2.ellipse(img_arr, (58 + x_off, 42 + y_off), (32 - (shape_seed % 6), 20 + (shape_seed % 5)), (shape_seed * 23) % 180, 0, 360, (40, 185, 70), -1)
    cv2.circle(img_arr, (52 + x_off, 38 + y_off), 4 + (shape_seed % 4), (90, 55, 35), -1)
    img_path = folder / f"{stem}.jpg"
    Image.fromarray(img_arr).save(img_path)

    xml_path = folder / f"{stem}.xml"
    xml_path.write_text(
        f"""<annotation>
  <filename>{stem}.jpg</filename>
  <size><width>{w}</width><height>{h}</height><depth>3</depth></size>
  <object>
    <name>{cls_name}</name>
    <bndbox><xmin>20</xmin><ymin>12</ymin><xmax>98</xmax><ymax>72</ymax></bndbox>
  </object>
</annotation>""",
        encoding="utf-8",
    )
    return img_path, xml_path


def test_existing_production_defaults_preserved():
    """Ensures YOLO26 v1 is enabled by default while legacy checkpoints and notebooks remain intact."""
    assert settings.YOLO26_ENABLED is True
    assert settings.YOLO26_INFERENCE_MODE == "one2many"
    assert abs(settings.YOLO26_CONF_THRESH - 0.25) < 1e-6
    assert Path("artifacts/yolo26_v1/checkpoints/yolo26_specimen_best.pt").exists()
    assert Path("cv/models/yolo_plantdoc_best.pt").exists()
    assert Path("cv/models/yolov8n_lesions_best.pt").exists()
    assert Path("cv/models/efficientnetv2_s_best.pt").exists()
    assert Path("cv/models/mobile_unet_best.pt").exists()
    assert Path("notebooks/historical/kagglesmartcropvisionsystem.ipynb").exists()
    assert Path("notebooks/YOLO26_TriDataset_Training_v1.ipynb").exists()


def test_prepare_yolo26_tridataset_dedup_canonical_and_pseudo_exclusion(tmp_path: Path):
    """Verifies canonical VOC conversion, dHash duplicate leak removal, PlantWild license, and PlantSeg masks."""
    raw_root = tmp_path / "raw"
    pdoc = raw_root / "plantdoc_od_repo"
    # Official test image + duplicate in train (must be pruned from train!) + unique train images
    _write_sample_voc_pair(pdoc / "test" / "Tomato leaf", "test_tomato", "Tomato leaf", color_rgb=(60, 150, 60), shape_seed=99)
    _write_sample_voc_pair(pdoc / "train" / "Tomato leaf", "dup_tomato", "Tomato leaf", color_rgb=(60, 150, 60), shape_seed=99)
    for i in range(4):
        _write_sample_voc_pair(
            pdoc / "train" / "Apple Scab Leaf",
            f"apple_scab_{i}",
            "Apple Scab Leaf",
            color_rgb=(30 + i * 25, 110 + i * 20, 40),
            shape_seed=i + 1,
        )

    # PlantSeg mock genuine LabelMe + binary mask
    pseg = raw_root / "PlantSeg"
    (pseg / "images" / "train").mkdir(parents=True)
    (pseg / "masks" / "train").mkdir(parents=True)
    seg_im = np.full((128, 128, 3), (40, 130, 50), dtype=np.uint8)
    cv2.imwrite(str(pseg / "images" / "train" / "leaf_01.jpg"), seg_im)
    seg_mask = np.zeros((128, 128), dtype=np.uint8)
    cv2.circle(seg_mask, (64, 64), 20, 255, -1)
    cv2.imwrite(str(pseg / "masks" / "train" / "leaf_01.png"), seg_mask)

    # Synthetic HSV pseudo-mask folder MUST be excluded cleanly
    pseudo_dir = tmp_path / "cv" / "datasets" / "processed" / "segmentation" / "masks" / "train"
    pseudo_dir.mkdir(parents=True)
    cv2.imwrite(str(pseudo_dir / "pseudo_01.png"), seg_mask)

    out_dir = tmp_path / "artifacts" / "yolo26_v1" / "datasets"
    rep = run_preparation([raw_root], out_dir, seed=42)

    pd_rep = rep["plantdoc_specimen_detection"]
    assert pd_rep["status"] == "READY"
    assert pd_rep["classes_count"] == 29
    assert pd_rep["cross_split_duplicates_removed"] == 1
    assert pd_rep["splits"]["test"] == 1
    # Verify canonical class index mapping: Apple Scab Leaf == 0
    apple_lbls = sorted((out_dir / "plantdoc_yolo26" / "labels" / "train").glob("*apple_scab*.txt"))
    assert len(apple_lbls) >= 1
    first_cid = int(apple_lbls[0].read_text().split()[0])
    assert first_cid == CANONICAL_PLANTDOC_29_CLASSES.index("Apple Scab Leaf") == 0

    ps_rep = rep["plantseg_lesion_segmentation"]
    assert ps_rep["status"] == "READY"
    assert ps_rep["splits"]["train"] == 1
    assert ps_rep["instances"]["train"] == 1
    assert rep["licenses_and_restrictions"]["PlantWild"]["commercial_use_allowed"] is False
    assert "CC BY-NC-ND 4.0" in rep["licenses_and_restrictions"]["PlantWild"]["license"]


def test_mask_to_yolo_polygons_and_letterbox_coordinate_transforms():
    """Verifies YOLO-Seg polygon extraction, letterbox reversal, and segmentation mask scaling."""
    mask = np.zeros((100, 200), dtype=np.uint8)
    cv2.rectangle(mask, (40, 20), (120, 70), 2, -1)
    polys = mask_to_yolo_polygons(mask, class_id=0)
    assert len(polys) == 1
    cid, coords = polys[0]
    assert cid == 0 and len(coords) >= 8
    assert all(0.0 <= c <= 1.0 for c in coords)

    # Reverse letterbox on 640x640 input for 1280x640 (2:1) original image: gain=0.5, pad_h=160
    unboxed = reverse_letterbox_xyxy([[100.0, 160.0, 540.0, 480.0]], 640, 640, 1280, 640)
    assert unboxed[0] == [200, 0, 1080, 640]

    scaled_mask = scale_segmentation_mask_to_original(mask, orig_w=400, orig_h=200)
    assert scaled_mask.shape == (200, 400)
    assert np.any(scaled_mask == 2)


def test_official_yolo26_architecture_vs_yolo11_rejection():
    """Verifies official YOLO26 Detect/Segment26 architecture and rejects YOLO11n aliases."""
    m_det = YOLO("yolo26n.yaml")
    arch_d, is_y26_d, _ = _inspect_yolo_architecture(m_det)
    assert arch_d == "YOLO26n" and is_y26_d is True

    m_det_s = YOLO("yolo26s.yaml")
    arch_s_det, is_y26_s_det, _ = _inspect_yolo_architecture(m_det_s)
    assert arch_s_det == "YOLO26s" and is_y26_s_det is True

    # Simulate Ultralytics >= 8.4.200 YAML init where head.end2end starts False despite one2one head
    m_det_s.model.model[-1].end2end = False
    arch_s_kgl, is_y26_s_kgl, _ = _inspect_yolo_architecture(m_det_s)
    assert arch_s_kgl == "YOLO26s" and is_y26_s_kgl is True

    m_seg = YOLO("yolo26n-seg.yaml")
    arch_s, is_y26_s, _ = _inspect_yolo_architecture(m_seg)
    assert arch_s == "YOLO26n" and is_y26_s is True

    # Test both end2end=True and end2end=False modes cleanly
    dummy = np.zeros((256, 256, 3), dtype=np.uint8)
    assert len(m_det.predict(dummy, imgsz=256, end2end=True, verbose=False)) == 1
    assert len(m_det.predict(dummy, imgsz=256, end2end=False, verbose=False)) == 1

    # Existing PlantDoc checkpoint must be detected truthfully as YOLO11 (not YOLO26)
    legacy_meta = inspect_yolo_checkpoint(Path("cv/models/yolo_plantdoc_best.pt"))
    assert legacy_meta["is_genuine_yolo26"] is False
    assert "YOLO11" in legacy_meta["detected_architecture"]


def test_registry_and_fastapi_optional_yolo26_route_non_destructive():
    """Verifies registry surfaces YOLO26 as primary when enabled and preserves legacy YOLO11n fallback."""
    client = TestClient(app)
    reg = client.get("/models/registry").json()
    ids = {m["model_id"]: m for m in reg["models"]}
    assert "yolo26_v1_specimen" in ids
    assert "yolo26_v1_lesion_seg" in ids
    spec_ckpt_exists = settings.YOLO26_SPECIMEN_MODEL_PATH.exists()
    assert ids["yolo26_v1_specimen"]["is_ready"] is bool(spec_ckpt_exists and settings.YOLO26_ENABLED)
    assert ids["yolo26_v1_specimen"]["tier"] == ("server" if settings.YOLO26_ENABLED else "yolo26_optional")
    assert ids["tier2_plantdoc_yolo"]["tier"] == ("legacy_baseline" if settings.YOLO26_ENABLED else "server")
    if spec_ckpt_exists:
        assert ids["yolo26_v1_specimen"]["checkpoint_path"] == "yolo26_specimen_best.pt"
        assert ids["yolo26_v1_specimen"]["checkpoint_size_mb"] > 18.0
        assert ids["yolo26_v1_specimen"]["sha256_hash"] == "d1be9b57b02b483aca1675bf33113fcc5622352295f6d0f99770425516875eb9"

    # Default server tier now routes to YOLO26 v1; explicit legacy fallback keeps YOLO11n intact
    img_path = Path("frontend/samples/tomato__fungal__early_blight.jpg")
    with open(img_path, "rb") as f:
        res_default = client.post(
            f"{settings.API_V1_STR}/vision/diagnose",
            files={"file": (img_path.name, f, "image/jpeg")},
            data={"model_tier": "server"},
        )
    assert res_default.status_code == 200
    d_def = res_default.json()
    assert "YOLO26 v1" in d_def["spatial_telemetry"]["detection_engine"]
    assert d_def["spatial_telemetry"]["taxonomy_version"] == "PlantDoc-29Class-YOLO26-v1.0"

    with open(img_path, "rb") as f:
        res_legacy = client.post(
            f"{settings.API_V1_STR}/vision/diagnose",
            files={"file": (img_path.name, f, "image/jpeg")},
            data={"model_tier": "yolo11"},
        )
    assert res_legacy.status_code == 200
    d_leg = res_legacy.json()
    assert "YOLO11n" in d_leg["spatial_telemetry"]["detection_engine"]
    assert d_leg["spatial_telemetry"]["taxonomy_version"] == "PlantDoc-29Class-YOLO-v1.0"

    eng = InferenceEngine.get_instance()
    orig_spec, orig_les = eng.model_yolo26_specimen, eng.model_yolo26_lesion
    orig_arch = eng.model_yolo26_specimen_arch
    try:
        # 1. Opt-in tier without loaded weights must warn honestly and never fake YOLO26 claims
        eng.model_yolo26_specimen = None
        eng.model_yolo26_lesion = None
        with open(img_path, "rb") as f:
            res_opt = client.post(
                f"{settings.API_V1_STR}/vision/diagnose",
                files={"file": (img_path.name, f, "image/jpeg")},
                data={"model_tier": "yolo26"},
            )
        assert res_opt.status_code == 200
        d_opt = res_opt.json()
        assert "YOLO11n" in d_opt["spatial_telemetry"]["detection_engine"]
        assert any("YOLO26 v1" in w for w in d_opt.get("warnings", []))

        # 2. With genuine trained YOLO26s specimen weights (if staged) or live YAML, execute real YOLO26 path
        if orig_spec is not None:
            eng.model_yolo26_specimen = orig_spec
            eng.model_yolo26_specimen_arch = orig_arch
            spec_meta = inspect_yolo_checkpoint(settings.YOLO26_SPECIMEN_MODEL_PATH)
            assert spec_meta["is_genuine_yolo26"] is True
            assert spec_meta["reg_max"] == 1
        else:
            eng.model_yolo26_specimen = YOLO("yolo26n.yaml")
        eng.model_yolo26_lesion = YOLO("yolo26n-seg.yaml")

        field_img = Path("cv/test_images/Strawberry-Leaves.jpg")
        target_img = field_img if (field_img.exists() and orig_spec is not None) else img_path
        with open(target_img, "rb") as f:
            res_live_y26 = client.post(
                f"{settings.API_V1_STR}/vision/diagnose",
                files={"file": (target_img.name, f, "image/jpeg")},
                data={"model_tier": "yolo26"},
            )
        assert res_live_y26.status_code == 200
        d_live = res_live_y26.json()
        assert "YOLO26 v1" in d_live["spatial_telemetry"]["detection_engine"]
        assert d_live["spatial_telemetry"]["taxonomy_version"] == "PlantDoc-29Class-YOLO26-v1.0"
        if orig_spec is not None:
            assert len(d_live["spatial_telemetry"]["specimen_detections"]) > 0
            for det in d_live["spatial_telemetry"]["specimen_detections"]:
                x1, y1, x2, y2 = det["bbox_xyxy"]
                assert 0 <= x1 < x2 and 0 <= y1 < y2
    finally:
        eng.model_yolo26_specimen, eng.model_yolo26_lesion = orig_spec, orig_les
        eng.model_yolo26_specimen_arch = orig_arch


def test_yolo26_lesion_v1_end_to_end_and_health():
    """Verifies promoted YOLO26s-Seg lesion model artifact, manifest SHA-256, health endpoint, and multi-crop inference."""
    import hashlib

    les_path = settings.YOLO26_LESION_MODEL_PATH
    assert les_path.exists(), f"Missing trained YOLO26 lesion checkpoint at {les_path}"
    les_meta = inspect_yolo_checkpoint(les_path)
    assert les_meta["is_genuine_yolo26"] is True
    assert les_meta["head_class"] == "Segment26"
    assert les_meta["reg_max"] == 1

    manifest_p = Path("artifacts/yolo26_lesion_v1/RELEASE_MANIFEST_YOLO26_LESION_v1.json")
    assert manifest_p.exists()
    manifest = json.loads(manifest_p.read_text(encoding="utf-8"))
    expected_sha = manifest["checkpoints"]["yolo26_lesion_best"]["sha256"]
    actual_sha = hashlib.sha256(les_path.read_bytes()).hexdigest()
    assert actual_sha == expected_sha

    with TestClient(app) as client:
        st_res = client.get(f"{settings.API_V1_STR}/models/status")
        assert st_res.status_code == 200
        models_by_name = {m["name"]: m for m in st_res.json()["models"]}
        assert any("YOLO26s-Seg v1 Lesion Localizer" in name and m["status"] == "ready" for name, m in models_by_name.items())

        # Infected multi-crop foliage sample should produce verified YOLO26s-Seg lesion detections
        grape_sample = Path("frontend/samples/grape__fungal__black_rot.jpg")
        if grape_sample.exists():
            with open(grape_sample, "rb") as f:
                r_inf = client.post(
                    f"{settings.API_V1_STR}/vision/diagnose",
                    files={"file": (grape_sample.name, f, "image/jpeg")},
                    data={"model_tier": "server"},
                )
            assert r_inf.status_code == 200
            d_inf = r_inf.json()
            assert d_inf["diagnosis"]["is_infected"] is True
            assert d_inf["diagnosis"]["lesion_foci_source"] == "yolo26_lesion_detector"
            assert len(d_inf["spatial_telemetry"]["lesion_detections"]) >= 1
            for lb in d_inf["spatial_telemetry"]["lesion_detections"]:
                x1, y1, x2, y2 = lb["bbox_xyxy"]
                assert 0 <= x1 < x2 and 0 <= y1 < y2

        # Healthy negative control must yield 0 lesion boxes
        healthy_sample = Path("frontend/samples/potato__healthy__healthy.jpg")
        if healthy_sample.exists():
            with open(healthy_sample, "rb") as f:
                r_hl = client.post(
                    f"{settings.API_V1_STR}/vision/diagnose",
                    files={"file": (healthy_sample.name, f, "image/jpeg")},
                    data={"model_tier": "server"},
                )
            assert r_hl.status_code == 200
            d_hl = r_hl.json()
            assert d_hl["diagnosis"]["lesion_foci_count"] == 0
            assert d_hl["spatial_telemetry"]["lesion_detections"] == []

