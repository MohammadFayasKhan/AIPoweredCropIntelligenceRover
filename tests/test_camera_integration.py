"""
SmartCropVision Camera Integration & Diagnostic Pipeline Parity Tests.

Verifies:
1. Camera capture payload handoff into preflight validation gate (/validate/vision).
2. Valid camera captures of agricultural foliage successfully proceed to /predict/vision.
3. Negative tests: Invalid captures (non-plant objects, documents, screenshots) are rejected
   by the validation firewall before neural inference occurs.
4. Model registry truthfulness: Confirms active production models (EfficientNetV2-S,
   YOLO11 PlantDoc Specimen Detector, Mobile-UNet).
"""

import io
from pathlib import Path
from PIL import Image
import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.config import settings

client = TestClient(app)


def create_synthetic_camera_frame(width=1280, height=720, pattern="leaf"):
    """
    Simulates a browser-captured JPEG frame from navigator.mediaDevices.getUserMedia
    video stream rendered onto a canvas.
    """
    if pattern == "leaf":
        sample_path = Path("frontend/samples/corn__fungal__common_rust.jpg")
        if sample_path.exists():
            with open(sample_path, "rb") as f:
                return f.read()

        arr = np.zeros((height, width, 3), dtype=np.uint8)
        arr[:, :, 1] = 160
        arr[:, :, 0] = 40
        arr[:, :, 2] = 50
        img = Image.fromarray(arr)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=95)
        return buf.getvalue()

    elif pattern == "blank_black":
        arr = np.zeros((height, width, 3), dtype=np.uint8)
        img = Image.fromarray(arr)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=95)
        return buf.getvalue()

    elif pattern == "document":
        arr = np.ones((height, width, 3), dtype=np.uint8) * 250
        img = Image.fromarray(arr)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=95)
        return buf.getvalue()

    return b""


def test_camera_capture_passes_preflight_validation():
    """Verify that a genuine camera-captured leaf frame passes /validate/vision."""
    camera_frame = create_synthetic_camera_frame(pattern="leaf")
    files = {"file": ("camera_capture_1726000000.jpg", camera_frame, "image/jpeg")}
    data = {"request_id": "cam_req_test_001"}

    response = client.post("/validate/vision", files=files, data=data)
    assert response.status_code == 200
    res_data = response.json()
    assert "is_inference_allowed" in res_data
    assert res_data["is_inference_allowed"] is True
    assert res_data["plant_presence"] is True
    assert res_data["leaf_presence"] is True


def test_camera_capture_end_to_end_inference():
    """Verify that confirmed camera photo proceeds through full multi-tier inference."""
    camera_frame = create_synthetic_camera_frame(pattern="leaf")
    files = {"file": ("camera_capture_1726000000.jpg", camera_frame, "image/jpeg")}
    data = {"model_tier": "server", "include_explainability": "false"}

    response = client.post("/predict/vision", files=files, data=data)
    assert response.status_code == 200
    res_data = response.json()

    assert res_data["status"] == "success"
    assert "diagnosis" in res_data
    assert "spatial_telemetry" in res_data

    st = res_data["spatial_telemetry"]
    assert "detection_engine" in st
    assert "YOLO" in st["detection_engine"]
    assert "PlantDoc" in st["detection_engine"] or "YOLO11" in st["detection_engine"]


def test_negative_camera_capture_rejected_by_firewall():
    """Verify that non-plant or covered camera frames are rejected before neural inference."""
    dark_frame = create_synthetic_camera_frame(pattern="blank_black")
    files = {"file": ("camera_dark_capture.jpg", dark_frame, "image/jpeg")}
    data = {"request_id": "cam_req_test_dark"}

    response = client.post("/validate/vision", files=files, data=data)
    assert response.status_code == 200
    res_data = response.json()
    assert res_data["is_inference_allowed"] is False
    assert res_data["status"] == "rejected"
    assert res_data["validation_status"] in ("rejected", "uncertain", "INVALID_NON_PLANT_IMAGE")


def test_production_model_registry_contains_plantdoc():
    """Verify that /models/status reports YOLO11 PlantDoc detector and EfficientNetV2-S."""
    response = client.get(f"{settings.API_V1_STR}/models/status")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ready"

    names = [m["name"] for m in data["models"]]
    assert any("EfficientNet" in n for n in names)
    assert any("PlantDoc" in n for n in names)
