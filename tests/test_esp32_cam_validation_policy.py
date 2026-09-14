"""
SmartCropVision ESP32-CAM Source-Aware Quality Policy & Semantic Plant Validation Tests.

Validates the architectural requirement:
- The ESP32-CAM is strictly an image acquisition device.
- Authenticated ESP32-CAM captures receive a camera-specific technical quality policy
  (tolerating OV2640 compression, mild optical blur, and lower resolution).
- Semantic plant validation firewall remains non-negotiable: non-plant images, screenshots,
  and documents are NEVER admitted to neural inference.
- Spoofed provenance without server-verified capture ID is rejected.
"""

import io
import time
from pathlib import Path
from PIL import Image, ImageFilter
import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.services.esp32_gateway import esp32_gateway
from backend.app.utils.domain_validation import validate_plant_image

client = TestClient(app)


def get_real_or_synthetic_leaf(degrade_blur=False, degrade_resolution=False):
    sample_path = Path("frontend/samples/corn__fungal__common_rust.jpg")
    if sample_path.exists():
        img = Image.open(sample_path).convert("RGB")
    else:
        arr = np.zeros((480, 640, 3), dtype=np.uint8)
        arr[:, :, 1] = 160
        arr[:, :, 0] = 50
        arr[:, :, 2] = 40
        img = Image.fromarray(arr)

    if degrade_resolution:
        img = img.resize((320, 240))  # Standard OV2640 QVGA preview resolution

    if degrade_blur:
        img = img.filter(ImageFilter.GaussianBlur(radius=1.2))  # Authentic mild optical sensor blur

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=70)  # Standard OV2640 JPEG compression
    return buf.getvalue()


def get_non_plant_screenshot():
    arr = np.ones((400, 600, 3), dtype=np.uint8) * 240
    # Simulate text / window chrome lines
    arr[0:40, :] = [30, 30, 30]
    arr[50:60, 40:200] = [10, 10, 10]
    arr[70:80, 40:400] = [10, 10, 10]
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def test_esp32_degraded_leaf_accepted_with_advisory():
    """
    Test that an authentic ESP32 capture of a plant with mild blur / low resolution
    is accepted for inference with a 'Camera Quality Degraded' advisory.
    """
    capture_id = f"SCAP_TEST_VAL_{int(time.time())}"
    esp32_gateway._verified_captures[capture_id] = {
        "capture_id": capture_id,
        "device_id": "esp32-field-cam-01",
        "timestamp": time.time(),
        "consumed": False,
        "resolution": "QVGA"
    }

    degraded_leaf_bytes = get_real_or_synthetic_leaf(degrade_blur=True, degrade_resolution=True)

    files = {"file": ("esp32_capture.jpg", degraded_leaf_bytes, "image/jpeg")}
    data = {
        "request_id": "val_esp32_test_01",
        "source": "esp32_cam",
        "esp32_capture_id": capture_id
    }

    res = client.post("/validate/vision", files=files, data=data)
    assert res.status_code == 200
    res_data = res.json()

    assert res_data["status"] == "valid"
    assert res_data["is_inference_allowed"] is True
    assert "Degraded" in res_data["image_quality"] or "Tolerated" in res_data["validation_reason"]


def test_esp32_non_plant_screenshot_strictly_rejected():
    """
    Critical security firewall test:
    Even when claimed as ESP32-CAM, a non-plant screenshot MUST be rejected.
    Semantic plant validation is never bypassed for ESP32 captures.
    """
    capture_id = f"SCAP_TEST_NONPLANT_{int(time.time())}"
    esp32_gateway._verified_captures[capture_id] = {
        "capture_id": capture_id,
        "device_id": "esp32-field-cam-01",
        "timestamp": time.time(),
        "consumed": False,
        "resolution": "SVGA"
    }

    screenshot_bytes = get_non_plant_screenshot()

    files = {"file": ("esp32_screenshot.jpg", screenshot_bytes, "image/jpeg")}
    data = {
        "request_id": "val_esp32_test_nonplant",
        "source": "esp32_cam",
        "esp32_capture_id": capture_id
    }

    res = client.post("/validate/vision", files=files, data=data)
    assert res.status_code == 200
    res_data = res.json()

    # Must be strictly rejected!
    assert res_data["status"] == "rejected"
    assert res_data["is_inference_allowed"] is False
    assert res_data["validation_status"] in ("INVALID_SCREENSHOT_OR_DOCUMENT", "INVALID_NON_PLANT_IMAGE", "rejected")


def test_esp32_unauthenticated_spoof_prevention():
    """
    Verify that an unauthenticated client claiming source=esp32_cam without
    a registered server capture_id is NOT granted the relaxed quality policy.
    """
    degraded_leaf_bytes = get_real_or_synthetic_leaf(degrade_blur=True, degrade_resolution=True)

    files = {"file": ("spoofed_capture.jpg", degraded_leaf_bytes, "image/jpeg")}
    data = {
        "request_id": "val_esp32_spoof",
        "source": "esp32_cam",
        "esp32_capture_id": "SCAP_INVALID_UNAUTHENTIC_99999"
    }

    res = client.post("/validate/vision", files=files, data=data)
    assert res.status_code == 200
    res_data = res.json()

    # The source should be downgraded to browser_upload server-side
    assert res_data.get("source") != "esp32_cam" or res_data.get("is_inference_allowed") is False


def test_esp32_end_to_end_inference_preserves_provenance():
    """
    Test that an authentic ESP32 capture runs through /predict/vision and returns
    full classification, YOLO26 detection, Mobile-UNet segmentation, and camera provenance.
    """
    capture_id = f"SCAP_E2E_{int(time.time())}"
    esp32_gateway._verified_captures[capture_id] = {
        "capture_id": capture_id,
        "device_id": "esp32-field-cam-01",
        "timestamp": time.time(),
        "consumed": False,
        "resolution": "SVGA"
    }

    leaf_bytes = get_real_or_synthetic_leaf(degrade_blur=False, degrade_resolution=False)

    files = {"file": ("esp32_foliage_sample.jpg", leaf_bytes, "image/jpeg")}
    data = {
        "model_tier": "server",
        "include_explainability": "false",
        "source": "esp32_cam",
        "esp32_capture_id": capture_id
    }

    res = client.post("/predict/vision", files=files, data=data)
    assert res.status_code == 200
    res_data = res.json()

    assert res_data["status"] == "success"
    assert "ESP32" in res_data["camera_source"]
    assert res_data["camera_device_id"] == "esp32-field-cam-01"
    assert res_data["camera_capture_id"] == capture_id
    assert "diagnosis" in res_data
    assert "spatial_telemetry" in res_data
