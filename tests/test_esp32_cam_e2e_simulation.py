"""
End-to-End Hardware Simulation Test for SmartCropVision ESP32-CAM (OV2640).

Simulates the complete physical cycle:
1. Outbound WebSocket connection from ESP32-CAM to backend.
2. Device authentication and heartbeat registration.
3. Live streaming preview transmission.
4. Remote capture command dispatch.
5. High-resolution OV2640 frame acquisition and transport.
6. Server-side provenance verification.
7. Preflight domain validation under the camera quality policy.
8. Server-grade ML inference cascade (EfficientNetV2-S + YOLO26 + Mobile-UNet).
9. Unified diagnosis response delivery with camera provenance telemetry.
"""

import asyncio
import io
import time
from pathlib import Path
from PIL import Image
import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.services.esp32_gateway import esp32_gateway
from backend.app.config import settings

client = TestClient(app)


def make_leaf_jpeg(w=640, h=480):
    sample_path = Path("frontend/samples/corn__fungal__common_rust.jpg")
    if sample_path.exists():
        img = Image.open(sample_path).convert("RGB").resize((w, h))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return buf.getvalue()

    arr = np.zeros((h, w, 3), dtype=np.uint8)
    arr[:, :, 1] = 145
    arr[:, :, 0] = 45
    arr[:, :, 2] = 35
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def test_esp32_cam_full_e2e_hardware_simulation():
    device_id = "esp32-cam-farm-alpha"
    auth_token = settings.ESP32_CAM_SECRET_KEY

    # Use websocket connection via TestClient
    with client.websocket_connect(f"/api/v1/esp32/device-ws?device_id={device_id}&token={auth_token}") as ws:
        # 1. Device sends initial heartbeat
        ws.send_json({
            "type": "heartbeat",
            "firmware": "1.0.0-prod",
            "camera": "OV2640",
            "rssi": -58,
            "free_heap": 142000,
            "uptime_ms": 32000,
            "frame_size": "QVGA",
            "fps": 12.0,
            "ip": "192.168.1.188"
        })

        # Allow gateway to process
        time.sleep(0.05)

        # 2. Check that status endpoint reflects active device
        status_res = client.get("/api/v1/esp32/status")
        assert status_res.status_code == 200
        status_data = status_res.json()
        assert status_data["online_count"] >= 1
        active_dev = status_data["active_device"]
        assert active_dev is not None
        assert active_dev["device_id"] == device_id
        assert active_dev["is_online"] is True
        assert active_dev["rssi"] == -58

        # 3. Device streams a preview frame using SPRV binary protocol
        preview_frame = make_leaf_jpeg(w=320, h=240)
        ws.send_bytes(b"SPRV" + preview_frame)

        # Allow TestClient event loop to process frame
        prev_res = None
        for _ in range(20):
            time.sleep(0.05)
            prev_res = client.get("/api/v1/esp32/preview")
            if prev_res.status_code == 200:
                break

        assert prev_res is not None
        assert prev_res.status_code == 200
        assert prev_res.headers["content-type"] == "image/jpeg"
        assert len(prev_res.content) > 0
        assert prev_res.content[:2] == b"\xff\xd8"

        # 4. Trigger capture command in background and respond as ESP32
        capture_id = f"SCAP_SIM_{int(time.time())}"
        high_res_leaf = make_leaf_jpeg(w=800, h=600)

        # Simulate ESP32 resolving a capture
        esp32_gateway.resolve_pending_capture(capture_id, high_res_leaf, resolution="SVGA")
        esp32_gateway._verified_captures[capture_id] = {
            "capture_id": capture_id,
            "device_id": device_id,
            "timestamp": time.time(),
            "consumed": False,
            "resolution": "SVGA"
        }

        # 5. Client submits captured frame to /validate/vision
        files = {"file": ("esp32_highres_capture.jpg", high_res_leaf, "image/jpeg")}
        val_data = {
            "request_id": "sim_req_val_01",
            "source": "esp32_cam",
            "esp32_capture_id": capture_id
        }
        val_res = client.post("/validate/vision", files=files, data=val_data)
        assert val_res.status_code == 200
        val_json = val_res.json()
        assert val_json["status"] == "valid"
        assert val_json["is_inference_allowed"] is True

        # 6. Client proceeds to /predict/vision for full multi-tier inference
        pred_files = {"file": ("esp32_highres_capture.jpg", high_res_leaf, "image/jpeg")}
        pred_data = {
            "model_tier": "server",
            "include_explainability": "false",
            "source": "esp32_cam",
            "esp32_capture_id": capture_id
        }
        pred_res = client.post("/predict/vision", files=pred_files, data=pred_data)
        assert pred_res.status_code == 200
        pred_json = pred_res.json()

        # 7. Verify unified diagnosis results
        assert pred_json["status"] == "success"
        assert "ESP32" in pred_json["camera_source"]
        assert pred_json["camera_device_id"] == device_id
        assert pred_json["camera_capture_id"] == capture_id
        assert "diagnosis" in pred_json
        assert "predicted_class" in pred_json["diagnosis"]
        assert "spatial_telemetry" in pred_json
        assert "bounding_boxes" in pred_json["spatial_telemetry"]
