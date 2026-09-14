"""
Tests for SmartCropVision ESP32-CAM Gateway, Device Registry, and Hardware Telemetry.
"""

import asyncio
import time
import io
from PIL import Image
import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.services.esp32_gateway import esp32_gateway, ESP32DeviceRecord

client = TestClient(app)


def make_dummy_jpeg(w=320, h=240, color=(50, 150, 50)):
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    arr[:, :] = color
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def test_esp32_status_endpoint_empty():
    """Verify status reports clean initial state with zero online devices."""
    res = client.get("/api/v1/esp32/status")
    assert res.status_code == 200
    data = res.json()
    assert data["gateway_enabled"] is True
    assert "devices" in data
    assert "online_count" in data


def test_esp32_device_record_lifecycle():
    """Test device registration, telemetry update, and online state calculation."""
    device_id = "test-esp32-cam-unit-01"
    dev = esp32_gateway.get_or_create_device(device_id)
    assert dev.device_id == device_id

    # Update heartbeat
    now = time.time()
    esp32_gateway.record_heartbeat(device_id, {
        "firmware": "1.0.0-test",
        "camera": "OV2640",
        "rssi": -62,
        "free_heap": 125000,
        "uptime_ms": 45000,
        "frame_size": "QVGA",
        "fps": 12.5,
        "ip": "192.168.1.150"
    })

    assert dev.status == "online"
    assert dev.rssi == -62
    assert dev.free_heap == 125000
    assert dev.uptime_ms == 45000
    assert dev.ip_address == "192.168.1.150"
    assert dev.is_alive(timeout_seconds=30) is True

    telemetry = dev.to_telemetry_dict(timeout_seconds=30)
    assert telemetry["is_online"] is True
    assert telemetry["camera_type"] == "OV2640"
    assert telemetry["rssi"] == -62


def test_esp32_heartbeat_timeout():
    """Verify that a device with stale heartbeat transitions to offline."""
    device_id = "test-esp32-cam-stale"
    dev = esp32_gateway.get_or_create_device(device_id)
    dev.status = "online"
    dev.last_heartbeat = time.time() - 45.0  # 45 seconds ago (> 30s timeout)

    assert dev.is_alive(timeout_seconds=30) is False
    telemetry = dev.to_telemetry_dict(timeout_seconds=30)
    assert telemetry["is_online"] is False
    assert telemetry["status"] == "offline"


def test_esp32_preview_frame_recording_and_endpoint():
    """Test recording a live preview frame and fetching it via /api/v1/esp32/preview."""
    device_id = "test-esp32-cam-streamer"
    jpeg_bytes = make_dummy_jpeg(w=320, h=240, color=(40, 140, 40))

    esp32_gateway.record_preview_frame(device_id, jpeg_bytes)

    res = client.get("/api/v1/esp32/preview")
    assert res.status_code == 200
    assert res.headers["content-type"] == "image/jpeg"
    assert len(res.content) > 0
    assert res.content[:2] == b"\xff\xd8"


def test_esp32_provenance_verification():
    """Test that authentic capture IDs are verified and fake capture IDs are rejected."""
    # Register an authentic capture in the gateway
    fake_cid = "SCAP_AUTHENTIC_TEST_01"
    esp32_gateway._verified_captures[fake_cid] = {
        "capture_id": fake_cid,
        "device_id": "test-esp32-cam-01",
        "timestamp": time.time(),
        "consumed": False,
        "resolution": "SVGA"
    }

    # Verify authentic token
    verified = esp32_gateway.verify_and_consume_provenance(fake_cid, consume=False)
    assert verified is not None
    assert verified["device_id"] == "test-esp32-cam-01"

    # Consume token
    consumed = esp32_gateway.verify_and_consume_provenance(fake_cid, consume=True)
    assert consumed is not None
    assert consumed["consumed"] is True

    # Fake token spoofing test
    spoofed = esp32_gateway.verify_and_consume_provenance("SCAP_SPOOFED_UNAUTHENTIC_99", consume=False)
    assert spoofed is None
