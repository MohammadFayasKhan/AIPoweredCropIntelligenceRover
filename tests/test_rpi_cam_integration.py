"""
Comprehensive Automated Integration Tests for Raspberry Pi 3 B+ (OV5647) Camera Gateway,
Multi-Camera Coexistence (ESP32-CAM + RPi), Provenance Validation, and Diagnosis Parity.
"""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import socket
import struct
import threading
import time
import httpx
from PIL import Image
import numpy as np
import pytest
import uvicorn
import websockets
from fastapi.testclient import TestClient

from backend.app.config import settings
from backend.app.main import app
from backend.app.services.esp32_gateway import esp32_gateway
from backend.app.services.rpi_gateway import rpi_gateway, _MAX_VERIFIED_CAPTURES
from rpi_camera_agent.agent import (
    LatestFrameSlot,
    RPiCameraAgent,
    validate_backend_ws_url,
)

client = TestClient(app)


def make_jpeg_bytes(w: int = 640, h: int = 480, pattern: str = "leaf") -> bytes:
    """Generates a valid JPEG test image (authentic leaf sample or synthetic pattern)."""
    if pattern == "leaf":
        sample_path = Path("frontend/samples/corn__fungal__common_rust.jpg")
        if sample_path.exists():
            img = Image.open(sample_path).convert("RGB").resize((w, h))
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=90)
            return buf.getvalue()
        arr = np.zeros((h, w, 3), dtype=np.uint8)
        arr[:, :, 1] = 155
        arr[:, :, 0] = 45
        arr[:, :, 2] = 40
    elif pattern == "screenshot":
        arr = np.ones((h, w, 3), dtype=np.uint8) * 242
        arr[0:40, :] = [30, 30, 30]
        arr[55:68, 40:260] = [12, 12, 12]
        arr[85:98, 40:420] = [12, 12, 12]
    else:
        arr = np.zeros((h, w, 3), dtype=np.uint8)
        arr[:, :, 1] = 130
        arr[:, :, 0] = 50
        arr[:, :, 2] = 50

    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def encode_rcap(capture_id: str, jpeg_bytes: bytes, device_id: str = "rpi-ov5647-test", request_id: str | None = None) -> bytes:
    """Encodes a binary RCAP still capture packet matching the Pi agent wire protocol."""
    sha = hashlib.sha256(jpeg_bytes).hexdigest()
    with Image.open(io.BytesIO(jpeg_bytes)) as im:
        w, h = im.size
    meta = {
        "capture_id": capture_id,
        "request_id": request_id,
        "device_id": device_id,
        "camera_model": "OV5647",
        "width": w,
        "height": h,
        "resolution": f"{w}x{h}",
        "sha256": sha,
        "timestamp": "2026-11-12T16:00:00Z",
        "camera_metadata": {"exposure_time_us": 12000, "analogue_gain": 1.5},
    }
    meta_bytes = json.dumps(meta).encode("utf-8")
    return b"RCAP" + struct.pack(">I", len(meta_bytes)) + meta_bytes + jpeg_bytes


@pytest.fixture(autouse=True)
def reset_rpi_gateway_state():
    """Ensures isolated gateway state for each test."""
    rpi_gateway._devices.clear()
    rpi_gateway._active_device_id = None
    rpi_gateway._pending_captures.clear()
    rpi_gateway._pending_request_ids.clear()
    rpi_gateway._recent_request_captures.clear()
    rpi_gateway._verified_captures.clear()
    yield
    rpi_gateway._devices.clear()
    rpi_gateway._active_device_id = None
    rpi_gateway._pending_captures.clear()
    rpi_gateway._pending_request_ids.clear()
    rpi_gateway._recent_request_captures.clear()
    rpi_gateway._verified_captures.clear()


def test_rpi_status_initial_offline():
    """Status endpoint starts cleanly with zero online devices and zero misleading indicators."""
    res = client.get("/api/v1/rpi/status")
    assert res.status_code == 200
    data = res.json()
    assert data["gateway_enabled"] is True
    assert data["online_count"] == 0
    assert data["active_device"] is None


def test_rpi_auth_rejects_invalid_token_and_invalid_device_id():
    """Unauthenticated WebSocket connections or bad device IDs are strictly rejected."""
    with pytest.raises(Exception):
        with client.websocket_connect("/api/v1/rpi/device-ws?device_id=rpi-ov5647-01&token=wrong-token"):
            pass

    with pytest.raises(Exception):
        with client.websocket_connect(
            "/api/v1/rpi/device-ws?device_id=bad/device!id",
            headers={"Authorization": f"Bearer {settings.RPI_CAM_SECRET_KEY}"},
        ):
            pass


def test_rpi_registration_lifecycle_and_camera_error_state():
    """
    Device is NOT reported online until authenticated registration completes.
    Camera error state (camera_ready=False) accurately surfaces 'camera_error'.
    """
    headers = {"Authorization": f"Bearer {settings.RPI_CAM_SECRET_KEY}"}
    with client.websocket_connect("/api/v1/rpi/device-ws?device_id=rpi-ov5647-01", headers=headers) as ws:
        # Before register packet, device is connecting (not online)
        pre_status = client.get("/api/v1/rpi/status").json()
        assert pre_status["online_count"] == 0
        assert pre_status["active_device"]["status"] == "connecting"
        assert pre_status["active_device"]["is_online"] is False

        # Register with camera_ready=False -> camera_error
        ws.send_json({
            "type": "register",
            "device_id": "rpi-ov5647-01",
            "device_name": "AgriRover Pi Test",
            "camera_model": "OV5647",
            "camera_ready": False,
            "camera_error": "OV5647 ribbon cable disconnected",
        })
        ack = ws.receive_json()
        assert ack["type"] == "registered"
        assert ack["status"] == "camera_error"

        err_status = client.get("/api/v1/rpi/status").json()
        assert err_status["online_count"] == 0
        assert err_status["active_device"]["status"] == "camera_error"
        assert err_status["active_device"]["camera_ready"] is False
        assert "ribbon cable" in err_status["active_device"]["camera_error"]

        # Capture attempt while in camera_error returns clear 503 error
        cap_res = client.post("/api/v1/rpi/capture", json={"request_id": "err_req_1"})
        assert cap_res.status_code == 503
        assert "ribbon cable" in cap_res.json()["detail"]

        # Recovery heartbeat with camera_ready=True transitions to online
        ws.send_json({
            "type": "heartbeat",
            "camera_ready": True,
            "camera_error": None,
            "wifi_ssid": "Eliyas",
            "wifi_signal_dbm": -52,
            "cpu_temp_c": 41.2,
            "fps": 11.8,
        })
        time.sleep(0.05)
        ok_status = client.get("/api/v1/rpi/status").json()
        assert ok_status["online_count"] == 1
        assert ok_status["active_device"]["status"] == "online"
        assert ok_status["active_device"]["is_online"] is True
        assert ok_status["active_device"]["wifi_signal_dbm"] == -52


def test_rpi_stale_heartbeat_detection_and_reconnect_recovery():
    """Stale heartbeat marks device offline; subsequent reconnect restores online state."""
    headers = {"X-Device-Token": settings.RPI_CAM_SECRET_KEY}
    with client.websocket_connect("/api/v1/rpi/device-ws?device_id=rpi-ov5647-stale", headers=headers) as ws:
        ws.send_json({"type": "register", "camera_ready": True})
        _ = ws.receive_json()
        _ = ws.receive_json()  # control_stream

        dev = rpi_gateway.get_device("rpi-ov5647-stale")
        assert dev is not None
        dev.last_heartbeat = time.time() - (settings.RPI_HEARTBEAT_TIMEOUT_SECONDS + 10)
        st = client.get("/api/v1/rpi/status").json()
        assert st["active_device"]["status"] == "offline"
        assert st["active_device"]["is_online"] is False

    # After disconnect, device status is offline
    after_disc = client.get("/api/v1/rpi/status").json()
    assert after_disc["online_count"] == 0
    assert after_disc["active_device"]["status"] == "offline"

    # Reconnect (simulating Wi-Fi recovery or backend restart recovery)
    with client.websocket_connect("/api/v1/rpi/device-ws?device_id=rpi-ov5647-stale", headers=headers) as ws2:
        ws2.send_json({"type": "register", "camera_ready": True})
        ack2 = ws2.receive_json()
        assert ack2["status"] == "online"
        rec_st = client.get("/api/v1/rpi/status").json()
        assert rec_st["online_count"] == 1
        assert rec_st["active_device"]["status"] == "online"


def test_rpi_preview_frame_delivery_and_bounded_frame_dropping():
    """
    LatestFrameSlot drops unconsumed stale frames in place without buffering lag,
    and /api/v1/rpi/preview serves only the newest frame.
    """
    slot = LatestFrameSlot()
    f1 = make_jpeg_bytes(320, 240, "other")
    f2 = make_jpeg_bytes(640, 480, "leaf")
    f3 = make_jpeg_bytes(640, 480, "leaf")
    slot.put(f1)
    slot.put(f2)
    slot.put(f3)
    assert slot.frames_produced == 3
    assert slot.frames_dropped == 2  # f1 and f2 dropped before being read
    latest, seq, _ = slot.take_if_newer(0)
    assert latest == f3
    assert seq == 3
    none_again, seq2, _ = slot.take_if_newer(seq)
    assert none_again is None
    assert seq2 == 3

    headers = {"Authorization": f"Bearer {settings.RPI_CAM_SECRET_KEY}"}
    with client.websocket_connect("/api/v1/rpi/device-ws?device_id=rpi-ov5647-prev", headers=headers) as ws:
        ws.send_json({"type": "register", "camera_ready": True, "preview_resolution": "640x480"})
        _ = ws.receive_json()
        _ = ws.receive_json()

        ws.send_bytes(b"RPRV" + f1)
        ws.send_bytes(b"RPRV" + f2)
        time.sleep(0.05)

        res = client.get("/api/v1/rpi/preview")
        assert res.status_code == 200
        assert res.headers["content-type"] == "image/jpeg"
        assert res.content == f2  # exact latest frame, unaltered


@pytest.fixture
def live_server_url():
    """Starts an isolated real uvicorn ASGI server on an ephemeral localhost port for live WS+HTTP tests."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()

    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    http_url = f"http://127.0.0.1:{port}"
    ws_url = f"ws://127.0.0.1:{port}"
    deadline = time.time() + 15.0
    while time.time() < deadline:
        try:
            r = httpx.get(f"{http_url}/health/live", timeout=1.0)
            if r.status_code == 200:
                break
        except Exception:
            time.sleep(0.1)
    else:
        raise RuntimeError("Live test uvicorn server failed to start")

    yield http_url, ws_url
    server.should_exit = True
    thread.join(timeout=5.0)


@pytest.mark.anyio
async def test_rpi_capture_deduplication_retry_upload_and_e2e_diagnosis(live_server_url):
    """
    Tests full capture command dispatch -> RCAP high-res upload -> duplicate request_id
    prevention -> /validate/vision -> /predict/vision parity with ESP32 pipeline.
    """
    http_url, ws_url = live_server_url
    headers = {"Authorization": f"Bearer {settings.RPI_CAM_SECRET_KEY}"}
    high_res_leaf = make_jpeg_bytes(1296, 972, "leaf")

    async with websockets.connect(f"{ws_url}/api/v1/rpi/device-ws?device_id=rpi-ov5647-e2e", additional_headers=headers) as ws:
        await ws.send(json.dumps({
            "type": "register",
            "device_id": "rpi-ov5647-e2e",
            "device_name": "Field RPi 3B+",
            "camera_model": "OV5647",
            "camera_ready": True,
            "preview_resolution": "640x480",
            "still_resolution": "1296x972",
        }))
        _ = json.loads(await ws.recv())  # registered
        _ = json.loads(await ws.recv())  # control_stream

        async with httpx.AsyncClient(base_url=http_url, timeout=30.0) as ac:
            import asyncio
            cap_task = asyncio.create_task(ac.post("/api/v1/rpi/capture", json={"request_id": "dedup_req_001", "quality": 92}))
            cmd = json.loads(await ws.recv())
            assert cmd["type"] == "capture"
            cid = cmd["capture_id"]
            assert cmd["request_id"] == "dedup_req_001"

            await ws.send(encode_rcap(cid, high_res_leaf, device_id="rpi-ov5647-e2e", request_id="dedup_req_001"))
            cap_res = await cap_task

            assert cap_res.status_code == 200
            cap_json = cap_res.json()
            assert cap_json["status"] == "success"
            assert cap_json["capture_id"] == cid
            assert cap_json["device_id"] == "rpi-ov5647-e2e"
            assert cap_json["camera_model"] == "OV5647"
            assert cap_json["resolution"] == "1296x972"
            assert cap_json["transfer_status"] == "verified"

            # Duplicate request_id must return exact same capture_id without dispatching another command
            dup_res = await ac.post("/api/v1/rpi/capture", json={"request_id": "dedup_req_001"})
            assert dup_res.status_code == 200
            assert dup_res.json()["capture_id"] == cid

            # Validate captured image via /validate/vision
            val_res = await ac.post(
                "/validate/vision",
                files={"file": ("rpi_leaf.jpg", high_res_leaf, "image/jpeg")},
                data={"source": "rpi_cam", "rpi_capture_id": cid, "request_id": "val_rpi_01"},
            )
            assert val_res.status_code == 200
            val_json = val_res.json()
            assert val_json["status"] == "valid"
            assert val_json["source"] == "rpi_cam"
            assert val_json["is_inference_allowed"] is True

            # Run full diagnosis via /predict/vision
            pred_res = await ac.post(
                "/predict/vision",
                files={"file": ("rpi_leaf.jpg", high_res_leaf, "image/jpeg")},
                data={"model_tier": "server", "source": "rpi_cam", "rpi_capture_id": cid},
            )
            assert pred_res.status_code == 200
            pred_json = pred_res.json()
            assert pred_json["status"] == "success"
            assert pred_json["camera_source"] == "Raspberry Pi · OV5647"
            assert pred_json["camera_device_id"] == "rpi-ov5647-e2e"
            assert pred_json["camera_capture_id"] == cid
            assert "diagnosis" in pred_json
            assert "spatial_telemetry" in pred_json
            assert "bounding_boxes" in pred_json["spatial_telemetry"]


@pytest.mark.anyio
async def test_rpi_http_retry_upload_fallback_and_integrity_checks(live_server_url):
    """HTTP retry endpoint authenticates token, validates JPEG & SHA-256, and resolves pending capture."""
    http_url, ws_url = live_server_url
    headers = {"Authorization": f"Bearer {settings.RPI_CAM_SECRET_KEY}"}
    high_res = make_jpeg_bytes(640, 480, "leaf")
    sha = hashlib.sha256(high_res).hexdigest()

    async with websockets.connect(f"{ws_url}/api/v1/rpi/device-ws?device_id=rpi-ov5647-retry", additional_headers=headers) as ws:
        await ws.send(json.dumps({"type": "register", "camera_ready": True}))
        _ = json.loads(await ws.recv())
        _ = json.loads(await ws.recv())

        async with httpx.AsyncClient(base_url=http_url, timeout=20.0) as ac:
            import asyncio
            cap_task = asyncio.create_task(ac.post("/api/v1/rpi/capture", json={"request_id": "retry_req_01"}))
            cmd = json.loads(await ws.recv())
            cid = cmd["capture_id"]

            # Simulate corrupted checksum upload rejected first
            bad_up = await ac.post(
                f"/api/v1/rpi/captures/{cid}/upload",
                headers=headers,
                files={"file": ("cap.jpg", high_res, "image/jpeg")},
                data={"sha256": "0" * 64},
            )
            assert bad_up.status_code == 400

            # Valid retry upload succeeds and resolves the pending capture
            good_up = await ac.post(
                f"/api/v1/rpi/captures/{cid}/upload",
                headers=headers,
                files={"file": ("cap.jpg", high_res, "image/jpeg")},
                data={"sha256": sha},
            )
            assert good_up.status_code == 200
            assert good_up.json()["status"] == "verified"

            cap_res = await cap_task
            assert cap_res.status_code == 200
            assert cap_res.json()["sha256"] == sha


@pytest.mark.anyio
async def test_rpi_non_plant_rejected_and_spoof_blocked(live_server_url):
    """
    Authentic Raspberry Pi captures of screenshots/non-plants are still strictly rejected,
    and tampered image bytes paired with an authentic rpi_capture_id are rejected.
    """
    http_url, ws_url = live_server_url
    headers = {"Authorization": f"Bearer {settings.RPI_CAM_SECRET_KEY}"}
    screenshot = make_jpeg_bytes(640, 480, "screenshot")
    leaf = make_jpeg_bytes(640, 480, "leaf")

    async with websockets.connect(f"{ws_url}/api/v1/rpi/device-ws?device_id=rpi-ov5647-sec", additional_headers=headers) as ws:
        await ws.send(json.dumps({"type": "register", "camera_ready": True}))
        _ = json.loads(await ws.recv())
        _ = json.loads(await ws.recv())

        async with httpx.AsyncClient(base_url=http_url, timeout=20.0) as ac:
            import asyncio
            cap_task = asyncio.create_task(ac.post("/api/v1/rpi/capture", json={"request_id": "sec_1"}))
            cmd = json.loads(await ws.recv())
            cid = cmd["capture_id"]
            await ws.send(encode_rcap(cid, screenshot, device_id="rpi-ov5647-sec"))
            cap_res = await cap_task
            assert cap_res.status_code == 200

            # Non-plant screenshot must still be rejected
            val_res = (await ac.post(
                "/validate/vision",
                files={"file": ("screen.jpg", screenshot, "image/jpeg")},
                data={"source": "rpi_cam", "rpi_capture_id": cid},
            )).json()
            assert val_res["status"] == "rejected"
            assert val_res["is_inference_allowed"] is False

            # Tampered file bytes mismatched with capture SHA-256 must fail HTTP 400
            mismatch = await ac.post(
                "/predict/vision",
                files={"file": ("swapped.jpg", leaf, "image/jpeg")},
                data={"source": "rpi_cam", "rpi_capture_id": cid},
            )
            assert mismatch.status_code == 400

    # Spoofed capture ID without authentic provenance downgrades to browser_upload
    spoof_res = client.post(
        "/validate/vision",
        files={"file": ("leaf.jpg", leaf, "image/jpeg")},
        data={"source": "rpi_cam", "rpi_capture_id": "rpi_fake_spoof_999"},
    ).json()
    assert spoof_res["source"] == "browser_upload"


def test_simultaneous_esp32_and_rpi_coexistence_and_independence():
    """
    ESP32-CAM and Raspberry Pi operate simultaneously as independent cameras;
    disconnecting one does not disturb the other.
    """
    esp_token = settings.ESP32_CAM_SECRET_KEY
    rpi_headers = {"Authorization": f"Bearer {settings.RPI_CAM_SECRET_KEY}"}

    with client.websocket_connect(f"/api/v1/esp32/device-ws?device_id=esp32-coexist-01&token={esp_token}") as ws_esp:
        ws_esp.send_json({"type": "heartbeat", "camera": "OV2640", "rssi": -55})

        with client.websocket_connect("/api/v1/rpi/device-ws?device_id=rpi-coexist-01", headers=rpi_headers) as ws_rpi:
            ws_rpi.send_json({"type": "register", "camera_model": "OV5647", "camera_ready": True})
            _ = ws_rpi.receive_json()
            _ = ws_rpi.receive_json()

            unified = client.get("/api/v1/cameras/status").json()
            assert unified["total_online_cameras"] == 2
            sources = {s["source_id"]: s for s in unified["sources"]}
            assert sources["esp32_cam"]["is_online"] is True
            assert sources["rpi_cam"]["is_online"] is True

        # After RPi disconnects, ESP32-CAM remains online & unaffected
        after = client.get("/api/v1/cameras/status").json()
        sources_after = {s["source_id"]: s for s in after["sources"]}
        assert sources_after["esp32_cam"]["is_online"] is True
        assert sources_after["rpi_cam"]["is_online"] is False


def test_rpi_bounded_capture_cache_and_agent_guardrails():
    """Verifies bounded capture storage, localhost URL guard, and zero ML dependencies in Pi agent."""
    sample = make_jpeg_bytes(320, 240, "leaf")
    for i in range(_MAX_VERIFIED_CAPTURES + 5):
        rpi_gateway._store_verified_capture(
            capture_id=f"rpi_bound_{i}",
            device_id="rpi-bound",
            camera_model="OV5647",
            frame_bytes=sample,
            img_meta={"file_size": len(sample), "width": 320, "height": 240, "resolution": "320x240", "format": "JPEG", "sha256": "abc"},
            elapsed_ms=10.0,
        )
    assert len(rpi_gateway._verified_captures) <= _MAX_VERIFIED_CAPTURES

    with pytest.raises(ValueError, match="localhost"):
        validate_backend_ws_url("ws://localhost:8000/api/v1/rpi/device-ws", allow_localhost=False)
    validate_backend_ws_url("ws://192.168.1.50:8000/api/v1/rpi/device-ws", allow_localhost=False)
    validate_backend_ws_url("wss://aipoweredcropintelligencerover.dpdns.org/api/v1/rpi/device-ws", allow_localhost=False)

    agent_src = Path("rpi_camera_agent/agent.py").read_text(encoding="utf-8")
    for forbidden in ("import torch", "import tensorflow", "import ultralytics", "import cv2"):
        assert forbidden not in agent_src


def test_rpi_indoor_selfie_and_cool_wall_strictly_rejected():
    """
    Regression test: Indoor scenes with warm skin tones plus cool/tinted walls or
    wooden cupboards captured via OV5647 (which can yield false positive 2G-R-B > 6)
    must be strictly rejected by both /validate/vision and /predict/vision.
    """
    import hashlib as _hl
    import time as _t
    import cv2

    # Construct a realistic 640x480 indoor selfie: cool/tinted room backdrop + warm wall + centered skin face + dark shirt
    scene = np.full((480, 640, 3), (78, 81, 76), dtype=np.uint8)  # BGR cool grey indoor wall (slightly higher G than R/B)
    scene[:, :160] = (50, 63, 64)  # Warm indoor wood/cupboard tone
    scene[300:, 180:460] = (32, 30, 28)  # Dark shirt
    # Centered skin-tone face (~18% of pixels)
    cv2.ellipse(scene, (320, 200), (110, 140), 0, 0, 360, (125, 155, 205), -1)
    _, enc = cv2.imencode(".jpg", scene, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    selfie_bytes = enc.tobytes()

    capture_id = f"rpi_selfie_reg_{int(_t.time())}"
    rpi_gateway._store_verified_capture(
        capture_id=capture_id,
        device_id="rpi-ov5647-01",
        camera_model="OV5647",
        frame_bytes=selfie_bytes,
        img_meta={"file_size": len(selfie_bytes), "width": 640, "height": 480, "resolution": "640x480", "format": "JPEG", "sha256": _hl.sha256(selfie_bytes).hexdigest()},
        elapsed_ms=20.0,
    )

    val_res = client.post(
        "/validate/vision",
        files={"file": ("rpi_selfie.jpg", selfie_bytes, "image/jpeg")},
        data={"source": "rpi_cam", "rpi_capture_id": capture_id},
    )
    assert val_res.status_code == 200
    val_json = val_res.json()
    assert val_json["status"] == "rejected"
    assert val_json["is_inference_allowed"] is False
    assert val_json["validation_status"] == "INVALID_NON_PLANT_IMAGE"

    pred_res = client.post(
        "/predict/vision",
        files={"file": ("rpi_selfie.jpg", selfie_bytes, "image/jpeg")},
        data={"source": "rpi_cam", "rpi_capture_id": capture_id, "include_explainability": "false"},
    )
    assert pred_res.status_code == 200
    pred_json = pred_res.json()
    assert pred_json["status"] == "rejected"
    assert pred_json.get("diagnosis") is None
