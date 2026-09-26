"""
Production Hardening Test Suite for SmartCropVision.
Validates:
1. Liveness and Readiness Probes (/health/live, /health/ready, /health).
2. Decompression-bomb defense and byte boundaries.
3. Request ID header propagation and tracing.
4. Security headers (X-Content-Type-Options, X-Frame-Options, etc.).
5. Concurrency safety: concurrent inference passes do not corrupt state.
6. Explainability toggle: fast-path prediction vs full 9-stage computation.
7. Component readiness semantics (HTTP 200 ready vs HTTP 503 degraded).
"""

import io
from pathlib import Path
import concurrent.futures
import pytest
from fastapi.testclient import TestClient
from PIL import Image
import numpy as np

from backend.app.main import app
from backend.app.config import settings
from backend.app.services.inference_service import inference_engine

client = TestClient(app)

def create_valid_test_leaf_bytes(width=400, height=400, color=(34, 139, 34)):
    """Generates an in-memory valid JPEG leaf image with texture exceeding minimum 2KB size."""
    if width == 400 and height == 400:
        sample_path = Path("cv/test_images/Strawberry-Leaves.jpg")
        if sample_path.exists():
            return sample_path.read_bytes()
    # Create image with texture so compressed JPEG > 2KB
    np_img = np.full((height, width, 3), color, dtype=np.uint8)
    np_img += np.random.randint(0, 30, (height, width, 3), dtype=np.uint8)
    img = Image.fromarray(np.clip(np_img, 0, 255).astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def test_liveness_probe_returns_200():
    """Verify /health/live returns process uptime and status 'alive'."""
    res = client.get("/health/live")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "alive"
    assert "uptime_seconds" in data
    assert data["version"] == settings.VERSION


def test_readiness_probe_returns_readiness_state():
    """Verify /health/ready returns detailed component states."""
    res = client.get("/health/ready")
    # In an environment with trained models, readiness should be 200
    assert res.status_code in [200, 503]
    data = res.json()
    assert "is_ready" in data
    assert "primary_classifier_ready" in data
    assert "components" in data
    assert "device" in data
    assert "tier1_server_classifier" in data["components"]


def test_request_id_header_and_tracing():
    """Verify X-Request-ID is generated and returned on all responses."""
    custom_req_id = "test-custom-trace-999"
    res = client.get("/health", headers={"X-Request-ID": custom_req_id})
    assert res.status_code == 200
    assert res.headers.get("X-Request-ID") == custom_req_id

    # Test auto-generation if not supplied
    res_auto = client.get("/health")
    assert res_auto.status_code == 200
    assert "X-Request-ID" in res_auto.headers
    assert len(res_auto.headers["X-Request-ID"]) > 0


def test_security_headers_present():
    """Verify standard security headers are attached to responses."""
    res = client.get("/health")
    assert res.headers.get("X-Content-Type-Options") == "nosniff"
    assert res.headers.get("X-Frame-Options") == "DENY"
    assert res.headers.get("Referrer-Policy") == "strict-origin-when-cross-origin"


def test_decompression_bomb_prevention():
    """Verify that oversized pixel payloads (decompression bomb attack) are rejected safely."""
    # Attempting to upload dimensions greater than MAX_IMAGE_DIMENSION (4096)
    img_bytes = create_valid_test_leaf_bytes(width=4500, height=4500)
    res = client.post(
        "/predict/vision",
        files={"file": ("bomb.jpg", img_bytes, "image/jpeg")},
        data={"model_tier": "server"}
    )
    assert res.status_code == 400
    data = res.json()
    assert data["status"] == "error"
    assert data["error_code"] == "IMAGE_VALIDATION_FAILED"
    assert "resolution" in data["message"].lower() or "dimension" in data["message"].lower() or "bomb" in data["message"].lower()


def test_fast_prediction_vs_explainability_toggle():
    """
    Verify performance optimization:
    include_explainability=False skips Grad-CAM and returns explainability=None.
    include_explainability=True returns all 9 computational stages.
    """
    img_bytes = create_valid_test_leaf_bytes()

    # Fast prediction path
    res_fast = client.post(
        "/predict/vision",
        files={"file": ("leaf.jpg", img_bytes, "image/jpeg")},
        data={"model_tier": "server", "include_explainability": "false"}
    )
    assert res_fast.status_code == 200
    fast_data = res_fast.json()
    assert fast_data["status"] == "success"
    assert fast_data["explainability"] is None
    assert fast_data["performance_benchmark"]["explainability_ms"] == 0.0

    # Detailed explainability path
    res_explain = client.post(
        "/predict/vision",
        files={"file": ("leaf.jpg", img_bytes, "image/jpeg")},
        data={"model_tier": "server", "include_explainability": "true"}
    )
    assert res_explain.status_code == 200
    explain_data = res_explain.json()
    assert explain_data["status"] == "success"
    assert explain_data["explainability"] is not None
    assert len(explain_data["explainability"]["stages"]) == 9


def test_concurrency_safety_multiple_simultaneous_inferences():
    """Verify that concurrent requests execute safely without state corruption or crash."""
    img_bytes = create_valid_test_leaf_bytes()

    def send_inference(idx):
        response = client.post(
            "/predict/vision",
            files={"file": (f"leaf_{idx}.jpg", img_bytes, "image/jpeg")},
            data={"model_tier": "server", "include_explainability": "false"},
            headers={"X-Request-ID": f"concurrent-req-{idx}"}
        )
        return response.status_code, response.json()

    # Launch 6 concurrent requests
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        futures = [executor.submit(send_inference, i) for i in range(6)]
        results = [f.result() for f in futures]

    for status_code, data in results:
        assert status_code == 200
        assert data["status"] == "success"
        assert data["diagnosis"]["crop"] is not None
        assert "request_id" in data
        assert data["request_id"].startswith("concurrent-req-")


def test_cors_preflight_options():
    """Verify CORS preflight returns correct allowed origins."""
    res = client.options(
        "/predict/vision",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "Content-Type, X-Request-ID"
        }
    )
    assert res.status_code == 200
    assert res.headers.get("access-control-allow-origin") in ["http://localhost:3000", "*"]
