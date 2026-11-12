"""
Shared Hardware Camera Provenance Helper.

Centralizes server-side capture ID authentication for both ESP32-CAM (OV2640)
and Raspberry Pi 3 B+ (OV5647) without duplicating logic across routes.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
from typing import Any, Dict, Optional, Tuple

from backend.app.services.esp32_gateway import esp32_gateway
from backend.app.services.rpi_gateway import rpi_gateway

logger = logging.getLogger("crop_rover.camera_provenance")


def resolve_camera_provenance(
    source: Optional[str] = "browser_upload",
    esp32_capture_id: Optional[str] = None,
    rpi_capture_id: Optional[str] = None,
    capture_id: Optional[str] = None,
    consume: bool = False,
) -> Tuple[str, Optional[Dict[str, Any]], Optional[bytes]]:
    """
    Authenticates hardware capture provenance and prevents client-side source spoofing.
    Returns (validated_source, camera_metadata, stored_frame_bytes).
    """
    eff_source = (source or "browser_upload").strip()
    eff_rpi_cid = (rpi_capture_id or (capture_id if eff_source == "rpi_cam" else None) or "").strip() or None
    eff_esp_cid = (esp32_capture_id or (capture_id if eff_source == "esp32_cam" else None) or "").strip() or None

    if eff_esp_cid:
        provenance = esp32_gateway.verify_and_consume_provenance(eff_esp_cid, consume=consume)
        if provenance:
            raw_rec = esp32_gateway._verified_captures.get(eff_esp_cid, {})
            meta = {
                "device_id": provenance["device_id"],
                "capture_id": eff_esp_cid,
                "capture_latency_ms": provenance.get("capture_latency_ms"),
                "camera_model": provenance.get("camera_model", "OV2640"),
                "camera_source": "ESP32-CAM · OV2640",
            }
            return "esp32_cam", meta, raw_rec.get("frame_bytes")
        logger.warning("Unverified or expired esp32_capture_id: %s; falling back to browser_upload", eff_esp_cid)
        return "browser_upload", None, None

    if eff_rpi_cid:
        provenance = rpi_gateway.verify_and_consume_provenance(eff_rpi_cid, consume=consume)
        if provenance:
            meta = {
                "device_id": provenance["device_id"],
                "capture_id": eff_rpi_cid,
                "capture_latency_ms": provenance.get("capture_latency_ms"),
                "camera_model": provenance.get("camera_model", "OV5647"),
                "camera_source": "Raspberry Pi · OV5647",
                "resolution": provenance.get("resolution"),
                "sha256": provenance.get("sha256"),
            }
            return "rpi_cam", meta, rpi_gateway.get_capture_bytes(eff_rpi_cid)
        logger.warning("Unverified or expired rpi_capture_id: %s; falling back to browser_upload", eff_rpi_cid)
        return "browser_upload", None, None

    if eff_source in ("browser_camera", "browser_upload"):
        return eff_source, None, None

    # Unauthenticated claim of hardware source without a valid capture ID
    return "browser_upload", None, None


def verify_uploaded_matches_provenance(file_bytes: bytes, camera_metadata: Optional[Dict[str, Any]]) -> bool:
    """
    Confirms uploaded image bytes match the verified capture SHA-256 when present.
    Returns True if no SHA-256 is recorded (e.g. unit test fixtures) or digests match.
    """
    if not camera_metadata:
        return True
    expected_sha = camera_metadata.get("sha256")
    if not expected_sha:
        return True
    actual_sha = hashlib.sha256(file_bytes).hexdigest()
    return hmac.compare_digest(actual_sha.lower(), str(expected_sha).lower())

