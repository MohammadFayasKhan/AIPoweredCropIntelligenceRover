"""
Unified Multi-Camera Hardware Registry Endpoint.

Aggregates independent status for all configured field camera sources
(ESP32-CAM OV2640 and Raspberry Pi 3 B+ OV5647) without altering
existing per-device endpoints.
"""

from __future__ import annotations

from typing import Any, Dict, List

from fastapi import APIRouter

from backend.app.services.esp32_gateway import esp32_gateway
from backend.app.services.rpi_gateway import rpi_gateway

router = APIRouter()


@router.get("/status", summary="Unified Multi-Camera Device Registry Status")
async def get_all_cameras_status() -> Dict[str, Any]:
    """
    Returns independent connection states, readiness, and metadata for both
    ESP32-CAM and Raspberry Pi Camera sources.
    """
    esp32_status = esp32_gateway.get_all_status()
    rpi_status = rpi_gateway.get_all_status()

    esp_dev = esp32_status.get("active_device") or (
        esp32_status["devices"][0] if esp32_status.get("devices") else None
    )
    rpi_dev = rpi_status.get("active_device") or (
        rpi_status["devices"][0] if rpi_status.get("devices") else None
    )

    sources: List[Dict[str, Any]] = [
        {
            "source_id": "esp32_cam",
            "label": "ESP32-CAM · OV2640",
            "gateway_enabled": esp32_status.get("gateway_enabled", True),
            "status": esp_dev.get("status", "offline") if esp_dev else "offline",
            "is_online": bool(esp_dev and esp_dev.get("is_online")),
            "online_count": esp32_status.get("online_count", 0),
            "device": esp_dev,
            "preview_url": "/api/v1/esp32/preview",
            "capture_url": "/api/v1/esp32/capture",
            "status_url": "/api/v1/esp32/status",
        },
        {
            "source_id": "rpi_cam",
            "label": "Raspberry Pi · OV5647",
            "gateway_enabled": rpi_status.get("gateway_enabled", True),
            "status": rpi_dev.get("status", "offline") if rpi_dev else "offline",
            "is_online": bool(rpi_dev and rpi_dev.get("is_online")),
            "online_count": rpi_status.get("online_count", 0),
            "device": rpi_dev,
            "preview_url": "/api/v1/rpi/preview",
            "capture_url": "/api/v1/rpi/capture",
            "status_url": "/api/v1/rpi/status",
        },
    ]

    return {
        "total_online_cameras": sum(1 for s in sources if s["is_online"]),
        "sources": sources,
        "esp32": esp32_status,
        "rpi": rpi_status,
    }

