"""
API v1 Router Aggregator.
Combines health check, models status, and vision diagnosis endpoints.
"""

from fastapi import APIRouter
from backend.app.api.v1.endpoints import health, diagnosis, crops, esp32, rpi, cameras, iot, xiaozhi

api_router = APIRouter()

api_router.include_router(health.router, tags=["Health & Status"])
api_router.include_router(diagnosis.router, tags=["Plant Intelligence Vision"])
api_router.include_router(crops.router, tags=["Crop Recommendation Intelligence"])
api_router.include_router(esp32.router, prefix="/esp32", tags=["ESP32-CAM Field Hardware"])
api_router.include_router(rpi.router, prefix="/rpi", tags=["Raspberry Pi OV5647 Field Camera"])
api_router.include_router(cameras.router, prefix="/cameras", tags=["Multi-Camera Device Registry"])
api_router.include_router(iot.router, prefix="/iot", tags=["AgriRover Field IoT & Telemetry"])
api_router.include_router(xiaozhi.router, prefix="/xiaozhi", tags=["XiaoZhi Voice Assistant & MCP"])
