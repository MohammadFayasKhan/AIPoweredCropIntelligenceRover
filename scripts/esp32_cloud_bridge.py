"""
SmartCropVision ESP32-CAM Cloud Bridge Daemon.
Bridges local ESP32-CAM telemetry, live frames, and capture dispatches
from the local gateway (http://127.0.0.1:8000) to the production Azure Cloud
gateway (wss://aipoweredcropintelligencerover.dpdns.org).

Enables the public web dashboard at https://aipoweredcropintelligencerover.dpdns.org
to display "ESP32 CAM Online" and stream live camera video in real time.
"""

import asyncio
import base64
import json
import logging
import sys
import time
import urllib.request
import websockets

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [CLOUD_BRIDGE] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("cloud_bridge")

LOCAL_BASE = "http://127.0.0.1:8000"
CLOUD_WS_URL = "wss://aipoweredcropintelligencerover.dpdns.org/api/v1/esp32/device-ws?device_id=esp32-cam-01&token=smartcropvision-esp32-token-innovex"

async def fetch_local_status():
    try:
        req = urllib.request.urlopen(f"{LOCAL_BASE}/api/v1/esp32/status", timeout=1.5)
        return json.loads(req.read().decode("utf-8"))
    except Exception:
        return None

async def fetch_local_frame():
    try:
        req = urllib.request.urlopen(f"{LOCAL_BASE}/api/v1/esp32/preview", timeout=1.5)
        return req.read()
    except Exception:
        return None

async def trigger_local_capture():
    try:
        req = urllib.request.Request(
            f"{LOCAL_BASE}/api/v1/esp32/capture",
            data=b"{}",
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        res = urllib.request.urlopen(req, timeout=8.0)
        return json.loads(res.read().decode("utf-8"))
    except Exception as e:
        logger.error(f"Local capture trigger failed: {e}")
        return None

async def run_bridge():
    while True:
        try:
            logger.info(f"Connecting to Cloud Gateway: {CLOUD_WS_URL.split('?')[0]}...")
            async with websockets.connect(CLOUD_WS_URL, ping_interval=15, ping_timeout=15) as ws:
                logger.info("Successfully connected to Azure Cloud Gateway! Bridge active.")
                
                last_hb = 0
                last_frame_bytes = None

                async def incoming_handler():
                    """Handles commands from Azure Cloud (e.g. Capture Specimen)."""
                    try:
                        async for message in ws:
                            if isinstance(message, str):
                                try:
                                    cmd = json.loads(message)
                                    if cmd.get("type") == "capture":
                                        cid = cmd.get("capture_id")
                                        logger.info(f"Received capture command [{cid}] from Cloud. Dispatching locally...")
                                        cap_res = await trigger_local_capture()
                                        if cap_res and cap_res.get("frame_bytes_base64"):
                                            resp_payload = {
                                                "type": "capture_response",
                                                "capture_id": cid,
                                                "image_base64": cap_res["frame_bytes_base64"],
                                                "resolution": cap_res.get("resolution", "VGA")
                                            }
                                            await ws.send(json.dumps(resp_payload))
                                            logger.info(f"Dispatched capture [{cid}] back to Cloud successfully!")
                                except Exception as cmd_err:
                                    logger.warning(f"Error handling cloud command: {cmd_err}")
                    except Exception as e:
                        logger.warning(f"Incoming loop closed: {e}")

                incoming_task = asyncio.create_task(incoming_handler())

                while True:
                    now = time.time()
                    local_status = await fetch_local_status()

                    # If local ESP32 is online, relay telemetry and frames
                    if local_status and local_status.get("active_device") and local_status["active_device"].get("is_online"):
                        dev = local_status["active_device"]

                        # 1. Periodic heartbeat to keep Cloud Gateway informed
                        if now - last_hb >= 2.5:
                            hb_pkt = {
                                "type": "heartbeat",
                                "device_id": dev.get("device_id", "esp32-cam-01"),
                                "firmware_version": dev.get("firmware_version", "v1.0.0-prod"),
                                "camera_type": dev.get("camera_type", "OV2640"),
                                "rssi": dev.get("rssi", -45),
                                "free_heap": dev.get("free_heap", 160000),
                                "uptime_ms": (dev.get("uptime_seconds", 60)) * 1000,
                                "frame_size": dev.get("frame_size", "QVGA"),
                                "fps": dev.get("fps", 2.5),
                                "stream_active": True
                            }
                            await ws.send(json.dumps(hb_pkt))
                            last_hb = now

                        # 2. Forward live preview frame at stable 2.5 FPS (~400ms interval)
                        if now - getattr(run_bridge, "last_frame_sent", 0) >= 0.35:
                            frame = await fetch_local_frame()
                            if frame and frame != last_frame_bytes:
                                await ws.send(b"SPRV" + frame)
                                last_frame_bytes = frame
                                run_bridge.last_frame_sent = now

                    await asyncio.sleep(0.1)

        except Exception as e:
            logger.warning(f"Cloud bridge connection lost ({e}). Reconnecting in 3 seconds...")
            await asyncio.sleep(3)

if __name__ == "__main__":
    asyncio.run(run_bridge())
