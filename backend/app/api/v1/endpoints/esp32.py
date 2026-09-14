"""
ESP32-CAM API Endpoints.
Provides authenticated device WebSocket gateway, runtime telemetry status,
live preview MJPEG stream relay, and asynchronous capture triggers.
"""

import asyncio
import base64
import json
import logging
from typing import Optional, Dict, Any
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query, HTTPException, status, Response
from fastapi.responses import StreamingResponse, JSONResponse

from backend.app.config import settings
from backend.app.services.esp32_gateway import esp32_gateway

router = APIRouter()
logger = logging.getLogger("smartcropvision.api.esp32")


@router.websocket("/device-ws")
async def esp32_device_websocket(
    websocket: WebSocket,
    device_id: str = Query("esp32-cam-01", description="Unique ESP32 device identifier"),
    token: Optional[str] = Query(None, description="Pre-shared authentication token")
):
    """
    Authenticated WebSocket gateway for field ESP32-CAM hardware.
    Handles device telemetry heartbeats, preview frame streaming,
    and high-resolution capture command execution.
    """
    # Authenticate device connection
    if not esp32_gateway.authenticate_device(token):
        logger.warning(f"Unauthorized ESP32 connection attempt for [{device_id}] with token: {token}")
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="Invalid device authentication token")
        return

    await websocket.accept()
    client_host = websocket.client.host if websocket.client else "unknown"
    await esp32_gateway.register_connection(device_id, websocket, client_host)

    try:
        while True:
            message = await websocket.receive()
            msg_type = message.get("type")

            if msg_type == "websocket.disconnect":
                logger.info(f"WebSocket client disconnected cleanly for [{device_id}]")
                break

            if msg_type == "websocket.receive":
                # Ensure device records this websocket as current
                dev = esp32_gateway.get_or_create_device(device_id)
                if dev.websocket != websocket:
                    dev.websocket = websocket
                    dev.status = "online"

                # Handle text/JSON telemetry or status packets
                if "text" in message and message["text"]:
                    try:
                        data = json.loads(message["text"])
                        packet_type = data.get("type", "unknown")

                        if packet_type == "heartbeat":
                            esp32_gateway.record_heartbeat(device_id, data)
                        elif packet_type == "capture_response":
                            # Base64-encoded capture fallback
                            cid = data.get("capture_id")
                            b64_img = data.get("image_base64")
                            if cid and b64_img:
                                frame_bytes = base64.b64decode(b64_img)
                                esp32_gateway.resolve_pending_capture(cid, frame_bytes, data.get("resolution"))
                        elif packet_type == "stream_status":
                            dev = esp32_gateway.get_or_create_device(device_id)
                            dev.stream_active = bool(data.get("active", False))
                    except Exception as parse_err:
                        logger.warning(f"Error parsing text packet from [{device_id}]: {parse_err}")

                # Handle binary preview frames or capture payloads
                elif "bytes" in message and message["bytes"]:
                    raw_bytes = message["bytes"]
                    if len(raw_bytes) >= 4:
                        header = raw_bytes[:4]
                        # SPRV: Live preview JPEG frame
                        if header == b"SPRV":
                            esp32_gateway.record_preview_frame(device_id, raw_bytes[4:])
                        # SCAP: High-resolution capture payload with 16-byte capture_id
                        elif header == b"SCAP" and len(raw_bytes) > 20:
                            cid = raw_bytes[4:20].decode("ascii", errors="ignore").strip()
                            frame_data = raw_bytes[20:]
                            esp32_gateway.resolve_pending_capture(cid, frame_data)
                        else:
                            # Direct JPEG without custom header
                            if raw_bytes[:2] == b"\xff\xd8":
                                esp32_gateway.record_preview_frame(device_id, raw_bytes)

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for [{device_id}]")
    except Exception as e:
        logger.error(f"Unexpected socket exception for [{device_id}]: {e}")
    finally:
        await esp32_gateway.handle_disconnect(device_id, websocket)


@router.get("/status", summary="ESP32-CAM Runtime Status & Telemetry")
async def get_esp32_status():
    """
    Returns authentic device connectivity, WiFi signal (RSSI), free heap,
    stream state, and hardware metrics. Never returns fabricated values.
    """
    return esp32_gateway.get_all_status()


@router.get("/preview", summary="Latest Live Preview Frame")
async def get_latest_preview_frame():
    """Returns the most recent authentic JPEG frame from the active ESP32-CAM."""
    frame_bytes, age_s = esp32_gateway.get_latest_preview_frame()
    if frame_bytes is None:
        # Return a 204 No Content if no frame has arrived yet
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    return Response(
        content=frame_bytes,
        media_type="image/jpeg",
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "X-Frame-Age-Seconds": str(round(age_s, 2)),
        }
    )


@router.get("/stream", summary="Live MJPEG Stream Relay")
async def stream_mjpeg_preview():
    """
    MJPEG stream relay for browser clients.
    Relays live OV2640 preview frames received from the ESP32-CAM WebSocket.
    """
    async def frame_generator():
        last_sent_time = 0.0
        while True:
            frame_bytes, timestamp = esp32_gateway.get_latest_preview_frame()
            if frame_bytes and timestamp != last_sent_time:
                last_sent_time = timestamp
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
                )
            await asyncio.sleep(0.06)  # ~15 FPS polling ceiling

    return StreamingResponse(
        frame_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"}
    )


@router.post("/capture", summary="Trigger Authentic OV2640 Hardware Capture")
async def trigger_hardware_capture(
    device_id: Optional[str] = Query(None, description="Target ESP32 device ID")
):
    """
    Commands the online ESP32-CAM to flush preview buffers, capture a genuine
    high-quality OV2640 frame, and return it with verified cryptographic provenance.
    """
    try:
        record = await esp32_gateway.request_capture(device_id=device_id)
        b64_img = base64.b64encode(record["frame_bytes"]).decode("ascii")

        return {
            "status": "success",
            "capture_id": record["capture_id"],
            "device_id": record["device_id"],
            "camera_model": record["camera_model"],
            "timestamp": record["timestamp"],
            "file_size": record["file_size"],
            "capture_latency_ms": record["capture_latency_ms"],
            "image_base64": f"data:image/jpeg;base64,{b64_img}",
        }
    except TimeoutError as te:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail=str(te)
        )
    except RuntimeError as re:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(re)
        )
    except Exception as exc:
        logger.exception("Error executing ESP32 capture")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc)
        )


@router.post("/stream-control", summary="Start or Stop ESP32 Stream")
async def control_preview_stream(payload: Dict[str, Any]):
    """
    Notifies the connected ESP32-CAM to begin or pause preview frame streaming
    over the WebSocket connection to conserve device bandwidth and CPU.
    """
    active = bool(payload.get("active", True))
    dev = esp32_gateway.get_active_device()
    if not dev or not dev.websocket:
        return {"status": "ignored", "message": "No active device online"}

    try:
        await dev.websocket.send_json({
            "type": "control_stream",
            "active": active
        })
        dev.stream_active = active
        return {"status": "success", "stream_active": active}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@router.post("/camera-config", summary="Update OV2640 Sensor Settings")
async def update_camera_config(payload: Dict[str, Any]):
    """
    Dynamically adjusts OV2640 sensor registers (wb_mode, saturation, contrast, brightness)
    on the connected ESP32-CAM over the live WebSocket channel.
    """
    success = await esp32_gateway.dispatch_camera_config(payload)
    if success:
        return {"status": "success", "config": payload}
    return {"status": "error", "message": "Failed to dispatch camera config. Camera may be offline."}
