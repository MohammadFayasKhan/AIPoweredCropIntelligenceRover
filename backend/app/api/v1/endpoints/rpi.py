"""
Raspberry Pi 3 B+ (OV5647) Camera API Endpoints.

Provides authenticated device WebSocket gateway, runtime device & camera health status,
low-latency live preview frame and MJPEG relay, duplicate-safe still capture triggers,
and authenticated retry upload fallback for interrupted transfers.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import struct
from typing import Any, Dict, Optional

from fastapi import (
    APIRouter,
    Body,
    File,
    Form,
    Header,
    HTTPException,
    Query,
    Response,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from fastapi.responses import StreamingResponse

from backend.app.config import settings
from backend.app.services.rpi_gateway import rpi_gateway, validate_device_id

router = APIRouter()
logger = logging.getLogger("crop_rover.api.rpi")


def _extract_ws_token(websocket: WebSocket, query_token: Optional[str]) -> Optional[str]:
    """Extracts device auth token from Authorization header, X-Device-Token header, or query."""
    auth_header = websocket.headers.get("authorization")
    if auth_header:
        return auth_header
    header_token = websocket.headers.get("x-device-token")
    if header_token:
        return header_token
    return query_token


@router.websocket("/device-ws")
async def rpi_device_websocket(
    websocket: WebSocket,
    device_id: str = Query("rpi-ov5647-01", description="Persistent Raspberry Pi device identifier"),
    token: Optional[str] = Query(None, description="Pre-shared device token (or pass Authorization/X-Device-Token header)"),
):
    """
    Authenticated persistent outbound WebSocket endpoint for Raspberry Pi 3 B+ OV5647 agent.
    Supports registration handshake, heartbeat telemetry, binary preview frames ('RPRV'),
    binary high-resolution captures ('RCAP'), and capture error notifications.
    """
    if not validate_device_id(device_id):
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="Invalid device_id format")
        return

    supplied_token = _extract_ws_token(websocket, token)
    if not rpi_gateway.authenticate_device(supplied_token):
        logger.warning("Unauthorized Raspberry Pi connection attempt for [%s]", device_id)
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="Invalid device authentication token")
        return

    await websocket.accept()
    client_host = websocket.client.host if websocket.client else "unknown"
    await rpi_gateway.begin_connection(device_id, websocket, client_host)
    disconnect_reason = "disconnected"

    try:
        while True:
            message = await websocket.receive()
            msg_type = message.get("type")

            if msg_type == "websocket.disconnect":
                logger.info("Raspberry Pi WebSocket closed cleanly for [%s]", device_id)
                break

            if msg_type != "websocket.receive":
                continue

            if message.get("text"):
                try:
                    data = json.loads(message["text"])
                    packet_type = data.get("type", "unknown")

                    if packet_type == "register":
                        ack = rpi_gateway.complete_registration(device_id, data, ip_address=client_host)
                        await websocket.send_json(ack)
                        if data.get("camera_ready", True):
                            await websocket.send_json({"type": "control_stream", "active": True})
                    elif packet_type == "heartbeat":
                        rpi_gateway.record_heartbeat(device_id, data)
                    elif packet_type == "stream_status":
                        dev = rpi_gateway.get_device(device_id)
                        if dev and dev.authenticated:
                            dev.stream_active = bool(data.get("active", False))
                    elif packet_type == "capture_error":
                        cid = data.get("capture_id")
                        err_msg = data.get("error") or "Camera capture failed on Raspberry Pi."
                        if cid:
                            rpi_gateway.fail_pending_capture(cid, err_msg)
                    elif packet_type == "capture_response":
                        cid = data.get("capture_id")
                        b64_img = data.get("image_base64")
                        if cid and b64_img:
                            raw_img = base64.b64decode(b64_img)
                            rpi_gateway.resolve_pending_capture(
                                cid,
                                raw_img,
                                expected_sha256=data.get("sha256"),
                                camera_metadata=data.get("camera_metadata"),
                            )
                    elif packet_type == "disconnecting":
                        if data.get("reason") == "reconnecting":
                            disconnect_reason = "reconnecting"
                except ValueError as val_err:
                    logger.warning("Validation error on RPi packet from [%s]: %s", device_id, val_err)
                except Exception as parse_err:
                    logger.warning("Error parsing text packet from RPi [%s]: %s", device_id, parse_err)

            elif message.get("bytes"):
                raw_bytes = message["bytes"]
                if len(raw_bytes) < 4:
                    continue
                header = raw_bytes[:4]

                # RPRV: Bounded latest preview JPEG frame
                if header == b"RPRV":
                    rpi_gateway.record_preview_frame(device_id, raw_bytes[4:])

                # RCAP: 4-byte 'RCAP' + 4-byte uint32 JSON meta length + JSON meta + JPEG bytes
                elif header == b"RCAP" and len(raw_bytes) > 8:
                    meta_len = struct.unpack(">I", raw_bytes[4:8])[0]
                    if meta_len <= 0 or (8 + meta_len) >= len(raw_bytes):
                        logger.warning("Malformed RCAP header from [%s] (meta_len=%d)", device_id, meta_len)
                        continue
                    try:
                        meta_json = json.loads(raw_bytes[8 : 8 + meta_len].decode("utf-8"))
                        jpeg_bytes = raw_bytes[8 + meta_len :]
                        cid = meta_json.get("capture_id")
                        if cid:
                            rpi_gateway.resolve_pending_capture(
                                cid,
                                jpeg_bytes,
                                expected_sha256=meta_json.get("sha256"),
                                camera_metadata=meta_json.get("camera_metadata") or meta_json,
                            )
                    except Exception as cap_err:
                        logger.warning("Failed processing RCAP capture from [%s]: %s", device_id, cap_err)
                        try:
                            meta_cid = json.loads(raw_bytes[8 : 8 + meta_len].decode("utf-8")).get("capture_id")
                            if meta_cid:
                                rpi_gateway.fail_pending_capture(meta_cid, f"Capture verification failed: {cap_err}")
                        except Exception:
                            pass

    except WebSocketDisconnect:
        logger.info("Raspberry Pi [%s] WebSocket disconnected.", device_id)
    except Exception as exc:
        logger.error("Unexpected socket exception for Raspberry Pi [%s]: %s", device_id, exc)
    finally:
        await rpi_gateway.handle_disconnect(device_id, websocket, reason=disconnect_reason)


@router.post("/captures/{capture_id}/upload", summary="Authenticated Retry Upload for Raspberry Pi Still Capture")
async def upload_pending_capture_retry(
    capture_id: str,
    file: UploadFile = File(...),
    sha256: Optional[str] = Form(None),
    camera_metadata: Optional[str] = Form(None),
    authorization: Optional[str] = Header(None),
    x_device_token: Optional[str] = Header(None),
):
    """
    Allows the Raspberry Pi agent to retry uploading an interrupted still capture over HTTPS/HTTP
    while preserving token authentication and full payload verification.
    """
    token = authorization or x_device_token
    if not rpi_gateway.authenticate_device(token):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid device authentication token")

    if capture_id not in rpi_gateway._pending_captures:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Capture transaction not pending or expired")

    try:
        frame_bytes = await file.read()
    finally:
        await file.close()

    parsed_meta: Dict[str, Any] = {}
    if camera_metadata:
        try:
            parsed_meta = json.loads(camera_metadata)
        except Exception:
            parsed_meta = {}

    try:
        img_meta = rpi_gateway.resolve_pending_capture(
            capture_id,
            frame_bytes,
            expected_sha256=sha256,
            camera_metadata=parsed_meta,
        )
    except ValueError as val_err:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(val_err))

    return {"status": "verified", "capture_id": capture_id, **img_meta}


@router.get("/status", summary="Raspberry Pi 3 B+ Camera Runtime Status & Telemetry")
async def get_rpi_status():
    """
    Returns authentic Raspberry Pi OV5647 connectivity, registration state,
    Wi-Fi signal, CPU temperature/memory, stream state, and frame drop metrics.
    """
    return rpi_gateway.get_all_status()


@router.get("/preview", summary="Latest Raspberry Pi OV5647 Live Preview Frame")
async def get_rpi_latest_preview(
    device_id: Optional[str] = Query(None, description="Optional Raspberry Pi device ID"),
):
    """Returns the most recent JPEG preview frame from the Raspberry Pi without buffering lag."""
    frame_bytes, age_s, resolved_id = rpi_gateway.get_latest_preview_frame(device_id=device_id)
    if frame_bytes is None:
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    dev = rpi_gateway.get_device(resolved_id)
    return Response(
        content=frame_bytes,
        media_type="image/jpeg",
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "X-Frame-Age-Seconds": str(round(age_s, 2)),
            "X-Device-Id": resolved_id or "",
            "X-Frame-Resolution": dev.preview_resolution if dev else "",
        },
    )


@router.get("/stream", summary="Raspberry Pi OV5647 Live MJPEG Stream Relay")
async def stream_rpi_mjpeg_preview(
    device_id: Optional[str] = Query(None, description="Optional Raspberry Pi device ID"),
):
    """MJPEG stream relay for browser clients displaying the Raspberry Pi preview."""

    async def frame_generator():
        last_seq = -1
        while True:
            dev = rpi_gateway.get_device(device_id)
            if dev and dev.latest_preview_frame and dev.latest_preview_seq != last_seq:
                last_seq = dev.latest_preview_seq
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + dev.latest_preview_frame + b"\r\n"
                )
            await asyncio.sleep(0.05)

    return StreamingResponse(
        frame_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )


@router.post("/capture", summary="Trigger Full-Quality OV5647 Still Capture on Raspberry Pi")
async def trigger_rpi_capture(
    device_id: Optional[str] = Query(None, description="Target Raspberry Pi device ID"),
    payload: Optional[Dict[str, Any]] = Body(default=None),
):
    """
    Dispatches an authenticated full-resolution OV5647 still capture command to the Raspberry Pi
    without stopping the live preview stream, verifies image integrity, and returns the image
    plus cryptographic capture provenance.
    """
    body = payload or {}
    target_device = device_id or body.get("device_id")
    request_id = body.get("request_id")
    quality = int(body.get("quality", 92) or 92)

    try:
        record = await rpi_gateway.request_capture(
            device_id=target_device,
            request_id=request_id,
            quality=quality,
        )
        b64_img = base64.b64encode(record["frame_bytes"]).decode("ascii")
        return {
            "status": "success",
            "capture_id": record["capture_id"],
            "request_id": record.get("request_id"),
            "device_id": record["device_id"],
            "camera_model": record["camera_model"],
            "camera_source": record["camera_source"],
            "timestamp": record["timestamp"],
            "width": record["width"],
            "height": record["height"],
            "resolution": record["resolution"],
            "file_size": record["file_size"],
            "sha256": record["sha256"],
            "transfer_status": record["transfer_status"],
            "capture_latency_ms": record["capture_latency_ms"],
            "camera_metadata": record.get("camera_metadata", {}),
            "image_base64": f"data:image/jpeg;base64,{b64_img}",
        }
    except TimeoutError as te:
        raise HTTPException(status_code=status.HTTP_504_GATEWAY_TIMEOUT, detail=str(te))
    except RuntimeError as re_err:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(re_err))
    except Exception as exc:
        logger.exception("Error executing Raspberry Pi capture")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))


@router.post("/stream-control", summary="Start, Pause, or Tune Raspberry Pi Preview Stream")
async def control_rpi_stream(payload: Dict[str, Any]):
    """Commands the Raspberry Pi camera agent to start/pause streaming or adjust preview rate."""
    active = bool(payload.get("active", True))
    device_id = payload.get("device_id")
    config_keys = {
        k: payload[k]
        for k in ("preview_width", "preview_height", "fps", "jpeg_quality")
        if k in payload
    }
    ok = await rpi_gateway.dispatch_stream_control(active=active, device_id=device_id, config=config_keys)
    if not ok:
        return {"status": "ignored", "message": "No authenticated Raspberry Pi camera online"}
    return {"status": "success", "stream_active": active, **config_keys}


@router.post("/camera-config", summary="Update Raspberry Pi OV5647 Preview / Capture Settings")
async def update_rpi_camera_config(payload: Dict[str, Any]):
    """Updates safe runtime camera parameters on the connected Raspberry Pi agent."""
    device_id = payload.get("device_id")
    ok = await rpi_gateway.dispatch_camera_config(payload, device_id=device_id)
    if ok:
        return {"status": "success", "config": payload}
    return {"status": "error", "message": "Failed to dispatch camera config. Raspberry Pi may be offline."}

