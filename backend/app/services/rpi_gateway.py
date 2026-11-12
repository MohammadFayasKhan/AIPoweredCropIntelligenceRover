"""
Raspberry Pi 3 B+ (OV5647) Camera Gateway & Device Registry Service.

Coordinates authenticated Raspberry Pi camera agent connections over outbound WebSockets,
heartbeat telemetry, bounded single-slot preview frame relays, duplicate-safe capture
commands, and cryptographic capture provenance for AI Powered Crop Intelligence Rover.

Architecture Flow:
  Pi Agent (Picamera2/libcamera) --WSS/WS--> /api/v1/rpi/device-ws --> RPIGatewayManager
    -> Bounded size-1 preview slot (drops stale frames, never queues lag)
    -> Full-res OV5647 still capture with SHA-256 & PIL decode verification
    -> Provenance-authenticated handoff to /validate/vision & /predict/vision
"""

from __future__ import annotations

import asyncio
from collections import OrderedDict
import hashlib
import hmac
import io
import logging
import re
import time
import uuid
from typing import Any, Dict, Optional, Tuple

from fastapi import WebSocket
from PIL import Image

from backend.app.config import settings

logger = logging.getLogger("crop_rover.rpi_gateway")

_DEVICE_ID_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.:-]{0,63}$")
_MAX_VERIFIED_CAPTURES = 20
_MAX_RECENT_REQUESTS = 32
_RECENT_REQUEST_TTL_S = 120.0
_MAX_CAPTURE_CACHE_BYTES = 32 * 1024 * 1024


def validate_device_id(device_id: Optional[str]) -> bool:
    """Validates that device_id is a safe bounded identifier."""
    return bool(device_id and _DEVICE_ID_RE.match(device_id.strip()))


def verify_jpeg_payload(frame_bytes: bytes, expected_sha256: Optional[str] = None) -> Dict[str, Any]:
    """
    Verifies raw still-capture payload format, bounds, optional SHA-256 digest,
    and PIL decoding integrity before accepting into the verified captures store.
    """
    if not frame_bytes:
        raise ValueError("Empty image payload received from Raspberry Pi camera.")

    size_bytes = len(frame_bytes)
    if size_bytes < settings.MIN_UPLOAD_SIZE_BYTES:
        raise ValueError(f"Captured frame too small ({size_bytes} bytes < {settings.MIN_UPLOAD_SIZE_BYTES}).")
    if size_bytes > settings.MAX_UPLOAD_SIZE_BYTES:
        raise ValueError(f"Captured frame exceeds max upload limit ({size_bytes} bytes).")

    if not frame_bytes.startswith(b"\xff\xd8\xff"):
        raise ValueError("Captured frame failed JPEG magic header check.")

    actual_sha256 = hashlib.sha256(frame_bytes).hexdigest()
    if expected_sha256 and not hmac.compare_digest(actual_sha256.lower(), str(expected_sha256).lower()):
        raise ValueError("Captured frame SHA-256 checksum mismatch during transfer.")

    prev_max = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = settings.MAX_IMAGE_PIXELS
    try:
        with Image.open(io.BytesIO(frame_bytes)) as img:
            img.load()
            width, height = img.size
            img_format = img.format or "JPEG"
    except Exception as exc:
        raise ValueError(f"Captured JPEG failed decode integrity check: {exc}") from exc
    finally:
        Image.MAX_IMAGE_PIXELS = prev_max

    if width < settings.MIN_IMAGE_DIMENSION or height < settings.MIN_IMAGE_DIMENSION:
        raise ValueError(f"Captured frame dimensions {width}x{height} below minimum.")
    if width > settings.MAX_IMAGE_DIMENSION or height > settings.MAX_IMAGE_DIMENSION:
        raise ValueError(f"Captured frame dimensions {width}x{height} exceed maximum.")

    return {
        "width": width,
        "height": height,
        "resolution": f"{width}x{height}",
        "format": img_format,
        "file_size": size_bytes,
        "sha256": actual_sha256,
    }


class RPIDeviceRecord:
    """Runtime connection, health, and camera telemetry state for a Raspberry Pi unit."""

    def __init__(self, device_id: str):
        self.device_id: str = device_id
        self.device_name: str = f"Raspberry Pi ({device_id})"
        self.camera_model: str = "OV5647"
        self.sensor_stack: str = "Picamera2/libcamera"
        self.os_version: str = "unknown"
        self.agent_version: str = "unknown"
        # Lifecycle: connecting | online | reconnecting | offline | camera_error
        self.status: str = "connecting"
        self.authenticated: bool = False
        self.camera_ready: bool = False
        self.camera_error: Optional[str] = None
        self.ip_address: Optional[str] = None
        self.wifi_ssid: Optional[str] = None
        self.wifi_signal_dbm: Optional[int] = None
        self.cpu_temp_c: Optional[float] = None
        self.cpu_percent: Optional[float] = None
        self.memory_used_mb: Optional[float] = None
        self.memory_total_mb: Optional[float] = None
        self.uptime_s: int = 0
        self.preview_resolution: str = "640x480"
        self.still_resolution: str = "2592x1944"
        self.fps: float = 0.0
        self.target_fps: int = 12
        self.jpeg_quality: int = 78
        self.stream_active: bool = False
        self.last_heartbeat: float = time.time()
        self.connected_at: Optional[float] = None
        self.disconnected_at: Optional[float] = None
        self.registered_at: Optional[float] = None
        self.websocket: Optional[WebSocket] = None
        self.total_frames_received: int = 0
        self.frames_dropped: int = 0
        self.total_captures_taken: int = 0
        # Bounded size-1 preview slot per device (overwritten in-place)
        self.latest_preview_frame: Optional[bytes] = None
        self.latest_preview_timestamp: float = 0.0
        self.latest_preview_seq: int = 0

    def is_alive(self, timeout_seconds: int) -> bool:
        """True only when device completed authenticated registration and has fresh heartbeat."""
        if not self.authenticated:
            return False
        if self.status in ("offline", "connecting", "reconnecting"):
            return False
        if self.websocket is None:
            return False
        return (time.time() - self.last_heartbeat) <= timeout_seconds

    def effective_status(self, timeout_seconds: int) -> str:
        """Resolves truthful UI connection state: online, connecting, reconnecting, camera_error, offline."""
        if self.status == "connecting" and self.websocket is not None and not self.authenticated:
            return "connecting"
        if self.status == "reconnecting":
            if self.disconnected_at and (time.time() - self.disconnected_at) <= timeout_seconds:
                return "reconnecting"
            return "offline"
        if not self.authenticated or self.websocket is None:
            return "offline"
        if (time.time() - self.last_heartbeat) > timeout_seconds:
            return "offline"
        if not self.camera_ready or self.status == "camera_error":
            return "camera_error"
        return "online"

    def to_telemetry_dict(self, timeout_seconds: int) -> Dict[str, Any]:
        """Serializes verified Raspberry Pi device state without synthetic values."""
        eff_status = self.effective_status(timeout_seconds)
        now = time.time()
        hb_age = round(max(0.0, now - self.last_heartbeat), 1) if self.registered_at else None
        preview_age = (
            round(max(0.0, now - self.latest_preview_timestamp), 2)
            if self.latest_preview_timestamp > 0
            else None
        )
        return {
            "source_type": "rpi_cam",
            "device_id": self.device_id,
            "device_name": self.device_name,
            "status": eff_status,
            "authenticated": bool(self.authenticated and eff_status in ("online", "camera_error")),
            "is_online": eff_status == "online",
            "camera_ready": self.camera_ready,
            "camera_error": self.camera_error,
            "camera_model": self.camera_model,
            "sensor_stack": self.sensor_stack,
            "os_version": self.os_version,
            "agent_version": self.agent_version,
            "ip_address": self.ip_address,
            "wifi_ssid": self.wifi_ssid,
            "wifi_signal_dbm": self.wifi_signal_dbm,
            "cpu_temp_c": self.cpu_temp_c,
            "cpu_percent": self.cpu_percent,
            "memory_used_mb": self.memory_used_mb,
            "memory_total_mb": self.memory_total_mb,
            "uptime_seconds": self.uptime_s,
            "preview_resolution": self.preview_resolution,
            "still_resolution": self.still_resolution,
            "fps": round(self.fps, 1),
            "target_fps": self.target_fps,
            "jpeg_quality": self.jpeg_quality,
            "stream_active": self.stream_active and eff_status == "online",
            "last_heartbeat_ago_s": hb_age,
            "has_preview_frame": self.latest_preview_frame is not None,
            "preview_frame_age_s": preview_age,
            "total_frames": self.total_frames_received,
            "frames_dropped": self.frames_dropped,
            "total_captures": self.total_captures_taken,
        }


class RPIGatewayManager:
    """
    Singleton gateway managing Raspberry Pi OV5647 device registrations,
    authenticated WebSockets, bounded preview caching, and capture provenance.
    """

    def __init__(self):
        self._devices: Dict[str, RPIDeviceRecord] = {}
        self._active_device_id: Optional[str] = None
        self._pending_captures: Dict[str, asyncio.Future] = {}
        self._pending_request_ids: Dict[str, str] = {}
        self._recent_request_captures: OrderedDict[str, Tuple[float, str]] = OrderedDict()
        self._verified_captures: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._lock = asyncio.Lock()

    def authenticate_device(self, token: Optional[str]) -> bool:
        """Constant-time verification of pre-shared Raspberry Pi device token."""
        if not settings.RPI_CAM_ENABLED:
            return False
        expected = (settings.RPI_CAM_SECRET_KEY or "").strip()
        if not expected:
            return False
        if not token:
            return False
        provided = token.strip()
        if provided.lower().startswith("bearer "):
            provided = provided[7:].strip()
        candidates = [c.strip() for c in expected.split(",") if c.strip()]
        return any(hmac.compare_digest(provided, c) for c in candidates)

    def get_or_create_device(self, device_id: str) -> RPIDeviceRecord:
        """Retrieves or initializes a Raspberry Pi device record."""
        if device_id not in self._devices:
            self._devices[device_id] = RPIDeviceRecord(device_id)
            if self._active_device_id is None:
                self._active_device_id = device_id
        return self._devices[device_id]

    def get_device(self, device_id: Optional[str] = None) -> Optional[RPIDeviceRecord]:
        """Returns requested device or the active/best available Raspberry Pi device."""
        if device_id:
            return self._devices.get(device_id)
        timeout = settings.RPI_HEARTBEAT_TIMEOUT_SECONDS
        if self._active_device_id and self._active_device_id in self._devices:
            dev = self._devices[self._active_device_id]
            if dev.effective_status(timeout) in ("online", "camera_error", "connecting", "reconnecting"):
                return dev
        for dev in self._devices.values():
            if dev.is_alive(timeout):
                self._active_device_id = dev.device_id
                return dev
        if self._active_device_id and self._active_device_id in self._devices:
            return self._devices[self._active_device_id]
        return next(iter(self._devices.values()), None)

    def get_active_device(self) -> Optional[RPIDeviceRecord]:
        """Returns the primary Raspberry Pi device only if authenticated and online."""
        timeout = settings.RPI_HEARTBEAT_TIMEOUT_SECONDS
        if self._active_device_id and self._active_device_id in self._devices:
            dev = self._devices[self._active_device_id]
            if dev.is_alive(timeout) and dev.camera_ready:
                return dev
        for dev in self._devices.values():
            if dev.is_alive(timeout) and dev.camera_ready:
                self._active_device_id = dev.device_id
                return dev
        return None

    def get_all_status(self) -> Dict[str, Any]:
        """Returns system-wide Raspberry Pi camera status for dashboard integration."""
        timeout = settings.RPI_HEARTBEAT_TIMEOUT_SECONDS
        target_dev = self.get_device()
        devices_summary = [d.to_telemetry_dict(timeout) for d in self._devices.values()]
        online_count = sum(1 for d in self._devices.values() if d.effective_status(timeout) == "online")
        active_dict = target_dev.to_telemetry_dict(timeout) if target_dev else None

        return {
            "gateway_enabled": settings.RPI_CAM_ENABLED,
            "source_type": "rpi_cam",
            "active_device": active_dict,
            "device_count": len(self._devices),
            "online_count": online_count,
            "devices": devices_summary,
            "has_preview_frame": bool(target_dev and target_dev.latest_preview_frame is not None),
            "preview_frame_age_s": active_dict.get("preview_frame_age_s") if active_dict else None,
        }

    async def begin_connection(self, device_id: str, websocket: WebSocket, ip_address: Optional[str] = None):
        """Marks device as connecting once WebSocket token auth succeeds, awaiting register packet."""
        async with self._lock:
            dev = self.get_or_create_device(device_id)
            if dev.websocket and dev.websocket != websocket:
                try:
                    await dev.websocket.close(code=1000, reason="Superseded by new session")
                except Exception:
                    pass
            dev.websocket = websocket
            dev.ip_address = ip_address or dev.ip_address
            dev.authenticated = False
            dev.status = "connecting"
            dev.connected_at = time.time()
            dev.disconnected_at = None
            self._active_device_id = device_id
            logger.info("Raspberry Pi [%s] socket authenticated from %s; awaiting registration.", device_id, ip_address)

    def complete_registration(self, device_id: str, payload: Dict[str, Any], ip_address: Optional[str] = None) -> Dict[str, Any]:
        """Completes authenticated device registration after register handshake packet."""
        dev = self.get_or_create_device(device_id)
        now = time.time()
        dev.authenticated = True
        dev.registered_at = now
        dev.last_heartbeat = now
        dev.disconnected_at = None
        dev.device_name = str(payload.get("device_name") or dev.device_name)[:80]
        dev.camera_model = str(payload.get("camera_model") or "OV5647")[:40]
        dev.sensor_stack = str(payload.get("sensor_stack") or dev.sensor_stack)[:64]
        dev.os_version = str(payload.get("os_version") or dev.os_version)[:80]
        dev.agent_version = str(payload.get("agent_version") or dev.agent_version)[:40]
        dev.ip_address = payload.get("ip_address") or ip_address or dev.ip_address
        dev.wifi_ssid = payload.get("wifi_ssid") or dev.wifi_ssid
        dev.wifi_signal_dbm = payload.get("wifi_signal_dbm", dev.wifi_signal_dbm)
        dev.preview_resolution = str(payload.get("preview_resolution") or dev.preview_resolution)
        dev.still_resolution = str(payload.get("still_resolution") or dev.still_resolution)
        dev.target_fps = int(payload.get("target_fps", dev.target_fps) or dev.target_fps)
        dev.jpeg_quality = int(payload.get("jpeg_quality", dev.jpeg_quality) or dev.jpeg_quality)
        dev.camera_ready = bool(payload.get("camera_ready", True))
        dev.camera_error = payload.get("camera_error")
        dev.stream_active = bool(payload.get("stream_active", dev.camera_ready))
        dev.status = "online" if dev.camera_ready else "camera_error"
        self._active_device_id = device_id

        logger.info(
            "Raspberry Pi [%s] registered (model=%s, stack=%s, camera_ready=%s, ip=%s).",
            device_id,
            dev.camera_model,
            dev.sensor_stack,
            dev.camera_ready,
            dev.ip_address,
        )
        return {
            "type": "registered",
            "device_id": device_id,
            "status": dev.status,
            "server_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "heartbeat_timeout_s": settings.RPI_HEARTBEAT_TIMEOUT_SECONDS,
        }

    async def handle_disconnect(self, device_id: str, websocket: Optional[WebSocket] = None, reason: str = "disconnected"):
        """Marks device offline or reconnecting upon disconnect and aborts pending captures."""
        async with self._lock:
            dev = self._devices.get(device_id)
            if not dev:
                return
            if websocket is not None and dev.websocket != websocket:
                logger.info("Superseded socket closed for Raspberry Pi [%s]; active session preserved.", device_id)
                return

            dev.websocket = None
            dev.stream_active = False
            dev.authenticated = False
            dev.disconnected_at = time.time()
            dev.status = "reconnecting" if reason == "reconnecting" else "offline"
            logger.warning("Raspberry Pi [%s] disconnected (%s). Status: %s", device_id, reason, dev.status.upper())

        for cid, fut in list(self._pending_captures.items()):
            if not fut.done():
                fut.set_exception(ConnectionError(f"Raspberry Pi [{device_id}] disconnected during capture."))

    def record_heartbeat(self, device_id: str, telemetry: Dict[str, Any]):
        """Updates live hardware and camera health metrics from an authenticated heartbeat."""
        dev = self.get_or_create_device(device_id)
        if not dev.authenticated:
            # Must complete registration before appearing online
            return
        dev.last_heartbeat = time.time()
        dev.camera_ready = bool(telemetry.get("camera_ready", dev.camera_ready))
        dev.camera_error = telemetry.get("camera_error")
        dev.status = "online" if dev.camera_ready else "camera_error"
        dev.ip_address = telemetry.get("ip_address") or dev.ip_address
        dev.wifi_ssid = telemetry.get("wifi_ssid", dev.wifi_ssid)
        dev.wifi_signal_dbm = telemetry.get("wifi_signal_dbm", dev.wifi_signal_dbm)
        dev.cpu_temp_c = telemetry.get("cpu_temp_c", dev.cpu_temp_c)
        dev.cpu_percent = telemetry.get("cpu_percent", dev.cpu_percent)
        dev.memory_used_mb = telemetry.get("memory_used_mb", dev.memory_used_mb)
        dev.memory_total_mb = telemetry.get("memory_total_mb", dev.memory_total_mb)
        dev.uptime_s = int(telemetry.get("uptime_s", dev.uptime_s) or dev.uptime_s)
        dev.preview_resolution = str(telemetry.get("preview_resolution") or dev.preview_resolution)
        dev.still_resolution = str(telemetry.get("still_resolution") or dev.still_resolution)
        if telemetry.get("fps") is not None:
            dev.fps = float(telemetry["fps"])
        if telemetry.get("target_fps") is not None:
            dev.target_fps = int(telemetry["target_fps"])
        if telemetry.get("jpeg_quality") is not None:
            dev.jpeg_quality = int(telemetry["jpeg_quality"])
        if telemetry.get("stream_active") is not None:
            dev.stream_active = bool(telemetry["stream_active"])
        if telemetry.get("frames_dropped") is not None:
            dev.frames_dropped = int(telemetry["frames_dropped"])

    def record_preview_frame(self, device_id: str, frame_bytes: bytes):
        """
        Stores only the latest JPEG preview frame in a bounded size-1 slot.
        Any unconsumed previous frame is overwritten in place so stale frames never queue.
        """
        if not frame_bytes or not frame_bytes.startswith(b"\xff\xd8"):
            return
        if len(frame_bytes) > settings.MAX_UPLOAD_SIZE_BYTES:
            return
        dev = self._devices.get(device_id)
        if not dev or not dev.authenticated:
            return
        now = time.time()
        dev.total_frames_received += 1
        dev.last_heartbeat = now
        if dev.camera_ready:
            dev.status = "online"
        dev.latest_preview_frame = frame_bytes
        dev.latest_preview_timestamp = now
        dev.latest_preview_seq += 1

    def get_latest_preview_frame(self, device_id: Optional[str] = None) -> Tuple[Optional[bytes], float, Optional[str]]:
        """Returns (frame_bytes, age_seconds, device_id) for the requested or active Pi device."""
        dev = self.get_device(device_id)
        if not dev or dev.latest_preview_frame is None:
            return None, 0.0, dev.device_id if dev else None
        age = max(0.0, time.time() - dev.latest_preview_timestamp)
        return dev.latest_preview_frame, age, dev.device_id

    def _prune_caches(self):
        """Enforces bounded temporary capture storage in memory."""
        now = time.time()
        for req_id, (ts, _) in list(self._recent_request_captures.items()):
            if (now - ts) > _RECENT_REQUEST_TTL_S or len(self._recent_request_captures) > _MAX_RECENT_REQUESTS:
                self._recent_request_captures.pop(req_id, None)

        while len(self._verified_captures) > _MAX_VERIFIED_CAPTURES:
            self._verified_captures.popitem(last=False)

        total_bytes = sum(len(v.get("frame_bytes", b"")) for v in self._verified_captures.values())
        while total_bytes > _MAX_CAPTURE_CACHE_BYTES and len(self._verified_captures) > 1:
            _, evicted = self._verified_captures.popitem(last=False)
            total_bytes -= len(evicted.get("frame_bytes", b""))

    def _store_verified_capture(
        self,
        capture_id: str,
        device_id: str,
        camera_model: str,
        frame_bytes: bytes,
        img_meta: Dict[str, Any],
        elapsed_ms: float,
        request_id: Optional[str] = None,
        extra_meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Stores a verified still capture with provenance metadata in bounded cache."""
        record = {
            "capture_id": capture_id,
            "request_id": request_id,
            "device_id": device_id,
            "camera_model": camera_model or "OV5647",
            "camera_source": "Raspberry Pi · OV5647",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "frame_bytes": frame_bytes,
            "file_size": img_meta["file_size"],
            "width": img_meta["width"],
            "height": img_meta["height"],
            "resolution": img_meta["resolution"],
            "format": img_meta["format"],
            "sha256": img_meta["sha256"],
            "transfer_status": "verified",
            "capture_latency_ms": round(elapsed_ms, 1),
            "camera_metadata": extra_meta or {},
            "consumed": False,
        }
        self._verified_captures[capture_id] = record
        if request_id:
            self._recent_request_captures[request_id] = (time.time(), capture_id)
        self._prune_caches()
        return record

    async def request_capture(
        self,
        device_id: Optional[str] = None,
        request_id: Optional[str] = None,
        quality: int = 92,
        timeout: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Dispatches an independent high-resolution still capture command to the Raspberry Pi.
        Deduplicates concurrent or repeated requests bearing the same request_id.
        Never silently substitutes a stale low-res preview frame when the camera is unavailable.
        """
        self._prune_caches()
        clean_req_id = request_id.strip() if isinstance(request_id, str) and request_id.strip() else None

        # 1. Duplicate capture prevention via request_id
        if clean_req_id:
            if clean_req_id in self._recent_request_captures:
                _, prev_cid = self._recent_request_captures[clean_req_id]
                cached = self._verified_captures.get(prev_cid)
                if cached and not cached.get("consumed"):
                    logger.info("Returning deduplicated capture [%s] for request_id=%s", prev_cid, clean_req_id)
                    return cached

            if clean_req_id in self._pending_request_ids:
                existing_cid = self._pending_request_ids[clean_req_id]
                existing_fut = self._pending_captures.get(existing_cid)
                if existing_fut and not existing_fut.done():
                    logger.info("Awaiting in-flight capture [%s] for duplicate request_id=%s", existing_cid, clean_req_id)
                    result = await asyncio.wait_for(asyncio.shield(existing_fut), timeout=float(timeout or settings.RPI_CAPTURE_TIMEOUT_SECONDS))
                    return self._verified_captures[result["capture_id"]]

        # 2. Verify online & camera-ready Raspberry Pi device
        target_dev = self.get_device(device_id)
        hb_timeout = settings.RPI_HEARTBEAT_TIMEOUT_SECONDS
        if not target_dev or not target_dev.authenticated or target_dev.websocket is None:
            raise RuntimeError("Raspberry Pi Camera is disconnected. Power on the Pi and verify its Wi-Fi/gateway connection.")

        eff_status = target_dev.effective_status(hb_timeout)
        if eff_status == "camera_error" or not target_dev.camera_ready:
            err_msg = target_dev.camera_error or "OV5647 camera sensor is unavailable."
            raise RuntimeError(f"Raspberry Pi Camera error on [{target_dev.device_id}]: {err_msg}")
        if eff_status != "online":
            raise RuntimeError(f"Raspberry Pi Camera [{target_dev.device_id}] is {eff_status}; cannot capture.")

        timeout_s = float(timeout or settings.RPI_CAPTURE_TIMEOUT_SECONDS)
        capture_id = f"rpi_{uuid.uuid4().hex[:14]}"
        capture_future = asyncio.get_running_loop().create_future()
        self._pending_captures[capture_id] = capture_future
        if clean_req_id:
            self._pending_request_ids[clean_req_id] = capture_id

        command_payload = {
            "type": "capture",
            "capture_id": capture_id,
            "request_id": clean_req_id,
            "quality": max(50, min(98, int(quality))),
            "timestamp": time.time(),
        }

        start_t = time.time()
        try:
            await target_dev.websocket.send_json(command_payload)
            logger.info("Dispatched still capture [%s] (req=%s) to RPi [%s]", capture_id, clean_req_id, target_dev.device_id)
            result = await asyncio.wait_for(capture_future, timeout=timeout_s)
            elapsed_ms = (time.time() - start_t) * 1000.0
            target_dev.total_captures_taken += 1

            record = self._store_verified_capture(
                capture_id=capture_id,
                device_id=target_dev.device_id,
                camera_model=target_dev.camera_model,
                frame_bytes=result["frame_bytes"],
                img_meta=result["img_meta"],
                elapsed_ms=elapsed_ms,
                request_id=clean_req_id,
                extra_meta=result.get("camera_metadata"),
            )
            return record
        except asyncio.TimeoutError as exc:
            logger.error("Raspberry Pi capture [%s] timed out after %.1fs on [%s]", capture_id, timeout_s, target_dev.device_id)
            raise TimeoutError(f"Raspberry Pi Camera [{target_dev.device_id}] capture timed out after {int(timeout_s)}s.") from exc
        finally:
            self._pending_captures.pop(capture_id, None)
            if clean_req_id:
                self._pending_request_ids.pop(clean_req_id, None)

    def resolve_pending_capture(
        self,
        capture_id: str,
        frame_bytes: bytes,
        expected_sha256: Optional[str] = None,
        camera_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Validates incoming still-capture bytes (format, dimensions, SHA-256, PIL decode)
        and resolves the awaiting capture future.
        """
        img_meta = verify_jpeg_payload(frame_bytes, expected_sha256=expected_sha256)
        fut = self._pending_captures.get(capture_id)
        if fut and not fut.done():
            fut.set_result({
                "capture_id": capture_id,
                "frame_bytes": frame_bytes,
                "img_meta": img_meta,
                "camera_metadata": camera_metadata or {},
            })
        return img_meta

    def fail_pending_capture(self, capture_id: str, error_message: str):
        """Immediately fails an awaiting capture when hardware reports an error."""
        fut = self._pending_captures.get(capture_id)
        if fut and not fut.done():
            fut.set_exception(RuntimeError(error_message or "Raspberry Pi hardware capture failed."))

    def verify_and_consume_provenance(self, capture_id: str, consume: bool = False) -> Optional[Dict[str, Any]]:
        """
        Validates whether a capture_id originated from an authenticated Raspberry Pi capture.
        Returns JSON-safe provenance metadata without raw frame_bytes.
        """
        record = self._verified_captures.get(capture_id)
        if not record:
            return None
        if consume:
            record["consumed"] = True
        return {k: v for k, v in record.items() if k != "frame_bytes"}

    def get_capture_bytes(self, capture_id: str) -> Optional[bytes]:
        """Retrieves verified JPEG bytes for a stored capture_id."""
        record = self._verified_captures.get(capture_id)
        return record.get("frame_bytes") if record else None

    async def dispatch_stream_control(
        self,
        active: bool,
        device_id: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Commands the connected Raspberry Pi to start/pause streaming or update preview params."""
        dev = self.get_device(device_id)
        if not dev or not dev.websocket or not dev.authenticated:
            return False
        payload: Dict[str, Any] = {
            "type": "control_stream",
            "active": bool(active),
            "timestamp": time.time(),
        }
        if config:
            payload.update(config)
        try:
            await dev.websocket.send_json(payload)
            dev.stream_active = bool(active)
            return True
        except Exception as exc:
            logger.error("Failed to dispatch stream control to RPi [%s]: %s", dev.device_id, exc)
            return False

    async def dispatch_camera_config(self, config: Dict[str, Any], device_id: Optional[str] = None) -> bool:
        """Dispatches safe preview/still configuration parameters to the connected Raspberry Pi."""
        dev = self.get_device(device_id)
        if not dev or not dev.websocket or not dev.authenticated:
            return False
        payload = {"type": "camera_config", "timestamp": time.time(), **config}
        try:
            await dev.websocket.send_json(payload)
            return True
        except Exception as exc:
            logger.error("Failed to dispatch camera_config to RPi [%s]: %s", dev.device_id, exc)
            return False


# Global singleton instance
rpi_gateway = RPIGatewayManager()
