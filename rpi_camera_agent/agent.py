#!/usr/bin/env python3
"""
AI Powered Crop Intelligence Rover - Raspberry Pi 3 B+ (OV5647) Camera Agent.

Lightweight headless device-agent service that:
  1. Auto-initializes the OV5647 CSI camera via the supported Picamera2 / libcamera stack.
  2. Maintains a persistent, authenticated outbound WebSocket connection to the FastAPI backend
     (WSS in production, WS over LAN during local development).
  3. Streams low-latency hardware-encoded preview frames (640x480 / 640x360) with a bounded
     single-slot frame buffer and adaptive rate pacing so stale frames never accumulate.
  4. Executes independent full-quality still captures without stopping the live preview,
     supporting request_id deduplication and HTTP retry uploads for interrupted transfers.
  5. Reports authentic OS, Wi-Fi signal, CPU temperature, memory, and camera health telemetry
     with exponential reconnect backoff + jitter.

Strictly an image acquisition agent: zero ML frameworks (no PyTorch, TensorFlow, or YOLO).
"""

from __future__ import annotations

import argparse
import asyncio
from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import io
import json
import logging
import os
from pathlib import Path
import random
import re
import signal
import socket
import struct
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from typing import Any, Dict, Optional, Tuple

AGENT_VERSION = "1.0.0-rpi-ov5647"

logging.basicConfig(
    level=getattr(logging, os.getenv("RPI_LOG_LEVEL", "INFO").upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] [RPI_AGENT] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("crop_rover.rpi_agent")


def load_env_file(env_path: Path) -> None:
    """Loads key=value pairs from an env file without overriding already-set environment vars."""
    if not env_path.exists():
        return
    try:
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = val
    except Exception as exc:
        logger.warning("Could not parse env file %s: %s", env_path, exc)


def resolve_persistent_device_id() -> str:
    """Returns configured RPI_DEVICE_ID or persists a hardware-derived ID across reboots."""
    env_id = os.getenv("RPI_DEVICE_ID", "").strip()
    if env_id:
        return env_id

    state_candidates = [
        Path("/var/lib/crop-rover-rpi/device_id"),
        Path.home() / ".config" / "crop-rover-rpi" / "device_id",
    ]
    for path in state_candidates:
        try:
            if path.exists():
                existing = path.read_text(encoding="utf-8").strip()
                if existing:
                    return existing
        except Exception:
            continue

    mac_hex = f"{uuid.getnode():012x}"[-6:]
    generated = f"rpi-ov5647-{mac_hex}"
    for path in state_candidates:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(generated, encoding="utf-8")
            break
        except Exception:
            continue
    return generated


def derive_http_base_url(ws_url: str, explicit_http: str = "") -> str:
    """Derives the HTTP/HTTPS backend base URL from the configured WebSocket URL."""
    if explicit_http.strip():
        return explicit_http.strip().rstrip("/")
    parsed = urllib.parse.urlparse(ws_url)
    scheme = "https" if parsed.scheme == "wss" else "http"
    return f"{scheme}://{parsed.netloc}"


def validate_backend_ws_url(ws_url: str, allow_localhost: bool = False) -> None:
    """
    Validates backend WebSocket URL and guards against misconfiguring localhost on the Pi.
    On the Raspberry Pi, localhost refers to the Pi itself rather than the host backend.
    """
    if not ws_url:
        raise ValueError("RPI_BACKEND_WS_URL is required (e.g. ws://192.168.1.10:8000/api/v1/rpi/device-ws).")
    parsed = urllib.parse.urlparse(ws_url)
    if parsed.scheme not in ("ws", "wss"):
        raise ValueError(f"Unsupported WebSocket scheme '{parsed.scheme}'; expected 'ws' or 'wss'.")
    host = (parsed.hostname or "").lower()
    if not host:
        raise ValueError(f"Invalid RPI_BACKEND_WS_URL (missing hostname): {ws_url}")
    if host in ("localhost", "127.0.0.1", "::1") and not allow_localhost:
        raise ValueError(
            "RPI_BACKEND_WS_URL cannot point to localhost/127.0.0.1 on the Raspberry Pi! "
            "Configure your host computer's LAN IP (e.g. ws://192.168.1.50:8000/api/v1/rpi/device-ws) "
            "or production wss:// URL."
        )


def read_system_telemetry() -> Dict[str, Any]:
    """Reads genuine Raspberry Pi OS, Wi-Fi, CPU temperature, and memory metrics."""
    info: Dict[str, Any] = {
        "os_version": "Linux",
        "ip_address": None,
        "wifi_ssid": None,
        "wifi_signal_dbm": None,
        "cpu_temp_c": None,
        "memory_used_mb": None,
        "memory_total_mb": None,
        "uptime_s": 0,
    }

    # OS release pretty name
    try:
        os_rel = Path("/etc/os-release").read_text(encoding="utf-8")
        m = re.search(r'^PRETTY_NAME="?(.*?)"?$', os_rel, re.MULTILINE)
        if m:
            info["os_version"] = m.group(1)
    except Exception:
        pass

    # Local outbound IP
    try:
        out = subprocess.check_output(["hostname", "-I"], text=True, timeout=1.5).strip()
        if out:
            info["ip_address"] = out.split()[0]
    except Exception:
        pass

    # Wi-Fi SSID (check iwgetid, iw, and NetworkManager nmcli)
    for cmd in (["iwgetid", "-r"], ["/sbin/iwgetid", "-r"], ["/usr/sbin/iwgetid", "-r"]):
        try:
            ssid = subprocess.check_output(cmd, text=True, timeout=1.5, stderr=subprocess.DEVNULL).strip()
            if ssid:
                info["wifi_ssid"] = ssid
                break
        except Exception:
            continue

    # Wi-Fi signal RSSI (dBm) via iw or /proc/net/wireless
    for iw_bin in ("iw", "/usr/sbin/iw", "/sbin/iw"):
        try:
            iw_out = subprocess.check_output([iw_bin, "dev", "wlan0", "link"], text=True, timeout=1.5, stderr=subprocess.DEVNULL)
            m_sig = re.search(r"signal:\s*(-?\d+)\s*dBm", iw_out)
            if m_sig:
                info["wifi_signal_dbm"] = int(m_sig.group(1))
            if not info["wifi_ssid"]:
                m_ssid = re.search(r"SSID:\s*(.+)", iw_out)
                if m_ssid:
                    info["wifi_ssid"] = m_ssid.group(1).strip()
            break
        except Exception:
            continue

    if info["wifi_signal_dbm"] is None:
        try:
            wireless = Path("/proc/net/wireless").read_text(encoding="utf-8")
            for line in wireless.splitlines():
                if "wlan0" in line:
                    parts = line.split()
                    if len(parts) >= 4:
                        val = float(parts[3].rstrip("."))
                        info["wifi_signal_dbm"] = int(val) if val < 0 else int(val - 256)
        except Exception:
            pass

    if not info["wifi_ssid"]:
        try:
            nm_ssid = subprocess.check_output(
                ["nmcli", "-t", "-f", "active,ssid", "dev", "wifi"],
                text=True,
                timeout=1.5,
                stderr=subprocess.DEVNULL,
            )
            for line in nm_ssid.splitlines():
                if line.startswith("yes:"):
                    info["wifi_ssid"] = line.split(":", 1)[1].strip() or None
                    break
        except Exception:
            pass

    # SoC temperature
    try:
        raw_temp = Path("/sys/class/thermal/thermal_zone0/temp").read_text(encoding="utf-8").strip()
        info["cpu_temp_c"] = round(int(raw_temp) / 1000.0, 1)
    except Exception:
        pass

    # Memory stats
    try:
        meminfo = Path("/proc/meminfo").read_text(encoding="utf-8")
        vals: Dict[str, int] = {}
        for line in meminfo.splitlines():
            parts = line.split(":")
            if len(parts) == 2:
                num = parts[1].strip().split()[0]
                if num.isdigit():
                    vals[parts[0].strip()] = int(num)
        total_kb = vals.get("MemTotal", 0)
        avail_kb = vals.get("MemAvailable", vals.get("MemFree", 0))
        if total_kb > 0:
            info["memory_total_mb"] = round(total_kb / 1024.0, 1)
            info["memory_used_mb"] = round((total_kb - avail_kb) / 1024.0, 1)
    except Exception:
        pass

    # System uptime
    try:
        uptime_raw = Path("/proc/uptime").read_text(encoding="utf-8").split()[0]
        info["uptime_s"] = int(float(uptime_raw))
    except Exception:
        pass

    return info


@dataclass
class AgentConfig:
    backend_ws_url: str
    backend_http_url: str
    device_id: str
    device_name: str
    device_token: str
    preview_width: int = 640
    preview_height: int = 480
    preview_fps: int = 12
    min_preview_fps: int = 4
    jpeg_quality: int = 78
    still_width: int = 2592
    still_height: int = 1944
    still_quality: int = 92
    heartbeat_interval_s: float = 5.0
    reconnect_base_s: float = 1.5
    reconnect_max_s: float = 30.0
    cache_dir: Path = Path("/tmp/crop_rover_rpi_cache")
    allow_localhost: bool = False

    @classmethod
    def from_env(cls, allow_localhost: bool = False) -> "AgentConfig":
        ws_url = os.getenv("RPI_BACKEND_WS_URL", "").strip()
        validate_backend_ws_url(ws_url, allow_localhost=allow_localhost)
        token = os.getenv("RPI_DEVICE_TOKEN", "").strip()
        if not token:
            raise ValueError("RPI_DEVICE_TOKEN must be set in environment or agent.env.")
        dev_id = resolve_persistent_device_id()
        dev_name = os.getenv("RPI_DEVICE_NAME", f"AgriRover RPi3B+ ({socket.gethostname()})").strip()
        http_url = derive_http_base_url(ws_url, os.getenv("RPI_BACKEND_HTTP_URL", ""))
        return cls(
            backend_ws_url=ws_url,
            backend_http_url=http_url,
            device_id=dev_id,
            device_name=dev_name,
            device_token=token,
            preview_width=int(os.getenv("RPI_PREVIEW_WIDTH", "640")),
            preview_height=int(os.getenv("RPI_PREVIEW_HEIGHT", "480")),
            preview_fps=max(1, min(15, int(os.getenv("RPI_PREVIEW_FPS", "12")))),
            min_preview_fps=max(1, min(10, int(os.getenv("RPI_MIN_PREVIEW_FPS", "4")))),
            jpeg_quality=max(40, min(95, int(os.getenv("RPI_JPEG_QUALITY", "78")))),
            still_width=int(os.getenv("RPI_STILL_WIDTH", "2592")),
            still_height=int(os.getenv("RPI_STILL_HEIGHT", "1944")),
            still_quality=max(60, min(98, int(os.getenv("RPI_STILL_QUALITY", "92")))),
            heartbeat_interval_s=max(2.0, float(os.getenv("RPI_HEARTBEAT_INTERVAL_S", "5.0"))),
            reconnect_base_s=max(0.5, float(os.getenv("RPI_RECONNECT_BASE_S", "1.5"))),
            reconnect_max_s=max(5.0, float(os.getenv("RPI_RECONNECT_MAX_S", "30.0"))),
            cache_dir=Path(os.getenv("RPI_CACHE_DIR", "/tmp/crop_rover_rpi_cache")),
            allow_localhost=allow_localhost,
        )


class LatestFrameSlot:
    """
    Bounded single-slot preview frame buffer.
    Overwriting an unconsumed frame drops it cleanly instead of queueing lag or growing RAM.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._frame: Optional[bytes] = None
        self._timestamp: float = 0.0
        self._seq: int = 0
        self.frames_produced: int = 0
        self.frames_dropped: int = 0
        self._last_read_seq: int = 0

    def put(self, jpeg_bytes: bytes) -> None:
        if not jpeg_bytes or not jpeg_bytes.startswith(b"\xff\xd8"):
            return
        with self._lock:
            if self._frame is not None and self._seq > self._last_read_seq:
                self.frames_dropped += 1
            self._frame = jpeg_bytes
            self._timestamp = time.monotonic()
            self._seq += 1
            self.frames_produced += 1

    def take_if_newer(self, last_seq: int) -> Tuple[Optional[bytes], int, float]:
        with self._lock:
            if self._frame is None or self._seq <= last_seq:
                return None, last_seq, 0.0
            self._last_read_seq = self._seq
            return self._frame, self._seq, self._timestamp


class _SlotOutput(io.BufferedIOBase):
    """File-like adapter feeding Picamera2 encoder output directly into LatestFrameSlot."""

    def __init__(self, slot: LatestFrameSlot):
        super().__init__()
        self._slot = slot

    def writable(self) -> bool:
        return True

    def write(self, buf: bytes) -> int:
        self._slot.put(bytes(buf))
        return len(buf)


class OV5647CameraController:
    """
    Manages the Raspberry Pi OV5647 CSI camera via Picamera2/libcamera.
    Runs a continuous dual-stream configuration:
      - lores (YUV420 @ 640x480 or 640x360) -> hardware MJPEGEncoder preview stream
      - main  (RGB888 @ 2592x1944 / 1296x972) -> independent high-quality still capture
    """

    def __init__(self, config: AgentConfig):
        self.config = config
        self.frame_slot = LatestFrameSlot()
        self._picam2: Any = None
        self._encoder: Any = None
        self._encoder_name: str = "unknown"
        self._lock = threading.Lock()
        self.camera_ready: bool = False
        self.camera_error: Optional[str] = None
        self.camera_model: str = "OV5647"
        self.sensor_stack: str = "Picamera2/libcamera"
        self.stream_active: bool = False
        self._recent_captures: OrderedDict[str, Dict[str, Any]] = OrderedDict()

    def start(self) -> bool:
        """Initializes OV5647 camera and starts hardware-accelerated lores preview encoding."""
        with self._lock:
            if self.camera_ready and self._picam2 is not None:
                return True

            self._cleanup_locked()
            try:
                from picamera2 import Picamera2
                from picamera2.encoders import JpegEncoder, MJPEGEncoder
                from picamera2.outputs import FileOutput

                picam2 = Picamera2()
                props = getattr(picam2, "camera_properties", {}) or {}
                if props.get("Model"):
                    self.camera_model = str(props["Model"]).upper()

                video_cfg = picam2.create_video_configuration(
                    main={"size": (self.config.still_width, self.config.still_height), "format": "RGB888"},
                    lores={"size": (self.config.preview_width, self.config.preview_height), "format": "YUV420"},
                    buffer_count=3,
                    controls={"FrameRate": float(self.config.preview_fps)},
                )
                picam2.configure(video_cfg)

                # Prefer V4L2 hardware MJPEGEncoder (/dev/video11) for low CPU on Pi 3 B+
                output = FileOutput(_SlotOutput(self.frame_slot))
                try:
                    bitrate = self._quality_to_bitrate(self.config.jpeg_quality)
                    encoder = MJPEGEncoder(bitrate=bitrate)
                    picam2.start_recording(encoder, output, name="lores")
                    self._encoder_name = "MJPEGEncoder(HW)"
                except Exception as hw_exc:
                    logger.warning("Hardware MJPEGEncoder unavailable (%s); falling back to JpegEncoder.", hw_exc)
                    encoder = JpegEncoder(q=self.config.jpeg_quality)
                    picam2.start_recording(encoder, output, name="lores")
                    self._encoder_name = "JpegEncoder(SW)"

                self._picam2 = picam2
                self._encoder = encoder
                self.camera_ready = True
                self.camera_error = None
                self.stream_active = True
                logger.info(
                    "Camera ready: model=%s encoder=%s preview=%dx%d@%dfps still=%dx%d",
                    self.camera_model,
                    self._encoder_name,
                    self.config.preview_width,
                    self.config.preview_height,
                    self.config.preview_fps,
                    self.config.still_width,
                    self.config.still_height,
                )
                return True
            except Exception as exc:
                self.camera_ready = False
                self.stream_active = False
                self.camera_error = f"OV5647 initialization failed: {exc}"
                logger.error(self.camera_error)
                self._cleanup_locked()
                return False

    @staticmethod
    def _quality_to_bitrate(quality: int) -> int:
        # Map JPEG quality 40..95 -> ~1.2 Mbps..4.5 Mbps for 640x480@12fps hardware MJPEG
        clamped = max(40, min(95, quality))
        return int(1_200_000 + ((clamped - 40) / 55.0) * 3_300_000)

    def _cleanup_locked(self) -> None:
        if self._picam2 is not None:
            try:
                self._picam2.stop_recording()
            except Exception:
                pass
            try:
                self._picam2.stop()
            except Exception:
                pass
            try:
                self._picam2.close()
            except Exception:
                pass
        self._picam2 = None
        self._encoder = None

    def stop(self) -> None:
        with self._lock:
            self.camera_ready = False
            self.stream_active = False
            self._cleanup_locked()

    def reconfigure_preview(
        self,
        preview_width: Optional[int] = None,
        preview_height: Optional[int] = None,
        fps: Optional[int] = None,
        jpeg_quality: Optional[int] = None,
    ) -> bool:
        """Applies runtime preview changes safely; restarts pipeline only when resolution/quality changes."""
        need_restart = False
        if preview_width and int(preview_width) != self.config.preview_width:
            self.config.preview_width = int(preview_width)
            need_restart = True
        if preview_height and int(preview_height) != self.config.preview_height:
            self.config.preview_height = int(preview_height)
            need_restart = True
        if jpeg_quality and int(jpeg_quality) != self.config.jpeg_quality:
            self.config.jpeg_quality = max(40, min(95, int(jpeg_quality)))
            need_restart = True
        if fps:
            self.config.preview_fps = max(1, min(15, int(fps)))
            if not need_restart and self._picam2 is not None and self.camera_ready:
                try:
                    self._picam2.set_controls({"FrameRate": float(self.config.preview_fps)})
                except Exception:
                    pass

        if need_restart:
            self.stop()
            return self.start()
        return self.camera_ready

    def capture_still(self, capture_id: str, request_id: Optional[str] = None, quality: Optional[int] = None) -> Dict[str, Any]:
        """
        Captures a full-resolution JPEG still from the main stream without stopping preview.
        Deduplicates by request_id when provided.
        """
        if request_id and request_id in self._recent_captures:
            return self._recent_captures[request_id]

        with self._lock:
            if not self.camera_ready or self._picam2 is None:
                raise RuntimeError(self.camera_error or "OV5647 camera is not ready.")

            t0 = time.monotonic()
            eff_quality = max(60, min(98, int(quality or self.config.still_quality)))
            buf = io.BytesIO()
            try:
                self._picam2.options["quality"] = eff_quality
                self._picam2.capture_file(buf, name="main", format="jpeg")
                jpeg_bytes = buf.getvalue()
                raw_meta = {}
                try:
                    raw_meta = dict(self._picam2.capture_metadata() or {})
                except Exception:
                    raw_meta = {}
            except Exception as exc:
                self.camera_error = f"Still capture failed: {exc}"
                raise RuntimeError(self.camera_error) from exc

            elapsed_ms = round((time.monotonic() - t0) * 1000.0, 1)
            sha256 = hashlib.sha256(jpeg_bytes).hexdigest()

            # Read actual decoded resolution via PIL
            from PIL import Image
            with Image.open(io.BytesIO(jpeg_bytes)) as im:
                width, height = im.size

            result = {
                "capture_id": capture_id,
                "request_id": request_id,
                "device_id": self.config.device_id,
                "camera_model": self.camera_model,
                "sensor_stack": self.sensor_stack,
                "encoder": self._encoder_name,
                "width": width,
                "height": height,
                "resolution": f"{width}x{height}",
                "jpeg_quality": eff_quality,
                "sha256": sha256,
                "capture_time_ms": elapsed_ms,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "camera_metadata": {
                    "exposure_time_us": raw_meta.get("ExposureTime"),
                    "analogue_gain": raw_meta.get("AnalogueGain"),
                    "lux": raw_meta.get("Lux"),
                    "colour_temperature": raw_meta.get("ColourTemperature"),
                    "encoder": self._encoder_name,
                },
                "jpeg_bytes": jpeg_bytes,
            }

            if request_id:
                self._recent_captures[request_id] = result
                while len(self._recent_captures) > 8:
                    self._recent_captures.popitem(last=False)

            return result


class RPiCameraAgent:
    """Headless async WebSocket client orchestrating registration, preview, heartbeats, and captures."""

    def __init__(self, config: AgentConfig):
        self.config = config
        self.camera = OV5647CameraController(config)
        self.stop_event = asyncio.Event()
        self._stream_enabled: bool = True
        self._effective_fps: float = float(config.preview_fps)
        self._measured_tx_fps: float = 0.0
        self._total_captures: int = 0
        self._pending_retry_uploads: OrderedDict[str, Dict[str, Any]] = OrderedDict()

    def _build_ws_url(self) -> str:
        parsed = urllib.parse.urlparse(self.config.backend_ws_url)
        q = urllib.parse.parse_qs(parsed.query)
        q["device_id"] = [self.config.device_id]
        if self.config.device_token:
            q["token"] = [self.config.device_token]
        new_query = urllib.parse.urlencode({k: v[0] for k, v in q.items()})
        return urllib.parse.urlunparse(parsed._replace(query=new_query))

    def _build_auth_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.config.device_token}",
            "X-Device-Token": self.config.device_token,
        }

    def _build_register_packet(self) -> Dict[str, Any]:
        sys_info = read_system_telemetry()
        return {
            "type": "register",
            "device_id": self.config.device_id,
            "device_name": self.config.device_name,
            "camera_model": self.camera.camera_model,
            "sensor_stack": f"{self.camera.sensor_stack} ({self.camera._encoder_name})",
            "agent_version": AGENT_VERSION,
            "os_version": sys_info["os_version"],
            "ip_address": sys_info["ip_address"],
            "wifi_ssid": sys_info["wifi_ssid"],
            "wifi_signal_dbm": sys_info["wifi_signal_dbm"],
            "preview_resolution": f"{self.config.preview_width}x{self.config.preview_height}",
            "still_resolution": f"{self.config.still_width}x{self.config.still_height}",
            "target_fps": self.config.preview_fps,
            "jpeg_quality": self.config.jpeg_quality,
            "camera_ready": self.camera.camera_ready,
            "camera_error": self.camera.camera_error,
            "stream_active": self._stream_enabled and self.camera.camera_ready,
        }

    def _build_heartbeat_packet(self) -> Dict[str, Any]:
        sys_info = read_system_telemetry()
        return {
            "type": "heartbeat",
            "device_id": self.config.device_id,
            "camera_ready": self.camera.camera_ready,
            "camera_error": self.camera.camera_error,
            "ip_address": sys_info["ip_address"],
            "wifi_ssid": sys_info["wifi_ssid"],
            "wifi_signal_dbm": sys_info["wifi_signal_dbm"],
            "cpu_temp_c": sys_info["cpu_temp_c"],
            "memory_used_mb": sys_info["memory_used_mb"],
            "memory_total_mb": sys_info["memory_total_mb"],
            "uptime_s": sys_info["uptime_s"],
            "preview_resolution": f"{self.config.preview_width}x{self.config.preview_height}",
            "still_resolution": f"{self.config.still_width}x{self.config.still_height}",
            "fps": round(self._measured_tx_fps, 1),
            "target_fps": int(round(self._effective_fps)),
            "jpeg_quality": self.config.jpeg_quality,
            "stream_active": self._stream_enabled and self.camera.camera_ready,
            "frames_dropped": self.camera.frame_slot.frames_dropped,
            "total_captures": self._total_captures,
        }

    @staticmethod
    def _encode_rcap_packet(capture: Dict[str, Any]) -> bytes:
        meta = {
            "capture_id": capture["capture_id"],
            "request_id": capture.get("request_id"),
            "device_id": capture["device_id"],
            "camera_model": capture["camera_model"],
            "width": capture["width"],
            "height": capture["height"],
            "resolution": capture["resolution"],
            "sha256": capture["sha256"],
            "timestamp": capture["timestamp"],
            "camera_metadata": capture.get("camera_metadata", {}),
        }
        meta_bytes = json.dumps(meta, separators=(",", ":")).encode("utf-8")
        return b"RCAP" + struct.pack(">I", len(meta_bytes)) + meta_bytes + capture["jpeg_bytes"]

    def _queue_retry_capture(self, capture: Dict[str, Any]) -> None:
        self.config.cache_dir.mkdir(parents=True, exist_ok=True)
        cid = capture["capture_id"]
        cache_file = self.config.cache_dir / f"{cid}.jpg"
        cache_file.write_bytes(capture["jpeg_bytes"])
        self._pending_retry_uploads[cid] = {
            "capture_id": cid,
            "sha256": capture["sha256"],
            "camera_metadata": capture.get("camera_metadata", {}),
            "path": cache_file,
        }
        while len(self._pending_retry_uploads) > 3:
            _, evicted = self._pending_retry_uploads.popitem(last=False)
            try:
                Path(evicted["path"]).unlink(missing_ok=True)
            except Exception:
                pass

    def _http_upload_capture_sync(self, entry: Dict[str, Any]) -> bool:
        """Uploads a cached capture via HTTP multipart fallback if WS delivery failed."""
        cid = entry["capture_id"]
        jpeg_bytes = Path(entry["path"]).read_bytes()
        boundary = f"----CropRoverBoundary{uuid.uuid4().hex[:12]}"
        body = bytearray()

        def add_field(name: str, value: str) -> None:
            body.extend(f"--{boundary}\r\n".encode())
            body.extend(f'Content-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode())

        add_field("sha256", entry["sha256"])
        add_field("camera_metadata", json.dumps(entry.get("camera_metadata", {})))
        body.extend(f"--{boundary}\r\n".encode())
        body.extend(f'Content-Disposition: form-data; name="file"; filename="{cid}.jpg"\r\n'.encode())
        body.extend(b"Content-Type: image/jpeg\r\n\r\n")
        body.extend(jpeg_bytes)
        body.extend(f"\r\n--{boundary}--\r\n".encode())

        url = f"{self.config.backend_http_url}/api/v1/rpi/captures/{cid}/upload"
        req = urllib.request.Request(
            url,
            data=bytes(body),
            method="POST",
            headers={
                "Content-Type": f"multipart/form-data; boundary={boundary}",
                **self._build_auth_headers(),
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=10.0) as resp:
                return 200 <= resp.status < 300
        except Exception as exc:
            logger.warning("HTTP retry upload failed for %s: %s", cid, exc)
            return False

    async def _flush_retry_uploads(self) -> None:
        for cid, entry in list(self._pending_retry_uploads.items()):
            ok = await asyncio.to_thread(self._http_upload_capture_sync, entry)
            if ok:
                logger.info("Retry upload succeeded for capture [%s]", cid)
                self._pending_retry_uploads.pop(cid, None)
                try:
                    Path(entry["path"]).unlink(missing_ok=True)
                except Exception:
                    pass

    async def _handle_command(self, ws: Any, cmd: Dict[str, Any]) -> None:
        ctype = cmd.get("type")
        if ctype == "registered":
            logger.info("Backend registration confirmed: %s", cmd.get("status"))
        elif ctype == "control_stream":
            self._stream_enabled = bool(cmd.get("active", True))
            if any(k in cmd for k in ("preview_width", "preview_height", "fps", "jpeg_quality")):
                await asyncio.to_thread(
                    self.camera.reconfigure_preview,
                    cmd.get("preview_width"),
                    cmd.get("preview_height"),
                    cmd.get("fps"),
                    cmd.get("jpeg_quality"),
                )
            await ws.send(json.dumps({"type": "stream_status", "active": self._stream_enabled}))
        elif ctype == "camera_config":
            await asyncio.to_thread(
                self.camera.reconfigure_preview,
                cmd.get("preview_width"),
                cmd.get("preview_height"),
                cmd.get("fps"),
                cmd.get("jpeg_quality"),
            )
        elif ctype == "capture":
            cid = cmd.get("capture_id")
            req_id = cmd.get("request_id")
            quality = cmd.get("quality")
            if not cid:
                return
            try:
                capture = await asyncio.to_thread(self.camera.capture_still, cid, req_id, quality)
                self._total_captures += 1
                packet = self._encode_rcap_packet(capture)
                try:
                    await ws.send(packet)
                    logger.info(
                        "Sent still capture [%s] %s (%d bytes, %.1fms)",
                        cid,
                        capture["resolution"],
                        len(capture["jpeg_bytes"]),
                        capture["capture_time_ms"],
                    )
                except Exception as send_err:
                    logger.warning("WS send failed for capture [%s] (%s); queuing HTTP retry.", cid, send_err)
                    self._queue_retry_capture(capture)
                    raise
            except Exception as cap_err:
                logger.error("Capture [%s] failed: %s", cid, cap_err)
                try:
                    await ws.send(json.dumps({"type": "capture_error", "capture_id": cid, "error": str(cap_err)}))
                except Exception:
                    pass

    async def run(self) -> None:
        import websockets

        # Start camera pipeline headlessly at boot
        self.camera.start()

        attempt = 0
        ws_url = self._build_ws_url()
        headers = self._build_auth_headers()

        while not self.stop_event.is_set():
            # Attempt recovery if camera failed at boot
            if not self.camera.camera_ready:
                await asyncio.to_thread(self.camera.start)

            try:
                logger.info("Connecting to backend gateway: %s", ws_url)
                async with websockets.connect(
                    ws_url,
                    additional_headers=headers,
                    ping_interval=10,
                    ping_timeout=10,
                    max_size=16 * 1024 * 1024,
                    close_timeout=5,
                ) as ws:
                    attempt = 0
                    logger.info("Authenticated WebSocket connected; sending registration handshake.")
                    await ws.send(json.dumps(self._build_register_packet()))
                    await self._flush_retry_uploads()

                    async def recv_loop() -> None:
                        async for raw_msg in ws:
                            if isinstance(raw_msg, str):
                                try:
                                    cmd = json.loads(raw_msg)
                                    await self._handle_command(ws, cmd)
                                except Exception as exc:
                                    logger.warning("Failed handling backend command: %s", exc)

                    async def heartbeat_loop() -> None:
                        while not self.stop_event.is_set():
                            if not self.camera.camera_ready:
                                await asyncio.to_thread(self.camera.start)
                            await ws.send(json.dumps(self._build_heartbeat_packet()))
                            await asyncio.sleep(self.config.heartbeat_interval_s)

                    async def preview_loop() -> None:
                        last_seq = 0
                        tx_window_count = 0
                        tx_window_start = time.monotonic()
                        while not self.stop_event.is_set():
                            frame_budget = 1.0 / max(1.0, self._effective_fps)
                            t_loop = time.monotonic()
                            if self._stream_enabled and self.camera.camera_ready:
                                frame_bytes, new_seq, _ = self.camera.frame_slot.take_if_newer(last_seq)
                                if frame_bytes is not None:
                                    last_seq = new_seq
                                    t_send0 = time.monotonic()
                                    await ws.send(b"RPRV" + frame_bytes)
                                    send_dt = time.monotonic() - t_send0
                                    tx_window_count += 1

                                    # Adaptive rate control: shed load on slow Wi-Fi, recover smoothly
                                    if send_dt > frame_budget * 1.25:
                                        self._effective_fps = max(
                                            float(self.config.min_preview_fps),
                                            self._effective_fps * 0.85,
                                        )
                                    elif send_dt < frame_budget * 0.65:
                                        self._effective_fps = min(
                                            float(self.config.preview_fps),
                                            self._effective_fps + 0.25,
                                        )

                            win_elapsed = time.monotonic() - tx_window_start
                            if win_elapsed >= 2.0:
                                self._measured_tx_fps = tx_window_count / win_elapsed
                                tx_window_count = 0
                                tx_window_start = time.monotonic()

                            sleep_s = max(0.01, frame_budget - (time.monotonic() - t_loop))
                            await asyncio.sleep(sleep_s)

                    tasks = [
                        asyncio.create_task(recv_loop()),
                        asyncio.create_task(heartbeat_loop()),
                        asyncio.create_task(preview_loop()),
                    ]
                    done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
                    for t in pending:
                        t.cancel()
                    for t in done:
                        exc = t.exception()
                        if exc:
                            raise exc

            except asyncio.CancelledError:
                break
            except Exception as exc:
                if self.stop_event.is_set():
                    break
                attempt += 1
                exp_delay = min(self.config.reconnect_max_s, self.config.reconnect_base_s * (2 ** min(attempt, 6)))
                jitter_delay = exp_delay * random.uniform(0.5, 1.0)
                logger.warning("Connection lost (%s). Reconnecting in %.1fs (attempt %d)...", exc, jitter_delay, attempt)
                try:
                    await asyncio.wait_for(self.stop_event.wait(), timeout=jitter_delay)
                except asyncio.TimeoutError:
                    pass

        self.camera.stop()
        logger.info("Raspberry Pi OV5647 agent shut down cleanly.")


def run_self_test(config: AgentConfig) -> int:
    """Verifies physical OV5647 camera detection, preview encoding, still capture, and backend reachability."""
    report: Dict[str, Any] = {
        "agent_version": AGENT_VERSION,
        "device_id": config.device_id,
        "backend_ws_url": config.backend_ws_url,
        "backend_http_url": config.backend_http_url,
        "system": read_system_telemetry(),
        "camera": {},
        "backend_health": {},
    }

    cam = OV5647CameraController(config)
    ok = cam.start()
    report["camera"]["initialized"] = ok
    report["camera"]["camera_model"] = cam.camera_model
    report["camera"]["error"] = cam.camera_error

    if ok:
        frame = None
        seq = 0
        for _ in range(25):
            frame, seq, _ = cam.frame_slot.take_if_newer(0)
            if frame:
                break
            time.sleep(0.1)
        report["camera"]["preview_frame_bytes"] = len(frame) if frame else 0
        report["camera"]["preview_seq"] = seq
        try:
            cap = cam.capture_still(capture_id="self_test_01", request_id="self_test_01")
            report["camera"]["still_resolution"] = cap["resolution"]
            report["camera"]["still_bytes"] = len(cap["jpeg_bytes"])
            report["camera"]["still_capture_ms"] = cap["capture_time_ms"]
        except Exception as exc:
            report["camera"]["still_error"] = str(exc)
            ok = False
    cam.stop()

    try:
        with urllib.request.urlopen(f"{config.backend_http_url}/health/live", timeout=4.0) as resp:
            report["backend_health"] = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        report["backend_health"] = {"reachable": False, "error": str(exc)}

    print(json.dumps(report, indent=2))
    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="AI Powered Crop Intelligence Rover - Raspberry Pi OV5647 Agent")
    parser.add_argument("--env-file", default="/etc/crop-rover-rpi/agent.env", help="Path to environment config file")
    parser.add_argument("--self-test", action="store_true", help="Run camera and backend health diagnostics and exit")
    parser.add_argument("--allow-localhost", action="store_true", help="Allow localhost backend URL (testing only)")
    args = parser.parse_args()

    load_env_file(Path(args.env_file))
    load_env_file(Path(__file__).resolve().parent / "config.env")

    try:
        config = AgentConfig.from_env(allow_localhost=args.allow_localhost)
    except ValueError as err:
        logger.error("Configuration error: %s", err)
        return 2

    if args.self_test:
        return run_self_test(config)

    agent = RPiCameraAgent(config)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    def _handle_sig(*_: Any) -> None:
        logger.info("Received termination signal; stopping agent...")
        loop.call_soon_threadsafe(agent.stop_event.set)

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _handle_sig)
        except NotImplementedError:
            signal.signal(sig, _handle_sig)

    try:
        loop.run_until_complete(agent.run())
    finally:
        loop.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
