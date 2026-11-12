# Raspberry Pi 3 B+ (OV5647) Camera Agent

Lightweight headless camera gateway for **AI Powered Crop Intelligence Rover** running on Raspberry Pi OS (Debian 12 Bookworm / Debian 13 Trixie) with `Picamera2` and `libcamera`.

## Features
- Outbound authenticated WebSocket (`WS` on LAN / `WSS` in production) with `Authorization: Bearer` header.
- Dual-stream OV5647 pipeline:
  - `lores` stream (`640x480` or `640x360` YUV420) encoded via hardware V4L2 `MJPEGEncoder` at 10–15 FPS.
  - `main` stream (`2592x1944` or `1296x972` RGB888) for independent full-resolution JPEG still capture without stopping preview.
- Bounded size-1 preview slot with frame dropping and adaptive send-rate throttling.
- `request_id` capture deduplication and HTTP multipart retry queue for interrupted transfers.
- Automatic boot startup via `systemd` after `network-online.target`, exponential backoff + jitter reconnection, and zero ML dependencies.

## Quick Setup on Raspberry Pi

1. Configure persistent Wi-Fi (once, via NetworkManager):
   ```bash
   ./wifi_provision.sh "YourSSID" "YourWiFiPassword"
   ```
2. Install dependencies & systemd service:
   ```bash
   chmod +x install.sh uninstall.sh wifi_provision.sh
   ./install.sh
   ```
3. Edit `/etc/crop-rover-rpi/agent.env` with your backend URL and shared `RPI_DEVICE_TOKEN`:
   - Local development (host Mac/PC LAN IP): `RPI_BACKEND_WS_URL="ws://<HOST_LAN_IP>:8000/api/v1/rpi/device-ws"`
   - Production cloud: `RPI_BACKEND_WS_URL="wss://aipoweredcropintelligencerover.dpdns.org/api/v1/rpi/device-ws"`
4. Restart and verify:
   ```bash
   sudo systemctl restart crop-rover-rpi-cam.service
   python3 /opt/crop-rover-rpi/agent.py --self-test
   ```

