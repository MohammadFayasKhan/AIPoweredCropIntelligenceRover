#!/usr/bin/env bash
set -euo pipefail

# Idempotent installation script for AI Powered Crop Intelligence Rover RPi OV5647 Agent.
# Does NOT run apt upgrade or modify kernel/firmware/camera drivers.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_DIR="/opt/crop-rover-rpi"
CONFIG_DIR="/etc/crop-rover-rpi"
STATE_DIR="/var/lib/crop-rover-rpi"
SERVICE_NAME="crop-rover-rpi-cam.service"

echo "[1/6] Checking Raspberry Pi OS & OV5647 camera stack..."
if ! command -v rpicam-hello >/dev/null 2>&1; then
  echo "ERROR: rpicam-hello not found. Verify Raspberry Pi OS camera tools." >&2
  exit 1
fi

if ! rpicam-hello --list-cameras 2>&1 | grep -qi "ov5647"; then
  echo "WARNING: OV5647 camera sensor was not detected in rpicam-hello --list-cameras." >&2
fi

echo "[2/6] Verifying Python dependencies (picamera2, PIL, websockets)..."
MISSING_PKGS=()
python3 -c "import picamera2" 2>/dev/null || MISSING_PKGS+=("python3-picamera2")
python3 -c "import PIL" 2>/dev/null || MISSING_PKGS+=("python3-pil")
python3 -c "import websockets" 2>/dev/null || MISSING_PKGS+=("python3-websockets")

if [ ${#MISSING_PKGS[@]} -gt 0 ]; then
  echo "Installing missing apt packages (without upgrading existing OS packages): ${MISSING_PKGS[*]}"
  sudo apt-get update -qq
  sudo apt-get install -y --no-upgrade "${MISSING_PKGS[@]}"
fi

echo "[3/6] Installing agent files to ${INSTALL_DIR}..."
sudo mkdir -p "${INSTALL_DIR}" "${CONFIG_DIR}" "${STATE_DIR}"
sudo cp "${SCRIPT_DIR}/agent.py" "${INSTALL_DIR}/agent.py"
sudo chmod 755 "${INSTALL_DIR}/agent.py"
sudo chown -R pi:pi "${INSTALL_DIR}" "${STATE_DIR}"

if [ ! -f "${CONFIG_DIR}/agent.env" ]; then
  sudo cp "${SCRIPT_DIR}/config.env.example" "${CONFIG_DIR}/agent.env"
  sudo chown root:pi "${CONFIG_DIR}/agent.env"
  sudo chmod 640 "${CONFIG_DIR}/agent.env"
  echo "Created ${CONFIG_DIR}/agent.env from template. Update RPI_BACKEND_WS_URL and RPI_DEVICE_TOKEN."
else
  echo "Preserving existing ${CONFIG_DIR}/agent.env"
fi

echo "[4/6] Ensuring user 'pi' is in video group and network-online wait service is enabled..."
sudo usermod -aG video pi || true
if systemctl list-unit-files | grep -q "NetworkManager-wait-online.service"; then
  sudo systemctl enable NetworkManager-wait-online.service >/dev/null 2>&1 || true
fi

echo "[5/6] Installing and enabling systemd unit ${SERVICE_NAME}..."
sudo cp "${SCRIPT_DIR}/${SERVICE_NAME}" "/etc/systemd/system/${SERVICE_NAME}"
sudo systemctl daemon-reload
sudo systemctl enable "${SERVICE_NAME}"
sudo systemctl restart "${SERVICE_NAME}"

echo "[6/6] Checking service health..."
sleep 2
sudo systemctl --no-pager --full status "${SERVICE_NAME}" | head -n 20 || true
echo "Installation complete. View live logs with: journalctl -u ${SERVICE_NAME} -f"

