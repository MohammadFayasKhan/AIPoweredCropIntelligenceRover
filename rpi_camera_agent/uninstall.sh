#!/usr/bin/env bash
set -euo pipefail

# Safe rollback / uninstall script for the Raspberry Pi OV5647 camera agent.
# Leaves OS, Wi-Fi profiles, camera drivers, and Python system packages untouched.

SERVICE_NAME="crop-rover-rpi-cam.service"

echo "Stopping and disabling ${SERVICE_NAME}..."
sudo systemctl stop "${SERVICE_NAME}" 2>/dev/null || true
sudo systemctl disable "${SERVICE_NAME}" 2>/dev/null || true
sudo rm -f "/etc/systemd/system/${SERVICE_NAME}"
sudo systemctl daemon-reload

echo "Removing /opt/crop-rover-rpi and temporary cache..."
sudo rm -rf /opt/crop-rover-rpi /tmp/crop_rover_rpi_cache

echo "NOTE: /etc/crop-rover-rpi/agent.env and /var/lib/crop-rover-rpi were preserved."
echo "Run 'sudo rm -rf /etc/crop-rover-rpi /var/lib/crop-rover-rpi' if you want a full purge."

