#!/usr/bin/env bash
set -euo pipefail

# Provision headless boot auto-connect Wi-Fi on Raspberry Pi OS (Bookworm/Trixie) via NetworkManager.
# Usage:
#   ./wifi_provision.sh "<WIFI_SSID>" "<WIFI_PASSWORD>" [PRIORITY]
# Passwords are passed directly to nmcli on the Pi and never stored in this repository.

if [ $# -lt 2 ]; then
  echo "Usage: $0 <WIFI_SSID> <WIFI_PASSWORD> [AUTOCONNECT_PRIORITY]" >&2
  exit 1
fi

SSID="$1"
PSK="$2"
PRIORITY="${3:-10}"

if ! command -v nmcli >/dev/null 2>&1; then
  echo "ERROR: nmcli (NetworkManager) is required." >&2
  exit 1
fi

echo "Configuring persistent NetworkManager Wi-Fi connection for SSID '${SSID}'..."
if nmcli -t -f NAME con show | grep -Fxq "${SSID}"; then
  sudo nmcli con modify "${SSID}" wifi-sec.key-mgmt wpa-psk wifi-sec.psk "${PSK}" \
    connection.autoconnect yes connection.autoconnect-priority "${PRIORITY}"
else
  sudo nmcli dev wifi connect "${SSID}" password "${PSK}" name "${SSID}"
  sudo nmcli con modify "${SSID}" connection.autoconnect yes connection.autoconnect-priority "${PRIORITY}"
fi

# Disable Wi-Fi power saving so persistent WebSocket streaming does not stall
sudo nmcli con modify "${SSID}" 802-11-wireless.powersave 2 || true

echo "Active network status:"
nmcli -t -f DEVICE,TYPE,STATE,CONNECTION dev

