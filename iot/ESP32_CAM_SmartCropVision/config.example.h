#ifndef CONFIG_EXAMPLE_H
#define CONFIG_EXAMPLE_H

/**
 * ══════════════════════════════════════════════════════════════════════════════
 * SmartCropVision ESP32-CAM Configuration Template
 * ══════════════════════════════════════════════════════════════════════════════
 * Copy this file to "config.h" and enter your local WiFi and backend parameters.
 * Never commit your real production passwords or secret tokens to source control!
 */

// ── Wireless Network Credentials ─────────────────────────────────────────────
#define WIFI_SSID             "YOUR_WIFI_SSID"
#define WIFI_PASSWORD         "YOUR_WIFI_PASSWORD"

// ── SmartCropVision Backend Server Configuration ─────────────────────────────
// For local development on same WiFi network: enter your computer's LAN IP
// For remote Azure cloud: enter your Azure host name (e.g. "aipoweredcropintelligencerover.dpdns.org")
#define BACKEND_HOST          "192.168.1.100"
#define BACKEND_PORT          8000
#define BACKEND_WS_PATH       "/api/v1/esp32/device-ws"
#define USE_SECURE_WSS        false  // Set true for wss:// on remote HTTPS deployments

// ── Device Identification & Security Token ───────────────────────────────────
#define DEVICE_ID             "esp32-cam-01"
#define DEVICE_TOKEN          "smartcropvision-esp32-token-innovex"
#define FIRMWARE_VERSION      "v1.0.0-prod"

// ── Camera Frame Resolution & Quality ────────────────────────────────────────
// Preview frame size: FRAMESIZE_VGA (640x480) for unified exposure & color consistency
// Capture frame size: FRAMESIZE_VGA (640x480) or FRAMESIZE_SVGA (800x600)
#define PREVIEW_FRAME_SIZE    FRAMESIZE_VGA
#define CAPTURE_FRAME_SIZE    FRAMESIZE_VGA
#define PREVIEW_JPEG_QUALITY  12   // 10-63 (lower = higher quality & larger payload)
#define CAPTURE_JPEG_QUALITY  10   // High detail for neural disease diagnosis

// ── Timing & Operational Parameters ──────────────────────────────────────────
#define HEARTBEAT_INTERVAL_MS 5000  // Send telemetry packet every 5 seconds
#define STREAM_FPS_TARGET     8     // Target preview frame rate (frames per sec)
#define LOCAL_STREAM_PORT     81    // Fallback local HTTP MJPEG server port

#endif // CONFIG_EXAMPLE_H
