/*
 * ══════════════════════════════════════════════════════════════════════════════
 *   SmartCropVision Production Firmware: ESP32-CAM OV2640 Field Unit
 * ══════════════════════════════════════════════════════════════════════════════
 *
 *   Team: Innovex
 *   Target Board: AI-Thinker ESP32-CAM (with OV2640 camera module & baseboard)
 *   Target Framework: Arduino-ESP32 Core (v2.0.x or v3.x)
 *
 *   Architectural Role:
 *     The ESP32-CAM acts strictly as an image acquisition and field streaming device.
 *     It does NOT perform on-device neural network disease inference.
 *     All classification (EfficientNetV2-S), detection (YOLO26), and segmentation
 *     (Mobile-UNet) remain securely hosted on the SmartCropVision server backend.
 *
 *   Communication Paths:
 *     1. Outbound WebSocket (/api/v1/esp32/device-ws):
 *        Connects to backend gateway for persistent telemetry, live preview relay,
 *        and capture dispatch. Works seamlessly through NAT/firewalls in Azure cloud.
 *     2. Local HTTP Server (Port 81):
 *        Serves direct local MJPEG stream (/stream) and snapshots (/capture)
 *        for direct LAN debugging without internet access.
 *
 *   Required Arduino Libraries:
 *     1. ArduinoWebsockets by Gil Maimon (install via Arduino Library Manager)
 *     2. ArduinoJson v6.x or v7.x by Benoit Blanchon
 * ══════════════════════════════════════════════════════════════════════════════
 */

#include "esp_camera.h"
#include <WiFi.h>
#include <WebServer.h>
#include <ArduinoWebsockets.h>
#include <ArduinoJson.h>

// ESP32 brownout detector control registers
#include "soc/soc.h"
#include "soc/rtc_cntl_reg.h"

#include "camera_pins.h"
#if __has_include("config.h")
  #include "config.h"
#else
  #include "config.example.h"
#endif

using namespace websockets;

// ── Global Runtime State ──────────────────────────────────────────────────────
WebsocketsClient wsClient;
WebServer localServer(LOCAL_STREAM_PORT);

bool g_cameraInitialized = false;
bool g_streamActive = true;
unsigned long g_lastHeartbeatMs = 0;
unsigned long g_lastFrameTimeMs = 0;
unsigned long g_lastWiFiAttemptMs = 0;
unsigned long g_wifiRetryIntervalMs = 10000;
bool g_wifiConnecting = false;
bool g_wifiWasConnected = false;
unsigned long g_totalFramesSent = 0;
unsigned long g_totalCapturesTaken = 0;
float g_currentFps = 0.0;
int g_framesInLastSecond = 0;
unsigned long g_lastFpsCalcMs = 0;

// ── Function Declarations ─────────────────────────────────────────────────────
bool initCameraHardware();
void scanAvailableNetworks();
void connectToWiFi();
void maintainWiFiConnection();
void onWiFiConnected();
void maintainWebSocketConnection();
void handleWebSocketMessage(WebsocketsMessage message);
void handleWebSocketEvents(WebsocketsEvent event, String data);
void sendHeartbeatPacket();
void streamPreviewFrame();
void executeCaptureCommand(const String& captureId);
void setupLocalHttpServer();

// ══════════════════════════════════════════════════════════════════════════════
//   SETUP: Hardware & Subsystem Initialization
// ══════════════════════════════════════════════════════════════════════════════
void setup() {
  // 0. Disable brownout detector immediately to prevent reset loops from RF / camera current spikes
  WRITE_PERI_REG(RTC_CNTL_BROWN_OUT_REG, 0);

  Serial.begin(115200);
  delay(500);

  Serial.println(F("\n========================================================"));
  Serial.println(F("    SmartCropVision ESP32-CAM Production Firmware      "));
  Serial.printf("    Firmware Version : %s\n", FIRMWARE_VERSION);
  Serial.printf("    Device ID        : %s\n", DEVICE_ID);
  Serial.println(F("========================================================"));

  // Configure onboard indicator LEDs if defined
  #if defined(STATUS_LED_PIN) && STATUS_LED_PIN >= 0
    pinMode(STATUS_LED_PIN, OUTPUT);
    digitalWrite(STATUS_LED_PIN, HIGH); // Off for active-low LED
  #endif

  #if defined(FLASH_LED_PIN) && FLASH_LED_PIN >= 0
    pinMode(FLASH_LED_PIN, OUTPUT);
    digitalWrite(FLASH_LED_PIN, LOW);  // Flash off initially
  #endif

  // 1. Initialize OV2640 camera sensor
  g_cameraInitialized = initCameraHardware();
  if (!g_cameraInitialized) {
    Serial.println(F("[ERROR] Camera initialization failed! Check ribbon cable seating."));
  }

  // 2. Configure WebSocket callbacks
  wsClient.onMessage(handleWebSocketMessage);
  wsClient.onEvent(handleWebSocketEvents);

  // 3. Scan 2.4 GHz RF environment to assist user network diagnostics
  scanAvailableNetworks();

  // 4. Initiate Wi-Fi connection
  connectToWiFi();

  // 5. Setup local HTTP fallback server on port 81
  setupLocalHttpServer();

  Serial.println(F("[SETUP] Initialization completed. Running cooperative main loop."));
}

// ══════════════════════════════════════════════════════════════════════════════
//   LOOP: Cooperative State Machine
// ══════════════════════════════════════════════════════════════════════════════
void loop() {
  unsigned long now = millis();

  // 1. Maintain Wi-Fi connectivity without blocking
  maintainWiFiConnection();

  // 2. Maintain outbound WebSocket link to backend gateway
  if (WiFi.status() == WL_CONNECTED) {
    maintainWebSocketConnection();
    wsClient.poll();
  }

  // 3. Service local HTTP client requests
  localServer.handleClient();

  // 4. Transmit periodic heartbeat telemetry
  if (wsClient.available() && (now - g_lastHeartbeatMs >= HEARTBEAT_INTERVAL_MS)) {
    g_lastHeartbeatMs = now;
    sendHeartbeatPacket();
  }

  // 5. Stream preview frame if requested by dashboard and interval has elapsed
  if (g_cameraInitialized && wsClient.available() && g_streamActive) {
    unsigned long frameInterval = 1000 / STREAM_FPS_TARGET;
    if (now - g_lastFrameTimeMs >= frameInterval) {
      g_lastFrameTimeMs = now;
      streamPreviewFrame();

      // Track running FPS calculation
      g_framesInLastSecond++;
      if (now - g_lastFpsCalcMs >= 1000) {
        g_currentFps = g_framesInLastSecond * 1000.0f / (now - g_lastFpsCalcMs);
        g_framesInLastSecond = 0;
        g_lastFpsCalcMs = now;
      }
    }
  }

  // Yield to FreeRTOS watchdog & network stack
  yield();
}

// ══════════════════════════════════════════════════════════════════════════════
//   CAMERA HARDWARE INITIALIZATION
// ══════════════════════════════════════════════════════════════════════════════
bool initCameraHardware() {
  Serial.println(F("[CAMERA] Configuring OV2640 pinout and buffer parameters..."));

  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000; // 20 MHz clock for crisp, low-noise capture
  config.pixel_format = PIXFORMAT_JPEG;

  // PSRAM (Pseudo-SRAM) check: AI-Thinker modules have 4MB external PSRAM
  if (psramFound()) {
    Serial.printf("[CAMERA] PSRAM detected (%d KB free). Enabling double frame buffers.\n", ESP.getFreePsram() / 1024);
    config.frame_size = CAPTURE_FRAME_SIZE; // Sized for max resolution (VGA) so DMA descriptors never overflow
    config.jpeg_quality = CAPTURE_JPEG_QUALITY;
    config.fb_count = 2;
    #if defined(CAMERA_GRAB_LATEST)
      config.grab_mode = CAMERA_GRAB_LATEST; // Always acquire freshest frame
    #endif
  } else {
    Serial.println(F("[CAMERA] WARNING: No PSRAM detected. Limiting to single frame buffer."));
    config.frame_size = FRAMESIZE_QVGA;
    config.jpeg_quality = 14;
    config.fb_count = 1;
  }

  // Initialize camera driver
  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("[CAMERA] Initialization failed with error code: 0x%x\n", err);
    return false;
  }

  // Tune OV2640 sensor registers for botanical inspection
  sensor_t *s = esp_camera_sensor_get();
  if (s != NULL) {
    // Switch to preview resolution and flush startup buffers
    s->set_framesize(s, PREVIEW_FRAME_SIZE);
    s->set_quality(s, PREVIEW_JPEG_QUALITY);
    s->set_brightness(s, 0);     // 0 = neutral clean natural lighting
    s->set_contrast(s, 1);       // 1 = natural leaf contrast (avoids blowing out highlights to pure white)
    s->set_saturation(s, 2);     // 2 = maximum vibrant saturation for rich green chlorophyll tones
    s->set_whitebal(s, 1);       // 1 = Auto White Balance enabled
    s->set_awb_gain(s, 1);       // 1 = AWB Gain enabled
    s->set_wb_mode(s, 1);        // 1 = Sunny/Daylight mode (5500K - stops OV2640 AWB from shifting foliage to purple/magenta!)
    s->set_special_effect(s, 0); // 0 = Normal / No special effect
    s->set_raw_gma(s, 1);        // 1 = Gamma curve correction for rich color depth
    s->set_lenc(s, 1);           // 1 = Lens correction (removes corner vignetting)
    s->set_dcw(s, 1);            // 1 = Downsize color interpolation (preserves vibrant chroma)
    s->set_exposure_ctrl(s, 1);  // 1 = Hardware auto exposure enabled
    s->set_aec2(s, 0);           // 0 = Disable AEC2 (eliminates color distortion and screen glare)
    s->set_ae_level(s, 0);       // Auto exposure target
    s->set_gain_ctrl(s, 1);      // 1 = Hardware auto gain enabled
    s->set_gainceiling(s, (gainceiling_t)1); // Gain ceiling 1 (suppresses chromatic CMOS sensor noise)
    s->set_bpc(s, 1);            // 1 = Black pixel correction
    s->set_wpc(s, 1);            // 1 = White pixel correction

    for (int i = 0; i < 2; i++) {
      camera_fb_t *fb = esp_camera_fb_get();
      if (fb) esp_camera_fb_return(fb);
    }
  }

  Serial.println(F("[CAMERA] OV2640 hardware successfully initialized and calibrated."));
  return true;
}

// ══════════════════════════════════════════════════════════════════════════════
//   WIRELESS (WiFi) MANAGEMENT & 2.4 GHz DIAGNOSTICS
// ══════════════════════════════════════════════════════════════════════════════
void scanAvailableNetworks() {
  Serial.println(F("\n[WIFI] Scanning 2.4 GHz wireless networks..."));
  WiFi.mode(WIFI_STA);
  WiFi.disconnect();
  delay(100);

  int n = WiFi.scanNetworks();
  if (n == 0) {
    Serial.println(F("[WIFI] No networks found. Check antenna!"));
  } else {
    Serial.printf("[WIFI] Found %d network(s):\n", n);
    bool targetFound = false;
    for (int i = 0; i < n; ++i) {
      String ssid = WiFi.SSID(i);
      int32_t rssi = WiFi.RSSI(i);
      int channel = WiFi.channel(i);
      Serial.printf("  [%2d] %-26s | RSSI: %3d dBm | Ch: %2d\n", i + 1, ssid.c_str(), rssi, channel);
      if (ssid.equals(WIFI_SSID)) {
        targetFound = true;
      }
    }
    if (!targetFound) {
      Serial.println(F("────────────────────────────────────────────────────────────────"));
      Serial.printf("  ⚠️  WARNING: Target SSID '%s' NOT detected in 2.4 GHz scan!\n", WIFI_SSID);
      Serial.println(F("  ────────────────────────────────────────────────────────────────"));
      Serial.println(F("  1. ESP32 hardware radio ONLY supports 2.4 GHz Wi-Fi (NOT 5 GHz)."));
      Serial.println(F("  2. If using iPhone Personal Hotspot:"));
      Serial.println(F("     👉 Go to Settings -> Personal Hotspot -> Turn ON 'Maximize Compatibility'"));
      Serial.println(F("  3. If using Android Personal Hotspot:"));
      Serial.println(F("     👉 Go to Hotspot Settings -> Set AP Band to '2.4 GHz Band'."));
      Serial.println(F("  4. Check SSID spelling (SSIDs are case-sensitive)."));
      Serial.println(F("────────────────────────────────────────────────────────────────\n"));
    } else {
      Serial.printf("[WIFI] Target SSID '%s' verified in 2.4 GHz band.\n\n", WIFI_SSID);
    }
  }
}

void connectToWiFi() {
  Serial.printf("[WIFI] Connecting to '%s'...\n", WIFI_SSID);
  WiFi.mode(WIFI_STA);
  WiFi.setAutoReconnect(true);
  WiFi.persistent(false);
  WiFi.setSleep(false); // Disables modem sleep to prevent RF voltage sags and frame drops
  WiFi.setTxPower(WIFI_POWER_15dBm); // Throttles RF peak current to prevent USB power supply brownout

  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  g_lastWiFiAttemptMs = millis();
  g_wifiConnecting = true;

  // Non-destructive synchronous wait up to 10 seconds for clean initial boot connection
  unsigned long start = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - start < 10000) {
    delay(500);
    Serial.print(F("."));
  }
  Serial.println();

  if (WiFi.status() == WL_CONNECTED) {
    onWiFiConnected();
  } else {
    Serial.println(F("[WIFI] Still associating in background. Main loop will maintain link."));
    g_lastWiFiAttemptMs = millis();
  }
}

String g_effectiveBackendHost = BACKEND_HOST;
uint8_t g_wsFailCount = 0;
uint8_t g_reconnectAttempts = 0;

void onWiFiConnected() {
  g_wifiConnecting = false;
  g_wifiWasConnected = true;
  g_wifiRetryIntervalMs = 10000;
  g_reconnectAttempts = 0;
  g_wsFailCount = 0;
  g_effectiveBackendHost = BACKEND_HOST;

  Serial.println(F("\n════════════════════════════════════════════════════════"));
  Serial.println(F("  [WIFI] Connected Successfully!"));
  Serial.printf("  IP Address   : %s\n", WiFi.localIP().toString().c_str());
  Serial.printf("  Subnet Mask  : %s\n", WiFi.subnetMask().toString().c_str());
  Serial.printf("  Gateway IP   : %s\n", WiFi.gatewayIP().toString().c_str());
  Serial.printf("  RSSI         : %d dBm\n", WiFi.RSSI());
  Serial.printf("  Local Stream : http://%s:%d/stream\n", WiFi.localIP().toString().c_str(), LOCAL_STREAM_PORT);
  Serial.printf("  Local Status : http://%s:%d/status\n", WiFi.localIP().toString().c_str(), LOCAL_STREAM_PORT);
  Serial.println(F("════════════════════════════════════════════════════════\n"));

  #if defined(STATUS_LED_PIN) && STATUS_LED_PIN >= 0
    digitalWrite(STATUS_LED_PIN, LOW); // Active-low LED ON indicates connected
  #endif
}

void maintainWiFiConnection() {
  wl_status_t status = WiFi.status();

  if (status == WL_CONNECTED) {
    if (g_wifiConnecting || !g_wifiWasConnected) {
      onWiFiConnected();
    }
    return;
  }

  unsigned long now = millis();

  // Handle sudden disconnection
  if (g_wifiWasConnected) {
    Serial.println(F("[WIFI] Connection lost! Waiting for hotspot / Wi-Fi beacon..."));
    g_wifiWasConnected = false;
    g_wifiConnecting = true;
    g_lastWiFiAttemptMs = now;
    g_wifiRetryIntervalMs = 3500; // Fast 3.5s interval for immediate hotspot detection

    #if defined(STATUS_LED_PIN) && STATUS_LED_PIN >= 0
      digitalWrite(STATUS_LED_PIN, HIGH); // LED off indicates disconnected
    #endif
  }

  // Fast, seamless hotspot auto-reconnection
  // Mobile hotspots (iPhone / Android) often sleep beacon when idle and resume when toggled.
  // WiFi.reconnect() re-associates instantly without clearing credentials.
  // Every 4th retry, refresh WiFi.begin() to force a complete 2.4GHz channel scan (Ch 1-13)
  // in case the mobile phone changed hotspot radio channel upon toggling ON.
  if (now - g_lastWiFiAttemptMs >= g_wifiRetryIntervalMs) {
    g_lastWiFiAttemptMs = now;
    g_reconnectAttempts++;

    if (g_reconnectAttempts % 4 == 0 || status == WL_NO_SSID_AVAIL || status == WL_CONNECT_FAILED) {
      Serial.printf("[WIFI] Refreshing 2.4 GHz channel scan for '%s' (attempt %d)...\n", WIFI_SSID, g_reconnectAttempts);
      WiFi.disconnect(false);
      WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
    } else {
      Serial.printf("[WIFI] Auto-reconnecting to '%s' (attempt %d)...\n", WIFI_SSID, g_reconnectAttempts);
      WiFi.reconnect();
    }
  }
}

// ══════════════════════════════════════════════════════════════════════════════
//   WEBSOCKET CLIENT MANAGEMENT (Outbound to Backend Gateway)
// ══════════════════════════════════════════════════════════════════════════════
void discoverLocalBackend() {
  #if USE_SECURE_WSS
    return; // Cloud domain uses DNS resolution, skip subnet probe
  #endif

  IPAddress localIp = WiFi.localIP();
  // If we are on an iPhone hotspot (172.20.10.x / 28) or typical LAN
  if (localIp[0] == 172 && localIp[1] == 20 && localIp[2] == 10) {
    Serial.println(F("[DISCOVERY] Probing mobile hotspot subnet for SmartCropVision backend on port 8000..."));
    WiFiClient probe;
    probe.setTimeout(120);

    IPAddress candidate = localIp;
    for (int host = 2; host <= 14; host++) {
      candidate[3] = host;
      if (candidate == localIp) continue; // Skip self

      if (probe.connect(candidate, BACKEND_PORT)) {
        probe.stop();
        g_effectiveBackendHost = candidate.toString();
        Serial.printf("[DISCOVERY] Found active SmartCropVision server at %s:%d!\n", 
                      g_effectiveBackendHost.c_str(), BACKEND_PORT);
        g_wsFailCount = 0;
        return;
      }
    }
  }
}

#if USE_SECURE_WSS
// DigiCert Global Root G2 Certificate (Authoritative root for Azure Cloud / dpdns.org)
const char DIGICERT_ROOT_CA[] PROGMEM = 
"-----BEGIN CERTIFICATE-----\n"
"MIIDjjCCAnagAwIBAgIQAzrx5qcRqaC7KGSxHQn65TANBgkqhkiG9w0BAQsFADBh\n"
"MQswCQYDVQQGEwJVUzEVMBMGA1UEChMMRGlnaUNlcnQgSW5jMRkwFwYDVQQLExB3\n"
"d3cuZGlnaWNlcnQuY29tMSAwHgYDVQQDExdEaWdpQ2VydCBHbG9iYWwgUm9vdCBH\n"
"MjAeFw0xMzA4MDExMjAwMDBaFw0zODAxMTUxMjAwMDBaMGExCzAJBgNVBAYTAlVT\n"
"MRUwEwYDVQQKEwxEaWdpQ2VydCBJbmMxGTAXBgNVBAsTEHd3dy5kaWdpY2VydC5j\n"
"b20xIDAeBgNVBAMTF0RpZ2lDZXJ0IEdsb2JhbCBSb290IEcyMIIBIjANBgkqhkiG\n"
"9w0BAQEFAAOCAQ8AMIIBCgKCAQEAuzfNNNx7a8myaJCtSnX/RrohCgiN9RlUyfuI\n"
"2/Ou8jqJkTx65qsGGmvPrC3oXgkkRLpimn7Wo6h+4FR1IAWsULecYxpsMNzaHxmx\n"
"1x7e/dfgy5SDN67sH0NO3Xss0r0upS/kqbitOtSZpLYl6ZtrAGCSYP9PIUkY92eQ\n"
"q2EGnI/yuum06ZIya7XzV+hdG82MHauVBJVJ8zUtluNJbd134/tJS7SsVQepj5Wz\n"
"tCO7TG1F8PapspUwtP1MVYwnSlcUfIKdzXOS0xZKBgyMUNGPHgm+F6HmIcr9g+UQ\n"
"vIOlCsRnKPZzFBQ9RnbDhxSJITRNrw9FDKZJobq7nMWxM4MphQIDAQABo0IwQDAP\n"
"BgNVHRMBAf8EBTADAQH/MA4GA1UdDwEB/wQEAwIBhjAdBgNVHQ4EFgQUTiJUIBiV\n"
"5uNu5g/6+rkS7QYXjzkwDQYJKoZIhvcNAQELBQADggEBAGBnKJRvDkhj6zHd6mcY\n"
"1Yl9PMWLSn/pvtsrF9+wX3N3KjITOYFnQoQj8kVnNeyIv/iPsGEMNKSuIEyExtv4\n"
"NeF22d+mQrvHRAiGfzZ0JFrabA0UWTW98kndth/Jsw1HKj2ZL7tcu7XUIOGZX1NG\n"
"Fdtom/DzMNU+MeKNhJ7jitralj41E6Vf8PlwUHBHQRFXGU7Aj64GxJUTFy8bJZ91\n"
"8rGOmaFvE7FBcf6IKshPECBV1/MUReXgRPTqh5Uykw7+U0b6LJ3/iyK5S9kJRaTe\n"
"pLiaWN0bfVKfjllDiIGknibVb63dDcY3fe0Dkhvld1927jyNxF1WW6LZZm6zNTfl\n"
"MrY=\n"
"-----END CERTIFICATE-----\n";
#endif

void maintainWebSocketConnection() {
  if (wsClient.available()) {
    return;
  }

  static unsigned long lastWsAttemptMs = 0;
  unsigned long now = millis();
  if (now - lastWsAttemptMs < 2500) {
    return; // Fast 2.5s reconnect throttle for seamless dashboard link
  }
  lastWsAttemptMs = now;

  // If initial configured BACKEND_HOST fails 4 times, probe the subnet for auto-discovery
  if (g_wsFailCount >= 4) {
    discoverLocalBackend();
  }

  #if USE_SECURE_WSS
    wsClient.setCACert(DIGICERT_ROOT_CA);
    wsClient.setInsecure(); // Dual verification: sets root CA and allows TLS handshake
  #endif

  // For standard HTTPS (443) or HTTP (80), omit explicit port to ensure clean RFC Host header
  String portPart = "";
  if (USE_SECURE_WSS && BACKEND_PORT != 443) {
    portPart = ":" + String(BACKEND_PORT);
  } else if (!USE_SECURE_WSS && BACKEND_PORT != 80) {
    portPart = ":" + String(BACKEND_PORT);
  }

  String wsUrl = String(USE_SECURE_WSS ? "wss://" : "ws://") +
                 g_effectiveBackendHost + portPart +
                 BACKEND_WS_PATH + "?device_id=" + DEVICE_ID +
                 "&token=" + DEVICE_TOKEN;

  Serial.printf("[WS] Connecting outbound to gateway: %s%s%s\n", g_effectiveBackendHost.c_str(), portPart.c_str(), BACKEND_WS_PATH);
  bool connected = wsClient.connect(wsUrl);
  if (connected) {
    Serial.println(F("[WS] Connected to SmartCropVision gateway. Status: ONLINE"));
    g_streamActive = true;
    g_wsFailCount = 0;
    sendHeartbeatPacket();
  } else {
    g_wsFailCount++;
    Serial.printf("[WS] Gateway connection to %s%s failed (fails=%d). Will retry.\n", 
                  g_effectiveBackendHost.c_str(), portPart.c_str(), g_wsFailCount);
  }
}

void handleWebSocketEvents(WebsocketsEvent event, String data) {
  if (event == WebsocketsEvent::ConnectionOpened) {
    Serial.println(F("[WS] Event: Channel Opened. Status: ONLINE"));
    g_streamActive = true;
  } else if (event == WebsocketsEvent::ConnectionClosed) {
    Serial.println(F("[WS] Event: Channel Closed. Status: OFFLINE"));
    g_streamActive = false;
  } else if (event == WebsocketsEvent::GotPing) {
    wsClient.pong();
  }
}

void handleWebSocketMessage(WebsocketsMessage message) {
  if (!message.isText()) return;

  String payload = message.data();
  StaticJsonDocument<256> doc;
  DeserializationError err = deserializeJson(doc, payload);
  if (err) {
    Serial.printf("[WS] JSON deserialization failed: %s\n", err.c_str());
    return;
  }

  const char* type = doc["type"] | "unknown";

  // 1. Capture command
  if (strcmp(type, "capture") == 0) {
    const char* cid = doc["capture_id"] | "unknown_capture";
    Serial.printf("[COMMAND] Hardware Capture requested: %s\n", cid);
    executeCaptureCommand(String(cid));
  }
  // 2. Stream control command
  else if (strcmp(type, "control_stream") == 0) {
    g_streamActive = doc["active"] | false;
    Serial.printf("[COMMAND] Preview streaming state updated: %s\n", g_streamActive ? "ACTIVE" : "PAUSED");
  }
  // 3. Dynamic camera configuration command
  else if (strcmp(type, "camera_config") == 0) {
    sensor_t *s = esp_camera_sensor_get();
    if (s != NULL) {
      if (doc.containsKey("wb_mode")) {
        int wb = doc["wb_mode"];
        s->set_wb_mode(s, wb);
        Serial.printf("[CONFIG] OV2640 wb_mode set to: %d\n", wb);
      }
      if (doc.containsKey("saturation")) {
        int sat = doc["saturation"];
        s->set_saturation(s, sat);
        Serial.printf("[CONFIG] OV2640 saturation set to: %d\n", sat);
      }
      if (doc.containsKey("contrast")) {
        int con = doc["contrast"];
        s->set_contrast(s, con);
        Serial.printf("[CONFIG] OV2640 contrast set to: %d\n", con);
      }
      if (doc.containsKey("brightness")) {
        int bri = doc["brightness"];
        s->set_brightness(s, bri);
        Serial.printf("[CONFIG] OV2640 brightness set to: %d\n", bri);
      }
    }
  }
  // 4. Ping command
  else if (strcmp(type, "ping") == 0) {
    sendHeartbeatPacket();
  }
}

// ══════════════════════════════════════════════════════════════════════════════
//   TELEMETRY & HEARTBEAT
// ══════════════════════════════════════════════════════════════════════════════
void sendHeartbeatPacket() {
  StaticJsonDocument<384> doc;
  doc["type"] = "heartbeat";
  doc["device_id"] = DEVICE_ID;
  doc["firmware_version"] = FIRMWARE_VERSION;
  doc["camera_type"] = "OV2640";
  doc["rssi"] = WiFi.RSSI();
  doc["free_heap"] = ESP.getFreeHeap();
  doc["uptime_ms"] = millis();
  doc["frame_size"] = (PREVIEW_FRAME_SIZE == FRAMESIZE_QVGA) ? "QVGA" : "CIF";
  doc["fps"] = round(g_currentFps * 10.0f) / 10.0f;
  doc["stream_active"] = g_streamActive;

  String output;
  serializeJson(doc, output);
  wsClient.send(output);
}

// ══════════════════════════════════════════════════════════════════════════════
//   IMAGE ACQUISITION: STREAMING & HIGH-RES CAPTURE
// ══════════════════════════════════════════════════════════════════════════════
void streamPreviewFrame() {
  camera_fb_t *fb = esp_camera_fb_get();
  if (!fb) {
    return;
  }

  // Prepend 4-byte header 'SPRV' (Smart Preview) to frame buffer
  size_t totalLen = 4 + fb->len;
  uint8_t *packet = psramFound() ? (uint8_t *)ps_malloc(totalLen) : (uint8_t *)malloc(totalLen);
  if (packet) {
    packet[0] = 'S';
    packet[1] = 'P';
    packet[2] = 'R';
    packet[3] = 'V';
    memcpy(packet + 4, fb->buf, fb->len);

    wsClient.sendBinary((const char *)packet, totalLen);
    free(packet);
    g_totalFramesSent++;
  }

  esp_camera_fb_return(fb);
}

void executeCaptureCommand(const String& captureId) {
  // Discard older queued buffer frame to get immediate real-time calibrated capture
  for (int i = 0; i < 2; i++) {
    camera_fb_t *staleFb = esp_camera_fb_get();
    if (staleFb) esp_camera_fb_return(staleFb);
  }

  sensor_t *s = esp_camera_sensor_get();
  // Only change frame size if capture resolution differs from preview
  if (s != NULL && psramFound() && (CAPTURE_FRAME_SIZE != PREVIEW_FRAME_SIZE)) {
    s->set_framesize(s, CAPTURE_FRAME_SIZE);
    s->set_quality(s, CAPTURE_JPEG_QUALITY);
    // Allow sensor AEC/AGC loop to converge to scene exposure
    for (int i = 0; i < 6; i++) {
      camera_fb_t *staleFb = esp_camera_fb_get();
      if (staleFb) esp_camera_fb_return(staleFb);
      delay(35);
    }
  }

  // Flash illumination assist
  #if defined(FLASH_LED_PIN) && FLASH_LED_PIN >= 0
    digitalWrite(FLASH_LED_PIN, HIGH);
    delay(50);
  #endif

  camera_fb_t *fb = esp_camera_fb_get();

  #if defined(FLASH_LED_PIN) && FLASH_LED_PIN >= 0
    digitalWrite(FLASH_LED_PIN, LOW);
  #endif

  if (!fb) {
    Serial.println(F("[CAMERA] Capture acquisition failed!"));
    if (s != NULL && psramFound() && (CAPTURE_FRAME_SIZE != PREVIEW_FRAME_SIZE)) {
      s->set_framesize(s, PREVIEW_FRAME_SIZE);
      s->set_quality(s, PREVIEW_JPEG_QUALITY);
    }
    return;
  }

  Serial.printf("[CAMERA] Frame captured! Size: %d bytes. Transmitting...\n", fb->len);

  // Format binary packet:
  // [4 bytes: 'SCAP'] + [16 bytes: capture_id padded with spaces] + [JPEG bytes]
  size_t headerLen = 20;
  size_t totalLen = headerLen + fb->len;
  uint8_t *packet = psramFound() ? (uint8_t *)ps_malloc(totalLen) : (uint8_t *)malloc(totalLen);

  if (packet) {
    packet[0] = 'S';
    packet[1] = 'C';
    packet[2] = 'A';
    packet[3] = 'P';

    // Copy 16-byte fixed capture ID
    memset(packet + 4, ' ', 16);
    size_t copyIdLen = min((size_t)16, (size_t)captureId.length());
    memcpy(packet + 4, captureId.c_str(), copyIdLen);

    // Copy JPEG data
    memcpy(packet + headerLen, fb->buf, fb->len);

    wsClient.sendBinary((const char *)packet, totalLen);
    free(packet);
    g_totalCapturesTaken++;
    Serial.printf("[CAPTURE] Capture [%s] successfully delivered to gateway.\n", captureId.c_str());
  } else {
    Serial.println(F("[CAMERA] Memory allocation failed for capture packet."));
  }

  esp_camera_fb_return(fb);

  // Restore preview resolution if changed
  if (s != NULL && psramFound() && (CAPTURE_FRAME_SIZE != PREVIEW_FRAME_SIZE)) {
    s->set_framesize(s, PREVIEW_FRAME_SIZE);
    s->set_quality(s, PREVIEW_JPEG_QUALITY);
    for (int i = 0; i < 2; i++) {
      camera_fb_t *staleFb = esp_camera_fb_get();
      if (staleFb) esp_camera_fb_return(staleFb);
    }
  }
}

// ══════════════════════════════════════════════════════════════════════════════
//   LOCAL HTTP FALLBACK SERVER (Port 81)
// ══════════════════════════════════════════════════════════════════════════════
void handleLocalStream() {
  WiFiClient client = localServer.client();
  String response = "HTTP/1.1 200 OK\r\n"
                    "Content-Type: multipart/x-mixed-replace; boundary=frame\r\n"
                    "Access-Control-Allow-Origin: *\r\n\r\n";
  client.print(response);

  while (client.connected()) {
    // Keep WebSocket connection and RTOS tasks serviced during local streaming
    wsClient.poll();
    yield();

    camera_fb_t *fb = esp_camera_fb_get();
    if (!fb) break;

    client.print("--frame\r\n");
    client.print("Content-Type: image/jpeg\r\n");
    client.printf("Content-Length: %d\r\n\r\n", fb->len);
    client.write(fb->buf, fb->len);
    client.print("\r\n");

    esp_camera_fb_return(fb);
    delay(80); // ~12 FPS
  }
}

void handleLocalCapture() {
  camera_fb_t *fb = esp_camera_fb_get();
  if (!fb) {
    localServer.send(500, "text/plain", "Camera acquisition failed");
    return;
  }
  localServer.sendHeader("Access-Control-Allow-Origin", "*");
  localServer.sendHeader("Cache-Control", "no-cache");
  localServer.setContentLength(fb->len);
  localServer.send(200, "image/jpeg", "");
  WiFiClient client = localServer.client();
  client.write(fb->buf, fb->len);
  esp_camera_fb_return(fb);
}

void handleLocalStatus() {
  StaticJsonDocument<256> doc;
  doc["device_id"] = DEVICE_ID;
  doc["firmware"] = FIRMWARE_VERSION;
  doc["camera"] = "OV2640";
  doc["rssi"] = WiFi.RSSI();
  doc["free_heap"] = ESP.getFreeHeap();
  doc["uptime_s"] = millis() / 1000;
  doc["fps"] = g_currentFps;

  String res;
  serializeJson(doc, res);
  localServer.sendHeader("Access-Control-Allow-Origin", "*");
  localServer.send(200, "application/json", res);
}

void setupLocalHttpServer() {
  localServer.on("/stream", HTTP_GET, handleLocalStream);
  localServer.on("/capture", HTTP_GET, handleLocalCapture);
  localServer.on("/status", HTTP_GET, handleLocalStatus);
  localServer.begin();
  Serial.printf("[HTTP] Local fallback server running on port %d (/stream, /capture, /status)\n", LOCAL_STREAM_PORT);
}
