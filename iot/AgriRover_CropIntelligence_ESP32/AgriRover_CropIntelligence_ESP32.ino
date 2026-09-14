/*
 * ═══════════════════════════════════════════════════════════════════════
 *   Smart Plant Intelligence System — ESP32 Dev Module Firmware v3.0
 *   16x2 I2C Character LCD Edition (HW-61 / PCF8574 Backpack)
 *   Target Board: ESP32 Dev Module (ESP32-WROOM-32 / 30-Pin NodeMCU ESP32)
 * ═══════════════════════════════════════════════════════════════════════
 *
 *   HARDWARE MODULES & PERIPHERALS:
 *     - ESP32 Dev Module (30-pin ESP32-WROOM-32)
 *     - 16x2 Character LCD with HW-61 I2C adapter (PCF8574 chip, 0x27 or 0x3F)
 *     - DHT11 Temperature + Humidity Sensor (3-pin or 4-pin module)
 *     - Capacitive Soil Moisture Sensor v1.2 (AO analog output)
 *     - Rain Drop Detection Sensor YL-83 (DO digital output + AO analog)
 *     - Water Level / Field Drainage Depth Sensor (AO analog output)
 *     - GPS NEO-6M Satellite Navigation Receiver (UART Serial)
 *
 * ─────────────────────────────────────────────────────────────────────
 *   ESP32 DEV MODULE WIRING DIAGRAM (30-PIN PINOUT)
 * ─────────────────────────────────────────────────────────────────────
 *
 *   ESP32 Dev Module         16x2 LCD (HW-61 I2C Backpack)
 *   ────────────────         ────────────────────────────
 *   GPIO 22 (SCL) ────────── SCL
 *   GPIO 21 (SDA) ────────── SDA
 *   VIN (5V Rail) ────────── VCC   ← MUST use 5V (VIN), not 3.3V, for clear LCD
 * contrast! GND           ────────── GND
 *
 *   ESP32 Dev Module         DHT11 Sensor (3-pin: GND | DATA | VCC)
 *   ────────────────         ─────────────────────────────────────
 *   GPIO 4        ────────── DATA
 *   3.3V          ────────── VCC
 *   GND           ────────── GND
 *
 *   ESP32 Dev Module         Capacitive Soil Moisture Sensor v1.2
 *   ────────────────         ─────────────────────────────────────
 *   GPIO 34 (ADC1_CH6) ───── AOUT  (Input-only ADC pin, 100% WiFi immune)
 *   3.3V          ────────── VCC
 *   GND           ────────── GND
 *
 *   ESP32 Dev Module         Rain Drop Sensor YL-83
 *   ────────────────         ───────────────────────
 *   GPIO 39 (VN)  ────────── DO    (Digital wet/dry flag: LOW = Rain, HIGH =
 * Dry) GPIO 36 (VP)  ────────── AO    (Precipitation intensity analog,
 * optional) 3.3V          ────────── VCC GND           ────────── GND
 *
 *   ESP32 Dev Module         Water Level / Drainage Depth Sensor
 *   ────────────────         ───────────────────────────────────
 *   GPIO 35 (ADC1_CH7) ───── SIG   (Field drainage depth / retention, WiFi
 * immune) 3.3V          ────────── VCC GND           ────────── GND
 *
 *   ESP32 Dev Module         GPS NEO-6M Receiver
 *   ────────────────         ────────────────────
 *   GPIO 16 (RX2) ────────── TXD   (GPS output transmits to ESP32 RX2)
 *   GPIO 17 (TX2) ────────── RXD   (ESP32 TX2 to GPS RX, optional)
 *   VIN (5V) or 3.3V ─────── VCC   (Accepts 3.6V - 5V)
 *   GND           ────────── GND
 *
 * ─────────────────────────────────────────────────────────────────────
 *   REQUIRED LIBRARIES — install via Arduino IDE Library Manager
 * ─────────────────────────────────────────────────────────────────────
 *   1. WiFi               — Built-in with ESP32 board package by Espressif
 *   2. HTTPClient         — Built-in with ESP32 board package
 *   3. ArduinoJson        — v6.x or v7.x by Benoit Blanchon
 *   4. DHT sensor library — by Adafruit
 *   5. LiquidCrystal I2C  — by Frank de Brabander or Marco Schwartz
 *   6. TinyGPSPlus        — by Mikal Hart
 *
 * ═══════════════════════════════════════════════════════════════════════
 */

// ── Library Includes ───────────────────────────────────────────────────
#include <ArduinoJson.h>
#include <DHT.h>
#include <HTTPClient.h>
#include <LiquidCrystal_I2C.h>
#include <TinyGPSPlus.h>
#include <WiFi.h>
#include <WiFiClient.h>
#include <WiFiClientSecure.h>
#include <Wire.h>

// ══════════════════════════════════════════════════════════════════════
//   ⚙️  WIFI & DEPLOYMENT CONFIGURATION
// ══════════════════════════════════════════════════════════════════════

const char *WIFI_SSID = "Fayas";     // WiFi hotspot name
const char *WIFI_PASS = "777888666"; // WiFi password

// ── Target Mode Selection ─────────────────────────────────────────────
//   0 = PUBLIC PRODUCTION CLOUD (https://aipoweredcropintelligencerover.dpdns.org)
//   1 = LOCAL MAC DEV           (http://172.20.10.4:8000)
//   2 = SMART DUAL-MODE         (Attempts Local Mac first; auto-failovers to Public Cloud if offline)
#define TARGET_MODE 2 // 2 = Seamless Auto-Failover (Recommended)

String g_localDevUrl = "http://172.20.10.4:8000/predict/compact";
const char *PUBLIC_CLOUD_URL = "https://aipoweredcropintelligencerover.dpdns.org/predict/compact";

#if TARGET_MODE == 0
const char *TARGET_LABEL = "dpdns.org PROD";
#elif TARGET_MODE == 1
const char *TARGET_LABEL = "172.20.10.4:8000";
#else
const char *TARGET_LABEL = "Auto Local+Cloud";
#endif

// DigiCert Global Root G2 Certificate (Authoritative root for Azure Cloud / dpdns.org, matching ESP32-CAM)
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

// ── Pin Definitions for ESP32 Dev Module ──────────────────────────────
#define DHT_PIN 4 // GPIO 4  — DHT11 DATA
#define DHT_TYPE DHT11
#define SOIL_PIN 34    // GPIO 34 (ADC1_CH6) — Capacitive Soil Moisture AO
#define WATER_PIN 35   // GPIO 35 (ADC1_CH7) — Water Level Sensor SIG
#define RAIN_DO_PIN 39 // GPIO 39 (VN)       — Rain Sensor DO
#define RAIN_AO_PIN 36 // GPIO 36 (VP)       — Rain Sensor AO
#define GPS_RX_PIN 16  // GPIO 16 (RX2)      — GPS NEO-6M TX
#define GPS_TX_PIN 17  // GPIO 17 (TX2)      — GPS NEO-6M RX
#define I2C_SDA 21     // GPIO 21            — LCD SDA
#define I2C_SCL 22     // GPIO 22            — LCD SCL

// ── LCD I2C Configuration ─────────────────────────────────────────────
#define LCD_I2C_ADDR 0x27 // Most HW-61 modules use 0x27. If blank, try 0x3F.
#define LCD_COLS 16
#define LCD_ROWS 2

// ── Timing ────────────────────────────────────────────────────────────
#define SEND_INTERVAL 12000  // POST to backend every 12 seconds
#define SCREEN_INTERVAL 4000 // Rotate LCD screen carousel every 4 seconds
#define SAMPLE_INTERVAL 2000 // Sample physical sensors every 2 seconds

// ── Sensor Calibration Constants (ESP32 12-Bit ADC) ───────────────────
#define SOIL_DRY_RAW 3200  // Bone dry air reading (0% moisture)
#define SOIL_WET_RAW 1350  // Saturated water reading (100% moisture)
#define WATER_BASE_RAW 400 // Dry / zero-level water baseline
#define WATER_SAT_RAW 2800 // Full immersion depth (~45mm standing water)

// ─────────────────────────────────────────────────────────────────────

// ── Object Instances ───────────────────────────────────────────────────
DHT dht(DHT_PIN, DHT_TYPE);
LiquidCrystal_I2C lcd(LCD_I2C_ADDR, LCD_COLS, LCD_ROWS);
TinyGPSPlus gps;
HardwareSerial gpsSerial(2);

WiFiClient plainClient;
WiFiClientSecure secureClient;

// ── Global Sensor State ────────────────────────────────────────────────
int g_temp = 25;
int g_hum = 60;
int g_soil = 50;
bool g_rainOn = false;
float g_waterMm = 0.0f;
int g_waterPct = 0;
double g_lat = 0.0;
double g_lon = 0.0;
float g_alt = 0.0f;
int g_sats = 0;
bool g_gpsValid = false;

// ── Global Prediction State ────────────────────────────────────────────
String g_crop = "";
float g_conf = 0.0;
String g_crop2 = "";
float g_conf2 = 0.0;
String g_crop3 = "";
float g_conf3 = 0.0;
int g_alerts = 0;
String g_alertName[4];
String g_alertSev[4];
int g_alertCount = 0;
bool g_serverOK = false;

// ── Display & Carousel State ───────────────────────────────────────────
int g_screen = 0;
int g_alertPage = 0;
unsigned long g_lastSend = 0;
unsigned long g_lastScreen = 0;
unsigned long g_lastSample = 0;

// ── Custom LCD 5x8 Characters (Pixel-Perfect Glyphs) ───────────────────
byte degreeChar[8] = {0b00110, 0b01001, 0b01001, 0b00110,
                      0b00000, 0b00000, 0b00000, 0b00000};
byte tickChar[8] = {0b00000, 0b00001, 0b00011, 0b10110,
                    0b11100, 0b01000, 0b00000, 0b00000};
byte alertChar[8] = {0b00100, 0b01110, 0b01110, 0b01110,
                     0b11111, 0b00000, 0b00100, 0b00000};
byte dropChar[8] = {0b00100, 0b00100, 0b01110, 0b01110,
                    0b11111, 0b11111, 0b01110, 0b00000};

// Character indices
#define CHAR_DEG 0
#define CHAR_TICK 1
#define CHAR_ALERT 2
#define CHAR_DROP 3

// Forward declarations
void initI2cLcd();
void showBoot();
void connectWiFi();
void readSensors();
void processGps();
void sendToServer();
void parseResponse(String &body);
void advanceScreen();
void drawScreen();
void screenSensors();
void screenCrop();
void screenAlerts();
void screenWater();
void screenGps();
void lcdStatus(String line0, String line1);

// ══════════════════════════════════════════════════════════════════════
//   SETUP
// ══════════════════════════════════════════════════════════════════════
void setup() {
  Serial.begin(115200);
  delay(150);
  Serial.println(F("\n\n=== Smart Plant Intelligence System — ESP32 v3.0 ==="));

  // 1. Hardware Pin & ADC Setup
  pinMode(RAIN_DO_PIN, INPUT);
  analogReadResolution(12);       // ESP32 12-bit ADC (0 - 4095)
  analogSetAttenuation(ADC_11db); // Full-scale 0 - 3.3V range

  // 2. DHT11 Initialization
  dht.begin();

  // 3. GPS NEO-6M HardwareSerial2 (RX2 = GPIO 16, TX2 = GPIO 17)
  gpsSerial.begin(9600, SERIAL_8N1, GPS_RX_PIN, GPS_TX_PIN);

  // 4. I2C Bus & LCD Initialization
  initI2cLcd();

  // 5. Initial Boot Screen Animation
  showBoot();

  // 6. Connect WiFi
  connectWiFi();

  // 7. Initial Sensor Acquisition & Immediate First Predict Sync
  readSensors();
  if (WiFi.status() == WL_CONNECTED) {
    sendToServer();
  }

  // Draw initial sensor screen
  drawScreen();
}

// ══════════════════════════════════════════════════════════════════════
//   MAIN NON-BLOCKING LOOP
// ══════════════════════════════════════════════════════════════════════
void loop() {
  unsigned long now = millis();

  // Continuous background GPS NMEA feed consumption
  processGps();

  // Re-connect WiFi if disconnected
  if (WiFi.status() != WL_CONNECTED) {
    g_serverOK = false;
    connectWiFi();
  }

  // Periodic sensor sampling
  if (now - g_lastSample >= SAMPLE_INTERVAL || g_lastSample == 0) {
    g_lastSample = now;
    readSensors();
  }

  // POST to server every SEND_INTERVAL
  if (now - g_lastSend >= SEND_INTERVAL || g_lastSend == 0) {
    g_lastSend = now;
    sendToServer();
  }

  // Rotate LCD screen carousel every SCREEN_INTERVAL
  if (now - g_lastScreen >= SCREEN_INTERVAL) {
    g_lastScreen = now;
    advanceScreen();
    drawScreen();
  }

  yield();
}

// ══════════════════════════════════════════════════════════════════════
//   I2C LCD INITIALIZATION
// ══════════════════════════════════════════════════════════════════════
void initI2cLcd() {
  Wire.begin(I2C_SDA, I2C_SCL);
  Wire.setClock(100000);

  lcd.init();
  lcd.backlight();

  // Register custom 5x8 glyphs
  lcd.createChar(CHAR_DEG, degreeChar);
  lcd.createChar(CHAR_TICK, tickChar);
  lcd.createChar(CHAR_ALERT, alertChar);
  lcd.createChar(CHAR_DROP, dropChar);

  Serial.printf("[LCD] I2C LCD online at address 0x%02X\n", LCD_I2C_ADDR);
}

// ══════════════════════════════════════════════════════════════════════
//   READ ALL PHYSICAL SENSORS (ESP32 CALIBRATED)
// ══════════════════════════════════════════════════════════════════════
void readSensors() {
  // ── DHT11 Temperature & Humidity ───────────────────────────────────
  float h = dht.readHumidity();
  float t = dht.readTemperature();

  if (!isnan(h) && !isnan(t) && h >= 10.0f && h <= 100.0f && t >= 0.0f &&
      t <= 60.0f) {
    g_temp = (int)round(t);
    g_hum = (int)round(h);
  } else {
    // Brief retry stabilization
    delay(200);
    h = dht.readHumidity();
    t = dht.readTemperature();
    if (!isnan(h) && !isnan(t)) {
      g_temp = (int)round(t);
      g_hum = (int)round(h);
    }
  }

  // ── Capacitive Soil Moisture (GPIO 34 ADC1) ─────────────────────────
  int rawSoil = analogRead(SOIL_PIN);
  int soilPct = map(rawSoil, SOIL_DRY_RAW, SOIL_WET_RAW, 0, 100);
  g_soil = constrain(soilPct, 0, 100);

  // ── Rain Drop Sensor (DO = GPIO 39, active LOW) ────────────────────
  g_rainOn = (digitalRead(RAIN_DO_PIN) == LOW);

  // ── Water Level Sensor (GPIO 35 ADC1) ──────────────────────────────
  int rawWater = analogRead(WATER_PIN);
  if (rawWater > WATER_BASE_RAW) {
    float depthMm = (float)(rawWater - WATER_BASE_RAW) /
                    (float)(WATER_SAT_RAW - WATER_BASE_RAW) * 45.0f;
    g_waterMm = constrain(depthMm, 0.0f, 55.0f);
    g_waterPct = constrain((int)((g_waterMm / 45.0f) * 100.0f), 0, 100);
  } else {
    g_waterMm = 0.0f;
    g_waterPct = 0;
  }

  Serial.printf(
      "[SENSORS] T:%dC H:%d%% Soil:%d%% Rain:%s Water:%.1fmm Sats:%d\n", g_temp,
      g_hum, g_soil, g_rainOn ? "YES" : "NO", g_waterMm, g_sats);
}

// ── Continuous GPS Parser ─────────────────────────────────────────────
void processGps() {
  while (gpsSerial.available() > 0) {
    gps.encode(gpsSerial.read());
  }

  if (gps.location.isValid() && gps.location.age() < 5000) {
    g_lat = gps.location.lat();
    g_lon = gps.location.lng();
    g_alt = gps.altitude.meters();
    g_sats = gps.satellites.value();
    g_gpsValid = true;
  } else {
    g_gpsValid = false;
    g_sats = gps.satellites.isValid() ? gps.satellites.value() : 0;
  }
}

// ══════════════════════════════════════════════════════════════════════
//   SEND TO SERVER + PARSE RESPONSE
// ══════════════════════════════════════════════════════════════════════
void sendToServer() {
  lcdStatus("Sending...", "");

  // Build compact JSON payload (compatible with ArduinoJson v6 and v7)
#if ARDUINOJSON_VERSION_MAJOR >= 7
  JsonDocument payload;
#else
  StaticJsonDocument<384> payload;
#endif
  payload["temperature"] = g_temp;
  payload["humidity"] = g_hum;
  payload["soil_moisture"] = g_soil;
  payload["rain"] = g_rainOn ? 1 : 0;
  payload["water_level"] = g_waterPct;
  payload["water_level_mm"] = (int)round(g_waterMm);
  payload["latitude"] = g_lat;
  payload["longitude"] = g_lon;
  payload["satellites"] = g_sats;

  String body;
  serializeJson(payload, body);

  HTTPClient http;
  int code = -1;
  bool success = false;

#if TARGET_MODE == 1 || TARGET_MODE == 2
  // Attempt local Mac server first
  Serial.println("[HTTP] Attempting LOCAL MAC → " + g_localDevUrl);
  http.begin(plainClient, g_localDevUrl);
  http.addHeader("Content-Type", "application/json");
  http.setTimeout(3000);
  code = http.POST(body);
  if (code == 200 || code == 201) {
    String resp = http.getString();
    Serial.println("[HTTP] Local Response: " + resp);
    parseResponse(resp);
    g_serverOK = true;
    success = true;
  } else {
    Serial.printf("[HTTP] Local unavailable (code %d)\n", code);
#if TARGET_MODE == 2
    discoverLocalBackend(); // Probe hotspot subnet in case Mac IP changed
#endif
  }
  http.end();
#endif

#if TARGET_MODE == 0 || TARGET_MODE == 2
  // Public Production Cloud (Azure / DPDNS) - Primary or Failover
  if (!success) {
#if TARGET_MODE == 2
    Serial.println("[HTTP] Failover: Routing telemetry to PUBLIC CLOUD (dpdns.org)...");
#else
    Serial.println("[HTTP] POST (PUBLIC CLOUD) → " + String(PUBLIC_CLOUD_URL));
#endif
    secureClient.setCACert(DIGICERT_ROOT_CA);
    http.begin(secureClient, PUBLIC_CLOUD_URL);
    http.addHeader("Content-Type", "application/json");
    http.setTimeout(8000);
    code = http.POST(body);
    if (code == 200 || code == 201) {
      String resp = http.getString();
      Serial.println("[HTTP] Cloud Response: " + resp);
      parseResponse(resp);
      g_serverOK = true;
      success = true;
    } else {
      // Fallback with setInsecure in case of clock or intermediate verification difference
      secureClient.setInsecure();
      http.begin(secureClient, PUBLIC_CLOUD_URL);
      http.addHeader("Content-Type", "application/json");
      http.setTimeout(8000);
      code = http.POST(body);
      if (code == 200 || code == 201) {
        String resp = http.getString();
        Serial.println("[HTTP] Cloud (Insecure fallback) Response: " + resp);
        parseResponse(resp);
        g_serverOK = true;
        success = true;
      }
    }
    http.end();
  }
#endif

  if (!success) {
    Serial.printf("[HTTP] All endpoints failed. Last code: %d\n", code);
    g_serverOK = false;
    lcdStatus("Server Error", "HTTP " + String(code));
    delay(1800);
  }
}

// ══════════════════════════════════════════════════════════════════════
//   LOCAL MAC SUBNET AUTO-DISCOVERY (For Mobile Hotspot 172.20.10.x)
// ══════════════════════════════════════════════════════════════════════
void discoverLocalBackend() {
  IPAddress localIp = WiFi.localIP();
  if (localIp[0] == 172 && localIp[1] == 20 && localIp[2] == 10) {
    Serial.println(F("[DISCOVERY] Probing hotspot subnet for Mac backend on port 8000..."));
    WiFiClient probe;
    probe.setTimeout(100);
    IPAddress candidate = localIp;
    for (int host = 2; host <= 14; host++) {
      candidate[3] = host;
      if (candidate == localIp) continue;
      if (probe.connect(candidate, 8000)) {
        probe.stop();
        g_localDevUrl = "http://" + candidate.toString() + ":8000/predict/compact";
        Serial.printf("[DISCOVERY] Found active Mac server at %s:8000!\n", candidate.toString().c_str());
        return;
      }
    }
  }
}

// ══════════════════════════════════════════════════════════════════════
//   PARSE JSON RESPONSE FROM /predict/compact
// ══════════════════════════════════════════════════════════════════════
void parseResponse(String &body) {
#if ARDUINOJSON_VERSION_MAJOR >= 7
  JsonDocument doc;
#else
  StaticJsonDocument<1024> doc;
#endif
  DeserializationError error = deserializeJson(doc, body);
  if (error) {
    lcdStatus("Parse Error", "Bad JSON");
    delay(1500);
    return;
  }

  if (doc.containsKey("ok") && doc["ok"] == 0) {
    lcdStatus("Pred Error", doc["err"] | "Unknown error");
    delay(1500);
    return;
  }

  g_crop = doc["crop"].as<String>();
  g_conf = doc["conf"].as<float>();
  g_crop2 = doc["t2"].as<String>();
  g_conf2 = doc["c2"].as<float>();
  g_crop3 = doc["t3"].as<String>();
  g_conf3 = doc["c3"].as<float>();

  g_alerts = doc["ac"].as<int>();
  g_alertCount = 0;

  JsonArray arr = doc["alerts"].as<JsonArray>();
  for (JsonObject a : arr) {
    if (g_alertCount >= 4)
      break;
    g_alertName[g_alertCount] = a["n"].as<String>();
    g_alertSev[g_alertCount] = a["s"].as<String>();
    g_alertCount++;
  }

  // Reset alert pagination on fresh prediction
  g_alertPage = 0;

  Serial.printf("[AI] Recommended: %s (%.1f%%) | Alerts: %d\n", g_crop.c_str(),
                g_conf, g_alerts);
}

// ══════════════════════════════════════════════════════════════════════
//   LCD SCREEN CAROUSEL ROTATION
// ══════════════════════════════════════════════════════════════════════
void advanceScreen() {
  /*
   * Carousel Progression:
   * Screen 0: Sensor readings (Temperature, Humidity, Soil, Rain)
   * Screen 1: Recommended crop & confidence
   * Screen 2: Disease risk alerts (cycles through each if alerts > 0)
   * Screen 3: Water level & drainage depth
   * Screen 4: GPS NEO-6M & Cloud API status
   */
  if (g_screen == 0) {
    g_screen = (g_crop.length() > 0) ? 1 : 3;
  } else if (g_screen == 1) {
    if (g_alerts > 0) {
      g_screen = 2;
      g_alertPage = 0;
    } else {
      g_screen = 3;
    }
  } else if (g_screen == 2) {
    g_alertPage++;
    if (g_alertPage >= g_alertCount) {
      g_screen = 3;
      g_alertPage = 0;
    }
  } else if (g_screen == 3) {
    g_screen = 4;
  } else if (g_screen == 4) {
    g_screen = 0;
  } else {
    g_screen = 0;
  }
}

void drawScreen() {
  lcd.clear();
  switch (g_screen) {
  case 0:
    screenSensors();
    break;
  case 1:
    screenCrop();
    break;
  case 2:
    screenAlerts();
    break;
  case 3:
    screenWater();
    break;
  case 4:
    screenGps();
    break;
  default:
    screenSensors();
    break;
  }
}

// ══════════════════════════════════════════════════════════════════════
//   LCD SCREEN 0 — SENSOR READINGS
// ══════════════════════════════════════════════════════════════════════
void screenSensors() {
  /*
   * Row 0: T:28^C H:72%
   * Row 1: Soil:45% R:[drop]WET  (or R:DRY)
   */
  lcd.setCursor(0, 0);
  lcd.print("T:");
  lcd.print(g_temp);
  lcd.write((uint8_t)CHAR_DEG);
  lcd.print("C H:");
  lcd.print(g_hum);
  lcd.print("%");

  lcd.setCursor(0, 1);
  lcd.print("Soil:");
  lcd.print(g_soil);
  lcd.print("% R:");
  if (g_rainOn) {
    lcd.write((uint8_t)CHAR_DROP);
    lcd.print("WET");
  } else {
    lcd.print("DRY");
  }
}

// ══════════════════════════════════════════════════════════════════════
//   LCD SCREEN 1 — CROP RECOMMENDATION
// ══════════════════════════════════════════════════════════════════════
void screenCrop() {
  if (g_crop.length() == 0) {
    lcd.setCursor(0, 0);
    lcd.print("No prediction");
    lcd.setCursor(0, 1);
    lcd.print("yet...");
    return;
  }

  /*
   * Row 0: [tick] PAPAYA
   * Row 1: Conf:62.1% [alert]1ALT  (or [tick]OK)
   */
  String cropUp = g_crop;
  cropUp.toUpperCase();

  lcd.setCursor(0, 0);
  lcd.write((uint8_t)CHAR_TICK);
  lcd.print(" ");
  lcd.print(cropUp.substring(0, 14));

  lcd.setCursor(0, 1);
  lcd.print("Conf:");
  lcd.print(g_conf, 1);
  lcd.print("% ");

  if (g_alerts > 0) {
    lcd.write((uint8_t)CHAR_ALERT);
    lcd.print(g_alerts);
    lcd.print("ALT");
  } else {
    lcd.write((uint8_t)CHAR_TICK);
    lcd.print("OK");
  }
}

// ══════════════════════════════════════════════════════════════════════
//   LCD SCREEN 2 — DISEASE ALERTS
// ══════════════════════════════════════════════════════════════════════
void screenAlerts() {
  if (g_alertCount == 0) {
    lcd.setCursor(3, 0);
    lcd.write((uint8_t)CHAR_TICK);
    lcd.print(" ALL CLEAR");
    lcd.setCursor(0, 1);
    lcd.print("No disease risk");
    return;
  }

  int i = g_alertPage;
  if (i >= g_alertCount)
    i = 0;

  String name = g_alertName[i];
  String sev = g_alertSev[i];

  /*
   * Row 0: [alert]Anthracnose
   * Row 1: HIGH        1/2
   */
  lcd.setCursor(0, 0);
  lcd.write((uint8_t)CHAR_ALERT);
  lcd.print(name.substring(0, 15));

  lcd.setCursor(0, 1);
  if (sev == "CRITICAL") {
    lcd.print("CRITICAL");
  } else if (sev == "HIGH") {
    lcd.print("HIGH    ");
  } else if (sev == "MODERATE") {
    lcd.print("MODERATE");
  } else {
    lcd.print("WATCH   ");
  }

  if (g_alertCount > 1) {
    lcd.setCursor(13, 1);
    lcd.print(i + 1);
    lcd.print("/");
    lcd.print(g_alertCount);
  }
}

// ══════════════════════════════════════════════════════════════════════
//   LCD SCREEN 3 — WATER LEVEL & DRAINAGE
// ══════════════════════════════════════════════════════════════════════
void screenWater() {
  /*
   * Row 0: Drainage / Water
   * Row 1: Lvl:12% (5mm)
   */
  lcd.setCursor(0, 0);
  lcd.print("Drainage / Water");

  lcd.setCursor(0, 1);
  lcd.print("Lvl:");
  lcd.print(g_waterPct);
  lcd.print("% (");
  lcd.print((int)round(g_waterMm));
  lcd.print("mm)");
}

// ══════════════════════════════════════════════════════════════════════
//   LCD SCREEN 4 — GPS LOCATION & STATUS
// ══════════════════════════════════════════════════════════════════════
void screenGps() {
  /*
   * Row 0: GPS: 3D FIX 8S
   * Row 1: WiFi:OK API:OK
   */
  lcd.setCursor(0, 0);
  if (g_gpsValid) {
    lcd.print("GPS: 3D FIX ");
    lcd.print(g_sats);
    lcd.print("S");
  } else {
    lcd.print("GPS: SRCH ");
    lcd.print(g_sats);
    lcd.print(" SAT");
  }

  lcd.setCursor(0, 1);
  bool wifiOk = (WiFi.status() == WL_CONNECTED);
  lcd.print("WiFi:");
  lcd.print(wifiOk ? "OK" : "NO");
  lcd.print(" API:");
  lcd.print(g_serverOK ? "OK" : "OFF");
}

// ══════════════════════════════════════════════════════════════════════
//   UTILITY FUNCTIONS
// ══════════════════════════════════════════════════════════════════════
void lcdStatus(String line0, String line1) {
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print(line0.substring(0, 16));
  lcd.setCursor(0, 1);
  lcd.print(line1.substring(0, 16));
}

// ── Boot Screen Animation ─────────────────────────────────────────────
void showBoot() {
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("SMART PLANT AI");
  lcd.setCursor(0, 1);
  lcd.print("Initializing...");
  delay(1500);

  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("Connecting WiFi");
  lcd.setCursor(0, 1);
  lcd.print(WIFI_SSID);
}

// ── WiFi Connection Animation ─────────────────────────────────────────
void connectWiFi() {
  Serial.print("[WIFI] Connecting to: ");
  Serial.println(WIFI_SSID);

  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASS);

  int tries = 0;
  while (WiFi.status() != WL_CONNECTED && tries < 40) {
    delay(500);
    Serial.print(".");
    tries++;
  }

  if (WiFi.status() == WL_CONNECTED) {
    Serial.println();
    Serial.print("[WIFI] Connected! IP: ");
    Serial.println(WiFi.localIP());

    lcd.clear();
    lcd.setCursor(0, 0);
    lcd.print("WiFi Connected!");
    lcd.setCursor(0, 1);
    lcd.print(WiFi.localIP());
    delay(2500);

    lcd.clear();
    lcd.setCursor(0, 0);
    lcd.print("Server:");
    lcd.setCursor(0, 1);
    lcd.print(TARGET_LABEL);
    delay(2000);
  } else {
    Serial.println(F("\n[WIFI] Connect FAILED!"));
    lcd.clear();
    lcd.setCursor(0, 0);
    lcd.print("!WiFi FAILED!");
    lcd.setCursor(0, 1);
    lcd.print("Check SSID/Pass");
    delay(4000);
  }
}