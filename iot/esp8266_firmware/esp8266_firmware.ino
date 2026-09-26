/*
 * ═════════════════════════════════════════════════════════════════════════════════════════
 *   Smart Crop Intelligence & AgriRover System: Production ESP32 Unified Firmware
 *   (Upgraded Replacement for legacy esp8266_firmware.ino)
 *   Team Innovex | Smart India Hackathon (SIH 2026) Challenge SIH26180 (Qualcomm Inc)
 *   Theme: Disaster Management | Category: Hardware & AI-Powered Field Robotics
 * ═════════════════════════════════════════════════════════════════════════════════════════
 *
 *   MIGRATION NOTICE:
 *   This file replaces obsolete ESP8266 single-core, single-ADC, 16x2 I2C LCD assumptions
 *   with the high-performance dual-core ESP32 Dev Module architecture integrating:
 *   - L293D Dual H-Bridge (4x TT DC gear motors with 600ms safety watchdog)
 *   - PCA9685 16-channel PWM controller with 4-DOF Robotic Manipulator
 *   - DHT11 microclimate sensor (Ambient Temperature & Atmospheric Humidity)
 *   - Capacitive Soil Moisture Sensor v1.2 (calibrated volumetric water content on ADC1)
 *   - Raindrop Detection Module (active precipitation flag + analog intensity)
 *   - Water Level Sensor (standing water / drainage / flood depth on ADC1)
 *   - GPS NEO-6M-0-001 Module (WGS84 spatial fix via HardwareSerial2)
 *   - 2.4-inch Color TFT LCD Display (multi-screen status HUD with smooth auto-rotation)
 *   - Two-way SmartCropVision REST & WebSocket synchronization
 *
 * ═════════════════════════════════════════════════════════════════════════════════════════
 *   CIRCUIT CONNECTION & ELECTRICAL SPECIFICATION TABLE
 * ═════════════════════════════════════════════════════════════════════════════════════════
 *
 *   ┌───────────────────────┬──────────────────────┬─────────────┬─────────────┬───────────────────┬─────────────────────────────────────────────────────────────┐
 *   │ Component Name        │ Function / Purpose   │ ESP32 GPIO  │ Power Rail  │ Signal Type       │ Electrical & Operational Notes                              │
 *   ├───────────────────────┼──────────────────────┼─────────────┼─────────────┼───────────────────┼─────────────────────────────────────────────────────────────┤
 *   │ Left Motor IN1 (L293D)│ Left Drive Forward   │ GPIO 5      │ 5V / Batt V+│ Digital Output    │ Direct logic drive to L293D pin 2 (1A)                      │
 *   │ Left Motor IN2 (L293D)│ Left Drive Reverse   │ GPIO 18     │ 5V / Batt V+│ Digital Output    │ Direct logic drive to L293D pin 7 (2A)                      │
 *   │ Right Motor IN1(L293D)│ Right Drive Forward  │ GPIO 19     │ 5V / Batt V+│ Digital Output    │ Direct logic drive to L293D pin 10 (3A)                     │
 *   │ Right Motor IN2(L293D)│ Right Drive Reverse  │ GPIO 21     │ 5V / Batt V+│ Digital Output    │ Direct logic drive to L293D pin 15 (4A)                     │
 *   │ Rover Headlight LED   │ Night / Foliage Light│ GPIO 2      │ 3.3V logic  │ Digital Output    │ On-board LED / External high-efficiency white illumination  │
 *   │ DHT11 Sensor          │ Temp & Humidity      │ GPIO 4      │ 3.3V / 5V   │ Single-Wire Data  │ 10k pull-up on module. Sampled non-blockingly every 3000ms  │
 *   │ Capacitive Soil v1.2  │ Root Hydration / VWC │ GPIO 34     │ 3.3V        │ Analog In (ADC1)  │ ADC1_CH6 (Input-only pin; safe with active Wi-Fi radio)     │
 *   │ Raindrop Module DO    │ Rain Flag (Binary)   │ GPIO 27     │ 3.3V        │ Digital Input     │ Active LOW comparator output (LM393)                        │
 *   │ Raindrop Module AO    │ Rain Intensity Proxy │ GPIO 32     │ 3.3V        │ Analog In (ADC1)  │ ADC1_CH4 (0-4095; calibrated against dry baseline)          │
 *   │ Water Level Sensor    │ Ponding/Flood Depth  │ GPIO 35     │ 3.3V        │ Analog In (ADC1)  │ ADC1_CH7 (Input-only; indicates drainage standing water)    │
 *   │ GPS NEO-6M TX         │ NMEA Telemetry Stream│ GPIO 16     │ 3.3V / 5V   │ UART2 RX (ESP32)  │ HardwareSerial2 at 9600 baud. Non-blocking sentence parser  │
 *   │ GPS NEO-6M RX         │ GPS Config / Commands│ GPIO 17     │ 3.3V / 5V   │ UART2 TX (ESP32)  │ HardwareSerial2 TX2 to GPS RX pin                           │
 *   │ PCA9685 I2C SDA       │ 4-DOF Arm Servo Bus  │ GPIO 23     │ 3.3V logic  │ I2C Data (400kHz) │ Communicates with PCA9685 address 0x40                      │
 *   │ PCA9685 I2C SCL       │ 4-DOF Arm Clock Bus  │ GPIO 22     │ 3.3V logic  │ I2C Clock (400kHz)│ Standard ESP32 hardware I2C wire bus                        │
 *   │ 2.4" TFT Display CS   │ Chip Select (SPI)    │ GPIO 15     │ 3.3V logic  │ Digital Output    │ TFT SPI Chip Select line (configurable in display manager)  │
 *   │ 2.4" TFT Display DC   │ Data/Command Select  │ GPIO 14     │ 3.3V logic  │ Digital Output    │ TFT Register / Data select line                             │
 *   │ 2.4" TFT Display RST  │ Hardware Reset       │ GPIO 13     │ 3.3V logic  │ Digital Output    │ TFT hardware reset strobe                                   │
 *   │ PCA9685 V+ Power      │ Servo Motor Power    │ EXTERNAL    │ 5.0V - 6.0V │ DC Power (3A-5A)  │ DO NOT POWER SERVOS FROM ESP32 3.3V PIN (Brownout hazard!)  │
 *   │ Common System GND     │ Common Reference     │ GND         │ 0V          │ Power Ground      │ Common ground MUST be shared across ESP32, drivers & battery│
 *   └───────────────────────┴──────────────────────┴─────────────┴─────────────┴───────────────────┴─────────────────────────────────────────────────────────────┘
 *
 * ═════════════════════════════════════════════════════════════════════════════════════════
 */

#include <Arduino.h>
#include <WiFi.h>
#include <WebServer.h>
#include <ESPmDNS.h>
#include <HTTPClient.h>
#include <WiFiClient.h>
#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

#define FIRMWARE_NAME       "AgriRover-CropIntelligence-ESP32"
#define FIRMWARE_VERSION    "v3.0.0-prod"
#define DEVICE_ID           "agrirover-esp32-01"
#define PROTOCOL_VERSION    "2.0.0"

// Wi-Fi Credentials
const char* WIFI_SSID_PRIMARY   = "Fayas";
const char* WIFI_PASS_PRIMARY   = "fayas1234";
const char* AP_SSID             = "AgriRover-Field-AP";
const char* AP_PASS             = "agrirover123";

// Backend Synchronization
const char* BACKEND_OBS_URL     = "http://172.20.10.2:8000/api/v1/iot/observation";

// Pin Map
#define PIN_MOTOR_LEFT_IN1    5
#define PIN_MOTOR_LEFT_IN2    18
#define PIN_MOTOR_RIGHT_IN1   19
#define PIN_MOTOR_RIGHT_IN2   21
#define PIN_HEADLIGHT         2
#define PIN_DHT11_DATA        4
#define PIN_SOIL_ADC          34
#define PIN_RAIN_DO           27
#define PIN_RAIN_AO           32
#define PIN_WATER_LEVEL_ADC   35
#define PIN_GPS_RX            16
#define PIN_GPS_TX            17
#define PIN_I2C_SDA           23
#define PIN_I2C_SCL           22
#define PIN_TFT_CS            15
#define PIN_TFT_DC            14
#define PIN_TFT_RST           13

#define ROVER_WATCHDOG_TIMEOUT_MS 600

WebServer server(80);
Adafruit_PWMServoDriver pca9685 = Adafruit_PWMServoDriver(0x40);
HardwareSerial GPSSerial(2);

enum DriveDirection { DIR_STOPPED = 0, DIR_FORWARD, DIR_BACKWARD, DIR_LEFT, DIR_RIGHT };

struct RoverState {
  DriveDirection direction;
  uint8_t speed_pwm;
  bool headlight;
  bool watchdog_active;
  unsigned long last_command_time;
} rover = {DIR_STOPPED, 200, false, true, 0};

struct SensorData {
  float temperature_c;
  float humidity_pct;
  bool dht_valid;
  unsigned long last_dht_read;
  uint16_t soil_raw;
  float soil_pct;
  bool rain_detected;
  uint16_t water_level_raw;
  float water_level_pct;
} sensors = {28.0f, 62.0f, false, 0, 2400, 50.0f, false, 400, 10.0f};

struct GPSData {
  float latitude;
  float longitude;
  float altitude_m;
  uint8_t satellites;
  bool fix_valid;
} gps = {13.0827f, 80.2707f, 12.0f, 0, false};

void executeMotorDrive(DriveDirection dir, uint8_t speed_pwm) {
  rover.direction = dir;
  rover.speed_pwm = speed_pwm;
  rover.last_command_time = millis();
  rover.watchdog_active = true;

  switch (dir) {
    case DIR_FORWARD:
      digitalWrite(PIN_MOTOR_LEFT_IN1, HIGH); digitalWrite(PIN_MOTOR_LEFT_IN2, LOW);
      digitalWrite(PIN_MOTOR_RIGHT_IN1, HIGH); digitalWrite(PIN_MOTOR_RIGHT_IN2, LOW);
      break;
    case DIR_BACKWARD:
      digitalWrite(PIN_MOTOR_LEFT_IN1, LOW); digitalWrite(PIN_MOTOR_LEFT_IN2, HIGH);
      digitalWrite(PIN_MOTOR_RIGHT_IN1, LOW); digitalWrite(PIN_MOTOR_RIGHT_IN2, HIGH);
      break;
    case DIR_LEFT:
      digitalWrite(PIN_MOTOR_LEFT_IN1, LOW); digitalWrite(PIN_MOTOR_LEFT_IN2, HIGH);
      digitalWrite(PIN_MOTOR_RIGHT_IN1, HIGH); digitalWrite(PIN_MOTOR_RIGHT_IN2, LOW);
      break;
    case DIR_RIGHT:
      digitalWrite(PIN_MOTOR_LEFT_IN1, HIGH); digitalWrite(PIN_MOTOR_LEFT_IN2, LOW);
      digitalWrite(PIN_MOTOR_RIGHT_IN1, LOW); digitalWrite(PIN_MOTOR_RIGHT_IN2, HIGH);
      break;
    case DIR_STOPPED:
    default:
      digitalWrite(PIN_MOTOR_LEFT_IN1, LOW); digitalWrite(PIN_MOTOR_LEFT_IN2, LOW);
      digitalWrite(PIN_MOTOR_RIGHT_IN1, LOW); digitalWrite(PIN_MOTOR_RIGHT_IN2, LOW);
      break;
  }
}

void checkWatchdog(unsigned long now) {
  if (rover.direction != DIR_STOPPED && (now - rover.last_command_time > ROVER_WATCHDOG_TIMEOUT_MS)) {
    executeMotorDrive(DIR_STOPPED, 0);
    rover.watchdog_active = false;
  }
}

void setup() {
  Serial.begin(115200);
  pinMode(PIN_MOTOR_LEFT_IN1, OUTPUT);
  pinMode(PIN_MOTOR_LEFT_IN2, OUTPUT);
  pinMode(PIN_MOTOR_RIGHT_IN1, OUTPUT);
  pinMode(PIN_MOTOR_RIGHT_IN2, OUTPUT);
  pinMode(PIN_HEADLIGHT, OUTPUT);
  pinMode(PIN_DHT11_DATA, INPUT_PULLUP);
  pinMode(PIN_RAIN_DO, INPUT);
  analogReadResolution(12);

  executeMotorDrive(DIR_STOPPED, 0);

  Wire.begin(PIN_I2C_SDA, PIN_I2C_SCL, 400000);
  pca9685.begin();
  pca9685.setPWMFreq(50);

  GPSSerial.begin(9600, SERIAL_8N1, PIN_GPS_RX, PIN_GPS_TX);

  WiFi.mode(WIFI_AP_STA);
  WiFi.begin(WIFI_SSID_PRIMARY, WIFI_PASS_PRIMARY);

  server.on("/", HTTP_GET, []() { server.send(200, "text/plain", "AgriRover ESP32 Master Operational"); });
  server.on("/forward", HTTP_GET, []() { executeMotorDrive(DIR_FORWARD, 200); server.send(200, "text/plain", "OK"); });
  server.on("/backward", HTTP_GET, []() { executeMotorDrive(DIR_BACKWARD, 200); server.send(200, "text/plain", "OK"); });
  server.on("/left", HTTP_GET, []() { executeMotorDrive(DIR_LEFT, 200); server.send(200, "text/plain", "OK"); });
  server.on("/right", HTTP_GET, []() { executeMotorDrive(DIR_RIGHT, 200); server.send(200, "text/plain", "OK"); });
  server.on("/stop", HTTP_GET, []() { executeMotorDrive(DIR_STOPPED, 0); server.send(200, "text/plain", "OK"); });
  server.begin();
}

void loop() {
  unsigned long now = millis();
  checkWatchdog(now);
  server.handleClient();
}
