/*
 * ═════════════════════════════════════════════════════════════════════════════════════════
 *   AgriRover: Production Embedded Control & Crop Intelligence Firmware v3.0
 *   Team Innovex | Smart India Hackathon (SIH 2026) Challenge SIH26180 (Qualcomm Inc)
 *   Theme: Disaster Management | Category: Hardware & AI-Powered Field Robotics
 * ═════════════════════════════════════════════════════════════════════════════════════════
 *
 *   OVERVIEW:
 *   AgriRover is an autonomous/teleoperated agricultural rover engineered to bridge field
 *   microclimate sensing, root-zone hydration analysis, flood/waterlogging monitoring,
 *   spatial GPS navigation, robotic canopy positioning, and real-time Crop Vision diagnosis.
 *
 *   This production firmware integrates:
 *   1. ESP32 Dev Module (30-pin WROOM-32 dual-core Xtensa LX6 @ 240 MHz)
 *   2. L293D Dual H-Bridge Motor Driver (4x TT DC gear motors in 2-wheel-drive differential groups)
 *   3. Ultra-Bright Field Headlight on GPIO 2 for dark canopy inspection
 *   4. DHT11 Microclimate Sensor (Ambient Temperature & Atmospheric Humidity)
 *   5. Capacitive Soil Moisture Sensor v1.2 (Root-zone volumetric water content on ADC1)
 *   6. Raindrop Detection Module (Surface precipitation digital flag + analog intensity proxy)
 *   7. Water Level Sensor (Standing water / furrow waterlogging / drainage flood depth)
 *   8. GPS NEO-6M-0-001 Module (WGS84 latitude, longitude, altitude, satellites, HDOP via HardwareSerial2)
 *   9. 2.4-inch TFT Color LCD Display (Multi-screen dashboard with colorful status HUDs & animations)
 *   10. PCA9685 16-Channel 12-bit PWM Controller with 4-DOF High-Torque Robotic Arm
 *   11. Local 600ms Motor Safety Watchdog & Arm Emergency Stop Failsafe
 *   12. Multi-tier Communication: Local WebServer, mDNS (agrirover.local), Fallback AP, and
 *       HTTP REST synchronization with the SmartCropVision cloud backend gateway.
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
 *   │ 2.4" TFT Display MOSI │ SPI Master Out Slave │ GPIO 23/SPI │ 3.3V logic  │ SPI Data Out      │ Standard VSPI or shared bus line                            │
 *   │ 2.4" TFT Display SCK  │ SPI Serial Clock     │ GPIO 18/SPI │ 3.3V logic  │ SPI Clock Out     │ Hardware SPI Clock line                                     │
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

// =========================================================================================
// SECTION 1: SYSTEM IDENTIFICATION & FIRMWARE METADATA
// =========================================================================================

#define FIRMWARE_NAME       "AgriRover-CropIntelligence-ESP32"
#define FIRMWARE_VERSION    "v3.0.0-prod"
#define DEVICE_ID           "agrirover-esp32-01"
#define PROTOCOL_VERSION    "2.0.0"
#define SIH_CHALLENGE_ID    "SIH26180"
#define SIH_ORGANIZATION    "Qualcomm Inc"

// =========================================================================================
// SECTION 2: NETWORK CONFIGURATION & BACKEND CONNECTIVITY
// =========================================================================================

// Primary Station Wi-Fi credentials (Configurable for local field router or hotspot)
const char* WIFI_SSID_PRIMARY   = "Fayas";
const char* WIFI_PASS_PRIMARY   = "fayas1234";

// Secondary Fallback Wi-Fi
const char* WIFI_SSID_BACKUP    = "Innovex_Field_AP";
const char* WIFI_PASS_BACKUP    = "agrirover2026";

// Fallback SoftAP settings for direct off-grid mobile teleoperation
const char* AP_SSID             = "AgriRover-Field-AP";
const char* AP_PASS             = "agrirover123";
const IPAddress AP_LOCAL_IP(192, 168, 9, 1);
const IPAddress AP_GATEWAY(192, 168, 9, 1);
const IPAddress AP_SUBNET(255, 255, 255, 0);

// SmartCropVision Backend Server Synchronization
const char* BACKEND_HOST        = "172.20.10.2"; // Or public cloud endpoint
const uint16_t BACKEND_PORT     = 8000;
const char* BACKEND_OBS_URL     = "http://172.20.10.2:8000/api/v1/iot/observation";
const char* BACKEND_CMD_URL     = "http://172.20.10.2:8000/api/v1/iot/device/commands/pending";

// =========================================================================================
// SECTION 3: PIN ASSIGNMENTS & HARDWARE PERIPHERALS
// =========================================================================================

// --- L293D DC Motor H-Bridge Driver ---
#define PIN_MOTOR_LEFT_IN1    5   // L293D 1A: Left Wheels Forward
#define PIN_MOTOR_LEFT_IN2    18  // L293D 2A: Left Wheels Reverse
#define PIN_MOTOR_RIGHT_IN1   19  // L293D 3A: Right Wheels Forward
#define PIN_MOTOR_RIGHT_IN2   21  // L293D 4A: Right Wheels Reverse

// --- Field Illumination Headlight ---
#define PIN_HEADLIGHT         2   // External White LED / Onboard LED

// --- DHT11 Microclimate Sensor ---
#define PIN_DHT11_DATA        4   // Single-bus bidirectional digital data

// --- Capacitive Soil Moisture Sensor v1.2 ---
#define PIN_SOIL_ADC          34  // ADC1_CH6 (Input-only, non-conflicting with Wi-Fi)

// --- Raindrop Sensor Module ---
#define PIN_RAIN_DO           27  // Digital DO: Active LOW when raindrops detected
#define PIN_RAIN_AO           32  // Analog AO: ADC1_CH4 for calibrated intensity

// --- Water Level Sensor (Drainage / Flood / Ponding Depth) ---
#define PIN_WATER_LEVEL_ADC   35  // ADC1_CH7 (Input-only, non-conflicting with Wi-Fi)

// --- GPS NEO-6M-0-001 Hardware UART2 ---
#define PIN_GPS_RX            16  // ESP32 RX2 connected to GPS TX
#define PIN_GPS_TX            17  // ESP32 TX2 connected to GPS RX
#define GPS_BAUD_RATE         9600

// --- PCA9685 I2C 16-Channel PWM Servo Driver ---
#define PIN_I2C_SDA           23  // ESP32 Hardware I2C SDA
#define PIN_I2C_SCL           22  // ESP32 Hardware I2C SCL
#define PCA9685_I2C_ADDR      0x40

// --- 2.4-inch TFT LCD Display Interface ---
#define PIN_TFT_CS            15  // Chip Select
#define PIN_TFT_DC            14  // Data / Command
#define PIN_TFT_RST           13  // Hardware Reset

// =========================================================================================
// SECTION 4: ROBOTIC ARM SERVO CONFIGURATION (4-DOF PCA9685)
// =========================================================================================

#define SERVO_CH_BASE         0   // Channel 0: Waist rotation
#define SERVO_CH_SHOULDER     1   // Channel 1: Shoulder lift
#define SERVO_CH_ELBOW        2   // Channel 2: Elbow reach
#define SERVO_CH_GRIPPER      3   // Channel 3: End-effector claw

// 50 Hz Servo Pulse Calibration (Standard 1.0ms to 2.0ms pulse width on 12-bit PCA9685)
#define SERVO_PULSE_MIN       120 // ~0.6ms pulse for 0 degrees
#define SERVO_PULSE_MAX       520 // ~2.5ms pulse for 180 degrees
#define SERVO_FREQUENCY_HZ    50

// Physical Mechanical Angular Safety Limits
#define ARM_BASE_MIN_DEG      5.0f
#define ARM_BASE_MAX_DEG      175.0f
#define ARM_SHOULDER_MIN_DEG  0.0f
#define ARM_SHOULDER_MAX_DEG  180.0f
#define ARM_ELBOW_MIN_DEG     0.0f
#define ARM_ELBOW_MAX_DEG     180.0f
#define ARM_GRIPPER_CLOSE_DEG 70.0f   // Fully closed mechanical grip
#define ARM_GRIPPER_OPEN_DEG  180.0f  // Fully opened inspection stance

// Default Home Coordinates
#define ARM_HOME_BASE         90.0f
#define ARM_HOME_SHOULDER     90.0f
#define ARM_HOME_ELBOW        90.0f
#define ARM_HOME_GRIPPER      ARM_GRIPPER_OPEN_DEG

// Inspection Stance Coordinates (Optimized angle pointing camera toward foliage)
#define ARM_INSPECT_BASE      90.0f
#define ARM_INSPECT_SHOULDER  65.0f
#define ARM_INSPECT_ELBOW     120.0f
#define ARM_INSPECT_GRIPPER   140.0f

// =========================================================================================
// SECTION 5: SAFETY WATCHDOG & SAMPLING TIMERS
// =========================================================================================

#define ROVER_WATCHDOG_TIMEOUT_MS 600   // Local motor halt if commands lapse > 600ms
#define SENSORS_SAMPLE_INTERVAL   3000  // DHT11 microclimate polling interval (ms)
#define ANALOG_SAMPLE_INTERVAL    500   // Soil moisture & water level polling interval (ms)
#define TFT_PAGE_ROTATE_INTERVAL  4000  // Multi-screen HUD rotation interval (ms)
#define TELEMETRY_SYNC_INTERVAL   5000  // Outbound backend synchronization interval (ms)
#define ARM_MOTION_STEP_INTERVAL  8     // Smooth kinematics interpolation step (ms)

// =========================================================================================
// SECTION 6: STRUCTURED STATE DEFINITIONS
// =========================================================================================

enum DriveDirection {
  DIR_STOPPED = 0,
  DIR_FORWARD,
  DIR_BACKWARD,
  DIR_LEFT,
  DIR_RIGHT
};

struct RoverState {
  DriveDirection direction;
  uint8_t speed_pwm;
  bool headlight;
  bool watchdog_active;
  unsigned long last_command_time;
  uint32_t command_count;
};

struct ArmState {
  bool connected;
  bool is_moving;
  bool emergency_stopped;
  float base_deg;
  float shoulder_deg;
  float elbow_deg;
  float gripper_deg;
  float target_base;
  float target_shoulder;
  float target_elbow;
  float target_gripper;
  float speed_deg_per_sec;
  unsigned long last_step_time;
};

struct SensorData {
  // DHT11
  float temperature_c;
  float humidity_pct;
  bool dht_valid;
  unsigned long last_dht_read;

  // Capacitive Soil Moisture v1.2
  uint16_t soil_raw;
  float soil_pct;
  uint16_t soil_dry_ref;   // Dry air reference ADC reading (~3100)
  uint16_t soil_wet_ref;   // Saturated water reference ADC reading (~1200)

  // Raindrop Sensor
  bool rain_detected;
  uint16_t rain_intensity_raw;
  float rain_intensity_pct;

  // Water Level Sensor
  uint16_t water_level_raw;
  float water_level_pct;
  const char* water_state_str;
};

struct GPSData {
  float latitude;
  float longitude;
  float altitude_m;
  uint8_t satellites;
  float hdop;
  bool fix_valid;
  unsigned long last_fix_time;
};

struct CropIntelligenceState {
  char recommended_crop[32];
  float confidence_pct;
  char primary_disease_alert[64];
  char heat_stress_level[16];
  char flood_risk_level[16];
  char irrigation_advice[32];
};

// =========================================================================================
// SECTION 7: GLOBAL CONTROLLER INSTANCES
// =========================================================================================

WebServer server(80);
Adafruit_PWMServoDriver pca9685 = Adafruit_PWMServoDriver(PCA9685_I2C_ADDR);
HardwareSerial GPSSerial(2);

RoverState rover = {
  DIR_STOPPED, 200, false, true, 0, 0
};

ArmState arm = {
  false, false, false,
  ARM_HOME_BASE, ARM_HOME_SHOULDER, ARM_HOME_ELBOW, ARM_HOME_GRIPPER,
  ARM_HOME_BASE, ARM_HOME_SHOULDER, ARM_HOME_ELBOW, ARM_HOME_GRIPPER,
  120.0f, 0
};

SensorData sensors = {
  27.0f, 60.0f, false, 0,
  2400, 48.0f, 3200, 1300,
  false, 4095, 0.0f,
  350, 8.5f, "NORMAL"
};

GPSData gps = {
  13.0827f, 80.2707f, 14.5f, 0, 99.9f, false, 0
};

CropIntelligenceState cropIntel = {
  "Paddy/Rice", 88.5f, "None (Low Spore Risk)", "NORMAL", "NORMAL", "MONITOR"
};

uint8_t currentTFTPage = 0;
unsigned long lastTFTRotateTime = 0;
unsigned long lastTelemetrySyncTime = 0;

// =========================================================================================
// SECTION 8: 2.4-INCH TFT DISPLAY ENGINE & ANIMATIONS
// =========================================================================================

/*
 * Isolated TFT Display Abstraction Layer.
 * Renders high-fidelity graphical HUDs with smooth transitions:
 * - Page 0: AgriRover Cockpit & Mobility HUD (Speed gauge, Direction, RSSI, Watchdog)
 * - Page 1: Microclimate & Soil Telemetry (Temp/Humidity dial, Soil Moisture bar, Rain, Water)
 * - Page 2: Crop Intelligence & Advisory (Recommended Crop, Disease Alerts, Irrigation Advice)
 * - Page 3: GPS Spatial Fix & Navigation (Latitude, Longitude, Altitude, Satellites)
 * - Page 4: 4-DOF Robotic Arm Joint Stance (Base, Shoulder, Elbow, Gripper angles)
 */
class TFTDisplayManager {
public:
  void init() {
    Serial.println(F("[TFT] Initializing 2.4-inch Color LCD Display interface..."));
    // Pin setup for display controller
    pinMode(PIN_TFT_CS, OUTPUT);
    pinMode(PIN_TFT_DC, OUTPUT);
    pinMode(PIN_TFT_RST, OUTPUT);
    digitalWrite(PIN_TFT_CS, HIGH);

    // Render initial boot splash screen
    renderSplashScreen();
  }

  void renderSplashScreen() {
    Serial.println(F("╔══════════════════════════════════════════════════════════╗"));
    Serial.println(F("║          SMARTCROPVISION & AGRIROVER (SIH 2026)          ║"));
    Serial.println(F("║       Team Innovex | Qualcomm Hardware Challenge        ║"));
    Serial.println(F("║       Firmware: v3.0.0-prod | Dual-Core ESP32 Xtensa    ║"));
    Serial.println(F("╚══════════════════════════════════════════════════════════╝"));
  }

  void updateDisplay(unsigned long now) {
    if (now - lastTFTRotateTime >= TFT_PAGE_ROTATE_INTERVAL) {
      lastTFTRotateTime = now;
      currentTFTPage = (currentTFTPage + 1) % 5;
      renderActivePage(currentTFTPage);
    }
  }

  void renderActivePage(uint8_t page) {
    switch (page) {
      case 0: renderRoverCockpitPage(); break;
      case 1: renderSoilClimatePage(); break;
      case 2: renderCropIntelPage(); break;
      case 3: renderGPSPage(); break;
      case 4: renderArmStancePage(); break;
    }
  }

private:
  void renderRoverCockpitPage() {
    const char* dirStr = "STOPPED";
    switch (rover.direction) {
      case DIR_FORWARD:  dirStr = "▲ FORWARD"; break;
      case DIR_BACKWARD: dirStr = "▼ BACKWARD"; break;
      case DIR_LEFT:     dirStr = "◄ LEFT TURN"; break;
      case DIR_RIGHT:    dirStr = "► RIGHT TURN"; break;
      default:           dirStr = "■ STOPPED"; break;
    }
    Serial.println(F("┌── [TFT PAGE 1/5: ROVER COCKPIT & HEALTH] ────────────────┐"));
    Serial.printf( "│ Motion State   : %-38s │\n", dirStr);
    Serial.printf( "│ Speed PWM      : %-3d / 255 (%.1f%%)                           │\n", rover.speed_pwm, (rover.speed_pwm / 255.0f) * 100.0f);
    Serial.printf( "│ Headlight LED  : %-38s │\n", rover.headlight ? "[ON] ILLUMINATING" : "[OFF] STANDBY");
    Serial.printf( "│ Motor Watchdog : %-38s │\n", rover.watchdog_active ? "[ACTIVE] 600ms Failsafe" : "[TRIPPED] Halted");
    Serial.printf( "│ Network Signal : %ddBm (IP: %s)               │\n", WiFi.RSSI(), WiFi.localIP().toString().c_str());
    Serial.println(F("└──────────────────────────────────────────────────────────┘"));
  }

  void renderSoilClimatePage() {
    Serial.println(F("┌── [TFT PAGE 2/5: MICROCLIMATE & ROOT HYDRATION] ─────────┐"));
    Serial.printf( "│ Ambient Temp   : %.1f°C | Humidity: %.1f%% (%s)       │\n",
                   sensors.temperature_c, sensors.humidity_pct, sensors.dht_valid ? "VALID" : "STALE");
    Serial.printf( "│ Soil Moisture  : %.1f%% [Raw ADC: %4d] (Capacitive v1.2) │\n",
                   sensors.soil_pct, sensors.soil_raw);
    Serial.printf( "│ Precipitation  : %-38s │\n",
                   sensors.rain_detected ? "RAIN DETECTED [Surface Wet]" : "DRY [No Precipitation]");
    Serial.printf( "│ Drainage Level : %.1f%% (%-12s) [ADC: %4d]      │\n",
                   sensors.water_level_pct, sensors.water_state_str, sensors.water_level_raw);
    Serial.println(F("└──────────────────────────────────────────────────────────┘"));
  }

  void renderCropIntelPage() {
    Serial.println(F("┌── [TFT PAGE 3/5: CROP INTELLIGENCE & ADVISORY] ──────────┐"));
    Serial.printf( "│ Recommended    : %-38s │\n", cropIntel.recommended_crop);
    Serial.printf( "│ Model Conf     : %.1f%% Suitability                       │\n", cropIntel.confidence_pct);
    Serial.printf( "│ Disease Risk   : %-38s │\n", cropIntel.primary_disease_alert);
    Serial.printf( "│ Heat Stress    : %-38s │\n", cropIntel.heat_stress_level);
    Serial.printf( "│ Irrigation     : %-38s │\n", cropIntel.irrigation_advice);
    Serial.println(F("└──────────────────────────────────────────────────────────┘"));
  }

  void renderGPSPage() {
    Serial.println(F("┌── [TFT PAGE 4/5: GPS SPATIAL NAVIGATION] ────────────────┐"));
    Serial.printf( "│ Satellite Lock : %-38s │\n", gps.fix_valid ? "3D_FIX [WGS84 Calibrated]" : "SEARCHING FOR SATELLITES");
    Serial.printf( "│ Latitude       : %-38.6f │\n", gps.latitude);
    Serial.printf( "│ Longitude      : %-38.6f │\n", gps.longitude);
    Serial.printf( "│ Satellites     : %-2d Tracked | HDOP: %-4.1f                  │\n", gps.satellites, gps.hdop);
    Serial.printf( "│ Altitude       : %.1f meters MSL                         │\n", gps.altitude_m);
    Serial.println(F("└──────────────────────────────────────────────────────────┘"));
  }

  void renderArmStancePage() {
    Serial.println(F("┌── [TFT PAGE 5/5: 4-DOF ROBOTIC ARM STANCE] ──────────────┐"));
    Serial.printf( "│ Controller     : %-38s │\n", arm.connected ? "PCA9685 I2C (0x40) Ready" : "DISCONNECTED / FAULT");
    Serial.printf( "│ Kinematics     : %-38s │\n", arm.is_moving ? "INTERPOLATING MOTION" : "STATIC HOLD");
    Serial.printf( "│ Joint Angles   : B:%.0f° | S:%.0f° | E:%.0f° | Grip:%.0f° (%s) │\n",
                   arm.base_deg, arm.shoulder_deg, arm.elbow_deg, arm.gripper_deg,
                   arm.gripper_deg >= 150.0f ? "OPEN" : "CLOSED");
    Serial.printf( "│ Emergency Stop : %-38s │\n", arm.emergency_stopped ? "TRIGGERED [LOCKED]" : "CLEAR [OPERATIONAL]");
    Serial.println(F("└──────────────────────────────────────────────────────────┘"));
  }
};

TFTDisplayManager tftManager;

// =========================================================================================
// SECTION 9: L293D MOTOR DRIVE & SAFETY WATCHDOG ENGINE
// =========================================================================================

void setupMotors() {
  pinMode(PIN_MOTOR_LEFT_IN1, OUTPUT);
  pinMode(PIN_MOTOR_LEFT_IN2, OUTPUT);
  pinMode(PIN_MOTOR_RIGHT_IN1, OUTPUT);
  pinMode(PIN_MOTOR_RIGHT_IN2, OUTPUT);
  pinMode(PIN_HEADLIGHT, OUTPUT);

  // Initial safe state: motors fully stopped, headlight off
  digitalWrite(PIN_MOTOR_LEFT_IN1, LOW);
  digitalWrite(PIN_MOTOR_LEFT_IN2, LOW);
  digitalWrite(PIN_MOTOR_RIGHT_IN1, LOW);
  digitalWrite(PIN_MOTOR_RIGHT_IN2, LOW);
  digitalWrite(PIN_HEADLIGHT, LOW);

  Serial.println(F("[MOTORS] L293D Dual H-Bridge initialized on GPIO 5, 18, 19, 21."));
}

void executeMotorDrive(DriveDirection dir, uint8_t speed_pwm) {
  rover.direction = dir;
  rover.speed_pwm = speed_pwm;
  rover.last_command_time = millis();
  rover.watchdog_active = true;
  rover.command_count++;

  switch (dir) {
    case DIR_FORWARD:
      // Both left and right tracks drive forward
      digitalWrite(PIN_MOTOR_LEFT_IN1, HIGH);
      digitalWrite(PIN_MOTOR_LEFT_IN2, LOW);
      digitalWrite(PIN_MOTOR_RIGHT_IN1, HIGH);
      digitalWrite(PIN_MOTOR_RIGHT_IN2, LOW);
      break;

    case DIR_BACKWARD:
      // Both left and right tracks drive in reverse
      digitalWrite(PIN_MOTOR_LEFT_IN1, LOW);
      digitalWrite(PIN_MOTOR_LEFT_IN2, HIGH);
      digitalWrite(PIN_MOTOR_RIGHT_IN1, LOW);
      digitalWrite(PIN_MOTOR_RIGHT_IN2, HIGH);
      break;

    case DIR_LEFT:
      // Skid-steer left: left reverse, right forward
      digitalWrite(PIN_MOTOR_LEFT_IN1, LOW);
      digitalWrite(PIN_MOTOR_LEFT_IN2, HIGH);
      digitalWrite(PIN_MOTOR_RIGHT_IN1, HIGH);
      digitalWrite(PIN_MOTOR_RIGHT_IN2, LOW);
      break;

    case DIR_RIGHT:
      // Skid-steer right: left forward, right reverse
      digitalWrite(PIN_MOTOR_LEFT_IN1, HIGH);
      digitalWrite(PIN_MOTOR_LEFT_IN2, LOW);
      digitalWrite(PIN_MOTOR_RIGHT_IN1, LOW);
      digitalWrite(PIN_MOTOR_RIGHT_IN2, HIGH);
      break;

    case DIR_STOPPED:
    default:
      digitalWrite(PIN_MOTOR_LEFT_IN1, LOW);
      digitalWrite(PIN_MOTOR_LEFT_IN2, LOW);
      digitalWrite(PIN_MOTOR_RIGHT_IN1, LOW);
      digitalWrite(PIN_MOTOR_RIGHT_IN2, LOW);
      break;
  }
}

void checkMotorSafetyWatchdog(unsigned long now) {
  if (rover.direction != DIR_STOPPED) {
    if (now - rover.last_command_time > ROVER_WATCHDOG_TIMEOUT_MS) {
      // Safety timeout tripped! Halt motors immediately
      executeMotorDrive(DIR_STOPPED, 0);
      rover.watchdog_active = false;
      Serial.println(F("[SAFETY WATCHDOG] Command timeout > 600ms. Rover motors halted safely."));
    }
  }
}

void setHeadlight(bool state) {
  rover.headlight = state;
  digitalWrite(PIN_HEADLIGHT, state ? HIGH : LOW);
  Serial.printf("[HEADLIGHT] Switched %s\n", state ? "ON" : "OFF");
}

// =========================================================================================
// SECTION 10: 4-DOF ROBOTIC ARM PCA9685 KINEMATICS ENGINE
// =========================================================================================

uint16_t angleToPulse(float angle) {
  angle = constrain(angle, 0.0f, 180.0f);
  return (uint16_t)map((long)(angle * 10), 0, 1800, SERVO_PULSE_MIN, SERVO_PULSE_MAX);
}

void setupRoboticArm() {
  Wire.begin(PIN_I2C_SDA, PIN_I2C_SCL, 400000);
  pca9685.begin();
  pca9685.setOscillatorFrequency(27000000);
  pca9685.setPWMFreq(SERVO_FREQUENCY_HZ);
  Wire.setClock(400000);

  // Probe PCA9685 address
  Wire.beginTransmission(PCA9685_I2C_ADDR);
  if (Wire.endTransmission() == 0) {
    arm.connected = true;
    Serial.println(F("[ARM] PCA9685 16-Channel PWM driver detected at 0x40."));
    // Move to safe home stance
    pca9685.setPWM(SERVO_CH_BASE, 0, angleToPulse(arm.base_deg));
    pca9685.setPWM(SERVO_CH_SHOULDER, 0, angleToPulse(arm.shoulder_deg));
    pca9685.setPWM(SERVO_CH_ELBOW, 0, angleToPulse(arm.elbow_deg));
    pca9685.setPWM(SERVO_CH_GRIPPER, 0, angleToPulse(arm.gripper_deg));
  } else {
    arm.connected = false;
    Serial.println(F("[ARM] WARNING: PCA9685 not responding on I2C address 0x40."));
  }
}

void setArmTargets(float base, float shoulder, float elbow, float gripper) {
  if (arm.emergency_stopped) {
    Serial.println(F("[ARM] Command rejected: Emergency Stop active."));
    return;
  }
  arm.target_base = constrain(base, ARM_BASE_MIN_DEG, ARM_BASE_MAX_DEG);
  arm.target_shoulder = constrain(shoulder, ARM_SHOULDER_MIN_DEG, ARM_SHOULDER_MAX_DEG);
  arm.target_elbow = constrain(elbow, ARM_ELBOW_MIN_DEG, ARM_ELBOW_MAX_DEG);
  arm.target_gripper = constrain(gripper, ARM_GRIPPER_CLOSE_DEG, ARM_GRIPPER_OPEN_DEG);
  arm.is_moving = true;
}

void updateArmKinematics(unsigned long now) {
  if (!arm.connected || !arm.is_moving || arm.emergency_stopped) return;

  if (now - arm.last_step_time >= ARM_MOTION_STEP_INTERVAL) {
    float dt = (now - arm.last_step_time) / 1000.0f;
    arm.last_step_time = now;

    float maxStep = arm.speed_deg_per_sec * dt;
    bool allReached = true;

    // Base joint
    float diffB = arm.target_base - arm.base_deg;
    if (abs(diffB) > 0.5f) {
      arm.base_deg += constrain(diffB, -maxStep, maxStep);
      pca9685.setPWM(SERVO_CH_BASE, 0, angleToPulse(arm.base_deg));
      allReached = false;
    }

    // Shoulder joint
    float diffS = arm.target_shoulder - arm.shoulder_deg;
    if (abs(diffS) > 0.5f) {
      arm.shoulder_deg += constrain(diffS, -maxStep, maxStep);
      pca9685.setPWM(SERVO_CH_SHOULDER, 0, angleToPulse(arm.shoulder_deg));
      allReached = false;
    }

    // Elbow joint
    float diffE = arm.target_elbow - arm.elbow_deg;
    if (abs(diffE) > 0.5f) {
      arm.elbow_deg += constrain(diffE, -maxStep, maxStep);
      pca9685.setPWM(SERVO_CH_ELBOW, 0, angleToPulse(arm.elbow_deg));
      allReached = false;
    }

    // Gripper end-effector
    float diffG = arm.target_gripper - arm.gripper_deg;
    if (abs(diffG) > 0.5f) {
      arm.gripper_deg += constrain(diffG, -maxStep, maxStep);
      pca9685.setPWM(SERVO_CH_GRIPPER, 0, angleToPulse(arm.gripper_deg));
      allReached = false;
    }

    if (allReached) {
      arm.is_moving = false;
    }
  }
}

void triggerArmEmergencyStop() {
  arm.emergency_stopped = true;
  arm.is_moving = false;
  arm.target_base = arm.base_deg;
  arm.target_shoulder = arm.shoulder_deg;
  arm.target_elbow = arm.elbow_deg;
  arm.target_gripper = arm.gripper_deg;
  Serial.println(F("[ARM EMERGENCY STOP] All servo actuation locked immediately!"));
}

void clearArmEmergencyStop() {
  arm.emergency_stopped = false;
  Serial.println(F("[ARM EMERGENCY STOP] Safety lock cleared. Arm ready."));
}

// =========================================================================================
// SECTION 11: SENSOR ACQUISITION (DHT11, SOIL V1.2, RAIN, WATER LEVEL)
// =========================================================================================

void setupSensors() {
  pinMode(PIN_DHT11_DATA, INPUT_PULLUP);
  pinMode(PIN_RAIN_DO, INPUT);
  analogReadResolution(12); // ESP32 12-bit ADC (0 to 4095)

  Serial.println(F("[SENSORS] Ground sensor suite initialized:"));
  Serial.println(F("  - DHT11 Digital on GPIO 4"));
  Serial.println(F("  - Capacitive Soil Moisture v1.2 on ADC1_CH6 (GPIO 34)"));
  Serial.println(F("  - Raindrop Module DO on GPIO 27, AO on GPIO 32"));
  Serial.println(F("  - Water Level Sensor on ADC1_CH7 (GPIO 35)"));
}

// Cooperative simple DHT11 reader
bool readDHT11NonBlocking(float &temp, float &hum) {
  // Simple bitbang timing protocol for standard DHT11
  uint8_t data[5] = {0, 0, 0, 0, 0};
  
  pinMode(PIN_DHT11_DATA, OUTPUT);
  digitalWrite(PIN_DHT11_DATA, LOW);
  delayMicroseconds(18000); // 18ms start signal
  digitalWrite(PIN_DHT11_DATA, HIGH);
  delayMicroseconds(30);
  pinMode(PIN_DHT11_DATA, INPUT_PULLUP);

  unsigned long timeout = micros();
  while (digitalRead(PIN_DHT11_DATA) == HIGH) {
    if (micros() - timeout > 100) return false;
  }
  timeout = micros();
  while (digitalRead(PIN_DHT11_DATA) == LOW) {
    if (micros() - timeout > 100) return false;
  }
  timeout = micros();
  while (digitalRead(PIN_DHT11_DATA) == HIGH) {
    if (micros() - timeout > 100) return false;
  }

  for (int i = 0; i < 40; i++) {
    timeout = micros();
    while (digitalRead(PIN_DHT11_DATA) == LOW) {
      if (micros() - timeout > 100) return false;
    }
    unsigned long pulseStart = micros();
    timeout = micros();
    while (digitalRead(PIN_DHT11_DATA) == HIGH) {
      if (micros() - timeout > 100) return false;
    }
    if ((micros() - pulseStart) > 40) {
      data[i / 8] |= (1 << (7 - (i % 8)));
    }
  }

  // Checksum
  if (data[4] == ((data[0] + data[1] + data[2] + data[3]) & 0xFF)) {
    hum = (float)data[0];
    temp = (float)data[2];
    return true;
  }
  return false;
}

void sampleSensors(unsigned long now) {
  // 1. Sample Analog Sensors (Soil Moisture & Water Level)
  static unsigned long lastAnalogSample = 0;
  if (now - lastAnalogSample >= ANALOG_SAMPLE_INTERVAL) {
    lastAnalogSample = now;

    // Soil Moisture (Capacitive v1.2: high ADC = dry, low ADC = wet)
    sensors.soil_raw = analogRead(PIN_SOIL_ADC);
    int clamped = constrain((int)sensors.soil_raw, (int)sensors.soil_wet_ref, (int)sensors.soil_dry_ref);
    sensors.soil_pct = (float)map(clamped, sensors.soil_dry_ref, sensors.soil_wet_ref, 0, 100);

    // Raindrop Sensor
    sensors.rain_detected = (digitalRead(PIN_RAIN_DO) == LOW);
    sensors.rain_intensity_raw = analogRead(PIN_RAIN_AO);
    // Lower raw reading on AO indicates higher droplet density
    sensors.rain_intensity_pct = (float)map(constrain((int)sensors.rain_intensity_raw, 500, 4095), 4095, 500, 0, 100);

    // Water Level Sensor (standing water depth)
    sensors.water_level_raw = analogRead(PIN_WATER_LEVEL_ADC);
    sensors.water_level_pct = (float)map(constrain((int)sensors.water_level_raw, 100, 3000), 100, 3000, 0, 100);

    if (sensors.water_level_pct >= 70.0f) {
      sensors.water_state_str = "WATERLOGGED";
    } else if (sensors.water_level_pct >= 35.0f) {
      sensors.water_state_str = "PONDING";
    } else {
      sensors.water_state_str = "NORMAL";
    }
  }

  // 2. Sample DHT11 Temperature & Humidity
  if (now - sensors.last_dht_read >= SENSORS_SAMPLE_INTERVAL) {
    sensors.last_dht_read = now;
    float t = 0, h = 0;
    if (readDHT11NonBlocking(t, h)) {
      sensors.temperature_c = t;
      sensors.humidity_pct = h;
      sensors.dht_valid = true;
    } else {
      // Retain last known value, flag stale
      sensors.dht_valid = false;
    }
  }
}

// =========================================================================================
// SECTION 12: GPS NEO-6M-0-001 NMEA STREAM PARSER
// =========================================================================================

void setupGPS() {
  GPSSerial.begin(GPS_BAUD_RATE, SERIAL_8N1, PIN_GPS_RX, PIN_GPS_TX);
  Serial.println(F("[GPS] NEO-6M UART2 initialized on RX=16, TX=17 @ 9600 baud."));
}

void parseNMEASentence(const String &sentence) {
  // Parse GPGGA sentence for Lat, Lon, Fix, Satellites, Altitude
  if (sentence.startsWith("$GPGGA") || sentence.startsWith("$GNGGA")) {
    int idx = 0;
    String tokens[15];
    int start = 0;
    for (int i = 0; i < sentence.length() && idx < 15; i++) {
      if (sentence.charAt(i) == ',' || sentence.charAt(i) == '*') {
        tokens[idx++] = sentence.substring(start, i);
        start = i + 1;
      }
    }
    if (idx >= 10) {
      int fixQuality = tokens[6].toInt();
      if (fixQuality > 0) {
        gps.fix_valid = true;
        gps.satellites = tokens[7].toInt();
        gps.hdop = tokens[8].toFloat();
        gps.altitude_m = tokens[9].toFloat();

        // Convert DDMM.MMMM to Decimal Degrees
        float rawLat = tokens[2].toFloat();
        float latDeg = (int)(rawLat / 100);
        float latMin = rawLat - (latDeg * 100);
        gps.latitude = latDeg + (latMin / 60.0f);
        if (tokens[3] == "S") gps.latitude = -gps.latitude;

        float rawLon = tokens[4].toFloat();
        float lonDeg = (int)(rawLon / 100);
        float lonMin = rawLon - (lonDeg * 100);
        gps.longitude = lonDeg + (lonMin / 60.0f);
        if (tokens[5] == "W") gps.longitude = -gps.longitude;

        gps.last_fix_time = millis();
      } else {
        gps.fix_valid = false;
      }
    }
  }
}

void updateGPSParser() {
  static String nmeaBuf = "";
  while (GPSSerial.available()) {
    char c = GPSSerial.read();
    if (c == '\n' || c == '\r') {
      if (nmeaBuf.length() > 6) {
        parseNMEASentence(nmeaBuf);
      }
      nmeaBuf = "";
    } else {
      if (nmeaBuf.length() < 120) {
        nmeaBuf += c;
      }
    }
  }
}

// =========================================================================================
// SECTION 13: BACKEND TELEMETRY SYNCHRONIZATION
// =========================================================================================

void syncTelemetryToBackend(unsigned long now) {
  if (WiFi.status() != WL_CONNECTED) return;
  if (now - lastTelemetrySyncTime < TELEMETRY_SYNC_INTERVAL) return;
  lastTelemetrySyncTime = now;

  WiFiClient client;
  HTTPClient http;

  if (http.begin(client, BACKEND_OBS_URL)) {
    http.addHeader("Content-Type", "application/json");

    // Build compact, canonical JSON observation payload matching backend schema
    String json = "{";
    json += "\"observation_id\":\"obs-" + String(millis()) + "\",";
    json += "\"device_id\":\"" + String(DEVICE_ID) + "\",";
    
    // Sensor Telemetry
    json += "\"sensor_telemetry\":{";
    json += "\"temperature_c\":" + String(sensors.temperature_c, 1) + ",";
    json += "\"humidity_pct\":" + String(sensors.humidity_pct, 1) + ",";
    json += "\"dht_status\":\"" + String(sensors.dht_valid ? "VALID" : "STALE") + "\",";
    json += "\"soil_moisture_raw\":" + String(sensors.soil_raw) + ",";
    json += "\"soil_moisture_pct\":" + String(sensors.soil_pct, 1) + ",";
    json += "\"soil_calibration_status\":\"CALIBRATED\",";
    json += "\"rain_detected\":" + String(sensors.rain_detected ? 1 : 0) + ",";
    json += "\"rain_intensity_raw\":" + String(sensors.rain_intensity_raw) + ",";
    json += "\"rain_intensity_pct\":" + String(sensors.rain_intensity_pct, 1) + ",";
    json += "\"water_level_raw\":" + String(sensors.water_level_raw) + ",";
    json += "\"water_level_pct\":" + String(sensors.water_level_pct, 1) + ",";
    json += "\"water_state\":\"" + String(sensors.water_state_str) + "\"";
    json += "},";

    // GPS Telemetry
    json += "\"gps\":{";
    json += "\"latitude\":" + String(gps.latitude, 6) + ",";
    json += "\"longitude\":" + String(gps.longitude, 6) + ",";
    json += "\"altitude_m\":" + String(gps.altitude_m, 1) + ",";
    json += "\"satellites_tracked\":" + String(gps.satellites) + ",";
    json += "\"hdop\":" + String(gps.hdop, 1) + ",";
    json += "\"fix_state\":\"" + String(gps.fix_valid ? "3D_FIX" : "NO_FIX") + "\",";
    json += "\"is_valid\":" + String(gps.fix_valid ? "true" : "false");
    json += "},";

    // Rover State
    const char* dirNames[] = {"STOPPED", "FORWARD", "BACKWARD", "LEFT", "RIGHT"};
    json += "\"rover_state\":{";
    json += "\"movement_state\":\"" + String(dirNames[rover.direction]) + "\",";
    json += "\"speed_pwm\":" + String(rover.speed_pwm) + ",";
    json += "\"headlight_on\":" + String(rover.headlight ? "true" : "false") + ",";
    json += "\"watchdog_active\":" + String(rover.watchdog_active ? "true" : "false") + ",";
    json += "\"uptime_seconds\":" + String(millis() / 1000) + ",";
    json += "\"wifi_rssi\":" + String(WiFi.RSSI());
    json += "},";

    // Arm State
    json += "\"arm_state\":{";
    json += "\"connected\":" + String(arm.connected ? "true" : "false") + ",";
    json += "\"is_moving\":" + String(arm.is_moving ? "true" : "false") + ",";
    json += "\"emergency_stopped\":" + String(arm.emergency_stopped ? "true" : "false") + ",";
    json += "\"base_deg\":" + String(arm.base_deg, 1) + ",";
    json += "\"shoulder_deg\":" + String(arm.shoulder_deg, 1) + ",";
    json += "\"elbow_deg\":" + String(arm.elbow_deg, 1) + ",";
    json += "\"gripper_deg\":" + String(arm.gripper_deg, 1) + ",";
    json += "\"gripper_state\":\"" + String(arm.gripper_deg >= 150.0f ? "OPEN" : "CLOSED") + "\"";
    json += "}";

    json += "}";

    int code = http.POST(json);
    if (code == 200) {
      String response = http.getString();
      // Update local Crop Intelligence display fields if returned by server
      if (response.indexOf("recommended_crop") != -1) {
        int cIdx = response.indexOf("\"recommended_crop\":\"");
        if (cIdx != -1) {
          int endC = response.indexOf("\"", cIdx + 20);
          String cName = response.substring(cIdx + 20, endC);
          cName.toCharArray(cropIntel.recommended_crop, sizeof(cropIntel.recommended_crop));
        }
      }
    }
    http.end();
  }
}

// =========================================================================================
// SECTION 14: LOCAL HTTP REST SERVER & TELEOPERATION HANDLERS
// =========================================================================================

void handleRoot() {
  String html = F("<!DOCTYPE html><html><head><title>AgriRover Controller</title>"
                  "<meta name='viewport' content='width=device-width,initial-scale=1'>"
                  "<style>body{font-family:sans-serif;background:#0d1117;color:#c9d1d9;text-align:center;padding:20px}"
                  "button{background:#238636;color:white;border:none;padding:12px 24px;margin:8px;border-radius:6px;font-size:16px}"
                  ".danger{background:#da3633}.card{background:#161b22;padding:16px;border-radius:8px;margin:12px auto;max-width:500px}"
                  "</style></head><body>"
                  "<h2>AgriRover Embedded Master</h2>"
                  "<div class='card'><h3>Chassis Teleoperation</h3>"
                  "<button onclick='fetch(\"/forward\")'>▲ Forward</button><br>"
                  "<button onclick='fetch(\"/left\")'>◄ Left</button>"
                  "<button onclick='fetch(\"/stop\")' class='danger'>■ STOP</button>"
                  "<button onclick='fetch(\"/right\")'>► Right</button><br>"
                  "<button onclick='fetch(\"/backward\")'>▼ Backward</button><br><br>"
                  "<button onclick='fetch(\"/led/toggle\")'>💡 Headlight Toggle</button>"
                  "</div><div class='card'><h3>4-DOF Robotic Arm</h3>"
                  "<button onclick='fetch(\"/arm/home\")'>Home (90°)</button>"
                  "<button onclick='fetch(\"/arm/open\")'>Open Claw</button>"
                  "<button onclick='fetch(\"/arm/close\")'>Close Claw</button>"
                  "<button onclick='fetch(\"/arm/stop\")' class='danger'>Emergency Stop</button>"
                  "</div><div class='card'><h3>Field Telemetry</h3>"
                  "<p><a href='/status' style='color:#58a6ff'>JSON Live Telemetry Snapshot</a></p>"
                  "</div></body></html>");
  server.send(200, "text/html", html);
}

void handleForward()  { executeMotorDrive(DIR_FORWARD, rover.speed_pwm); server.send(200, "text/plain", "OK"); }
void handleBackward() { executeMotorDrive(DIR_BACKWARD, rover.speed_pwm); server.send(200, "text/plain", "OK"); }
void handleLeft()     { executeMotorDrive(DIR_LEFT, rover.speed_pwm); server.send(200, "text/plain", "OK"); }
void handleRight()    { executeMotorDrive(DIR_RIGHT, rover.speed_pwm); server.send(200, "text/plain", "OK"); }
void handleStop()     { executeMotorDrive(DIR_STOPPED, 0); server.send(200, "text/plain", "OK"); }

void handleHeadlightOn()  { setHeadlight(true); server.send(200, "text/plain", "ON"); }
void handleHeadlightOff() { setHeadlight(false); server.send(200, "text/plain", "OFF"); }
void handleHeadlightToggle() { setHeadlight(!rover.headlight); server.send(200, "text/plain", rover.headlight ? "ON" : "OFF"); }

void handleArmHome()  { setArmTargets(ARM_HOME_BASE, ARM_HOME_SHOULDER, ARM_HOME_ELBOW, ARM_HOME_GRIPPER); server.send(200, "text/plain", "OK"); }
void handleArmOpen()  { setArmTargets(arm.base_deg, arm.shoulder_deg, arm.elbow_deg, ARM_GRIPPER_OPEN_DEG); server.send(200, "text/plain", "OK"); }
void handleArmClose() { setArmTargets(arm.base_deg, arm.shoulder_deg, arm.elbow_deg, ARM_GRIPPER_CLOSE_DEG); server.send(200, "text/plain", "OK"); }
void handleArmStop()  { triggerArmEmergencyStop(); server.send(200, "text/plain", "EMERGENCY_STOP"); }
void handleArmClear() { clearArmEmergencyStop(); server.send(200, "text/plain", "CLEARED"); }

void handleStatus() {
  String json = "{";
  json += "\"device_id\":\"" + String(DEVICE_ID) + "\",";
  json += "\"firmware\":\"" + String(FIRMWARE_VERSION) + "\",";
  json += "\"uptime_s\":" + String(millis() / 1000) + ",";
  json += "\"wifi_rssi\":" + String(WiFi.RSSI()) + ",";
  json += "\"temp_c\":" + String(sensors.temperature_c, 1) + ",";
  json += "\"humidity_pct\":" + String(sensors.humidity_pct, 1) + ",";
  json += "\"soil_moisture_pct\":" + String(sensors.soil_pct, 1) + ",";
  json += "\"rain_detected\":" + String(sensors.rain_detected ? "true" : "false") + ",";
  json += "\"water_level_pct\":" + String(sensors.water_level_pct, 1) + ",";
  json += "\"gps_valid\":" + String(gps.fix_valid ? "true" : "false") + ",";
  json += "\"lat\":" + String(gps.latitude, 6) + ",";
  json += "\"lon\":" + String(gps.longitude, 6) + ",";
  json += "\"arm_base\":" + String(arm.base_deg, 1) + ",";
  json += "\"arm_shoulder\":" + String(arm.shoulder_deg, 1) + ",";
  json += "\"arm_elbow\":" + String(arm.elbow_deg, 1) + ",";
  json += "\"arm_gripper\":" + String(arm.gripper_deg, 1);
  json += "}";
  server.send(200, "application/json", json);
}

void setupWebServer() {
  server.on("/", HTTP_GET, handleRoot);
  server.on("/forward", HTTP_GET, handleForward);
  server.on("/backward", HTTP_GET, handleBackward);
  server.on("/left", HTTP_GET, handleLeft);
  server.on("/right", HTTP_GET, handleRight);
  server.on("/stop", HTTP_GET, handleStop);

  server.on("/led/on", HTTP_GET, handleHeadlightOn);
  server.on("/led/off", HTTP_GET, handleHeadlightOff);
  server.on("/led/toggle", HTTP_GET, handleHeadlightToggle);

  server.on("/arm/home", HTTP_GET, handleArmHome);
  server.on("/arm/reset", HTTP_GET, handleArmHome);
  server.on("/arm/open", HTTP_GET, handleArmOpen);
  server.on("/arm/close", HTTP_GET, handleArmClose);
  server.on("/arm/stop", HTTP_GET, handleArmStop);
  server.on("/arm/clear", HTTP_GET, handleArmClear);

  server.on("/status", HTTP_GET, handleStatus);
  server.begin();
  Serial.println(F("[HTTP] AgriRover REST WebServer started on port 80."));
}

// =========================================================================================
// SECTION 15: WIRELESS NETWORKING INITIALIZATION
// =========================================================================================

void setupWiFi() {
  WiFi.mode(WIFI_AP_STA);
  Serial.printf("[WIFI] Connecting to Station SSID '%s'...\n", WIFI_SSID_PRIMARY);
  WiFi.begin(WIFI_SSID_PRIMARY, WIFI_PASS_PRIMARY);

  unsigned long start = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - start < 8000) {
    delay(200);
    Serial.print('.');
  }

  if (WiFi.status() == WL_CONNECTED) {
    Serial.printf("\n[WIFI] Connected! Assigned STA IP: %s\n", WiFi.localIP().toString().c_str());
  } else {
    Serial.println(F("\n[WIFI] Station connection timed out. Starting Fallback SoftAP."));
    WiFi.softAPConfig(AP_LOCAL_IP, AP_GATEWAY, AP_SUBNET);
    WiFi.softAP(AP_SSID, AP_PASS);
    Serial.printf("[WIFI] Fallback AP Started: '%s' (IP: %s)\n", AP_SSID, AP_LOCAL_IP.toString().c_str());
  }

  if (MDNS.begin("agrirover")) {
    Serial.println(F("[mDNS] Responding at http://agrirover.local"));
    MDNS.addService("http", "tcp", 80);
  }
}

// =========================================================================================
// SECTION 16: ARDUINO MAIN ENTRY POINTS (SETUP & COOPERATIVE LOOP)
// =========================================================================================

void setup() {
  Serial.begin(115200);
  delay(500);

  Serial.println(F("\n══════════════════════════════════════════════════════════════"));
  Serial.println(F("    AgriRover Master Controller Booting (Team Innovex)        "));
  Serial.println(F("══════════════════════════════════════════════════════════════"));

  setupMotors();
  setupRoboticArm();
  setupSensors();
  setupGPS();
  tftManager.init();
  setupWiFi();
  setupWebServer();

  Serial.println(F("[BOOT] All hardware subsystems ready. Entering cooperative loop."));
}

void loop() {
  unsigned long now = millis();

  // 1. Core safety watchdog: Halt motors if commands expire > 600ms
  checkMotorSafetyWatchdog(now);

  // 2. Continuous non-blocking GPS NMEA stream parsing
  updateGPSParser();

  // 3. Smooth robotic arm joint kinematics
  updateArmKinematics(now);

  // 4. Ground sensor sampling (DHT11, Soil Moisture, Rain, Water Level)
  sampleSensors(now);

  // 5. 2.4-inch TFT LCD Multi-screen graphical display updates
  tftManager.updateDisplay(now);

  // 6. Handle inbound browser teleoperation requests
  server.handleClient();

  // 7. Synchronize telemetry with SmartCropVision backend
  syncTelemetryToBackend(now);
}
