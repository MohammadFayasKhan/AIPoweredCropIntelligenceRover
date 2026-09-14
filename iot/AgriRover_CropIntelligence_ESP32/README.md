# 🌾 SmartCropVision & AgriRover — Crop Intelligence Sensor Node
## 16x2 Character LCD Edition (I2C / PCF8574 Backpack)

**Firmware Version:** `v3.0.0-PROD`  
**Target Hardware:** ESP32-WROOM-32 (30-Pin)  
**Display:** 16x2 Character LCD with PCF8574 I2C Backpack (Addresses `0x27` / `0x3F`)  
**SIH Challenge:** SIH26180 (Qualcomm Inc. / Agriculture)  
**Author:** Team Innovex  

---

### 1. Hardware Overview & Wiring Table

| Peripheral | Pin Name | ESP32 GPIO | Voltage Rail | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **16x2 I2C LCD** | VCC | **VIN (5V)** | 5V | Essential for HD44780 contrast |
| | GND | **GND** | Ground | Common system ground |
| | SDA | **GPIO 21** | 3.3V | I2C Wire Data Bus (Internal pull-up) |
| | SCL | **GPIO 22** | 3.3V | I2C Wire Clock Bus (Internal pull-up) |
| **DHT11** | DATA | **GPIO 4** | 3.3V | Air temperature & humidity |
| **Soil Moisture** | AOUT | **GPIO 34** | 3.3V | ADC1_CH6 (Capacitive v1.2, WiFi safe) |
| **Water Level** | SIG | **GPIO 35** | 3.3V | ADC1_CH7 (Drainage / furrow depth) |
| **Rain Sensor** | AO (Analog) | **GPIO 36** | 3.3V | ADC1_CH0 (VP, precipitation intensity) |
| | DO (Digital)| **GPIO 39** | 3.3V | ADC1_CH3 (VN, active LOW rain flag) |
| **GPS NEO-6M** | TXD | **GPIO 16** | 3.3V / 5V | HardwareSerial2 RX2 |
| | RXD | **GPIO 17** | 3.3V / 5V | HardwareSerial2 TX2 |

*Note: All legacy 2.4" 8-bit parallel TFT pins (`LCD_D0..D7`, `WR`, `RD`, `RS`, `CS`, `RST`) and drivers have been completely removed.*

---

### 2. Startup I2C Auto-Detection & Fallback Policy

1. **Auto-Detection:** At boot, the firmware probes `0x27` and `0x3F` (the standard PCF8574 backpack addresses), followed by a bus scan if needed.
2. **Headless Fallback:** If no I2C LCD is detected, the firmware reports `[LCD] Not detected (Headless operational mode)` over Serial and continues non-blocking sensor acquisition, GPS processing, Wi-Fi connectivity, and backend telemetry synchronization. The display never blocks the node.

---

### 3. Custom LCD Glyphs

The firmware creates 4 custom 5x8 characters:
- `CHAR_DEG` (0): Degree symbol `°` for temperature display
- `CHAR_TICK` (1): Checkmark `✓` for verified states and recommendations
- `CHAR_ALERT` (2): Warning exclamation icon `!` for pathogen/environmental stress
- `CHAR_DROP` (3): Water droplet icon for active precipitation

---

### 4. Non-Blocking Screen Carousel Sequence (~4s per screen)

- **Screen 0 — Canopy Atmosphere & Root Moisture:**
  ```text
  Row 0: T:28.4°C H:72%
  Row 1: Soil:45% R:WET
  ```
- **Screen 1 — Authoritative Crop Recommendation:**
  ```text
  Row 0: [tick] RICE
  Row 1: Conf:94.0% [tick]OK
  ```
  *(If backend is unreachable: `[tick] NO CROP` / `AI: OFFLINE`)*
- **Screen 2 — Disease & Pathogen Microclimate Alerts:**
  ```text
  Row 0: [alert]Anthracnose
  Row 1: HIGH        1/2
  ```
  *(If no disease risk: `[tick] ALL CLEAR` / `No disease risk`)*
- **Screen 3 — Field Hydrology & Irrigation Advisory:**
  ```text
  Row 0: Irrig: NOMINAL
  Row 1: Soil:45% W:0mm
  ```
- **Screen 4 — GPS & Connectivity Diagnostics:**
  ```text
  Row 0: GPS: FIX 10 SAT
  Row 1: WiFi:OK API:OK
  ```
- **Screen 5 — Production Node & Observation Identification:**
  ```text
  Row 0: SCV NODE v3.0
  Row 1: Obs: #OBS-12345
  ```

---

### 5. Backend Deployment Targets & Auto-Failover Modes

- **Mode 0 (`TARGET_MODE 0` — Public Production Cloud [Default / Primary]):**
  - **Endpoint:** `https://aipoweredcropintelligencerover.dpdns.org/predict/compact`
  - **Transport:** HTTPS with DigiCert Global Root G2 certificate validation (with resilient fallback to TLS insecure if system clock unset)
  - **Use Case:** Direct outdoor/field operations connecting straight to the live public web dashboard.

- **Mode 1 (`TARGET_MODE 1` — Local Mac Development):**
  - **Endpoint:** `http://172.20.10.4:8000/predict/compact`
  - **Transport:** Fast plaintext HTTP
  - **Subnet Auto-Discovery:** Automatically scans `172.20.10.x` hotspot subnet to detect the Mac's IP dynamically if it changes.
  - **Use Case:** Local paired development and rapid debugging on Mac.

- **Mode 2 (`TARGET_MODE 2` — Smart Dual-Mode Auto-Failover):**
  - Routes telemetry to the **Public Production Cloud (`https://aipoweredcropintelligencerover.dpdns.org`) as PRIMARY**.
  - If the public cloud is ever unreachable (e.g., loss of cellular internet), it automatically fails over to the Local Mac dev server (`http://172.20.10.4:8000`).
  - Guarantees zero telemetry loss across network transitions.

- **Method:** `POST`
- **Content-Type:** `application/json`
- **Response Format:** Compact character-safe JSON (`ok`, `crop`, `conf`, `ac`, `t2`, `c2`, `t3`, `c3`, `alerts`) parsed without buffer overflow on 16x2 LCD.
