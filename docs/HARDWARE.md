# Hardware & Embedded Systems Architecture
## AI Powered Crop Intelligence Rover

```
SIH Challenge: SIH26180 | Qualcomm Inc.
Category: Hardware | Theme: Disaster Management
Team: Innovex (SIH-H-B2-048)
```

---

## 1. Physical Hardware Overview & Separation of Concerns

The **AI Powered Crop Intelligence Rover** decouples locomotion and mechanical manipulation from environmental sensing and cloud synchronization. This architectural boundary prevents inductive motor spikes and servo PWM switching from corrupting analog sensor readings or stalling the primary telemetry loops.

```
┌────────────────────────────────────────────────────────────────────────┐
│                        AGRIROVER PHYSICAL PLATFORM                     │
├──────────────────────────────────┬─────────────────────────────────────┤
│   SUBSYSTEM A: MOBILITY & ARM    │   SUBSYSTEM B: CROP SENSOR NODE     │
│   Controller: ESP32 DevKit       │   Controller: ESP32-WROOM-32        │
│   ├─ L293D Dual H-Bridge Motor   │   ├─ 16x2 I2C LCD (PCF8574)         │
│   │  Driver (4WD Skid Steer)     │   ├─ DHT11 (Air Temp & Humidity)    │
│   ├─ PCA9685 16-Ch 12-Bit PWM    │   ├─ Capacitive Soil Moisture v1.2  │
│   │  Servo Driver (I2C 0x40)     │   ├─ Optical Rain Drop Sensor       │
│   ├─ 4-DOF Robotic Arm Joints    │   ├─ Resistive Water Level Sensor   │
│   └─ Web Teleoperation Console   │   ├─ NEO-6M GPS Module (UART2)      │
│      (http://agrirover.local)    │   └─ Dual-Failover Cloud Gateway    │
└──────────────────────────────────┴─────────────────────────────────────┘
```

---

## 2. Subsystem A: Mobility & 4-DOF Robotic Arm Controller

The mobility and mechanical arm controller runs on an ESP32 DevKit, hosting a standalone Wi-Fi teleoperation console with hardware interlocking and safety watchdogs.

### 2.1 ESP32 to L293D H-Bridge Motor Pinout (Chassis Drive)

The rover chassis utilizes a 4-wheel drive skid-steer configuration driven by an L293D dual H-bridge motor driver. Left-side motors (front and rear) and right-side motors (front and rear) are wired in parallel pairs.

| ESP32 GPIO | L293D Terminal | Signal / Function | Drive Group |
| :--- | :--- | :--- | :--- |
| **GPIO 5** | M1 IN1 | Digital OUT | Left Motors Forward |
| **GPIO 18** | M1 IN2 | Digital OUT | Left Motors Reverse |
| **GPIO 19** | M2 IN3 | Digital OUT | Right Motors Forward |
| **GPIO 21** | M2 IN4 | Digital OUT | Right Motors Reverse |
| **GPIO 2** | Anode | Digital OUT | Rover Field Headlight / Illumination LED |
| **GND** | GND | Power | Common System Ground |

*Note: L293D enable jumpers (`EN1`, `EN2`) are tied to 5V. A software movement watchdog halts motors if no heartbeat is received within 600 ms.*

### 2.2 ESP32 to PCA9685 PWM Servo Driver (4-DOF Arm)

The 4-DOF robotic arm is driven by an Adafruit PCA9685 16-channel 12-bit PWM controller communicating over hardware I2C at address `0x40`.

| ESP32 Pin | PCA9685 Header | Wire Bus | Voltage Level |
| :--- | :--- | :--- | :--- |
| **GPIO 23** | **SDA** | I2C Data | 3.3V Logic |
| **GPIO 22** | **SCL** | I2C Clock | 3.3V Logic |
| **3.3V** | **VCC** | Logic Supply | 3.3V |
| **GND** | **GND** | Common Ground | Ground |

> **Power Supply Rule:** Servos are strictly powered through the external terminal block on the PCA9685 using a dedicated **5V to 6V DC power supply (3A to 5A capacity)**. Servos are never driven from the ESP32 onboard regulator rails. External power supply ground is tied to ESP32 ground.

### 2.3 Servo Channel Allocation & Angular Limits

| PCA9685 Channel | Joint Identification | Physical Bounds | Safe Park Angle | Angular Velocity | Function |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **CH 0** | **Base Turntable** | 5° to 175° | **90°** | 360°/s | Azimuth alignment toward crop canopy |
| **CH 1** | **Shoulder Pitch** | 0° to 180° | **90°** | 300°/s | Vertical elevation lift arm |
| **CH 2** | **Elbow Extension** | 0° to 180° | **90°** | 340°/s | Forearm depth positioning into crop rows |
| **CH 3** | **End-Effector Gripper** | 70° to 180° | **180°** (Open) | 500°/s | Foliar leaf clamp (70° = Full grip, 180° = Open) |

---

## 3. Subsystem B: Crop Intelligence Sensor Node

The environmental telemetry node operates autonomously on a dedicated ESP32-WROOM-32 microcontroller, reading multivariable soil, precipitation, atmospheric, and positioning sensors before transmitting authenticated observations to the cloud backend.

### 3.1 Sensor Node Pinout & Interface Mapping

| Peripheral Component | Pin Name | ESP32 GPIO | Channel / Bus | Voltage Rail | Sensor Semantics |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **16x2 Character LCD** | SDA | **GPIO 21** | Wire I2C Data | 5V (Backpack VIN) | HD44780 controller via PCF8574 backpack |
| | SCL | **GPIO 22** | Wire I2C Clock | 5V (Backpack VIN) | Internal I2C pull-ups enabled |
| **DHT11 Microclimate** | DATA | **GPIO 4** | 1-Wire Digital | 3.3V | Ambient temperature (±2°C) & humidity (±5% RH) |
| **Capacitive Soil v1.2**| AOUT | **GPIO 34** | ADC1_CH6 | 3.3V | Volumetric root moisture (analog calibration) |
| **Water Level Sensor** | SIG | **GPIO 35** | ADC1_CH7 | 3.3V | Furrow water pooling / drainage depth |
| **Rain Drop Sensor** | AO (Analog) | **GPIO 36** | ADC1_CH0 (VP) | 3.3V | Precipitation intensity (raw ADC counts) |
| | DO (Digital) | **GPIO 39** | ADC1_CH3 (VN) | 3.3V | Binary rain threshold flag (active LOW) |
| **NEO-6M GPS Module** | TXD | **GPIO 16** | UART2 RX2 | 3.3V / 5V | NMEA sentence stream at 9600 baud |
| | RXD | **GPIO 17** | UART2 TX2 | 3.3V / 5V | Configuration commands |

> **ADC Channel Selection Note:** All analog sensors are mapped strictly to **ADC1** (`GPIO 34, 35, 36, 39`). ADC2 pins are avoided because the ESP32 Wi-Fi driver claims ADC2 during active network communication.

---

## 4. 16x2 I2C Display System & Glyph Engine

The Crop Intelligence node uses a 16-character by 2-line alphanumeric LCD with a PCF8574 I2C expander. All legacy 8-bit parallel TFT display code has been eliminated.

### 4.1 Bus Auto-Discovery & Headless Recovery

1. **Auto-Detection:** At boot, firmware probes `0x27` and `0x3F` (standard PCF8574 addresses) followed by a full I2C bus scan if required.
2. **Headless Fallback:** If the display is disconnected or absent, the firmware enters headless mode (`[LCD] Not detected (Headless operational mode)`) and continues non-blocking sensor acquisition, GPS processing, and cloud transmission without stalling.

### 4.2 Custom 5x8 Alphanumeric Glyphs

The firmware registers 4 custom pixel bitmaps into the LCD CGRAM:

* `CHAR_DEG` (`\x00`): Degree symbol `°` for temperature readouts.
* `CHAR_TICK` (`\x01`): Checkmark `✓` confirming verified state and model health.
* `CHAR_ALERT` (`\x02`): Exclamation icon `!` signalling disease favorability or water stress.
* `CHAR_DROP` (`\x03`): Droplet icon representing active rainfall.

### 4.3 Alphanumeric Screen Carousel Sequence

The display iterates through a non-blocking carousel (~4 seconds per screen):

```text
Screen 0: Canopy Atmosphere & Root Moisture:
┌────────────────┐
│T:28.4°C  H:72% │
│Soil:45%  R:WET │
└────────────────┘

Screen 1: AI Crop Recommendation:
┌────────────────┐
│✓ RICE          │
│Conf:94.0%  ✓OK │
└────────────────┘

Screen 2: Disease Pathogen Microclimate Alerts:
┌────────────────┐
│!Anthracnose    │
│HIGH         1/2│
└────────────────┘

Screen 3: Field Hydrology & Irrigation Advisory:
┌────────────────┐
│Irrig: NOMINAL  │
│Soil:45%   W:0mm│
└────────────────┘

Screen 4: GPS & Connectivity Diagnostics:
┌────────────────┐
│GPS: FIX 10 SAT │
│WiFi:OK   API:OK│
└────────────────┘

Screen 5: Production Node & Observation ID:
┌────────────────┐
│SCV NODE v3.0   │
│Obs: #OBS-10482 │
└────────────────┘
```

---

## 5. Cloud Telemetry & Dual-Failover Routing

The sensor node firmware embeds dual-mode failover logic for network resilience during field trials:

```
[Sensors + GPS]
       │
       ▼
[ESP32 Aggregator] ──HTTPS POST──► [PRIMARY: https://aipoweredcropintelligencerover.dpdns.org]
       │                                     │
       │ (On 3 consecutive network timeouts) │
       ▼                                     ▼
[FALLBACK: Local Subnet Gateway]    [FastAPI Processing Engine]
```

1. **Primary Target (Public Cloud):**
   * Endpoint: `https://aipoweredcropintelligencerover.dpdns.org/predict/compact`
   * Transport: TLS 1.3 / HTTPS with DigiCert Global Root G2 certificate verification.
   * Transmits JSON payload with authenticated observation tokens every 10 seconds.
2. **Fallback Target (Local Development Gateway):**
   * Automatically falls back to local paired subnet (`http://172.20.10.x:8000/predict/compact`) if public cloud DNS or route fails during offline field testing.
3. **Observation Persistence:**
   * Each telemetry packet contains an incremental observation counter, hardware uptime, and authentic GPS coordinates (latitude, longitude, altitude, satellite count, HDOP). If GPS fix is unavailable, fields report `stale` or `searching` rather than synthetic coordinates.

---

## 6. Physical Specifications & Power Budget

| Subsystem Component | Supply Rail | Peak Current | Average Current | Power Source |
| :--- | :--- | :--- | :--- | :--- |
| **ESP32 Mobility + Arm** | 5V regulated | 240 mA | 120 mA | Dual 18650 Li-ion (7.4V) via Buck Converter |
| **L293D + 4 DC Gear Motors** | 7.4V raw | 1.8 A (Stall) | 650 mA | Dual 18650 Li-ion battery pack |
| **PCA9685 + 4 Servos** | 5.5V regulated | 3.2 A (Under load) | 800 mA | High-discharge LiPo battery pack |
| **ESP32 Sensor Node** | 5V regulated | 260 mA (Wi-Fi TX) | 95 mA | 5000 mAh Li-ion power bank |
| **16x2 LCD + Backlight** | 5V | 40 mA | 25 mA | From sensor node 5V rail |
| **NEO-6M GPS** | 3.3V | 67 mA | 45 mA | From ESP32 3.3V rail |
| **DHT11 + Moisture + Rain** | 3.3V | 15 mA | 8 mA | From ESP32 3.3V rail |
