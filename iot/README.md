# 🚜 AgriRover + 🦾 4-DOF Robotic Arm Master Controller (v2.0.0)

**Team:** Innovex  
**SIH Challenge ID:** SH26180  
**Organization:** Qualcomm Inc  
**Category:** Hardware | **Theme:** Disaster Management  
**Target Platform:** ESP32 DevKit (ESP32-WROOM-32)  

---

## 📌 System Architecture

AgriRover v2.0 integrates the **4-Wheel Drive Rover Chassis** and the **4-DOF Robotic Arm** onto a single unified **ESP32 DevKit** controller.

### 🔄 What Changed from the Legacy System?
- ❌ **Eliminated:** Arduino Nano micro-controller.
- ❌ **Eliminated:** HC-05 / HC-06 Bluetooth module and unreliable `SoftwareSerial` (9600 baud).
- ✅ **Integrated:** Adafruit PCA9685 16-Channel 12-bit PWM Servo Driver directly over **I2C** (`0x40`).
- ✅ **Unified Control:** High-speed Wi-Fi Web Teleoperation console (`http://agrirover.local`) serving dual controls:
  1. 🚜 **Chassis D-Pad Drive** with automatic movement safety watchdog.
  2. 🦾 **4-DOF Robotic Arm** with interactive joint sliders, micro-steppers, gripper controls, velocity scaling, agronomic inspection presets, and teach-and-repeat waypoint sequencing.
- ✅ **Hardware Interlocking:** Zero pin conflicts between L293D H-bridge DC motors and PCA9685 I2C lines.

---

## ⚡ Hardware Pinout & Wiring Specification

### 1. ESP32 to L293D Dual H-Bridge Motor Driver (Chassis Drive)
| ESP32 GPIO | L293D Pin | Function / Drive Group |
|:---:|:---:|:---|
| **GPIO 5** | M1 IN1 | Left Motors Forward (Front & Rear Left in parallel) |
| **GPIO 18** | M1 IN2 | Left Motors Reverse |
| **GPIO 19** | M2 IN3 | Right Motors Forward (Front & Rear Right in parallel) |
| **GPIO 21** | M2 IN4 | Right Motors Reverse |
| **GPIO 2** | LED Anode | Rover Field Headlight / Visual Telemetry LED |
| **GND** | GND | Common System Ground |

> **Note:** L293D enable pins `EN1` and `EN2` should be connected to 5V (using driver board jumper caps).

---

### 2. ESP32 to PCA9685 16-Channel PWM Servo Driver (Robotic Arm)
| ESP32 Pin | PCA9685 Pin | Description |
|:---:|:---:|:---|
| **GPIO 23** | **SDA** | I2C Serial Data (Hardware Wire Bus) |
| **GPIO 22** | **SCL** | I2C Serial Clock (Hardware Wire Bus) |
| **3.3V or 5V** | **VCC** | PCA9685 Logic Power |
| **GND** | **GND** | Common System Ground |

> ⚠️ **CRITICAL POWER SUPPLY RULE (Brownout Prevention):**  
> **DO NOT** power the servos from the ESP32 3.3V or 5V rail!  
> Connect a dedicated **external 5V–6V DC power supply (minimum 3A to 5A capacity)** to the **PCA9685 screw terminal (V+ and GND)**.  
> Ensure the **GND** of the external servo power supply is connected to the **ESP32 GND** (Common Ground).

---

### 3. PCA9685 Servo Channel Allocation & Physical Motion Limits
| Channel | Joint Name | Physical Limits | Safe Home | Default Speed | Description |
|:---:|:---|:---:|:---:|:---:|:---|
| **CH 0** | **Base Rotation** | 5° – 175° | **90°** | 360°/s | Azimuth rotation for crop inspection alignment |
| **CH 1** | **Shoulder Pitch** | 0° – 180° | **90°** | 300°/s | Elevation lift arm |
| **CH 2** | **Elbow Extension** | 0° – 180° | **90°** | 340°/s | Forearm reach & canopy depth positioning |
| **CH 3** | **End-Effector Gripper** | 70° – 180° | **180°** (Open) | 500°/s | Sample collector / foliar leaf clamp (70° = Full Grip, 180° = Wide Open) |

---

## 🌐 Network & Access

1. **Station Mode (Default):**
   - Connects to Wi-Fi SSID: `Fayas`
   - Access URL: `http://agrirover.local` or direct ESP32 IP reported in Serial Monitor.
2. **Autonomous Fallback Access Point (AP Mode):**
   - If Wi-Fi is unreachable after 20 attempts, the rover automatically spawns its own network:
   - **SSID:** `AgriRover`
   - **Password:** `12345678`
   - **Gateway / Web Dashboard IP:** `http://192.168.9.1`

---

## 🎛️ Web Teleoperation Console Features

Accessing `http://agrirover.local` loads the unified mobile-optimized Web App:

- 🚜 **Chassis Tab:**
  - High-response virtual D-Pad with multi-touch pointer capture.
  - Keyboard shortcuts (`W`, `A`, `S`, `D`, `Space` for emergency stop).
  - Headlight toggle switch with live illumination glow.
  - Heartbeat watchdog timer (stops motors if connection drops > 600ms).
- 🦾 **Robotic Arm Tab (MIT App Inventor Layout & Graphical SVG Kinematics):**
  - **Complete 4-DOF Robotic Arm Vector Illustration:**
    - Left-hand canvas displaying the full robotic arm (gripper claw, forearm, bicep, and base turntable).
    - Alignment callout pointer lines extending directly from each joint to its respective slider on the right.
    - Live SVG kinematics: the claw physically closes and opens as the user moves the Gripper slider.
  - **4-DOF Joint Controls with Value Display Boxes:**
    - **Grip (`Servo_06`):** Rectangular value readout box, range slider (70°–180°), micro-steps `[-5°]`, `[+5°]`.
    - **Elbow (`Servo_03`):** Rectangular value readout box, range slider (0°–180°), micro-steps `[-5°]`, `[+5°]`.
    - **Shoulder (`Servo_02`):** Rectangular value readout box, range slider (0°–180°), micro-steps `[-5°]`, `[+5°]`.
    - **Base (`Servo_01`):** Rectangular value readout box, range slider (5°–175°), micro-steps `[-5°]`, `[+5°]`.
  - **Arm Velocity Speed Control (`ss`):**
    - Slider (5% to 100%) with live percentage readout.
  - **MIT App Inventor Primary Action Blocks:**
    - `[💾 SAVE]`: Sends `SAVE`, records pose to waypoint buffer, increments `Positions: X` count.
    - `[▶️ RUN / ⏸️ PAUSE]`: Starts green with label `RUN`. When clicked, sends `RUN`, changes label to `PAUSE` and background to Red. When clicked again, sends `PAUSE`, reverts to `RUN` and Green.
    - `[🔄 RESET]`: Sends `RESET`, returns servos to Safe Home `[90, 90, 90, 180]`, resets `RUN` button to green, resets positions to 0.
    - `[🗑️ CLEAR]`: Sends `CLEAR`, flushes stored waypoints.
    - **Stored Waypoints Counter:** Displays `Positions: X / 20`.
  - **Quick Agricultural Presets:**
    - 🏠 **Safe Home:** `[90°, 90°, 90°, 180°]` (Neutral stowed stance).
    - 🔍 **Canopy High:** `[90°, 130°, 65°, 180°]` (Elevated angle for upper foliage).
    - 🌿 **Foliar Mid:** `[90°, 100°, 115°, 180°]` (Horizontal reach for mid-tier leaves).
    - 🧪 **Ground Sample:** `[90°, 45°, 140°, 70°]` (Low-elevation grasping stance).
    - ✋ **Open Grip** (180°) and ✊ **Close Grip** (70°).
- ⚡ **Dual Console Tab:**
  - Side-by-side view for tablets or desktop monitors allowing simultaneous driving and manipulator guidance.
- 📜 **Unified Activity Log:**
  - Real-time timestamped event log tracking both rover drive commands and arm movements.

---

## 📡 REST API Reference

All endpoints return HTTP 200 with standard headers (`Cache-Control: no-cache`).

### Arm Control Endpoints
| Endpoint | Parameters | Example | Description |
|:---|:---|:---|:---|
| `/arm/set` | `base`, `shoulder`, `elbow`, `gripper` | `/arm/set?base=90&shoulder=120` | Sets target angles for specified joints |
| `/arm/servo` | `joint` (1, 2, 3, 6), `angle` | `/arm/servo?joint=1&angle=45` | Sets single joint target angle |
| `/arm/home` | *none* | `/arm/home` | Returns arm to safe default home stance |
| `/arm/open` | *none* | `/arm/open` | Opens gripper to 180° |
| `/arm/close` | *none* | `/arm/close` | Closes gripper to 70° |
| `/arm/speed` | `val` (5–100) | `/arm/speed?val=85` | Sets dynamic velocity ratio |
| `/arm/preset` | `name` | `/arm/preset?name=canopy_high` | Activates named agricultural preset |
| `/arm/save` | *none* | `/arm/save` | Saves current arm pose to waypoint buffer |
| `/arm/run` | *none* | `/arm/run` | Starts autonomous waypoint sequence |
| `/arm/pause` | *none* | `/arm/pause` | Pauses sequence playback |
| `/arm/resume` | *none* | `/arm/resume` | Resumes paused sequence |
| `/arm/stop` | *none* | `/arm/stop` | Halts sequence playback |
| `/arm/clear` | *none* | `/arm/clear` | Clears all stored waypoints |
| `/arm/status` | *none* | `/arm/status` | Returns detailed JSON arm telemetry |
| `/cmd` or `/arm/cmd` | `c` or `cmd` | `/cmd?c=s190` or `/cmd?c=SAVE` | Executes raw legacy MIT App Inventor command |

### Unified System Status
- **`GET /status`** returns:
```json
{
  "command": "STOPPED",
  "led": false,
  "rssi": -48,
  "uptime": 142,
  "ip": "192.168.1.105",
  "clients": 0,
  "commands": 18,
  "arm": {
    "base": 90.0,
    "shoulder": 90.0,
    "elbow": 90.0,
    "gripper": 180.0,
    "target": {
      "base": 90.0,
      "shoulder": 90.0,
      "elbow": 90.0,
      "gripper": 180.0
    },
    "speed": 70,
    "is_moving": false,
    "seq_running": false,
    "seq_paused": false,
    "seq_step": 0,
    "saved_count": 3
  }
}
```

---

## 🧭 Complete End-to-End System Architecture

### 1. Inbound Telemetry & Intelligence Data Flow
```
Field Crop Canopy → AgriRover Mobile Chassis → ESP32 Dev Module Controller → Ground Sensors (DHT11, Capacitive Soil v1.2, Raindrop, Water Level) + GPS NEO-6M + OV2640 Camera → FastAPI IoT Gateway (/api/v1/iot/observation) → Environmental Risk Engine + Crop Intelligence Recommendation → 3-Tier Crop Vision Inference (EfficientNetV2-S + YOLO11 Nano + Mobile UNet) → Live WebSocket Broadcast (/api/v1/iot/live-ws) → SmartCropVision Dashboard + 2.4-inch TFT LCD Status HUD + n8n Automated Webhooks + XiaoZhi Conversational Assistant
```

### 2. Reverse Physical Actuation & Teleoperation Flow
```
Dashboard Operator / XiaoZhi Voice Command → Backend Command Gateway (/api/v1/iot/device/command) → Physical Envelope & Angular Safety Gate → ESP32 Dev Module Master Controller → Local 600ms Hardware Safety Watchdog → L293D Dual H-Bridge (Motors) / PCA9685 I2C (4-DOF Arm) → Physical Motion Execution → Hardware State Acknowledgement → Backend Gateway → Live Dashboard & 2.4-inch TFT Update
```

---

## ⚡ Master Circuit Connection & Interface Table

| Subsystem / Peripheral | Hardware Component | ESP32 Pin / Interface | Supply Voltage | Signal Type | Role & Electrical Notes |
|:---|:---|:---:|:---:|:---:|:---|
| **Chassis Motors** | L293D Left In 1 (1A) | **GPIO 5** | 5V / 7.4V Batt | Digital Out | Left tracks forward logic |
| **Chassis Motors** | L293D Left In 2 (2A) | **GPIO 18** | 5V / 7.4V Batt | Digital Out | Left tracks reverse logic |
| **Chassis Motors** | L293D Right In 1 (3A) | **GPIO 19** | 5V / 7.4V Batt | Digital Out | Right tracks forward logic |
| **Chassis Motors** | L293D Right In 2 (4A) | **GPIO 21** | 5V / 7.4V Batt | Digital Out | Right tracks reverse logic |
| **Field Illumination** | Rover White Headlight | **GPIO 2** | 3.3V Logic | Digital Out | Foliage inspection illumination |
| **Microclimate Sensor**| DHT11 Temp & Humidity | **GPIO 4** | 3.3V / 5V | Digital Bus | Single-wire data protocol (10k pull-up) |
| **Root Water Status** | Capacitive Soil v1.2 | **GPIO 34** | 3.3V | Analog In (ADC1) | ADC1_CH6 (Input only; Wi-Fi radio safe) |
| **Rain Detection** | Raindrop Module (DO) | **GPIO 27** | 3.3V | Digital In | Active LOW precipitation flag |
| **Rain Intensity** | Raindrop Module (AO) | **GPIO 32** | 3.3V | Analog In (ADC1) | ADC1_CH4 analog droplet density proxy |
| **Drainage Probe** | Water Level Sensor | **GPIO 35** | 3.3V | Analog In (ADC1) | ADC1_CH7 standing water & flood depth |
| **GPS Spatial Fix** | GPS NEO-6M TX | **GPIO 16 (RX2)** | 3.3V / 5V | UART2 Serial | HardwareSerial2 at 9600 baud |
| **GPS Spatial Fix** | GPS NEO-6M RX | **GPIO 17 (TX2)** | 3.3V / 5V | UART2 Serial | HardwareSerial2 transmit to GPS |
| **4-DOF Robotic Arm** | PCA9685 SDA | **GPIO 23** | 3.3V Logic | I2C Data | Hardware I2C Wire bus at address `0x40` |
| **4-DOF Robotic Arm** | PCA9685 SCL | **GPIO 22** | 3.3V Logic | I2C Clock | Hardware I2C Clock at 400 kHz |
| **2.4" TFT Display** | Color LCD Controller | **SPI / VSPI** | 3.3V Logic | SPI Bus | CS: 15, DC: 14, RST: 13, MOSI: 23, SCK: 18 |
| **Arm Servo Power** | External Battery Pack | **V+ (PCA9685)** | **5.0V – 6.0V** | High-Current DC | **External 3A–5A supply only; NEVER from ESP32** |
| **Common System GND**| All Boards & Drivers | **GND** | 0V | System Ground | Mandatory common ground reference |

---

## 🔧 Comprehensive Field Troubleshooting Guide

1. **Wi-Fi & SoftAP Issues:**
   - If the station Wi-Fi fails to connect within 8 seconds, the ESP32 automatically starts `AgriRover-Field-AP` at `192.168.9.1` (Password: `agrirover123`).
   - Connect to the SoftAP on your mobile device or laptop and navigate to `http://192.168.9.1` or `http://agrirover.local`.
2. **GPS NEO-6M Satellites Searching:**
   - The NEO-6M requires clear line-of-sight to the sky. Under dense metal roofs, satellite acquisition will stay in `NO_FIX` state.
   - Place the ceramic patch antenna facing upward outdoors. Lock typically occurs within 45 seconds (indicated by flashing blue LED on module).
3. **Capacitive Soil Moisture Calibration:**
   - Dry air baseline reading is approximately `3100–3300` raw ADC.
   - Submerged in water reading is approximately `1200–1400` raw ADC.
   - Adjust `soil_dry_ref` and `soil_wet_ref` constants in firmware configuration if custom soil type requires shifted calibration.
4. **Water Level vs. Soil Moisture Disambiguation:**
   - Capacitive soil moisture indicates direct root dehydration and drought.
   - Water level sensor indicates standing drainage water, flood accumulation, or furrow pooling. Never confuse water level depth with root water stress.
5. **Motor Watchdog Tripping:**
   - The ESP32 enforces a local 600ms hardware watchdog. If browser heartbeats lapse (due to tab backgrounding or Wi-Fi packet drop), motors stop automatically.
   - Re-tap any direction button on the dashboard to immediately resume motion.
6. **PCA9685 Servo Jitter or Brownout:**
   - Jitter occurs when servos pull high surge current from an inadequate power source. Always power servos from a dedicated 5V–6V 3A+ battery pack or step-down converter, never from the ESP32 3.3V or VIN pins.
7. **n8n Alerts and Webhook Testing:**
   - Set `N8N_WEBHOOK_URL` in `.env`. Test alerts using the automated test suite `pytest tests/test_iot_integration.py`.
   - Alerts feature a built-in 5-minute cooldown to prevent notification spam across Telegram, WhatsApp, or email.
8. **XiaoZhi Voice Assistant & MCP Queries:**
   - XiaoZhi queries `/api/v1/xiaozhi/chat` and reads live values directly from `_latest_observation`.
   - Actuation tools are bounded: `base` (5°–175°), `shoulder` (0°–180°), `elbow` (0°–180°), `gripper` (70°–180°), rover duration max 2500ms.
