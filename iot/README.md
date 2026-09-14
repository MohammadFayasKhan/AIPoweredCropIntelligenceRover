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

## 💻 USB Serial Command Line (Backwards Compatibility)
The USB Serial interface operates at **115200 baud** and accepts both text words and legacy HC-05 format commands:
- `s1<ang>`: Set Base Angle (e.g. `s190`)
- `s2<ang>`: Set Shoulder Angle (e.g. `s2120`)
- `s3<ang>`: Set Elbow Angle (e.g. `s360`)
- `s6<ang>`: Set Gripper Angle (e.g. `s6180`)
- `ss<spd>`: Set Arm Speed (e.g. `ss80`)
- `SAVE`: Record current pose as waypoint
- `RUN`: Begin sequence execution
- `PAUSE` / `RESUME` / `STOP`: Sequence control
- `RESET`: Return to Home stance
- `CLEAR`: Flush waypoint records
- `POSITION`: Print current joint targets
- `RECORDS`: Print all recorded waypoints

---

## 🛠️ Arduino IDE Setup & Libraries Required
To compile and flash `iot/AgriRover_Innovex_SIH.ino`:
1. **Board Manager:** Install `esp32` by Espressif Systems (v2.0.x or v3.x).
   - Select Board: `ESP32 Dev Module`.
2. **Required Libraries (Install via Arduino Library Manager):**
   - `Adafruit PWM Servo Driver Library` (by Adafruit)
   - `Wire` (built-in)
   - `WiFi` (built-in)
   - `WebServer` (built-in)
   - `ESPmDNS` (built-in)
3. **Upload Settings:**
   - Upload Speed: `921600` or `115200`
   - CPU Frequency: `240MHz (WiFi/BT)`
   - Flash Frequency: `80MHz`
   - Partition Scheme: `Default 4MB with spiffs (1.2MB APP / 1.5MB SPIFFS)` or `Huge APP (3MB No OTA)`
