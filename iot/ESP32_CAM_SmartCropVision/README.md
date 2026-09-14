# SmartCropVision ESP32-CAM Field Unit Firmware

## 1. Overview & System Architecture

This directory contains the production Arduino firmware for the **SmartCropVision ESP32-CAM** field unit equipped with an **OV2640** camera sensor.

```
┌─────────────────────────┐          Outbound WSS / WS          ┌──────────────────────────┐
│   ESP32-CAM + OV2640    │ ─────────────────────────────────>  │  SmartCropVision Backend │
│  (Field Image Capture)  │   • Non-blocking WiFi & Backoff     │  (FastAPI + Model Hub)   │
│                         │   • Authentic Heartbeat Telemetry   │                          │
│  [FRAMESIZE_QVGA (prev)]│   • Binary Preview Relay (SPRV)     │  • Preflight Validator   │
│  [FRAMESIZE_VGA  (cap)] │   • High-Res Capture Dispatch (SCAP)│  • EfficientNetV2-S (T1) │
└─────────────────────────┘                                     │  • YOLO26 PlantDoc (T2)  │
             ▲                                                  │  • Mobile-UNet Foliar(T3)│
             │                                                  └──────────────────────────┘
      Direct Local LAN                                                        ▲
      HTTP Port 81                                                            │ REST / Preview
      (/stream & /capture)                                                    │
             │                                                  ┌──────────────────────────┐
             └────────────────────────────────────────────────  │   Web Dashboard (SPA)    │
                                                                │ [Upload][Camera][ESP32]  │
                                                                └──────────────────────────┘
```

### Core Responsibilities
- **Field Acquisition Only**: The ESP32-CAM strictly captures, previews, and streams images. It does **not** perform on-device neural disease classification.
- **Authoritative Server Inference**: Classification (EfficientNetV2-S), Specimen Detection (YOLO26), and Foliar Segmentation (Mobile-UNet) remain hosted on the server backend.
- **Source-Aware Validation**: Authenticated ESP32-CAM captures receive an adjusted optical quality tolerance (tolerating mild sensor noise and blur) while strictly maintaining semantic plant validation.

---

## 2. Hardware Specifications & Board Selection

- **Microcontroller**: Espressif ESP32-D0WD-V3 (Dual Core 240MHz, 520KB SRAM).
- **External Memory**: 4MB External Pseudo-SRAM (PSRAM) on the AI-Thinker module.
- **Camera Sensor**: OmniVision OV2640 (2-Megapixel, UXGA 1600×1200 native).
- **Flash / Illuminator**: Onboard white power LED on GPIO 4.
- **Status Indicator**: Onboard red indicator on GPIO 33 (inverted logic).
- **Baseboard**: ESP32-CAM-MB micro-USB programmer baseboard (CH340G USB-UART).

---

## 3. Arduino IDE Setup & Configuration

### Step A: Install the ESP32 Board Package
1. Open **Arduino IDE**.
2. Go to **File → Preferences** (or **Arduino IDE → Settings** on macOS).
3. In **Additional Board Manager URLs**, add:
   ```text
   https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json
   ```
4. Go to **Tools → Board → Boards Manager**, search for `esp32` by **Espressif Systems**, and install the latest stable version.

### Step B: Install Required Libraries
Open **Tools → Manage Libraries...** and install:
1. **ArduinoWebsockets** (by *Gil Maimon*)
2. **ArduinoJson** (by *Benoit Blanchon*, v6.x or v7.x)

### Step C: Select Exact Board Parameters
In the **Tools** menu, configure the following settings:
- **Board**: `"AI Thinker ESP32-CAM"`
- **CPU Frequency**: `240MHz (WiFi/BT)`
- **Flash Frequency**: `80MHz`
- **Flash Mode**: `QIO`
- **Partition Scheme**: `"Huge APP (3MB No OTA/1MB SPIFFS)"`
- **Core Debug Level**: `"None"` (or `"Info"` during initial testing)
- **PSRAM**: `"Enabled"` ⚠️ **(Crucial for double-buffering & high-res capture)**
- **Upload Speed**: `115200` (or `921600` if your USB cable supports high-speed transfer)

---

## 4. Configuration (`config.h`)

Copy `config.example.h` to `config.h`:
```bash
cp config.example.h config.h
```
Edit `config.h` with your environment parameters:
```c
// WiFi Network
#define WIFI_SSID             "Your_WiFi_Name"
#define WIFI_PASSWORD         "Your_WiFi_Password"

// SmartCropVision Backend Server
#define BACKEND_HOST          "192.168.1.100"      // Mac/Server IP on local LAN
#define BACKEND_PORT          8000                 // FastAPI server port
#define BACKEND_WS_PATH       "/api/v1/esp32/device-ws"
#define USE_SECURE_WSS        false                // Set true for wss:// in cloud deployments

// Authentication
#define DEVICE_ID             "esp32-cam-01"
#define DEVICE_TOKEN          "smartcropvision-esp32-token-innovex"
```

---

## 5. Flashing & Uploading Instructions

### Method 1: Using ESP32-CAM-MB Baseboard (Recommended)
1. Snap the ESP32-CAM board directly onto the ESP32-CAM-MB daughterboard.
2. Plug the micro-USB cable into your computer.
3. Select the correct serial port under **Tools → Port** (e.g., `/dev/cu.usbserial-110` on macOS).
4. Click the **Upload** button (`→`) in Arduino IDE.
5. Once upload completes (`Leaving... Hard resetting via RTS pin...`), open **Serial Monitor** at **115200 baud**.

### Method 2: Using FTDI / USB-to-UART Adapter
If flashing without the baseboard:
```
FTDI Programmer           ESP32-CAM
─────────────────         ─────────
VCC (5V)         ───────> 5V (or 3.3V)
GND              ───────> GND
TX               ───────> U0R (GPIO 3)
RX               ───────> U0T (GPIO 1)
IO0              ───────> GND  <── Jumper required for flashing!
```
1. Connect **IO0 to GND**.
2. Press the **RST** button on the ESP32-CAM to enter download bootloader mode.
3. Click **Upload** in Arduino IDE.
4. When finished, **disconnect IO0 from GND** and press **RST** again to run the firmware.

---

## 6. Verification & Serial Diagnostics

Upon successful boot, the serial monitor will display:
```text
========================================================
    SmartCropVision ESP32-CAM Production Firmware      
    Firmware Version : v1.0.0-prod
    Device ID        : esp32-cam-01
========================================================
[CAMERA] PSRAM detected (4092 KB free). Enabling double frame buffers.
[CAMERA] OV2640 hardware successfully initialized and calibrated.
[WIFI] Connecting to configured network: Fayas
[HTTP] Local fallback server running on port 81 (/stream, /capture, /status)
[WS] Connecting outbound to gateway: 192.168.1.100:8000/api/v1/esp32/device-ws
[WS] Connected to SmartCropVision gateway. Status: ONLINE
```

---

## 7. Troubleshooting Matrix

| Issue | Root Cause | Solution |
| :--- | :--- | :--- |
| `Brownout detector was triggered` | Insufficient supply current | ESP32-CAM draws up to 350mA peak during WiFi transmission. Connect to a dedicated 5V 2A USB power source, or add a 470µF electrolytic capacitor across 5V and GND. |
| `Camera init failed with error 0x20003` | Ribbon cable loose or bad contact | Gently unlock the FPC camera connector, re-seat the OV2640 gold contacts firmly, and close the black latch. |
| `No PSRAM detected` | Incorrect board setting | In Arduino IDE, verify **Tools → PSRAM → Enabled** is selected. |
| `WiFi offline. Retrying...` | SSID / password error or weak signal | Check credentials in `config.h`. Ensure 2.4GHz WiFi band (ESP32 does not support 5GHz). |
| `WS Gateway connection failed` | Backend not reachable | Verify backend is running on the specified IP (`curl -I http://<BACKEND_HOST>:8000/docs`). |
