# AI Powered Crop Intelligence Rover: System Architecture & Data Engineering

> **Official Architecture Specification for Smart India Hackathon 2026**  
> **Challenge ID:** SIH26180 | **Theme:** Disaster Management | **Category:** Hardware  
> **Team:** Innovex (SIH-H-B2-048)  
> **Canonical Identity:** AI Powered Crop Intelligence Rover  

---

## 1. Architectural Philosophy & Core Advantages

The **AI Powered Crop Intelligence Rover** is engineered as a cyber-physical agronomic intelligence system that combines autonomous mobile ground robotics, multi-point microclimate telemetry, and a multi-tier computer vision inference cascade.

The system is built upon four fundamental design principles:

1. **Strict Decoupling of Visual Pathology and Environmental Favorability:**
   * **Optical Foliar Diagnosis:** Evaluates macro and microscopic leaf tissue symptoms (what necrotic lesions, chlorosis, or pustules currently exist on the plant).
   * **Microclimatic Disease Risk Engine:** Continuously computes ambient fungal and bacterial incubation risks from temperature, relative humidity, rain accumulation, and soil moisture (what environmental hazards will trigger pathogen proliferation).
   * This decoupled architecture ensures that farmers receive both immediate curative advisories for observed infections and proactive preventive warnings before airborne spores spread.

2. **Server-Side Neural Network Weight Execution:**
   * High-capacity neural networks (EfficientNetV2-S at 78 MB, YOLO-PlantDoc at 5.2 MB, and Mobile-UNet at 1.9 MB) execute strictly on accelerated servers.
   * Edge devices (smartphones, low-power field tablets, ESP32 microcontrollers) remain ultra-lightweight and battery-efficient, communicating via lightweight REST and WebSocket protocols without downloading or running model weights locally.

3. **Calibrated Confidence & Zero False Hallucination Policy:**
   * Temperature-scaled probability calibration ($T = 1.42$, ECE: 0.0803) prevents overconfident erroneous classifications.
   * A rigorous pre-inference domain validation gate filters out blurry imagery, non-plant items, and software screenshots before invoking compute-heavy networks.
   * Healthy foliage produces zero synthetic bounding boxes, preserving honest diagnostic evidence.

4. **Multi-Stage Botanical Explainability by Default:**
   * Layer-by-layer Grad-CAM heatmaps verify that network activation focuses on genuine pathological lesions rather than background soil, shadows, or specular sunlight.

---

## 2. End-to-End System Topology

```text
                           FIELD / PHYSICAL EDGE LAYER
      ┌─────────────────────────────────────────────────────────────────┐
      │  AgriRover Mobile Chassis (ESP32 + L293D Dual H-Bridge)         │
      │  Crop Intelligence Sensor Node (ESP32 + DHT11 + Capacitive Soil │
      │  + Rain Sensor + Water Level + NEO-6M GPS + 16x2 I2C LCD)       │
      │  OV2640 Field Camera Module (ESP32-CAM Streaming Gateway)       │
      │  Web Teleoperation Dashboard (Vanilla JS SPA)                   │
      └────────────────────────────────┬────────────────────────────────┘
                                       │ HTTPS / WSS Telemetry Stream
                                       ▼
                      FASTAPI EDGE / CLOUD GATEWAY (backend/)
      ┌─────────────────────────────────────────────────────────────────┐
      │  Security Headers & Rate Limiting                               │
      │  Image Decompression Bomb Guard (Max 16 MP)                     │
      │  Request ID Tracing & Structured Logging                        │
      └───────┬─────────────────────────────────────────────────┬───────┘
              │                                                 │
       [ Image Upload ]                                  [ IoT Telemetry ]
              ▼                                                 ▼
  ┌───────────────────────────┐                     ┌───────────────────────────┐
  │ Pre-Inference Domain Gate │                     │ Environmental Risk Engine │
  │ • Laplacian Blur Check    │                     │ • Rothfusz Heat Index     │
  │ • Foliar Presence Ratio   │                     │ • Fungal Incubation Risk  │
  │ • Screenshot Filter       │                     │ • Flood / Drought Bounds  │
  └───────────┬───────────────┘                     └───────────┬───────────────┘
              ▼ (Valid Leaf)                                    │
  ┌──────────────────────────────────────────────────┐          │
  │       Multi-Tier Computer Vision Cascade         │          │
  │                                                  │          │
  │ Tier 1: EfficientNetV2-S (38 Classes, Acc 95.1%) │          │
  │   └── Temperature Scaled Calibration (ECE: 0.08) │          │
  │                                                  │          │
  │ Tier 2: YOLO PlantDoc (Canopy Detection, 29 Cls) │          │
  │   └── Zero synthetic boxes on healthy foliage    │          │
  │                                                  │          │
  │ Tier 3: Mobile-UNet (Sub-pixel Segmentation)     │          │
  │   └── Quantified Foliar Damage Index (0-100%)    │          │
  │                                                  │          │
  │ Explainability: 9-Stage Dynamic Grad-CAM         │          │
  │   └── Verifiable lesion activation heatmaps      │          │
  └───────────────────┬──────────────────────────────┘          │
                      │                                         │
                      ▼                                         ▼
  ┌─────────────────────────────────────────────────────────────────────┐
  │             Agronomic Advisory & Decision Synthesis Engine           │
  │  • Random Forest Cultivar Recommendation (22 Agronomic Crops)       │
  │  • CIBRC Approved Chemical & Organic Remedies (FRAC Codes)          │
  │  • Real-Time WebSocket Telemetry Broadcast to Web UI & 16x2 LCD     │
  └─────────────────────────────────────────────────────────────────────┘
```

---

## 3. Computer Vision Multi-Tier Cascade

### Tier 1: Universal Foliar Classification (EfficientNetV2-S)
* **Architecture:** EfficientNetV2-S initialized with ImageNet-1k weights and fine-tuned on 38 botanical classes.
* **Input Resolution:** $256 \times 256 \times 3$ normalized with standard ImageNet mean and variance.
* **Accuracy:** **95.13% Top-1 Test Accuracy** | **Macro F1: 0.9354**.
* **Confidence Calibration:** Applies post-training temperature scaling ($T = 1.42$). Uncalibrated deep networks often assign $>99\%$ probability to out-of-distribution inputs; temperature scaling flattens overconfident logits, yielding an Expected Calibration Error (ECE) of **0.0803**.

### Tier 2: Spatial Canopy Localization (YOLO PlantDoc)
* **Architecture:** Ultralytics YOLO11 nano fine-tuned on the 2,581 Pascal VOC image-annotation pairs of the PlantDoc benchmark.
* **Input Resolution:** $640 \times 640$ letterboxed tensor.
* **Validation mAP@50:** **0.3362** | **Validation mAP@50:95:** **0.2361**.
* **Zero-Hallucination Policy:** Healthy leaves produce strictly zero bounding boxes. The model never fabricates lesion boxes where no pathological target exists.

### Tier 3: Sub-Pixel Foliar Tissue Segmentation (Mobile-UNet)
* **Architecture:** Custom lightweight Mobile-UNet with depthwise separable convolutions (total size: **1.90 MB**).
* **Objective:** Segments input foliage into 3 distinct pixel classes:
  1. Class 0: Background / Non-leaf artifacts
  2. Class 1: Healthy foliar tissue
  3. Class 2: Necrotic lesion / fungal chlorosis
* **Metrics:** **0.7485 Mean IoU** | **0.8412 Dice Score**.
* **Foliar Damage Calculation:** Automatically computes:
  $$\text{Foliar Damage \%} = \frac{\sum \text{Lesion Pixels}}{\sum \text{Healthy Pixels} + \sum \text{Lesion Pixels}} \times 100$$

### 9-Stage Explainability Pipeline (Grad-CAM)
* Dynamically resolves the final convolutional layer of EfficientNetV2-S (`conv_head`).
* Computes gradients of the target class score with respect to feature activation maps.
* Renders 9 visual inspection stages: Input foliage $\rightarrow$ RGB channel distribution $\rightarrow$ Preprocessed normalized tensor $\rightarrow$ Feature activation collage $\rightarrow$ Raw gradient heatmap $\rightarrow$ Thresholded attention mask $\rightarrow$ Alpha-blended Grad-CAM overlay $\rightarrow$ Spatial bounding overlay $\rightarrow$ Diagnostic summary card.

---

## 4. Environmental Intelligence & Agro-Climatic Recommender

### Multi-Point Physical Sensing
The Crop Intelligence node continuously aggregates:
* **Temperature & Relative Humidity (DHT11):** Computes atmospheric vapor pressure deficit and Rothfusz Heat Index.
* **Capacitive Soil Moisture Probe (v1.2):** High-frequency analog sensing impervious to direct electrochemical corrosion.
* **Raindrop Presence Sensor (YL-83):** Distinguishes between dry canopy, light drizzle, and heavy rainfall.
* **Water Depth Probe:** Measures ground puddle depth to flag waterlogging.
* **NEO-6M GPS Module:** Tags environmental observations with latitude, longitude, and altitude for geospatial field mapping.

### Random Forest Crop Recommendation Model
* **Dataset:** 2,200 curated field records across 22 crop types (Rice, Maize, Chickpea, Kidney Beans, Pigeonpeas, Mothbeans, Mungbean, Blackgram, Lentil, Pomegranate, Banana, Mango, Grapes, Watermelon, Muskmelon, Apple, Orange, Papaya, Coconut, Cotton, Jute, Coffee).
* **Features:** Nitrogen (N), Phosphorus (P), Potassium (K), Temperature, Humidity, Soil pH, and Rainfall.
* **Accuracy:** **99.32% test accuracy** on stratified 5-fold cross-validation.
* **Output:** Recommended primary cultivar with confidence score and ranked runner-up alternatives.

---

## 5. Hardware Interface & Telemetry Flow

The hardware stack is partitioned into distinct modular subsystems:

1. **AgriRover Chassis Controller:**
   * Microcontroller: ESP32 NodeMCU.
   * Motor Driver: L293D / L298N Dual H-Bridge driving four high-torque DC geared motors.
   * Power Source: 12V 3S Li-ion battery pack with LM2596 buck converters delivering isolated 5V logic rails.
2. **Crop Intelligence Sensor Node:**
   * Microcontroller: ESP32 Dev Module.
   * Display: 16x2 Character LCD driven over I2C (PCF8574 adapter at `0x27` or `0x3F`).
   * LCD Renders 4 rotating telemetry screens with custom glyphs (degree symbol, checkmark, warning alert, and water drop).
3. **Canopy Inspection Robotic Arm:**
   * Microcontroller: Arduino Nano.
   * Servo Driver: PCA9685 16-channel 12-bit PWM I2C breakout controlling 4 MG996R / SG90 servos for multi-angle under-canopy camera orientation.
4. **Field Camera Gateway:**
   * Camera Module: ESP32-CAM with OV2640 sensor streaming JPEG frames over authenticated WebSocket (`/api/v1/camera/ws`) or HTTP multipart upload.
