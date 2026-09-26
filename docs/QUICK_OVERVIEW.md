# AI Powered Crop Intelligence Rover: Quick Overview 🌿🔬

> **Smart India Hackathon 2026** | **Problem Statement:** SIH26180 (Qualcomm Inc.)  
> **Theme:** Disaster Management | **Category:** Hardware  
> **Team:** Innovex (**SIH-H-B2-048**)  
> **Live Production URL:** [https://aipoweredcropintelligencerover.dpdns.org](https://aipoweredcropintelligencerover.dpdns.org)  
> **GitHub Repository:** [https://github.com/MohammadFayasKhan/AIPoweredCropIntelligenceRover](https://github.com/MohammadFayasKhan/AIPoweredCropIntelligenceRover)  

---

## 1. The Core Agricultural Challenge

Smallholder farmers lose **20% to 40% of their annual crop yields** to foliar diseases (early blight, late blight, powdery mildew, rust, leaf scorch). Farmers and agronomists face two compounding failure modes:

1. **Diagnostic Latency & Hallucination:** Traditional manual inspection is slow, agricultural extension officers are scarce, and existing mobile apps typically rely on closed-set black-box classifiers that hallucinate confident diagnoses on non-plant images or blurry photos.
2. **Decoupling of Symptoms from Microclimate:** Visual symptoms indicate past infection, whereas ambient humidity, temperature spikes, soil waterlogging, and rainfall patterns dictate future pathogen spore germination. Treating leaf lesions without correlating field microclimate leads to reactive, mistimed chemical applications and wasted input costs.

---

## 2. The AI Powered Crop Intelligence Rover Solution

The **AI Powered Crop Intelligence Rover** unifies field robotics, environmental Internet of Things (IoT) sensing, and a multi-tier computer vision cascade into an integrated decision-support platform:

* **Dual-Stream Field Intelligence:** Strictly separates optical foliar diagnosis (analyzing what is damaged on the leaf) from microclimatic environmental disease risk (modeling pathogen transmission risks from sensor telemetry).
* **Multi-Tier Perception Cascade:**
  * **Domain Validation Gate:** Filters out non-plant imagery, software screenshots, and unreadable files before invoking compute-heavy neural models.
  * **Tier 1 Classification (EfficientNetV2-S):** 38-class crop/disease classifier operating at **95.13% Top-1 Accuracy** with temperature scaling ($T=1.42$, ECE: 0.0803) to eliminate false certainty.
  * **Tier 2 Detection (YOLO-PlantDoc):** 29-class canopy detector localizing foliar boundaries without synthetic bounding box hallucinations.
  * **Tier 3 Segmentation (Mobile-UNet):** Lightweight 1.9 MB model computing pixel-level foliar damage ratios (healthy vs necrotic tissue).
  * **9-Stage Explainability (Grad-CAM):** Dynamic gradient activation heatmaps proving that predictions stem from genuine foliar lesions rather than background soil or specular lighting.
* **Agro-Climatic Recommendation:** Random Forest multi-class model (**99.32% test accuracy**) matching soil N-P-K, ambient temperature, humidity, pH, and rainfall to 22 optimal crop cultivars.
* **Autonomous Ground-Canopy Rover:** ESP32-powered physical scout equipped with a 16x2 I2C LCD, DHT11 atmospheric sensor, capacitive soil moisture probe, rain sensor, water level sensor, and NEO-6M GPS module.

---

## 3. Key Engineering Advantages & Innovations

1. **Zero Client Weight Execution:** Heavy neural networks execute exclusively on high-performance servers. Edge devices (smartphones, field tablets, ESP32 nodes) send lightweight HTTP/WebSocket requests, consuming negligible device battery and bandwidth.
2. **Temperature-Scaled Confidence Calibration:** Reduces Expected Calibration Error to 0.0803, ensuring low-confidence borderline samples trigger a visual verification recommendation rather than an authoritative misdiagnosis.
3. **Hardware Truthfulness:** The 16x2 I2C LCD renders real-time field readings using custom glyphs (degree, checkmark, warning alert, water drop) directly on the rover chassis.
4. **Dual-Mode Failover Resilience:** Firmware automatically redirects telemetry to a local edge gateway if cloud network connectivity drops, ensuring continuous logging under remote field conditions.

---

## 4. End-to-End System Workflow

```text
Field Rover / User Upload
   │
   ▼
[ Pre-Inference Domain Gate ] ──(Invalid / Non-Plant)──► Graceful Rejection (Zero False Metrics)
   │
   ▼ (Valid Plant Foliage)
[ Tier 1: EfficientNetV2-S ] ──► Calibrated 38-Class Foliar Diagnosis (Top-1 Acc: 95.13%)
   │
   ├──► [ Tier 2: PlantDoc YOLO ] ─────► Specimen & Foliar Bounding Boxes (29 Classes)
   │
   ├──► [ Tier 3: Mobile-UNet ] ───────► Foliar Tissue Segmentation (Healthy vs Necrotic)
   │
   └──► [ 9-Stage Grad-CAM ] ──────────► Verifiable Spatial Activation Heatmaps
   │
   ▼
[ Telemetry & Advisory Engine ] ◄── Field IoT Sensors (Temp, Humidity, Soil, Rain, GPS)
   │
   ├──► Environmental Microclimate Disease Alerts (Early Warning)
   ├──► Random Forest Crop Recommendation (22 Cultivars)
   └──► Targeted Agronomic Advisory & Safe Treatment Guidance (CIBRC / FRAC Codes)
   │
   ▼
[ Live Dashboard & 16x2 LCD Character Stream ]
```

---

## 5. Technology Stack Summary

* **Frontend:** Vanilla HTML5, CSS3 Glassmorphism, ECMAScript 2022, Chart.js, WebSocket telemetry client.
* **Backend:** FastAPI, Python 3.10+, PyTorch 2.0+, Ultralytics YOLO, Scikit-Learn, Uvicorn ASGI server.
* **Embedded IoT:** ESP32 Dev Modules (WiFi/WSS, ADC, I2C, UART), Arduino Nano (PCA9685 PWM servo driver), DHT11, Capacitive Soil Probe v1.2, YL-83 Rain Sensor, NEO-6M GPS, PCF8574 16x2 LCD.
* **Deployment:** Docker, Docker Compose, Linux x86_64 / ARM64, Caddy reverse proxy with automated SSL.
