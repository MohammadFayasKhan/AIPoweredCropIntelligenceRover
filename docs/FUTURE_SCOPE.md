# AI Powered Crop Intelligence Rover: Future Scope & Research Roadmap

> **Strategic Engineering Horizons, Field Scaling, and Agronomic Enhancements**  
> **Team:** Innovex | **SIH 2026 Challenge ID:** SIH26180  

---

## 1. Computer Vision & Edge AI Evolution

1. **Edge TPU & NPU Compilation (Qualcomm Snapdragon / Coral):**
   * Compile the Tier 1 classifier and Mobile-UNet into INT8 quantized models using Qualcomm Neural Processing SDK (QNN) and TensorFlow Lite.
   * Target: Sub-30ms offline inference directly on an onboard Snapdragon edge processor mounted to the rover chassis.
2. **Open-Set Fine-Grained Anomaly Detection:**
   * Extend the pre-inference gate with Energy-Based Out-of-Distribution (OOD) scoring to detect completely novel fungal pathogens without misclassifying them into the 38 closed-set categories.
3. **Multispectral Canopy Indices:**
   * Integrate an affordable dual-band Near-Infrared (NIR) filter onto the rover camera to compute Normalized Difference Vegetation Index (NDVI) values directly from canopy reflection.

---

## 2. Robotics & Physical Autonomy Scaling

1. **Row-Following Autonomous Visual Odometry:**
   * Deploy visual SLAM (Simultaneous Localization and Mapping) using stereo cameras or dual OV2640 modules to enable autonomous navigation down crop furrows without GPS reliance in dense canopies.
2. **Precision Micro-Dose Spraying End-Effector:**
   * Transition the 4-DOF canopy robotic arm from passive inspection to active localized intervention by equipping it with a micro-solenoid valve nozzle that dispenses targeted fungicide micro-doses exclusively to detected lesion coordinates.
3. **Solar-Assisted Power Architecture:**
   * Mount a 20W monocrystalline solar panel with an MPPT charge controller to extend field scouting endurance from 4 hours to dawn-to-dusk continuous patrol.

---

## 3. Agronomic Grounding & Institutional Partnerships

1. **Krishi Vigyan Kendra (KVK) Field Trials:**
   * Partner with local agricultural research stations across Andhra Pradesh and Telangana to calibrate capacitive soil probe voltage thresholds across red sandy loam, black cotton, and alluvial soils.
2. **Real-Time Mandi Market & Agmarknet API Integration:**
   * Integrate live e-NAM (National Agriculture Market) and AGMARKNET APIs to correlate field yield predictions and disease severity with real-time regional mandi wholesale prices, storage advisories, and harvest window recommendations.
3. **Pradhan Mantri Fasal Bima Yojana (PMFBY) Ground Truth Verification:**
   * Utilize tamper-evident GPS-tagged telemetry and damage quantification reports to accelerate crop insurance loss assessments after extreme weather events.
