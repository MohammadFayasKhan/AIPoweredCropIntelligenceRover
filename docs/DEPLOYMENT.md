# AI Powered Crop Intelligence Rover Deployment & Operations Guide

## 1. Verified Live Production Deployment

The project maintains an active, publicly accessible production instance for field validation and SIH2026 evaluations:

* **Live Web Application URL**: `https://aipoweredcropintelligencerover.dpdns.org/`
* **Live Health Check (Liveness)**: `https://aipoweredcropintelligencerover.dpdns.org/health/live`
* **Live Model Readiness (Readiness)**: `https://aipoweredcropintelligencerover.dpdns.org/health/ready`
* **Interactive OpenAPI Documentation**: `https://aipoweredcropintelligencerover.dpdns.org/docs`

All vision inference runs strictly server side. Client browsers and mobile field units never download or execute heavy neural network weight files.

---

## 2. Quick Start Local Development

### 2.1 Prerequisites
* Python 3.10, 3.11, 3.12, or 3.13
* PyTorch 2.0 or higher
* Node.js (optional, for frontend testing)

### 2.2 Installation
```bash
# 1. Clone repository
git clone https://github.com/MohammadFayasKhan/AIPoweredCropIntelligenceRover.git
cd AIPoweredCropIntelligenceRover

# 2. Create and activate a clean virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install production dependencies
pip install -r requirements.txt
```

### 2.3 Launching the Backend
```bash
# Start FastAPI backend server
python3 backend/run.py
```
The server preloads the model registry, warms up GPU or CPU kernels, and binds to `http://0.0.0.0:8000`.

### 2.4 Opening the Web Client
The frontend is a vanilla single page application requiring zero compilation or bundling steps:
```bash
# Option A: Open directly in your browser
open frontend/index.html

# Option B: Serve through a local HTTP server
python3 -m http.server 3000 --directory frontend
```
Navigate to `http://localhost:3000` to interact with the diagnostic dashboard.

---

## 3. Docker Container Deployment

The provided `Dockerfile` builds a production container running as an unprivileged system user.

### 3.1 Building the Container
```bash
docker build -t crop-intelligence-rover:v2.2 .
```

### 3.2 Running with Docker Compose
```bash
docker compose up -d
```
The container exposes port 8000 and includes automatic healthcheck polling at `/health/live`.

---

## 4. Air Gapped & Kaggle Deployment

For deployments on air gapped edge servers or Kaggle kernels where external internet connectivity is restricted, offline wheel packages are bundled in `packages/`:

```bash
# Install required libraries 100% offline
pip install --no-index --find-links packages/ \
    ultralytics \
    albumentations \
    albucore \
    nvidia_ml_py \
    ultralytics_thop
```

---

## 5. Operational Verification Checklist

### 5.1 Confirm Models Remain Server Side
To verify that large weight files are never transmitted to user browsers:
1. Open your browser Developer Tools (`F12`) and select the Network tab.
2. Upload a leaf sample and click Analyze Leaf Specimen.
3. Verify that the only outbound network request is a POST to `/api/v1/vision/diagnose` transmitting the image bytes.
4. Verify that zero requests attempt to fetch `.pt`, `.pth`, or `.pkl` weight files.

### 5.2 Confirm Loaded Model Versions
Query the live registry endpoint:
```bash
curl -s http://localhost:8000/models/status | python3 -m json.tool
```
Confirm that `models_ready` matches expected production checkpoints and that the active compute device is truthfully identified (MPS, CUDA, or CPU).

---

## 6. Rollback Procedure

If a deployed checkpoint exhibits an unexpected anomaly:
1. Identify the previous stable release tag from `RELEASE_MANIFEST.json`.
2. Verify the SHA 256 signature of the backup checkpoint in `cv/models/`.
3. Update `SERVER_TIER1_MODEL_PATH` in `.env` to point to the verified checkpoint.
4. Restart the backend service. The startup preflight will validate the signature before declaring readiness.
