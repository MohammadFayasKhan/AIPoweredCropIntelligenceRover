# AI Powered Crop Intelligence Rover: Setup & Execution Guide

> **Step-by-step local setup, environment configuration, and verification instructions.**  
> **Team:** Innovex | **SIH 2026 Challenge:** SIH26180  

---

## 1. Prerequisites

Ensure your host machine satisfies the following minimum requirements:
* **Operating System:** Linux (Ubuntu 20.04+), macOS (Apple Silicon / Intel), or Windows 10/11 with WSL2.
* **Python Runtime:** Python 3.10, 3.11, or 3.12 (standard CPython).
* **Hardware Acceleration (Optional):** NVIDIA GPU with CUDA 11.8+ or Apple Silicon with MPS. CPU inference is fully supported out of the box with zero code changes.
* **Network Ports:** Port 8000 (FastAPI backend) and Port 3000 (optional standalone frontend static server).

---

## 2. Local Environment Setup

### Step 1: Clone the Repository
```bash
git clone https://github.com/MohammadFayasKhan/AIPoweredCropIntelligenceRover.git
cd AIPoweredCropIntelligenceRover
```

### Step 2: Create and Activate Virtual Environment
```bash
python3 -m venv .venv

# On Linux / macOS:
source .venv/bin/activate

# On Windows (cmd):
.venv\Scripts\activate.bat

# On Windows (PowerShell):
.venv\Scripts\Activate.ps1
```

### Step 3: Install Production Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 3. Environment Configuration

Copy the example environment configuration template:
```bash
cp .env.example .env
```

Review `.env` settings:
```ini
PROJECT_NAME="AI Powered Crop Intelligence Rover API"
ENVIRONMENT="development"
HOST="0.0.0.0"
PORT=8000
API_V1_STR="/api/v1"
CORS_ORIGINS="*"

# Model Paths
SERVER_TIER1_MODEL_PATH="cv/models/efficientnetv2_s_best.pt"
TIER2_PLANTDOC_MODEL_PATH="cv/models/yolo_plantdoc_best.pt"
TIER3_MODEL_PATH="cv/models/mobile_unet_best.pt"
CROP_MODEL_DIR="cv/models/crop_recommender"

# Assistant Configuration
XIAOZHI_ENABLED="true"
```

---

## 4. Running the Application

### Option A: Direct Python Execution (Recommended for Development)
From the repository root directory:
```bash
python3 backend/run.py
```
* **Backend API & Swagger UI:** `http://127.0.0.1:8000/docs`
* **Web Teleoperation Dashboard:** Open `http://127.0.0.1:8000/` in any modern web browser. The backend automatically serves the static dashboard directly from `frontend/`.

### Option B: Standalone Frontend Dev Server (Optional)
If you wish to serve the frontend on a separate port:
```bash
python3 -m http.server 3000 --directory frontend/
```
Open `http://127.0.0.1:3000` in your browser.

---

## 5. Docker Deployment

To build and run the production-hardened container:

```bash
# Build the Docker image
docker build -t crop-intelligence-rover:latest .

# Run the container
docker run -d -p 8000:8000 --name crop-rover crop-intelligence-rover:latest
```

Using Docker Compose:
```bash
docker compose up --build -d
```

Verify container health:
```bash
curl http://127.0.0.1:8000/health/live
curl http://127.0.0.1:8000/health/ready
```

---

## 6. Running Automated Tests

Run the complete test suite:
```bash
pytest tests/ -v -k "not test_camera_integration and not test_esp32_cam"
```
*(Physical camera integration tests are automatically skipped in offline environments where physical USB/IP camera hardware is not plugged in).*
