# ==============================================================================
# AI Powered Crop Intelligence Rover Production Inference Container
# Lightweight, security-hardened container for CPU and CUDA GPU deployment
# ==============================================================================

FROM python:3.11-slim as base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000 \
    ENVIRONMENT=production \
    MPLCONFIGDIR=/tmp/matplotlib \
    YOLO_CONFIG_DIR=/tmp/ultralytics \
    TORCH_HOME=/tmp/torch \
    NUMBA_CACHE_DIR=/tmp/numba

# Install minimal OS dependencies for OpenCV and image operations
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy production code, models, configs, artifacts, and frontend
COPY backend/ ./backend/
COPY cv/ ./cv/
COPY artifacts/ ./artifacts/
COPY frontend/ ./frontend/
COPY RELEASE_MANIFEST*.json ./

# Create unprivileged system user and prepare writable directories
RUN useradd -u 1001 -m -s /bin/bash smartcrop && \
    mkdir -p /tmp/matplotlib /tmp/ultralytics /tmp/torch /tmp/numba && \
    chown -R smartcrop:smartcrop /app /tmp/matplotlib /tmp/ultralytics /tmp/torch /tmp/numba && \
    chmod -R 777 /tmp

USER smartcrop

EXPOSE 8000

# Healthcheck probe using liveness endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health/live || exit 1

CMD ["sh", "-c", "python3 -m uvicorn backend.app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
