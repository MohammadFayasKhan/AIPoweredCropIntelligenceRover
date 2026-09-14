"""
SmartCropVision Production Application Entrypoint.
Unified, production-grade inference service providing 3-Tier plant pathology diagnosis,
agro-climatic crop recommendation, real-time IoT hardware telemetry, and dashboard serving.
"""

from contextlib import asynccontextmanager
from collections import deque
from datetime import datetime, timezone
import logging
import uuid
import time
from pathlib import Path
from typing import Optional, Dict, Any

from fastapi import FastAPI, Request, UploadFile, File, Form, HTTPException, status, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from backend.app.config import settings
from backend.app.services.inference_service import inference_engine
from backend.app.services.crop_service import crop_service
from backend.app.schemas.crop import CropRecommendationRequest
from backend.app.schemas.diagnosis import (
    HealthStatusResponse,
    LivenessStatusResponse,
    ReadinessStatusResponse,
    ModelsStatusResponse,
    ModelRegistryResponse,
    ErrorResponse,
)
from backend.app.services.model_registry import model_registry
from backend.app.services.esp32_gateway import esp32_gateway
from backend.app.api.v1.router import api_router
from backend.app.api.v1.endpoints.health import health_check, models_status, liveness_probe, readiness_probe
from backend.app.utils.image_processing import ImageValidationError, validate_uploaded_image
from backend.app.utils.domain_validation import validate_plant_image

# Configure structured application logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("smartcropvision.server")

# In-memory IoT reading history (last 50 physical or verified readings)
history: deque = deque(maxlen=50)

# Latest reading from physical ESP8266 IoT hardware
latest_iot: Optional[Dict[str, Any]] = None

FRONTEND_DIR = settings.PROJECT_ROOT / "frontend"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager: Preloads ML vision models and crop recommendation artifacts at startup."""
    logger.info("Initializing SmartCropVision production service...")
    try:
        inference_engine.load_models()
        logger.info("ML vision suite preloaded on compute device: %s.", inference_engine.device_name)
    except Exception as e:
        logger.error("Failed to preload vision inference models: %s", e)

    try:
        crop_service.load_artifacts()
        logger.info("Crop recommendation service preloaded successfully.")
    except Exception as e:
        logger.error("Failed to preload crop recommendation service: %s", e)

    yield
    logger.info("Gracefully shutting down SmartCropVision API service.")


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description=(
        "Production-grade agricultural intelligence API executing a 3-Tier vision cascade: "
        "Tier 1 (Server EfficientNet-B2 / Edge MobileNetV2 38-class screening), "
        "Tier 2 (YOLOv8-nano lesion localization), "
        "Tier 3 (Mobile-UNet sub-pixel foliar damage segmentation), "
        "and Crop Recommendation Intelligence (Random Forest 22-class agro-climatic matching)."
    ),
    lifespan=lifespan,
    docs_url="/docs" if settings.ENVIRONMENT != "production" or settings.DEBUG else "/docs",
    redoc_url="/redoc" if settings.ENVIRONMENT != "production" or settings.DEBUG else "/redoc"
)

# ── Security & Tracing Middleware ──────────────────────────────────────────────
@app.middleware("http")
async def request_tracing_and_security_middleware(request: Request, call_next):
    req_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
    request.state.request_id = req_id

    start_time = time.time()
    try:
        response: Response = await call_next(request)
    except Exception as exc:
        logger.error("Unhandled request exception [req_id=%s]: %s", req_id, exc, exc_info=True)
        response = JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "status": "error",
                "error_code": "INTERNAL_SERVER_ERROR",
                "message": "An unexpected error occurred during request processing.",
                "recovery_hint": "Please retry the request. System telemetry has logged this incident.",
                "request_id": req_id
            }
        )

    duration_ms = (time.time() - start_time) * 1000.0
    response.headers["X-Request-ID"] = req_id
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

    if not request.url.path.startswith(("/static", "/samples")):
        logger.info("%s %s [%d] - %.1fms (req_id=%s)",
                    request.method, request.url.path, response.status_code, duration_ms, req_id)
    return response


# ── CORS Middleware Configuration ──────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=settings.ALLOW_CREDENTIALS,
    allow_methods=["GET", "POST", "OPTIONS", "HEAD"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID"]
)

# ── Mount API v1 Router ────────────────────────────────────────────────────────
app.include_router(api_router, prefix=settings.API_V1_STR)

# ── Static File Mounts for Frontend Dashboard ──────────────────────────────────
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
    samples_dir = FRONTEND_DIR / "samples"
    if samples_dir.exists():
        app.mount("/samples", StaticFiles(directory=samples_dir), name="samples")


# ── IoT / Frontend Sensor Input Model ──────────────────────────────────────────
class SensorReading(BaseModel):
    temperature: float = Field(..., ge=-10.0, le=60.0, description="Ambient temperature in °C")
    humidity: float = Field(..., ge=0.0, le=100.0, description="Relative humidity percentage")
    soil_moisture: float = Field(..., ge=0.0, le=100.0, description="Volumetric soil moisture percentage")
    rain: int = Field(0, ge=0, le=1, description="Binary rain state (1=raining, 0=dry)")
    rainfall_mm: Optional[float] = Field(None, ge=0.0, le=500.0)
    rainfall: Optional[float] = None
    N: Optional[float] = None
    P: Optional[float] = None
    K: Optional[float] = None
    ph: Optional[float] = None


# ── Root UI & Static Endpoints ────────────────────────────────────────────────
NO_CACHE_HEADERS = {
    "Cache-Control": "no-cache, no-store, must-revalidate, max-age=0",
    "Pragma": "no-cache",
    "Expires": "0"
}

@app.get("/", summary="Dashboard Index or API Info")
async def serve_index(request: Request):
    """Serves the dashboard index.html when accessed via browser, or returns API metadata."""
    accept_header = request.headers.get("accept", "")
    index_path = FRONTEND_DIR / "index.html"
    if index_path.exists() and "text/html" in accept_header and "application/json" not in accept_header:
        return FileResponse(index_path, media_type="text/html", headers=NO_CACHE_HEADERS)
    return {
        "project": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "environment": settings.ENVIRONMENT,
        "device": inference_engine.device_name,
        "docs_url": "/docs",
        "health_url": "/health",
        "readiness_url": "/health/ready",
        "liveness_url": "/health/live",
        "models_status_url": "/models/status",
        "vision_diagnose_url": "/predict/vision",
        "crop_recommend_url": "/predict",
        "api_v1_prefix": settings.API_V1_STR
    }


@app.get("/style.css", include_in_schema=False)
async def serve_css():
    css_path = FRONTEND_DIR / "style.css"
    if css_path.exists():
        return FileResponse(css_path, media_type="text/css", headers=NO_CACHE_HEADERS)
    raise HTTPException(status_code=404, detail="style.css not found")


@app.get("/app.js", include_in_schema=False)
async def serve_js():
    js_path = FRONTEND_DIR / "app.js"
    if js_path.exists():
        return FileResponse(js_path, media_type="application/javascript", headers=NO_CACHE_HEADERS)
    raise HTTPException(status_code=404, detail="app.js not found")


# ── Root-Level Health & Status Aliases ─────────────────────────────────────────
@app.get("/health", response_model=HealthStatusResponse, summary="System Health Liveness Check")
async def root_health():
    return await health_check()


@app.get("/health/live", response_model=LivenessStatusResponse, summary="Liveness Probe Alias")
async def root_health_live():
    return await liveness_probe()


@app.get("/health/ready", response_model=ReadinessStatusResponse, summary="Readiness Probe Alias")
async def root_health_ready(response: Response):
    return await readiness_probe(response=response)


@app.get("/models/status", response_model=ModelsStatusResponse, summary="Platform Models Checkpoint Audit")
async def root_models_status():
    return await models_status()


@app.get("/models/registry", response_model=ModelRegistryResponse, summary="Platform Model Provenance & Artifact Registry")
async def root_models_registry():
    """Returns complete versioned registry of active and candidate model checkpoints."""
    return model_registry.get_registry_response(
        device=inference_engine.device_name,
        active_tier=settings.DEFAULT_TIER1_MODEL
    )


# ── Root-Level Crop Recommendation (/predict) ──────────────────────────────────
@app.post("/predict", summary="Unified Crop Recommendation and Agro-Climatic Risk")
async def predict(reading: SensorReading):
    """
    Evaluates incoming environmental parameters (from dashboard sliders or sensors)
    and returns top-3 crop recommendations and active microclimate disease alerts.
    """
    try:
        eff_rain_mm = reading.rainfall_mm if reading.rainfall_mm is not None else reading.rainfall
        rec_req = CropRecommendationRequest(
            temperature=reading.temperature,
            humidity=reading.humidity,
            soil_moisture=reading.soil_moisture,
            rain=reading.rain,
            rainfall_mm=eff_rain_mm
        )
        rec = crop_service.recommend(rec_req)

        alerts = [
            {
                "id": a.id,
                "name": a.name,
                "type": a.type,
                "severity": a.severity,
                "trigger": a.trigger,
                "symptoms": a.symptoms,
                "pesticide": a.pesticide,
                "technique": a.technique,
            }
            for a in rec.disease_alerts
        ]
        top3 = [
            {
                "crop": c.crop,
                "confidence": c.confidence_pct,
                "rank": c.rank,
                "suitability_level": c.suitability_level
            }
            for c in rec.top3_candidates
        ]

        response = {
            "status": "success",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "recommended_crop": rec.recommended_crop,
            "confidence": rec.confidence_pct,
            "top3": top3,
            "disease_alerts": alerts,
            "alert_count": len(alerts),
            "features": {
                "temperature": rec.features_used.temperature,
                "humidity": rec.features_used.humidity,
                "soil_moisture": rec.features_used.soil_moisture,
                "rain": rec.features_used.rain,
                "rainfall_mm": rec.features_used.rainfall_mm,
                "fungal_risk": rec.features_used.fungal_risk,
                "drought_risk": rec.features_used.drought_risk,
                "waterlog_risk": rec.features_used.waterlog_risk,
            },
            "agronomic_summary": rec.agronomic_summary
        }

        history.append({
            "timestamp": response["timestamp"],
            "source": "dashboard",
            "temperature": reading.temperature,
            "humidity": reading.humidity,
            "soil_moisture": reading.soil_moisture,
            "rain": reading.rain,
            "recommended_crop": rec.recommended_crop,
            "confidence": rec.confidence_pct,
            "alert_count": len(alerts),
        })

        return response
    except Exception as e:
        logger.error("Prediction failure in /predict: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


# ── ESP8266 Microcontroller Compact Endpoint (/predict/compact) ────────────────
@app.post("/predict/compact", summary="ESP8266 Compact Telemetry Ingestion")
async def predict_compact(reading: SensorReading):
    """
    Compact JSON response optimized for memory-constrained microcontrollers
    updating 16x2 LCD character displays without buffer overflow.
    """
    try:
        eff_rain_mm = reading.rainfall_mm if reading.rainfall_mm is not None else reading.rainfall
        rec_req = CropRecommendationRequest(
            temperature=reading.temperature,
            humidity=reading.humidity,
            soil_moisture=reading.soil_moisture,
            rain=reading.rain,
            rainfall_mm=eff_rain_mm
        )
        rec = crop_service.recommend(rec_req)

        top3 = rec.top3_candidates
        alerts = rec.disease_alerts

        iot_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source": "iot",
            "temperature": reading.temperature,
            "humidity": reading.humidity,
            "soil_moisture": reading.soil_moisture,
            "rain": reading.rain,
            "recommended_crop": rec.recommended_crop,
            "confidence": rec.confidence_pct,
            "alert_count": len(alerts),
        }
        history.append(iot_entry)

        global latest_iot
        latest_iot = iot_entry

        return {
            "ok": 1,
            "crop": rec.recommended_crop[:14],
            "conf": rec.confidence_pct,
            "ac": len(alerts),
            "t2": top3[1].crop[:12] if len(top3) > 1 else "",
            "c2": top3[1].confidence_pct if len(top3) > 1 else 0,
            "t3": top3[2].crop[:12] if len(top3) > 2 else "",
            "c3": top3[2].confidence_pct if len(top3) > 2 else 0,
            "alerts": [
                {"n": d.name[:18], "s": d.severity}
                for d in alerts[:4]
            ],
        }
    except Exception as e:
        logger.error("Compact prediction error: %s", e, exc_info=True)
        return {"ok": 0, "err": str(e)[:40]}


# ── Real-Time IoT Telemetry Polling (/latest & /history) ───────────────────────
@app.get("/latest", summary="Most Recent Genuine IoT Sensor Packet")
async def get_latest():
    """
    Returns the most recent physical sensor packet transmitted by an ESP8266.
    Returns HTTP 404 if no physical hardware packet has been recorded.
    """
    if latest_iot is None:
        return {
            "status": "no_data",
            "temperature": None,
            "humidity": None,
            "soil_moisture": None,
            "rain": None,
            "message": "No IoT reading received yet. Waiting for ESP8266 transmission..."
        }
    return latest_iot


@app.get("/history", summary="Telemetry & Prediction History")
async def get_history(limit: int = 20):
    """Returns the last N sensor readings and their model predictions."""
    items = list(history)[-limit:]
    items.reverse()
    return {"count": len(items), "readings": items}


# ── Pre Inference Domain Validation Endpoint (/validate/vision) ───────────────
@app.post("/validate/vision", summary="Pre Inference Botanical Domain and Suitability Gate")
async def validate_vision(
    request: Request,
    file: UploadFile = File(...),
    request_id: Optional[str] = Form(None),
    source: str = Form("browser_upload"),
    esp32_capture_id: Optional[str] = Form(None)
):
    """
    Dedicated pre inference validation gate:
    Evaluates whether an uploaded image is a valid plant leaf photograph before
    any disease classification or diagnostic inference is permitted.
    Includes source-aware technical tolerance for authenticated ESP32-CAM captures.
    """
    if isinstance(request_id, str) and request_id.strip():
        req_id = request_id.strip()
    else:
        req_id = request.headers.get("X-Request-ID") or getattr(request.state, "request_id", None) or uuid.uuid4().hex[:12]

    # Verify provenance for esp32_cam to prevent unauthenticated client spoofing
    trusted_source = "browser_upload"
    camera_meta = None
    if source == "esp32_cam" and esp32_capture_id:
        provenance = esp32_gateway.verify_and_consume_provenance(esp32_capture_id, consume=False)
        if provenance:
            trusted_source = "esp32_cam"
            camera_meta = provenance
        else:
            trusted_source = "browser_upload"
    elif source in ("browser_camera", "browser_upload"):
        trusted_source = source

    if not file or not file.filename:
        raise HTTPException(status_code=400, detail="No image file received.")

    filename = file.filename or "uploaded_leaf.jpg"
    content_type = file.content_type

    try:
        file_bytes = await file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read file: {e}")
    finally:
        await file.close()

    try:
        img_bgr, meta = validate_uploaded_image(file_bytes, filename=filename, content_type=content_type)
        if getattr(inference_engine, "model_tier2_plantdoc", None) is None and getattr(inference_engine, "model_tier2", None) is None:
            inference_engine.load_models()
        detector_candidate = getattr(inference_engine, "model_tier2_plantdoc", None) or getattr(inference_engine, "model_tier2", None)
        val_result = validate_plant_image(
            img_bgr,
            detector_model=detector_candidate,
            filename=filename,
            source=trusted_source,
            camera_metadata=camera_meta
        )
        
        return {
            "status": "valid" if val_result.is_inference_allowed else "rejected",
            "request_id": req_id,
            "filename": filename,
            "source": trusted_source,
            "validation_status": val_result.validation_status,
            "validation_reason": val_result.validation_reason,
            "validation_confidence": val_result.validation_confidence,
            "plant_presence": val_result.plant_presence,
            "leaf_presence": val_result.leaf_presence,
            "screenshot_or_document_probability": val_result.screenshot_or_document_probability,
            "image_quality": val_result.image_quality,
            "is_inference_allowed": val_result.is_inference_allowed,
            "inference_allowed": val_result.inference_allowed,
            "telemetry": val_result.telemetry
        }
    except ImageValidationError as e:
        return {
            "status": "rejected",
            "request_id": req_id,
            "filename": filename,
            "validation_status": "INVALID_NON_PLANT_IMAGE",
            "validation_reason": str(e),
            "validation_confidence": 1.0,
            "plant_presence": False,
            "leaf_presence": False,
            "screenshot_or_document_probability": 0.0,
            "image_quality": "Unreadable File",
            "is_inference_allowed": False,
            "inference_allowed": False,
            "telemetry": {}
        }


# ── Root Level Computer Vision Diagnosis (/predict/vision) ─────────────────────
@app.post("/predict/vision", summary="Authoritative 3 Tier Plant Pathology Diagnosis")
async def predict_vision(
    request: Request,
    file: Optional[UploadFile] = File(None),
    model_tier: Optional[str] = Form("server"),
    include_explainability: Optional[bool] = Form(False),
    request_id: Optional[str] = Form(None),
    force_inference: Optional[bool] = Form(False),
    source: Optional[str] = Form("browser_upload"),
    esp32_capture_id: Optional[str] = Form(None)
):
    """
    Executes the 3-Tier Computer Vision Cascade on uploaded leaf photography:
      • Tier 1: Server-Grade EfficientNetV2-S (authoritative 256x256, Focal Loss)
      • Tier 2: YOLO PlantDoc / YOLO26 spatial necrotic lesion localization
      • Tier 3: Mobile-UNet sub-pixel pathology segmentation
      • Explainability: Controlled 9-Stage Pipeline when include_explainability is true
      • Source-Aware Technical Policy: Authentic ESP32-CAM captures receive sensor-tuned thresholds
    """
    if isinstance(request_id, str) and request_id.strip():
        req_id = request_id.strip()
    else:
        req_id = request.headers.get("X-Request-ID") or getattr(request.state, "request_id", None) or uuid.uuid4().hex[:12]

    camera_metadata = None
    validated_source = source or "browser_upload"

    if esp32_capture_id:
        provenance = esp32_gateway.verify_and_consume_provenance(esp32_capture_id)
        if provenance:
            validated_source = "esp32_cam"
            camera_metadata = {
                "device_id": provenance["device_id"],
                "capture_id": esp32_capture_id,
                "capture_latency_ms": provenance.get("capture_latency_ms"),
                "camera_model": provenance.get("camera_model", "OV2640"),
                "camera_source": "ESP32-CAM · OV2640"
            }
        else:
            validated_source = "browser_upload"

    if file and file.filename:
        filename = file.filename or "uploaded_leaf.jpg"
        content_type = file.content_type

        try:
            file_bytes = await file.read()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to read file: {e}")
        finally:
            await file.close()
    elif camera_metadata and "frame_bytes" in provenance:
        file_bytes = provenance["frame_bytes"]
        filename = f"{esp32_capture_id}.jpg"
        content_type = "image/jpeg"
    else:
        raise HTTPException(status_code=400, detail="No image file or authentic ESP32 capture received.")

    try:
        resp = inference_engine.run_inference(
            file_bytes=file_bytes,
            filename=filename,
            content_type=content_type,
            model_tier=model_tier or "server",
            include_explainability=bool(include_explainability),
            request_id=req_id,
            force_inference=bool(force_inference),
            source=validated_source,
            camera_metadata=camera_metadata
        )

        if resp.status == "rejected":
            val = resp.image_validation
            is_screenshot = bool(val and val.validation_status == "INVALID_SCREENSHOT_OR_DOCUMENT")
            return {
                "response_schema_version": resp.response_schema_version,
                "status": "rejected",
                "request_id": req_id,
                "filename": filename,
                "validation_status": val.validation_status if val else "INVALID_NON_PLANT_IMAGE",
                "validation_reason": val.validation_reason if val else "Specimen rejected by domain validation.",
                "validation_confidence": val.validation_confidence if val else 1.0,
                "plant_presence": val.plant_presence if val else False,
                "leaf_presence": val.leaf_presence if val else False,
                "screenshot_or_document_detection": is_screenshot,
                "screenshot_or_document_probability": val.screenshot_or_document_probability if val else (1.0 if is_screenshot else 0.0),
                "image_quality": val.image_quality if val else "Out of Domain / Non-Plant",
                "inference_allowed": False,
                "is_inference_allowed": False,
                "image_validation": val.model_dump() if val else None,
                "warnings": resp.warnings,
                "camera_source": resp.camera_source,
                "camera_device_id": resp.camera_device_id,
                "camera_capture_id": resp.camera_capture_id,
                "camera_capture_latency_ms": resp.camera_capture_latency_ms
            }

        return {
            "response_schema_version": resp.response_schema_version,
            "status": resp.status,
            "request_id": req_id,
            "filename": filename,
            "detection_status": resp.detection_status,
            "segmentation_status": resp.segmentation_status,
            "explainability_status": resp.explainability_status,
            "short_explanation": resp.short_explanation,
            "what_to_check": resp.what_to_check,
            "model_metadata": resp.model_metadata.model_dump(),
            "image_validation": resp.image_validation.model_dump() if resp.image_validation else None,
            "image_quality": resp.image_quality.model_dump() if resp.image_quality else None,
            "uncertainty": resp.uncertainty.model_dump() if resp.uncertainty else None,
            "diagnosis": resp.diagnosis.model_dump(),
            "spatial_telemetry": resp.spatial_telemetry.model_dump(),
            "segmentation_mask_b64": resp.segmentation_mask_b64,
            "mask_raw_b64": resp.mask_raw_b64,
            "cam_heatmap_b64": resp.cam_heatmap_b64,
            "cam_overlay_b64": resp.cam_overlay_b64,
            "explainability": resp.explainability.model_dump() if resp.explainability else None,
            "advisory": {
                **resp.advisory.model_dump(),
                "immediate": resp.advisory.immediate_action,
                "treatment": resp.advisory.treatment_protocol,
                "cultural": resp.advisory.cultural_practices,
            },
            "latency_ms": {
                "tier1_ms": resp.performance_benchmark.tier1_mobilenetv2_ms,
                "tier1_model_name": resp.performance_benchmark.tier1_model_name,
                "tier2_ms": resp.performance_benchmark.tier2_yolov8n_ms,
                "tier3_ms": resp.performance_benchmark.tier3_mobile_unet_ms,
                "explainability_ms": resp.performance_benchmark.explainability_ms,
                "total_ms": resp.performance_benchmark.total_pipeline_ms,
                "device": resp.performance_benchmark.compute_device
            },
            "performance_benchmark": resp.performance_benchmark.model_dump(),
            "warnings": resp.warnings,
            "camera_source": resp.camera_source,
            "camera_device_id": resp.camera_device_id,
            "camera_capture_id": resp.camera_capture_id,
            "camera_capture_latency_ms": resp.camera_capture_latency_ms
        }
    except ImageValidationError as ive:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "status": "error",
                "error_code": "IMAGE_VALIDATION_FAILED",
                "message": ive.message,
                "recovery_hint": ive.recovery_hint,
                "request_id": req_id
            }
        )
    except RuntimeError as rte:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "error",
                "error_code": "MODEL_UNAVAILABLE",
                "message": str(rte),
                "recovery_hint": "Please verify model checkpoint files are loaded.",
                "request_id": req_id
            }
        )
    except Exception as exc:
        logger.error("Inference error in /predict/vision [req_id=%s]: %s", req_id, exc, exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "status": "error",
                "error_code": "INFERENCE_ERROR",
                "message": "Vision inference pipeline encountered an unexpected issue.",
                "recovery_hint": "Please retry the image. System logs have recorded the trace.",
                "request_id": req_id
            }
        )


@app.post("/predict/vision/explain", summary="Convenience Route: 3-Tier Diagnosis + Full 9-Stage Explainability")
async def predict_vision_explain(
    request: Request,
    file: UploadFile = File(...),
    model_tier: Optional[str] = Form("server"),
    request_id: Optional[str] = Form(None)
):
    """Executes the vision cascade and forces include_explainability=True."""
    return await predict_vision(
        request=request,
        file=file,
        model_tier=model_tier,
        include_explainability=True,
        request_id=request_id if isinstance(request_id, str) else None
    )
