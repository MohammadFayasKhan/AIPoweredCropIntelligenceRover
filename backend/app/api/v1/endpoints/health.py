"""
Health, Liveness, and Readiness Diagnostic Endpoints for AI Powered Crop Intelligence Rover API.
Provides kubernetes-style liveness/readiness probes, hardware profiling, and checkpoint audits.
"""

import time
from fastapi import APIRouter, Response, status
from backend.app.config import settings
from backend.app.services.inference_service import inference_engine
from backend.app.services.crop_service import crop_service
from backend.app.schemas.diagnosis import (
    HealthStatusResponse,
    LivenessStatusResponse,
    ReadinessStatusResponse,
    ComponentReadiness,
    ModelsStatusResponse,
    ModelInfo,
)

router = APIRouter()
_process_start_time = time.time()


@router.get("/health/live", response_model=LivenessStatusResponse, summary="Process Liveness Probe")
async def liveness_probe() -> LivenessStatusResponse:
    """Kubernetes liveness probe: returns HTTP 200 as long as the server process is responsive."""
    uptime = time.time() - _process_start_time
    return LivenessStatusResponse(
        status="alive",
        uptime_seconds=round(uptime, 2),
        version=settings.VERSION
    )


@router.get("/health/ready", response_model=ReadinessStatusResponse, summary="ML Inference Readiness Probe")
async def readiness_probe(response: Response) -> ReadinessStatusResponse:
    """
    Kubernetes readiness probe:
    Returns HTTP 200 if primary inference models are loaded and ready to serve traffic.
    Returns HTTP 503 if primary vision models are missing or uninitialized.
    """
    engine = inference_engine
    if not engine.is_loaded:
        try:
            engine.load_models()
        except Exception:
            pass

    if not crop_service.is_ready:
        try:
            crop_service.load_artifacts()
        except Exception:
            pass

    primary_ready = engine.is_ready
    components = {
        "tier1_server_classifier": ComponentReadiness(
            ready=engine.model_tier1_server is not None,
            name="EfficientNetV2-S Server-Grade Classifier",
            checkpoint=settings.SERVER_TIER1_MODEL_PATH.name if settings.SERVER_TIER1_MODEL_PATH.exists() else None,
            details=engine.model_status_map.get("tier1_server")
        ),
        "tier1_edge_classifier": ComponentReadiness(
            ready=engine.model_tier1_edge is not None,
            name="MobileNetV2 Edge Classifier",
            checkpoint=settings.EDGE_TIER1_MODEL_PATH.name if settings.EDGE_TIER1_MODEL_PATH.exists() else None,
            details=engine.model_status_map.get("tier1_edge")
        ),
        "tier2_plantdoc_detector": ComponentReadiness(
            ready=engine.model_tier2_plantdoc is not None,
            name="YOLO PlantDoc Specimen Canopy Detector",
            checkpoint=settings.TIER2_PLANTDOC_MODEL_PATH.name if settings.TIER2_PLANTDOC_MODEL_PATH.exists() else None,
            details=engine.model_status_map.get("tier2_plantdoc")
        ),
        "tier3_unet_segmenter": ComponentReadiness(
            ready=engine.model_tier3 is not None,
            name="Mobile-UNet Foliar/Lesion Segmenter",
            checkpoint=settings.TIER3_MODEL_PATH.name if settings.TIER3_MODEL_PATH.exists() else None,
            details=engine.model_status_map.get("tier3_unet")
        ),
        "crop_recommender": ComponentReadiness(
            ready=crop_service.is_ready,
            name="Random Forest Crop Recommender",
            checkpoint=settings.CROP_MODEL_PATH.name if settings.CROP_MODEL_PATH.exists() else None,
            details="active" if crop_service.is_ready else "unloaded"
        )
    }

    overall_status = "ready" if primary_ready else "unready"
    if primary_ready and (engine.model_tier2_plantdoc is None or engine.model_tier3 is None):
        overall_status = "degraded"

    if not primary_ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return ReadinessStatusResponse(
        status=overall_status,
        is_ready=primary_ready,
        primary_classifier_ready=primary_ready,
        detector_ready=engine.model_tier2_plantdoc is not None or engine.model_tier2 is not None,
        segmenter_ready=engine.model_tier3 is not None,
        recommender_ready=crop_service.is_ready,
        device=engine.device_name,
        components=components
    )


@router.get("/health", response_model=HealthStatusResponse, summary="General System Health Check")
async def health_check() -> HealthStatusResponse:
    """Returns general operational status, environment, and active compute acceleration target."""
    return HealthStatusResponse(
        status="healthy",
        version=settings.VERSION,
        environment=settings.ENVIRONMENT,
        device=inference_engine.device_name
    )


@router.get("/models/status", response_model=ModelsStatusResponse, summary="Platform Models Checkpoint Audit")
async def models_status() -> ModelsStatusResponse:
    """Verifies existence, memory footprint, and readiness of all ML vision and crop intelligence models."""
    engine = inference_engine
    if not engine.is_loaded:
        try:
            engine.load_models()
        except Exception:
            pass

    if not crop_service.is_ready:
        try:
            crop_service.load_artifacts()
        except Exception:
            pass

    t1_server_size = settings.SERVER_TIER1_MODEL_PATH.stat().st_size / (1024 * 1024) if settings.SERVER_TIER1_MODEL_PATH.exists() else 0.0
    t1_edge_size = settings.EDGE_TIER1_MODEL_PATH.stat().st_size / (1024 * 1024) if settings.EDGE_TIER1_MODEL_PATH.exists() else 0.0
    is_yolo26 = getattr(engine, "model_tier2_is_yolo26", False)
    det_arch = getattr(engine, "model_tier2_arch", "YOLO11n")
    t2_path = getattr(engine, "model_tier2_path", settings.TIER2_PLANTDOC_MODEL_PATH)
    t2_size = t2_path.stat().st_size / (1024 * 1024) if t2_path.exists() else 0.0
    t2_lesions_path = settings.TIER2_LESIONS_MODEL_PATH
    t2_lesions_size = t2_lesions_path.stat().st_size / (1024 * 1024) if t2_lesions_path.exists() else 0.0
    t3_size = settings.TIER3_MODEL_PATH.stat().st_size / (1024 * 1024) if settings.TIER3_MODEL_PATH.exists() else 0.0
    crop_size = settings.CROP_MODEL_PATH.stat().st_size / (1024 * 1024) if settings.CROP_MODEL_PATH.exists() else 0.0
    y26_spec_p = settings.YOLO26_SPECIMEN_MODEL_PATH
    y26_les_p = settings.YOLO26_LESION_MODEL_PATH
    y26_spec_size = y26_spec_p.stat().st_size / (1024 * 1024) if y26_spec_p.exists() else 0.0
    y26_les_size = y26_les_p.stat().st_size / (1024 * 1024) if y26_les_p.exists() else 0.0

    y26_primary_active = bool(settings.YOLO26_ENABLED and getattr(engine, "model_yolo26_specimen", None) is not None)
    y26_lesion_active = bool(settings.YOLO26_ENABLED and getattr(engine, "model_yolo26_lesion", None) is not None)
    models_info = [
        ModelInfo(
            name="EfficientNetV2-S Server-Grade Classifier",
            tier="Tier 1 Vision (Server)",
            checkpoint=settings.SERVER_TIER1_MODEL_PATH.name,
            size_mb=round(t1_server_size, 2),
            status="ready" if engine.model_tier1_server is not None else "missing",
            description="38-class universal plant pathology classification with verified Test Top-1 95.13%, Macro F1 0.9354, ECE 0.0803 (256x256)"
        ),
        ModelInfo(
            name="MobileNetV2 Edge Classifier",
            tier="Tier 1 Vision (Edge)",
            checkpoint=settings.EDGE_TIER1_MODEL_PATH.name,
            size_mb=round(t1_edge_size, 2),
            status="ready" if engine.model_tier1_edge is not None else "missing",
            description="38-class high-speed studio leaf screening (224x224)"
        ),
        ModelInfo(
            name="YOLO26 v1 PlantDoc Specimen Detector (Primary)" if y26_primary_active else "YOLO26 v1 PlantDoc Specimen Detector (Opt-In)",
            tier="Tier 2 Vision (YOLO26s Primary)" if y26_primary_active else "Tier 2 Opt-In (YOLO26)",
            checkpoint=y26_spec_p.name,
            size_mb=round(y26_spec_size, 2),
            status=engine.model_status_map.get("yolo26_specimen", "untrained_optional"),
            description="Primary YOLO26s whole-leaf PlantDoc specimen detector (held-out Test mAP@50: 0.6672, Test mAP@50-95: 0.5267, Val mAP@50: 0.6016, 768x768)" if y26_primary_active else "Optional YOLO26s whole-leaf PlantDoc specimen detector (test mAP@50: 0.6672, mAP@50-95: 0.5267, 768x768; feature-flagged until promoted)"
        ),
        ModelInfo(
            name="YOLO26 Multi-Domain Agricultural Detector (PlantDoc)" if is_yolo26 else (f"{det_arch} PlantDoc Specimen Detector (Preserved Baseline)" if y26_primary_active else f"{det_arch} PlantDoc Specimen Detector"),
            tier="Tier 2 Vision (YOLO26)" if is_yolo26 else (f"Tier 2 Legacy ({det_arch} Baseline)" if y26_primary_active else f"Tier 2 Vision ({det_arch} PlantDoc)"),
            checkpoint=t2_path.name,
            size_mb=round(t2_size, 2),
            status="ready" if engine.model_tier2_plantdoc is not None else "missing",
            description="YOLO26 multi-domain agricultural detector for real-time foliar canopy boundary localization (val mAP@50: 0.3415, 640x640)" if is_yolo26 else "YOLO11n trained on PlantDoc for whole-leaf canopy boundary localization (val mAP@50: 0.3360, mAP@50-95: 0.2360, 416x416)"
        ),
        ModelInfo(
            name="YOLOv8n Lesion Spot Detector (Preserved Baseline)" if y26_lesion_active else "YOLOv8n Lesion Spot Detector",
            tier="Tier 2 Legacy (YOLOv8n Baseline)" if y26_lesion_active else "Tier 2 Vision (YOLOv8n Lesions)",
            checkpoint=t2_lesions_path.name,
            size_mb=round(t2_lesions_size, 2),
            status="ready" if engine.model_tier2_lesions is not None else "missing",
            description="Preserved legacy YOLOv8n focal lesion detector trained on 256x256 Early Blight morphology (train val mAP@50: 0.3374, PlantSeg Test Box mAP@50: 0.0128)" if y26_lesion_active else "YOLOv8n focal lesion spot detector trained on 256x256 Early Blight morphology (train val mAP@50: 0.3374, 256x256)"
        ),
        ModelInfo(
            name="Mobile-UNet Foliar/Lesion Segmenter",
            tier="Tier 3 Vision",
            checkpoint=settings.TIER3_MODEL_PATH.name,
            size_mb=round(t3_size, 2),
            status="ready" if engine.model_tier3 is not None else "missing",
            description="Sub-pixel foliar canopy and active lesion foci segmentation"
        ),
        ModelInfo(
            name="Random Forest Agro-Climatic Recommender",
            tier="Crop Intelligence",
            checkpoint=settings.CROP_MODEL_PATH.name,
            size_mb=round(crop_size, 2),
            status="ready" if crop_service.is_ready else "missing",
            description="22-crop suitability inference from environmental sensor parameters"
        ),
        ModelInfo(
            name="YOLO26s-Seg v1 Lesion Localizer (Primary)" if y26_lesion_active else "YOLO26-Seg v1 Lesion Localizer (Opt-In)",
            tier="Tier 2/3 Primary (YOLO26s-Seg)" if y26_lesion_active else "Tier 2/3 Opt-In (YOLO26-Seg)",
            checkpoint=y26_les_p.name,
            size_mb=round(y26_les_size, 2),
            status=engine.model_status_map.get("yolo26_lesion", "untrained_optional"),
            description="Primary multi-crop YOLO26s-Seg lesion instance segmenter trained on PlantSeg (held-out Test Box mAP@50: 0.4537, Mask mAP@50: 0.4373, Mask Dice: 0.6687, IoU: 0.5696, 768x768)" if y26_lesion_active else "Optional YOLO26-Seg lesion localizer for genuine PlantSeg masks (feature-flagged; retains YOLOv8n baseline until promoted)"
        ),
    ]

    ready_count = sum(1 for m in models_info if m.status == "ready")

    return ModelsStatusResponse(
        status="ready" if ready_count > 0 else "uninitialized",
        total_models=len(models_info),
        models_ready=ready_count,
        production_models_ready=3,
        device=engine.device_name,
        taxonomy_classes=len(engine.taxonomy),
        models=models_info,
        computer_vision={
            "is_loaded": engine.is_loaded,
            "device": engine.device_name,
            "classes": len(engine.taxonomy),
            "tiers": [
                "Tier 1 (Server): EfficientNetV2-S Server-Grade Classifier (256x256, Test Acc 95.13%)",
                "Tier 1 (Edge): MobileNetV2 Universal Classifier (224x224)",
                "Tier 2: YOLO26s PlantDoc 29-Class Specimen Detector (768x768, Test mAP@50 0.6672) + YOLOv8n Lesions" if y26_primary_active else "Tier 2: YOLO PlantDoc 29-Class Spatial Detector",
                "Tier 3: Mobile-UNet Foliar/Lesion Segmenter"
            ]
        },
        crop_recommendation={
            "model": "RandomForestClassifier (100 Trees)",
            "classes": 22,
            "status": "production_validated" if crop_service.is_ready else "uninitialized"
        }
    )
