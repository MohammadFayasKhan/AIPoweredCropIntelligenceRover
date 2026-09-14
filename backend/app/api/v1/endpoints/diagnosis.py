"""
Image-Based Plant Intelligence Inference Endpoint.
Accepts multipart leaf photograph uploads, coordinates server-side validation,
triggers hierarchical ML vision cascade, and returns typed agronomic diagnosis.
"""

from typing import Optional
import uuid
import logging
from fastapi import APIRouter, Request, UploadFile, File, Form, HTTPException, status
from fastapi.responses import JSONResponse

logger = logging.getLogger("smartcropvision.diagnosis")

from backend.app.services.inference_service import inference_engine
from backend.app.services.esp32_gateway import esp32_gateway
from backend.app.utils.image_processing import ImageValidationError
from backend.app.schemas.diagnosis import DiagnosisResponse, ErrorResponse

router = APIRouter()


@router.post(
    "/vision/diagnose",
    response_model=DiagnosisResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid image format, corrupted stream, or size violation"},
        503: {"model": ErrorResponse, "description": "Model weights or inference engine currently unavailable"},
        500: {"model": ErrorResponse, "description": "Internal model execution or inference failure"}
    },
    summary="Diagnose Plant Leaf Image",
    description="Upload a plant leaf photograph or provide an authenticated ESP32-CAM capture ID to execute the 3-Tier SmartCropVision pipeline."
)
async def diagnose_leaf_image(
    request: Request,
    file: Optional[UploadFile] = File(None, description="Plant leaf photograph (JPEG, PNG, or WebP up to 15 MB)"),
    source: Optional[str] = Form("browser_upload", description="Source: 'browser_upload', 'browser_camera', or 'esp32_cam'"),
    esp32_capture_id: Optional[str] = Form(None, description="Authentic capture ID from ESP32-CAM hardware capture"),
    model_tier: Optional[str] = Form("server", description="Vision model tier: 'server', 'edge', or 'ensemble'"),
    include_explainability: bool = Form(False, description="Whether to compute the full 9-stage explainability suite"),
    crop_context: Optional[str] = Form(None, description="Optional crop hint (e.g., Tomato, Corn, Grape)"),
    temperature_c: Optional[float] = Form(None, description="Optional ambient temperature in Celsius"),
    humidity_pct: Optional[float] = Form(None, description="Optional relative humidity percentage"),
    symptom_notes: Optional[str] = Form(None, description="Optional grower symptom description"),
    attach_iot_context: Optional[bool] = Form(False, description="Whether to attach live field IoT telemetry to multimodal context")
) -> DiagnosisResponse:
    """
    Authoritative plant diagnostic pipeline:
    1. Validates upload stream, magic bytes, dimensions, and MIME format.
    2. Runs Tier 1 classification: Server-grade EfficientNetV2-S (default), Edge MobileNetV2, or Ensemble.
    3. Runs Tier 2 YOLO26/YOLO11 lesion localization if infected.
    4. Runs Tier 3 Mobile-UNet sub-pixel segmentation if infected.
    5. Applies source-aware quality policy for authentic ESP32-CAM captures while maintaining strict semantic plant validation.
    6. Incorporates optional paired multimodal context with explicit modality traceability.
    7. Formulates grounded agronomic action advisory and returns payload.
    """
    request_id = getattr(request.state, "request_id", None) or uuid.uuid4().hex[:12]
    camera_metadata = None
    validated_source = source or "browser_upload"

    # Authenticate ESP32 capture provenance if claimed
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
            # Stale or unverified capture ID - do not permit esp32_cam quality policy bypass
            validated_source = "browser_upload"

    if file is not None and file.filename:
        filename = file.filename or "uploaded_leaf.jpg"
        content_type = file.content_type

        try:
            file_bytes = await file.read()
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "status": "error",
                    "error_code": "FILE_READ_ERROR",
                    "message": f"Failed to read image stream: {str(e)}",
                    "recovery_hint": "Ensure the image file is not corrupted and try re-uploading.",
                    "request_id": request_id
                }
            )
        finally:
            await file.close()
    elif camera_metadata and "frame_bytes" in provenance:
        file_bytes = provenance["frame_bytes"]
        filename = f"{esp32_capture_id}.jpg"
        content_type = "image/jpeg"
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "status": "error",
                "error_code": "NO_IMAGE_PROVIDED",
                "message": "Neither file upload nor authentic ESP32 capture ID was provided.",
                "recovery_hint": "Please choose an image file or capture a frame from ESP32 CAM.",
                "request_id": request_id
            }
        )

    # Enrich multimodal context with live field sensors if explicitly requested or if authentic esp32_cam capture
    if attach_iot_context or (validated_source == "esp32_cam" and (temperature_c is None or humidity_pct is None)):
        try:
            from backend.app.api.v1.endpoints.iot import _latest_observation
            if _latest_observation is not None:
                s = _latest_observation.sensor_telemetry
                if temperature_c is None:
                    temperature_c = s.temperature_c
                if humidity_pct is None:
                    humidity_pct = s.humidity_pct
        except Exception:
            pass

    # Build optional multimodal context dict if provided
    multimodal_context = None
    if any([crop_context, temperature_c is not None, humidity_pct is not None, symptom_notes]):
        multimodal_context = {
            k: v for k, v in {
                "crop_context": crop_context,
                "temperature_c": temperature_c,
                "humidity_pct": humidity_pct,
                "symptom_notes": symptom_notes
            }.items() if v is not None and v != ""
        }

    try:
        response = inference_engine.run_inference(
            file_bytes=file_bytes,
            filename=filename,
            content_type=content_type,
            model_tier=model_tier or "server",
            include_explainability=include_explainability,
            request_id=request_id,
            multimodal_context=multimodal_context,
            source=validated_source,
            camera_metadata=camera_metadata
        )

        # Synchronize diagnosis into unified field observation & enrich with disease knowledge
        try:
            from backend.app.api.v1.endpoints.iot import _latest_observation, ws_manager
            from backend.app.services.crop_service import crop_service

            if _latest_observation is not None:
                s = _latest_observation.sensor_telemetry
                g = _latest_observation.gps
                response.environmental_context = {
                    "temperature_c": s.temperature_c,
                    "humidity_pct": s.humidity_pct,
                    "soil_moisture_pct": s.soil_moisture_pct,
                    "soil_moisture_raw": s.soil_moisture_raw,
                    "rain_detected": s.rain_detected,
                    "water_level_pct": s.water_level_pct,
                    "gps_latitude": g.latitude if (g.is_valid or g.latitude is not None) else None,
                    "gps_longitude": g.longitude if (g.is_valid or g.longitude is not None) else None,
                    "gps_fix": g.is_valid,
                    "observation_id": _latest_observation.observation_id,
                    "timestamp": _latest_observation.timestamp,
                    "freshness": "LIVE"
                }

                # Query disease knowledge engine with vision diagnosis + live sensors
                diag = response.diagnosis
                disease_name = diag.disease_common_name if diag else "Healthy"
                dk = crop_service.get_disease_knowledge(
                    disease_name=disease_name,
                    temperature=s.temperature_c,
                    humidity=s.humidity_pct,
                    rain=s.rain_detected,
                    soil_moisture=s.soil_moisture_pct
                )
                if dk:
                    response.disease_knowledge = dk

                # Synchronize vision result into unified field observation
                _latest_observation.crop_vision = {
                    "crop_name": diag.crop if diag else "Unknown",
                    "disease_name": diag.disease_common_name if diag else "Unknown",
                    "confidence_pct": diag.confidence_pct if diag else 0.0,
                    "is_healthy": not diag.is_infected if diag else True,
                    "severity_level": diag.severity_level if diag else "Low",
                    "timestamp": response.timestamp
                }
        except Exception as e:
            logger.warning(f"Error enriching diagnosis with field context: {e}")

        return response
    except ImageValidationError as ive:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "status": "error",
                "error_code": "IMAGE_VALIDATION_FAILED",
                "message": ive.message,
                "recovery_hint": ive.recovery_hint,
                "request_id": request_id
            }
        )
    except RuntimeError as rte:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "error",
                "error_code": "MODEL_UNAVAILABLE",
                "message": str(rte),
                "recovery_hint": "Please verify model checkpoint files are mounted in the model directory.",
                "request_id": request_id
            }
        )
    except Exception as exc:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "status": "error",
                "error_code": "INFERENCE_PIPELINE_ERROR",
                "message": "An unexpected error occurred during neural inference execution.",
                "recovery_hint": "Please verify image integrity and retry. System logs contain the trace.",
                "request_id": request_id
            }
        )


@router.post(
    "/vision/explain",
    response_model=DiagnosisResponse,
    summary="Diagnose Plant Leaf Image with Full Explainability Suite",
    description="Upload a plant leaf photograph to execute the 3-Tier cascade and generate all 9 explainability stages."
)
async def explain_leaf_image(
    request: Request,
    file: UploadFile = File(..., description="Plant leaf photograph"),
    model_tier: Optional[str] = Form("server", description="Vision model tier"),
    crop_context: Optional[str] = Form(None, description="Optional crop hint"),
    temperature_c: Optional[float] = Form(None, description="Optional temperature in Celsius"),
    humidity_pct: Optional[float] = Form(None, description="Optional humidity percentage"),
    symptom_notes: Optional[str] = Form(None, description="Optional symptom notes")
) -> DiagnosisResponse:
    """Convenience endpoint that explicitly sets include_explainability=True."""
    return await diagnose_leaf_image(
        request=request,
        file=file,
        model_tier=model_tier,
        include_explainability=True,
        crop_context=crop_context,
        temperature_c=temperature_c,
        humidity_pct=humidity_pct,
        symptom_notes=symptom_notes
    )
