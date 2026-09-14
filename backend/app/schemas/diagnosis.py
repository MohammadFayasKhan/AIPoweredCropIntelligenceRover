"""
Pydantic Schemas for Smart Plant Intelligence Inference API.
Enforces typed request and response contracts for image diagnosis, spatial telemetry,
agronomic advisory, multi-stage explainability, edge latency benchmarks, and system health/readiness.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, model_validator

class TopPrediction(BaseModel):
    class_id: str = Field(..., description="Canonical folder or taxonomy ID")
    label: str = Field(..., description="Human-readable condition label")
    confidence_pct: float = Field(..., description="Confidence percentage (0 to 100)")

class DiagnosisSummary(BaseModel):
    predicted_class: str = Field(..., description="Raw taxonomy folder identifier")
    disease_common_name: str = Field(..., description="Common disease name or healthy status")
    crop: str = Field(..., description="Plant crop species (e.g., Tomato, Corn, Grape)")
    condition_type: str = Field(..., description="Condition category (healthy, fungal, bacterial, viral)")
    confidence_pct: float = Field(..., description="Primary prediction confidence percentage")
    confidence_level: str = Field(..., description="Certainty level: HIGH, MEDIUM, or LOW_UNCERTAIN")
    is_low_confidence: bool = Field(False, description="True if confidence is below authoritative threshold (<50%)")
    uncertainty_score: float = Field(0.0, description="Normalized model prediction uncertainty score (0.0 to 1.0)")
    entropy: float = Field(0.0, description="Shannon entropy across class prediction distribution")
    top3_predictions: List[TopPrediction] = Field(default_factory=list, description="Top-3 differential diagnostic classes")
    is_infected: bool = Field(..., description="True if pathological symptoms detected, False if healthy")
    triage_stage: str = Field(..., description="Agronomic triage classification")
    segmentation_status: str = Field("available", description="Status of segmentation: 'available', 'unavailable', or 'healthy_not_applicable'")
    foliar_damage_pct: Optional[float] = Field(None, description="Sub-pixel necrotic leaf damage percentage (None if unavailable)")
    lesion_foci_count: int = Field(..., description="Total verified lesion objects identified by detector")
    lesion_foci_source: Optional[str] = Field(default="none", description="Source of lesion foci: 'yolov8n_lesions_detector', 'mobile_unet_segmentation', or 'none'")
    short_explanation: Optional[str] = Field(None, description="Clear plain-language explanation of what this diagnosis means")
    what_to_check: Optional[str] = Field(None, description="Practical observational steps to inspect nearby foliage")
    detection_status: str = Field("available", description="Status: 'available', 'unavailable', 'not_requested', or 'failed'")
    model_architecture: Optional[str] = Field(default="EfficientNetV2-S Server-Grade", description="Active classification backbone")
    model_tier: Optional[str] = Field(default="server", description="Model tier executed (server, edge, ensemble)")

class BoundingBox(BaseModel):
    label: str = Field(..., description="Detected object label")
    bbox_xyxy: List[int] = Field(..., description="Pixel coordinates [x1, y1, x2, y2]")
    confidence: float = Field(..., description="Detection confidence score (0.0 to 1.0)")
    centroid_norm: List[float] = Field(..., description="Normalized centroid coordinates [cx, cy]")
    box_type: Optional[str] = Field(default="lesion", description="leaf, spot, lesion, or canopy")
    severity: Optional[str] = Field(default="moderate", description="mild, moderate, severe, or healthy")
    color_hex: Optional[str] = Field(default=None, description="Hex color code for rendering")
    detection_id: Optional[str] = Field(default=None, description="Unique stable identifier for frontend keying and audit")
    class_id: Optional[int] = Field(default=None, description="Genuine detector class ID from model ontology")
    class_name: Optional[str] = Field(default=None, description="Genuine detector class name from model ontology")
    category_type: Optional[str] = Field(default="lesion", description="Category: 'canopy' (leaf boundary) or 'lesion' (pathological focus)")
    source_model: Optional[str] = Field(default=None, description="Model identifier that produced this detection")
    coordinate_space: Optional[str] = Field(default=None, description="Coordinate system representation (e.g. pixel_orig_WxH)")
    box_id: Optional[str] = Field(default=None, description="Legacy identifier e.g. box_1")
    x1: Optional[int] = Field(default=None, description="Top-left x pixel coordinate")
    y1: Optional[int] = Field(default=None, description="Top-left y pixel coordinate")
    x2: Optional[int] = Field(default=None, description="Bottom-right x pixel coordinate")
    y2: Optional[int] = Field(default=None, description="Bottom-right y pixel coordinate")
    center_x: Optional[float] = Field(default=None, description="Normalized horizontal center")
    center_y: Optional[float] = Field(default=None, description="Normalized vertical center")
    area_px: Optional[int] = Field(default=None, description="Box area in square pixels")

    @model_validator(mode="after")
    def populate_derived_fields(self):
        if self.bbox_xyxy and len(self.bbox_xyxy) == 4:
            if self.x1 is None: self.x1 = self.bbox_xyxy[0]
            if self.y1 is None: self.y1 = self.bbox_xyxy[1]
            if self.x2 is None: self.x2 = self.bbox_xyxy[2]
            if self.y2 is None: self.y2 = self.bbox_xyxy[3]
            if self.area_px is None:
                self.area_px = max(0, (self.bbox_xyxy[2] - self.bbox_xyxy[0]) * (self.bbox_xyxy[3] - self.bbox_xyxy[1]))
        if self.centroid_norm and len(self.centroid_norm) == 2:
            if self.center_x is None: self.center_x = self.centroid_norm[0]
            if self.center_y is None: self.center_y = self.centroid_norm[1]
        if self.box_id is None:
            self.box_id = self.detection_id or "box"
        if self.box_type == "leaf" and (self.category_type is None or self.category_type == "lesion"):
            self.category_type = "canopy"
        return self

class SpatialTelemetry(BaseModel):
    detection_engine: str = Field(..., description="Model and version utilized for spatial localization")
    taxonomy_version: Optional[str] = Field(default="YOLO26-29Class-MultiDomain-v2.5", description="Detection taxonomy version identifier")
    bounding_boxes: List[BoundingBox] = Field(default_factory=list, description="List of all localized spatial bounding boxes")
    specimen_detections: List[BoundingBox] = Field(default_factory=list, description="Specimen / leaf / canopy boundary detections")
    lesion_detections: List[BoundingBox] = Field(default_factory=list, description="Genuine pathology / lesion focus detections")
    nozzle_actuation_targets: int = Field(..., description="Number of targeted spray coordinates")
    variable_rate_dosage_multiplier: float = Field(..., description="VRA chemical dose multiplier (1.0 to 2.5)")
    raw_detection_count: Optional[int] = Field(default=0, description="Total raw detections from detector before filtering")
    post_filtering_count: Optional[int] = Field(default=0, description="Total detections preserved after threshold and NMS")
    canopy_box_count: Optional[int] = Field(default=0, description="Count of canopy/foliage boundary boxes")
    specimen_box_count: Optional[int] = Field(default=0, description="Count of specimen boundary boxes")
    lesion_box_count: Optional[int] = Field(default=0, description="Count of active disease lesion boxes")
    localization_capability: str = Field(default="specimen_boundary_only", description="Level of spatial localization provided: 'specimen_boundary_only', 'lesion_foci_only', or 'specimen_boundary_and_lesion_foci'")
    localization_notice: Optional[str] = Field(default=None, description="Honest scientific notice explaining localization capability and scope")
    raw_specimen_count: Optional[int] = Field(default=0, description="Raw candidate specimen detections")
    raw_lesion_count: Optional[int] = Field(default=0, description="Raw candidate lesion detections")
    post_nms_specimen_count: Optional[int] = Field(default=0, description="Specimen detections preserved after threshold/NMS")
    post_nms_lesion_count: Optional[int] = Field(default=0, description="Lesion detections preserved after threshold/NMS")
    lesion_foci_source: Optional[str] = Field(default="none", description="Source of lesion foci: 'yolov8n_lesions_detector', 'mobile_unet_segmentation', or 'none'")

class AgronomicAdvisory(BaseModel):
    immediate_action: str = Field(..., description="Actionable step for the farmer")
    treatment_protocol: str = Field(..., description="Prescribed chemical or biological application")
    cultural_practices: str = Field(..., description="Sanitation, irrigation, and airflow management")
    uncertainty_guidance: Optional[str] = Field(None, description="Guidance when confidence is low or ambiguous")

class LatencyBenchmark(BaseModel):
    tier1_mobilenetv2_ms: float = Field(..., description="Classifier screening latency in milliseconds")
    tier1_model_name: Optional[str] = Field(default="EfficientNet-B2 Server-Grade", description="Name of executed Tier 1 backbone")
    tier2_yolov8n_ms: float = Field(..., description="Tier 2 spatial localization latency in milliseconds")
    tier3_mobile_unet_ms: float = Field(..., description="Tier 3 sub-pixel segmentation latency in milliseconds")
    explainability_ms: float = Field(0.0, description="Explainability (Grad-CAM & feature map) latency in milliseconds")
    total_pipeline_ms: float = Field(..., description="Complete multi-tier processing time in milliseconds")
    effective_fps: float = Field(..., description="Inference frame rate throughput")
    compute_device: str = Field(..., description="Underlying acceleration hardware (MPS / CUDA GPU name / CPU)")

class ImageValidationAssessment(BaseModel):
    validation_status: str = Field(..., description="Validation state: VALID_PLANT_IMAGE, INVALID_SCREENSHOT_OR_DOCUMENT, INVALID_NON_PLANT_IMAGE, LOW_QUALITY_IMAGE, or VALIDATION_UNCERTAIN")
    validation_reason: str = Field(..., description="User facing plain language validation verdict or guidance")
    validation_confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in domain validation assessment")
    plant_presence: bool = Field(..., description="True if botanical plant foliage or tissue was detected")
    leaf_presence: bool = Field(..., description="True if genuine leaf or canopy structure is present")
    screenshot_or_document_probability: float = Field(default=0.0, ge=0.0, le=1.0, description="Estimated probability that input is a screenshot or document")
    image_quality: str = Field(..., description="Qualitative quality summary")
    is_inference_allowed: bool = Field(..., description="True only if image passed domain validation and inference proceeded")
    inference_allowed: bool = Field(default=False, description="Direct alias for is_inference_allowed")
    telemetry: Dict[str, Any] = Field(default_factory=dict, description="Raw computed domain signals")

class ImageQualityAssessment(BaseModel):
    quality_score: float = Field(..., ge=0.0, le=1.0, description="Overall image quality score (0.0 to 1.0)")
    quality_level: str = Field("Good", description="Image quality tier: 'Good', 'Acceptable', or 'Poor'")
    summary_text: Optional[str] = Field(None, description="Human-readable image quality summary")
    is_acceptable: bool = Field(..., description="True if specimen is sharp, well-lit, and contains foliar tissue")
    is_usable: bool = Field(True, description="Usability flag for diagnostic inference")
    blur_score: float = Field(..., description="Laplacian edge variance (higher = sharper, <40 indicates blur)")
    brightness_mean: float = Field(..., description="Mean pixel intensity across RGB channels (0 to 255)")
    contrast_std: float = Field(..., description="Standard deviation of pixel intensity (contrast dynamic range)")
    greenness_ratio: float = Field(..., description="Vegetation index / foliar color ratio indicating plant tissue presence")
    quality_issues: List[str] = Field(default_factory=list, description="Specific quality defects identified during preflight")
    warnings: List[str] = Field(default_factory=list, description="Specific warning notices for the user")
    recapture_guidance: Optional[str] = Field(None, description="Actionable photographer guidance if quality is degraded")
    recommendation: Optional[str] = Field(None, description="Actionable photographer guidance alias")


class UncertaintyMetrics(BaseModel):
    prediction_margin: float = Field(..., description="Difference between Top-1 and Top-2 class probabilities (0.0 to 1.0)")
    entropy_nats: float = Field(..., description="Shannon entropy across softmax prediction distribution")
    normalized_uncertainty: float = Field(..., description="Entropy normalized by maximum possible entropy ln(N)")
    ood_status: str = Field("IN_DISTRIBUTION", description="Out-of-distribution state: IN_DISTRIBUTION, BORDERLINE, or OUT_OF_DISTRIBUTION")
    is_low_confidence: bool = Field(False, description="True if top-1 confidence is below threshold (<50%) or OOD")

class ExplainabilityStage(BaseModel):
    stage_number: int = Field(..., description="Step index (1 to 9)")
    title: str = Field(..., description="Human-readable title of the inference stage")
    technical_name: str = Field(..., description="Technical pipeline phase identifier")
    image_b64: Optional[str] = Field(None, description="Base64 encoded JPEG/PNG visual artifact")
    explanation: str = Field(..., description="Primary narrative description of what this stage processed and discovered")
    simple_explanation: Optional[str] = Field(None, description="Plain-language botanical explanation for growers and students")
    technical_explanation: Optional[str] = Field(None, description="Detailed computational and algorithmic breakdown for developers")
    metrics: Dict[str, Any] = Field(default_factory=dict, description="Stage-specific telemetry metadata")

class ExplainabilityPipeline(BaseModel):
    stages: List[ExplainabilityStage] = Field(default_factory=list, description="Step-by-step pipeline stages")
    synthesis: str = Field(..., description="Grounded agronomic decision synthesis explaining final verdict")

class ModelMetadata(BaseModel):
    model_name: str = Field(..., description="Authoritative active model identifier")
    model_version: str = Field(..., description="Active checkpoint release tag")
    architecture: str = Field("EfficientNetV2-S", description="Neural network architecture family")
    taxonomy_version: str = Field("38-class-canonical", description="Taxonomy release identifier")
    classification_taxonomy_version: str = Field("PlantVillage-38Class-v2.0", description="Independent classification taxonomy identifier")
    detection_taxonomy_version: str = Field("PlantDoc-29Class-YOLO-v1.0", description="Independent object detection taxonomy identifier")
    segmentation_taxonomy_version: str = Field("FoliarLesions-Binary-v1.0", description="Independent segmentation mask taxonomy identifier")
    input_resolution: str = Field("256x256", description="Model input image resolution (HxW)")
    is_server_authoritative: bool = Field(True, description="True if running authoritative server-grade tier")
    device: Optional[str] = Field(None, description="Actual runtime execution device")
    test_top1_accuracy: Optional[float] = Field(0.9513, description="Verified benchmark Test Top-1 Accuracy")
    macro_f1: Optional[float] = Field(0.9354, description="Verified benchmark Test Macro F1 Score")
    expected_calibration_error: Optional[float] = Field(0.0803, description="Verified benchmark ECE")
    sha256_hash: Optional[str] = Field(None, description="Verified checkpoint SHA-256 fingerprint")

class ModelRegistryEntry(BaseModel):
    model_id: str = Field(..., description="Unique model identifier in registry")
    name: str = Field(..., description="Human-readable model title")
    architecture: str = Field(..., description="Neural architecture family")
    task: str = Field(..., description="Vision task: classification, detection, segmentation, crop_recommendation")
    tier: str = Field(..., description="Deployment tier: server, edge, ensemble, benchmark")
    version: str = Field(..., description="Model release version tag")
    checkpoint_path: str = Field(..., description="Safe filename of model checkpoint")
    checkpoint_size_mb: float = Field(..., description="File size in megabytes")
    input_resolution: str = Field(..., description="Model input image resolution")
    taxonomy_version: str = Field(..., description="Supported taxonomy release version")
    is_ready: bool = Field(..., description="True if checkpoint exists and loaded successfully")
    device: Optional[str] = Field(None, description="Compute device running this model")
    sha256_hash: Optional[str] = Field(None, description="SHA-256 fingerprint of the checkpoint file")

class ModelRegistryResponse(BaseModel):
    status: str = Field("ready", description="Overall registry status")
    total_models: int = Field(..., description="Total models cataloged")
    models_ready: int = Field(..., description="Models initialized and ready for inference")
    active_classifier_tier: str = Field(..., description="Currently active classification tier")
    models: List[ModelRegistryEntry] = Field(default_factory=list, description="Registered model entries")

class DiagnosisResponse(BaseModel):
    response_schema_version: str = Field("1.0", description="Canonical response schema version")
    status: str = Field("success", description="Response status indicator")
    pipeline_version: str = Field("CV-06-Universal-MultiCrop-v2.2", description="Inference pipeline release version")
    sample_id: str = Field(..., description="Client uploaded image identifier")
    request_id: Optional[str] = Field(None, description="Unique trace identifier for request auditing")
    timestamp: str = Field(..., description="ISO 8601 evaluation timestamp")
    model_metadata: ModelMetadata = Field(..., description="Model versioning and checkpoint metadata")
    image_validation: Optional[ImageValidationAssessment] = Field(None, description="Pre-inference domain validation and rejection assessment")
    image_quality: Optional[ImageQualityAssessment] = Field(None, description="Pre-inference image quality evaluation")
    uncertainty: Optional[UncertaintyMetrics] = Field(None, description="Normalized entropy, margin, and OOD assessment")
    diagnosis: Optional[DiagnosisSummary] = Field(None, description="Complete pathology screening and triage summary (None if rejected)")
    spatial_telemetry: Optional[SpatialTelemetry] = Field(None, description="Spatial bounding boxes and nozzle coordinates (None if rejected)")
    detection_status: str = Field("available", description="Status of detection: 'available', 'unavailable', 'not_requested', or 'failed'")
    segmentation_status: str = Field("available", description="Status of segmentation: 'available', 'unavailable', 'not_requested', 'failed', or 'healthy_not_applicable'")
    explainability_status: str = Field("available", description="Status of explainability: 'available', 'unavailable', 'not_requested', or 'failed'")
    segmentation_mask_b64: Optional[str] = Field(None, description="Base64 PNG translucent foliar lesion overlay")
    mask_raw_b64: Optional[str] = Field(None, description="Base64 PNG colorized discrete foliar segmentation mask")
    cam_heatmap_b64: Optional[str] = Field(None, description="Base64 PNG standalone Grad-CAM saliency heatmap")
    cam_overlay_b64: Optional[str] = Field(None, description="Base64 PNG Grad-CAM saliency blended with specimen image")
    explainability: Optional[ExplainabilityPipeline] = Field(None, description="Full 9-stage inference explainability suite")
    advisory: Optional[AgronomicAdvisory] = Field(None, description="Actionable agronomic guidance (None if rejected)")
    performance_benchmark: LatencyBenchmark = Field(..., description="Hardware profiling telemetry")
    modalities_used: List[str] = Field(default_factory=lambda: ["image"], description="List of validated modalities utilized in diagnosis (e.g. ['image'], ['image', 'environmental_context'])")
    multimodal_context: Optional[Dict[str, Any]] = Field(None, description="Paired agronomic or environmental context")
    short_explanation: Optional[str] = Field(None, description="Clear plain-language explanation of what this diagnosis means")
    what_to_check: Optional[str] = Field(None, description="Practical observational steps to inspect nearby foliage")
    warnings: List[str] = Field(default_factory=list, description="Operational warnings or degraded component notices")
    camera_source: Optional[str] = Field(None, description="Provenance source of image: e.g. 'ESP32-CAM · OV2640', 'Browser Upload', 'Device Camera'")
    camera_device_id: Optional[str] = Field(None, description="Authenticated hardware device ID (if captured via ESP32-CAM)")
    camera_capture_id: Optional[str] = Field(None, description="Verified capture transaction ID")
    camera_capture_latency_ms: Optional[float] = Field(None, description="Hardware frame acquisition latency in milliseconds")
    environmental_context: Optional[Dict[str, Any]] = Field(None, description="Paired live field sensor telemetry snapshot (DHT11, Soil, Rain, Water, GPS)")
    disease_knowledge: Optional[Dict[str, Any]] = Field(None, description="Enriched disease knowledge base entry with environmental favorability evaluation")


class ErrorResponse(BaseModel):
    status: str = Field("error", description="Error status indicator")
    error_code: str = Field(..., description="Machine-readable error classification")
    message: str = Field(..., description="Human-readable error description")
    recovery_hint: str = Field(..., description="Actionable suggestion for user recovery")
    request_id: Optional[str] = Field(None, description="Unique trace identifier for request auditing")
    timestamp: Optional[str] = Field(None, description="ISO 8601 error timestamp")

class HealthStatusResponse(BaseModel):
    status: str = Field("healthy", description="API operational health status")
    version: str = Field(..., description="API version")
    environment: str = Field(..., description="Deployment environment")
    device: str = Field(..., description="Active compute device")

class LivenessStatusResponse(BaseModel):
    status: str = Field("alive", description="Process liveness check")
    uptime_seconds: float = Field(..., description="Seconds since process startup")
    version: str = Field(..., description="Application version")

class ComponentReadiness(BaseModel):
    ready: bool = Field(..., description="Component availability flag")
    name: str = Field(..., description="Human-readable component name")
    checkpoint: Optional[str] = Field(None, description="Loaded checkpoint filename")
    details: Optional[str] = Field(None, description="Component status summary")

class ReadinessStatusResponse(BaseModel):
    status: str = Field(..., description="'ready' when core models loaded, 'degraded' or 'unready'")
    is_ready: bool = Field(..., description="True if primary classification service is usable")
    primary_classifier_ready: bool = Field(..., description="Authoritative Tier 1 classifier state")
    detector_ready: bool = Field(..., description="YOLOv8 spatial detector state")
    segmenter_ready: bool = Field(..., description="Mobile-UNet segmenter state")
    recommender_ready: bool = Field(..., description="Random Forest crop recommender state")
    device: str = Field(..., description="Active compute hardware")
    components: Dict[str, ComponentReadiness] = Field(default_factory=dict)

class ModelInfo(BaseModel):
    name: str
    tier: str
    checkpoint: str
    size_mb: float
    status: str
    description: str

class ModelsStatusResponse(BaseModel):
    status: str
    total_models: int
    models_ready: int
    production_models_ready: int = 3
    device: str
    taxonomy_classes: int
    models: List[ModelInfo]
    computer_vision: Optional[dict] = None
    crop_recommendation: Optional[dict] = None
