# AI Powered Crop Intelligence Rover REST & WebSocket API Specification

## 1. Gateway Overview

The AI Powered Crop Intelligence Rover backend provides REST endpoints and telemetry streams built with FastAPI. All inference endpoints operate on the server side and support request tracking via unique identifiers (`request_id`).

* **Base URL**: `http://localhost:8000`
* **API Version**: `v1` (`/api/v1`)
* **Interactive Documentation**: `http://localhost:8000/docs` (OpenAPI Swagger UI)

---

## 2. Health & Model Readiness Endpoints

### 2.1 General System Health
* **Method & Route**: `GET /health`
* **Description**: Returns general process liveness, version, and active compute acceleration device.
* **Sample Response**:
```json
{
  "status": "healthy",
  "version": "2.2.0",
  "environment": "development",
  "device": "MPS"
}
```

### 2.2 Deep Readiness Audit
* **Method & Route**: `GET /health/ready`
* **Description**: Audits readiness of all vision and crop recommendation models in memory. Returns HTTP 503 if primary models are unready.
* **Sample Response**:
```json
{
  "status": "ready",
  "is_ready": true,
  "primary_classifier_ready": true,
  "detector_ready": true,
  "segmenter_ready": true,
  "recommender_ready": true,
  "device": "MPS",
  "components": {
    "tier1_server_classifier": {"ready": true, "name": "EfficientNetV2 S", "checkpoint": "efficientnetv2_s_best.pt"},
    "tier2_plantdoc_detector": {"ready": true, "name": "YOLO PlantDoc", "checkpoint": "yolo_plantdoc_best.pt"},
    "tier2_lesion_detector": {"ready": true, "name": "YOLOv8n Lesions", "checkpoint": "yolov8n_lesions_best.pt"},
    "tier3_unet_segmenter": {"ready": true, "name": "Mobile UNet", "checkpoint": "mobile_unet_best.pt"},
    "crop_recommender": {"ready": true, "name": "Random Forest", "checkpoint": "crop_model.pkl"}
  }
}
```

### 2.3 Model Status & Catalog
* **Method & Route**: `GET /models/status`
* **Description**: Reports loaded production models, file sizes, verified checksums, and benchmark scores.

---

## 3. Computer Vision Diagnostic API

### 3.1 Foliar Diagnosis Endpoint
* **Method & Route**: `POST /api/v1/vision/diagnose`
* **Content Type**: multipart form data
* **Request Parameters**:

| Field Name | Type | Required | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `file` | Binary (JPEG, PNG, WebP) | Yes | | Leaf photograph (Max 15 MB, Max 16M pixels) |
| `model_tier` | String | No | `"server"` | `"server"` (EfficientNetV2 S) or `"gen2"` (Tri Domain) |
| `explainability` | Boolean | No | `false` | Enables 9 stage explainability with heatmaps |
| `ambient_temperature` | Float | No | `null` | Ambient air temperature in Celsius |
| `relative_humidity` | Float | No | `null` | Air relative humidity percentage |
| `soil_moisture` | Float | No | `null` | Soil moisture volumetric percentage |
| `rain_detected` | Integer (0 or 1) | No | `null` | Binary rain drop sensor reading |
| `rainfall_mm` | Float | No | `null` | Continuous rainfall in millimeters |

* **Sample Response Structure**:
```json
{
  "status": "success",
  "pipeline_version": "CV-06-Universal-MultiCrop-v2.2",
  "request_id": "ab6eff79bf0d",
  "model_metadata": {
    "model_name": "EfficientNetV2-S Server-Grade Classifier",
    "architecture": "EfficientNetV2-S",
    "device": "MPS",
    "sha256_hash": "07e84d39481f0757898c5088cc7fb9e9d28817fbd71e16769fea5eb056eace74"
  },
  "image_quality": {
    "quality_score": 0.956,
    "quality_level": "Good",
    "blur_score": 701.0,
    "is_usable": true
  },
  "uncertainty": {
    "ood_status": "IN_DISTRIBUTION",
    "entropy_nats": 1.498,
    "is_low_confidence": false
  },
  "diagnosis": {
    "crop": "Strawberry",
    "predicted_class": "Strawberry___healthy",
    "disease_common_name": "Healthy Strawberry Leaf",
    "confidence_pct": 70.07,
    "foliar_damage_pct": 0.0,
    "lesion_foci_count": 0,
    "lesion_foci_source": "none"
  },
  "spatial_telemetry": {
    "specimen_detections": [
      {"class_name": "Strawberry leaf", "confidence": 0.96, "box_256": [45, 30, 210, 195]}
    ],
    "lesion_detections": []
  }
}
```

---

## 4. Agro Climatic Crop Recommendation API

### 4.1 Crop Recommendation
* **Method & Route**: `POST /api/v1/crops/recommend`
* **Content Type**: `application/json`
* **Sample Payload**:
```json
{
  "temperature": 26.5,
  "humidity": 75.0,
  "soil_moisture": 60.0,
  "rain": 0,
  "rainfall_mm": 110.0,
  "nitrogen": 85.0,
  "phosphorus": 45.0,
  "potassium": 40.0,
  "ph": 6.5
}
```

### 4.2 Regional Presets
* **Method & Route**: `GET /api/v1/crops/presets`
* **Description**: Returns curated agro climatic profiles (Arid Semi Desert, Monsoonal Paddy, Subtropical Coastal, etc.).

### 4.3 Microclimate Disease Alerts
* **Method & Route**: `GET /api/v1/crops/diseases`
* **Description**: Returns meteorological rule based fungal and bacterial pathogen risk alerts based on current temperature and humidity.

---

## 5. Standard Error Taxonomy

| HTTP Status | Error Code | Description | Client Action |
| :--- | :--- | :--- | :--- |
| `400 Bad Request` | `IMAGE_VALIDATION_FAILED` | File is not an image or exceeds 16M pixel decompression bounds. | Upload valid JPEG or PNG leaf image. |
| `422 Unprocessable` | `VALIDATION_ERROR` | Request payload fails Pydantic schema validation. | Correct JSON types and required parameters. |
| `503 Service Unavailable`| `MODELS_NOT_READY` | Requested vision model failed to load into memory. | Check server logs and model checkpoint files. |
