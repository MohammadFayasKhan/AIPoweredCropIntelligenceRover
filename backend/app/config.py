"""
SmartCropVision Application Configuration Module.
Centralizes environment-driven configuration for production, testing, and development.
Provides dynamic checkpoint resolution, upload bounds, security limits, and CORS policies.
"""

from __future__ import annotations

from pathlib import Path
import os
import json
from typing import List, Optional, Dict, Any

# Resolve project root directory dynamically relative to this file
BACKEND_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BACKEND_DIR.parent

try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except Exception:
    pass

def _parse_cors_origins(raw: str) -> List[str]:
    """Safely parse comma-separated or JSON string of CORS origins."""
    if not raw:
        return [
            "http://localhost:3000",
            "http://127.0.0.1:3000",
            "http://localhost:8000",
            "http://127.0.0.1:8000",
        ]
    raw = raw.strip()
    if raw.startswith("[") and raw.endswith("]"):
        try:
            return json.loads(raw)
        except Exception:
            pass
    return [o.strip() for o in raw.split(",") if o.strip()]


class Settings:
    """Production settings resolved from environment variables with sensible defaults."""
    
    PROJECT_NAME: str = os.getenv("PROJECT_NAME", "SmartCropVision API")
    PROJECT_ROOT: Path = PROJECT_ROOT
    VERSION: str = "2.2.0"
    API_V1_STR: str = "/api/v1"
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development").lower()
    DEBUG: bool = os.getenv("DEBUG", "false").lower() in ("true", "1", "yes")
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()

    # Server network bindings
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))

    # Security & CORS
    CORS_ORIGINS: List[str] = _parse_cors_origins(os.getenv("CORS_ORIGINS", ""))
    ALLOW_CREDENTIALS: bool = os.getenv("ALLOW_CREDENTIALS", "true").lower() in ("true", "1", "yes")
    SECRET_KEY: str = os.getenv("SECRET_KEY", "smartcropvision-insecure-dev-key-change-in-prod")

    # Groq Cloud Multimodal Intelligence & Semantic Validation Gateway
    GROQ_API_KEY: Optional[str] = os.getenv("GROQ_API_KEY", None)
    GROQ_VISION_MODEL: str = os.getenv("GROQ_VISION_MODEL", "qwen/qwen3.8-27b")
    GROQ_VISION_ENABLED: bool = os.getenv("GROQ_VISION_ENABLED", "true").lower() in ("true", "1", "yes")
    GROQ_VISION_TIMEOUT: float = float(os.getenv("GROQ_VISION_TIMEOUT", "3.5"))

    # Image upload bounds and decompression bomb protections
    MAX_UPLOAD_SIZE_BYTES: int = int(os.getenv("MAX_UPLOAD_SIZE_BYTES", str(15 * 1024 * 1024)))  # 15 MB
    MIN_UPLOAD_SIZE_BYTES: int = int(os.getenv("MIN_UPLOAD_SIZE_BYTES", "2048"))  # 2 KB
    MAX_IMAGE_PIXELS: int = int(os.getenv("MAX_IMAGE_PIXELS", str(16_000_000)))  # 4000x4000 pixels max
    MAX_IMAGE_DIMENSION: int = int(os.getenv("MAX_IMAGE_DIMENSION", "4096"))
    MIN_IMAGE_DIMENSION: int = int(os.getenv("MIN_IMAGE_DIMENSION", "64"))
    ALLOWED_MIME_TYPES: List[str] = [
        "image/jpeg",
        "image/jpg",
        "image/png",
        "image/webp"
    ]
    ALLOWED_EXTENSIONS: List[str] = [".jpg", ".jpeg", ".png", ".webp"]

    # Model Directory & Checkpoint Paths
    MODEL_DIR: Path = Path(os.getenv("MODEL_DIR", str(PROJECT_ROOT / "cv" / "models")))
    CONFIG_DIR: Path = Path(os.getenv("CONFIG_DIR", str(PROJECT_ROOT / "cv" / "configs")))

    TAXONOMY_PATH: Path = Path(os.getenv("TAXONOMY_PATH", str(MODEL_DIR / "taxonomy_38.json")))
    if not TAXONOMY_PATH.exists() and (CONFIG_DIR / "taxonomy_38classes.json").exists():
        TAXONOMY_PATH = CONFIG_DIR / "taxonomy_38classes.json"

    PREPROCESSING_CONFIG_PATH: Path = Path(os.getenv("PREPROCESSING_CONFIG_PATH", str(MODEL_DIR / "preprocessing_config.json")))
    RELEASE_MANIFEST_PATH: Path = Path(os.getenv("RELEASE_MANIFEST_PATH", str(PROJECT_ROOT / "RELEASE_MANIFEST.json")))

    # Verified production model checkpoints
    SERVER_TIER1_MODEL_PATH: Path = Path(os.getenv("SERVER_TIER1_MODEL_PATH", str(MODEL_DIR / "efficientnetv2_s_best.pt")))
    EDGE_TIER1_MODEL_PATH: Path = Path(os.getenv("EDGE_TIER1_MODEL_PATH", str(MODEL_DIR / "mobilenet_v2_38classes_best.pth")))
    TIER1_MODEL_PATH: Path = SERVER_TIER1_MODEL_PATH
    TIER2_MODEL_PATH: Path = Path(os.getenv("TIER2_MODEL_PATH", str(MODEL_DIR / "yolo_plantdoc_best.pt")))
    TIER2_PLANTDOC_MODEL_PATH: Path = Path(os.getenv("TIER2_PLANTDOC_MODEL_PATH", str(MODEL_DIR / "yolo_plantdoc_best.pt")))
    TIER3_MODEL_PATH: Path = Path(os.getenv("TIER3_MODEL_PATH", str(MODEL_DIR / "mobile_unet_best.pt")))
    DEFAULT_TIER1_MODEL: str = os.getenv("DEFAULT_TIER1_MODEL", "server")

    # Generation 2 Multi-Domain Tri-Dataset Checkpoints
    GEN2_TIER1_MODEL_PATH: Path = Path(os.getenv("GEN2_TIER1_MODEL_PATH", str(MODEL_DIR / "efficientnetv2_s_tri_domain_best.pt")))
    GEN2_TIER2_YOLO26_MODEL_PATH: Path = Path(os.getenv("GEN2_TIER2_YOLO26_MODEL_PATH", str(MODEL_DIR / "yolo26_tri_domain_best.pt")))
    GEN2_TIER3_UNET_MODEL_PATH: Path = Path(os.getenv("GEN2_TIER3_UNET_MODEL_PATH", str(MODEL_DIR / "mobile_unet_best.pt")))
    GEN2_RELEASE_MANIFEST_PATH: Path = Path(os.getenv("GEN2_RELEASE_MANIFEST_PATH", str(PROJECT_ROOT / "RELEASE_MANIFEST_GEN2.json")))

    # Authoritative verified checksums & metrics
    PROMOTED_CLASSIFIER_SHA256: str = "07e84d39481f0757898c5088cc7fb9e9d28817fbd71e16769fea5eb056eace74"
    GEN2_CLASSIFIER_SHA256: str = "7ebe13a085819231878c776092c112b2c0babf678d39e4d0f2a2e1ee258f25c7"
    GEN2_DETECTOR_SHA256: str = "e98545716c774d796a4796f620138022f6388a063bf8a9d36dfc5937372366b6"
    GEN2_SEGMENTER_SHA256: str = "86bba6df9222b92c2d8b019fb2a10eacc2654f1a61b6e2b5996d8c8ee3cc04cc"
    CLASSIFIER_INPUT_RESOLUTION: int = 256
    CLASSIFIER_TEST_TOP1_ACC: float = 0.9513
    CLASSIFIER_TEST_MACRO_F1: float = 0.9354
    CLASSIFIER_ECE: float = 0.0803
    DETECTOR_VAL_MAP50: float = 0.3362
    DETECTOR_VAL_MAP50_95: float = 0.2361

    # Crop Recommendation Artifacts
    CROP_MODEL_DIR: Path = Path(os.getenv("CROP_MODEL_DIR", str(MODEL_DIR / "crop_recommender")))
    CROP_MODEL_PATH: Path = CROP_MODEL_DIR / "crop_model.pkl"
    CROP_SCALER_PATH: Path = CROP_MODEL_DIR / "scaler.pkl"
    CROP_LABEL_ENCODER_PATH: Path = CROP_MODEL_DIR / "label_encoder.pkl"
    CROP_FEATURE_NAMES_PATH: Path = CROP_MODEL_DIR / "feature_names.pkl"

    # Inference & Operational Controls
    REQUEST_TIMEOUT_SECONDS: float = float(os.getenv("REQUEST_TIMEOUT_SECONDS", "30.0"))
    EXPLAINABILITY_ENABLED_DEFAULT: bool = os.getenv("EXPLAINABILITY_ENABLED_DEFAULT", "false").lower() in ("true", "1", "yes")
    CONF_THRESHOLD_SCREENING: float = float(os.getenv("CONF_THRESHOLD_SCREENING", "0.50"))
    YOLO_CONF_THRESH: float = float(os.getenv("YOLO_CONF_THRESH", "0.15"))
    YOLO_PLANTDOC_CONF_THRESH: float = float(os.getenv("YOLO_PLANTDOC_CONF_THRESH", "0.12"))
    YOLO_LESIONS_CONF_THRESH: float = float(os.getenv("YOLO_LESIONS_CONF_THRESH", "0.10"))
    YOLO_IOU_THRESH: float = float(os.getenv("YOLO_IOU_THRESH", "0.45"))


settings = Settings()

