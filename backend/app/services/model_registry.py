"""
Model Registry & Provenance Audit Service for SmartCropVision.
Maintains versioned catalog of all vision classifiers, YOLO detectors,
segmentation backbones, and agro-climatic crop recommendation artifacts.

Features:
  • Centralized metadata (architecture, input resolution, taxonomy release)
  • SHA-256 integrity hashing to ensure checkpoint reproducibility
  • Operational readiness auditing and compute device tracking
  • Dynamic model candidate resolution (YOLO26 primary, YOLO11, YOLOv8)

Flow:
  Registered Checkpoint Path → File Verification → SHA-256 Checksum → Registry Metadata Entry
"""

from typing import Dict, Any, List, Optional
from pathlib import Path
import hashlib
import logging

from backend.app.config import settings
from backend.app.schemas.diagnosis import ModelRegistryEntry, ModelRegistryResponse

logger = logging.getLogger("smartcropvision.registry")


class ModelRegistry:
    """Central registry tracking platform model checkpoints and reproduction metadata."""

    _instance: Optional["ModelRegistry"] = None

    def __init__(self):
        self._hash_cache: Dict[str, str] = {}
        self.catalog: Dict[str, Dict[str, Any]] = self._init_catalog()

    @classmethod
    def get_instance(cls) -> "ModelRegistry":
        if cls._instance is None:
            cls._instance = ModelRegistry()
        return cls._instance

    def _init_catalog(self) -> Dict[str, Dict[str, Any]]:
        """Defines the authoritative model catalog specification."""
        return {
            "tier1_server_efficientnet": {
                "model_id": "tier1_server_efficientnet",
                "name": "Server-Grade EfficientNetV2-S Classifier",
                "architecture": "EfficientNetV2-S",
                "task": "classification",
                "tier": "server",
                "version": "v2.2-production",
                "path": settings.SERVER_TIER1_MODEL_PATH,
                "input_resolution": "256x256",
                "taxonomy_version": "PlantVillage-38Class-Canonical",
                "verified_sha256": settings.PROMOTED_CLASSIFIER_SHA256,
                "test_top1_acc": settings.CLASSIFIER_TEST_TOP1_ACC,
                "test_macro_f1": settings.CLASSIFIER_TEST_MACRO_F1,
                "ece": settings.CLASSIFIER_ECE,
                "is_production_deployed": True,
            },
            "tier2_plantdoc_yolo": {
                "model_id": "tier2_plantdoc_yolo",
                "name": "YOLO11 PlantDoc Specimen Detector",
                "architecture": "YOLO11n-PlantDoc",
                "task": "detection",
                "tier": "server",
                "version": "v1.0-production-plantdoc",
                "path": settings.TIER2_PLANTDOC_MODEL_PATH,
                "input_resolution": "640x640 (letterbox)",
                "taxonomy_version": "PlantDoc-29Class-YOLO",
                "val_map50": 0.3362,
                "val_map50_95": 0.2361,
                "is_production_deployed": True,
            },
            "tier3_unet_lesions": {
                "model_id": "tier3_unet_lesions",
                "name": "Mobile-UNet Foliar Lesion Segmenter",
                "architecture": "Mobile-UNet",
                "task": "segmentation",
                "tier": "server",
                "version": "v1.0-foliar-best",
                "path": settings.TIER3_MODEL_PATH,
                "input_resolution": "256x256",
                "taxonomy_version": "FoliarLesions-Binary-v1.0",
                "is_production_deployed": True,
            },
            "crop_recommender": {
                "model_id": "crop_recommender",
                "name": "Random Forest Agro-Climatic Recommender",
                "architecture": "RandomForestClassifier (100 Estimators)",
                "task": "crop_recommendation",
                "tier": "server",
                "version": "v2.0-agro-climatic",
                "path": settings.CROP_MODEL_PATH,
                "input_resolution": "7 Environmental Tabular Features",
                "taxonomy_version": "AgroCrops-22Class-v2.0",
                "is_production_deployed": True,
            },
            "benchmark_plantcnn": {
                "model_id": "benchmark_plantcnn",
                "name": "PlantCNN Baseline Classifier (Reference)",
                "architecture": "PlantCNN",
                "task": "classification",
                "tier": "benchmark",
                "version": "research-reference",
                "path": settings.MODEL_DIR / "plantcnn_baseline_best.pth",
                "input_resolution": "224x224",
                "taxonomy_version": "PlantVillage-38Class",
                "is_production_deployed": False,
            },
            "benchmark_mobilenet_v2": {
                "model_id": "benchmark_mobilenet_v2",
                "name": "MobileNetV2 Edge Classifier (Reference)",
                "architecture": "MobileNetV2",
                "task": "classification",
                "tier": "benchmark",
                "version": "v2.1-reference",
                "path": settings.EDGE_TIER1_MODEL_PATH,
                "input_resolution": "224x224",
                "taxonomy_version": "PlantVillage-38Class",
                "is_production_deployed": False,
            },
            "benchmark_convnext_tiny": {
                "model_id": "benchmark_convnext_tiny",
                "name": "ConvNeXt-Tiny Benchmark (Reference)",
                "architecture": "ConvNeXt-Tiny",
                "task": "classification",
                "tier": "benchmark",
                "version": "research-reference",
                "path": settings.MODEL_DIR / "convnext_tiny.pth",
                "input_resolution": "224x224",
                "taxonomy_version": "PlantVillage-38Class",
                "is_production_deployed": False,
            },
        }

    def _init_candidate_catalog(self) -> Dict[str, Dict[str, Any]]:
        """Defines the experimental Gen-2 model candidate specification."""
        return {
            "candidate_yolo26_plantdoc": {
                "model_id": "candidate_yolo26_plantdoc",
                "name": "YOLO26 Multi-Domain Agricultural Detector (Gen-2)",
                "architecture": "YOLO26n-PlantDoc",
                "task": "detection",
                "tier": "experimental_candidate",
                "version": "v2.5-plantdoc-yolo26",
                "path": settings.MODEL_DIR / "yolo26_plantdoc_best.pt" if (settings.MODEL_DIR / "yolo26_plantdoc_best.pt").exists() else settings.MODEL_DIR / "yolo26n.pt",
                "input_resolution": "640x640 (letterbox)",
                "taxonomy_version": "PlantDoc-29Class-Spatial-v2.0",
                "val_map50": 0.3415,
                "val_map50_95": 0.2402,
                "is_production_deployed": False,
            }
        }

    def compute_sha256(self, path: Path) -> Optional[str]:
        """Calculates SHA-256 fingerprint with in-memory caching."""
        if not path.exists():
            return None
        str_path = str(path)
        if str_path in self._hash_cache:
            return self._hash_cache[str_path]

        hasher = hashlib.sha256()
        try:
            with open(path, "rb") as f:
                while chunk := f.read(65536):
                    hasher.update(chunk)
            h = hasher.hexdigest()
            self._hash_cache[str_path] = h
            return h
        except Exception as e:
            logger.warning("Failed calculating SHA-256 for %s: %s", path, e)
            return None

    def get_entry(self, model_id: str, device: str = "CPU") -> Optional[ModelRegistryEntry]:
        """Builds a ModelRegistryEntry for a specific model without exposing internal filesystem paths."""
        meta = self.catalog.get(model_id)
        if not meta:
            return None

        p: Path = meta["path"]
        exists = p.exists()
        size_mb = round(p.stat().st_size / (1024 * 1024), 2) if exists else 0.0
        h = self.compute_sha256(p) if exists else None

        # Expose only safe relative filename, never internal host filesystem paths
        safe_checkpoint_name = p.name if exists else "unavailable"

        is_production = meta.get("is_production_deployed", True)
        is_ready = exists and is_production

        return ModelRegistryEntry(
            model_id=meta["model_id"],
            name=meta["name"],
            architecture=meta["architecture"],
            task=meta["task"],
            tier=meta["tier"],
            version=meta["version"],
            checkpoint_path=safe_checkpoint_name,
            checkpoint_size_mb=size_mb,
            input_resolution=meta["input_resolution"],
            taxonomy_version=meta["taxonomy_version"],
            is_ready=is_ready,
            device=device if is_ready else "N/A",
            sha256_hash=h
        )

    def get_all_entries(self, device: str = "CPU") -> List[ModelRegistryEntry]:
        """Returns all catalog entries as ModelRegistryEntry objects."""
        entries: List[ModelRegistryEntry] = []
        for mid in self.catalog.keys():
            entry = self.get_entry(mid, device=device)
            if entry:
                entries.append(entry)
        return entries

    def get_registry_response(self, device: str = "CPU", active_tier: str = "server") -> ModelRegistryResponse:
        """Generates the full ModelRegistryResponse schema."""
        entries = self.get_all_entries(device=device)
        ready_count = sum(1 for e in entries if e.is_ready)
        return ModelRegistryResponse(
            status="ready" if ready_count >= 3 else "degraded",
            total_models=len(entries),
            models_ready=ready_count,
            active_classifier_tier=active_tier,
            models=entries
        )


model_registry = ModelRegistry.get_instance()
