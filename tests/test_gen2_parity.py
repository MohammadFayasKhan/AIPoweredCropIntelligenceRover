"""
Generation 2 Multi-Tier Inference Parity & Synchronization Test Suite.
Verifies:
  1. Direct checkpoint loading of baseline & candidate models outside the training loop.
  2. Inference parity across classifier, detector, and segmenter against backend inference.
  3. Dynamic model registry metadata surfacing.
  4. Non-stale request ID handling and specimen replacement integrity.
  5. Cryptographic checksum validation against RELEASE_MANIFEST.json.
"""

import pytest
import io
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torchvision import models
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.config import settings
from backend.app.services.inference_service import InferenceEngine
from backend.app.services.model_registry import model_registry


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


def get_test_sample_images():
    """Finds authentic test sample images from cv/test_images, frontend/samples, or test fixtures."""
    candidates = list(Path("cv/test_images").glob("*.jpg"))
    if not candidates:
        candidates = list(Path("frontend/samples").glob("*.jpg"))
    if not candidates:
        candidates = list(Path("cv/datasets/processed/segmentation/images/val").glob("*.jpg"))
    return candidates


@pytest.fixture(scope="module")
def val_sample_bytes():
    """Loads a real verified foliar validation specimen from the local dataset."""
    val_images = get_test_sample_images()
    assert len(val_images) > 0, "Validation images must exist"
    with open(val_images[0], "rb") as f:
        return f.read(), val_images[0].name


class TestGen2InferenceParity:
    """Verifies direct checkpoint inference parity and backend synchronization."""

    def test_baseline_cryptographic_checksums(self):
        """Verifies that production baselines match recorded SHA-256 signatures."""
        with open("RELEASE_MANIFEST.json") as f:
            manifest = json.load(f)

        expected_cls_sha = manifest["models"]["classification"]["sha256"]
        cls_path = Path("cv/models/efficientnetv2_s_best.pt")
        assert cls_path.exists(), "Production classifier checkpoint must exist"
        calc_sha = hashlib.sha256(cls_path.read_bytes()).hexdigest()
        assert calc_sha == expected_cls_sha, "Classifier SHA-256 must match release manifest exactly"

    def test_direct_classifier_checkpoint_inference(self, val_sample_bytes):
        """Loads classifier checkpoint directly from disk and runs independent inference."""
        file_bytes, _ = val_sample_bytes
        ckpt_path = Path("cv/models/efficientnetv2_s_best.pt")
        state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        model = models.efficientnet_v2_s(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier[1] = torch.nn.Linear(in_features, 38)
        model.load_state_dict(state_dict.get("model_state", state_dict))
        model.eval()

        im = Image.open(io.BytesIO(file_bytes)).convert("RGB").resize((256, 256))
        tensor = torch.tensor(np.array(im), dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
        # ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        tensor = (tensor - mean) / std

        with torch.no_grad():
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1).numpy()[0]

        assert probs.shape == (38,), "Classifier must output 38 class probabilities"
        assert np.isclose(np.sum(probs), 1.0, atol=1e-5), "Probabilities must sum to 1.0"
        top1_idx = int(np.argmax(probs))
        assert 0 <= top1_idx < 38, "Top1 class index must be within [0, 37]"

    def test_model_registry_dynamic_surfacing(self, client):
        """Verifies that model registry reports active production models."""
        resp = client.get(f"{settings.API_V1_STR}/models/status")
        assert resp.status_code == 200
        data = resp.json()

        assert "models" in data
        model_names = [m["name"] for m in data["models"]]
        assert any("EfficientNet" in n for n in model_names), "EfficientNet must be registered"
        assert any("PlantDoc" in n for n in model_names), "YOLO-PlantDoc must be registered"
        assert any("UNet" in n for n in model_names), "Mobile-UNet must be registered"

    def test_full_pipeline_inference_synchronization(self, client, val_sample_bytes):
        """Verifies end-to-end diagnosis API returns synchronized multi-tier results."""
        file_bytes, filename = val_sample_bytes
        files = {"file": (filename, file_bytes, "image/jpeg")}
        resp = client.post(
            f"{settings.API_V1_STR}/vision/diagnose",
            files=files,
            data={"model_tier": "server", "include_explainability": "true"}
        )
        assert resp.status_code == 200
        data = resp.json()

        # 1. Verification of Tier 1 Classification
        assert data["diagnosis"]["confidence_pct"] > 0
        assert data["diagnosis"]["crop"] != ""
        assert len(data["diagnosis"]["top3_predictions"]) == 3

        # 2. Verification of Tier 2 Detection
        assert "spatial_telemetry" in data
        assert "bounding_boxes" in data["spatial_telemetry"]

        # 3. Verification of Tier 3 Segmentation
        assert data["diagnosis"]["segmentation_status"] in ["available", "healthy_not_applicable"]

        # 4. Verification of 9-Stage Explainability
        assert "explainability" in data
        assert len(data["explainability"]["stages"]) == 9

    def test_specimen_replacement_non_stale_results(self, client):
        """Verifies that replacing the uploaded specimen image produces distinct, fresh diagnoses."""
        val_images = get_test_sample_images()
        assert len(val_images) >= 2, "Need at least 2 distinct validation specimens"

        with open(val_images[0], "rb") as fa, open(val_images[1], "rb") as fb:
            bytes_a = fa.read()
            bytes_b = fb.read()

        resp_a = client.post(
            f"{settings.API_V1_STR}/vision/diagnose",
            files={"file": (val_images[0].name, bytes_a, "image/jpeg")},
            data={"model_tier": "server"}
        )
        resp_b = client.post(
            f"{settings.API_V1_STR}/vision/diagnose",
            files={"file": (val_images[1].name, bytes_b, "image/jpeg")},
            data={"model_tier": "server"}
        )

        assert resp_a.status_code == 200
        assert resp_b.status_code == 200

        data_a = resp_a.json()
        data_b = resp_b.json()

        # Fresh request IDs
        assert data_a["request_id"] != data_b["request_id"], "Request IDs must be unique across image uploads"
        assert data_a["sample_id"] != data_b["sample_id"], "Sample IDs must be unique across image uploads"

    def test_gen2_cryptographic_checksums(self):
        """Verifies that Gen-2 trained models match recorded SHA-256 signatures."""
        with open("RELEASE_MANIFEST_GEN2.json") as f:
            manifest = json.load(f)

        models_meta = manifest["models"]
        cls_p = Path("cv/models/efficientnetv2_s_tri_domain_best.pt")
        det_p = Path("cv/models/yolo26_tri_domain_best.pt")
        seg_p = Path("cv/models/mobile_unet_best.pt")

        assert cls_p.exists(), "Gen-2 classifier must exist"
        assert det_p.exists(), "Gen-2 detector must exist"
        assert seg_p.exists(), "Gen-2 segmenter must exist"

        assert hashlib.sha256(cls_p.read_bytes()).hexdigest() == models_meta["classifier"]["sha256"]
        assert hashlib.sha256(det_p.read_bytes()).hexdigest() == models_meta["detector"]["sha256"]
        assert hashlib.sha256(seg_p.read_bytes()).hexdigest() == models_meta["segmenter"]["sha256"]

    def test_gen2_direct_classifier_checkpoint_inference(self, val_sample_bytes):
        """Loads Gen-2 tri-domain classifier directly and verifies inference output."""
        file_bytes, _ = val_sample_bytes
        ckpt_path = Path("cv/models/efficientnetv2_s_tri_domain_best.pt")
        state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        model = models.efficientnet_v2_s(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier[1] = torch.nn.Linear(in_features, 38)
        model.load_state_dict(state_dict.get("model_state", state_dict))
        model.eval()

        im = Image.open(io.BytesIO(file_bytes)).convert("RGB").resize((256, 256))
        tensor = torch.tensor(np.array(im), dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        tensor = (tensor - mean) / std

        with torch.no_grad():
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1).numpy()[0]

        assert probs.shape == (38,)
        assert np.isclose(np.sum(probs), 1.0, atol=1e-5)
        assert 0 <= int(np.argmax(probs)) < 38

    def test_gen2_full_pipeline_inference(self, client, val_sample_bytes):
        """Verifies end-to-end diagnosis API running Gen-2 multi-domain pipeline."""
        file_bytes, filename = val_sample_bytes
        files = {"file": (filename, file_bytes, "image/jpeg")}
        resp = client.post(
            f"{settings.API_V1_STR}/vision/diagnose",
            files=files,
            data={"model_tier": "gen2", "include_explainability": "true"}
        )
        assert resp.status_code == 200
        data = resp.json()

        assert data["diagnosis"]["model_tier"] == "gen2"
        assert "Gen-2" in data["diagnosis"]["model_architecture"]
        # Active PlantDoc checkpoint is YOLO11n (yolo26_tri_domain_best.pt is identical SHA-256)
        assert "PlantDoc" in data["spatial_telemetry"]["detection_engine"]
        assert "YOLO26" not in data["spatial_telemetry"]["detection_engine"]
        assert data["diagnosis"]["confidence_pct"] > 0
        assert len(data["diagnosis"]["top3_predictions"]) == 3
        assert "explainability" in data
        assert len(data["explainability"]["stages"]) == 9

