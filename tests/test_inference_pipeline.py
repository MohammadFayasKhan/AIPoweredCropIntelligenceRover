"""
SmartCropVision Test Suite: Core Inference Pipeline & Architectural Safeguards
Mandatory regression tests:
1. Fake-box prevention: Zero detections must yield zero boxes, never synthetic replacements.
2. Fake-override prevention: Low-confidence healthy leaves must never be transformed into diseases.
3. Low-confidence handling: Epistemic uncertainty flagged, calm non-urgent advisory.
4. Dynamic device detection: Reflects real hardware.
5. Segmentation availability honesty: No fabricated damage percentage when unavailable.
6. Taxonomy separation: Classification 38 != PlantDoc 29.
"""

import pytest
import numpy as np
import torch
from pathlib import Path
from PIL import Image

from backend.app.services.inference_service import inference_engine
from backend.app.schemas.diagnosis import DiagnosisResponse, ModelMetadata


@pytest.fixture(scope="module")
def loaded_engine():
    inference_engine.load_models()
    return inference_engine


@pytest.fixture
def clean_healthy_leaf_path():
    """Returns path to an authentic healthy leaf specimen in the repository."""
    path = Path("frontend/samples/apple__healthy__healthy.jpg")
    assert path.exists()
    return str(path)


def test_fake_box_prevention_on_clean_image(loaded_engine, clean_healthy_leaf_path):
    """
    MANDATORY REGRESSION TEST:
    Verifies that when no lesions are detected by YOLO, the system produces
    EXACTLY zero bounding boxes and zero foci, with zero morphological dilation replacements.
    """
    # Force detector to return empty list
    orig_pdoc = loaded_engine.model_tier2_plantdoc
    orig_tier2 = loaded_engine.model_tier2
    loaded_engine.model_tier2_plantdoc = None
    loaded_engine.model_tier2 = None

    try:
        res = loaded_engine.predict_vision(clean_healthy_leaf_path, model_tier="server")
        assert isinstance(res, DiagnosisResponse)
        # Verify bounding boxes are strictly empty
        assert len(res.spatial_telemetry.bounding_boxes) == 0, (
            f"Expected 0 boxes, got {len(res.spatial_telemetry.bounding_boxes)}. "
            "Synthetic boxes must never be generated!"
        )
        assert res.diagnosis.lesion_foci_count == 0
        assert res.spatial_telemetry.nozzle_actuation_targets == 0
    finally:
        loaded_engine.model_tier2_plantdoc = orig_pdoc
        loaded_engine.model_tier2 = orig_tier2


def test_fake_override_prevention(loaded_engine, clean_healthy_leaf_path):
    """
    MANDATORY REGRESSION TEST:
    Verifies that low-confidence healthy leaves are NEVER converted to
    'Foliar Lesions & Spotting (Atypical Specimen)' or 'Severe Foliar Necrosis'.
    """
    res = loaded_engine.predict_vision(clean_healthy_leaf_path, model_tier="server")
    disease_name = res.diagnosis.disease_common_name
    assert "Foliar Lesions & Spotting" not in disease_name, (
        "Synthetic override detected! Low confidence must not invent disease diagnoses."
    )
    assert "Severe Foliar Necrosis" not in disease_name


def test_low_confidence_handling_and_advisory(loaded_engine, clean_healthy_leaf_path):
    """
    Verifies that when confidence is low (<50%):
    - is_low_confidence is True
    - confidence_level is LOW_UNCERTAIN
    - Advisory does not trigger emergency/urgent quarantine or spraying
    """
    res = loaded_engine.predict_vision(clean_healthy_leaf_path, model_tier="server")
    if res.diagnosis.confidence_pct < 50.0:
        assert res.diagnosis.is_low_confidence is True
        assert res.diagnosis.confidence_level in ("LOW_UNCERTAIN", "REJECTED")
        # Verify advisory guidance is non-urgent
        assert "emergency" not in res.advisory.immediate_action.lower()
        assert res.advisory.uncertainty_guidance is not None
        assert len(res.advisory.uncertainty_guidance) > 10


def test_dynamic_device_detection(loaded_engine):
    """
    Verifies that hardware acceleration reflects the actual runtime device,
    not a hardcoded string literal.
    """
    device_name = loaded_engine.device_name
    if torch.cuda.is_available():
        assert "CUDA" in device_name
        assert torch.cuda.get_device_name(0) in device_name
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        assert "MPS" in device_name
    else:
        assert "CPU" in device_name


def test_segmentation_unavailable_honesty(loaded_engine):
    """
    Verifies that if segmentation is unavailable for an infected leaf,
    foliar_damage_pct is None and segmentation_status == 'unavailable'.
    Never fabricates 0% or fake area calculation.
    """
    orig_unet = loaded_engine.model_tier3
    loaded_engine.model_tier3 = None
    sample_path = "frontend/samples/apple__fungal__apple_scab.jpg"
    try:
        res = loaded_engine.predict_vision(sample_path, model_tier="server")
        assert res.diagnosis.segmentation_status == "unavailable"
        assert res.diagnosis.foliar_damage_pct is None
    finally:
        loaded_engine.model_tier3 = orig_unet


def test_taxonomy_separation(loaded_engine):
    """
    Verifies that the 38-class classification taxonomy and the 29-class PlantDoc
    detection taxonomy are tracked as distinct, independent namespaces.
    """
    assert len(loaded_engine.taxonomy) == 38
    # Verify metadata exposes separate classification and detection taxonomy versions
    meta = loaded_engine.get_model_metadata("server")
    assert isinstance(meta, ModelMetadata)
    assert meta.classification_taxonomy_version == "PlantVillage-38Class-v2.0"
    assert meta.detection_taxonomy_version == "PlantDoc-29Class-YOLO26-v1.0"
    assert meta.segmentation_taxonomy_version == "FoliarLesions-Binary-v1.0"
    meta_legacy = loaded_engine.get_model_metadata("yolo11")
    assert meta_legacy.detection_taxonomy_version == "PlantDoc-29Class-YOLO-v1.0"
