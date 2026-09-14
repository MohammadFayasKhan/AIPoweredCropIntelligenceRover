"""
Crop Recommendation and Agro-Climatic Intelligence Service.
Loads RandomForestClassifier, StandardScaler, and LabelEncoder artifacts.
Executes probabilistic crop prediction across 22 classes, maps binary rain
and continuous rainfall measurements, and runs the microclimate disease rules engine.
"""

import time
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from backend.app.config import settings
from backend.app.schemas.crop import (
    CropRecommendationRequest,
    CropRecommendationResponse,
    CropCandidate,
    AgroClimaticAlert,
    EnvironmentalFeaturesUsed,
    CropPreset
)

logger = logging.getLogger("smartplant.crop_service")

# Severity order for sorting disease alerts
SEVERITY_ORDER = {"CRITICAL": 0, "HIGH": 1, "MODERATE": 2, "WATCH": 3}

# Curated disease knowledge base with agronomic guidance
DISEASE_KNOWLEDGE_BASE = [
    {
        "id": "late_blight",
        "name": "Late Blight (Phytophthora infestans)",
        "type": "Fungal",
        "severity": "CRITICAL",
        "trigger": "Humidity > 85% and Temperature 10°C to 25°C",
        "condition": lambda t, h, rain, sm: h > 85 and 10 <= t <= 25,
        "symptoms": "Brown-black lesions on leaves with white fungal mold at edges. Affected tissue collapses rapidly.",
        "pesticide": "Mancozeb 75% WP: 2.5 g/L water, or Cymoxanil + Mancozeb (Curzate M8) 2.5 g/L. Spray every 7 days.",
        "technique": "Remove infected plant tissue immediately. Avoid overhead irrigation. Increase row spacing for air circulation.",
    },
    {
        "id": "powdery_mildew",
        "name": "Powdery Mildew (Erysiphales)",
        "type": "Fungal",
        "severity": "HIGH",
        "trigger": "Humidity 60% to 80% and Temperature 20°C to 30°C",
        "condition": lambda t, h, rain, sm: 60 <= h <= 80 and 20 <= t <= 30,
        "symptoms": "White powdery talcum-like coating on upper leaf surfaces. Leaves curl upward and yellow.",
        "pesticide": "Sulphur 80% WP: 2 g/L, or Hexaconazole (Contaf 5 EC) 1 mL/L. Apply at first visual sign.",
        "technique": "Improve canopy ventilation. Avoid excessive nitrogenous fertilizers. Prune shaded inner leaves.",
    },
    {
        "id": "gray_mold",
        "name": "Gray Mold (Botrytis cinerea)",
        "type": "Fungal",
        "severity": "HIGH",
        "trigger": "Humidity > 90% and Temperature 15°C to 25°C",
        "condition": lambda t, h, rain, sm: h > 90 and 15 <= t <= 25,
        "symptoms": "Soft gray fuzzy mold on flowers, fruits, and foliage. Rapid watery decay of delicate tissues.",
        "pesticide": "Iprodione (Rovral 50 WP) 2 g/L, or Carbendazim 50% WP 1 g/L.",
        "technique": "Reduce greenhouse/field humidity immediately. Prune dense foliage to allow morning drying.",
    },
    {
        "id": "foliar_humidity_stress",
        "name": "High Foliar Humidity / Spore Germination",
        "type": "Fungal",
        "severity": "HIGH",
        "trigger": "Humidity ≥ 80% and Temperature 20°C to 35°C",
        "condition": lambda t, h, rain, sm: h >= 80 and 20 <= t <= 35,
        "symptoms": "High boundary layer moisture induces foliar water-congestion and accelerates fungal spore germination.",
        "pesticide": "Preventative bio-fungicide: Trichoderma viride 5 g/L or Copper Oxychloride 2 g/L.",
        "technique": "Enhance plant spacing and canopy aeration; avoid overhead sprinkler wetting.",
    },
    {
        "id": "anthracnose",
        "name": "Anthracnose (Colletotrichum spp.)",
        "type": "Fungal",
        "severity": "HIGH",
        "trigger": "Humidity > 80%, Temperature 24°C to 32°C, and Rain Detected",
        "condition": lambda t, h, rain, sm: h > 80 and 24 <= t <= 32 and rain == 1,
        "symptoms": "Dark sunken circular necrotic lesions on fruits and stems with salmon-pink spore droplets.",
        "pesticide": "Copper Oxychloride 50% WP: 3 g/L, or Azoxystrobin (Amistar) 1 mL/L. Spray before rain events.",
        "technique": "Transition to drip irrigation from overhead sprinklers. Disinfect harvesting tools.",
    },
    {
        "id": "downy_mildew",
        "name": "Downy Mildew (Peronosporaceae)",
        "type": "Fungal",
        "severity": "HIGH",
        "trigger": "Humidity > 85% and Temperature 10°C to 18°C",
        "condition": lambda t, h, rain, sm: h > 85 and 10 <= t <= 18,
        "symptoms": "Angular yellow patches bounded by leaf veins on upper surface, gray-violet down beneath.",
        "pesticide": "Metalaxyl + Mancozeb (Ridomil Gold MZ) 2.5 g/L, or Cymoxanil 8% + Mancozeb 64% WP 2 g/L.",
        "technique": "Avoid late-evening irrigation. Ensure rapid drainage from furrows.",
    },
    {
        "id": "bacterial_blight",
        "name": "Bacterial Blight (Xanthomonas spp.)",
        "type": "Bacterial",
        "severity": "CRITICAL",
        "trigger": "Temperature > 30°C, Humidity > 75%, and Rain Detected",
        "condition": lambda t, h, rain, sm: t > 30 and h > 75 and rain == 1,
        "symptoms": "Water-soaked translucent lesions that rapidly turn brown with bacterial exudate droplets.",
        "pesticide": "Copper Hydroxide (Kocide 77 WP) 3 g/L, or Streptomycin Sulphate + Tetracycline 0.5 g/L.",
        "technique": "Do not prune or cultivate when canopy is wet. Burn severely infected crop debris.",
    },
    {
        "id": "root_rot",
        "name": "Root Rot / Damping Off (Pythium spp.)",
        "type": "Oomycete",
        "severity": "CRITICAL",
        "trigger": "Rain Detected and Soil Moisture > 85%",
        "condition": lambda t, h, rain, sm: rain == 1 and sm > 85,
        "symptoms": "Severe wilting despite wet soil. Waterlogged, brown, necrotic root cortex.",
        "pesticide": "Metalaxyl (Ridomil) 2 mL/L root drench, or Fosetyl-Al (Aliette 80 WP) 2.5 g/L.",
        "technique": "Construct raised planting beds. Cut drainage trenches to shed surface standing water.",
    },
    {
        "id": "spider_mite",
        "name": "Two-Spotted Spider Mite (Tetranychus urticae)",
        "type": "Pest",
        "severity": "HIGH",
        "trigger": "Temperature > 32°C and Humidity < 40%",
        "condition": lambda t, h, rain, sm: t > 32 and h < 40,
        "symptoms": "Fine silk webbing on leaf undersides, bronzed speckled foliage, premature defoliation.",
        "pesticide": "Abamectin (Vertimec 1.8 EC) 0.5 mL/L, or Spiromesifen (Oberon 240 SC) 0.5 mL/L.",
        "technique": "Introduce predatory mites (Phytoseiulus persimilis). Light misting to elevate boundary humidity.",
    },
    {
        "id": "aphid",
        "name": "Aphid Infestation (Aphididae)",
        "type": "Pest",
        "severity": "MODERATE",
        "trigger": "Temperature 18°C to 28°C and Humidity < 50%",
        "condition": lambda t, h, rain, sm: 18 <= t <= 28 and h < 50,
        "symptoms": "Colonies clustered on apical shoots, honeydew deposits leading to black sooty mold.",
        "pesticide": "Imidacloprid (Confidor 200 SL) 0.3 mL/L, or cold-pressed Neem oil (5 mL/L with mild surfactant).",
        "technique": "Preserve ladybird beetle and hoverfly populations. Avoid excessive soluble nitrogen.",
    },
    {
        "id": "heat_stress",
        "name": "Thermal Heat Stress",
        "type": "Abiotic",
        "severity": "HIGH",
        "trigger": "Temperature > 38°C",
        "condition": lambda t, h, rain, sm: t > 38,
        "symptoms": "Blossom abortion, fruit sunburn, leaf marginal scorch, pollen sterility.",
        "pesticide": "Kaolin clay suspension (Surround WP) 50 g/L for foliar solar reflection, or Salicylic acid 0.5 mM.",
        "technique": "Deploy 35% to 50% agricultural shade netting. Irrigate pre-dawn to maintain transpiration cooling.",
    },
    {
        "id": "drought_stress",
        "name": "Soil Moisture Drought Deficit",
        "type": "Abiotic",
        "severity": "HIGH",
        "trigger": "No Rain Detected and Soil Moisture < 25%",
        "condition": lambda t, h, rain, sm: rain == 0 and sm < 25,
        "symptoms": "Foliage epinasty and inward rolling, reduced cell turgor, leaf edge crisping.",
        "pesticide": "Potassium Silicate foliar spray (3 g/L) to strengthen cuticle integrity against transpiration.",
        "technique": "Apply organic straw/hay mulch 8 cm deep. Implement deficit drip scheduling at root zone.",
    },
]

# Standard curated regional presets for user convenience
CURATED_PRESETS: List[CropPreset] = [
    CropPreset(
        id="preset_monsoon_wetland",
        title="Monsoon Wetland",
        region_description="Tropical floodplains, high precipitation, heavy saturated soils.",
        temperature=28.5,
        humidity=84.0,
        soil_moisture=82.0,
        rain=1,
        rainfall_mm=160.0,
        expected_crop="Rice / Jute"
    ),
    CropPreset(
        id="preset_semi_arid",
        title="Semi-Arid Drylands",
        region_description="Warm inland plateaus, low humidity, water-stressed sandy loam.",
        temperature=31.0,
        humidity=38.0,
        soil_moisture=22.0,
        rain=0,
        rainfall_mm=18.0,
        expected_crop="Mothbeans / Chickpea"
    ),
    CropPreset(
        id="preset_temperate_orchard",
        title="Temperate Orchard Valley",
        region_description="Cool highlands, moderate humidity, well-drained loamy slopes.",
        temperature=19.5,
        humidity=65.0,
        soil_moisture=48.0,
        rain=0,
        rainfall_mm=35.0,
        expected_crop="Apple / Grapes"
    ),
    CropPreset(
        id="preset_subtropical_coastal",
        title="Subtropical Coastal Plains",
        region_description="Maritime breezes, warm temperatures, consistent moisture.",
        temperature=27.0,
        humidity=78.0,
        soil_moisture=62.0,
        rain=0,
        rainfall_mm=65.0,
        expected_crop="Banana / Coconut"
    ),
]


class CropInferenceService:
    """
    Singleton inference manager for Agro-Climatic Crop Recommendation.
    Maintains cached Random Forest classifier and feature scalers in memory.
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(CropInferenceService, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance
        
    def _initialize(self):
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.feature_names = ["temperature", "humidity", "rainfall"]
        self.classes: List[str] = []
        self.is_ready = False
        
        self.RAIN_TO_MM = {1: 120.0, 0: 20.0}
        self.load_artifacts()
        
    def load_artifacts(self):
        """Loads scikit-learn model, scaler, and label encoder from disk."""
        logger.info("Initializing Crop Recommendation Service artifacts...")
        
        try:
            if not settings.CROP_MODEL_PATH.exists():
                raise FileNotFoundError(f"Crop model artifact missing: {settings.CROP_MODEL_PATH}")
            if not settings.CROP_SCALER_PATH.exists():
                raise FileNotFoundError(f"Crop scaler artifact missing: {settings.CROP_SCALER_PATH}")
            if not settings.CROP_LABEL_ENCODER_PATH.exists():
                raise FileNotFoundError(f"Crop label encoder missing: {settings.CROP_LABEL_ENCODER_PATH}")
                
            self.model = joblib.load(str(settings.CROP_MODEL_PATH))
            self.scaler = joblib.load(str(settings.CROP_SCALER_PATH))
            self.label_encoder = joblib.load(str(settings.CROP_LABEL_ENCODER_PATH))
            
            if settings.CROP_FEATURE_NAMES_PATH.exists():
                self.feature_names = list(joblib.load(str(settings.CROP_FEATURE_NAMES_PATH)))
                
            self.classes = [str(c) for c in self.label_encoder.classes_]
            self.is_ready = True
            logger.info(f"Crop Recommendation Service ready. Total classes: {len(self.classes)} ({', '.join(self.classes[:5])}...)")
            
        except Exception as e:
            logger.error(f"Failed to load crop recommendation artifacts: {str(e)}", exc_info=True)
            self.is_ready = False
            raise
            
    def get_status(self) -> Dict[str, Any]:
        """Provides status audit for system readiness probes."""
        return {
            "is_ready": self.is_ready,
            "model_type": type(self.model).__name__ if self.model else "None",
            "features": self.feature_names,
            "classes_count": len(self.classes),
            "classes_sample": self.classes[:6] if self.classes else [],
            "artifacts": {
                "model_path": str(settings.CROP_MODEL_PATH),
                "scaler_path": str(settings.CROP_SCALER_PATH),
                "label_encoder_path": str(settings.CROP_LABEL_ENCODER_PATH)
            }
        }
        
    def get_presets(self) -> List[CropPreset]:
        """Returns the curated regional agricultural presets."""
        return CURATED_PRESETS
        
    def detect_disease_risks(
        self,
        temperature: float,
        humidity: float,
        rain: int,
        soil_moisture: float
    ) -> List[AgroClimaticAlert]:
        """
        Evaluates microclimate environmental thresholds against the disease database.
        Returns triggered alerts ordered by severity.
        """
        alerts: List[AgroClimaticAlert] = []
        rain_int = int(rain)
        
        for rule in DISEASE_KNOWLEDGE_BASE:
            try:
                if rule["condition"](temperature, humidity, rain_int, soil_moisture):
                    alerts.append(
                        AgroClimaticAlert(
                            id=rule["id"],
                            name=rule["name"],
                            type=rule["type"],
                            severity=rule["severity"],
                            trigger=rule["trigger"],
                            symptoms=rule["symptoms"],
                            pesticide=rule["pesticide"],
                            technique=rule["technique"]
                        )
                    )
            except Exception as err:
                logger.warning(f"Error evaluating rule {rule.get('id')}: {err}")
                
        alerts.sort(key=lambda a: SEVERITY_ORDER.get(a.severity, 99))
        return alerts

    def get_disease_knowledge(
        self,
        disease_name: str,
        temperature: Optional[float] = None,
        humidity: Optional[float] = None,
        rain: Optional[int] = None,
        soil_moisture: Optional[float] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Cross-references predicted foliar pathology with curated agronomic knowledge base.
        Evaluates whether current microclimate (DHT11, capacitive soil, rain) favors pathogen proliferation.
        """
        if not disease_name or "healthy" in disease_name.lower():
            return None

        d_lower = disease_name.lower().replace("_", " ")
        matched_rule = None

        for rule in DISEASE_KNOWLEDGE_BASE:
            r_id = rule["id"].lower().replace("_", " ")
            r_name = rule["name"].lower()
            if r_id in d_lower or any(word in d_lower for word in r_id.split()):
                matched_rule = rule
                break
            elif any(part in d_lower for part in ["blight", "mildew", "mold", "rot", "mite", "aphid", "anthracnose"]):
                if any(part in r_name for part in ["blight", "mildew", "mold", "rot", "mite", "aphid", "anthracnose"] if part in d_lower):
                    matched_rule = rule
                    break

        if not matched_rule:
            if "blight" in d_lower:
                matched_rule = next((r for r in DISEASE_KNOWLEDGE_BASE if r["id"] == "late_blight"), None)
            elif "mildew" in d_lower:
                matched_rule = next((r for r in DISEASE_KNOWLEDGE_BASE if r["id"] == "powdery_mildew"), None)

        if not matched_rule:
            return None

        is_favorable = False
        favorability_text = "UNKNOWN_NO_SENSORS"
        contributing_signals = []

        if temperature is not None and humidity is not None:
            r_int = int(rain) if rain is not None else 0
            sm_val = float(soil_moisture) if soil_moisture is not None else 50.0
            try:
                is_favorable = bool(matched_rule["condition"](temperature, humidity, r_int, sm_val))
                if is_favorable:
                    favorability_text = "FAVORABLE_HIGH_PATHOGEN_PRESSURE"
                    contributing_signals.append(f"Ambient microclimate ({temperature:.1f}°C, {humidity:.1f}% RH) satisfies pathogen trigger: {matched_rule['trigger']}")
                else:
                    favorability_text = "UNFAVORABLE_CONDITIONS_CONTAINED"
                    contributing_signals.append(f"Current microclimate ({temperature:.1f}°C, {humidity:.1f}% RH) does not trigger active spore release")
            except Exception as e:
                logger.warning(f"Error evaluating environmental favorability: {e}")

        return {
            "disease_id": matched_rule["id"],
            "disease_name": matched_rule["name"],
            "pathology_type": matched_rule["type"],
            "severity": matched_rule["severity"],
            "environmental_trigger": matched_rule["trigger"],
            "is_environment_favorable": is_favorable,
            "environmental_favorability": favorability_text,
            "contributing_signals": contributing_signals,
            "symptoms": matched_rule["symptoms"],
            "pesticide_prescription": matched_rule["pesticide"],
            "cultural_management": matched_rule["technique"]
        }

    def recommend(self, req: CropRecommendationRequest) -> CropRecommendationResponse:
        """
        Runs the complete agro-climatic crop inference pipeline.
        Calculates feature matrix, applies StandardScaler, computes Random Forest
        class distributions, and constructs the recommendation response.
        """
        if not self.is_ready or self.model is None:
            raise RuntimeError("Crop Recommendation model is not initialized or failed to load.")
            
        t_start = time.time()
        
        # Determine continuous rainfall representation:
        # If the caller provided a continuous rainfall_mm, prioritize it.
        # Otherwise, translate the binary rain flag using the calibrated mapping.
        if req.rainfall_mm is not None and req.rainfall_mm >= 0.0:
            effective_rainfall = float(req.rainfall_mm)
        else:
            effective_rainfall = self.RAIN_TO_MM.get(int(req.rain), 20.0)
            
        # Build pandas DataFrame with exact column names expected by scaler
        feature_df = pd.DataFrame(
            [[req.temperature, req.humidity, effective_rainfall]],
            columns=self.feature_names
        )
        
        # Standardize features using the trained pipeline scaler
        feature_scaled = self.scaler.transform(feature_df)
        
        # Compute predicted class probabilities
        probas = self.model.predict_proba(feature_scaled)[0]
        top3_indices = probas.argsort()[-3:][::-1]
        
        top3_candidates: List[CropCandidate] = []
        for rank_idx, idx in enumerate(top3_indices, start=1):
            crop_name = str(self.label_encoder.inverse_transform([int(idx)])[0])
            conf_pct = round(float(probas[idx]) * 100.0, 1)
            
            if rank_idx == 1:
                suitability = "Optimal" if conf_pct >= 40.0 else "Viable"
            elif rank_idx == 2:
                suitability = "Viable" if conf_pct >= 15.0 else "Marginal"
            else:
                suitability = "Marginal"
                
            top3_candidates.append(
                CropCandidate(
                    crop=crop_name.capitalize(),
                    confidence_pct=conf_pct,
                    rank=rank_idx,
                    suitability_level=suitability
                )
            )
            
        primary_pred_idx = int(probas.argmax())
        recommended_crop = str(self.label_encoder.inverse_transform([primary_pred_idx])[0]).capitalize()
        primary_confidence = round(float(probas[primary_pred_idx]) * 100.0, 1)
        
        # Evaluate microclimate risk alerts
        disease_alerts = self.detect_disease_risks(
            temperature=req.temperature,
            humidity=req.humidity,
            rain=req.rain,
            soil_moisture=req.soil_moisture
        )
        
        # Environmental risk flags
        rain_int = int(req.rain)
        fungal_risk = 1 if (req.humidity >= 75.0 and req.temperature >= 20.0) else 0
        drought_risk = 1 if (rain_int == 0 and req.soil_moisture < 35.0) else 0
        waterlog_risk = 1 if (rain_int == 1 or req.soil_moisture > 75.0) else 0
        
        features_used = EnvironmentalFeaturesUsed(
            temperature=req.temperature,
            humidity=req.humidity,
            soil_moisture=req.soil_moisture,
            rain=rain_int,
            rainfall_mm=round(effective_rainfall, 1),
            fungal_risk=fungal_risk,
            drought_risk=drought_risk,
            waterlog_risk=waterlog_risk
        )
        
        # Synthesize human-readable agronomic summary grounded in parameters
        summary_parts = [
            f"Based on ambient temperature of {req.temperature:.1f}°C, relative humidity of {req.humidity:.1f}%, "
            f"and effective rainfall of {effective_rainfall:.1f} mm, the system identifies {recommended_crop} as the "
            f"primary agricultural match ({primary_confidence}% confidence)."
        ]
        
        if req.soil_moisture < 30.0:
            summary_parts.append(
                f"Volumetric soil moisture is currently constrained at {req.soil_moisture:.1f}%, favoring drought-resilient "
                f"root architectures or requiring scheduled supplemental irrigation."
            )
        elif req.soil_moisture > 75.0:
            summary_parts.append(
                f"Elevated soil moisture ({req.soil_moisture:.1f}%) indicates high hydration levels, supporting moisture-demanding "
                f"crops but requiring root zone drainage monitoring."
            )
            
        if disease_alerts:
            critical_alerts = [a.name for a in disease_alerts if a.severity == "CRITICAL"]
            if critical_alerts:
                summary_parts.append(
                    f"Notice: Environmental thresholds currently trigger critical risk alerts for {', '.join(critical_alerts)}. "
                    f"Preventative cultural practices and fungicide protection are advised."
                )
            else:
                summary_parts.append(
                    f"Notice: Microclimate conditions trigger {len(disease_alerts)} active advisory watch warnings. "
                    f"Inspect foliage regularly for early disease symptoms."
                )
                
        agronomic_summary = " ".join(summary_parts)
        latency_ms = round((time.time() - t_start) * 1000.0, 2)
        
        return CropRecommendationResponse(
            status="success",
            recommended_crop=recommended_crop,
            confidence_pct=primary_confidence,
            top3_candidates=top3_candidates,
            disease_alerts=disease_alerts,
            features_used=features_used,
            agronomic_summary=agronomic_summary,
            latency_ms=latency_ms
        )

# Module singleton instance
crop_service = CropInferenceService()
