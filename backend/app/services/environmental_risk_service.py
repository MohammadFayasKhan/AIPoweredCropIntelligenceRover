"""
Environmental Risk Engine & Irrigation Intelligence Service.
Author: Team Innovex (SIH 2026 Challenge SIH26180 - Qualcomm Inc).

Provides transparent, explainable agronomic risk evaluation from field sensor telemetry:
- DHT11 microclimate (Ambient Temperature & Atmospheric Humidity)
- Capacitive Soil Moisture Sensor v1.2 (Direct root zone volumetric water content)
- Raindrop Detection Module (Precipitation presence & analog intensity)
- Water Level Sensor (Standing water / drainage / flood / waterlogging depth)

Calculates:
1. Heat Stress Index (apparent microclimate thermal load)
2. Water Stress & Drought Assessment (root hydration deficit)
3. Flood & Waterlogging Risk (hypoxia and drainage saturation)
4. Actionable Irrigation Scheduling Advisory
"""

from __future__ import annotations

import math
from typing import Tuple, List, Dict, Any
from backend.app.schemas.iot import SensorTelemetry, EnvironmentalRisk, IrrigationAdvisory


def compute_heat_index(temp_c: float, humidity_pct: float) -> float:
    """
    Calculate apparent heat index in °C using standard Rothfusz regression equation.
    Converts to Fahrenheit internally for standard meteorological constants, then returns °C.
    """
    if temp_c < 25.0 or humidity_pct < 35.0:
        # Below threshold for heat stress regression; return ambient dry bulb temperature
        return round(temp_c, 1)

    t_f = (temp_c * 9.0 / 5.0) + 32.0
    r = humidity_pct

    # Rothfusz polynomial coefficients
    hi_f = (
        -42.379
        + 2.04901523 * t_f
        + 10.14333127 * r
        - 0.22475541 * t_f * r
        - 0.00683783 * (t_f ** 2)
        - 0.05481717 * (r ** 2)
        + 0.00122874 * (t_f ** 2) * r
        + 0.00085282 * t_f * (r ** 2)
        - 0.00000199 * (t_f ** 2) * (r ** 2)
    )

    # Convert back to Celsius
    hi_c = (hi_f - 32.0) * 5.0 / 9.0
    return round(max(temp_c, hi_c), 1)


class EnvironmentalRiskEngine:
    """Production agronomic rule engine for multi-sensor field intelligence."""

    @staticmethod
    def evaluate_heat_stress(temp_c: float, humidity_pct: float) -> Tuple[float, str, List[str]]:
        """Evaluate ambient heat stress index and alert classification."""
        hi_c = compute_heat_index(temp_c, humidity_pct)
        contributions = []

        if hi_c >= 45.0:
            level = "SEVERE"
            contributions.append(f"Heat Index {hi_c}°C exceeds severe danger limit (≥45°C); high risk of crop thermal shock")
        elif hi_c >= 38.0:
            level = "HIGH"
            contributions.append(f"Heat Index {hi_c}°C in danger zone (38-44°C); stomatal closure and wilting likely")
        elif hi_c >= 32.0:
            level = "MODERATE"
            contributions.append(f"Heat Index {hi_c}°C in caution zone (32-37°C); elevated crop transpiration demand")
        else:
            level = "NORMAL"
            contributions.append(f"Heat Index {hi_c}°C within normal physiological tolerance (<32°C)")

        return hi_c, level, contributions

    @staticmethod
    def evaluate_water_stress(soil_pct: float, temp_c: float) -> Tuple[str, List[str]]:
        """Evaluate root zone moisture deficit from calibrated capacitive soil moisture v1.2."""
        contributions = []

        if soil_pct < 15.0:
            level = "SEVERE_DROUGHT"
            contributions.append(f"Capacitive soil moisture at {soil_pct:.1f}% indicates severe root-zone dehydration (<15%)")
        elif soil_pct < 28.0:
            level = "MODERATE_DROUGHT"
            contributions.append(f"Capacitive soil moisture at {soil_pct:.1f}% indicates moderate water stress (15-28%)")
        elif soil_pct < 40.0:
            level = "MILD_STRESS"
            contributions.append(f"Capacitive soil moisture at {soil_pct:.1f}% shows mild depletion below optimal field capacity")
        else:
            level = "OPTIMAL"
            contributions.append(f"Capacitive soil moisture at {soil_pct:.1f}% is within healthy field capacity (≥40%)")

        if soil_pct < 30.0 and temp_c > 35.0:
            contributions.append(f"Coupled stress: Low soil moisture ({soil_pct:.1f}%) exacerbated by high ambient temperature ({temp_c}°C)")

        return level, contributions

    @staticmethod
    def evaluate_flood_risk(
        water_level_pct: float,
        soil_pct: float,
        rain_detected: int,
        rain_intensity_pct: float | None
    ) -> Tuple[str, List[str]]:
        """Evaluate standing water accumulation and drainage saturation from water level and rain sensors."""
        contributions = []
        rain_intensity = rain_intensity_pct or (50.0 if rain_detected else 0.0)

        if water_level_pct >= 75.0 or (water_level_pct >= 55.0 and rain_detected):
            level = "CRITICAL_WATERLOGGING"
            contributions.append(
                f"Water level sensor at {water_level_pct:.1f}% with "
                f"{'active precipitation' if rain_detected else 'standing water'}; acute root hypoxia hazard"
            )
        elif water_level_pct >= 45.0 or (soil_pct >= 85.0 and rain_detected):
            level = "WARNING"
            contributions.append(
                f"Water level at {water_level_pct:.1f}% and soil saturation at {soil_pct:.1f}%; furrow drainage bottleneck"
            )
        elif water_level_pct >= 25.0 or rain_detected:
            level = "WATCH"
            contributions.append(
                f"Water level reading {water_level_pct:.1f}% with "
                f"{'precipitation detected' if rain_detected else 'mild furrow ponding'}"
            )
        else:
            level = "NORMAL"
            contributions.append("No significant standing water detected by drainage probe (<25%)")

        return level, contributions

    @classmethod
    def analyze(cls, telemetry: SensorTelemetry) -> EnvironmentalRisk:
        """Calculate composite environmental risk score and primary agronomic hazard."""
        # 1. Heat stress
        hi_c, heat_level, heat_contribs = cls.evaluate_heat_stress(
            telemetry.temperature_c, telemetry.humidity_pct
        )

        # 2. Water stress / drought
        water_stress_level, drought_contribs = cls.evaluate_water_stress(
            telemetry.soil_moisture_pct, telemetry.temperature_c
        )

        # 3. Flood / waterlogging
        flood_level, flood_contribs = cls.evaluate_flood_risk(
            telemetry.water_level_pct,
            telemetry.soil_moisture_pct,
            telemetry.rain_detected,
            telemetry.rain_intensity_pct,
        )

        all_contributions = heat_contribs + drought_contribs + flood_contribs

        # Map levels to numeric severity weights (0.0 to 1.0)
        heat_scores = {"NORMAL": 0.05, "MODERATE": 0.35, "HIGH": 0.70, "SEVERE": 0.95}
        drought_scores = {"OPTIMAL": 0.05, "MILD_STRESS": 0.30, "MODERATE_DROUGHT": 0.65, "SEVERE_DROUGHT": 0.95}
        flood_scores = {"NORMAL": 0.05, "WATCH": 0.30, "WARNING": 0.70, "CRITICAL_WATERLOGGING": 1.0}

        h_score = heat_scores.get(heat_level, 0.05)
        d_score = drought_scores.get(water_stress_level, 0.05)
        f_score = flood_scores.get(flood_level, 0.05)

        # Determine dominant primary hazard
        threat_tuples = [
            ("HEAT_STRESS", h_score),
            ("WATER_STRESS", d_score),
            ("EXCESSIVE_WATER", f_score),
        ]
        threat_tuples.sort(key=lambda x: x[1], reverse=True)
        primary_threat, max_score = threat_tuples[0]

        if max_score <= 0.15:
            primary_threat = "BALANCED_CONDITIONS"
            overall_score = 0.08
            rationale = "Field microclimate, root hydration, and surface drainage are within optimal physiological thresholds."
        else:
            # Composite non-linear risk index
            overall_score = round(min(1.0, max_score * 0.75 + (h_score + d_score + f_score) * 0.10), 2)
            if primary_threat == "HEAT_STRESS":
                rationale = f"Primary agro-threat is acute heat stress (Heat Index {hi_c}°C, {heat_level}). Stomatal defense required."
            elif primary_threat == "WATER_STRESS":
                rationale = f"Primary agro-threat is soil water depletion ({telemetry.soil_moisture_pct:.1f}%, {water_stress_level}). Urgent hydration required."
            else:
                rationale = f"Primary agro-threat is excessive water accumulation ({telemetry.water_level_pct:.1f}%, {flood_level}). Drainage action required."

        return EnvironmentalRisk(
            heat_stress_index=hi_c,
            heat_risk_level=heat_level,
            water_stress_level=water_stress_level,
            flood_risk_level=flood_level,
            overall_risk_score=overall_score,
            primary_threat=primary_threat,
            contributing_signals=all_contributions,
            rationale=rationale,
        )

    @classmethod
    def generate_irrigation_advisory(
        cls, telemetry: SensorTelemetry, risk: EnvironmentalRisk
    ) -> IrrigationAdvisory:
        """
        Derives actionable irrigation scheduling advisory.
        Separates drought stress (soil moisture) from standing water (water level).
        """
        soil = telemetry.soil_moisture_pct
        rain = telemetry.rain_detected
        water_level = telemetry.water_level_pct

        # 1. Flood / waterlogged condition
        if risk.flood_risk_level in ("WARNING", "CRITICAL_WATERLOGGING") or water_level >= 50.0:
            return IrrigationAdvisory(
                recommendation="EXCESSIVE_WATER_RISK",
                water_volume_proxy="NONE",
                urgency="IMMEDIATE",
                rationale=(
                    f"Water level probe detects standing water at {water_level:.1f}% depth. "
                    "Irrigation must be prohibited and drainage furrows cleared to avoid root rot."
                ),
            )

        # 2. Active rainfall
        if rain == 1 or (telemetry.rain_intensity_pct and telemetry.rain_intensity_pct > 20.0):
            return IrrigationAdvisory(
                recommendation="DELAY_IRRIGATION",
                water_volume_proxy="NONE",
                urgency="LOW",
                rationale="Active rainfall detected by raindrop module. Suspend all automated irrigation cycles.",
            )

        # 3. Severe soil water deficit
        if soil < 18.0:
            return IrrigationAdvisory(
                recommendation="IRRIGATE_NOW",
                water_volume_proxy="DEEP_ROOT_SOAKING",
                urgency="IMMEDIATE",
                rationale=(
                    f"Critical root-zone water deficit detected (Soil Moisture: {soil:.1f}%). "
                    "Deliver deep-root soaking immediately to avert permanent wilting."
                ),
            )

        # 4. Moderate water deficit
        if soil < 30.0:
            vol = "STANDARD_IRRIGATION" if telemetry.temperature_c < 32.0 else "DEEP_ROOT_SOAKING"
            return IrrigationAdvisory(
                recommendation="IRRIGATE_NOW",
                water_volume_proxy=vol,
                urgency="HIGH",
                rationale=(
                    f"Soil moisture is depleted ({soil:.1f}%) below the 30% safety margin. "
                    "Standard irrigation cycle recommended during morning or evening."
                ),
            )

        # 5. Mild depletion
        if soil < 45.0:
            return IrrigationAdvisory(
                recommendation="MONITOR",
                water_volume_proxy="LIGHT_APPLICATION",
                urgency="MEDIUM",
                rationale=(
                    f"Soil moisture at {soil:.1f}% is adequate but approaching lower boundary. "
                    "Monitor soil trend over next 6-12 hours."
                ),
            )

        # 6. Well hydrated / High moisture
        if soil > 80.0:
            return IrrigationAdvisory(
                recommendation="DELAY_IRRIGATION",
                water_volume_proxy="NONE",
                urgency="LOW",
                rationale=(
                    f"Soil is near maximum saturation at {soil:.1f}%. "
                    "Hold irrigation until natural percolation and transpiration deplete moisture."
                ),
            )

        # 7. Optimal
        return IrrigationAdvisory(
            recommendation="MONITOR",
            water_volume_proxy="NONE",
            urgency="LOW",
            rationale=(
                f"Soil moisture ({soil:.1f}%) is in the optimal agronomic envelope (45-80%). "
                "No irrigation intervention needed."
            ),
        )


environmental_risk_engine = EnvironmentalRiskEngine()
