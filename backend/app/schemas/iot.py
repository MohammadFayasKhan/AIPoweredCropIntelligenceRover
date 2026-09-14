"""
Pydantic Schemas for AgriRover IoT, Field Sensors, and Unified Environmental Intelligence.
Defines typed contracts for ESP32 telemetry, GPS NEO-6M spatial telemetry, 4-DOF robotic arm state,
capacitive soil moisture, water level, rain intensity, DHT11 microclimate,
environmental risk reasoning, and two-way teleoperation commands.
"""

from __future__ import annotations

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime, timezone


def _utc_now_str() -> str:
    return datetime.now(timezone.utc).isoformat()


class GPSData(BaseModel):
    """Spatial telemetry parsed from GPS NEO-6M-0-001 hardware UART."""
    latitude: Optional[float] = Field(None, ge=-90.0, le=90.0, description="WGS84 latitude coordinate in degrees")
    longitude: Optional[float] = Field(None, ge=-180.0, le=180.0, description="WGS84 longitude coordinate in degrees")
    altitude_m: Optional[float] = Field(None, description="Altitude above sea level in meters")
    satellites_tracked: int = Field(0, ge=0, description="Count of visible satellites currently tracked")
    hdop: Optional[float] = Field(None, description="Horizontal Dilution of Precision (spatial accuracy proxy)")
    fix_state: str = Field("NO_FIX", description="GPS fix status: NO_FIX, 2D_FIX, 3D_FIX, DGPS_FIX")
    is_valid: bool = Field(False, description="True only when satellite fix passes WGS84 validity bounds")
    fix_age_ms: Optional[int] = Field(None, description="Age of last valid NMEA fix in milliseconds")


class SensorTelemetry(BaseModel):
    """Live field environmental sensor group readings."""
    # DHT11 Temperature & Relative Humidity
    temperature_c: float = Field(..., ge=-20.0, le=70.0, description="Ambient air temperature in °C")
    humidity_pct: float = Field(..., ge=0.0, le=100.0, description="Relative atmospheric humidity %")
    dht_status: str = Field("VALID", description="DHT11 data freshness: VALID, STALE, SENSOR_FAULT")

    # Capacitive Soil Moisture Sensor v1.2 (primary drought / root water status)
    soil_moisture_raw: int = Field(..., ge=0, le=4095, description="12-bit ADC raw output on ESP32 GPIO 34")
    soil_moisture_pct: float = Field(..., ge=0.0, le=100.0, description="Calibrated volumetric water content %")
    soil_calibration_status: str = Field("CALIBRATED", description="CALIBRATED, DEFAULT_REF, OUT_OF_BOUNDS")

    # Raindrop Detection Module (precipitation proxy)
    rain_detected: int = Field(0, ge=0, le=1, description="Digital DO flag: 1 if surface raindrops detected, 0 dry")
    rain_intensity_raw: Optional[int] = Field(None, ge=0, le=4095, description="Analog AO 12-bit ADC value (GPIO 32)")
    rain_intensity_pct: Optional[float] = Field(None, ge=0.0, le=100.0, description="Calibrated precipitation intensity proxy %")

    # Water Level Sensor (standing water / drainage / flood / waterlogging depth)
    water_level_raw: int = Field(0, ge=0, le=4095, description="12-bit ADC reading on ESP32 GPIO 35")
    water_level_pct: float = Field(0.0, ge=0.0, le=100.0, description="Normalized standing water depth %")
    water_state: str = Field("NORMAL", description="Water level state: NORMAL, RETENTION_WATCH, WATERLOGGED, SUBMERGED")
    sensor_semantic_role: str = Field("FIELD_DRAINAGE", description="Semantic role: FIELD_DRAINAGE, FURROW, TANK")


class RoverState(BaseModel):
    """AgriRover mobility and chassis watchdog status."""
    movement_state: str = Field("STOPPED", description="Active drive state: FORWARD, BACKWARD, LEFT, RIGHT, STOPPED")
    speed_pwm: int = Field(200, ge=0, le=255, description="L293D motor drive speed PWM")
    headlight_on: bool = Field(False, description="Field inspection headlight LED status on GPIO 2")
    watchdog_active: bool = Field(True, description="Local 600ms motor safety watchdog state")
    watchdog_timeout_ms: int = Field(600, description="Local safety timeout window in milliseconds")
    wifi_rssi: Optional[int] = Field(None, description="Wi-Fi signal strength in dBm")
    local_ip: Optional[str] = Field(None, description="Assigned STA or AP IP address")
    uptime_seconds: int = Field(0, ge=0, description="ESP32 hardware uptime since boot")
    command_counter: int = Field(0, ge=0, description="Total movement commands acknowledged")


class ArmState(BaseModel):
    """4-DOF robotic manipulator stance and PCA9685 controller state."""
    connected: bool = Field(True, description="True if PCA9685 responds on I2C address 0x40")
    is_moving: bool = Field(False, description="True while smooth kinematics interpolation is active")
    emergency_stopped: bool = Field(False, description="Safety override: true if emergency stop is triggered")
    base_deg: float = Field(90.0, ge=5.0, le=175.0, description="Waist rotation servo angle (Channel 0)")
    shoulder_deg: float = Field(90.0, ge=0.0, le=180.0, description="Shoulder lift servo angle (Channel 1)")
    elbow_deg: float = Field(90.0, ge=0.0, le=180.0, description="Elbow reach servo angle (Channel 2)")
    gripper_deg: float = Field(180.0, ge=70.0, le=180.0, description="Gripper servo angle (Channel 3, 180=Open)")
    gripper_state: str = Field("OPEN", description="OPEN (180°), CLOSED (70°), or INTERMEDIATE")
    current_preset: Optional[str] = Field("HOME", description="HOME, INSPECTION, RETRACTED, REACH, CUSTOM")
    fault_code: Optional[str] = Field(None, description="Arm controller diagnostic fault code if any")


class EnvironmentalRisk(BaseModel):
    """Transparent, explainable agronomic risk indices."""
    heat_stress_index: float = Field(..., description="Calculated apparent heat index temperature in °C")
    heat_risk_level: str = Field(..., description="NORMAL, MODERATE, HIGH, SEVERE")
    water_stress_level: str = Field(..., description="OPTIMAL, MILD_STRESS, MODERATE_DROUGHT, SEVERE_DROUGHT")
    flood_risk_level: str = Field(..., description="NORMAL, WATCH, WARNING, CRITICAL_WATERLOGGING")
    overall_risk_score: float = Field(..., ge=0.0, le=1.0, description="Composite agronomic threat score (0.0=Safe to 1.0=Hazard)")
    primary_threat: str = Field(..., description="Dominant immediate environmental risk")
    contributing_signals: List[str] = Field(default_factory=list, description="Sensor factors driving the risk calculation")
    rationale: str = Field(..., description="Transparent technical explanation of derived risk")


class IrrigationAdvisory(BaseModel):
    """Actionable irrigation scheduling recommendations."""
    recommendation: str = Field(..., description="IRRIGATE_NOW, MONITOR, DELAY_IRRIGATION, EXCESSIVE_WATER_RISK, INSUFFICIENT_DATA")
    water_volume_proxy: str = Field("NONE", description="NONE, LIGHT_APPLICATION, STANDARD_IRRIGATION, DEEP_ROOT_SOAKING")
    urgency: str = Field("LOW", description="LOW, MEDIUM, HIGH, IMMEDIATE")
    rationale: str = Field(..., description="Soil moisture and weather conditions driving the irrigation decision")


class UnifiedObservation(BaseModel):
    """
    Canonical, versioned field observation contract.
    Combines rover kinematics, robotic arm state, microclimate, soil water, rain,
    water level, GPS spatial coordinates, Crop Intelligence recommendation, and Crop Vision inference.
    """
    observation_id: str = Field(..., description="Unique UUID or timestamp-keyed observation identifier")
    device_id: str = Field("agrirover-esp32-01", description="Authoritative field controller identifier")
    timestamp: str = Field(default_factory=_utc_now_str, description="UTC ISO8601 timestamp")
    
    # Raw & Normalized Telemetry
    sensor_telemetry: SensorTelemetry = Field(..., description="Ground sensor group measurements")
    gps: GPSData = Field(default_factory=GPSData, description="Geographic location and satellite fix")
    rover_state: RoverState = Field(default_factory=RoverState, description="Chassis drive status and safety watchdog")
    arm_state: ArmState = Field(default_factory=ArmState, description="4-DOF robotic manipulator stance")
    
    # Intelligence Layer Inferences
    environmental_risk: Optional[EnvironmentalRisk] = Field(None, description="Derived agronomic hazard evaluation")
    irrigation_advisory: Optional[IrrigationAdvisory] = Field(None, description="Actionable water management guidance")
    crop_intelligence: Optional[Dict[str, Any]] = Field(None, description="Crop recommendation and microclimate pathogen alerts")
    crop_vision: Optional[Dict[str, Any]] = Field(None, description="EfficientNetV2-S + YOLO11 vision diagnosis")
    
    # Metadata & Quality Assurance
    is_simulated: bool = Field(False, description="True if generated by synthetic hardware mock loop")
    processing_latency_ms: Optional[float] = Field(None, description="End-to-end backend processing duration in ms")


class DeviceCommandRequest(BaseModel):
    """Structured teleoperation or arm actuation command sent to ESP32."""
    command_id: str = Field(..., description="Correlation ID for command tracing and deduplication")
    device_id: str = Field("agrirover-esp32-01", description="Target hardware controller ID")
    command_type: str = Field(..., description="ROVER_MOVE, ROVER_STOP, HEADLIGHT, ARM_TARGET, ARM_GRIPPER, ARM_HOME, ARM_EMERGENCY_STOP, INSPECTION_RUN")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Command-specific parameters (direction, angle, speed, etc.)")
    source: str = Field("DASHBOARD", description="Initiator: DASHBOARD, XIAOZHI_VOICE, N8N_AUTOMATION, EMERGENCY_OVERRIDE")
    timestamp: str = Field(default_factory=_utc_now_str)


class DeviceCommandResponse(BaseModel):
    """Acknowledgement returned by backend command gateway."""
    command_id: str
    status: str = Field("ACCEPTED", description="ACCEPTED, EXECUTING, REJECTED, FAILED, COMPLETED")
    message: str
    safety_validated: bool = Field(True, description="True if command passed physical envelope and rate checks")
    timestamp: str = Field(default_factory=_utc_now_str)


class DeviceRegistration(BaseModel):
    """Hardware capability handshake on ESP32 boot."""
    device_id: str
    firmware_version: str
    hardware_profile: str = "ESP32_WROOM_32D"
    protocol_version: str = "2.0.0"
    capabilities: List[str] = Field(
        default_factory=lambda: [
            "rover_control", "arm_control", "dht11", "capacitive_soil_moisture",
            "raindrop_sensor", "water_level_sensor", "gps_neo6m", "tft_display_2_4in"
        ]
    )
    uptime_ms: int = 0
    free_heap_bytes: Optional[int] = None
