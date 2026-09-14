"""
Comprehensive Integration and Unit Tests for AgriRover IoT & Environmental Intelligence.
Author: Team Innovex (SIH 2026 Challenge SIH26180 - Qualcomm Inc).

Tests:
1. SensorTelemetry & UnifiedObservation schema validation.
2. EnvironmentalRiskEngine: Heat Index, Drought assessment, Flood risk, and Irrigation Advisory.
3. IoT API Endpoints: /api/v1/iot/observation, /observation/latest, /device/state.
4. Device Command Gateway: Direction validation, Arm angle limit enforcement, Emergency Stop.
5. XiaoZhi Voice Assistant & MCP tool definitions and execution.
6. n8n automation alert dispatch and cooldown deduplication.
"""

import pytest
from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.schemas.iot import (
    SensorTelemetry,
    GPSData,
    RoverState,
    ArmState,
    UnifiedObservation,
    DeviceCommandRequest,
)
from backend.app.services.environmental_risk_service import (
    compute_heat_index,
    environmental_risk_engine,
)
from backend.app.services.xiaozhi_gateway import xiaozhi_gateway
from backend.app.services.n8n_service import n8n_dispatcher

client = TestClient(app)


def test_heat_index_calculation():
    """Verify standard Rothfusz heat index regression equations."""
    # Under 25°C, returns dry bulb ambient temperature
    assert compute_heat_index(22.0, 50.0) == 22.0

    # 35°C at 75% humidity produces severe heat index
    hi = compute_heat_index(35.0, 75.0)
    assert hi > 45.0  # Acute heat hazard


def test_environmental_risk_engine_drought_and_heat():
    """Verify drought detection from capacitive soil moisture and heat stress from DHT11."""
    telemetry = SensorTelemetry(
        temperature_c=39.0,
        humidity_pct=65.0,
        dht_status="VALID",
        soil_moisture_raw=3100,
        soil_moisture_pct=12.0,  # Critical root dehydration
        soil_calibration_status="CALIBRATED",
        rain_detected=0,
        water_level_raw=200,
        water_level_pct=5.0,
        water_state="NORMAL",
    )

    risk = environmental_risk_engine.analyze(telemetry)
    assert risk.heat_risk_level in ("HIGH", "SEVERE")
    assert risk.water_stress_level == "SEVERE_DROUGHT"
    assert risk.flood_risk_level == "NORMAL"
    assert risk.overall_risk_score > 0.70

    advisory = environmental_risk_engine.generate_irrigation_advisory(telemetry, risk)
    assert advisory.recommendation == "IRRIGATE_NOW"
    assert advisory.water_volume_proxy == "DEEP_ROOT_SOAKING"
    assert advisory.urgency == "IMMEDIATE"


def test_environmental_risk_engine_flood_and_rain():
    """Verify flood and waterlogging alert when water level is high with active rain."""
    telemetry = SensorTelemetry(
        temperature_c=24.0,
        humidity_pct=92.0,
        dht_status="VALID",
        soil_moisture_raw=1100,
        soil_moisture_pct=95.0,  # Fully saturated soil
        soil_calibration_status="CALIBRATED",
        rain_detected=1,         # Active rainfall
        rain_intensity_pct=80.0,
        water_level_raw=2800,
        water_level_pct=78.0,    # High standing water
        water_state="WATERLOGGED",
    )

    risk = environmental_risk_engine.analyze(telemetry)
    assert risk.flood_risk_level == "CRITICAL_WATERLOGGING"
    assert risk.primary_threat == "EXCESSIVE_WATER"

    advisory = environmental_risk_engine.generate_irrigation_advisory(telemetry, risk)
    assert advisory.recommendation == "EXCESSIVE_WATER_RISK"
    assert advisory.water_volume_proxy == "NONE"


def test_iot_observation_ingestion_and_retrieval():
    """Test full round-trip of /api/v1/iot/observation and /observation/latest."""
    payload = {
        "observation_id": "obs-test-unit-001",
        "device_id": "agrirover-esp32-01",
        "sensor_telemetry": {
            "temperature_c": 28.0,
            "humidity_pct": 58.0,
            "dht_status": "VALID",
            "soil_moisture_raw": 2100,
            "soil_moisture_pct": 55.0,
            "soil_calibration_status": "CALIBRATED",
            "rain_detected": 0,
            "rain_intensity_raw": 4095,
            "rain_intensity_pct": 0.0,
            "water_level_raw": 350,
            "water_level_pct": 8.0,
            "water_state": "NORMAL",
            "sensor_semantic_role": "FIELD_DRAINAGE",
        },
        "gps": {
            "latitude": 13.0827,
            "longitude": 80.2707,
            "altitude_m": 12.0,
            "satellites_tracked": 7,
            "hdop": 1.2,
            "fix_state": "3D_FIX",
            "is_valid": True,
        },
        "rover_state": {
            "movement_state": "STOPPED",
            "speed_pwm": 200,
            "headlight_on": False,
            "watchdog_active": True,
            "watchdog_timeout_ms": 600,
            "uptime_seconds": 120,
        },
        "arm_state": {
            "connected": True,
            "is_moving": False,
            "emergency_stopped": False,
            "base_deg": 90.0,
            "shoulder_deg": 90.0,
            "elbow_deg": 90.0,
            "gripper_deg": 180.0,
            "gripper_state": "OPEN",
        },
        "is_simulated": False,
    }

    # Ingest
    res_post = client.post("/api/v1/iot/observation", json=payload)
    assert res_post.status_code == 200
    data = res_post.json()
    assert data["observation_id"] == "obs-test-unit-001"
    assert "environmental_risk" in data
    assert "irrigation_advisory" in data

    # Retrieve latest
    res_latest = client.get("/api/v1/iot/observation/latest")
    assert res_latest.status_code == 200
    latest_data = res_latest.json()
    assert latest_data["observation_id"] == "obs-test-unit-001"
    assert latest_data["sensor_telemetry"]["temperature_c"] == 28.0

    # Verify legacy /latest synchronization
    res_legacy = client.get("/latest")
    assert res_legacy.status_code == 200
    legacy_data = res_legacy.json()
    assert legacy_data["temperature"] == 28.0
    assert legacy_data["soil_moisture"] == 55.0


def test_device_command_safety_envelope():
    """Verify physical safety boundary checks on rover and robotic arm commands."""
    # 1. Valid rover forward command
    valid_rover_cmd = {
        "command_id": "cmd-test-01",
        "device_id": "agrirover-esp32-01",
        "command_type": "ROVER_MOVE",
        "parameters": {"direction": "FORWARD", "speed": 210},
        "source": "DASHBOARD",
    }
    res = client.post("/api/v1/iot/device/command", json=valid_rover_cmd)
    assert res.status_code == 200
    assert res.json()["status"] == "ACCEPTED"
    assert res.json()["safety_validated"] is True

    # 2. Invalid rover direction
    invalid_rover_cmd = {
        "command_id": "cmd-test-02",
        "device_id": "agrirover-esp32-01",
        "command_type": "ROVER_MOVE",
        "parameters": {"direction": "FLY_UPWARD"},
        "source": "DASHBOARD",
    }
    res_bad = client.post("/api/v1/iot/device/command", json=invalid_rover_cmd)
    assert res_bad.status_code == 200
    assert res_bad.json()["status"] == "REJECTED"
    assert res_bad.json()["safety_validated"] is False

    # 3. Arm angle out-of-bounds rejection (Base must be 5° to 175°)
    invalid_arm_cmd = {
        "command_id": "cmd-test-03",
        "device_id": "agrirover-esp32-01",
        "command_type": "ARM_TARGET",
        "parameters": {"base": 240.0},
        "source": "DASHBOARD",
    }
    res_arm_bad = client.post("/api/v1/iot/device/command", json=invalid_arm_cmd)
    assert res_arm_bad.status_code == 200
    assert res_arm_bad.json()["status"] == "REJECTED"
    assert "safety limits" in res_arm_bad.json()["message"]


def test_xiaozhi_voice_assistant_and_mcp():
    """Test XiaoZhi MCP tool discovery and conversational execution."""
    # 1. List MCP tools
    tools_res = client.get("/api/v1/xiaozhi/mcp/tools")
    assert tools_res.status_code == 200
    tools = tools_res.json()["tools"]
    tool_names = [t["name"] for t in tools]
    assert "get_current_rover_status" in tool_names
    assert "get_sensor_status" in tool_names
    assert "get_current_environment" in tool_names
    assert "move_rover" in tool_names

    # 2. Execute read tool
    exec_res = client.post(
        "/api/v1/xiaozhi/execute",
        json={"tool_name": "get_sensor_status", "arguments": {}},
    )
    assert exec_res.status_code == 200
    result = exec_res.json()["result"]
    assert "temperature_c" in result

    # 3. Conversational query
    chat_res = client.post(
        "/api/v1/xiaozhi/chat",
        json={"query": "What is the current field temperature and humidity?", "language": "en"},
    )
    assert chat_res.status_code == 200
    assert "temperature" in chat_res.json()["response"].lower()


def test_n8n_alert_cooldown_deduplication():
    """Verify that n8n dispatcher suppresses duplicate alerts during cooldown."""
    dispatcher = n8n_dispatcher
    assert dispatcher.is_in_cooldown("heat_risk_detected", "test_plot_1") is False

    dispatcher.mark_dispatched("heat_risk_detected", "test_plot_1")
    assert dispatcher.is_in_cooldown("heat_risk_detected", "test_plot_1") is True
    # Different key should not be in cooldown
    assert dispatcher.is_in_cooldown("heat_risk_detected", "test_plot_2") is False


def test_disease_knowledge_base_environmental_favorability():
    """Verify disease knowledge engine correctly evaluates pathogen trigger conditions against sensor telemetry."""
    from backend.app.services.crop_service import crop_service

    # Case 1: Late Blight with high humidity (88%) and temperate climate (20°C) -> Favorable
    dk_favorable = crop_service.get_disease_knowledge(
        disease_name="Tomato Late Blight",
        temperature=20.0,
        humidity=88.0,
        rain=1,
        soil_moisture=70.0
    )
    assert dk_favorable is not None
    assert dk_favorable["disease_id"] == "late_blight"
    assert dk_favorable["is_environment_favorable"] is True
    assert dk_favorable["environmental_favorability"] == "FAVORABLE_HIGH_PATHOGEN_PRESSURE"
    assert "Mancozeb" in dk_favorable["pesticide_prescription"]

    # Case 2: Late Blight with low humidity (40%) and high temperature (35°C) -> Unfavorable
    dk_unfavorable = crop_service.get_disease_knowledge(
        disease_name="Tomato Late Blight",
        temperature=35.0,
        humidity=40.0,
        rain=0,
        soil_moisture=30.0
    )
    assert dk_unfavorable is not None
    assert dk_unfavorable["is_environment_favorable"] is False
    assert dk_unfavorable["environmental_favorability"] == "UNFAVORABLE_CONDITIONS_CONTAINED"

    # Case 3: Healthy foliage -> No disease knowledge match
    dk_healthy = crop_service.get_disease_knowledge("Tomato Healthy")
    assert dk_healthy is None


def test_multimodal_vision_enrichment_with_iot_context():
    """Verify that leaf diagnosis attaches live field sensor telemetry and evaluates disease knowledge."""
    from tests.test_api import get_test_sample_images
    val_images = get_test_sample_images()
    sample_path = val_images[0]
    with open(sample_path, "rb") as f:
        file_bytes = f.read()

    # 1. Ingest known IoT observation
    payload = {
        "observation_id": "obs-test-enrichment-01",
        "device_id": "agrirover-esp32-01",
        "sensor_telemetry": {
            "temperature_c": 22.5,
            "humidity_pct": 89.0,
            "soil_moisture_raw": 2100,
            "soil_moisture_pct": 65.0,
            "rain_detected": 1,
            "rain_intensity_raw": 1500,
            "rain_intensity_pct": 85.0,
            "water_level_raw": 600,
            "water_level_pct": 18.0,
        },
        "gps": {
            "fix_valid": True,
            "latitude": 12.971598,
            "longitude": 77.594562,
            "satellites": 9,
            "hdop": 1.1,
        }
    }
    obs_res = client.post("/api/v1/iot/observation", json=payload)
    assert obs_res.status_code == 200

    # 2. Diagnose image with attach_iot_context=True
    diag_res = client.post(
        "/api/v1/vision/diagnose",
        files={"file": (sample_path.name, file_bytes, "image/jpeg")},
        data={"model_tier": "server", "attach_iot_context": "true"}
    )
    assert diag_res.status_code == 200
    data = diag_res.json()

    assert data["environmental_context"] is not None
    assert data["environmental_context"]["temperature_c"] == 22.5
    assert data["environmental_context"]["humidity_pct"] == 89.0
    assert data["environmental_context"]["rain_detected"] == 1
    assert data["environmental_context"]["gps_fix"] is True
    assert data["environmental_context"]["observation_id"] == "obs-test-enrichment-01"

