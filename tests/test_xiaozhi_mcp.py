"""
Tests for XiaoZhi AI Voice Assistant & Model Context Protocol (MCP) Integration.
Validates tool schemas, tool execution, freshness categorization, knowledge lookup,
physical actuation blocking, and standard JSON-RPC 2.0 protocol handling.
"""

import json
import time
import pytest
from datetime import datetime, timezone

from backend.app.schemas.iot import (
    UnifiedObservation,
    SensorTelemetry,
    GPSData,
    RoverState,
    ArmState,
    EnvironmentalRisk,
    IrrigationAdvisory,
)
from backend.app.services.xiaozhi_gateway import (
    xiaozhi_gateway,
    compute_freshness,
)
from backend.app.services.xiaozhi_mcp_client import (
    xiaozhi_mcp_client,
    mask_token,
)


EXPECTED_TOOLS = {
    "get_current_field_status",
    "get_latest_observation",
    "get_observation_history",
    "get_device_status",
    "get_crop_recommendation",
    "get_disease_status",
    "get_environmental_risk",
    "get_irrigation_status",
    "get_crop_or_disease_knowledge",
    "get_latest_image_analysis",
    "get_field_summary",
    "search_current_agricultural_guidance",
    "search_pesticide_guidance",
    "search_current_product_prices",
    "search_weather",
    "estimate_treatment_budget",
}



@pytest.fixture
def sample_live_observation():
    """Generates an authentic unified field observation timestamped to now."""
    now_iso = datetime.now(timezone.utc).isoformat()
    return UnifiedObservation(
        observation_id="obs-test-xz-001",
        device_id="agrirover-esp32-node01",
        timestamp=now_iso,
        sensor_telemetry=SensorTelemetry(
            temperature_c=28.5,
            humidity_pct=76.0,
            dht_status="VALID",
            soil_moisture_raw=2150,
            soil_moisture_pct=34.0,
            soil_calibration_status="CALIBRATED",
            rain_detected=0,
            rain_intensity_raw=4095,
            rain_intensity_pct=0.0,
            water_level_raw=550,
            water_level_pct=14.5,
            water_state="NORMAL",
        ),
        gps=GPSData(
            latitude=13.0827,
            longitude=80.2707,
            altitude_m=16.2,
            satellites_tracked=8,
            hdop=1.2,
            fix_state="3D_FIX",
            is_valid=True,
        ),
        rover_state=RoverState(
            movement_state="STOPPED",
            speed_pwm=0,
            headlight_on=False,
            watchdog_active=True,
            uptime_seconds=3600,
            wifi_rssi=-62,
            local_ip="192.168.1.105",
        ),
        arm_state=ArmState(
            base_deg=90.0,
            shoulder_deg=90.0,
            elbow_deg=90.0,
            gripper_deg=180.0,
            gripper_state="OPEN",
            is_moving=False,
        ),
        environmental_risk=EnvironmentalRisk(
            heat_stress_index=31.2,
            heat_risk_level="NORMAL",
            water_stress_level="MILD_STRESS",
            flood_risk_level="NORMAL",
            overall_risk_score=0.35,
            primary_threat="Water Deficit",
            rationale="Soil moisture is below optimal 40% threshold.",
            contributing_factors=["Soil moisture at 34%"],
        ),
        irrigation_advisory=IrrigationAdvisory(
            recommendation="MONITOR",
            water_volume_proxy="NONE",
            urgency="MODERATE",
            rationale="Moisture is slightly below optimal. Monitor field before irrigating.",
        ),
        crop_intelligence={
            "recommended_crop": "rice",
            "confidence_pct": 92.5,
            "crop_candidates": [
                {"crop": "rice", "probability_pct": 92.5},
                {"crop": "jute", "probability_pct": 5.0},
            ],
            "environmental_features_used": {"temperature": 28.5, "humidity": 76.0},
        },
        crop_vision={
            "crop_name": "Tomato",
            "disease_name": "Early Blight",
            "confidence_pct": 88.0,
            "is_healthy": False,
            "severity_level": "Moderate",
            "foliar_damage_pct": 14.2,
            "timestamp": now_iso,
        },
    )


def test_mcp_tool_definitions_completeness():
    """Verify all 11 required farmer-oriented tools are exposed with valid JSON Schema."""
    tools = xiaozhi_gateway.get_mcp_tool_definitions()
    tool_names = {t["name"] for t in tools}

    assert EXPECTED_TOOLS.issubset(tool_names), f"Missing tools: {EXPECTED_TOOLS - tool_names}"
    assert len(tools) >= len(EXPECTED_TOOLS)

    for tool in tools:
        assert "name" in tool and isinstance(tool["name"], str)
        assert "description" in tool and len(tool["description"]) > 10
        assert "parameters" in tool
        assert tool["parameters"]["type"] == "object"


def test_freshness_computation():
    """Verify temporal freshness categorizes observations into LIVE, RECENT, STALE, OFFLINE."""
    now = time.time()

    # < 60s
    status_live, age_live = compute_freshness(now - 15)
    assert status_live == "LIVE"
    assert 14.0 <= age_live <= 16.0

    # 5 minutes (300s)
    status_recent, _ = compute_freshness(now - 300)
    assert status_recent == "RECENT"

    # 25 minutes (1500s)
    status_stale, _ = compute_freshness(now - 1500)
    assert status_stale == "STALE"

    # 2 hours (7200s)
    status_offline, _ = compute_freshness(now - 7200)
    assert status_offline == "OFFLINE"

    # None / Missing
    status_none, _ = compute_freshness(None)
    assert status_none == "NO_DATA"


def test_get_current_field_status(sample_live_observation):
    """Verify get_current_field_status returns accurate live readings and speech summary."""
    xiaozhi_gateway.update_observation(sample_live_observation)
    res = xiaozhi_gateway.execute_tool("get_current_field_status", {})

    assert res["status"] == "ok"
    assert res["temperature_c"] == 28.5
    assert res["humidity_pct"] == 76.0
    assert res["soil_moisture_pct"] == 34.0
    assert res["rain_state"] == "Dry"
    assert res["freshness"] == "LIVE"
    assert res["gps"]["has_fix"] is True
    assert "spoken_summary" in res
    assert "28.5" in res["spoken_summary"]
    assert "34" in res["spoken_summary"]


def test_get_crop_recommendation(sample_live_observation):
    """Verify crop recommendation cites Random Forest model output truthfully."""
    xiaozhi_gateway.update_observation(sample_live_observation)
    res = xiaozhi_gateway.execute_tool("get_crop_recommendation", {})

    assert res["status"] == "ok"
    assert res["recommended_crop"] == "rice"
    assert res["confidence_pct"] == 92.5
    assert "rice" in res["spoken_summary"].lower()


def test_get_disease_status_separation(sample_live_observation):
    """Verify disease diagnosis is strictly decoupled from microclimate environmental risk."""
    xiaozhi_gateway.update_observation(sample_live_observation)
    res = xiaozhi_gateway.execute_tool("get_disease_status", {})

    assert res["status"] == "ok"
    assert res["has_diagnosis"] is True
    assert res["disease_name"] == "Early Blight"
    assert res["crop_name"] == "Tomato"
    assert res["is_healthy"] is False
    assert res["environmental_disease_risk"] == "Water Deficit"
    assert "early blight" in res["spoken_summary"].lower()


def test_get_irrigation_status(sample_live_observation):
    """Verify irrigation status reflects backend irrigation advisory."""
    xiaozhi_gateway.update_observation(sample_live_observation)
    res = xiaozhi_gateway.execute_tool("get_irrigation_status", {})

    assert res["status"] == "ok"
    assert res["recommendation"] == "MONITOR"
    assert res["soil_moisture_pct"] == 34.0
    assert "monitoring" in res["spoken_summary"].lower() or "monitor" in res["spoken_summary"].lower()


def test_get_crop_or_disease_knowledge():
    """Verify agronomic knowledge lookup retrieves actionable advice from knowledge bases."""
    # 1. Existing disease lookup
    res_blight = xiaozhi_gateway.execute_tool("get_crop_or_disease_knowledge", {"query": "late blight"})
    assert res_blight["status"] == "ok"
    assert "Late Blight" in res_blight["disease_name"]
    assert res_blight["immediate_action"] is not None
    assert "spoken_summary" in res_blight

    # 2. Non-existent lookup
    res_fake = xiaozhi_gateway.execute_tool("get_crop_or_disease_knowledge", {"query": "martian fungus"})
    assert res_fake["status"] == "NOT_FOUND"


def test_get_field_summary(sample_live_observation):
    """Verify field summary produces complete compact farmer synthesis."""
    xiaozhi_gateway.update_observation(sample_live_observation)
    res = xiaozhi_gateway.execute_tool("get_field_summary", {})

    assert res["status"] == "ok"
    assert res["soil_moisture_pct"] == 34.0
    assert res["recommended_crop"] == "rice"
    assert res["irrigation_recommendation"] == "MONITOR"
    assert "spoken_summary" in res
    assert "rice" in res["spoken_summary"].lower()


def test_dynamic_follow_up_suggestions_presence(sample_live_observation):
    """Verify EVERY tool execution attaches 2 to 4 dynamic context-aware follow-up suggestions."""
    xiaozhi_gateway.update_observation(sample_live_observation)
    for tool_name in EXPECTED_TOOLS:
        args = {}
        if tool_name == "get_crop_or_disease_knowledge":
            args = {"query": "early blight"}
        elif tool_name == "search_current_agricultural_guidance":
            args = {"topic": "late blight management"}
        elif tool_name == "search_pesticide_guidance":
            args = {"crop": "tomato", "disease_or_pest": "early blight"}
        elif tool_name == "search_current_product_prices":
            args = {"product_name": "Mancozeb"}
        elif tool_name == "estimate_treatment_budget":
            args = {"crop": "tomato", "problem": "early blight", "field_area_acres": 1.0}

        res = xiaozhi_gateway.execute_tool(tool_name, args)
        assert "follow_up_suggestions" in res, f"Tool {tool_name} missing follow_up_suggestions"
        suggestions = res["follow_up_suggestions"]
        assert isinstance(suggestions, list)
        assert 2 <= len(suggestions) <= 4, f"Tool {tool_name} generated {len(suggestions)} suggestions (expected 2-4)"
        for s in suggestions:
            assert isinstance(s, str) and len(s) > 5 and s.endswith("?")


def test_search_current_agricultural_guidance():
    """Verify agricultural search retrieves authoritative ICAR/university guidance."""
    res = xiaozhi_gateway.execute_tool("search_current_agricultural_guidance", {"topic": "late blight management", "crop": "potato"})
    assert res["status"] == "ok"
    assert "guidance_summary" in res
    assert "source" in res
    assert "spoken_summary" in res
    assert "late blight" in res["spoken_summary"].lower()


def test_search_pesticide_guidance():
    """Verify pesticide guidance correctly distinguishes fungicides from insecticides with CIBRC sources."""
    # 1. Valid fungicide lookup for tomato early blight
    res = xiaozhi_gateway.execute_tool("search_pesticide_guidance", {"crop": "tomato", "disease_or_pest": "early blight"})
    assert res["status"] == "ok"
    assert res["category"] == "Fungicide"
    assert any("Mancozeb" in ai or "Chlorothalonil" in ai or "Difenoconazole" in ai for ai in res["verified_active_ingredients"])
    assert res["pre_harvest_interval_days"] == 7
    assert "CIBRC" in res["authoritative_source"]
    assert "follow_up_suggestions" in res

    # 2. Valid insecticide lookup for aphids
    res_aphid = xiaozhi_gateway.execute_tool("search_pesticide_guidance", {"crop": "mustard", "disease_or_pest": "aphids"})
    assert res_aphid["status"] == "ok"
    assert res_aphid["category"] == "Insecticide"
    assert any("Imidacloprid" in ai or "Thiamethoxam" in ai for ai in res_aphid["verified_active_ingredients"])

    # 3. Missing info prompts
    res_no_crop = xiaozhi_gateway.execute_tool("search_pesticide_guidance", {"crop": "", "disease_or_pest": "early blight"})
    assert res_no_crop["status"] == "NEED_INFO"
    assert "Which crop" in res_no_crop["spoken_summary"]


def test_search_current_product_prices():
    """Verify product price lookup provides realistic market ranges and dealer variance disclaimer."""
    res = xiaozhi_gateway.execute_tool("search_current_product_prices", {"product_name": "Mancozeb"})
    assert res["status"] == "ok"
    assert len(res["matched_products"]) > 0
    assert "₹" in res["spoken_summary"]
    assert "disclaimer" in res


def test_search_weather_spray_suitability(sample_live_observation):
    """Verify weather check evaluates spray suitability based on temperature, rain, and humidity."""
    # 1. Dry conditions -> favorable
    sample_live_observation.sensor_telemetry.rain_detected = 0
    sample_live_observation.sensor_telemetry.temperature_c = 26.0
    sample_live_observation.sensor_telemetry.humidity_pct = 65.0
    xiaozhi_gateway.update_observation(sample_live_observation)

    res_dry = xiaozhi_gateway.execute_tool("search_weather", {})
    assert res_dry["status"] == "ok"
    assert res_dry["spray_suitability"] == "FAVORABLE"
    assert "suitable" in res_dry["spoken_summary"].lower()

    # 2. Raining conditions -> unfavorable
    sample_live_observation.sensor_telemetry.rain_detected = 1
    xiaozhi_gateway.update_observation(sample_live_observation)
    res_rain = xiaozhi_gateway.execute_tool("search_weather", {})
    assert res_rain["status"] == "ok"
    assert res_rain["spray_suitability"] == "UNFAVORABLE"
    assert "rain" in res_rain["spoken_summary"].lower()


def test_estimate_treatment_budget():
    """Verify treatment budget scales properly across field acreages and outlines assumptions."""
    # 1. 1 acre
    res_1ac = xiaozhi_gateway.execute_tool("estimate_treatment_budget", {
        "crop": "tomato",
        "problem": "early blight",
        "field_area_acres": 1.0
    })
    assert res_1ac["status"] == "ok"
    cost_1ac_low = res_1ac["estimated_cost_low_inr"]
    cost_1ac_high = res_1ac["estimated_cost_high_inr"]
    assert cost_1ac_low > 0 and cost_1ac_high >= cost_1ac_low
    assert len(res_1ac["assumptions"]) >= 4

    # 2. 2 acres (cost should be double 1 acre)
    res_2ac = xiaozhi_gateway.execute_tool("estimate_treatment_budget", {
        "crop": "tomato",
        "problem": "early blight",
        "field_area_acres": 2.0
    })
    assert res_2ac["status"] == "ok"
    assert res_2ac["estimated_cost_low_inr"] == cost_1ac_low * 2
    assert res_2ac["estimated_cost_high_inr"] == cost_1ac_high * 2

    # 3. Missing acreage -> prompts farmer
    res_no_acre = xiaozhi_gateway.execute_tool("estimate_treatment_budget", {
        "crop": "tomato",
        "problem": "early blight",
        "field_area_acres": 0
    })
    assert res_no_acre["status"] == "NEED_INFO"
    assert "How many acres" in res_no_acre["spoken_summary"]


def test_physical_actuation_blocked():

    """Verify chassis and arm movements are strictly blocked over voice tools."""
    for forbidden_tool in ("move_rover", "stop_rover", "move_arm", "home_arm", "open_gripper"):
        res = xiaozhi_gateway.execute_tool(forbidden_tool, {"direction": "FORWARD"})
        assert res["status"] == "BLOCKED"
        assert "blocked" in res["message"].lower()


import asyncio


def test_mcp_jsonrpc_lifecycle(sample_live_observation):
    """Verify JSON-RPC 2.0 message handling for initialize, tools/list, tools/call, and ping."""
    async def _run():
        xiaozhi_gateway.update_observation(sample_live_observation)

        # 1. Initialize
        init_msg = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
        init_res = await xiaozhi_mcp_client._handle_incoming_message(init_msg)
        assert init_res is not None
        init_data = json.loads(init_res)
        assert init_data["id"] == 1
        assert init_data["result"]["protocolVersion"] == "2024-11-05"
        assert init_data["result"]["serverInfo"]["name"] == "SmartCropVision"

        # 2. notifications/initialized (no response required)
        notify_msg = json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}})
        notify_res = await xiaozhi_mcp_client._handle_incoming_message(notify_msg)
        assert notify_res is None

        # 3. tools/list
        list_msg = json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
        list_res = await xiaozhi_mcp_client._handle_incoming_message(list_msg)
        assert list_res is not None
        list_data = json.loads(list_res)
        assert list_data["id"] == 2
        assert len(list_data["result"]["tools"]) >= 16

        # 4. tools/call
        call_msg = json.dumps({
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": "get_current_field_status", "arguments": {}},
        })
        call_res = await xiaozhi_mcp_client._handle_incoming_message(call_msg)
        assert call_res is not None
        call_data = json.loads(call_res)
        assert call_data["id"] == 3
        assert call_data["result"]["isError"] is False
        assert len(call_data["result"]["content"]) > 0
        assert call_data["result"]["content"][0]["type"] == "text"

        # 5. ping
        ping_msg = json.dumps({"jsonrpc": "2.0", "id": 4, "method": "ping", "params": {}})
        ping_res = await xiaozhi_mcp_client._handle_incoming_message(ping_msg)
        assert ping_res is not None
        ping_data = json.loads(ping_res)
        assert ping_data["id"] == 4
        assert ping_data["result"] == {}

    asyncio.run(_run())


def test_token_masking():
    """Verify security masking prevents token exposure in logs."""
    sample_url = "wss://api.xiaozhi.me/mcp/?token=dummy_test_secret_token_123456789"
    masked = mask_token(sample_url)
    assert "dummy_test_secret" not in masked
    assert "token=***MASKED***" in masked

    raw_token = "dummy_secret_token_abcdef"
    masked_raw = mask_token(raw_token)
    assert "secret" not in masked_raw
    assert "..." in masked_raw


from fastapi.testclient import TestClient
from backend.app.main import app


def test_chat_endpoint_suggestions(sample_live_observation):
    """Verify POST /api/v1/xiaozhi/chat returns context-aware suggestion chips for frontend rendering."""
    xiaozhi_gateway.update_observation(sample_live_observation)
    client = TestClient(app)

    # 1. Soil query
    res_soil = client.post("/api/v1/xiaozhi/chat", json={"query": "What is the soil moisture?", "language": "en"})
    assert res_soil.status_code == 200
    data_soil = res_soil.json()
    assert "response" in data_soil
    assert "suggestions" in data_soil
    assert len(data_soil["suggestions"]) >= 2
    for s in data_soil["suggestions"]:
        assert "text" in s and "type" in s and s["text"].endswith("?")

    # 2. Pesticide query
    res_pest = client.post("/api/v1/xiaozhi/chat", json={"query": "What should I spray on tomato for early blight?", "language": "en"})
    assert res_pest.status_code == 200
    data_pest = res_pest.json()
    assert "Mancozeb" in data_pest["response"] or "Chlorothalonil" in data_pest["response"]
    assert len(data_pest["suggestions"]) >= 2

    # 3. Budget query
    res_budget = client.post("/api/v1/xiaozhi/chat", json={"query": "How much will it cost for 2 acres?", "language": "en"})
    assert res_budget.status_code == 200
    data_budget = res_budget.json()
    assert "₹" in data_budget["response"]
    assert len(data_budget["suggestions"]) >= 2


def test_hindi_agricultural_queries_and_dashboard_alias():
    """Verify gateway correctly normalizes Hindi terms (टमाटर, पछेती झुलसा) and supports dashboard summary aliases."""
    # 1. Hindi pesticide lookup: tomato (टमाटर) and late blight (पछेती झुलसा)
    res_hi = xiaozhi_gateway.execute_tool("search_pesticide_guidance", {"crop": "टमाटर", "disease_or_pest": "पछेती झुलसा"})
    assert res_hi["status"] == "ok"
    assert res_hi["category"] == "Fungicide"
    assert any("Mancozeb" in ai or "Metalaxyl" in ai for ai in res_hi["active_ingredients"])
    assert "spoken_summary_hi" in res_hi
    assert "मैन्कोजेब" in res_hi["spoken_summary_hi"] or "CIBRC" in res_hi["spoken_summary_hi"]

    # 2. Hindi budget lookup: 2 acres for late blight
    res_bud = xiaozhi_gateway.execute_tool("estimate_treatment_budget", {"crop": "टमाटर", "problem": "पछेती झुलसा", "field_area_acres": 2.0})
    assert res_bud["status"] == "ok"
    assert "spoken_summary_hi" in res_bud
    assert "एकड़" in res_bud["spoken_summary_hi"]
    assert res_bud["estimated_cost_low_inr"] > 0

    # 3. Dashboard summary alias
    res_dash = xiaozhi_gateway.execute_tool("get_dashboard_summary", {})
    assert res_dash["status"] == "ok"
    assert "SmartCropVision" in res_dash["spoken_summary"]
    assert "spoken_summary_hi" in res_dash
    assert "डैशबोर्ड" in res_dash["spoken_summary_hi"]


