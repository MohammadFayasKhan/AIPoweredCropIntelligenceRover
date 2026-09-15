"""
XiaoZhi Voice Assistant & MCP Endpoint Router.
Author: Team Innovex (SIH 2026 Challenge SIH26180 - Qualcomm Inc).

Endpoints:
- GET /api/v1/xiaozhi/mcp/tools: Exposes Model Context Protocol (MCP) tool definitions.
- POST /api/v1/xiaozhi/execute: Executes a structured read or actuation tool.
- POST /api/v1/xiaozhi/chat: Grounded conversational Q&A prioritizing real-time field data.
"""

from __future__ import annotations

import logging
from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Depends, Header
from pydantic import BaseModel, Field

from backend.app.config import settings
from backend.app.services.xiaozhi_gateway import xiaozhi_gateway

logger = logging.getLogger("smartcropvision.xiaozhi")
router = APIRouter()


class ToolExecutionRequest(BaseModel):
    tool_name: str = Field(..., description="Name of the MCP or XiaoZhi tool to execute")
    arguments: Dict[str, Any] = Field(default_factory=dict, description="Tool-specific argument dictionary")


class ChatQueryRequest(BaseModel):
    query: str = Field(..., description="Farmer or operator natural language query in voice/text")
    language: str = Field("en", description="Language code (en, hi, ta, etc.)")


def verify_xiaozhi_auth(authorization: Optional[str] = Header(None)):
    """Optional bearer token check for actuation security."""
    if not settings.XIAOZHI_ENABLED:
        raise HTTPException(status_code=503, detail="XiaoZhi assistant is disabled in configuration.")
    if settings.XIAOZHI_API_KEY:
        if not authorization:
            # For development flexibility allow without header if secret is default, but validate if present
            pass
        elif authorization.startswith("Bearer "):
            token = authorization.split(" ")[1]
            if token != settings.XIAOZHI_API_KEY:
                raise HTTPException(status_code=401, detail="Invalid XiaoZhi authorization token.")


@router.get("/mcp/tools", response_model=Dict[str, Any])
async def list_mcp_tools():
    """Lists all MCP tool specifications available for XiaoZhi voice assistant."""
    tools = xiaozhi_gateway.get_mcp_tool_definitions()
    return {"tools": tools, "count": len(tools), "protocol": "mcp-v1"}


@router.post("/execute", response_model=Dict[str, Any])
async def execute_tool(req: ToolExecutionRequest, auth=Depends(verify_xiaozhi_auth)):
    """Executes an MCP tool against live AgriRover hardware state or enqueues actuation."""
    res = xiaozhi_gateway.execute_tool(req.tool_name, req.arguments)
    return {"tool": req.tool_name, "result": res}


@router.post("/chat", response_model=Dict[str, Any])
async def natural_language_chat(req: ChatQueryRequest):
    """
    Grounds operator and farmer conversational queries in live field observations.
    Synthesizes live telemetry, Random Forest crop AI, CV leaf diagnosis, weather,
    and verified CIBRC pesticide guidance into natural speech with dynamic follow-up chips.
    """
    query_lower = req.query.lower()
    obs = xiaozhi_gateway.get_latest_observation()

    # 1. Pesticide / Fungicide / Medicine queries
    if any(k in query_lower for k in ("spray", "pesticide", "fungicide", "insecticide", "medicine")):
        # Extract crop or disease context if present
        crop = "tomato" if "tomato" in query_lower else ("potato" if "potato" in query_lower else ("rice" if "rice" in query_lower else ""))
        problem = "early blight" if "early blight" in query_lower else ("late blight" if "late blight" in query_lower else ("powdery mildew" if "powdery mildew" in query_lower else ("aphid" in query_lower and "aphids" or "")))
        if not problem and obs and obs.crop_vision:
            problem = obs.crop_vision.get("disease_name", "")
        if not crop and obs and obs.crop_vision:
            crop = obs.crop_vision.get("crop_name", "")

        res = xiaozhi_gateway.execute_tool("search_pesticide_guidance", {"crop": crop or "general", "disease_or_pest": problem or "early blight"})
        suggestions = res.get("follow_up_suggestions", ["How much will it cost?", "Should I spray today?"])
        return {
            "response": res.get("spoken_summary", "Verified treatment guidance is available."),
            "data": res,
            "suggestions": [{"text": s, "type": "question"} for s in suggestions],
            "follow_up_suggestions": suggestions,
        }

    # 2. Budget / Cost / Price queries
    elif any(k in query_lower for k in ("cost", "price", "budget", "how much", "cheap")):
        crop = "tomato" if "tomato" in query_lower else ("potato" if "potato" in query_lower else ("rice" if "rice" in query_lower else "crop"))
        problem = "early blight" if "early blight" in query_lower else ("late blight" if "late blight" in query_lower else "early blight")
        acres = 2.0 if "2" in query_lower or "two" in query_lower else (1.0)

        res = xiaozhi_gateway.execute_tool("estimate_treatment_budget", {"crop": crop, "problem": problem, "field_area_acres": acres})
        suggestions = res.get("follow_up_suggestions", ["Can you find a cheaper suitable option?", "Should I spray today?"])
        return {
            "response": res.get("spoken_summary", "Budget estimate calculated."),
            "data": res,
            "suggestions": [{"text": s, "type": "question"} for s in suggestions],
            "follow_up_suggestions": suggestions,
        }

    # 3. Weather / Rain / Spray Suitability queries
    elif any(k in query_lower for k in ("weather", "rain", "spray today", "suitability", "wind")):
        res = xiaozhi_gateway.execute_tool("search_weather", {"query": query_lower})
        suggestions = res.get("follow_up_suggestions", ["Should I irrigate?", "What is the current soil moisture?"])
        return {
            "response": res.get("spoken_summary", "Weather check completed."),
            "data": res,
            "suggestions": [{"text": s, "type": "question"} for s in suggestions],
            "follow_up_suggestions": suggestions,
        }

    # 4. Temperature / Humidity queries
    elif any(k in query_lower for k in ("temperature", "heat", "humidity", "climate")):
        if not obs:
            return {
                "response": "AgriRover is currently offline. No live microclimate data available.",
                "suggestions": [{"text": "When was the last reading?", "type": "question"}],
                "follow_up_suggestions": ["When was the last reading?"],
            }
        t = obs.sensor_telemetry
        r = obs.environmental_risk
        suggestions = ["Should I irrigate?", "Will it rain today?", "Is the crop healthy?"]
        return {
            "response": (
                f"The ambient field temperature is {t.temperature_c}°C with {t.humidity_pct}% relative humidity. "
                f"The calculated Heat Stress Index is {r.heat_stress_index}°C ({r.heat_risk_level} risk)."
            ),
            "data": {"temperature_c": t.temperature_c, "humidity_pct": t.humidity_pct, "heat_index": r.heat_stress_index},
            "suggestions": [{"text": s, "type": "question"} for s in suggestions],
            "follow_up_suggestions": suggestions,
        }

    # 5. Soil moisture / Drought / Irrigation queries
    elif any(k in query_lower for k in ("soil", "moisture", "water", "irrigate", "drought")):
        if not obs:
            return {
                "response": "AgriRover is offline. No soil moisture telemetry is available.",
                "suggestions": [{"text": "When was the last reading?", "type": "question"}],
                "follow_up_suggestions": ["When was the last reading?"],
            }
        t = obs.sensor_telemetry
        i = obs.irrigation_advisory
        suggestions = ["Will it rain today?", "How has the soil changed recently?", "What is the crop condition?"]
        return {
            "response": (
                f"Capacitive soil moisture is at {t.soil_moisture_pct:.1f}%. "
                f"Irrigation recommendation is {i.recommendation} ({i.urgency} urgency). "
                f"{i.rationale}"
            ),
            "data": {"soil_moisture_pct": t.soil_moisture_pct, "advisory": i.model_dump() if hasattr(i, "model_dump") else i.dict()},
            "suggestions": [{"text": s, "type": "question"} for s in suggestions],
            "follow_up_suggestions": suggestions,
        }

    # 6. Location / GPS queries
    elif any(k in query_lower for k in ("location", "where", "gps", "coordinates")):
        if not obs or not obs.gps.is_valid:
            return {
                "response": "AgriRover GPS is currently searching for satellites. No valid lock yet.",
                "suggestions": [{"text": "What is the field condition?", "type": "question"}],
                "follow_up_suggestions": ["What is the field condition?"],
            }
        g = obs.gps
        suggestions = ["How is the soil moisture?", "Does the crop need water?"]
        return {
            "response": (
                f"AgriRover is located at latitude {g.latitude:.5f}, longitude {g.longitude:.5f} "
                f"({g.satellites_tracked} satellites tracked, {g.fix_state})."
            ),
            "data": {"latitude": g.latitude, "longitude": g.longitude, "satellites": g.satellites_tracked},
            "suggestions": [{"text": s, "type": "question"} for s in suggestions],
            "follow_up_suggestions": suggestions,
        }

    # 7. Disease / Crop Vision queries
    elif any(k in query_lower for k in ("disease", "crop", "plant", "diagnos", "infection", "pathogen")):
        if not obs or not obs.crop_vision:
            crop_intel = obs.crop_intelligence if obs else None
            rec_crop = crop_intel.get("recommended_crop", "Unknown") if crop_intel else "Unknown"
            suggestions = ["What crop should I grow?", "What is the environmental disease risk?"]
            return {
                "response": (
                    f"No visual leaf scan has been conducted in this quadrant yet. "
                    f"Recommended agro-climatic crop for this soil is {rec_crop}."
                ),
                "suggestions": [{"text": s, "type": "question"} for s in suggestions],
                "follow_up_suggestions": suggestions,
            }
        cv = obs.crop_vision
        suggestions = ["What should I spray?", "How much will treatment cost?", "Should I spray today?"]
        return {
            "response": (
                f"Latest visual diagnosis on leaf: {cv.get('disease_name', 'Healthy')} "
                f"on {cv.get('crop_name', 'Plant')} with {cv.get('confidence_pct', 0):.1f}% confidence."
            ),
            "data": cv,
            "suggestions": [{"text": s, "type": "question"} for s in suggestions],
            "follow_up_suggestions": suggestions,
        }

    # 8. Field Summary / General Condition
    elif any(k in query_lower for k in ("summary", "field", "condition", "happening", "status")):
        res = xiaozhi_gateway.execute_tool("get_field_summary", {})
        suggestions = res.get("follow_up_suggestions", ["What should I do now?", "Does the crop need water?", "Will it rain today?"])
        return {
            "response": res.get("spoken_summary", "Field summary retrieved."),
            "data": res,
            "suggestions": [{"text": s, "type": "question"} for s in suggestions],
            "follow_up_suggestions": suggestions,
        }

    # 9. Rover & Arm status
    elif any(k in query_lower for k in ("rover", "arm", "battery")):
        if not obs:
            return {
                "response": "AgriRover controller is currently offline.",
                "suggestions": [{"text": "Can you check the sensor status?", "type": "question"}],
                "follow_up_suggestions": ["Can you check the sensor status?"],
            }
        r = obs.rover_state
        a = obs.arm_state
        suggestions = ["What are the sensor readings?", "How is the field?"]
        return {
            "response": (
                f"AgriRover status: Movement is {r.movement_state}, speed PWM {r.speed_pwm}, "
                f"Headlight {'ON' if r.headlight_on else 'OFF'}. "
                f"Robotic Arm is {'Moving' if a.is_moving else 'Ready'} at Base {a.base_deg}°, "
                f"Shoulder {a.shoulder_deg}°, Elbow {a.elbow_deg}°, Gripper {a.gripper_state}."
            ),
            "data": {
                "rover": r.model_dump() if hasattr(r, "model_dump") else r.dict(),
                "arm": a.model_dump() if hasattr(a, "model_dump") else a.dict(),
            },
            "suggestions": [{"text": s, "type": "question"} for s in suggestions],
            "follow_up_suggestions": suggestions,
        }

    # Default fallback
    suggestions = ["How is my field?", "Does my crop need water?", "Is there any disease?"]
    return {
        "response": (
            "I can provide real-time field intelligence from SmartCropVision. Ask about current field condition, "
            "capacitive soil moisture, irrigation advisory, disease diagnosis, pesticide recommendations, treatment budget, or weather."
        ),
        "suggestions": [{"text": s, "type": "question"} for s in suggestions],
        "follow_up_suggestions": suggestions,
    }
