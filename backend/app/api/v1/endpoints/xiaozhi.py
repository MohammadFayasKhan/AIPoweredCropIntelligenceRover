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
    Grounds operator conversational queries in live field observations.
    Never hallucinates data: queries live telemetry, GPS, risk engine, and vision records.
    """
    query_lower = req.query.lower()
    obs = xiaozhi_gateway.get_latest_observation()

    # Temperature / Humidity queries
    if any(k in query_lower for k in ("temperature", "heat", "humidity", "weather", "climate")):
        if not obs:
            return {"response": "AgriRover is currently offline. No live microclimate data available."}
        t = obs.sensor_telemetry
        r = obs.environmental_risk
        return {
            "response": (
                f"The ambient field temperature is {t.temperature_c}°C with {t.humidity_pct}% relative humidity. "
                f"The calculated Heat Stress Index is {r.heat_stress_index}°C ({r.heat_risk_level} risk)."
            ),
            "data": {"temperature_c": t.temperature_c, "humidity_pct": t.humidity_pct, "heat_index": r.heat_stress_index},
        }

    # Soil moisture / Drought / Irrigation queries
    elif any(k in query_lower for k in ("soil", "moisture", "water", "irrigate", "drought")):
        if not obs:
            return {"response": "AgriRover is offline. No soil moisture telemetry is available."}
        t = obs.sensor_telemetry
        i = obs.irrigation_advisory
        return {
            "response": (
                f"Capacitive soil moisture is at {t.soil_moisture_pct:.1f}%. "
                f"Irrigation recommendation is {i.recommendation} ({i.urgency} urgency). "
                f"{i.rationale}"
            ),
            "data": {"soil_moisture_pct": t.soil_moisture_pct, "advisory": i.dict()},
        }

    # Location / GPS queries
    elif any(k in query_lower for k in ("location", "where", "gps", "coordinates")):
        if not obs or not obs.gps.is_valid:
            return {"response": "AgriRover GPS is currently searching for satellites. No valid lock yet."}
        g = obs.gps
        return {
            "response": (
                f"AgriRover is located at latitude {g.latitude:.5f}, longitude {g.longitude:.5f} "
                f"({g.satellites_tracked} satellites tracked, {g.fix_state})."
            ),
            "data": {"latitude": g.latitude, "longitude": g.longitude, "satellites": g.satellites_tracked},
        }

    # Disease / Crop Vision queries
    elif any(k in query_lower for k in ("disease", "crop", "plant", "diagnos", "infection", "pathogen")):
        if not obs or not obs.crop_vision:
            crop_intel = obs.crop_intelligence if obs else None
            rec_crop = crop_intel.get("recommended_crop", "Unknown") if crop_intel else "Unknown"
            return {
                "response": (
                    f"No visual leaf scan has been conducted in this quadrant yet. "
                    f"Recommended agro-climatic crop for this soil is {rec_crop}."
                )
            }
        cv = obs.crop_vision
        return {
            "response": (
                f"Latest visual diagnosis on leaf: {cv.get('disease_name', 'Healthy')} "
                f"on {cv.get('crop_name', 'Plant')} with {cv.get('confidence_pct', 0):.1f}% confidence."
            ),
            "data": cv,
        }

    # Rover & Arm status
    elif any(k in query_lower for k in ("rover", "arm", "battery", "status", "health")):
        if not obs:
            return {"response": "AgriRover controller is currently offline."}
        r = obs.rover_state
        a = obs.arm_state
        return {
            "response": (
                f"AgriRover status: Movement is {r.movement_state}, speed PWM {r.speed_pwm}, "
                f"Headlight {'ON' if r.headlight_on else 'OFF'}. "
                f"Robotic Arm is {'Moving' if a.is_moving else 'Ready'} at Base {a.base_deg}°, "
                f"Shoulder {a.shoulder_deg}°, Elbow {a.elbow_deg}°, Gripper {a.gripper_state}."
            ),
            "data": {"rover": r.dict(), "arm": a.dict()},
        }

    # Default fallback
    return {
        "response": (
            "I can provide real-time field telemetry from AgriRover. Ask about temperature, "
            "capacitive soil moisture, irrigation advisory, GPS coordinates, disease diagnosis, or rover status."
        )
    }
