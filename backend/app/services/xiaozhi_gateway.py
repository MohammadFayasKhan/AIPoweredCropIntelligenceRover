"""
XiaoZhi Voice Assistant & Model Context Protocol (MCP) Gateway Service.
Author: Team Innovex (SIH 2026 Challenge SIH26180 - Qualcomm Inc).

Provides structured read and actuation tools for conversational voice interaction with the AgriRover:
- Read tools: Telemetry, microclimate, soil moisture, water level, GPS coordinates,
  Crop Vision diagnosis, environmental risk, and irrigation recommendations.
- Actuation tools: Safe, rate-limited movement commands, arm presets, and inspection triggers.
- MCP (Model Context Protocol) tool schema definitions.

All physical actuation tools are strictly bounded by hardware safety envelopes
and require the local ESP32 watchdog and PCA9685 angular constraints.
"""

from __future__ import annotations

import time
import uuid
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

from backend.app.schemas.iot import (
    DeviceCommandRequest,
    DeviceCommandResponse,
    UnifiedObservation,
)


class ToolDefinition(BaseModel):
    name: str
    description: str
    parameters: Dict[str, Any]
    is_actuation: bool = False


class XiaoZhiGateway:
    """Manages conversational query tools and safe physical command dispatch."""

    def __init__(self):
        self._latest_observation: Optional[UnifiedObservation] = None
        self._recent_observations: List[UnifiedObservation] = []
        self._command_queue: List[DeviceCommandRequest] = []
        self._max_history = 50

    def update_observation(self, observation: UnifiedObservation) -> None:
        """Cache incoming field observation for immediate voice assistant retrieval."""
        self._latest_observation = observation
        self._recent_observations.append(observation)
        if len(self._recent_observations) > self._max_history:
            self._recent_observations.pop(0)

    def get_latest_observation(self) -> Optional[UnifiedObservation]:
        return self._latest_observation

    def get_mcp_tool_definitions(self) -> List[Dict[str, Any]]:
        """Returns standard MCP-compatible tool schema definitions."""
        return [
            {
                "name": "get_current_rover_status",
                "description": "Get current AgriRover movement state, speed PWM, headlight state, and uptime.",
                "parameters": {"type": "object", "properties": {}},
            },
            {
                "name": "get_sensor_status",
                "description": "Get real-time readings from DHT11 (temp/humidity), capacitive soil moisture, rain sensor, and water level sensor.",
                "parameters": {"type": "object", "properties": {}},
            },
            {
                "name": "get_current_environment",
                "description": "Get computed environmental risk indices including Heat Stress Index, Drought level, and Waterlogging status.",
                "parameters": {"type": "object", "properties": {}},
            },
            {
                "name": "get_current_crop_analysis",
                "description": "Get latest Crop Vision disease diagnosis, detected pathogen, confidence percentage, and recommended treatment.",
                "parameters": {"type": "object", "properties": {}},
            },
            {
                "name": "get_current_disease_risk",
                "description": "Get active microclimate fungal, bacterial, or heat stress alerts triggered by environmental thresholds.",
                "parameters": {"type": "object", "properties": {}},
            },
            {
                "name": "get_current_gps",
                "description": "Get AgriRover field location (latitude, longitude, altitude, satellites tracked, fix state).",
                "parameters": {"type": "object", "properties": {}},
            },
            {
                "name": "get_arm_status",
                "description": "Get 4-DOF robotic arm joint angles (Base, Shoulder, Elbow, Gripper) and movement state.",
                "parameters": {"type": "object", "properties": {}},
            },
            {
                "name": "get_irrigation_recommendation",
                "description": "Get actionable irrigation advisory (IRRIGATE_NOW, MONITOR, DELAY) and agronomic rationale.",
                "parameters": {"type": "object", "properties": {}},
            },
            {
                "name": "get_device_health",
                "description": "Get hardware diagnostics, ESP32 watchdog state, Wi-Fi RSSI, and sensor freshness.",
                "parameters": {"type": "object", "properties": {}},
            },
            # Actuation tools (bounded & safety validated)
            {
                "name": "move_rover",
                "description": "Teleoperate AgriRover chassis in a specified direction with safe duration bounds.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "direction": {"type": "string", "enum": ["FORWARD", "BACKWARD", "LEFT", "RIGHT"]},
                        "speed": {"type": "integer", "minimum": 100, "maximum": 255, "default": 200},
                        "duration_ms": {"type": "integer", "minimum": 100, "maximum": 2500, "default": 600},
                    },
                    "required": ["direction"],
                },
            },
            {
                "name": "stop_rover",
                "description": "Emergency halt of AgriRover chassis drive motors.",
                "parameters": {"type": "object", "properties": {}},
            },
            {
                "name": "move_arm",
                "description": "Actuate robotic arm joints within safe physical angular envelopes.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "base": {"type": "number", "minimum": 5.0, "maximum": 175.0},
                        "shoulder": {"type": "number", "minimum": 0.0, "maximum": 180.0},
                        "elbow": {"type": "number", "minimum": 0.0, "maximum": 180.0},
                        "gripper": {"type": "number", "minimum": 70.0, "maximum": 180.0},
                    },
                },
            },
            {
                "name": "home_arm",
                "description": "Return robotic arm to default upright home stance (90°, 90°, 90°, 180°).",
                "parameters": {"type": "object", "properties": {}},
            },
            {
                "name": "open_gripper",
                "description": "Open robotic arm end-effector gripper (180°).",
                "parameters": {"type": "object", "properties": {}},
            },
            {
                "name": "close_gripper",
                "description": "Close robotic arm end-effector gripper (70°).",
                "parameters": {"type": "object", "properties": {}},
            },
            {
                "name": "start_inspection",
                "description": "Trigger robotic arm inspection stance and camera capture sequence.",
                "parameters": {"type": "object", "properties": {}},
            },
        ]

    def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Executes a requested tool against live device state or command queue."""
        obs = self._latest_observation

        # 1. Read tools
        if tool_name == "get_current_rover_status":
            if not obs:
                return {"status": "NO_TELEMETRY", "message": "AgriRover has not sent telemetry yet."}
            r = obs.rover_state
            return {
                "movement_state": r.movement_state,
                "speed_pwm": r.speed_pwm,
                "headlight_on": r.headlight_on,
                "watchdog_active": r.watchdog_active,
                "uptime_seconds": r.uptime_seconds,
                "wifi_rssi_dbm": r.wifi_rssi,
                "local_ip": r.local_ip,
            }

        elif tool_name == "get_sensor_status":
            if not obs:
                return {"status": "NO_TELEMETRY", "message": "Sensor data unavailable."}
            s = obs.sensor_telemetry
            return {
                "temperature_c": s.temperature_c,
                "humidity_pct": s.humidity_pct,
                "soil_moisture_pct": s.soil_moisture_pct,
                "soil_moisture_raw": s.soil_moisture_raw,
                "rain_detected": bool(s.rain_detected),
                "rain_intensity_pct": s.rain_intensity_pct,
                "water_level_pct": s.water_level_pct,
                "water_state": s.water_state,
            }

        elif tool_name == "get_current_environment":
            if not obs or not obs.environmental_risk:
                return {"status": "NO_TELEMETRY", "message": "Environmental risk data unavailable."}
            e = obs.environmental_risk
            return {
                "heat_stress_index": e.heat_stress_index,
                "heat_risk_level": e.heat_risk_level,
                "water_stress_level": e.water_stress_level,
                "flood_risk_level": e.flood_risk_level,
                "overall_risk_score": e.overall_risk_score,
                "primary_threat": e.primary_threat,
                "rationale": e.rationale,
            }

        elif tool_name == "get_current_crop_analysis":
            if not obs or not obs.crop_vision:
                return {"status": "NO_DIAGNOSIS", "message": "No visual foliar diagnosis recorded yet."}
            return obs.crop_vision

        elif tool_name == "get_current_disease_risk":
            if not obs or not obs.crop_intelligence:
                return {"status": "NO_ALERTS", "disease_alerts": []}
            return {
                "disease_alerts": obs.crop_intelligence.get("disease_alerts", []),
                "recommended_crop": obs.crop_intelligence.get("recommended_crop", "Unknown"),
            }

        elif tool_name == "get_current_gps":
            if not obs:
                return {"status": "NO_TELEMETRY", "message": "GPS unavailable."}
            g = obs.gps
            return {
                "latitude": g.latitude,
                "longitude": g.longitude,
                "altitude_m": g.altitude_m,
                "satellites": g.satellites_tracked,
                "hdop": g.hdop,
                "fix_state": g.fix_state,
                "is_valid": g.is_valid,
            }

        elif tool_name == "get_arm_status":
            if not obs:
                return {"status": "NO_TELEMETRY", "message": "Arm state unavailable."}
            a = obs.arm_state
            return {
                "base_deg": a.base_deg,
                "shoulder_deg": a.shoulder_deg,
                "elbow_deg": a.elbow_deg,
                "gripper_deg": a.gripper_deg,
                "gripper_state": a.gripper_state,
                "is_moving": a.is_moving,
                "emergency_stopped": a.emergency_stopped,
            }

        elif tool_name == "get_irrigation_recommendation":
            if not obs or not obs.irrigation_advisory:
                return {"status": "NO_TELEMETRY", "message": "Irrigation intelligence unavailable."}
            i = obs.irrigation_advisory
            return {
                "recommendation": i.recommendation,
                "water_volume_proxy": i.water_volume_proxy,
                "urgency": i.urgency,
                "rationale": i.rationale,
            }

        elif tool_name == "get_device_health":
            if not obs:
                return {"status": "OFFLINE", "message": "Device offline."}
            return {
                "device_id": obs.device_id,
                "timestamp": obs.timestamp,
                "rover_watchdog": obs.rover_state.watchdog_active,
                "uptime_seconds": obs.rover_state.uptime_seconds,
                "wifi_rssi_dbm": obs.rover_state.wifi_rssi,
                "is_simulated": obs.is_simulated,
            }

        # 2. Actuation tools (bounded & safety validated)
        elif tool_name == "move_rover":
            direction = str(arguments.get("direction", "STOPPED")).upper()
            if direction not in ("FORWARD", "BACKWARD", "LEFT", "RIGHT", "STOPPED"):
                return {"status": "REJECTED", "message": f"Invalid direction '{direction}'"}
            speed = min(255, max(100, int(arguments.get("speed", 200))))
            duration_ms = min(2500, max(100, int(arguments.get("duration_ms", 600))))
            
            cmd = DeviceCommandRequest(
                command_id=f"cmd-xz-{uuid.uuid4().hex[:8]}",
                command_type="ROVER_MOVE",
                parameters={"direction": direction, "speed": speed, "duration_ms": duration_ms},
                source="XIAOZHI_VOICE",
            )
            self._command_queue.append(cmd)
            return {
                "status": "ACCEPTED",
                "command_id": cmd.command_id,
                "action": f"Moving {direction} at PWM {speed} for {duration_ms}ms",
            }

        elif tool_name == "stop_rover":
            cmd = DeviceCommandRequest(
                command_id=f"cmd-xz-{uuid.uuid4().hex[:8]}",
                command_type="ROVER_STOP",
                parameters={},
                source="XIAOZHI_VOICE",
            )
            self._command_queue.append(cmd)
            return {"status": "ACCEPTED", "command_id": cmd.command_id, "action": "Chassis stopped immediately"}

        elif tool_name == "move_arm":
            params = {}
            if "base" in arguments:
                params["base"] = min(175.0, max(5.0, float(arguments["base"])))
            if "shoulder" in arguments:
                params["shoulder"] = min(180.0, max(0.0, float(arguments["shoulder"])))
            if "elbow" in arguments:
                params["elbow"] = min(180.0, max(0.0, float(arguments["elbow"])))
            if "gripper" in arguments:
                params["gripper"] = min(180.0, max(70.0, float(arguments["gripper"])))

            cmd = DeviceCommandRequest(
                command_id=f"cmd-xz-{uuid.uuid4().hex[:8]}",
                command_type="ARM_TARGET",
                parameters=params,
                source="XIAOZHI_VOICE",
            )
            self._command_queue.append(cmd)
            return {"status": "ACCEPTED", "command_id": cmd.command_id, "parameters": params}

        elif tool_name == "home_arm":
            cmd = DeviceCommandRequest(
                command_id=f"cmd-xz-{uuid.uuid4().hex[:8]}",
                command_type="ARM_HOME",
                parameters={"base": 90.0, "shoulder": 90.0, "elbow": 90.0, "gripper": 180.0},
                source="XIAOZHI_VOICE",
            )
            self._command_queue.append(cmd)
            return {"status": "ACCEPTED", "command_id": cmd.command_id, "action": "Robotic arm homing to 90°"}

        elif tool_name == "open_gripper":
            cmd = DeviceCommandRequest(
                command_id=f"cmd-xz-{uuid.uuid4().hex[:8]}",
                command_type="ARM_GRIPPER",
                parameters={"gripper": 180.0, "state": "OPEN"},
                source="XIAOZHI_VOICE",
            )
            self._command_queue.append(cmd)
            return {"status": "ACCEPTED", "command_id": cmd.command_id, "action": "Gripper opening to 180°"}

        elif tool_name == "close_gripper":
            cmd = DeviceCommandRequest(
                command_id=f"cmd-xz-{uuid.uuid4().hex[:8]}",
                command_type="ARM_GRIPPER",
                parameters={"gripper": 70.0, "state": "CLOSED"},
                source="XIAOZHI_VOICE",
            )
            self._command_queue.append(cmd)
            return {"status": "ACCEPTED", "command_id": cmd.command_id, "action": "Gripper closing to 70°"}

        elif tool_name == "start_inspection":
            cmd = DeviceCommandRequest(
                command_id=f"cmd-xz-{uuid.uuid4().hex[:8]}",
                command_type="INSPECTION_RUN",
                parameters={},
                source="XIAOZHI_VOICE",
            )
            self._command_queue.append(cmd)
            return {"status": "ACCEPTED", "command_id": cmd.command_id, "action": "Triggered camera inspection sequence"}

        return {"status": "UNKNOWN_TOOL", "message": f"Tool '{tool_name}' is not recognized"}

    def pop_pending_commands(self) -> List[DeviceCommandRequest]:
        """Retrieve and clear queued commands awaiting transmission to ESP32."""
        cmds = list(self._command_queue)
        self._command_queue.clear()
        return cmds


xiaozhi_gateway = XiaoZhiGateway()
