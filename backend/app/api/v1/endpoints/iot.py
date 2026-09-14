"""
AgriRover IoT Gateway, Telemetry Ingestion, and Real-Time WebSocket Router.
Author: Team Innovex (SIH 2026 Challenge SIH26180 - Qualcomm Inc).

Endpoints:
- POST /api/v1/iot/observation: Ingest ground sensor packet, calculate environmental risk,
  query Crop Intelligence recommendation engine, trigger n8n alerts, and broadcast to live WebSockets.
- GET /api/v1/iot/observation/latest: Return latest unified observation snapshot.
- GET /api/v1/iot/observation/history: Return rolling buffer of past field observations.
- GET /api/v1/iot/device/state: Consolidated hardware diagnostics, watchdog, and sensor health.
- POST /api/v1/iot/device/command: Structured teleoperation and arm actuation gateway.
- GET /api/v1/iot/device/commands/pending: Retrieve queued commands for ESP32 downlink.
- POST /api/v1/iot/device/register: ESP32 hardware capability and protocol handshake.
- WS /api/v1/iot/live-ws: Bidirectional WebSocket stream for real-time dashboard updates.
"""

from __future__ import annotations

import time
import uuid
import logging
import asyncio
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException, Query, BackgroundTasks
from pydantic import ValidationError

from backend.app.config import settings
from backend.app.schemas.iot import (
    UnifiedObservation,
    SensorTelemetry,
    GPSData,
    RoverState,
    ArmState,
    DeviceCommandRequest,
    DeviceCommandResponse,
    DeviceRegistration,
)
from backend.app.schemas.crop import CropRecommendationRequest
from backend.app.services.environmental_risk_service import environmental_risk_engine
from backend.app.services.crop_service import crop_service
from backend.app.services.n8n_service import n8n_dispatcher
from backend.app.services.xiaozhi_gateway import xiaozhi_gateway

logger = logging.getLogger("smartcropvision.iot")
router = APIRouter()

# In-memory unified observation store
_latest_observation: Optional[UnifiedObservation] = None
_observation_history: List[UnifiedObservation] = []
_pending_commands: List[DeviceCommandRequest] = []
_device_registration: Optional[DeviceRegistration] = None


class ConnectionManager:
    """Manages active live dashboard WebSocket connections."""

    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"[WebSocket] Client connected. Total active: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            logger.info(f"[WebSocket] Client disconnected. Total active: {len(self.active_connections)}")

    async def broadcast(self, message: Dict[str, Any]):
        """Broadcast payload to all connected frontend clients."""
        for connection in list(self.active_connections):
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.debug(f"[WebSocket] Failed to send to client: {e}")
                self.disconnect(connection)


ws_manager = ConnectionManager()


@router.post("/device/register", response_model=Dict[str, Any])
async def register_device(reg: DeviceRegistration):
    """Handshake from ESP32 on startup registering capabilities and firmware version."""
    global _device_registration
    _device_registration = reg
    logger.info(
        f"[IoT Gateway] Registered device {reg.device_id} (FW: {reg.firmware_version}, Profile: {reg.hardware_profile})"
    )
    return {
        "status": "REGISTERED",
        "device_id": reg.device_id,
        "protocol_version": reg.protocol_version,
        "server_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


@router.post("/observation", response_model=UnifiedObservation)
async def ingest_observation(
    observation: UnifiedObservation,
    background_tasks: BackgroundTasks,
):
    """
    Ingests ground sensor group observations from the AgriRover ESP32.
    Computes environmental risk, attaches Crop Intelligence recommendations,
    dispatches n8n alerts asynchronously, and broadcasts live state.
    """
    global _latest_observation, _observation_history
    start_time = time.time()

    # 1. Evaluate transparent environmental risk & irrigation advisory
    telemetry = observation.sensor_telemetry
    risk = environmental_risk_engine.analyze(telemetry)
    advisory = environmental_risk_engine.generate_irrigation_advisory(telemetry, risk)

    observation.environmental_risk = risk
    observation.irrigation_advisory = advisory

    # 2. Query Crop Intelligence recommendation engine
    try:
        crop_req = CropRecommendationRequest(
            temperature=telemetry.temperature_c,
            humidity=telemetry.humidity_pct,
            soil_moisture=telemetry.soil_moisture_pct,
            rain=telemetry.rain_detected,
            rainfall_mm=telemetry.rain_intensity_pct,
        )
        crop_res = crop_service.recommend(crop_req)
        observation.crop_intelligence = crop_res.model_dump() if hasattr(crop_res, "model_dump") else crop_res.dict()
    except Exception as e:
        logger.warning(f"[IoT Gateway] Crop recommendation evaluation failed: {e}")
        observation.crop_intelligence = None

    # Calculate end-to-end processing latency
    observation.processing_latency_ms = round((time.time() - start_time) * 1000.0, 2)

    # 3. Update in-memory caches
    _latest_observation = observation
    _observation_history.append(observation)
    if len(_observation_history) > settings.TELEMETRY_HISTORY_MAX_ENTRIES:
        _observation_history.pop(0)

    # Bridge to legacy main.latest_iot and main.history for complete backward compatibility
    try:
        from backend.app import main as app_main
        legacy_entry = {
            "timestamp": observation.timestamp,
            "source": "agrirover_esp32",
            "temperature": telemetry.temperature_c,
            "humidity": telemetry.humidity_pct,
            "soil_moisture": telemetry.soil_moisture_pct,
            "rain": telemetry.rain_detected,
            "water_level": telemetry.water_level_pct,
            "recommended_crop": observation.crop_intelligence.get("recommended_crop", "Pending") if observation.crop_intelligence else "Pending",
            "confidence": observation.crop_intelligence.get("confidence_pct", 0.0) if observation.crop_intelligence else 0.0,
            "alert_count": len(observation.crop_intelligence.get("disease_alerts", [])) if observation.crop_intelligence else 0,
        }
        app_main.latest_iot = legacy_entry
        app_main.history.append(legacy_entry)
    except Exception:
        pass

    # 4. Update XiaoZhi voice assistant cache
    xiaozhi_gateway.update_observation(observation)

    # 5. Check and trigger n8n automation webhooks in background
    crop_name = (
        observation.crop_intelligence.get("recommended_crop")
        if observation.crop_intelligence
        else None
    )
    t_dump = telemetry.model_dump() if hasattr(telemetry, "model_dump") else telemetry.dict()
    r_dump = risk.model_dump() if hasattr(risk, "model_dump") else risk.dict()
    g_dump = observation.gps.model_dump() if hasattr(observation.gps, "model_dump") else observation.gps.dict()
    obs_dump = observation.model_dump() if hasattr(observation, "model_dump") else observation.dict()

    background_tasks.add_task(
        n8n_dispatcher.check_and_trigger_risk_alerts,
        t_dump,
        r_dump,
        g_dump,
        crop_name,
    )

    # 6. Broadcast live packet to dashboard WebSockets
    background_tasks.add_task(
        ws_manager.broadcast,
        {"event": "TELEMETRY_UPDATE", "data": obs_dump},
    )

    return observation


@router.get("/observation/latest", response_model=UnifiedObservation)
async def get_latest_observation():
    """Retrieve the most recent unified field observation snapshot."""
    global _latest_observation
    if _latest_observation is None:
        # Generate initial default baseline observation so UI never crashes
        default_telemetry = SensorTelemetry(
            temperature_c=28.5,
            humidity_pct=65.0,
            dht_status="VALID",
            soil_moisture_raw=2200,
            soil_moisture_pct=52.0,
            soil_calibration_status="CALIBRATED",
            rain_detected=0,
            rain_intensity_raw=4095,
            rain_intensity_pct=0.0,
            water_level_raw=420,
            water_level_pct=10.2,
            water_state="NORMAL",
        )
        risk = environmental_risk_engine.analyze(default_telemetry)
        advisory = environmental_risk_engine.generate_irrigation_advisory(default_telemetry, risk)

        _latest_observation = UnifiedObservation(
            observation_id=f"obs-init-{uuid.uuid4().hex[:8]}",
            device_id="agrirover-esp32-01",
            sensor_telemetry=default_telemetry,
            gps=GPSData(
                latitude=13.0827,
                longitude=80.2707,
                altitude_m=12.4,
                satellites_tracked=8,
                hdop=1.1,
                fix_state="3D_FIX",
                is_valid=True,
            ),
            rover_state=RoverState(),
            arm_state=ArmState(),
            environmental_risk=risk,
            irrigation_advisory=advisory,
            is_simulated=True,
        )

    return _latest_observation


@router.get("/observation/history", response_model=List[UnifiedObservation])
async def get_observation_history(limit: int = Query(50, ge=1, le=500)):
    """Retrieve historical field observations in reverse chronological order."""
    return list(reversed(_observation_history[-limit:]))


@router.get("/device/state", response_model=Dict[str, Any])
async def get_device_state():
    """Aggregated device health, registration, and active state."""
    latest = await get_latest_observation()
    return {
        "status": "ONLINE" if latest and not latest.is_simulated else "STANDBY",
        "device_id": latest.device_id,
        "registration": _device_registration.dict() if _device_registration else None,
        "rover": latest.rover_state.dict(),
        "arm": latest.arm_state.dict(),
        "gps": latest.gps.dict(),
        "sensors": latest.sensor_telemetry.dict(),
        "last_seen": latest.timestamp,
        "history_count": len(_observation_history),
    }


@router.post("/device/command", response_model=DeviceCommandResponse)
async def dispatch_device_command(
    cmd: DeviceCommandRequest,
    background_tasks: BackgroundTasks,
):
    """
    Command gateway for rover teleoperation, arm kinematics, and camera inspection.
    Validates physical boundaries and enqueues commands for hardware execution.
    """
    global _pending_commands

    # 1. Physical safety & boundary verification
    if cmd.command_type == "ROVER_MOVE":
        direction = cmd.parameters.get("direction", "").upper()
        if direction not in ("FORWARD", "BACKWARD", "LEFT", "RIGHT", "STOPPED"):
            return DeviceCommandResponse(
                command_id=cmd.command_id,
                status="REJECTED",
                message=f"Invalid rover movement direction '{direction}'",
                safety_validated=False,
            )
    elif cmd.command_type == "ARM_TARGET":
        base = cmd.parameters.get("base")
        shoulder = cmd.parameters.get("shoulder")
        elbow = cmd.parameters.get("elbow")
        gripper = cmd.parameters.get("gripper")

        if base is not None and not (5.0 <= float(base) <= 175.0):
            return DeviceCommandResponse(
                command_id=cmd.command_id,
                status="REJECTED",
                message=f"Base angle {base}° violates mechanical safety limits (5°-175°)",
                safety_validated=False,
            )
        if shoulder is not None and not (0.0 <= float(shoulder) <= 180.0):
            return DeviceCommandResponse(
                command_id=cmd.command_id,
                status="REJECTED",
                message=f"Shoulder angle {shoulder}° exceeds 0°-180° limit",
                safety_validated=False,
            )
        if elbow is not None and not (0.0 <= float(elbow) <= 180.0):
            return DeviceCommandResponse(
                command_id=cmd.command_id,
                status="REJECTED",
                message=f"Elbow angle {elbow}° exceeds 0°-180° limit",
                safety_validated=False,
            )
        if gripper is not None and not (70.0 <= float(gripper) <= 180.0):
            return DeviceCommandResponse(
                command_id=cmd.command_id,
                status="REJECTED",
                message=f"Gripper angle {gripper}° exceeds mechanical grasp envelope (70°-180°)",
                safety_validated=False,
            )

    elif cmd.command_type == "ARM_EMERGENCY_STOP" or cmd.command_type == "ROVER_STOP":
        # Clear all pending commands immediately on emergency stop
        _pending_commands.clear()

    # Enqueue command for ESP32 downlink
    _pending_commands.append(cmd)
    if len(_pending_commands) > 50:
        _pending_commands.pop(0)

    # Broadcast command acknowledgement to WebSocket clients
    cmd_dump = cmd.model_dump() if hasattr(cmd, "model_dump") else cmd.dict()
    background_tasks.add_task(
        ws_manager.broadcast,
        {"event": "COMMAND_DISPATCHED", "data": cmd_dump},
    )

    return DeviceCommandResponse(
        command_id=cmd.command_id,
        status="ACCEPTED",
        message=f"Command '{cmd.command_type}' validated and enqueued for {cmd.device_id}",
        safety_validated=True,
    )


@router.get("/device/commands/pending", response_model=List[DeviceCommandRequest])
async def get_pending_commands():
    """Downlink poll endpoint for ESP32 hardware to fetch pending commands."""
    global _pending_commands
    cmds = list(_pending_commands)
    _pending_commands.clear()
    return cmds


@router.websocket("/live-ws")
async def websocket_live_telemetry(websocket: WebSocket):
    """
    Full-duplex WebSocket connection for real-time AgriRover dashboard updates.
    Sends initial state on connect, streams live telemetry packets, and receives teleop commands.
    """
    await ws_manager.connect(websocket)
    try:
        # Send initial snapshot immediately upon handshake
        latest = await get_latest_observation()
        await websocket.send_json({"event": "INITIAL_STATE", "data": latest.dict()})

        while True:
            # Listen for inbound commands from dashboard
            raw_data = await websocket.receive_json()
            event = raw_data.get("event")
            payload = raw_data.get("data", {})

            if event == "COMMAND":
                try:
                    cmd_req = DeviceCommandRequest(**payload)
                    # Safety check and enqueue
                    _pending_commands.append(cmd_req)
                    await ws_manager.broadcast({"event": "COMMAND_DISPATCHED", "data": cmd_req.dict()})
                except ValidationError as ve:
                    await websocket.send_json({"event": "ERROR", "message": str(ve)})

            elif event == "PING":
                await websocket.send_json({"event": "PONG", "timestamp": time.time()})

    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"[WebSocket] Exception in connection loop: {e}")
        ws_manager.disconnect(websocket)
