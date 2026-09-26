"""
AI Powered Crop Intelligence Rover  -  n8n Automation & Alert Webhook Dispatcher.

This service integrates the physical rover and intelligence backend with n8n workflow pipelines:
1. Event Emission: Generates typed JSON payloads for critical agronomic events
   (pathogen outbreaks, root zone drought, flooding/ponding, heat stress, motor stalls).
2. Anti-Flooding Cooldown Deduplication: Maintains an in-memory TTL cache
   (default 300 seconds / 5 minutes) keyed by event type and entity identifier. This prevents
   alert storms across SMS, WhatsApp, and Telegram channels when sensor readings hover near thresholds.
3. Fail-Safe Asynchronous Dispatch: Dispatches webhooks using non-blocking HTTP client calls;
   network drops or slow external webhook receivers never block rover real-time control loops.
"""

from __future__ import annotations

import logging
import time
import asyncio
from typing import Dict, Any, Optional
import httpx

from backend.app.config import settings

logger = logging.getLogger("crop_rover.n8n")


class N8NAlertDispatcher:
    """Dispatches webhook alerts to n8n automation workflows with deduplication."""

    def __init__(self):
        # Maps event_key -> last_dispatched_timestamp
        # Prevents alert spam by ensuring an identical condition cannot fire repeatedly
        self._cooldown_cache: Dict[str, float] = {}
        self._cooldown_seconds: int = settings.N8N_COOLDOWN_SECONDS
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Lazily initialize an asynchronous HTTP client session with strict connection timeouts."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=4.0)
        return self._client

    def is_in_cooldown(self, event_type: str, deduplication_key: str) -> bool:
        """Check if an alert is currently suppressed by the cooldown window."""
        key = f"{event_type}:{deduplication_key}"
        now = time.time()
        last_sent = self._cooldown_cache.get(key, 0.0)
        if now - last_sent < self._cooldown_seconds:
            return True
        return False

    def mark_dispatched(self, event_type: str, deduplication_key: str) -> None:
        """Record dispatch timestamp for cooldown enforcement."""
        key = f"{event_type}:{deduplication_key}"
        self._cooldown_cache[key] = time.time()

    async def dispatch_event(
        self,
        event_type: str,
        deduplication_key: str,
        payload: Dict[str, Any],
    ) -> bool:
        """
        Asynchronously post an event payload to n8n webhook.
        Fails safely without raising exceptions into rover or sensor loops.
        """
        if not settings.N8N_ALERTS_ENABLED:
            logger.debug(f"[n8n] Alerts disabled via config; skipping event {event_type}")
            return False

        webhook_url = settings.N8N_WEBHOOK_URL
        if not webhook_url:
            logger.debug(f"[n8n] No N8N_WEBHOOK_URL configured; skipping event {event_type}")
            return False

        if self.is_in_cooldown(event_type, deduplication_key):
            logger.info(f"[n8n] Suppressing duplicate event '{event_type}:{deduplication_key}' (in cooldown)")
            return False

        # Construct comprehensive event packet
        packet = {
            "source": "AI-Powered-Crop-Intelligence-Rover-Gateway",
            "event_type": event_type,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "deduplication_key": deduplication_key,
            "data": payload,
        }

        try:
            client = await self._get_client()
            resp = await client.post(webhook_url, json=packet)
            if resp.status_code in (200, 201, 202):
                self.mark_dispatched(event_type, deduplication_key)
                logger.info(f"[n8n] Dispatched {event_type} to n8n successfully ({resp.status_code})")
                return True
            else:
                logger.warning(f"[n8n] Webhook responded with status {resp.status_code}: {resp.text[:120]}")
                return False
        except Exception as e:
            logger.error(f"[n8n] Non-fatal webhook dispatch error: {e}")
            return False

    async def check_and_trigger_risk_alerts(
        self,
        telemetry: Dict[str, Any],
        risk: Dict[str, Any],
        gps: Dict[str, Any],
        crop_name: Optional[str] = None
    ) -> None:
        """
        Evaluates risk state and fires appropriate n8n webhooks if thresholds are breached.
        """
        lat = gps.get("latitude")
        lon = gps.get("longitude")
        location_str = f"{lat:.5f}, {lon:.5f}" if lat is not None and lon is not None else "GPS Searching"

        # 1. Heat stress alert
        if risk.get("heat_risk_level") in ("HIGH", "SEVERE"):
            heat_payload = {
                "heat_index_c": risk.get("heat_stress_index"),
                "ambient_temp_c": telemetry.get("temperature_c"),
                "humidity_pct": telemetry.get("humidity_pct"),
                "heat_level": risk.get("heat_risk_level"),
                "location": location_str,
                "crop": crop_name or "General Canopy",
                "recommended_action": "Apply shaded misting or protective antitranspirant spray.",
            }
            await self.dispatch_event("heat_risk_detected", f"heat_{risk.get('heat_risk_level')}", heat_payload)

        # 2. Severe drought / water stress alert
        if risk.get("water_stress_level") in ("MODERATE_DROUGHT", "SEVERE_DROUGHT"):
            drought_payload = {
                "soil_moisture_pct": telemetry.get("soil_moisture_pct"),
                "ambient_temp_c": telemetry.get("temperature_c"),
                "stress_level": risk.get("water_stress_level"),
                "location": location_str,
                "crop": crop_name or "General Canopy",
                "recommended_action": "Trigger drip irrigation line immediately to prevent permanent root wilting.",
            }
            await self.dispatch_event("water_stress_detected", f"water_{risk.get('water_stress_level')}", drought_payload)

        # 3. Excessive water / flood alert
        if risk.get("flood_risk_level") in ("WARNING", "CRITICAL_WATERLOGGING"):
            flood_payload = {
                "water_level_pct": telemetry.get("water_level_pct"),
                "rain_detected": telemetry.get("rain_detected"),
                "flood_level": risk.get("flood_risk_level"),
                "location": location_str,
                "crop": crop_name or "General Canopy",
                "recommended_action": "Open field drainage sluices to mitigate hypoxic root damage.",
            }
            await self.dispatch_event("excessive_water_detected", f"flood_{risk.get('flood_risk_level')}", flood_payload)


n8n_dispatcher = N8NAlertDispatcher()
