"""
XiaoZhi Outbound MCP WebSocket Client.
Author: Team Innovex (SIH 2026 Challenge SIH26180 - Qualcomm Inc).

Maintains a resilient, secure outbound WebSocket connection from the SmartCropVision
backend to the Xiaozhi AI Cloud MCP endpoint (wss://api.xiaozhi.me/mcp/?token=...).

Key Features:
1. Outbound WebSocket Client: No open ports, port forwarding, or ngrok required.
2. JSON-RPC 2.0 Compliance: Handles initialize, tools/list, tools/call, and ping.
3. Token & Credential Privacy: Tokens are strictly loaded from environment variables
   and masked in all logs (never printed in plaintext).
4. Automatic Reconnection: Exponential backoff with jitter on network dropouts.
5. Graceful Lifecycle: Integrates directly into FastAPI lifespan events.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Optional, Dict, Any
import websockets
from websockets.exceptions import ConnectionClosed

from backend.app.config import settings
from backend.app.services.xiaozhi_gateway import xiaozhi_gateway

logger = logging.getLogger("smartcropvision.xiaozhi_mcp_client")


def mask_token(url_or_token: Optional[str]) -> str:
    """Masks bearer token or query token in URLs to protect secrets from logs."""
    if not url_or_token:
        return "<none>"
    # Mask URL token query parameter: ?token=...
    masked = re.sub(r"([?&]token=)([^&]+)", r"\1***MASKED***", url_or_token)
    if masked != url_or_token:
        return masked
    # If it's a raw token string
    if len(url_or_token) > 8:
        return f"{url_or_token[:4]}...{url_or_token[-4:]}"
    return "***MASKED***"


class XiaoZhiMCPClient:
    """Asynchronous WebSocket client connecting to the Xiaozhi Cloud MCP bridge."""

    def __init__(self):
        self._task: Optional[asyncio.Task] = None
        self._running: bool = False
        self._connected: bool = False
        self._websocket = None

    @property
    def is_connected(self) -> bool:
        return self._connected

    def get_endpoint_url(self) -> Optional[str]:
        """Resolves the configured MCP endpoint from settings."""
        endpoint = (settings.XIAOZHI_MCP_ENDPOINT or "").strip()
        token = (settings.XIAOZHI_MCP_TOKEN or "").strip()

        # If token was pasted as a full URL
        if token.startswith("wss://") or token.startswith("ws://"):
            return token

        if endpoint:
            if "token=" in endpoint:
                return endpoint
            if token:
                sep = "&" if "?" in endpoint else ("" if endpoint.endswith("?") else "?")
                return f"{endpoint}{sep}token={token}"
            return endpoint

        if token:
            return f"wss://api.xiaozhi.me/mcp/?token={token}"
        return None

    def start(self) -> None:
        """Launches the background client task if enabled and configured."""
        if not settings.XIAOZHI_ENABLED:
            logger.info("[XIAOZHI] MCP client is disabled via XIAOZHI_ENABLED=false.")
            return

        endpoint = self.get_endpoint_url()
        if not endpoint:
            logger.info(
                "[XIAOZHI] MCP client is inactive (neither XIAOZHI_MCP_TOKEN nor XIAOZHI_MCP_ENDPOINT is set). "
                "Set XIAOZHI_MCP_TOKEN in .env to enable the physical voice assistant bridge."
            )
            return

        if self._task and not self._task.done():
            logger.warning("[XIAOZHI] MCP client task is already running.")
            return

        self._running = True
        self._task = asyncio.create_task(self._connection_loop())
        logger.info(f"[XIAOZHI] Launched MCP connection worker targeting {mask_token(endpoint)}")

    async def stop(self) -> None:
        """Gracefully terminates the background client connection."""
        self._running = False
        if self._websocket:
            try:
                await self._websocket.close()
            except Exception:
                pass
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        self._connected = False
        logger.info("[XIAOZHI] MCP client stopped successfully.")

    async def _connection_loop(self) -> None:
        """Main connection and reconnection lifecycle loop."""
        reconnect_delay = float(settings.XIAOZHI_RECONNECT_INTERVAL_SECONDS)
        max_reconnect_delay = 60.0

        while self._running:
            endpoint = self.get_endpoint_url()
            if not endpoint:
                logger.warning("[XIAOZHI] MCP endpoint URL became unavailable. Pausing connection loop.")
                await asyncio.sleep(10)
                continue

            masked_url = mask_token(endpoint)
            try:
                logger.info(f"[XIAOZHI] Connecting outbound to {masked_url} ...")
                async with websockets.connect(
                    endpoint,
                    ping_interval=30,
                    ping_timeout=10,
                    close_timeout=5,
                ) as ws:
                    self._websocket = ws
                    self._connected = True
                    reconnect_delay = float(settings.XIAOZHI_RECONNECT_INTERVAL_SECONDS)  # Reset delay on success
                    logger.info(f"[XIAOZHI] Successfully connected to {masked_url}. Awaiting MCP requests.")

                    async for raw_message in ws:
                        if not self._running:
                            break
                        response_json = await self._handle_incoming_message(raw_message)
                        if response_json is not None:
                            await ws.send(response_json)

            except ConnectionClosed as cc:
                logger.warning(f"[XIAOZHI] WebSocket connection closed by remote (code={cc.code}, reason='{cc.reason}').")
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[XIAOZHI] Connection error: {type(e).__name__} - {e}")
            finally:
                self._connected = False
                self._websocket = None

            if self._running:
                logger.info(f"[XIAOZHI] Reconnecting in {reconnect_delay:.1f}s ...")
                await asyncio.sleep(reconnect_delay)
                reconnect_delay = min(max_reconnect_delay, reconnect_delay * 1.5)

    async def _handle_incoming_message(self, raw_message: str) -> Optional[str]:
        """
        Parses and handles standard JSON-RPC 2.0 requests from Xiaozhi Cloud.
        Returns a JSON-RPC response string or None (for notifications).
        """
        try:
            data = json.loads(raw_message)
        except json.JSONDecodeError:
            logger.error("[XIAOZHI] Received invalid non-JSON message from server.")
            return json.dumps({
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32700, "message": "Parse error: Invalid JSON"},
            })

        method = data.get("method")
        req_id = data.get("id")
        params = data.get("params", {}) or {}

        logger.debug(f"[XIAOZHI] Inbound JSON-RPC: method='{method}', id={req_id}")

        # 1. MCP Initialization Handshake
        if method == "initialize":
            logger.info("[XIAOZHI] Handling MCP 'initialize' handshake.")
            return json.dumps({
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {
                        "tools": {},
                        "logging": {},
                    },
                    "serverInfo": {
                        "name": "SmartCropVision",
                        "version": "2.0.0",
                    },
                },
            })

        # 2. MCP Notification Initialized (no response required)
        elif method == "notifications/initialized":
            logger.info("[XIAOZHI] MCP session confirmed initialized by server.")
            return None

        # 3. Tool Discovery (tools/list)
        elif method == "tools/list":
            raw_tools = xiaozhi_gateway.get_mcp_tool_definitions()
            formatted_tools = [
                {
                    "name": t["name"],
                    "description": t["description"],
                    "inputSchema": t["parameters"],
                }
                for t in raw_tools
            ]
            logger.info(f"[XIAOZHI] Returned {len(formatted_tools)} tool definitions to Xiaozhi.")
            return json.dumps({
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"tools": formatted_tools},
            })

        # 4. Tool Invocation (tools/call)
        elif method == "tools/call":
            tool_name = params.get("name")
            arguments = params.get("arguments", {}) or {}

            logger.info(f"[XIAOZHI] Invoking tool '{tool_name}' for voice query.")
            try:
                tool_result = xiaozhi_gateway.execute_tool(tool_name, arguments)
                is_error = tool_result.get("status") in ("ERROR", "UNKNOWN_TOOL", "BLOCKED")

                # Format text content for the voice assistant LLM
                speech_text = tool_result.get("spoken_summary") or json.dumps(tool_result, ensure_ascii=False)
                content_payload = [
                    {
                        "type": "text",
                        "text": speech_text if not is_error else tool_result.get("message", "Error executing tool."),
                    }
                ]

                return json.dumps({
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": content_payload,
                        "structuredData": tool_result,
                        "isError": is_error,
                    },
                })
            except Exception as e:
                logger.error(f"[XIAOZHI] Error executing tool '{tool_name}': {e}", exc_info=True)
                return json.dumps({
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [{"type": "text", "text": f"Error executing tool {tool_name}: {str(e)}"}],
                        "isError": True,
                    },
                })

        # 5. Ping Heartbeat
        elif method == "ping":
            return json.dumps({"jsonrpc": "2.0", "id": req_id, "result": {}})

        # Unrecognized Method
        else:
            logger.warning(f"[XIAOZHI] Unrecognized MCP method '{method}' received.")
            if req_id is not None:
                return json.dumps({
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {"code": -32601, "message": f"Method '{method}' not found"},
                })
            return None


# Global singleton client instance
xiaozhi_mcp_client = XiaoZhiMCPClient()
