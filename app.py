"""
AI Powered Crop Intelligence Rover  -  Unified Production Server Entrypoint.

This module serves as the primary executable entrypoint for containerized and cloud deployments:
- Mounts the authoritative FastAPI application instance from `backend.app.main`.
- Inspects runtime environment variables (`HOST`, `PORT`) with automatic fallbacks:
    - Default Port: 7860 (optimized for cloud app hosts like Hugging Face Spaces and container engines).
    - Default Host: 0.0.0.0 (enables ingress from container networks, Docker bridges, and external proxies).
- Executes the high-performance Uvicorn ASGI server with standard access logging and signal handling.
"""

import os
import uvicorn
from backend.app.main import app

if __name__ == "__main__":
    # Resolve host and port dynamically from container environment or default to cloud port 7860
    port = int(os.environ.get("PORT", 7860))
    host = os.environ.get("HOST", "0.0.0.0")

    print(f"================================================================================")
    print(f" Starting AI Powered Crop Intelligence Rover Unified Production Server")
    print(f" Bound to: http://{host}:{port}")
    print(f"================================================================================")

    # Launch production ASGI server loop
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info",
        access_log=True,
    )
