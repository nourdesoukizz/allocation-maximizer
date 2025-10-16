#!/usr/bin/env python3
"""
SIMPLE Frontend + API Server for Railway
NO complex middleware, NO backend interference
"""

import os
import sys
import logging
import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create a completely fresh app
app = FastAPI(title="Allocation Maximizer", redirect_slashes=False)

# Health check
@app.get("/health")
async def health():
    return {"status": "healthy"}

# Check if frontend exists
frontend_dist_path = os.path.join(os.path.dirname(__file__), 'frontend', 'dist')
logger.info(f"Looking for frontend at: {frontend_dist_path}")
logger.info(f"Frontend exists: {os.path.exists(frontend_dist_path)}")

if os.path.exists(frontend_dist_path):
    logger.info(f"Frontend contents: {os.listdir(frontend_dist_path)}")
    
    # Mount static files
    app.mount("/assets", StaticFiles(directory=os.path.join(frontend_dist_path, "assets")), name="assets")
    
    # Serve frontend
    @app.get("/")
    async def serve_frontend():
        index_path = os.path.join(frontend_dist_path, 'index.html')
        logger.info(f"Serving: {index_path}")
        return FileResponse(index_path)
else:
    @app.get("/")
    async def no_frontend():
        return {"error": "Frontend not found", "path": frontend_dist_path}

# Add backend API
try:
    backend_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'backend', 'api', 'module4')
    sys.path.insert(0, backend_path)
    from backend.api.module4.main import app as backend_app
    app.mount("/api", backend_app)
    logger.info("Backend API mounted at /api")
except ImportError as e:
    logger.error(f"Backend import failed: {e}")
    @app.get("/api/status")
    async def api_status():
        return {"error": "Backend not available"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"Starting server on 0.0.0.0:{port}")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info"
    )