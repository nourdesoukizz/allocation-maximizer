#!/usr/bin/env python3
"""
Main entry point for Allocation Maximizer Backend
For Railway deployment - serves both API and frontend
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

# Add the backend path to sys.path
backend_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'backend', 'api', 'module4')
sys.path.insert(0, backend_path)

logger.info(f"Backend path: {backend_path}")
logger.info(f"sys.path: {sys.path[:3]}")

# Import backend app with error handling
try:
    from backend.api.module4.main import app as backend_app
    logger.info("Successfully imported backend app")
except ImportError as e:
    logger.error(f"Failed to import backend app: {e}")
    # Fallback: create minimal backend app
    backend_app = FastAPI(title="Backend API - Initialization Failed")
    @backend_app.get("/health")
    async def backend_health():
        return {"status": "degraded", "error": "Backend initialization failed"}

# Create main app that serves both API and frontend
# Disable redirect_slashes to prevent 307 redirects on health checks
app = FastAPI(
    title="Allocation Maximizer - Full Stack",
    redirect_slashes=False
)

# Add root-level health endpoint for Railway - simple and direct
# This MUST be defined before mounting sub-apps and catch-all routes
@app.get("/health")
async def railway_health_check():
    """Simple health check for Railway deployment"""
    return {"status": "healthy", "service": "allocation-maximizer"}

# Mount the backend API under /api
app.mount("/api", backend_app)

# Serve static frontend files
frontend_dist_path = os.path.join(os.path.dirname(__file__), 'frontend', 'dist')
if os.path.exists(frontend_dist_path):
    app.mount("/assets", StaticFiles(directory=os.path.join(frontend_dist_path, 'assets')), name="assets")
    
    # Add root handler
    @app.get("/")
    async def serve_root():
        """Serve frontend at root"""
        index_path = os.path.join(frontend_dist_path, 'index.html')
        return FileResponse(index_path)
    
    @app.get("/{full_path:path}")
    async def serve_frontend(full_path: str):
        """Serve frontend for all non-API routes (SPA routing)"""
        # Serve index.html for all frontend routes
        index_path = os.path.join(frontend_dist_path, 'index.html')
        return FileResponse(index_path)
else:
    @app.get("/")
    async def no_frontend():
        """Message when frontend is not built"""
        return {"message": "Frontend not built. Run 'cd frontend && npm run build' to build the frontend."}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"Starting server on 0.0.0.0:{port}")
    logger.info(f"Environment PORT variable: {os.environ.get('PORT', 'not set')}")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info"
    )