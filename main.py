#!/usr/bin/env python3
"""
ISOLATED Frontend + API Server for Railway
Frontend routes FIRST, then backend
"""

import os
import sys
import logging
import uvicorn
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create a completely fresh app
app = FastAPI(title="Allocation Maximizer", redirect_slashes=False)

# Check if frontend exists FIRST
frontend_dist_path = os.path.join(os.path.dirname(__file__), 'frontend', 'dist')
logger.info(f"Looking for frontend at: {frontend_dist_path}")
logger.info(f"Frontend exists: {os.path.exists(frontend_dist_path)}")

if os.path.exists(frontend_dist_path):
    logger.info(f"Frontend contents: {os.listdir(frontend_dist_path)}")
    
    # Mount static assets FIRST - highest priority
    app.mount("/assets", StaticFiles(directory=os.path.join(frontend_dist_path, "assets")), name="assets")
    
    # Add a static file route for other static files
    app.mount("/static", StaticFiles(directory=frontend_dist_path), name="static")

# Health check for Railway
@app.get("/health")
async def health():
    return {"status": "healthy"}

# API routes BEFORE mounting backend to avoid conflicts
@app.get("/api/test")
async def api_test():
    return {"message": "Main app API working"}

# Add backend API at /api path - but AFTER our routes
try:
    backend_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'backend', 'api', 'module4')
    sys.path.insert(0, backend_path)
    from backend.api.module4.main import app as backend_app
    app.mount("/api/backend", backend_app)  # Mount at /api/backend to avoid conflict
    logger.info("Backend API mounted at /api/backend")
except ImportError as e:
    logger.error(f"Backend import failed: {e}")

# Frontend route LAST - this catches everything else
if os.path.exists(frontend_dist_path):
    @app.get("/{full_path:path}")
    async def serve_frontend(full_path: str, request: Request):
        logger.info(f"Frontend request for: {full_path} from {request.url}")
        
        # If it's a file request, try to serve it
        if "." in full_path:
            file_path = os.path.join(frontend_dist_path, full_path)
            if os.path.exists(file_path):
                logger.info(f"Serving file: {file_path}")
                return FileResponse(file_path)
        
        # For everything else, serve index.html (SPA)
        index_path = os.path.join(frontend_dist_path, 'index.html')
        logger.info(f"Serving SPA index: {index_path}")
        return FileResponse(index_path)
else:
    @app.get("/{full_path:path}")
    async def no_frontend(full_path: str):
        return {"error": "Frontend not found", "path": frontend_dist_path, "requested": full_path}

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