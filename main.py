#!/usr/bin/env python3
"""
FRONTEND ONLY Server for Railway
NO backend mounting - frontend serves all routes
"""

import os
import logging
import uvicorn
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create a completely fresh app with NO backend interference
app = FastAPI(title="Allocation Maximizer Frontend", redirect_slashes=False)

# Check if frontend exists
frontend_dist_path = os.path.join(os.path.dirname(__file__), 'frontend', 'dist')
logger.info(f"Looking for frontend at: {frontend_dist_path}")
logger.info(f"Frontend exists: {os.path.exists(frontend_dist_path)}")

if os.path.exists(frontend_dist_path):
    logger.info(f"Frontend contents: {os.listdir(frontend_dist_path)}")
    
    # Mount static assets with highest priority
    app.mount("/assets", StaticFiles(directory=os.path.join(frontend_dist_path, "assets")), name="assets")

# Health check for Railway
@app.get("/health")
async def health():
    return {"status": "healthy", "service": "frontend_only"}

# Simple API status (no backend)
@app.get("/api/status")
async def api_status():
    return {"message": "Frontend server running, no backend mounted"}

# Frontend route - serves all non-API requests
if os.path.exists(frontend_dist_path):
    @app.get("/{full_path:path}")
    async def serve_frontend(full_path: str, request: Request):
        logger.info(f"Frontend request for: /{full_path}")
        
        # Skip API routes 
        if full_path.startswith("api/"):
            return {"error": "API not available in frontend-only mode"}
        
        # If it's a file request, try to serve it directly
        if "." in full_path and not full_path.startswith("assets/"):
            file_path = os.path.join(frontend_dist_path, full_path)
            if os.path.exists(file_path):
                logger.info(f"Serving file: {file_path}")
                return FileResponse(file_path)
        
        # For everything else (SPA routes), serve index.html
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