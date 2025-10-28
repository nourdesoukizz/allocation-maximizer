#!/usr/bin/env python3
"""
Production Frontend Server for Railway
Serves the built React frontend with fallback to index.html for SPA routing
"""

import os
import logging
import uvicorn
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create main app
app = FastAPI(
    title="Allocation Maximizer Frontend",
    description="Production frontend server for Railway deployment",
    version="1.0.0"
)

# CORS configuration for production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check for Railway
@app.get("/health")
async def health():
    return {
        "status": "healthy", 
        "service": "frontend",
        "frontend": "enabled",
        "version": "1.0.0"
    }

# Check if frontend exists and mount it
frontend_dist_path = os.path.join(os.path.dirname(__file__), 'frontend', 'dist')
logger.info(f"Looking for frontend at: {frontend_dist_path}")
logger.info(f"Frontend exists: {os.path.exists(frontend_dist_path)}")

if os.path.exists(frontend_dist_path):
    logger.info(f"Frontend contents: {os.listdir(frontend_dist_path)}")
    
    # Mount static assets
    app.mount("/assets", StaticFiles(directory=os.path.join(frontend_dist_path, "assets")), name="assets")
    
    # Frontend route - serves all non-API/health requests (catch-all for SPA)
    @app.get("/{full_path:path}")
    async def serve_frontend(full_path: str, request: Request):        
        # If it's a file request, try to serve it directly
        if "." in full_path and not full_path.startswith("assets/"):
            file_path = os.path.join(frontend_dist_path, full_path)
            if os.path.exists(file_path):
                return FileResponse(file_path)
        
        # For everything else (SPA routes), serve index.html
        index_path = os.path.join(frontend_dist_path, 'index.html')
        return FileResponse(index_path)
else:
    logger.error("Frontend dist directory not found! Build may have failed.")
    
    @app.get("/{full_path:path}")
    async def no_frontend(full_path: str):
        return {
            "error": "Frontend not found", 
            "path": frontend_dist_path, 
            "requested": full_path,
            "message": "Please check if frontend build succeeded"
        }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"Starting frontend server on 0.0.0.0:{port}")
    logger.info(f"Health check available at: /health")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info"
    )