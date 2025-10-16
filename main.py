#!/usr/bin/env python3
"""
Main entry point for Allocation Maximizer Backend
For Railway deployment - serves both API and frontend
"""

import os
import sys
import logging
import uvicorn
from fastapi import FastAPI, HTTPException
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

# Add root-level health endpoint FIRST - before any mounts or middleware
# This ensures it gets priority in routing and avoids backend middleware
@app.get("/health")
async def railway_health_check():
    """Simple health check for Railway deployment - must be first route"""
    logger.info("Health check endpoint called")
    return {"status": "healthy", "service": "allocation-maximizer"}

# Add a middleware to intercept requests before they reach the backend
@app.middleware("http")
async def intercept_frontend_routes(request, call_next):
    """Intercept frontend routes before backend middleware can handle them"""
    path = request.url.path
    
    # Handle root route for frontend
    if path == "/":
        logger.info("Middleware intercepting root route for frontend")
        frontend_dist_path = os.path.join(os.path.dirname(__file__), 'frontend', 'dist')
        if os.path.exists(frontend_dist_path):
            index_path = os.path.join(frontend_dist_path, 'index.html')
            logger.info(f"Serving frontend from: {index_path}")
            logger.info(f"Frontend dist contents: {os.listdir(frontend_dist_path) if os.path.exists(frontend_dist_path) else 'NOT FOUND'}")
            assets_dir = os.path.join(frontend_dist_path, 'assets')
            if os.path.exists(assets_dir):
                logger.info(f"Assets directory contents: {os.listdir(assets_dir)}")
            from fastapi.responses import FileResponse
            return FileResponse(index_path)
        else:
            logger.error("Frontend dist directory not found!")
            return {"error": "Frontend not built"}
    
    # Continue with normal request processing
    return await call_next(request)

# Mount the backend API under /api AFTER middleware setup
app.mount("/api", backend_app)

# Serve static frontend files FIRST
frontend_dist_path = os.path.join(os.path.dirname(__file__), 'frontend', 'dist')
if os.path.exists(frontend_dist_path):
    # Mount the entire dist directory for static files (handles vite.svg, etc.)
    app.mount("/static", StaticFiles(directory=frontend_dist_path), name="static")
    
    # Mount assets directory specifically 
    assets_path = os.path.join(frontend_dist_path, 'assets')
    if os.path.exists(assets_path):
        app.mount("/assets", StaticFiles(directory=assets_path), name="assets")
    
    # Handle specific static files at root level
    @app.get("/vite.svg")
    async def serve_vite_svg():
        """Serve vite.svg from frontend dist"""
        svg_path = os.path.join(frontend_dist_path, 'vite.svg')
        if os.path.exists(svg_path):
            return FileResponse(svg_path)
        raise HTTPException(status_code=404, detail="vite.svg not found")
    
    # Serve index.html for all frontend routes (SPA)
    @app.get("/")
    async def serve_root():
        """Serve frontend at root"""
        index_path = os.path.join(frontend_dist_path, 'index.html')
        logger.info(f"Serving frontend from: {index_path}")
        logger.info(f"Frontend dist contents: {os.listdir(frontend_dist_path) if os.path.exists(frontend_dist_path) else 'NOT FOUND'}")
        assets_dir = os.path.join(frontend_dist_path, 'assets')
        if os.path.exists(assets_dir):
            logger.info(f"Assets directory contents: {os.listdir(assets_dir)}")
        return FileResponse(index_path)
    
    # Catch-all route for SPA routing (must be last)
    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        """Serve SPA for all non-API routes"""
        # Skip API routes, health, assets, and static files
        if (full_path.startswith("api/") or 
            full_path == "health" or 
            full_path.startswith("assets/") or 
            full_path.startswith("static/") or
            full_path.endswith(('.js', '.css', '.svg', '.png', '.jpg', '.ico'))):
            raise HTTPException(status_code=404, detail="Not found")
        
        index_path = os.path.join(frontend_dist_path, 'index.html')
        logger.info(f"SPA routing: {full_path} -> {index_path}")
        return FileResponse(index_path)
else:
    @app.get("/")
    async def no_frontend():
        return {"message": "Frontend not built. Build the frontend with 'cd frontend && npm run build'"}

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