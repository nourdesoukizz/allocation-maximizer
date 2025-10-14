#!/usr/bin/env python3
"""
Main entry point for Allocation Maximizer Backend
For Railway deployment - serves both API and frontend
"""

import os
import sys
import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Add the backend path to sys.path
backend_path = os.path.join(os.path.dirname(__file__), 'backend', 'api', 'module4')
sys.path.insert(0, backend_path)

from main import app as backend_app

# Create main app that serves both API and frontend
app = FastAPI(title="Allocation Maximizer - Full Stack")

# Mount the backend API
app.mount("/api", backend_app)

# Serve static frontend files
frontend_dist_path = os.path.join(os.path.dirname(__file__), 'frontend', 'dist')
if os.path.exists(frontend_dist_path):
    app.mount("/assets", StaticFiles(directory=os.path.join(frontend_dist_path, 'assets')), name="assets")
    
    @app.get("/{full_path:path}")
    async def serve_frontend(full_path: str):
        """Serve frontend for all non-API routes"""
        if full_path.startswith("api/"):
            # This shouldn't happen due to mount order, but just in case
            return {"error": "API route not found"}
        
        # Serve index.html for all frontend routes (SPA routing)
        index_path = os.path.join(frontend_dist_path, 'index.html')
        return FileResponse(index_path)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=False
    )