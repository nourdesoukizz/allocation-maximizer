"""
FastAPI application for Module 4: Allocation Maximizer
"""

import logging
import uvicorn
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import Dict, Any
import traceback

from routers import optimization, health
from config import get_settings
from services.cache_service import get_cache_service, close_cache_service
from middleware.security import (
    setup_rate_limiting, 
    request_logging_middleware, 
    security_middleware
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get application settings
settings = get_settings()

# Create FastAPI application
app = FastAPI(
    title="Allocation Maximizer API",
    description="Module 4: Advanced allocation optimization with priority and fair share strategies",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# Add security middleware
app.middleware("http")(request_logging_middleware)

# Setup rate limiting
setup_rate_limiting(app)

# Include routers
app.include_router(health.router, prefix="/health", tags=["Health"])
app.include_router(optimization.router, prefix="/optimization", tags=["Optimization"])

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for unhandled errors"""
    logger.error(f"Unhandled exception: {exc}")
    logger.error(f"Traceback: {traceback.format_exc()}")
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred. Please try again.",
            "request_id": getattr(request.state, 'request_id', 'unknown')
        }
    )

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "service": "Allocation Maximizer API",
        "module": "Module 4",
        "version": "1.0.0",
        "status": "running",
        "docs_url": "/docs",
        "health_check": "/health"
    }

@app.on_event("startup")
async def startup_event():
    """Application startup event"""
    logger.info("Starting Allocation Maximizer API (Module 4)")
    logger.info(f"Environment: {settings.environment}")
    logger.info(f"Debug mode: {settings.debug}")
    
    # Initialize cache service
    try:
        cache_service = await get_cache_service()
        if hasattr(cache_service, 'redis_client') and cache_service.redis_client:
            logger.info("Redis cache service initialized successfully")
        else:
            logger.info("In-memory cache service initialized (Redis not available)")
    except Exception as e:
        logger.warning(f"Cache service initialization failed: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown event"""
    logger.info("Shutting down Allocation Maximizer API (Module 4)")
    
    # Close cache service
    try:
        await close_cache_service()
        logger.info("Cache service closed successfully")
    except Exception as e:
        logger.error(f"Error closing cache service: {e}")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level="info"
    )