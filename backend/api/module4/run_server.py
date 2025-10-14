#!/usr/bin/env python3
"""
Run script for Module 4 Allocation Maximizer API
"""

import uvicorn
from main import app
from config import get_settings

if __name__ == "__main__":
    settings = get_settings()
    
    print(f"Starting Allocation Maximizer API (Module 4)")
    print(f"Server: http://{settings.host}:{settings.port}")
    print(f"API Documentation: http://{settings.host}:{settings.port}/docs")
    print(f"Environment: {settings.environment}")
    
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level="info"
    )