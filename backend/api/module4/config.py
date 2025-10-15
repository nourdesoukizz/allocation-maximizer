"""
Configuration settings for Module 4 API
"""

import os
from typing import List, Optional
from functools import lru_cache

try:
    from pydantic_settings import BaseSettings
except ImportError:
    from pydantic import BaseSettings


class Settings(BaseSettings):
    """Application settings"""
    
    # Application settings
    app_name: str = "Allocation Maximizer API"
    environment: str = "development"
    debug: bool = True
    version: str = "1.0.0"
    
    # Server settings
    host: str = "localhost"
    port: int = 8004
    
    # CORS settings
    allowed_origins: List[str] = ["*"]  # Allow all origins for production deployment
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Override CORS origins from environment variable if set
        cors_origins = os.getenv("CORS_ORIGINS", "").strip()
        if cors_origins:
            if cors_origins == "*":
                self.allowed_origins = ["*"]
            else:
                self.allowed_origins = [origin.strip() for origin in cors_origins.split(",")]
        elif self.environment == "production":
            # In production, allow all origins unless explicitly set
            self.allowed_origins = ["*"]
    
    # Data paths
    data_directory: str = "../../data/module4"
    upload_directory: str = "../../data/module4/uploads"
    
    # Optimization settings
    max_optimization_time: int = 300  # 5 minutes timeout
    max_file_size_mb: int = 10
    supported_file_formats: List[str] = ["csv", "xlsx", "json"]
    
    # Logging settings
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Cache settings
    redis_url: str = "redis://localhost:6379/0"
    cache_ttl_default: int = 3600  # 1 hour
    cache_ttl_file_processing: int = 1800  # 30 minutes
    cache_ttl_optimization: int = 3600  # 1 hour
    
    class Config:
        env_file = ".env"
        env_prefix = "MODULE4_"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()