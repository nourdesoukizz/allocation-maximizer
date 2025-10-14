"""
Health check endpoints for Module 4 API
"""

import logging
import time
from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException
from typing import Dict, Any

from optimizers.base_optimizer import OptimizerFactory, OptimizerType
from services.cache_service import get_cache_service
from middleware.security import request_logging_middleware, security_middleware
from config import get_settings

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/")
async def health_check() -> Dict[str, Any]:
    """
    Basic health check endpoint
    
    Returns:
        Health status information
    """
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "service": "Allocation Maximizer API",
        "module": "Module 4",
        "version": "1.0.0"
    }

@router.get("/detailed")
async def detailed_health_check() -> Dict[str, Any]:
    """
    Detailed health check with component status
    
    Returns:
        Detailed health status information
    """
    start_time = time.time()
    settings = get_settings()
    
    health_info = {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "service": "Allocation Maximizer API",
        "module": "Module 4",
        "version": "1.0.0",
        "environment": settings.environment,
        "components": {},
        "performance": {}
    }
    
    try:
        # Check optimizer components
        optimizers_status = await _check_optimizers()
        health_info["components"]["optimizers"] = optimizers_status
        
        # Check configuration
        config_status = await _check_configuration()
        health_info["components"]["configuration"] = config_status
        
        # Check cache service
        cache_status = await _check_cache_service()
        health_info["components"]["cache"] = cache_status
        
        # Performance metrics
        response_time = time.time() - start_time
        health_info["performance"]["response_time_ms"] = round(response_time * 1000, 2)
        health_info["performance"]["uptime_seconds"] = round(time.time() - start_time, 2)
        
        # Overall health assessment
        component_statuses = [comp["status"] for comp in health_info["components"].values()]
        if all(status == "healthy" for status in component_statuses):
            health_info["status"] = "healthy"
        elif any(status == "unhealthy" for status in component_statuses):
            health_info["status"] = "unhealthy"
        else:
            health_info["status"] = "degraded"
            
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        health_info["status"] = "unhealthy"
        health_info["error"] = str(e)
        
    return health_info

@router.get("/status")
async def service_status() -> Dict[str, Any]:
    """
    Service status endpoint
    
    Returns:
        Current service status and metrics
    """
    return {
        "service": "Allocation Maximizer API",
        "status": "running",
        "module": "Module 4",
        "capabilities": [
            "priority_based_optimization",
            "fair_share_optimization",
            "hybrid_optimization",
            "automatic_strategy_selection",
            "sku_substitution",
            "allocation_validation"
        ],
        "supported_file_formats": ["csv", "xlsx", "json"],
        "api_endpoints": {
            "health": "/health",
            "optimization": "/optimization",
            "documentation": "/docs"
        }
    }

async def _check_optimizers() -> Dict[str, Any]:
    """Check optimizer components health"""
    try:
        # Test optimizer factory
        available_optimizers = OptimizerFactory.get_available_optimizers()
        
        optimizer_tests = {}
        for optimizer_type in available_optimizers:
            try:
                optimizer = OptimizerFactory.create_optimizer(optimizer_type)
                optimizer_tests[optimizer_type.value] = {
                    "status": "healthy",
                    "type": optimizer_type.value,
                    "initialized": True
                }
            except Exception as e:
                optimizer_tests[optimizer_type.value] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
        
        return {
            "status": "healthy",
            "available_optimizers": len(available_optimizers),
            "optimizer_details": optimizer_tests
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }

@router.get("/metrics")
async def performance_metrics() -> Dict[str, Any]:
    """
    Get performance and security metrics
    
    Returns:
        Performance metrics including request counts, response times, and error rates
    """
    try:
        # Get request metrics from middleware
        request_metrics = request_logging_middleware.get_metrics()
        
        # Get cache stats
        cache_service = await get_cache_service()
        if hasattr(cache_service, 'get_cache_stats'):
            cache_stats = await cache_service.get_cache_stats()
        elif hasattr(cache_service, 'get_stats'):
            cache_stats = cache_service.get_stats()
        else:
            cache_stats = {"type": "unavailable"}
        
        return {
            "status": "healthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "performance": {
                "requests": request_metrics,
                "cache": cache_stats
            },
            "security": {
                "api_keys_configured": len(security_middleware.api_keys),
                "security_headers_enabled": True,
                "input_validation_enabled": True,
                "rate_limiting_enabled": True
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get performance metrics: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

@router.post("/security/add-api-key")
async def add_api_key(
    key_name: str,
    permissions: list = None
) -> Dict[str, Any]:
    """
    Add a new API key (for administrative use)
    
    Args:
        key_name: Name for the API key
        permissions: List of permissions for the key
        
    Returns:
        Generated API key information
    """
    try:
        import secrets
        
        # Generate a secure API key
        api_key = f"ak_{secrets.token_urlsafe(32)}"
        
        # Add to security middleware
        security_middleware.add_api_key(api_key, key_name, permissions)
        
        return {
            "success": True,
            "api_key": api_key,
            "name": key_name,
            "permissions": permissions or ["read", "write"],
            "message": "API key created successfully",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to create API key: {e}")
        return {
            "success": False,
            "error": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

async def _check_configuration() -> Dict[str, Any]:
    """Check configuration health"""
    try:
        settings = get_settings()
        
        return {
            "status": "healthy",
            "environment": settings.environment,
            "debug_mode": settings.debug,
            "host": settings.host,
            "port": settings.port,
            "cors_configured": len(settings.allowed_origins) > 0,
            "data_directory": settings.data_directory
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }

@router.get("/metrics")
async def performance_metrics() -> Dict[str, Any]:
    """
    Get performance and security metrics
    
    Returns:
        Performance metrics including request counts, response times, and error rates
    """
    try:
        # Get request metrics from middleware
        request_metrics = request_logging_middleware.get_metrics()
        
        # Get cache stats
        cache_service = await get_cache_service()
        if hasattr(cache_service, 'get_cache_stats'):
            cache_stats = await cache_service.get_cache_stats()
        elif hasattr(cache_service, 'get_stats'):
            cache_stats = cache_service.get_stats()
        else:
            cache_stats = {"type": "unavailable"}
        
        return {
            "status": "healthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "performance": {
                "requests": request_metrics,
                "cache": cache_stats
            },
            "security": {
                "api_keys_configured": len(security_middleware.api_keys),
                "security_headers_enabled": True,
                "input_validation_enabled": True,
                "rate_limiting_enabled": True
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get performance metrics: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

@router.post("/security/add-api-key")
async def add_api_key(
    key_name: str,
    permissions: list = None
) -> Dict[str, Any]:
    """
    Add a new API key (for administrative use)
    
    Args:
        key_name: Name for the API key
        permissions: List of permissions for the key
        
    Returns:
        Generated API key information
    """
    try:
        import secrets
        
        # Generate a secure API key
        api_key = f"ak_{secrets.token_urlsafe(32)}"
        
        # Add to security middleware
        security_middleware.add_api_key(api_key, key_name, permissions)
        
        return {
            "success": True,
            "api_key": api_key,
            "name": key_name,
            "permissions": permissions or ["read", "write"],
            "message": "API key created successfully",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to create API key: {e}")
        return {
            "success": False,
            "error": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

@router.get("/cache-stats")
async def cache_statistics() -> Dict[str, Any]:
    """
    Get cache statistics and health information
    
    Returns:
        Cache statistics and performance metrics
    """
    try:
        cache_service = await get_cache_service()
        
        if hasattr(cache_service, 'get_cache_stats'):
            stats = await cache_service.get_cache_stats()
        elif hasattr(cache_service, 'get_stats'):
            stats = cache_service.get_stats()
        else:
            stats = {"type": "in_memory", "available": True}
        
        return {
            "status": "healthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "cache_type": "redis" if hasattr(cache_service, 'redis_client') else "in_memory",
            "statistics": stats
        }
        
    except Exception as e:
        logger.error(f"Failed to get cache statistics: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

@router.post("/cache-clear")
async def clear_cache() -> Dict[str, Any]:
    """
    Clear all cache data (use with caution!)
    
    Returns:
        Cache clear operation result
    """
    try:
        cache_service = await get_cache_service()
        
        if hasattr(cache_service, 'clear_cache'):
            success = await cache_service.clear_cache()
        elif hasattr(cache_service, 'clear'):
            cache_service.clear()
            success = True
        else:
            success = False
        
        return {
            "success": success,
            "message": "Cache cleared successfully" if success else "Failed to clear cache",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to clear cache: {e}")
        return {
            "success": False,
            "error": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

@router.delete("/cache/{pattern}")
async def invalidate_cache_pattern(pattern: str) -> Dict[str, Any]:
    """
    Invalidate cache entries matching a pattern
    
    Args:
        pattern: Cache key pattern to invalidate (e.g., "optimization:*")
        
    Returns:
        Cache invalidation result
    """
    try:
        cache_service = await get_cache_service()
        
        if hasattr(cache_service, 'delete_pattern'):
            deleted_count = await cache_service.delete_pattern(pattern)
        else:
            # For in-memory cache, manually delete matching keys
            deleted_count = 0
            if hasattr(cache_service, '_cache'):
                keys_to_delete = [
                    key for key in cache_service._cache.keys() 
                    if pattern.replace('*', '') in key
                ]
                for key in keys_to_delete:
                    cache_service.delete(key)
                    deleted_count += 1
        
        return {
            "success": True,
            "pattern": pattern,
            "deleted_count": deleted_count,
            "message": f"Invalidated {deleted_count} cache entries matching pattern '{pattern}'",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to invalidate cache pattern '{pattern}': {e}")
        return {
            "success": False,
            "error": str(e),
            "pattern": pattern,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

async def _check_cache_service() -> Dict[str, Any]:
    """Check cache service health"""
    try:
        cache_service = await get_cache_service()
        
        if hasattr(cache_service, 'redis_client') and cache_service.redis_client:
            # Redis cache
            try:
                stats = await cache_service.get_cache_stats()
                return {
                    "status": "healthy",
                    "type": "redis",
                    "connected": stats.get("connected", False),
                    "memory_used": stats.get("used_memory", "Unknown"),
                    "hit_rate": stats.get("hit_rate", 0)
                }
            except Exception as e:
                return {
                    "status": "degraded",
                    "type": "redis", 
                    "error": str(e),
                    "fallback": "using in-memory cache"
                }
        else:
            # In-memory cache
            if hasattr(cache_service, 'get_stats'):
                stats = cache_service.get_stats()
                return {
                    "status": "healthy",
                    "type": "in_memory",
                    "entries": stats.get("entries", 0),
                    "hit_rate": stats.get("hit_rate", 0)
                }
            else:
                return {
                    "status": "healthy",
                    "type": "in_memory",
                    "note": "Basic cache service available"
                }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }

@router.get("/metrics")
async def performance_metrics() -> Dict[str, Any]:
    """
    Get performance and security metrics
    
    Returns:
        Performance metrics including request counts, response times, and error rates
    """
    try:
        # Get request metrics from middleware
        request_metrics = request_logging_middleware.get_metrics()
        
        # Get cache stats
        cache_service = await get_cache_service()
        if hasattr(cache_service, 'get_cache_stats'):
            cache_stats = await cache_service.get_cache_stats()
        elif hasattr(cache_service, 'get_stats'):
            cache_stats = cache_service.get_stats()
        else:
            cache_stats = {"type": "unavailable"}
        
        return {
            "status": "healthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "performance": {
                "requests": request_metrics,
                "cache": cache_stats
            },
            "security": {
                "api_keys_configured": len(security_middleware.api_keys),
                "security_headers_enabled": True,
                "input_validation_enabled": True,
                "rate_limiting_enabled": True
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get performance metrics: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

@router.post("/security/add-api-key")
async def add_api_key(
    key_name: str,
    permissions: list = None
) -> Dict[str, Any]:
    """
    Add a new API key (for administrative use)
    
    Args:
        key_name: Name for the API key
        permissions: List of permissions for the key
        
    Returns:
        Generated API key information
    """
    try:
        import secrets
        
        # Generate a secure API key
        api_key = f"ak_{secrets.token_urlsafe(32)}"
        
        # Add to security middleware
        security_middleware.add_api_key(api_key, key_name, permissions)
        
        return {
            "success": True,
            "api_key": api_key,
            "name": key_name,
            "permissions": permissions or ["read", "write"],
            "message": "API key created successfully",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to create API key: {e}")
        return {
            "success": False,
            "error": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }