"""
Caching service for allocation data and results with Redis support
"""

import asyncio
import json
import gzip
import pickle
import logging
from typing import Any, Optional, Union, Dict, List
from datetime import datetime, timedelta
from dataclasses import dataclass
import redis.asyncio as redis
from redis.asyncio import Redis
from config import get_settings

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Cache entry with metadata"""
    data: Any
    created_at: datetime
    expires_at: Optional[datetime] = None
    access_count: int = 0
    last_accessed: Optional[datetime] = None


class InMemoryCache:
    """In-memory caching service with TTL support"""
    
    def __init__(self, default_ttl_minutes: int = 15):
        """
        Initialize cache service
        
        Args:
            default_ttl_minutes: Default TTL for cache entries
        """
        self._cache: Dict[str, CacheEntry] = {}
        self.default_ttl_minutes = default_ttl_minutes
        self._stats = {
            'hits': 0,
            'misses': 0,
            'sets': 0,
            'evictions': 0
        }
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found/expired
        """
        if key not in self._cache:
            self._stats['misses'] += 1
            return None
        
        entry = self._cache[key]
        
        # Check if expired
        if entry.expires_at and datetime.now() > entry.expires_at:
            logger.debug(f"Cache key '{key}' expired, removing")
            del self._cache[key]
            self._stats['misses'] += 1
            self._stats['evictions'] += 1
            return None
        
        # Update access metadata
        entry.access_count += 1
        entry.last_accessed = datetime.now()
        
        self._stats['hits'] += 1
        logger.debug(f"Cache hit for key '{key}'")
        return entry.data
    
    def set(
        self, 
        key: str, 
        value: Any, 
        ttl_minutes: Optional[int] = None
    ) -> None:
        """
        Set value in cache
        
        Args:
            key: Cache key
            value: Value to cache
            ttl_minutes: TTL in minutes (uses default if not specified)
        """
        ttl = ttl_minutes or self.default_ttl_minutes
        expires_at = datetime.now() + timedelta(minutes=ttl) if ttl > 0 else None
        
        entry = CacheEntry(
            data=value,
            created_at=datetime.now(),
            expires_at=expires_at,
            access_count=0
        )
        
        self._cache[key] = entry
        self._stats['sets'] += 1
        
        logger.debug(f"Cached key '{key}' with TTL {ttl} minutes")
    
    def delete(self, key: str) -> bool:
        """
        Delete key from cache
        
        Args:
            key: Cache key to delete
            
        Returns:
            True if key was deleted, False if not found
        """
        if key in self._cache:
            del self._cache[key]
            logger.debug(f"Deleted cache key '{key}'")
            return True
        return False
    
    def clear(self) -> None:
        """Clear all cache entries"""
        count = len(self._cache)
        self._cache.clear()
        self._stats['evictions'] += count
        logger.info(f"Cleared {count} cache entries")
    
    def cleanup_expired(self) -> int:
        """
        Remove expired entries from cache
        
        Returns:
            Number of entries removed
        """
        now = datetime.now()
        expired_keys = [
            key for key, entry in self._cache.items()
            if entry.expires_at and now > entry.expires_at
        ]
        
        for key in expired_keys:
            del self._cache[key]
        
        if expired_keys:
            self._stats['evictions'] += len(expired_keys)
            logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")
        
        return len(expired_keys)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics
        
        Returns:
            Dictionary with cache statistics
        """
        total_requests = self._stats['hits'] + self._stats['misses']
        hit_rate = (self._stats['hits'] / total_requests * 100) if total_requests > 0 else 0
        
        return {
            'entries': len(self._cache),
            'hits': self._stats['hits'],
            'misses': self._stats['misses'],
            'sets': self._stats['sets'],
            'evictions': self._stats['evictions'],
            'hit_rate': round(hit_rate, 2),
            'total_requests': total_requests
        }
    
    def get_cache_info(self) -> Dict[str, Dict]:
        """
        Get detailed information about cache entries
        
        Returns:
            Dictionary with cache entry details
        """
        info = {}
        now = datetime.now()
        
        for key, entry in self._cache.items():
            age_minutes = (now - entry.created_at).total_seconds() / 60
            ttl_remaining = None
            
            if entry.expires_at:
                ttl_remaining = (entry.expires_at - now).total_seconds() / 60
                ttl_remaining = max(0, ttl_remaining)  # Don't show negative TTL
            
            info[key] = {
                'age_minutes': round(age_minutes, 2),
                'ttl_remaining_minutes': round(ttl_remaining, 2) if ttl_remaining is not None else None,
                'access_count': entry.access_count,
                'last_accessed': entry.last_accessed.isoformat() if entry.last_accessed else None,
                'data_type': type(entry.data).__name__,
                'data_size_approx': len(str(entry.data))
            }
        
        return info


class DataCache:
    """High-level cache service for allocation data"""
    
    def __init__(self, cache_service: InMemoryCache):
        """
        Initialize data cache
        
        Args:
            cache_service: Underlying cache service
        """
        self.cache = cache_service
    
    # CSV Data Caching
    def get_csv_data(self) -> Optional[Any]:
        """Get cached CSV data"""
        return self.cache.get("csv_data")
    
    def set_csv_data(self, data: Any, ttl_minutes: int = 30) -> None:
        """Cache CSV data"""
        self.cache.set("csv_data", data, ttl_minutes)
    
    # Data Summary Caching
    def get_data_summary(self) -> Optional[Any]:
        """Get cached data summary"""
        return self.cache.get("data_summary")
    
    def set_data_summary(self, summary: Any, ttl_minutes: int = 15) -> None:
        """Cache data summary"""
        self.cache.set("data_summary", summary, ttl_minutes)
    
    # Filtered Data Caching
    def get_filtered_data(self, filter_key: str) -> Optional[Any]:
        """Get cached filtered data"""
        return self.cache.get(f"filtered_data:{filter_key}")
    
    def set_filtered_data(self, filter_key: str, data: Any, ttl_minutes: int = 10) -> None:
        """Cache filtered data"""
        self.cache.set(f"filtered_data:{filter_key}", data, ttl_minutes)
    
    # ML Model Results Caching
    def get_ml_prediction(self, model_key: str) -> Optional[Any]:
        """Get cached ML model prediction"""
        return self.cache.get(f"ml_prediction:{model_key}")
    
    def set_ml_prediction(self, model_key: str, prediction: Any, ttl_minutes: int = 60) -> None:
        """Cache ML model prediction"""
        self.cache.set(f"ml_prediction:{model_key}", prediction, ttl_minutes)
    
    # Allocation Results Caching
    def get_allocation_result(self, batch_id: str) -> Optional[Any]:
        """Get cached allocation result"""
        return self.cache.get(f"allocation_result:{batch_id}")
    
    def set_allocation_result(self, batch_id: str, result: Any, ttl_minutes: int = 120) -> None:
        """Cache allocation result"""
        self.cache.set(f"allocation_result:{batch_id}", result, ttl_minutes)
    
    # Cache Management
    def invalidate_csv_cache(self) -> None:
        """Invalidate all CSV-related cache entries"""
        keys_to_delete = []
        for key in self.cache._cache.keys():
            if key.startswith(('csv_data', 'data_summary', 'filtered_data:')):
                keys_to_delete.append(key)
        
        for key in keys_to_delete:
            self.cache.delete(key)
        
        logger.info(f"Invalidated {len(keys_to_delete)} CSV-related cache entries")
    
    def cleanup_old_predictions(self, max_age_hours: int = 24) -> int:
        """
        Clean up old ML predictions
        
        Args:
            max_age_hours: Maximum age for predictions
            
        Returns:
            Number of entries cleaned up
        """
        now = datetime.now()
        cutoff_time = now - timedelta(hours=max_age_hours)
        
        keys_to_delete = []
        for key, entry in self.cache._cache.items():
            if (key.startswith('ml_prediction:') and 
                entry.created_at < cutoff_time):
                keys_to_delete.append(key)
        
        for key in keys_to_delete:
            self.cache.delete(key)
        
        if keys_to_delete:
            logger.info(f"Cleaned up {len(keys_to_delete)} old ML predictions")
        
        return len(keys_to_delete)


class RedisCache:
    """
    Redis-based caching service with compression and TTL support
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.redis_client: Optional[Redis] = None
        self.default_ttl = 3600  # 1 hour default TTL
        self.compression_threshold = 1024  # Compress values larger than 1KB
        
    async def connect(self):
        """Connect to Redis server"""
        try:
            self.redis_client = redis.from_url(
                self.settings.redis_url,
                encoding="utf-8",
                decode_responses=False,  # We'll handle encoding ourselves
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True,
                health_check_interval=30
            )
            
            # Test the connection
            await self.redis_client.ping()
            logger.info("Connected to Redis successfully")
            
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self.redis_client = None
            raise
    
    async def disconnect(self):
        """Disconnect from Redis server"""
        if self.redis_client:
            await self.redis_client.close()
            self.redis_client = None
            logger.info("Disconnected from Redis")
    
    def _serialize_value(self, value: Any) -> bytes:
        """Serialize value to bytes with optional compression"""
        try:
            # First, serialize to JSON if possible (for better readability)
            if isinstance(value, (dict, list, str, int, float, bool)) or value is None:
                serialized = json.dumps(value, default=str).encode('utf-8')
            else:
                # Fall back to pickle for complex objects
                serialized = pickle.dumps(value)
            
            # Compress if value is large enough
            if len(serialized) > self.compression_threshold:
                compressed = gzip.compress(serialized)
                # Only use compression if it actually reduces size
                if len(compressed) < len(serialized):
                    return b'GZIP:' + compressed
            
            return b'RAW:' + serialized
            
        except Exception as e:
            logger.error(f"Failed to serialize value: {e}")
            raise
    
    def _deserialize_value(self, data: bytes) -> Any:
        """Deserialize bytes back to original value"""
        try:
            if data.startswith(b'GZIP:'):
                # Decompress and deserialize
                compressed_data = data[5:]  # Remove 'GZIP:' prefix
                decompressed = gzip.decompress(compressed_data)
                
                # Try JSON first, then pickle
                try:
                    return json.loads(decompressed.decode('utf-8'))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    return pickle.loads(decompressed)
                    
            elif data.startswith(b'RAW:'):
                # Direct deserialization
                raw_data = data[4:]  # Remove 'RAW:' prefix
                
                # Try JSON first, then pickle
                try:
                    return json.loads(raw_data.decode('utf-8'))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    return pickle.loads(raw_data)
            else:
                # Legacy format fallback
                try:
                    return json.loads(data.decode('utf-8'))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    return pickle.loads(data)
                    
        except Exception as e:
            logger.error(f"Failed to deserialize value: {e}")
            raise
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        if not self.redis_client:
            logger.warning("Redis client not connected")
            return None
            
        try:
            data = await self.redis_client.get(key)
            if data is None:
                return None
                
            return self._deserialize_value(data)
            
        except Exception as e:
            logger.error(f"Failed to get cache key '{key}': {e}")
            return None
    
    async def set(
        self, 
        key: str, 
        value: Any, 
        ttl: Optional[Union[int, timedelta]] = None
    ) -> bool:
        """Set value in cache with optional TTL"""
        if not self.redis_client:
            logger.warning("Redis client not connected")
            return False
            
        try:
            serialized_value = self._serialize_value(value)
            
            # Convert timedelta to seconds
            if isinstance(ttl, timedelta):
                ttl = int(ttl.total_seconds())
            elif ttl is None:
                ttl = self.default_ttl
                
            await self.redis_client.setex(key, ttl, serialized_value)
            return True
            
        except Exception as e:
            logger.error(f"Failed to set cache key '{key}': {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete key from cache"""
        if not self.redis_client:
            logger.warning("Redis client not connected")
            return False
            
        try:
            result = await self.redis_client.delete(key)
            return result > 0
            
        except Exception as e:
            logger.error(f"Failed to delete cache key '{key}': {e}")
            return False
    
    async def delete_pattern(self, pattern: str) -> int:
        """Delete all keys matching pattern"""
        if not self.redis_client:
            logger.warning("Redis client not connected")
            return 0
            
        try:
            keys = await self.redis_client.keys(pattern)
            if keys:
                deleted = await self.redis_client.delete(*keys)
                return deleted
            return 0
            
        except Exception as e:
            logger.error(f"Failed to delete cache pattern '{pattern}': {e}")
            return 0
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        if not self.redis_client:
            return False
            
        try:
            return bool(await self.redis_client.exists(key))
            
        except Exception as e:
            logger.error(f"Failed to check cache key existence '{key}': {e}")
            return False
    
    async def get_ttl(self, key: str) -> Optional[int]:
        """Get TTL for a key"""
        if not self.redis_client:
            return None
            
        try:
            ttl = await self.redis_client.ttl(key)
            return ttl if ttl > 0 else None
            
        except Exception as e:
            logger.error(f"Failed to get TTL for key '{key}': {e}")
            return None
    
    async def extend_ttl(self, key: str, ttl: Union[int, timedelta]) -> bool:
        """Extend TTL for an existing key"""
        if not self.redis_client:
            return False
            
        try:
            if isinstance(ttl, timedelta):
                ttl = int(ttl.total_seconds())
                
            return bool(await self.redis_client.expire(key, ttl))
            
        except Exception as e:
            logger.error(f"Failed to extend TTL for key '{key}': {e}")
            return False
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        if not self.redis_client:
            return {"connected": False}
            
        try:
            info = await self.redis_client.info()
            stats = {
                "connected": True,
                "used_memory": info.get("used_memory_human", "Unknown"),
                "used_memory_bytes": info.get("used_memory", 0),
                "total_connections_received": info.get("total_connections_received", 0),
                "total_commands_processed": info.get("total_commands_processed", 0),
                "keyspace_hits": info.get("keyspace_hits", 0),
                "keyspace_misses": info.get("keyspace_misses", 0),
                "connected_clients": info.get("connected_clients", 0),
                "uptime_in_seconds": info.get("uptime_in_seconds", 0),
            }
            
            # Calculate hit rate
            hits = stats["keyspace_hits"]
            misses = stats["keyspace_misses"]
            total_requests = hits + misses
            stats["hit_rate"] = (hits / total_requests * 100) if total_requests > 0 else 0
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get cache stats: {e}")
            return {"connected": False, "error": str(e)}
    
    async def clear_cache(self) -> bool:
        """Clear all cache data (use with caution!)"""
        if not self.redis_client:
            return False
            
        try:
            await self.redis_client.flushdb()
            logger.info("Cache cleared successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
            return False


class CacheKeys:
    """Cache key generators for different data types"""
    
    @staticmethod
    def csv_data(file_path: str) -> str:
        """Generate cache key for CSV data"""
        return f"csv_data:{hash(file_path)}"
    
    @staticmethod
    def optimization_result(request_hash: str) -> str:
        """Generate cache key for optimization results"""
        return f"optimization:{request_hash}"
    
    @staticmethod
    def model_prediction(model_name: str, data_hash: str) -> str:
        """Generate cache key for model predictions"""
        return f"model:{model_name}:{data_hash}"
    
    @staticmethod
    def strategy_comparison(data_hash: str) -> str:
        """Generate cache key for strategy comparisons"""
        return f"strategy_comparison:{data_hash}"
    
    @staticmethod
    def data_validation(data_hash: str) -> str:
        """Generate cache key for data validation results"""
        return f"validation:{data_hash}"


# Cache service instance
_redis_cache = None
_memory_cache = None

async def get_redis_cache() -> RedisCache:
    """Get the global Redis cache service instance"""
    global _redis_cache
    
    if _redis_cache is None:
        _redis_cache = RedisCache()
        try:
            await _redis_cache.connect()
        except Exception as e:
            logger.warning(f"Redis not available, falling back to in-memory cache: {e}")
            _redis_cache = None
    
    return _redis_cache

def get_memory_cache() -> InMemoryCache:
    """Get the global in-memory cache service instance"""
    global _memory_cache
    
    if _memory_cache is None:
        _memory_cache = InMemoryCache(default_ttl_minutes=15)
    
    return _memory_cache

async def get_cache_service() -> Union[RedisCache, InMemoryCache]:
    """Get the best available cache service (Redis preferred, fallback to in-memory)"""
    redis_cache = await get_redis_cache()
    if redis_cache is not None:
        return redis_cache
    
    logger.info("Using in-memory cache as Redis is not available")
    return get_memory_cache()

async def close_cache_service():
    """Close the global cache service instances"""
    global _redis_cache, _memory_cache
    
    if _redis_cache is not None:
        await _redis_cache.disconnect()
        _redis_cache = None
        
    if _memory_cache is not None:
        _memory_cache.clear()
        _memory_cache = None


# Global cache instances for backward compatibility
memory_cache = get_memory_cache()
data_cache = DataCache(memory_cache)