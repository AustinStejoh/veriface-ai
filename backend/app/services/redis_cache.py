"""
Redis Cache Module
Provides caching functionality for model predictions and expensive operations
"""

import redis
import hashlib
import json
import time
from typing import Any, Optional, Callable
from functools import wraps
from PIL import Image
import io
from logger_config import cache_logger, log_cache_operation

# Redis configuration
REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_DB = 0
CACHE_TTL = 3600  # 1 hour default TTL

# Cache key prefixes
PREDICTION_CACHE_PREFIX = "pred:"
HEATMAP_CACHE_PREFIX = "hm:"
ROBUSTNESS_CACHE_PREFIX = "robust:"
METRICS_CACHE_PREFIX = "metrics:"


class RedisCache:
    """Redis caching client with automatic connection management"""
    
    def __init__(self, host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB):
        self.host = host
        self.port = port
        self.db = db
        self._client = None
        self._connected = False
    
    @property
    def client(self) -> Optional[redis.Redis]:
        """Lazy initialization of Redis client"""
        if self._client is None:
            try:
                self._client = redis.Redis(
                    host=self.host,
                    port=self.port,
                    db=self.db,
                    decode_responses=False,
                    socket_connect_timeout=5,
                    socket_keepalive=True,
                    health_check_interval=30
                )
                # Test connection
                self._client.ping()
                self._connected = True
                cache_logger.info(f"[OK] Connected to Redis at {self.host}:{self.port}")
            except Exception as e:
                cache_logger.warning(f"⚠ Redis connection failed: {e}. Running without cache.")
                self._connected = False
                self._client = None
        
        return self._client
    
    def is_connected(self) -> bool:
        """Check if Redis is available"""
        if not self._connected:
            try:
                self.client.ping()
                self._connected = True
            except:
                self._connected = False
        return self._connected
    
    def get_image_hash(self, image_data: bytes) -> str:
        """
        Generate a unique hash for image data
        
        Args:
            image_data: Raw image bytes
        
        Returns:
            SHA256 hash of image
        """
        return hashlib.sha256(image_data).hexdigest()[:16]
    
    def set(self, key: str, value: Any, ttl: int = CACHE_TTL) -> bool:
        """
        Set a value in cache
        
        Args:
            key: Cache key
            value: Value to cache (will be JSON serialized)
            ttl: Time-to-live in seconds
        
        Returns:
            True if successful, False otherwise
        """
        if not self.is_connected():
            return False
        
        try:
            start_time = time.time()
            serialized = json.dumps(value).encode('utf-8')
            self.client.setex(key, ttl, serialized)
            duration_ms = (time.time() - start_time) * 1000
            log_cache_operation("SET", key, duration_ms=duration_ms)
            return True
        except Exception as e:
            cache_logger.error(f"Cache SET failed for {key}: {e}")
            return False
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get a value from cache
        
        Args:
            key: Cache key
        
        Returns:
            Cached value or None if not found/expired
        """
        if not self.is_connected():
            return None
        
        try:
            start_time = time.time()
            value = self.client.get(key)
            duration_ms = (time.time() - start_time) * 1000
            
            if value is not None:
                log_cache_operation("GET", key, hit=True, duration_ms=duration_ms)
                return json.loads(value.decode('utf-8'))
            else:
                log_cache_operation("GET", key, hit=False, duration_ms=duration_ms)
                return None
        except Exception as e:
            cache_logger.error(f"Cache GET failed for {key}: {e}")
            return None
    
    def delete(self, key: str) -> bool:
        """Delete a key from cache"""
        if not self.is_connected():
            return False
        
        try:
            self.client.delete(key)
            log_cache_operation("DEL", key)
            return True
        except Exception as e:
            cache_logger.error(f"Cache DELETE failed for {key}: {e}")
            return False
    
    def clear_prefix(self, prefix: str) -> int:
        """Delete all keys with a specific prefix"""
        if not self.is_connected():
            return 0
        
        try:
            pattern = f"{prefix}*"
            keys = self.client.keys(pattern)
            if keys:
                deleted = self.client.delete(*keys)
                cache_logger.info(f"Cleared {deleted} keys with prefix: {prefix}")
                return deleted
            return 0
        except Exception as e:
            cache_logger.error(f"Cache CLEAR_PREFIX failed for {prefix}: {e}")
            return 0
    
    def get_stats(self) -> dict:
        """Get Redis statistics"""
        if not self.is_connected():
            return {"connected": False}
        
        try:
            info = self.client.info()
            return {
                "connected": True,
                "used_memory_human": info.get("used_memory_human"),
                "connected_clients": info.get("connected_clients"),
                "total_commands_processed": info.get("total_commands_processed"),
                "uptime_in_days": info.get("uptime_in_days"),
            }
        except Exception as e:
            cache_logger.error(f"Failed to get Redis stats: {e}")
            return {"connected": False, "error": str(e)}


# Global cache instance
cache = RedisCache()


def cache_prediction(ttl: int = CACHE_TTL):
    """
    Decorator to cache prediction results
    
    Args:
        ttl: Cache time-to-live in seconds
    
    Example:
        @cache_prediction(ttl=7200)
        def predict(image_data):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(image_data: bytes, *args, **kwargs):
            # Generate cache key
            image_hash = cache.get_image_hash(image_data)
            cache_key = f"{PREDICTION_CACHE_PREFIX}{image_hash}"
            
            # Try to get from cache
            cached_result = cache.get(cache_key)
            if cached_result is not None:
                cache_logger.info(f"✓ Prediction cache HIT for image {image_hash}")
                return cached_result
            
            # Cache miss - execute function
            cache_logger.info(f"✗ Prediction cache MISS for image {image_hash}")
            result = func(image_data, *args, **kwargs)
            
            # Store in cache
            cache.set(cache_key, result, ttl)
            
            return result
        
        return wrapper
    return decorator


def cache_result(prefix: str = "cache:", ttl: int = CACHE_TTL):
    """
    Generic decorator to cache function results
    
    Args:
        prefix: Cache key prefix
        ttl: Cache time-to-live in seconds
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key from function name and args
            cache_key = f"{prefix}{func.__name__}:{hash(str(args) + str(kwargs))}"
            
            # Try to get from cache
            cached_result = cache.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Cache miss - execute function
            result = func(*args, **kwargs)
            
            # Store in cache
            cache.set(cache_key, result, ttl)
            
            return result
        
        return wrapper
    return decorator
