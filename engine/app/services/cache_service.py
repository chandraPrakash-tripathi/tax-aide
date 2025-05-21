import json
import pickle
import logging
from typing import Any, Dict, List, Optional, Union
import redis
from datetime import timedelta
import time

from app.config import settings

logger = logging.getLogger(__name__)

class CacheService:
    """Service for caching data using Redis"""
    
    def __init__(self):
        self.redis_client = None
        self._initialize_redis()
    
    def _initialize_redis(self):
        """Initialize Redis connection if enabled in settings"""
        if settings.REDIS_ENABLED:
            try:
                self.redis_client = redis.Redis(
                    host=settings.REDIS_HOST,
                    port=settings.REDIS_PORT,
                    db=settings.REDIS_DB,
                    decode_responses=False  # We'll handle decoding ourselves
                )
                # Test connection
                self.redis_client.ping()
                logger.info("Redis cache initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize Redis cache: {str(e)}")
                self.redis_client = None
    
    def set(self, key: str, value: Any, expires_in: int = 3600) -> bool:
        """
        Store a value in the cache
        
        Args:
            key: The cache key
            value: The value to store (will be pickled)
            expires_in: Time in seconds until expiration (default: 1 hour)
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.redis_client:
            return False
        
        try:
            pickled_value = pickle.dumps(value)
            expiry = timedelta(seconds=expires_in)
            
            self.redis_client.set(key, pickled_value, ex=expiry)
            return True
        except Exception as e:
            logger.error(f"Cache set error for key {key}: {str(e)}")
            return False
    
    def get(self, key: str) -> Optional[Any]:
        """
        Retrieve a value from the cache
        
        Args:
            key: The cache key
            
        Returns:
            The stored value or None if not found or error
        """
        if not self.redis_client:
            return None
        
        try:
            value = self.redis_client.get(key)
            if value:
                return pickle.loads(value)
            return None
        except Exception as e:
            logger.error(f"Cache get error for key {key}: {str(e)}")
            return None
    
    def delete(self, key: str) -> bool:
        """
        Delete a value from the cache
        
        Args:
            key: The cache key
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.redis_client:
            return False
        
        try:
            self.redis_client.delete(key)
            return True
        except Exception as e:
            logger.error(f"Cache delete error for key {key}: {str(e)}")
            return False
    
    def cache_reconciliation_result(self, user_id: str, result_data: Dict[str, Any]) -> str:
        """
        Cache reconciliation results with a unique key
        
        Args:
            user_id: User ID for whom to cache results
            result_data: The reconciliation result data to cache
            
        Returns:
            str: Cache key for later retrieval
        """
        import uuid
        import time
        
        cache_key = f"reconciliation:{user_id}:{uuid.uuid4()}"
        timestamp = int(time.time())
        
        # Add metadata to the cached result
        data_to_cache = {
            "timestamp": timestamp,
            "user_id": user_id,
            "result": result_data
        }
        
        self.set(cache_key, data_to_cache, expires_in=settings.RECONCILIATION_CACHE_TTL)
        return cache_key
    
    def get_reconciliation_result(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached reconciliation results
        
        Args:
            cache_key: The unique cache key
            
        Returns:
            Optional[Dict[str, Any]]: The cached result or None if not found
        """
        return self.get(cache_key)
    
    def cache_user_files(self, user_id: str, file_paths: Dict[str, str]) -> str:
        """
        Cache information about uploaded files for a user
        
        Args:
            user_id: User ID
            file_paths: Dictionary mapping file types to file paths
            
        Returns:
            str: Cache key for later retrieval
        """
        cache_key = f"files:{user_id}:{int(time.time())}"
        self.set(cache_key, file_paths, expires_in=settings.FILE_CACHE_TTL)
        return cache_key
    
    def get_user_files(self, cache_key: str) -> Optional[Dict[str, str]]:
        """
        Retrieve cached file information
        
        Args:
            cache_key: The unique cache key
            
        Returns:
            Optional[Dict[str, str]]: The cached file paths or None if not found
        """
        return self.get(cache_key)