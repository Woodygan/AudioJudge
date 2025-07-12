import json
import base64
import os
import time
import wave
import audioop
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass
import hashlib
import functools
import diskcache

@dataclass
class APICache:
    """Standalone API cache that can be used as a decorator"""
    
    def __init__(self, 
                 cache_dir: Optional[str] = None,
                 expire_seconds: int = 60 * 60 * 24 * 30,  # 30 days
                 disable_cache: bool = False):
        """
        Initialize API cache
        
        Args:
            cache_dir: Cache directory path
            expire_seconds: Cache expiration time in seconds
            disable_cache: Whether to disable caching
        """
        self.cache_dir = cache_dir or os.environ.get("EVAL_CACHE_DIR", ".eval_cache")
        self.expire_seconds = expire_seconds
        self.disable_cache = disable_cache or os.environ.get("EVAL_DISABLE_CACHE", "").lower() in ("true", "1", "yes")
        
        if not self.disable_cache:
            os.makedirs(self.cache_dir, exist_ok=True)
            self.cache_store = diskcache.Cache(self.cache_dir)
            print(f"API cache initialized at: {os.path.abspath(self.cache_dir)}")
        else:
            self.cache_store = None
            print("API caching disabled")
    
    def _serialize_for_cache(self, value):
        """
        Serialize values for cache key creation, handling bytes specially
        """
        if isinstance(value, bytes):
            # Convert bytes to a hash instead of trying to serialize them
            return f"bytes_hash:{hashlib.md5(value).hexdigest()}"
        elif isinstance(value, list):
            return [self._serialize_for_cache(item) for item in value]
        elif isinstance(value, dict):
            return {k: self._serialize_for_cache(v) for k, v in value.items()}
        elif isinstance(value, tuple):
            return tuple(self._serialize_for_cache(item) for item in value)
        else:
            return value
    
    def _create_cache_key(self, func_name: str, *args, **kwargs) -> str:
        """Create a unique cache key based on function name and arguments"""
        # Serialize arguments safely, handling bytes
        serialized_args = self._serialize_for_cache(args)
        serialized_kwargs = self._serialize_for_cache(kwargs)
        
        cache_data = {
            "function": func_name,
            "args": serialized_args,
            "kwargs": serialized_kwargs
        }
        
        # Now we can safely serialize to JSON
        cache_str = json.dumps(cache_data, sort_keys=True, default=str)
        cache_seed = os.environ.get("EVAL_CACHE_SEED", "")
        key_str = f"{cache_str}:{cache_seed}"
        
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def __call__(self, func: Callable) -> Callable:
        """Decorator implementation"""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Skip caching if disabled
            if self.disable_cache or self.cache_store is None:
                return func(*args, **kwargs)
            
            # Create cache key
            cache_key = self._create_cache_key(func.__name__, *args, **kwargs)
            
            # Try to get from cache
            cached_result = self.cache_store.get(cache_key)
            if cached_result is not None:
                print(f"Cache hit for {func.__name__}")
                return cached_result
            
            # Call the original function
            result = func(*args, **kwargs)
            
            # Store in cache if we got a valid result
            if result is not None:
                self.cache_store.set(cache_key, result, expire=self.expire_seconds)
                print(f"Cached result for {func.__name__}")
            
            return result
        
        # Add cache management methods to the wrapper
        wrapper.clear_cache = lambda: self.clear_cache()
        wrapper.clear_none_cache = lambda: self.clear_none_cache()
        wrapper.get_cache_stats = lambda: self.get_cache_stats()
        
        return wrapper
    
    def clear_cache(self):
        """Clear the entire cache"""
        if self.cache_store is not None:
            self.cache_store.clear()
            print(f"Cleared API cache at {self.cache_dir}")
    
    def clear_none_cache(self):
        """Clear only cache entries that returned None"""
        if self.cache_store is None:
            return 0
        
        none_keys = []
        valid_count = 0
        
        for key in self.cache_store:
            value = self.cache_store.get(key)
            if value is None:
                none_keys.append(key)
            else:
                valid_count += 1
        
        for key in none_keys:
            self.cache_store.delete(key)
        
        print(f"Cleared {len(none_keys)} None entries, kept {valid_count} valid entries")
        return valid_count
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        if self.cache_store is None:
            return {"cache_enabled": False}
        
        total_entries = len(self.cache_store)
        return {
            "cache_enabled": True,
            "cache_dir": self.cache_dir,
            "total_entries": total_entries,
            "expire_seconds": self.expire_seconds
        }