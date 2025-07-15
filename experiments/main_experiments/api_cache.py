import json
import hashlib
import functools
import os
import diskcache
from typing import Callable, Any

# Initialize disk cache for API calls
CACHE_DIR = os.environ.get("EVAL_CACHE_DIR", ".eval_cache")
api_cache_store = diskcache.Cache(CACHE_DIR)

# Cache expiration time (default: 30 days)
CACHE_EXPIRE_SECONDS = int(os.environ.get("EVAL_CACHE_EXPIRE", 60 * 60 * 24 * 30))


def create_cache_key(*args, **kwargs) -> str:
    """
    Create a unique cache key based on function arguments

    Args:
        *args, **kwargs: Arguments to create hash from

    Returns:
        A unique hash string to use as cache key
    """
    # Serialize all arguments to a string
    args_str = json.dumps(args, sort_keys=True, default=str)
    kwargs_str = json.dumps(kwargs, sort_keys=True, default=str)

    # Get hash of arguments
    args_hash = hashlib.md5((args_str + kwargs_str).encode()).hexdigest()

    # Include cache seed if available
    cache_seed = os.environ.get("EVAL_CACHE_SEED", "")

    # Create final key
    key_str = f"{args_hash}:{cache_seed}"
    return hashlib.md5(key_str.encode()).hexdigest()


def api_cache(func: Callable) -> Callable:
    """
    Decorator to cache API responses on disk

    Args:
        func: The function to decorate

    Returns:
        Wrapped function with caching
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Skip caching if disabled
        if os.environ.get("EVAL_DISABLE_CACHE", "").lower() in ("true", "1", "yes"):
            return func(*args, **kwargs)

        # Create cache key from all arguments
        cache_key = create_cache_key(*args, **kwargs)

        # Try to get from cache
        cached_result = api_cache_store.get(cache_key)
        if cached_result is not None:
            # Extract model name for logging if available
            model_name = args[0] if len(args) > 0 else kwargs.get("model", "unknown")
            print(f"Cache hit for API call: {model_name}")
            return cached_result

        # Call the original function
        result = func(*args, **kwargs)

        # Store in cache if we got a valid result
        if result is not None:
            # Extract model name for logging if available
            model_name = args[0] if len(args) > 0 else kwargs.get("model", "unknown")
            api_cache_store.set(cache_key, result, expire=CACHE_EXPIRE_SECONDS)
            print(f"Cached API response for {model_name}")

        return result

    return wrapper


def clear_cache():
    """Clear the entire API cache"""
    api_cache_store.clear()
    print(f"Cleared API cache at {CACHE_DIR}")


def clear_none_cache():
    """Clear only cache entries that returned None"""
    count = 0
    none_keys = []
    count_none = 0
    # First, collect all keys with None results
    for key in api_cache_store:
        value = api_cache_store.get(key)
        if value is None:
            none_keys.append(key)
        else:
            count += 1

    # Delete collected keys
    for key in none_keys:
        api_cache_store.delete(key)
        count_none += 1

    print(f"Cleared {count_none} None-returning entries from API cache at {CACHE_DIR}")
    print(f"Kept {count} valid entries in API cache at {CACHE_DIR}")
    return count
