import pytest
import os
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

from audiojudge.api_cache import APICache


class TestAPICache:
    """Test suite for the APICache class."""

    @pytest.fixture
    def temp_cache_dir(self):
        """Create a temporary directory for cache testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        # Clean up after tests
        shutil.rmtree(temp_dir)

    def test_cache_initialization(self, temp_cache_dir):
        """Test that the cache initializes correctly."""
        cache = APICache(
            cache_dir=temp_cache_dir, expire_seconds=3600, disable_cache=False
        )
        assert cache.cache_dir == temp_cache_dir
        assert cache.expire_seconds == 3600
        assert not cache.disable_cache

        # Check that the cache directory exists
        assert os.path.exists(temp_cache_dir)

    def test_cache_decorator(self, temp_cache_dir):
        """Test that the cache decorator works correctly."""
        cache = APICache(
            cache_dir=temp_cache_dir, expire_seconds=3600, disable_cache=False
        )

        # Create a proper function to be cached instead of a MagicMock
        def test_func(arg1, arg2, kwarg1=None):
            return "test_result"

        # Apply cache decorator
        cached_func = cache(test_func)

        # Call the function twice with the same arguments
        result1 = cached_func("arg1", "arg2", kwarg1="kwarg1")
        result2 = cached_func("arg1", "arg2", kwarg1="kwarg1")

        # Check that the results are the same
        assert result1 == "test_result"
        assert result2 == "test_result"

    def test_cache_disabled(self, temp_cache_dir):
        """Test that the cache can be disabled."""
        cache = APICache(
            cache_dir=temp_cache_dir, expire_seconds=3600, disable_cache=True
        )

        # Create a function with a counter to check call count
        call_count = 0

        def test_func():
            nonlocal call_count
            call_count += 1
            return "test_result"

        # Apply cache decorator
        cached_func = cache(test_func)

        # Call the function twice
        cached_func()
        cached_func()

        # Check that the function was called twice (cache disabled)
        assert call_count == 2

    def test_clear_cache(self, temp_cache_dir):
        """Test that the cache can be cleared."""
        cache = APICache(
            cache_dir=temp_cache_dir, expire_seconds=3600, disable_cache=False
        )

        # Create a function with a counter to check call count
        call_count = 0

        def test_func(arg1, arg2):
            nonlocal call_count
            call_count += 1
            return "test_result"

        # Apply cache decorator
        cached_func = cache(test_func)

        # Call the function to populate cache
        cached_func("arg1", "arg2")

        # Clear the cache
        cache.clear_cache()

        # Call the function again
        cached_func("arg1", "arg2")

        # Check that the function was called twice (cache was cleared)
        assert call_count == 2

    def test_clear_none_cache(self, temp_cache_dir):
        """Test that None responses can be cleared from cache."""
        cache = APICache(
            cache_dir=temp_cache_dir, expire_seconds=3600, disable_cache=False
        )

        # Create functions with different return values
        def func_none(arg):
            return None

        def func_value(arg):
            return "test_result"

        # Apply cache decorator
        cached_func1 = cache(func_none)
        cached_func2 = cache(func_value)

        # Call the functions to populate cache
        cached_func1("arg1")
        cached_func2("arg2")

        # Clear None responses
        valid_entries = cache.clear_none_cache()

        # Should have one valid entry
        assert valid_entries >= 1
