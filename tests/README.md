# AudioJudge Tests

This directory contains tests for the AudioJudge package.

## Test Structure

- `conftest.py`: Contains shared fixtures and configuration for all tests
- `test_init.py`: Tests for package imports
- `test_core.py`: Tests for the core AudioJudge class functionality
- `test_utils.py`: Tests for utility classes and functions
- `test_api_cache.py`: Tests for the API caching functionality
- `test_integration.py`: Integration tests that make actual API calls
- `test_audiojudge.py`: Basic functionality tests

## Running Tests

### Prerequisites

- Python 3.10 or higher
- Install the package in development mode:
  ```bash
  pip install -e .
  ```
- Install test dependencies:
  ```bash
  pip install -e ".[dev]"
  ```

### Running All Tests

```bash
pytest
```

### Running Specific Tests

```bash
# Run tests in a specific file
pytest tests/test_core.py

# Run tests with a specific name pattern
pytest -k "test_cache"

# Run tests and show output
pytest -v
```

### Skipping Integration Tests

Integration tests require API keys to be set as environment variables:

```bash
export OPENAI_API_KEY="your-openai-key"
export GOOGLE_API_KEY="your-google-key"
```

If these keys are not available, integration tests will be skipped automatically.

## Writing New Tests

When adding new tests:

1. Follow the existing naming conventions
2. Use fixtures from `conftest.py` where appropriate
3. Mock external API calls when testing functionality
4. Add integration tests for new features, but make them skippable

## Code Coverage

To generate a code coverage report:

```bash
pytest --cov=audiojudge tests/
``` 