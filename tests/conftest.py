import pytest
import os
import sys
from pathlib import Path

# Add the src directory to the Python path for tests
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


@pytest.fixture
def sample_audio_paths():
    """Fixture to provide sample audio paths for testing."""
    base_dir = Path("src/audiojudge/signal_audios")
    return {
        "audio1": str(base_dir / "audio_1.wav"),
        "audio2": str(base_dir / "audio_2.wav"),
        "instruction": str(base_dir / "instruction.wav"),
        "test_audio": str(base_dir / "test_audio.wav"),
        "audio": str(base_dir / "audio.wav"),
    }
