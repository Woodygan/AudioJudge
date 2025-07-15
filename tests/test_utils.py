import pytest
from pathlib import Path

from audiojudge.utils import AudioExample, AudioExamplePointwise


@pytest.fixture
def sample_audio_paths():
    """Fixture to provide sample audio paths for testing."""
    base_dir = Path("src/audiojudge/signal_audios")
    return {
        "audio1": str(base_dir / "audio_1.wav"),
        "audio2": str(base_dir / "audio_2.wav"),
        "instruction": str(base_dir / "instruction.wav"),
    }


class TestAudioExample:
    """Test suite for the AudioExample class."""

    def test_audio_example_creation(self, sample_audio_paths):
        """Test creating an AudioExample instance."""
        example = AudioExample(
            audio1_path=sample_audio_paths["audio1"],
            audio2_path=sample_audio_paths["audio2"],
            output="Audio 1 is better.",
        )

        assert example.audio1_path == sample_audio_paths["audio1"]
        assert example.audio2_path == sample_audio_paths["audio2"]
        assert example.output == "Audio 1 is better."
        assert example.instruction_path is None

    def test_audio_example_with_instruction(self, sample_audio_paths):
        """Test creating an AudioExample with instruction."""
        example = AudioExample(
            audio1_path=sample_audio_paths["audio1"],
            audio2_path=sample_audio_paths["audio2"],
            instruction_path=sample_audio_paths["instruction"],
            output="Audio 2 is better.",
        )

        assert example.audio1_path == sample_audio_paths["audio1"]
        assert example.audio2_path == sample_audio_paths["audio2"]
        assert example.instruction_path == sample_audio_paths["instruction"]
        assert example.output == "Audio 2 is better."


class TestAudioExamplePointwise:
    """Test suite for the AudioExamplePointwise class."""

    def test_audio_example_pointwise_creation(self, sample_audio_paths):
        """Test creating an AudioExamplePointwise instance."""
        example = AudioExamplePointwise(
            audio_path=sample_audio_paths["audio1"], output="8/10 - Good quality audio."
        )

        assert example.audio_path == sample_audio_paths["audio1"]
        assert example.output == "8/10 - Good quality audio."
