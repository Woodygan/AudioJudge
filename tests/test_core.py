import pytest
import os
import json
from unittest.mock import patch, MagicMock
from pathlib import Path

from audiojudge import AudioJudge
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


@pytest.fixture
def mock_audiojudge():
    """Create a mocked AudioJudge instance that doesn't make actual API calls."""
    with (
        patch("audiojudge.core.OpenAI"),
        patch("audiojudge.core.genai"),
        patch.object(AudioJudge, "_get_model_response", return_value="Mocked response"),
    ):
        judge = AudioJudge(
            openai_api_key="fake_key", google_api_key="fake_key", disable_cache=True
        )
        yield judge


class TestAudioJudge:
    """Test suite for the AudioJudge class."""

    def test_initialization(self):
        """Test that AudioJudge initializes correctly."""
        with patch("audiojudge.core.OpenAI"), patch("audiojudge.core.genai"):
            judge = AudioJudge(
                openai_api_key="fake_key",
                google_api_key="fake_key",
                temp_dir="test_temp",
                signal_folder="test_signals",
                cache_dir="test_cache",
                disable_cache=True,
            )

            # Check that directories were created
            assert os.path.exists("test_temp")
            assert os.path.exists("test_signals")

            # Clean up
            os.rmdir("test_temp")
            os.rmdir("test_signals")

    def test_judge_audio_basic(self, mock_audiojudge, sample_audio_paths):
        """Test the basic judge_audio functionality."""
        result = mock_audiojudge.judge_audio(
            audio1_path=sample_audio_paths["audio1"],
            audio2_path=sample_audio_paths["audio2"],
            system_prompt="Compare these two audio clips.",
            model="gpt-4o-audio-preview",
        )

        assert result["success"] is True
        assert result["response"] == "Mocked response"
        assert result["model"] == "gpt-4o-audio-preview"
        assert result["audio1_path"] == sample_audio_paths["audio1"]
        assert result["audio2_path"] == sample_audio_paths["audio2"]

    def test_judge_audio_with_instruction(self, mock_audiojudge, sample_audio_paths):
        """Test judge_audio with instruction audio."""
        result = mock_audiojudge.judge_audio(
            audio1_path=sample_audio_paths["audio1"],
            audio2_path=sample_audio_paths["audio2"],
            instruction_path=sample_audio_paths["instruction"],
            system_prompt="Compare these two audio clips based on the instruction.",
            model="gpt-4o-audio-preview",
        )

        assert result["success"] is True
        assert result["instruction_path"] == sample_audio_paths["instruction"]

    def test_judge_audio_pointwise(self, mock_audiojudge, sample_audio_paths):
        """Test the judge_audio_pointwise functionality."""
        result = mock_audiojudge.judge_audio_pointwise(
            audio_path=sample_audio_paths["audio1"],
            system_prompt="Rate this audio clip.",
            model="gpt-4o-audio-preview",
        )

        assert result["success"] is True
        assert result["response"] == "Mocked response"
        assert result["model"] == "gpt-4o-audio-preview"
        assert result["audio_path"] == sample_audio_paths["audio1"]

    def test_error_handling(self, mock_audiojudge, sample_audio_paths):
        """Test error handling for non-existent files."""
        result = mock_audiojudge.judge_audio(
            audio1_path="nonexistent_file.wav",
            audio2_path=sample_audio_paths["audio2"],
            system_prompt="Compare these two audio clips.",
            model="gpt-4o-audio-preview",
        )

        assert result["success"] is False
        assert "error" in result
