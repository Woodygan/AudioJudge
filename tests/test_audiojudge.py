import pytest
from audiojudge import AudioJudge
import os
from pathlib import Path


class TestBasicFunctionality:
    """Basic functionality tests for AudioJudge."""

    @pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY"), reason="OpenAI API key not available"
    )
    def test_no_concatenation(self, sample_audio_paths):
        """Test judge_audio with no_concatenation method."""
        judge = AudioJudge()

        result = judge.judge_audio(
            audio1_path=sample_audio_paths["audio1"],
            audio2_path=sample_audio_paths["audio2"],
            system_prompt="Compare these two audio clips for quality.",
            model="gpt-4o-audio-preview",
            concatenation_method="no_concatenation",
        )

        assert result["success"] is True
        assert isinstance(result["response"], str)
        assert len(result["response"]) > 0
        assert result["concatenation_method"] == "no_concatenation"

    @pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY"), reason="OpenAI API key not available"
    )
    def test_examples_and_test_concatenation(self, sample_audio_paths):
        """Test judge_audio with examples_and_test_concatenation method."""
        judge = AudioJudge()

        result = judge.judge_audio(
            audio1_path=sample_audio_paths["audio1"],
            audio2_path=sample_audio_paths["audio2"],
            system_prompt="Compare these two audio clips for quality.",
            model="gpt-4o-audio-preview",
            concatenation_method="examples_and_test_concatenation",
        )
        print("--------------------------------")
        print(result)
        print("--------------------------------")

        assert result["success"] is True
        assert isinstance(result["response"], str)
        assert len(result["response"]) > 0
        assert result["concatenation_method"] == "examples_and_test_concatenation"
