import pytest
import os
from pathlib import Path

from audiojudge import AudioJudge
from audiojudge.utils import AudioExample, AudioExamplePointwise


# Skip all tests in this file if API keys are not available
pytestmark = pytest.mark.skipif(
    not (os.environ.get("OPENAI_API_KEY") or os.environ.get("GOOGLE_API_KEY")),
    reason="API keys not available",
)


class TestIntegration:
    """Integration tests that make actual API calls.

    These tests are skipped if API keys are not available.
    """

    @pytest.fixture
    def judge(self):
        """Create an AudioJudge instance for testing."""
        return AudioJudge(
            temp_dir="temp_audio_test",
            signal_folder="signal_audios_test",
            cache_dir=".test_cache",
            disable_cache=False,
        )

    @pytest.fixture(autouse=True)
    def cleanup(self):
        """Clean up temporary directories after tests."""
        yield
        import shutil

        for dir_path in ["temp_audio_test", "signal_audios_test", ".test_cache"]:
            if os.path.exists(dir_path):
                shutil.rmtree(dir_path)

    @pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY"), reason="OpenAI API key not available"
    )
    def test_openai_integration(self, judge, sample_audio_paths):
        """Test integration with OpenAI API."""
        result = judge.judge_audio(
            audio1_path=sample_audio_paths["audio1"],
            audio2_path=sample_audio_paths["audio2"],
            system_prompt="Compare these two audio clips briefly.",
            model="gpt-4o-audio-preview",
            max_tokens=100,
        )

        assert result["success"] is True
        assert isinstance(result["response"], str)
        assert len(result["response"]) > 0

    @pytest.mark.skipif(
        not os.environ.get("GOOGLE_API_KEY"), reason="Google API key not available"
    )
    def test_gemini_integration(self, judge, sample_audio_paths):
        """Test integration with Google Gemini API."""
        result = judge.judge_audio(
            audio1_path=sample_audio_paths["audio1"],
            audio2_path=sample_audio_paths["audio2"],
            system_prompt="Compare these two audio clips briefly.",
            model="gemini-1.5-flash",
            max_tokens=100,
        )

        assert result["success"] is True
        assert isinstance(result["response"], str)
        assert len(result["response"]) > 0

    @pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY"), reason="OpenAI API key not available"
    )
    def test_pointwise_integration(self, judge, sample_audio_paths):
        """Test pointwise evaluation integration."""
        result = judge.judge_audio_pointwise(
            audio_path=sample_audio_paths["audio"],
            system_prompt="Rate this audio clip from 1-10.",
            model="gpt-4o-audio-preview",
            max_tokens=100,
        )

        assert result["success"] is True
        assert isinstance(result["response"], str)
        assert len(result["response"]) > 0

    @pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY"), reason="OpenAI API key not available"
    )
    def test_cache_functionality(self, judge, sample_audio_paths):
        """Test that caching works in integration."""
        # First call should hit the API
        result1 = judge.judge_audio(
            audio1_path=sample_audio_paths["audio1"],
            audio2_path=sample_audio_paths["audio2"],
            system_prompt="Compare these two audio clips briefly.",
            model="gpt-4o-audio-preview",
            max_tokens=100,
        )

        # Second call with same parameters should use cache
        result2 = judge.judge_audio(
            audio1_path=sample_audio_paths["audio1"],
            audio2_path=sample_audio_paths["audio2"],
            system_prompt="Compare these two audio clips briefly.",
            model="gpt-4o-audio-preview",
            max_tokens=100,
        )

        # Results should be identical
        assert result1["response"] == result2["response"]

        # Get cache stats
        stats = judge.get_cache_stats()
        assert stats["total_entries"] > 0
