
"""
AudioJudge: A simple package for audio comparison using large language models.

This package provides an easy-to-use interface for comparing audio files
using large language models with optional in-context learning examples.
"""

from .core import AudioJudge
from .utils import AudioExample

__version__ = "0.1.0"
__author__ = "Woody Gan"
__email__ = "woodygan@usc.edu"

__all__ = [
    "AudioJudge",
    "AudioExample", 
    "judge_audio_simple",
    "__version__"
]
