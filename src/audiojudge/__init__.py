
"""
AudioJudge: A simple package for audio comparison using large audio models.

This package provides an easy-to-use interface for comparing audio files
using large audio models with optional in-context learning examples.
"""

from .core import AudioJudge
from .utils import AudioExample, AudioExamplePointwise

__version__ = "0.1.0"
__author__ = "Woody Gan"
__email__ = "woodygan@usc.edu"

__all__ = [
    "AudioJudge",
    "AudioExample", 
    "AudioExamplePointwise",
    "__version__"
]
