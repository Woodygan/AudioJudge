import json
import base64
import os
import time
from pathlib import Path
import wave
import pandas as pd
import numpy as np
from openai import OpenAI
from tqdm import tqdm
import json
from typing import Counter, Dict, List, Tuple, Any, Optional
import re
from api_cache import api_cache
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import seaborn as sns
from pydub import AudioSegment

load_dotenv()
import audioop

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


def get_transcript(
    audio_path: str, transcript_type: str, dataset_name
) -> Optional[str]:
    """
    Get transcript for the audio based on the transcript type.

    Parameters:
    - audio_path: Path to the audio file
    - transcript_type: Type of transcription ('groundtruth' or 'asr')

    Returns:
    - The transcript or None if not available
    """
    if transcript_type == "groundtruth":
        # Get transcripts from the dataset
        dataset_path = f"datasets/{dataset_name}_dataset.json"
        with open(dataset_path, "r") as f:
            data = json.load(f)

        # Find the matching entry in the dataset
        for item in data:
            if item["audio1_path"] == audio_path:
                return item.get("transcript1")
            elif item["audio2_path"] == audio_path:
                return item.get("transcript2")
            elif item.get("instruction_path", "") == audio_path:
                return item.get("instruction_transcript")

        raise ValueError(f"Audio file {audio_path} not found in dataset.")

    elif transcript_type == "asr":
        return get_asr_transcription(audio_path)

    return None


@api_cache
def get_asr_transcription(audio_path: str) -> str:
    audio_file = open(audio_path, "rb")
    transcription = client.audio.transcriptions.create(
        model="gpt-4o-transcribe", file=audio_file
    )
    return transcription.text
