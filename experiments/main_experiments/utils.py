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
from system_prompts import SYSTEM_PROMPTS
from utils_gemini import get_prompt_gemini
from helper import get_transcript, get_asr_transcription

load_dotenv()
import audioop

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

import google.generativeai as genai

genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))


def concatenate_audio_files(
    audio_paths: List[str],
    output_path: str,
    add_signals: bool = True,
    signal_folder: str = "signal_audios",
    is_test: bool = False,
    idx: int = 0,
) -> str:
    """
    Concatenate multiple audio files into a single file using Base64 encoding

    Parameters:
    - audio_paths: List of paths to audio files
    - output_path: Path to save the concatenated audio
    - add_signals: Whether to add spoken signals between audio clips
    - signal_folder: Folder to store signal audio files for reuse
    - is_test: Whether this is a test audio (True) or example audio (False)

    Returns:
    - Path to the concatenated audio file
    """
    # Create signal folder if it doesn't exist
    if add_signals:
        os.makedirs(signal_folder, exist_ok=True)

    # First, determine the target sample rate
    # Open all files and get their sample rates to pick the most common one
    sample_rates = []
    for audio_path in audio_paths:
        try:
            with wave.open(audio_path, "rb") as w:
                sample_rates.append(w.getframerate())
        except Exception as e:
            print(f"Error reading {audio_path}: {e}")

    if sample_rates:
        from collections import Counter

        target_sample_rate = Counter(sample_rates).most_common(1)[0][0]
    else:
        target_sample_rate = 24000
    if idx != 0 and len(audio_paths) != 2:
        raise ValueError("idx setting only work for two audio files")
    print(f"Using target sample rate: {target_sample_rate} Hz")

    # Now open the first file to get other parameters
    with wave.open(audio_paths[0], "rb") as first_file:
        params = first_file.getparams()
        # Update the framerate in params to our target rate
        params = params._replace(framerate=target_sample_rate)
        nchannels = params.nchannels
        sampwidth = params.sampwidth

    # Dictionary to store generated signal files (already resampled)
    signal_segments = {}

    # Generate all required signal files if needed
    if add_signals:
        # Generate signals with exact specified format
        required_signals = []

        if is_test:
            required_signals.append(("Test", "test.wav"))
        else:
            # Generate Example X signals
            for i in range((len(audio_paths) + 1) // 2):
                if idx == 0:
                    required_signals.append(
                        (f"Example {i + 1}", f"example_{i + 1}.wav")
                    )
                else:
                    required_signals.append((f"Example {idx}", f"example_{idx}.wav"))

        # Generate Audio 1/Audio 2 signals
        required_signals.append(("Audio 1", "audio_1.wav"))
        required_signals.append(("Audio 2", "audio_2.wav"))

        # Create all signal files
        for signal_text, signal_filename in required_signals:
            signal_path = os.path.join(signal_folder, signal_filename)
            resampled_path = os.path.join(signal_folder, f"resampled_{signal_filename}")

            # Create the signal if it doesn't exist
            if not os.path.exists(signal_path):
                with client.audio.speech.with_streaming_response.create(
                    model="gpt-4o-mini-tts",
                    voice="coral",
                    input=signal_text,
                    instructions="Speak in a clear and instructive tone",
                    response_format="wav",
                ) as response:
                    response.stream_to_file(signal_path)

            # Resample the signal to match target sample rate
            with wave.open(signal_path, "rb") as w:
                signal_rate = w.getframerate()
                signal_frames = w.readframes(w.getnframes())
                signal_channels = w.getnchannels()
                signal_width = w.getsampwidth()

                # Resample if needed
                if signal_rate != target_sample_rate:
                    signal_frames, _ = audioop.ratecv(
                        signal_frames,
                        signal_width,
                        signal_channels,
                        signal_rate,
                        target_sample_rate,
                        None,
                    )

                    # Save resampled version for debugging if needed
                    with wave.open(resampled_path, "wb") as out:
                        out.setparams(
                            (
                                signal_channels,
                                signal_width,
                                target_sample_rate,
                                0,
                                "NONE",
                                "not compressed",
                            )
                        )
                        out.writeframes(signal_frames)

                # Store the resampled signal
                signal_segments[signal_filename] = signal_frames

    # Create combined audio data
    final_data = bytearray()

    # Create a new WAV file for the output
    with wave.open(output_path, "wb") as output_file:
        output_file.setparams(params)

        # Process pairs of audio files (example by example)
        for i in range(0, len(audio_paths), 2):
            # Add signals if needed
            if add_signals:
                # Add Example X or Test signal
                if is_test:
                    signal_filename = "test.wav"
                else:
                    if idx == 0:
                        signal_filename = f"example_{(i // 2) + 1}.wav"
                    else:
                        signal_filename = f"example_{idx}.wav"

                # Add the signal
                output_file.writeframes(signal_segments[signal_filename])

                # Add silence (0.5 seconds)
                silence_frames = b"\x00" * (
                    int(0.5 * target_sample_rate) * sampwidth * nchannels
                )
                output_file.writeframes(silence_frames)

                # Add "Audio 1" signal
                output_file.writeframes(signal_segments["audio_1.wav"])

                # Add silence
                output_file.writeframes(silence_frames)

            # Add first audio file of the pair (resampled if needed)
            with wave.open(audio_paths[i], "rb") as w:
                audio_rate = w.getframerate()
                audio_frames = w.readframes(w.getnframes())
                audio_channels = w.getnchannels()
                audio_width = w.getsampwidth()

                # Resample if needed
                if audio_rate != target_sample_rate:
                    audio_frames, _ = audioop.ratecv(
                        audio_frames,
                        audio_width,
                        audio_channels,
                        audio_rate,
                        target_sample_rate,
                        None,
                    )

                # Add the audio data
                output_file.writeframes(audio_frames)

            # Add silence
            silence_frames = b"\x00" * (
                int(0.5 * target_sample_rate) * sampwidth * nchannels
            )
            output_file.writeframes(silence_frames)

            # Check if there's a second file in this pair
            if i + 1 < len(audio_paths):
                # Add "Audio 2" signal if signals are enabled
                if add_signals:
                    output_file.writeframes(signal_segments["audio_2.wav"])
                    output_file.writeframes(silence_frames)

                # Add second audio file (resampled if needed)
                with wave.open(audio_paths[i + 1], "rb") as w:
                    audio_rate = w.getframerate()
                    audio_frames = w.readframes(w.getnframes())
                    audio_channels = w.getnchannels()
                    audio_width = w.getsampwidth()

                    if audio_rate != target_sample_rate:
                        audio_frames, _ = audioop.ratecv(
                            audio_frames,
                            audio_width,
                            audio_channels,
                            audio_rate,
                            target_sample_rate,
                            None,
                        )

                    # Add the audio data
                    output_file.writeframes(audio_frames)

                # Add silence
                output_file.writeframes(silence_frames)

    return output_path


def encode_audio_file(file_path: str) -> str:
    """
    Encode an audio file to base64
    """
    with open(file_path, "rb") as audio_file:
        encoded_string = base64.b64encode(audio_file.read()).decode("utf-8")
    return encoded_string


def get_prompt(
    prompt_type: str,
    audio1_path: str,
    audio2_path: str,
    dataset_name: str = "pronunciation",
    n_shots: int = 0,
    transcript_type: str = "none",
    concat_fewshot: bool = False,
    concat_test: bool = False,
    two_turns: bool = False,
    aggregate_fewshot: bool = False,
) -> List[Dict]:
    """
    Create a prompt for the API call using the selected prompt type and audio files
    with options for transcription and audio concatenation

    Parameters:
    - prompt_type: Type of prompt strategy to use ('no_cot', 'standard_cot', 'phonetic_cot', etc.)
    - audio1_path: Path to the first audio file
    - audio2_path: Path to the second audio file
    - dataset_name: Name of the dataset ("pronunciation", "speaker", or "speed")
    - n_shots: Number of examples to include (0-8). 0 means no few-shot examples.
    - transcript_type: Type of transcription to use ('none', 'groundtruth', 'asr')
    - concat_fewshot: Whether to concatenate few-shot example audio files (each datapoint in one file)
    - concat_test: Whether to concatenate test audio files
    - two_turns: Whether to send each audio file in a separate message turn (overrides concat_test if True)
    - aggregate_fewshot: Whether to aggregate all few-shot examples into a single audio file (original concat)

    Returns:
    - List of message dictionaries for the API call
    """
    system_prompts = SYSTEM_PROMPTS.get(dataset_name)
    system_prompt = system_prompts.get(prompt_type)
    messages = [{"role": "system", "content": system_prompt}]
    user_message = ""
    # Add dataset-specific instructions
    if dataset_name == "pronunciation":
        if prompt_type != "no_cot":
            user_message = (
                "Please analyze these two recordings strictly for pronunciation details (phonemes, syllables, stress, emphasis). "
                "Ignore differences solely due to accent. Respond ONLY in text and output valid JSON with keys 'reasoning' and 'match' (boolean)."
            )
        else:
            user_message = (
                "Please analyze these two recordings strictly for pronunciation details (phonemes, syllables, stress, emphasis). "
                "Ignore differences solely due to accent. Respond ONLY in text and output valid JSON with key 'match' (boolean)."
            )
    elif dataset_name == "speaker":
        if prompt_type != "no_cot":
            user_message = (
                "Please analyze if these two recordings are from the same speaker. "
                "Respond ONLY in text and output valid JSON with keys 'reasoning' and 'match' (boolean)."
            )
        else:
            user_message = (
                "Please analyze if these two recordings are from the same speaker. "
                "Respond ONLY in text and output valid JSON with key 'match' (boolean)."
            )
    elif dataset_name == "speed":
        if prompt_type != "no_cot":
            user_message = (
                "Please analyze which of the two recordings has faster speech. "
                "Respond ONLY in text and output valid JSON with keys 'reasoning' and 'label' (string, either '1' or '2')."
            )
        else:
            user_message = (
                "Please analyze which of the two recordings has faster speech. "
                "Respond ONLY in text and output valid JSON with key 'label' (string, either '1' or '2')."
            )
    elif (
        dataset_name == "tmhintq"
        or dataset_name == "somos"
        or dataset_name == "thaimos"
    ):
        if prompt_type != "no_cot":
            user_message = (
                "Please analyze which of the two recordings is better (has better speech quality). "
                "Respond ONLY in text and output valid JSON with keys 'reasoning' and 'label' (string, either '1' or '2')."
            )
        else:
            user_message = (
                "Please analyze which of the two recordings is better (has better speech quality). "
                "Respond ONLY in text and output valid JSON with key 'label' (string, either '1' or '2')."
            )

    # Load dataset-specific few-shot examples
    with open("few_shots_examples.json", "r") as f:
        few_shots_examples = json.load(f)

    # Check if examples exist for the specified dataset
    if dataset_name not in few_shots_examples:
        raise ValueError(f"No few-shot examples found for dataset: {dataset_name}")
    else:
        available_examples = few_shots_examples[dataset_name]

    # If few-shot examples are requested, add them before the user query
    if n_shots > 0 and available_examples:
        n_shots = min(n_shots, len(available_examples))
        if two_turns:
            # Handle two-turn examples
            for i in range(n_shots):
                example = available_examples[i]

                # First audio turn
                example_audio1 = encode_audio_file(example["audio1_path"])
                first_content = [
                    {"type": "text", "text": f"Here is the first audio clip:"},
                    {
                        "type": "input_audio",
                        "input_audio": {"data": example_audio1, "format": "wav"},
                    },
                ]

                if transcript_type != "none":
                    transcript1_key = (
                        "transcript1" if "transcript1" in example else "word"
                    )
                    first_content.append(
                        {
                            "type": "text",
                            "text": f'Transcript for this audio: "{example[transcript1_key]}"',
                        }
                    )

                messages.append({"role": "user", "content": first_content})

                # Assistant acknowledgement
                messages.append(
                    {
                        "role": "assistant",
                        "content": f"I've heard the first audio clip.",
                    }
                )

                # Second audio turn
                example_audio2 = encode_audio_file(example["audio2_path"])
                second_content = [
                    {"type": "text", "text": f"Here is the second audio clip:"},
                    {
                        "type": "input_audio",
                        "input_audio": {"data": example_audio2, "format": "wav"},
                    },
                ]

                if transcript_type != "none":
                    transcript2_key = (
                        "transcript2" if "transcript2" in example else "word"
                    )
                    second_content.append(
                        {
                            "type": "text",
                            "text": f'Transcript for this audio: "{example[transcript2_key]}"',
                        }
                    )

                second_content.append({"type": "text", "text": user_message})

                messages.append({"role": "user", "content": second_content})

                # Add assistant response
                assistant_response = json.dumps(example["assistant_message"])
                messages.append({"role": "assistant", "content": assistant_response})

        elif aggregate_fewshot:
            all_example_audio_paths = []
            examples_data = []

            for i in range(n_shots):
                example = available_examples[i]
                all_example_audio_paths.append(example["audio1_path"])
                all_example_audio_paths.append(example["audio2_path"])

                # Extract transcript keys based on dataset
                if dataset_name == "pronunciation":
                    transcript1 = example.get("transcript1", example.get("word"))
                    transcript2 = example.get("transcript2", example.get("word"))
                    example_result = example["assistant_message"].get("match")
                else:
                    transcript1 = example.get("transcript1", "")
                    transcript2 = example.get("transcript2", "")
                    if (
                        dataset_name == "speed"
                        or dataset_name == "tmhintq"
                        or dataset_name == "somos"
                        or dataset_name == "thaimos"
                    ):
                        example_result = example["assistant_message"].get("label")
                    else:  # speaker dataset
                        example_result = example["assistant_message"].get("match")

                examples_data.append(
                    {
                        "transcript1": transcript1,
                        "transcript2": transcript2,
                        "result": example_result,
                    }
                )

            os.makedirs("temp_audio", exist_ok=True)

            concat_examples_path = os.path.join(
                "temp_audio", f"concat_examples_{time.time()}.wav"
            )
            concatenate_audio_files(
                all_example_audio_paths, concat_examples_path, add_signals=True
            )

            # Encode concatenated examples audio
            examples_encoded = encode_audio_file(concat_examples_path)

            # Create content for examples
            examples_content = [
                {"type": "text", "text": "Here are some examples for reference:"},
                {
                    "type": "input_audio",
                    "input_audio": {"data": examples_encoded, "format": "wav"},
                },
            ]

            # Add examples data
            example_text = "Examples information:\n"
            for i, example in enumerate(examples_data):
                example_text += f"Example {i + 1}:\n"
                if transcript_type != "none" or dataset_name == "pronunciation":
                    example_text += (
                        f'- First audio transcript: "{example["transcript1"]}"\n'
                    )
                    example_text += (
                        f'- Second audio transcript: "{example["transcript2"]}"\n'
                    )

                # Output depends on dataset type
                if (
                    dataset_name == "speed"
                    or dataset_name == "tmhintq"
                    or dataset_name == "somos"
                    or dataset_name == "thaimos"
                ):
                    example_text += f"- Label: {json.dumps(example['result'])}\n\n"
                else:  # pronunciation or speaker
                    example_text += f"- Match: {json.dumps(example['result'])}\n\n"

            examples_content.append({"type": "text", "text": example_text})
            # concatenate 1 example in one turn
            messages.append({"role": "user", "content": examples_content})

            messages.append(
                {
                    "role": "assistant",
                    "content": "I understand these examples. I'll apply this understanding to analyze the new audio clips you provide.",
                }
            )
            # Clean up the temporary file
            os.remove(concat_examples_path)

        elif concat_fewshot:
            # New behavior: concatenate each datapoint's audio files (2 audios in 1 file per example)
            for i in range(n_shots):
                example = available_examples[i]

                # Create temp directory if it doesn't exist
                os.makedirs("temp_audio", exist_ok=True)

                # Concatenate the two audio files for this example into one file
                concat_example_path = os.path.join(
                    "temp_audio", f"concat_example_{i + 1}_{time.time()}.wav"
                )
                concatenate_audio_files(
                    [example["audio1_path"], example["audio2_path"]],
                    concat_example_path,
                    add_signals=True,
                    is_test=False,
                    idx=i + 1,
                )

                # Encode the concatenated example
                example_encoded = encode_audio_file(concat_example_path)

                content = [
                    {"type": "text", "text": f"Please analyze these audio clips:"},
                    {
                        "type": "input_audio",
                        "input_audio": {"data": example_encoded, "format": "wav"},
                    },
                ]

                if transcript_type != "none":
                    # Get transcript keys based on dataset
                    if dataset_name == "pronunciation":
                        transcript1 = example.get("transcript1", example.get("word"))
                        transcript2 = example.get("transcript2", example.get("word"))
                    else:
                        transcript1 = example.get("transcript1", "")
                        transcript2 = example.get("transcript2", "")

                    content.append(
                        {
                            "type": "text",
                            "text": f'Transcript for first audio: "{transcript1}"',
                        }
                    )
                    content.append(
                        {
                            "type": "text",
                            "text": f'Transcript for second audio: "{transcript2}"',
                        }
                    )

                content.append({"type": "text", "text": user_message})

                messages.append({"role": "user", "content": content})

                # Add assistant response
                assistant_response = json.dumps(example["assistant_message"])
                messages.append({"role": "assistant", "content": assistant_response})

                # Clean up the temporary file
                os.remove(concat_example_path)

        else:
            # Original separate behavior
            for i in range(n_shots):
                example = available_examples[i]
                example_audio1 = encode_audio_file(example["audio1_path"])
                example_audio2 = encode_audio_file(example["audio2_path"])

                # Get transcript info based on dataset
                if dataset_name == "pronunciation":
                    transcript1 = example.get("transcript1", example.get("word"))
                    transcript2 = example.get("transcript2", example.get("word"))
                else:
                    transcript1 = example.get("transcript1", "")
                    transcript2 = example.get("transcript2", "")

                content = [
                    {"type": "text", "text": f"Here is the first audio clip:"},
                    {
                        "type": "input_audio",
                        "input_audio": {"data": example_audio1, "format": "wav"},
                    },
                    {"type": "text", "text": f"Here is the second audio clip:"},
                    {
                        "type": "input_audio",
                        "input_audio": {"data": example_audio2, "format": "wav"},
                    },
                ]
                if transcript_type != "none":
                    content.append(
                        {
                            "type": "text",
                            "text": f'Transcript for first audio: "{transcript1}"',
                        }
                    )
                    content.append(
                        {
                            "type": "text",
                            "text": f'Transcript for second audio: "{transcript2}"',
                        }
                    )

                content.append({"type": "text", "text": user_message})
                messages.append({"role": "user", "content": content})

                # Add assistant response
                assistant_response = json.dumps(example["assistant_message"])
                messages.append({"role": "assistant", "content": assistant_response})

    # Handle the test audio files
    if two_turns:
        # Two turns for test audios - this completely overrides concat_test
        # First audio turn
        audio1_encoded = encode_audio_file(audio1_path)
        first_content = [
            {"type": "text", "text": "Here is the first test audio clip:"},
            {
                "type": "input_audio",
                "input_audio": {"data": audio1_encoded, "format": "wav"},
            },
        ]

        if transcript_type == "groundtruth" or transcript_type == "asr":
            # Get transcript for the first audio
            transcript1 = get_transcript(audio1_path, transcript_type, dataset_name)
            if transcript1:
                first_content.append(
                    {
                        "type": "text",
                        "text": f'Transcript for this audio: "{transcript1}"',
                    }
                )

        messages.append({"role": "user", "content": first_content})

        # Assistant acknowledgement
        messages.append(
            {"role": "assistant", "content": "I've heard the first test audio clip."}
        )

        # Second audio turn
        audio2_encoded = encode_audio_file(audio2_path)
        second_content = [
            {"type": "text", "text": "Here is the second test audio clip:"},
            {
                "type": "input_audio",
                "input_audio": {"data": audio2_encoded, "format": "wav"},
            },
        ]

        if transcript_type == "groundtruth" or transcript_type == "asr":
            # Get transcript for the second audio
            transcript2 = get_transcript(audio2_path, transcript_type, dataset_name)
            if transcript2:
                second_content.append(
                    {
                        "type": "text",
                        "text": f'Transcript for this audio: "{transcript2}"',
                    }
                )

        second_content.append({"type": "text", "text": user_message})

        messages.append({"role": "user", "content": second_content})
    else:
        if concat_test:
            os.makedirs("temp_audio", exist_ok=True)

            concat_test_path = os.path.join(
                "temp_audio", f"concat_test_{time.time()}.wav"
            )
            concatenate_audio_files(
                [audio1_path, audio2_path],
                concat_test_path,
                add_signals=True,
                is_test=True,
            )

            test_encoded = encode_audio_file(concat_test_path)

            user_content = [
                {"type": "text", "text": "Please analyze these audio clips:"},
                {
                    "type": "input_audio",
                    "input_audio": {"data": test_encoded, "format": "wav"},
                },
            ]
            os.remove(concat_test_path)
        else:
            audio1_encoded = encode_audio_file(audio1_path)
            audio2_encoded = encode_audio_file(audio2_path)

            user_content = [
                {"type": "text", "text": "Here is the first audio clip:"},
                {
                    "type": "input_audio",
                    "input_audio": {"data": audio1_encoded, "format": "wav"},
                },
                {"type": "text", "text": "Here is the second audio clip:"},
                {
                    "type": "input_audio",
                    "input_audio": {"data": audio2_encoded, "format": "wav"},
                },
            ]

        if transcript_type == "groundtruth":
            transcript1 = get_transcript(audio1_path, transcript_type, dataset_name)
            transcript2 = get_transcript(audio2_path, transcript_type, dataset_name)

            if transcript1 and transcript2:
                user_content.append(
                    {
                        "type": "text",
                        "text": f'Transcript for first audio: "{transcript1}"',
                    }
                )
                user_content.append(
                    {
                        "type": "text",
                        "text": f'Transcript for second audio: "{transcript2}"',
                    }
                )

        elif transcript_type == "asr":
            transcript1 = get_asr_transcription(audio1_path)
            print(f"Transcription for first audio: {transcript1}")
            transcript2 = get_asr_transcription(audio2_path)

            user_content.append(
                {"type": "text", "text": f'Transcript for first audio: "{transcript1}"'}
            )
            user_content.append(
                {
                    "type": "text",
                    "text": f'Transcript for second audio: "{transcript2}"',
                }
            )

        user_content.append({"type": "text", "text": user_message})

        messages.append({"role": "user", "content": user_content})

    return messages


def evaluate_prompt_strategy(
    df: pd.DataFrame,
    prompt_type: str,
    model: str,
    dataset_name: str = "pronunciation",
    n_samples: int = 100,
    n_shots: int = 0,
    result_dir: str = "results",
    majority_vote: bool = False,
    vote_prompt_types: List[str] = None,
    transcript_type: str = "none",
    concat_fewshot: bool = False,
    concat_test: bool = False,
    two_turns: bool = False,
    aggregate_fewshot: bool = False,
) -> Dict:
    """
    Evaluate a prompt strategy on the dataset with options for transcription and audio concatenation

    Parameters:
    - df: DataFrame containing the dataset
    - prompt_type: Type of prompt to use (e.g., 'vanilla', 'standard_cot', 'phonetic_cot')
    - model: Model to use for evaluation
    - dataset_name: Name of the dataset ('pronunciation', 'speaker', or 'speed')
    - n_samples: Number of samples to evaluate
    - n_shots: Number of examples to use for few-shot prompting (0 means no examples)
    - result_dir: Directory to save results
    - majority_vote: Whether to use majority voting across different prompt types
    - vote_prompt_types: List of prompt types to use for majority voting
    - transcript_type: Type of transcription to use ('none', 'groundtruth', 'asr')
    - concat_fewshot: Whether to concatenate few-shot example audio files (each datapoint in one file)
    - concat_test: Whether to concatenate test audio files
    - two_turns: Whether to send each audio file in a separate message turn(both for fewshot and test)
    - aggregate_fewshot: Whether to aggregate all few-shot examples into a single audio file (original concat)
    """
    try:
        # If majority voting is enabled, use the majority vote evaluation function
        if majority_vote and vote_prompt_types:
            return evaluate_with_majority_vote(
                df,
                vote_prompt_types,
                model,
                dataset_name,
                n_samples,
                n_shots,
                result_dir,
                transcript_type,
                concat_fewshot,
                concat_test,
                two_turns,
                aggregate_fewshot,
            )

        results = []

        # Take a sample of the dataset
        sample_df = df.head(min(n_samples, len(df)))

        for _, row in tqdm(
            sample_df.iterrows(),
            total=len(sample_df),
            desc=f"Evaluating {prompt_type} with {n_shots} shots, {transcript_type} transcript, "
            f"fewshot_aggregate={aggregate_fewshot}, fewshot_concat={concat_fewshot}, "
            f"test_concat={concat_test}, two_turns={two_turns}",
        ):
            audio1_path = row["audio1_path"]
            audio2_path = row["audio2_path"]

            # Extract ground truth based on dataset type
            if (
                dataset_name == "speed"
                or dataset_name == "tmhintq"
                or dataset_name == "somos"
                or dataset_name == "thaimos"
            ):
                ground_truth = row.get("label")
            else:  # pronunciation or speaker
                ground_truth = row.get("match")
            if "gpt" in model:
                messages = get_prompt(
                    prompt_type=prompt_type,
                    audio1_path=audio1_path,
                    audio2_path=audio2_path,
                    dataset_name=dataset_name,
                    n_shots=n_shots,
                    transcript_type=transcript_type,
                    concat_fewshot=concat_fewshot,
                    concat_test=concat_test,
                    two_turns=two_turns,
                    aggregate_fewshot=aggregate_fewshot,
                )
            elif "gemini" in model:
                messages = get_prompt_gemini(
                    prompt_type=prompt_type,
                    audio1_path=audio1_path,
                    audio2_path=audio2_path,
                    dataset_name=dataset_name,
                    n_shots=n_shots,
                    transcript_type=transcript_type,
                    concat_fewshot=concat_fewshot,
                    concat_test=concat_test,
                    two_turns=two_turns,
                    aggregate_fewshot=aggregate_fewshot,
                )
            response_data = get_model_response(model, messages)

            if response_data is None:
                print(f"Failed to get response for {audio1_path} and {audio2_path}")
                continue

            _, prediction_text = response_data
            prediction_json = extract_json_from_response(prediction_text)

            if prediction_json is None:
                print(
                    f"Failed to extract JSON from response for {audio1_path} and {audio2_path}"
                )
                continue

            # Extract the prediction based on dataset type
            if (
                dataset_name == "speed"
                or dataset_name == "tmhintq"
                or dataset_name == "somos"
                or dataset_name == "thaimos"
            ):
                prediction = prediction_json.get("label", None)
            else:  # pronunciation or speaker
                prediction = prediction_json.get("match", None)

            reasoning = prediction_json.get("reasoning", "")

            if prediction is None:
                print(
                    f"No prediction field in extracted JSON for {audio1_path} and {audio2_path}"
                )
                continue

            results.append(
                {
                    "audio1_path": audio1_path,
                    "audio2_path": audio2_path,
                    "ground_truth": ground_truth,
                    "prediction": prediction,
                    "reasoning": reasoning,
                    "correct": ground_truth == prediction,
                    "prompt_type": prompt_type,
                    "dataset_name": dataset_name,
                    "n_shots": n_shots,
                    "transcript_type": transcript_type,
                    "concat_fewshot": concat_fewshot,
                    "concat_test": concat_test,
                    "two_turns": two_turns,
                    "aggregate_fewshot": aggregate_fewshot,
                }
            )

        # Create a meaningful prompt_type name for output
        output_prompt_type = prompt_type

        # Calculate metrics
        results_df = pd.DataFrame(results)

        if len(results_df) == 0:
            return {
                "prompt_type": output_prompt_type,
                "model": model,
                "dataset_name": dataset_name,
                "n_shots": n_shots,
                "transcript_type": transcript_type,
                "concat_fewshot": concat_fewshot,
                "concat_test": concat_test,
                "two_turns": two_turns,
                "aggregate_fewshot": aggregate_fewshot,
                "accuracy": 0,
                "false_positive_rate": 0,
                "false_negative_rate": 0,
                "num_samples": 0,
            }
        accuracy = results_df["correct"].mean()
        # Calculate metrics based on dataset type
        if (
            dataset_name == "speed"
            or dataset_name == "tmhintq"
            or dataset_name == "somos"
            or dataset_name == "thaimos"
        ):
            # For speed dataset, calculate accuracy only
            metric_summary = {"accuracy": accuracy, "num_samples": len(results_df)}
        else:  # pronunciation or speaker
            # Calculate false positive rate and false negative rate for binary classification tasks
            true_negatives = results_df[
                (results_df["ground_truth"] == False)
                & (results_df["prediction"] == False)
            ]
            false_positives = results_df[
                (results_df["ground_truth"] == False)
                & (results_df["prediction"] == True)
            ]
            true_positives = results_df[
                (results_df["ground_truth"] == True)
                & (results_df["prediction"] == True)
            ]
            false_negatives = results_df[
                (results_df["ground_truth"] == True)
                & (results_df["prediction"] == False)
            ]

            false_positive_rate = (
                len(false_positives) / (len(false_positives) + len(true_negatives))
                if (len(false_positives) + len(true_negatives)) > 0
                else 0
            )
            false_negative_rate = (
                len(false_negatives) / (len(false_negatives) + len(true_positives))
                if (len(false_negatives) + len(true_positives)) > 0
                else 0
            )
            unbalanced_accuracy = accuracy
            accuracy = 1 - (false_negative_rate + false_positive_rate) / 2

            metric_summary = {
                "accuracy": accuracy,
                "false_positive_rate": false_positive_rate,
                "false_negative_rate": false_negative_rate,
                "num_samples": len(results_df),
                "unbalanced_accuracy": unbalanced_accuracy,
            }

        # Save detailed results
        os.makedirs(result_dir, exist_ok=True)

        # Create a more descriptive filename that includes all configurations
        config_desc = (
            f"{dataset_name}_{output_prompt_type}_{n_shots}_shots_{transcript_type}_transcript_"
            f"{'two_turns' if two_turns else 'single_turn'}_"
            f"fewshot_{'aggregate' if aggregate_fewshot else ('concat' if concat_fewshot else 'separate')}_"
            f"test_{'concat' if concat_test else 'separate'}_"
            f"{model.replace('-', '_')}.csv"
        )

        results_output_path = os.path.join(result_dir, config_desc)
        results_df.to_csv(results_output_path, index=False)

        print(f"Results for {dataset_name} - {output_prompt_type}:")
        print(f"  Accuracy: {accuracy:.4f}")
        if (
            dataset_name != "speed"
            and dataset_name != "tmhintq"
            and dataset_name != "somos"
            and dataset_name != "thaimos"
        ):
            print(f"  False Positive Rate: {false_positive_rate:.4f}")
            print(f"  False Negative Rate: {false_negative_rate:.4f}")
        print(f"  Number of samples: {len(results_df)}")

        # Return results with appropriate metrics
        result_dict = {
            "prompt_type": output_prompt_type,
            "model": model,
            "dataset_name": dataset_name,
            "n_shots": n_shots,
            "transcript_type": transcript_type,
            "concat_fewshot": concat_fewshot,
            "concat_test": concat_test,
            "two_turns": two_turns,
            "aggregate_fewshot": aggregate_fewshot,
            **metric_summary,
        }

        return result_dict
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        return {
            "prompt_type": prompt_type,
            "model": model,
            "dataset_name": dataset_name,
            "n_shots": n_shots,
            "transcript_type": transcript_type,
            "concat_fewshot": concat_fewshot,
            "concat_test": concat_test,
            "two_turns": two_turns,
            "aggregate_fewshot": aggregate_fewshot,
            "accuracy": 0,
            "false_positive_rate": 0 if dataset_name != "speed" else None,
            "false_negative_rate": 0 if dataset_name != "speed" else None,
            "num_samples": 0,
            "error": str(e),
        }


def evaluate_with_majority_vote(
    df: pd.DataFrame,
    prompt_types: List[str],
    model: str,
    dataset_name: str = "pronunciation",
    n_samples: int = 100,
    n_shots: int = 0,
    result_dir: str = "results",
    transcript_type: str = "none",
    concat_fewshot: bool = False,
    concat_test: bool = False,
    two_turns: bool = False,
    aggregate_fewshot: bool = False,
) -> Dict:
    """
    Evaluate using majority voting across multiple prompt types for different datasets

    Parameters:
    - df: DataFrame containing the dataset
    - prompt_types: List of prompt types to use in voting
    - model: Model name to use for evaluation
    - dataset_name: Name of the dataset ('pronunciation', 'speaker', or 'speed')
    - n_samples: Number of samples to evaluate
    - n_shots: Number of examples to use for few-shot prompting
    - result_dir: Directory to save results
    - transcript_type: Type of transcription to use
    - concat_fewshot: Whether to concatenate few-shot example audio files
    - concat_test: Whether to concatenate test audio files
    - two_turns: Whether to send each audio file in a separate message turn
    - aggregate_fewshot: Whether to aggregate all few-shot examples into a single audio file

    Returns:
    - Dictionary containing evaluation metrics
    """
    # Sample the dataset
    sample_df = df.head(min(n_samples, len(df)))

    # Set the prediction key based on dataset type
    if (
        dataset_name == "speed"
        or dataset_name == "tmhintq"
        or dataset_name == "somos"
        or dataset_name == "thaimos"
    ):
        prediction_key = "label"
    else:  # pronunciation or speaker
        prediction_key = "match"

    # Dictionary to store predictions for each sample across prompt types
    all_predictions = {}

    # Loop through each prompt type
    for prompt_type in prompt_types:
        print(f"Running {prompt_type} for majority voting on {dataset_name} dataset...")

        # Process each sample with the current prompt type
        for _, row in tqdm(
            sample_df.iterrows(),
            total=len(sample_df),
            desc=f"Evaluating {dataset_name} with {prompt_type}",
        ):
            audio1_path = row["audio1_path"]
            audio2_path = row["audio2_path"]
            sample_id = f"{audio1_path}_{audio2_path}"

            # Extract ground truth based on dataset type
            if (
                dataset_name == "speed"
                or dataset_name == "tmhintq"
                or dataset_name == "somos"
                or dataset_name == "thaimos"
            ):
                ground_truth = row.get("label")
            else:  # pronunciation or speaker
                ground_truth = row.get("match")

            # Initialize prediction tracking for this sample if not already done
            if sample_id not in all_predictions:
                all_predictions[sample_id] = {
                    "audio1_path": audio1_path,
                    "audio2_path": audio2_path,
                    "ground_truth": ground_truth,
                    "votes": [],
                    "reasonings": {},
                }

            # Create the prompt for this sample
            if "gpt" in model:
                messages = get_prompt(
                    prompt_type=prompt_type,
                    audio1_path=audio1_path,
                    audio2_path=audio2_path,
                    dataset_name=dataset_name,
                    n_shots=n_shots,
                    transcript_type=transcript_type,
                    concat_fewshot=concat_fewshot,
                    concat_test=concat_test,
                    two_turns=two_turns,
                    aggregate_fewshot=aggregate_fewshot,
                )
            elif "gemini" in model:
                messages = get_prompt_gemini(
                    prompt_type=prompt_type,
                    audio1_path=audio1_path,
                    audio2_path=audio2_path,
                    dataset_name=dataset_name,
                    n_shots=n_shots,
                    transcript_type=transcript_type,
                    concat_fewshot=concat_fewshot,
                    concat_test=concat_test,
                    two_turns=two_turns,
                    aggregate_fewshot=aggregate_fewshot,
                )

            # Get model response
            response_data = get_model_response(model, messages)

            # Handle failed responses
            if response_data is None:
                print(
                    f"Failed to get response for {audio1_path} and {audio2_path} with {prompt_type}"
                )
                continue

            # Extract prediction from response
            _, prediction_text = response_data
            prediction_json = extract_json_from_response(prediction_text)

            if prediction_json is None:
                print(
                    f"Failed to extract JSON from response for {audio1_path} and {audio2_path} with {prompt_type}"
                )
                continue

            # Get the prediction and reasoning
            prediction = prediction_json.get(prediction_key, None)
            reasoning = prediction_json.get("reasoning", "")

            if prediction is None:
                print(
                    f"No '{prediction_key}' field in extracted JSON for {audio1_path} and {audio2_path} with {prompt_type}"
                )
                continue

            # Store prediction and reasoning
            all_predictions[sample_id]["votes"].append(prediction)
            all_predictions[sample_id]["reasonings"][prompt_type] = reasoning

    # Compile results with majority voting
    results = []
    from collections import Counter

    for sample_id, data in all_predictions.items():
        # Skip if no votes (all methods failed)
        if not data["votes"]:
            continue

        # Count votes and determine majority
        vote_counter = Counter(data["votes"])
        majority_prediction = vote_counter.most_common(1)[0][0]

        # Create result with majority vote
        result = {
            "audio1_path": data["audio1_path"],
            "audio2_path": data["audio2_path"],
            "ground_truth": data["ground_truth"],
            "prediction": majority_prediction,
            "vote_count": dict(vote_counter),
            "majority_size": vote_counter[majority_prediction],
            "total_votes": len(data["votes"]),
            "correct": data["ground_truth"] == majority_prediction,
            "prompt_type": "majority_vote",
            "dataset_name": dataset_name,
            "n_shots": n_shots,
            "transcript_type": transcript_type,
            "concat_fewshot": concat_fewshot,
            "concat_test": concat_test,
            "two_turns": two_turns,
            "aggregate_fewshot": aggregate_fewshot,
            "reasonings": data["reasonings"],
        }
        results.append(result)

    # Create results dataframe
    results_df = pd.DataFrame(results)

    # Create output name
    prompt_types_short = "_".join([p.replace("_cot", "") for p in prompt_types])
    if len(prompt_types_short) > 50:  # Truncate if too long
        prompt_types_short = prompt_types_short[:50] + "..."

    output_name = f"majority_vote_{prompt_types_short}"

    # Handle empty results
    if len(results_df) == 0:
        empty_result = {
            "prompt_type": output_name,
            "model": model,
            "dataset_name": dataset_name,
            "n_shots": n_shots,
            "transcript_type": transcript_type,
            "concat_fewshot": concat_fewshot,
            "concat_test": concat_test,
            "two_turns": two_turns,
            "aggregate_fewshot": aggregate_fewshot,
            "accuracy": 0,
            "num_samples": 0,
        }

        # Add dataset-specific metrics
        if (
            dataset_name != "speed"
            and dataset_name != "tmhintq"
            and dataset_name != "somos"
            and dataset_name != "thaimos"
        ):
            empty_result.update({"false_positive_rate": 0, "false_negative_rate": 0})

        return empty_result

    # Calculate basic metrics
    accuracy = results_df["correct"].mean()

    # Initialize metrics dictionary
    metrics = {"accuracy": accuracy, "num_samples": len(results_df)}

    # Calculate dataset-specific metrics
    if (
        dataset_name != "speed"
        and dataset_name != "tmhinqt"
        and dataset_name != "somos"
        or dataset_name == "thaimos"
    ):  # For binary classification datasets (pronunciation, speaker)
        # Calculate classification metrics
        true_negatives = results_df[
            (results_df["ground_truth"] == False) & (results_df["prediction"] == False)
        ]
        false_positives = results_df[
            (results_df["ground_truth"] == False) & (results_df["prediction"] == True)
        ]
        true_positives = results_df[
            (results_df["ground_truth"] == True) & (results_df["prediction"] == True)
        ]
        false_negatives = results_df[
            (results_df["ground_truth"] == True) & (results_df["prediction"] == False)
        ]

        # Calculate rates
        false_positive_rate = (
            len(false_positives) / (len(false_positives) + len(true_negatives))
            if (len(false_positives) + len(true_negatives)) > 0
            else 0
        )
        false_negative_rate = (
            len(false_negatives) / (len(false_negatives) + len(true_positives))
            if (len(false_negatives) + len(true_positives)) > 0
            else 0
        )

        balanced_accuracy = 1 - (false_negative_rate + false_positive_rate) / 2
        metrics["accuracy"] = balanced_accuracy

        # Add classification metrics
        metrics.update(
            {
                "false_positive_rate": false_positive_rate,
                "false_negative_rate": false_negative_rate,
            }
        )

    # Save detailed results
    os.makedirs(result_dir, exist_ok=True)
    results_filename = (
        f"{dataset_name}_{output_name}_{n_shots}_shots_{transcript_type}_transcript_"
        f"{'two_turns' if two_turns else 'single_turn'}_"
        f"fewshot_{'aggregate' if aggregate_fewshot else ('concat' if concat_fewshot else 'separate')}_"
        f"test_{'concat' if concat_test else 'separate'}_{model.replace('-', '_')}.csv"
    )
    results_output_path = os.path.join(result_dir, results_filename)
    results_df.to_csv(results_output_path, index=False)

    # Print summary
    print(
        f"\nResults for {dataset_name} - majority voting across {', '.join(prompt_types)}:"
    )
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    if (
        dataset_name != "speed"
        and dataset_name != "tmhintq"
        and dataset_name != "somos"
        or dataset_name == "thaimos"
    ):
        print(f"  False Positive Rate: {metrics['false_positive_rate']:.4f}")
        print(f"  False Negative Rate: {metrics['false_negative_rate']:.4f}")
    print(f"  Number of samples: {metrics['num_samples']}")
    print(f"  Results saved to: {results_output_path}")

    # Return complete results dictionary
    return {
        "prompt_type": output_name,
        "model": model,
        "dataset_name": dataset_name,
        "n_shots": n_shots,
        "transcript_type": transcript_type,
        "concat_fewshot": concat_fewshot,
        "concat_test": concat_test,
        "two_turns": two_turns,
        "aggregate_fewshot": aggregate_fewshot,
        **metrics,
    }


@api_cache  # Apply the decorator
def get_model_response(
    model: str, messages: List[Dict], max_retries: int = 6
) -> Optional[Tuple[Any, str]]:
    """
    Get a response from the model with retry logic
    """
    if "gpt" in model.lower():
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    modalities=["text"],
                    max_tokens=800,
                    temperature=0.00000001,
                )
                prediction = response.choices[0].message.content.strip()
                return response, prediction
            except Exception as e:
                print(
                    f"Error in getting response (attempt {attempt + 1}/{max_retries}): {str(e)}"
                )

                # Special handling for rate limit errors
                if "rate limit" in str(e).lower():
                    sleep_time = 2 ** (attempt + 2)  # Longer backoff for rate limits
                    print(f"Rate limit hit. Waiting for {sleep_time}s before retry...")
                else:
                    sleep_time = 2**attempt  # Standard exponential backoff

                time.sleep(sleep_time)
    elif "gemini" in model.lower():
        import google.generativeai as genai

        # Initialize Gemini model
        genai_model = genai.GenerativeModel(f"models/{model}")

        for attempt in range(max_retries):
            try:
                # Generate content with Gemini model
                response = genai_model.generate_content(messages)
                prediction = response.text.strip()
                return response, prediction
            except Exception as e:
                print(
                    f"Error in getting response (attempt {attempt + 1}/{max_retries}): {str(e)}"
                )

                # Special handling for rate limit errors
                if "rate limit" in str(e).lower():
                    sleep_time = 2 ** (attempt + 2)  # Longer backoff for rate limits
                    print(f"Rate limit hit. Waiting for {sleep_time}s before retry...")
                else:
                    sleep_time = 2**attempt  # Standard exponential backoff

                time.sleep(sleep_time)

    return None


def extract_json_from_response(response: str) -> Optional[Dict]:
    """
    Extract JSON from the model's response
    """
    try:
        # Try direct JSON parsing first
        return json.loads(response)
    except json.JSONDecodeError:
        # If that fails, try to find JSON in the text using regex
        json_match = re.search(r"({.*?})", response.replace("\n", " "), re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass
    return None


def load_dataset(file_path: str) -> pd.DataFrame:
    """
    Load the dataset from a JSON file
    """
    with open(file_path, "r") as f:
        data = json.load(f)
    for item in data:
        if isinstance(item.get("match", None), str):
            item["match"] = item["match"].lower() == "true"
        if isinstance(item.get("gpt4o_correct", None), str):
            item["gpt4o_correct"] = item["gpt4o_correct"].lower() == "true"
    # Convert to DataFrame for easier handling
    df = pd.DataFrame(data)
    return df
