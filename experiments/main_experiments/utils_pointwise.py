import json
import base64
import os
import time
from pathlib import Path
import traceback
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
from system_prompts_pointwise import SYSTEM_PROMPTS
from utils_gemini import get_prompt_gemini
from helper import get_transcript, get_asr_transcription
from utils import get_model_response, extract_json_from_response, encode_audio_file
from utils_gemini import convert_to_16kHz_bytes

load_dotenv()
import audioop

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


def concatenate_audio_files(
    audio_paths: List[str],
    output_path: str,
    add_signals: bool = True,
    signal_folder: str = "signal_audios",
) -> str:
    """
    Concatenate multiple audio files into a single file with spoken labels

    Parameters:
    - audio_paths: List of paths to audio files
    - output_path: Path to save the concatenated audio
    - add_signals: Whether to add spoken signals before each audio clip
    - signal_folder: Folder to store signal audio files for reuse

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
        # Generate example labels
        required_signals = []
        for i in range(len(audio_paths)):
            required_signals.append(
                (f"Example {i + 1}, audio:", f"example_{i + 1}_audio.wav")
            )

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

    # Create a new WAV file for the output
    with wave.open(output_path, "wb") as output_file:
        output_file.setparams(params)

        # Process each audio file sequentially
        for i, audio_path in enumerate(audio_paths):
            # Add the example label if signals are enabled
            if add_signals:
                signal_filename = f"example_{i + 1}_audio.wav"

                # Add the signal
                output_file.writeframes(signal_segments[signal_filename])

                # Add short silence (0.3 seconds)
                silence_frames = b"\x00" * (
                    int(0.3 * target_sample_rate) * sampwidth * nchannels
                )
                output_file.writeframes(silence_frames)

            # Add the audio file (resampled if needed)
            with wave.open(audio_path, "rb") as w:
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

            # Add silence between examples (0.7 seconds)
            silence_frames = b"\x00" * (
                int(0.7 * target_sample_rate) * sampwidth * nchannels
            )
            output_file.writeframes(silence_frames)

    return output_path


def concatenate_audios_speakbench(
    audio_paths: List[str], output_path: str, n_shots: int
) -> str:
    """
    Concatenate audio files following the pattern: "example i+1, instruction, audio"

    Parameters:
    - audio_paths: List of paths to audio files (should contain instruction and audio pairs)
    - output_path: Path to save the concatenated audio
    - n_shots: Number of example shots (used to determine the pattern)

    Returns:
    - Path to the concatenated audio file
    """
    # Create signal folder if it doesn't exist
    signal_folder = "signal_audios"
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

    # Generate spoken labels for examples, instructions, and audio
    spoken_labels = []

    # Add example number labels
    for i in range(1, n_shots + 1):
        spoken_labels.append((f"Example {i}", f"example_{i}.wav"))

    # Add instruction and audio labels
    spoken_labels.append(("Instruction", "instruction.wav"))
    spoken_labels.append(("Audio response", "audio_response.wav"))

    # Create all signal files
    for signal_text, signal_filename in spoken_labels:
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

    # Create a new WAV file for the output
    with wave.open(output_path, "wb") as output_file:
        output_file.setparams(params)

        # Create the pattern: "example i+1, instruction, audio"
        for i in range(n_shots):
            # Add "Example X" signal
            example_signal = signal_segments[f"example_{i + 1}.wav"]
            output_file.writeframes(example_signal)

            # Add silence (0.3 seconds)
            silence_frames = b"\x00" * (
                int(0.3 * target_sample_rate) * sampwidth * nchannels
            )
            output_file.writeframes(silence_frames)

            # Add "Instruction" signal
            instruction_signal = signal_segments["instruction.wav"]
            output_file.writeframes(instruction_signal)

            # Add silence
            output_file.writeframes(silence_frames)

            # Add instruction audio
            instruction_idx = i * 2  # Each example has instruction + audio (2 files)
            with wave.open(audio_paths[instruction_idx], "rb") as w:
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
            output_file.writeframes(silence_frames)

            # Add "Audio response" signal
            audio_signal = signal_segments["audio_response.wav"]
            output_file.writeframes(audio_signal)

            # Add silence
            output_file.writeframes(silence_frames)

            # Add audio response
            audio_idx = i * 2 + 1  # Each example has instruction + audio (2 files)
            with wave.open(audio_paths[audio_idx], "rb") as w:
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

            # Add longer silence between examples (0.7 seconds)
            long_silence = b"\x00" * (
                int(0.7 * target_sample_rate) * sampwidth * nchannels
            )
            output_file.writeframes(long_silence)

    return output_path


def get_prompt(
    prompt_type: str,
    instruction_path: str,
    audio_path: str,
    dataset_name: str = "pronunciation",
    n_shots: int = 0,
    transcript_type: str = "none",
    aggregate_fewshot: bool = False,
) -> List[Dict]:
    """
    Create a prompt for the API call using the selected prompt type and a single audio file
    with options for transcription and aggregated few-shot examples

    Parameters:
    - prompt_type: Type of prompt strategy to use ('no_cot', 'standard_cot', 'phonetic_cot', etc.)
    - audio_path: Path to the audio file to evaluate
    - dataset_name: Name of the dataset ("pronunciation", "speaker", "speed", etc.)
    - n_shots: Number of examples to include (0-8). 0 means no few-shot examples.
    - transcript_type: Type of transcription to use ('none', 'groundtruth', 'asr')
    - aggregate_fewshot: Whether to aggregate all few-shot examples into a single audio file

    Returns:
    - List of message dictionaries for the API call
    """
    system_prompts = SYSTEM_PROMPTS.get(dataset_name)
    system_prompt = system_prompts.get(prompt_type)
    messages = [{"role": "system", "content": system_prompt}]
    user_message = ""

    if (
        dataset_name == "tmhintq"
        or dataset_name == "somos"
        or dataset_name == "thaimos"
    ):
        if prompt_type != "no_cot":
            user_message = (
                "Please analyze the speech quality of this recording. "
                "Respond ONLY in text and output valid JSON with keys 'reasoning' and 'score' (int from 1-5)."
            )
        else:
            user_message = (
                "Please analyze the speech quality of this recording. "
                "Respond ONLY in text and output valid JSON with key 'score' (int from 1-5)."
            )
    elif dataset_name == "speakbench508" or dataset_name == "speakbench":
        if prompt_type != "no_cot":
            user_message = (
                "Please analyze how well this recording follows the instruction. "
                "Respond ONLY in text and output valid JSON with keys 'reasoning' and 'score' (int from 1-5)."
            )
        else:
            user_message = (
                "Please analyze which of the two recordings follows the instruction better, or tie. "
                "Respond ONLY in text and output valid JSON with key 'score' (int from 1-5)."
            )

    # Load dataset-specific few-shot examples
    with open("few_shots_examples_pointwise.json", "r") as f:
        few_shots_examples = json.load(f)

    # Check if examples exist for the specified dataset
    if dataset_name not in few_shots_examples:
        raise ValueError(f"No few-shot examples found for dataset: {dataset_name}")
    else:
        available_examples = few_shots_examples[dataset_name]
    if (
        dataset_name == "tmhintq"
        or dataset_name == "somos"
        or dataset_name == "thaimos"
    ):
        # If few-shot examples are requested, add them before the user query
        if n_shots > 0 and available_examples:
            n_shots = min(n_shots, len(available_examples))

            if aggregate_fewshot:
                # This is the only few-shot option we're keeping
                all_example_audio_paths = []
                examples_data = []

                for i in range(n_shots):
                    example = available_examples[i]
                    # For single audio examples, we only use audio1 from each example
                    all_example_audio_paths.append(example["audio_path"])

                    transcript = example.get("transcript")
                    # For other datasets, assume we have adapted examples with scores
                    example_score = example.get("score")

                    examples_data.append(
                        {"transcript": transcript, "score": example_score}
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
                    {
                        "type": "text",
                        "text": "Here are some examples for reference (Note that their scores are mean option scores so are not necessarily integer, but you will need to give an integer score):",
                    },
                    {
                        "type": "input_audio",
                        "input_audio": {"data": examples_encoded, "format": "wav"},
                    },
                ]

                # Add examples data
                example_text = "Examples information:\n"
                for i, example in enumerate(examples_data):
                    example_text += f"Example {i + 1}:\n"
                    if transcript_type != "none":
                        example_text += (
                            f'- Audio transcript: "{example["transcript"]}"\n'
                        )

                    # Output the score
                    example_text += f"- Score: {example['score']}\n\n"

                examples_content.append({"type": "text", "text": example_text})

                messages.append({"role": "user", "content": examples_content})

                messages.append(
                    {
                        "role": "assistant",
                        "content": "I understand these examples. I'll apply this understanding to analyze the new audio clip you provide.",
                    }
                )

                # Clean up the temporary file
                os.remove(concat_examples_path)
            else:
                # Add each example separately
                for i in range(n_shots):
                    example = available_examples[i]
                    example_audio_path = example["audio_path"]

                    # Encode the audio file
                    audio_encoded = encode_audio_file(example_audio_path)
                    if i == 0:
                        messages.append(
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": "Here are some examples for reference (Note that their scores are mean option scores so are not necessarily integer, but you will need to give an integer score):",
                                    }
                                ],
                            }
                        )
                    user_content = [
                        {"type": "text", "text": f"Example {i + 1}:"},
                        {
                            "type": "input_audio",
                            "input_audio": {"data": audio_encoded, "format": "wav"},
                        },
                    ]

                    transcript = example.get("transcript")
                    if transcript_type != "none":
                        user_content.append(
                            {
                                "type": "text",
                                "text": f'Transcript for audio: "{transcript}"',
                            }
                        )
                    user_content.append(
                        {"type": "text", "text": f"Score: {example.get('score')}"}
                    )

        # Handle the test audio file
        audio_encoded = encode_audio_file(audio_path)
        print(f"Encoded audio file {audio_path}")
        user_content = [
            {"type": "text", "text": "Please analyze this audio clip:"},
            {
                "type": "input_audio",
                "input_audio": {"data": audio_encoded, "format": "wav"},
            },
        ]

        # Add transcript if needed
        if transcript_type == "groundtruth":
            transcript = get_transcript(audio_path, transcript_type, dataset_name)

            if transcript:
                user_content.append(
                    {"type": "text", "text": f'Transcript for audio: "{transcript}"'}
                )

        elif transcript_type == "asr":
            transcript = get_asr_transcription(audio_path)
            user_content.append(
                {"type": "text", "text": f'Transcript for audio: "{transcript}"'}
            )

        user_content.append({"type": "text", "text": user_message})

        messages.append({"role": "user", "content": user_content})
    elif dataset_name == "speakbench508" or dataset_name == "speakbench":
        if n_shots > 0 and available_examples:
            n_shots = min(n_shots, len(available_examples))

            if aggregate_fewshot:
                # Aggregate all few-shot examples into a single audio file
                all_example_audio_paths = []
                examples_data = []

                for i in range(n_shots):
                    example = available_examples[i]
                    # For each example, we need instruction and the audio
                    all_example_audio_paths.append(example["instruction_path"])
                    all_example_audio_paths.append(example["audio_path"])

                    # Extract data
                    instruction_transcript = example.get("instruction_text")
                    example_score = example.get("score")

                    examples_data.append(
                        {
                            "instruction_text": instruction_transcript,
                            "score": example_score,
                        }
                    )

                os.makedirs("temp_audio", exist_ok=True)

                # Concatenate all example audio files with the pattern: "example i+1, instruction, audio"
                concat_examples_path = os.path.join(
                    "temp_audio", f"concat_examples_{time.time()}.wav"
                )
                concatenate_audios_speakbench(
                    all_example_audio_paths, concat_examples_path, n_shots
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
                for i in range(n_shots):
                    example = examples_data[i]
                    example_text += f"Example {i + 1}:\n"
                    if transcript_type != "none":
                        example_text += (
                            f'- Instruction: "{example["instruction_text"]}"\n'
                        )

                    if example["score"] is not None:
                        example_text += f"- Score: {example['score']}\n\n"
                    else:
                        raise ValueError(f"Score not available for example {i + 1}")

                examples_content.append({"type": "text", "text": example_text})

                messages.append({"role": "user", "content": examples_content})

                messages.append(
                    {
                        "role": "assistant",
                        "content": "I understand these examples. I'll apply this understanding to analyze the new audio clip you provide.",
                    }
                )

                # Clean up the temporary file
                os.remove(concat_examples_path)
            else:
                # Add each example separately
                for i in range(n_shots):
                    example = available_examples[i]

                    # Encode the instruction and audio files
                    instruction_encoded = encode_audio_file(example["instruction_path"])
                    audio_encoded = encode_audio_file(example["audio_path"])

                    # Get data
                    instruction_text = example.get("instruction_text")
                    example_score = example.get("score")

                    # Create content for this example
                    if i == 0:
                        messages.append(
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": "Here are some examples for reference:",
                                    }
                                ],
                            }
                        )

                    content = [
                        {"type": "text", "text": f"Example {i + 1}:"},
                        {"type": "text", "text": "Here is the instruction:"},
                        {
                            "type": "input_audio",
                            "input_audio": {
                                "data": instruction_encoded,
                                "format": "wav",
                            },
                        },
                    ]
                    if transcript_type != "none":
                        content.append(
                            {
                                "type": "text",
                                "text": f'Instruction text: "{instruction_text}"',
                            }
                        )
                    (
                        content.extend(
                            [
                                {"type": "text", "text": "Here is the audio response:"},
                                {
                                    "type": "input_audio",
                                    "input_audio": {
                                        "data": audio_encoded,
                                        "format": "wav",
                                    },
                                },
                            ]
                        ),
                    )
                    if example_score is not None:
                        content.append(
                            {"type": "text", "text": f"Score: {example_score}"}
                        )
                    else:
                        raise ValueError(f"Score not available for example {i + 1}")

                    messages.append({"role": "user", "content": content})

                    # Add assistant acknowledgment
                    messages.append(
                        {
                            "role": "assistant",
                            "content": f"I understand Example {i + 1}.",
                        }
                    )

        # Handle the test audio file
        instruction_encoded = encode_audio_file(instruction_path)
        audio_encoded = encode_audio_file(audio_path)

        user_content = [
            {"type": "text", "text": "Please analyze this example:"},
            {"type": "text", "text": "Here is the instruction:"},
            {
                "type": "input_audio",
                "input_audio": {"data": instruction_encoded, "format": "wav"},
            },
        ]

        # Add instruction transcript if needed
        if transcript_type == "groundtruth":
            instruction_transcript = get_transcript(
                instruction_path, transcript_type, dataset_name
            )
            if instruction_transcript:
                user_content.append(
                    {
                        "type": "text",
                        "text": f'Instruction text: "{instruction_transcript}"',
                    }
                )
        elif transcript_type == "asr":
            instruction_transcript = get_asr_transcription(instruction_path)
            user_content.append(
                {
                    "type": "text",
                    "text": f'Instruction text: "{instruction_transcript}"',
                }
            )

        # Add audio
        user_content.extend(
            [
                {"type": "text", "text": "Here is the audio response:"},
                {
                    "type": "input_audio",
                    "input_audio": {"data": audio_encoded, "format": "wav"},
                },
            ]
        )

        user_content.append({"type": "text", "text": user_message})

        messages.append({"role": "user", "content": user_content})

    return messages


def get_prompt_gemini(
    prompt_type: str,
    instruction_path: str,
    audio_path: str,
    dataset_name: str = "pronunciation",
    n_shots: int = 0,
    transcript_type: str = "none",
    aggregate_fewshot: bool = False,
) -> List:
    """
    Create a prompt for the Gemini API call using the selected prompt type and a single audio file
    with options for transcription and aggregated few-shot examples

    Parameters:
    - prompt_type: Type of prompt strategy to use ('no_cot', 'standard_cot', 'phonetic_cot', etc.)
    - instruction_path: Path to the instruction file
    - audio_path: Path to the audio file to evaluate
    - dataset_name: Name of the dataset ("pronunciation", "speaker", "speed", etc.)
    - n_shots: Number of examples to include (0-8). 0 means no few-shot examples.
    - transcript_type: Type of transcription to use ('none', 'groundtruth', 'asr')
    - aggregate_fewshot: Whether to aggregate all few-shot examples into a single audio file

    Returns:
    - List of message items for the Gemini API call
    """
    system_prompts = SYSTEM_PROMPTS.get(dataset_name)
    system_prompt = system_prompts.get(prompt_type)

    # Initialize messages with the system prompt
    messages = [system_prompt]

    user_message = ""

    if (
        dataset_name == "tmhintq"
        or dataset_name == "somos"
        or dataset_name == "thaimos"
    ):
        if prompt_type != "no_cot":
            user_message = (
                "Please analyze the speech quality of this recording. "
                "Respond ONLY in text and output valid JSON with keys 'reasoning' and 'score' (int from 1-5)."
            )
        else:
            user_message = (
                "Please analyze the speech quality of this recording. "
                "Respond ONLY in text and output valid JSON with key 'score' (int from 1-5)."
            )
    elif dataset_name == "speakbench" or dataset_name == "speakbench508":
        pass

    # Load dataset-specific few-shot examples
    with open("few_shots_examples_pointwise.json", "r") as f:
        few_shots_examples = json.load(f)

    # Check if examples exist for the specified dataset
    if dataset_name not in few_shots_examples:
        raise ValueError(f"No few-shot examples found for dataset: {dataset_name}")
    else:
        available_examples = few_shots_examples[dataset_name]

    # If few-shot examples are requested, add them before the user query
    if n_shots > 0 and available_examples:
        n_shots = min(n_shots, len(available_examples))

        if aggregate_fewshot:
            # Aggregate all few-shot examples into a single audio file
            all_example_audio_paths = []
            examples_data = []

            for i in range(n_shots):
                example = available_examples[i]
                # For single audio examples, we only use audio1 from each example
                all_example_audio_paths.append(example["audio_path"])

                # Extract transcript based on dataset
                transcript = example.get("transcript")
                # For datasets with scores
                example_score = example.get("score")

                examples_data.append({"transcript": transcript, "score": example_score})

            os.makedirs("temp_audio", exist_ok=True)

            concat_examples_path = os.path.join(
                "temp_audio", f"concat_examples_{time.time()}.wav"
            )
            concatenate_audio_files(
                all_example_audio_paths, concat_examples_path, add_signals=True
            )

            # Convert audio to bytes format for Gemini
            examples_audio_bytes = convert_to_16kHz_bytes(concat_examples_path)

            # Add examples message
            messages.append(
                "Here are some examples for reference (Note that their scores are mean option scores so are not necessarily integer, but you will need to give an integer score):"
            )
            messages.append({"mime_type": "audio/wav", "data": examples_audio_bytes})

            # Add examples data
            example_text = "Examples information:\n"
            for i, example in enumerate(examples_data):
                example_text += f"Example {i + 1}:\n"
                if transcript_type != "none":
                    example_text += f'- Audio transcript: "{example["transcript"]}"\n'

                # Output the score
                example_text += f"- Score: {example['score']}\n\n"

            messages.append(example_text)
            messages.append(
                "I'll apply this understanding to analyze the new audio clip you provide."
            )

            # Clean up the temporary file
            os.remove(concat_examples_path)
        else:
            # Add each example separately
            for i in range(n_shots):
                example = available_examples[i]
                audio_example_path = example["audio_path"]

                # Convert audio to bytes format for Gemini
                audio_bytes = convert_to_16kHz_bytes(audio_example_path)

                if i == 0:
                    messages.append(
                        "Here are some examples for reference (Note that their scores are mean option scores so are not necessarily integer, but you will need to give an integer score):"
                    )

                user_content = [
                    f"Example {i + 1}:",
                    {"mime_type": "audio/wav", "data": audio_bytes},
                ]

                transcript = example.get("transcript")
                if transcript_type != "none":
                    user_content.append(f'Transcript for audio: "{transcript}"')
                user_content.append(f"Score: {example.get('score')}")

                messages.append(user_content)

    # Handle the test audio file
    audio_bytes = convert_to_16kHz_bytes(audio_path)

    # Add audio message
    messages.append("Please analyze this audio clip:")
    messages.append({"mime_type": "audio/wav", "data": audio_bytes})

    # Add transcript if needed
    if transcript_type == "groundtruth":
        transcript = get_transcript(audio_path, transcript_type, dataset_name)

        if transcript:
            messages.append(f'Transcript for audio: "{transcript}"')

    elif transcript_type == "asr":
        transcript = get_asr_transcription(audio_path)
        messages.append(f'Transcript for audio: "{transcript}"')

    messages.append(user_message)

    return messages


def evaluate_prompt_strategy_pointwise(
    df: pd.DataFrame,
    prompt_type: str,
    model: str,
    dataset_name: str,
    n_samples: int = 100,
    n_shots: int = 0,
    result_dir: str = "results_pointwise",
    transcript_type: str = "none",
    aggregate_fewshot: bool = False,
) -> Dict:
    """
    Evaluate a prompt strategy on the dataset with pairwise comparison (evaluate each audio separately)

    Parameters:
    - df: DataFrame containing the dataset
    - prompt_type: Type of prompt to use (e.g., 'vanilla', 'standard_cot', 'phonetic_cot')
    - model: Model to use for evaluation
    - dataset_name: Name of the dataset ('pronunciation', 'speaker', or 'speed')
    - n_samples: Number of samples to evaluate
    - n_shots: Number of examples to use for few-shot prompting (0 means no examples)
    - result_dir: Directory to save results
    - transcript_type: Type of transcription to use ('none', 'groundtruth', 'asr')
    - aggregate_fewshot: Whether to aggregate all few-shot examples into a single audio file
    """
    try:
        results = []
        # Take a sample of the dataset
        sample_df = df.head(min(n_samples, len(df)))

        # Lists to calculate MSE
        model_scores1 = []
        model_scores2 = []
        gt_scores1 = []
        gt_scores2 = []

        for _, row in tqdm(
            sample_df.iterrows(),
            total=len(sample_df),
            desc=f"Evaluating {prompt_type} with {n_shots} shots, {transcript_type} transcript, "
            f"fewshot_aggregate={aggregate_fewshot}",
        ):
            audio1_path = row["audio1_path"]
            audio2_path = row["audio2_path"]
            if dataset_name == "speakbench" or dataset_name == "speakbench508":
                instruction_path = row["instruction_path"]
                instruction_id = row.get("instruction_id", None)
                instruction_text = row["instruction_text"]
            else:
                instruction_path = None
                instruction_id = None
                instruction_text = None
            ground_truth = row.get("label")

            # Get groundtruth scores based on dataset
            if dataset_name == "somos":
                groundtruth1 = row.get("mos_a")
                groundtruth2 = row.get("mos_b")
            elif dataset_name == "thaimos":
                groundtruth1 = row.get("pronunciation_a")
                groundtruth2 = row.get("pronunciation_b")
            elif dataset_name == "tmhintq":
                groundtruth1 = row.get("human_quality_a")
                groundtruth2 = row.get("human_quality_b")
            else:
                # For datasets without explicit groundtruth scores
                groundtruth1 = None
                groundtruth2 = None

            # Get response for audio 1
            if "gpt" in model:
                messages1 = get_prompt(
                    prompt_type=prompt_type,
                    instruction_path=instruction_path,
                    audio_path=audio1_path,
                    dataset_name=dataset_name,
                    n_shots=n_shots,
                    transcript_type=transcript_type,
                    aggregate_fewshot=aggregate_fewshot,
                )
            elif "gemini" in model:
                messages1 = get_prompt_gemini(
                    prompt_type=prompt_type,
                    instruction_path=instruction_path,
                    audio_path=audio1_path,
                    dataset_name=dataset_name,
                    n_shots=n_shots,
                    transcript_type=transcript_type,
                    aggregate_fewshot=aggregate_fewshot,
                )

            response_data1 = get_model_response(model, messages1)

            # Get response for audio 2
            if "gpt" in model:
                messages2 = get_prompt(
                    prompt_type=prompt_type,
                    instruction_path=instruction_path,
                    audio_path=audio2_path,
                    dataset_name=dataset_name,
                    n_shots=n_shots,
                    transcript_type=transcript_type,
                    aggregate_fewshot=aggregate_fewshot,
                )
            elif "gemini" in model:
                messages2 = get_prompt_gemini(
                    prompt_type=prompt_type,
                    instruction_path=instruction_path,
                    audio_path=audio2_path,
                    dataset_name=dataset_name,
                    n_shots=n_shots,
                    transcript_type=transcript_type,
                    aggregate_fewshot=aggregate_fewshot,
                )

            response_data2 = get_model_response(model, messages2)

            if response_data1 is None or response_data2 is None:
                print(f"Failed to get response for {audio1_path} or {audio2_path}")
                continue

            _, prediction_text1 = response_data1
            _, prediction_text2 = response_data2

            prediction_json1 = extract_json_from_response(prediction_text1)
            prediction_json2 = extract_json_from_response(prediction_text2)

            if prediction_json1 is None or prediction_json2 is None:
                print(
                    f"Failed to extract JSON from response for {audio1_path} or {audio2_path}"
                )
                continue

            # Extract scores for each audio
            score1 = prediction_json1.get("score", None)
            score2 = prediction_json2.get("score", None)

            reasoning1 = prediction_json1.get("reasoning", "")
            reasoning2 = prediction_json2.get("reasoning", "")

            if score1 is None or score2 is None:
                print(
                    f"No score field in extracted JSON for {audio1_path} or {audio2_path}"
                )
                continue

            # Compare scores to determine the prediction
            if score1 > score2:
                prediction = "1"
            elif score2 > score1:
                prediction = "2"
            else:
                prediction = "tie"  # Tie case

            # Calculate correctness (tie counts as 0.5 correct)
            if (
                dataset_name == "somos"
                or dataset_name == "thaimos"
                or dataset_name == "tmhintq"
            ):
                if prediction == "tie":
                    correct = 0.5  # Count tie as half correct
                else:
                    correct = 1.0 if ground_truth == prediction else 0.0
            elif dataset_name == "speakbench" or dataset_name == "speakbench508":
                correct = 1.0 if prediction == ground_truth else 0.0

            # Store scores for MSE calculation if groundtruth scores are available
            if groundtruth1 is not None and groundtruth2 is not None:
                model_scores1.append(score1)
                model_scores2.append(score2)
                gt_scores1.append(groundtruth1)
                gt_scores2.append(groundtruth2)

            # Add result to the list
            result_dict = {
                "audio1_path": audio1_path,
                "audio2_path": audio2_path,
                "ground_truth": ground_truth,
                "prediction": prediction,
                "score1": score1,
                "score2": score2,
                "reasoning1": reasoning1,
                "reasoning2": reasoning2,
                "correct": correct,
                "prompt_type": prompt_type,
                "dataset_name": dataset_name,
                "n_shots": n_shots,
                "transcript_type": transcript_type,
                "aggregate_fewshot": aggregate_fewshot,
                "instruction_id": instruction_id,
            }

            # Add groundtruth scores to results if available
            if groundtruth1 is not None and groundtruth2 is not None:
                result_dict["groundtruth1"] = groundtruth1
                result_dict["groundtruth2"] = groundtruth2

            results.append(result_dict)

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
                "aggregate_fewshot": aggregate_fewshot,
                "accuracy": 0,
                "num_samples": 0,
                "mse_score1": float("nan"),
                "mse_score2": float("nan"),
                "mse_overall": float("nan"),
            }

        # Calculate accuracy (supporting ties as 0.5 correct)
        accuracy = results_df["correct"].mean()

        mse_overall = float("nan")

        if len(model_scores1) > 0 and len(gt_scores1) > 0:
            # Calculate overall MSE
            all_model_scores = model_scores1 + model_scores2
            all_gt_scores = gt_scores1 + gt_scores2
            mse_overall = np.mean(
                [(m - g) ** 2 for m, g in zip(all_model_scores, all_gt_scores)]
            )

        metric_summary = {
            "accuracy": accuracy,
            "num_samples": len(results_df),
            "mse_overall": mse_overall,
        }

        # Save detailed results
        os.makedirs(result_dir, exist_ok=True)

        # Create a more descriptive filename that includes all configurations
        config_desc = (
            f"{dataset_name}_{output_prompt_type}_{n_shots}_shots_{transcript_type}_transcript_"
            f"fewshot_{'aggregate' if aggregate_fewshot else 'separate'}_"
            f"{model.replace('-', '_')}.csv"
        )

        results_output_path = os.path.join(result_dir, config_desc)
        results_df.to_csv(results_output_path, index=False)

        print(f"Results for {dataset_name} - {output_prompt_type}:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Number of samples: {len(results_df)}")

        # Print MSE metrics if available
        if not np.isnan(mse_overall):
            print(f"  Overall MSE: {mse_overall:.4f}")

        # Return results with appropriate metrics
        result_dict = {
            "prompt_type": output_prompt_type,
            "model": model,
            "dataset_name": dataset_name,
            "n_shots": n_shots,
            "transcript_type": transcript_type,
            "aggregate_fewshot": aggregate_fewshot,
            **metric_summary,
        }

        return result_dict
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        traceback.print_exc()
        return {
            "prompt_type": output_prompt_type,
            "model": model,
            "dataset_name": dataset_name,
            "n_shots": n_shots,
            "transcript_type": transcript_type,
            "aggregate_fewshot": aggregate_fewshot,
            "accuracy": 0,
            "num_samples": 0,
            "mse_score1": float("nan"),
            "mse_score2": float("nan"),
            "mse_overall": float("nan"),
            "error": str(e),
        }
