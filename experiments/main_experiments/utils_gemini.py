import io
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
from helper import get_transcript, get_asr_transcription
import audioop


def convert_to_16kHz_bytes(audio_path):
    """
    Convert an audio file to 16kHz and return as bytes

    Parameters:
    - audio_path: Path to the audio file

    Returns:
    - Binary data of the audio file resampled to 16kHz
    """
    # Load the audio file
    audio = AudioSegment.from_file(audio_path)

    # Check if resampling is necessary
    if audio.frame_rate != 16000:
        print(f"Resampling from {audio.frame_rate} Hz to 16 kHz...")
        audio = audio.set_frame_rate(16000)

    # Export the audio to an in-memory buffer in WAV format
    output_buffer = io.BytesIO()
    audio.export(output_buffer, format="wav")
    output_buffer.seek(0)  # Reset the buffer to the beginning

    # Return the binary data
    return output_buffer.read()


def get_prompt_gemini(
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
) -> List:
    """
    Create a prompt for the Gemini API call using the selected prompt type and audio files
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
    - two_turns: Whether to send each audio file in a separate message turn (not supported in Gemini)
    - aggregate_fewshot: Whether to aggregate all few-shot examples into a single audio file

    Returns:
    - List of message items for the Gemini API call
    """
    # Check if two_turns is enabled (not supported in Gemini)
    if two_turns:
        raise ValueError(
            "The 'two_turns' setting is not supported with Gemini. Please set it to False."
        )

    system_prompts = SYSTEM_PROMPTS.get(dataset_name)
    system_prompt = system_prompts.get(prompt_type)

    # Initialize messages with the system prompt
    messages = [system_prompt]

    # Set the user message based on dataset and prompt type
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

        if aggregate_fewshot:
            # Aggregate all few-shot examples into a single audio file
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

            # Convert audio to bytes format for Gemini
            examples_audio_bytes = convert_to_16kHz_bytes(concat_examples_path)

            # Add examples message
            messages.append("Here are some examples for reference:")
            messages.append({"mime_type": "audio/wav", "data": examples_audio_bytes})

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

            messages.append(example_text)
            # messages.append("I understand these examples. I'll apply this understanding to analyze the new audio clips you provide.")

            # Clean up the temporary file
            os.remove(concat_examples_path)

        elif concat_fewshot:
            # Concatenate each datapoint's audio files
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

                # Convert audio to bytes format for Gemini
                example_audio_bytes = convert_to_16kHz_bytes(concat_example_path)

                # Add example message
                messages.append(f"Please analyze these audio clips:")
                messages.append({"mime_type": "audio/wav", "data": example_audio_bytes})

                if transcript_type != "none":
                    # Get transcript keys based on dataset
                    if dataset_name == "pronunciation":
                        transcript1 = example.get("transcript1", example.get("word"))
                        transcript2 = example.get("transcript2", example.get("word"))
                    else:
                        transcript1 = example.get("transcript1", "")
                        transcript2 = example.get("transcript2", "")

                    messages.append(f'Transcript for first audio: "{transcript1}"')
                    messages.append(f'Transcript for second audio: "{transcript2}"')

                messages.append(user_message)
                messages.append("Here is the assistant's response for this example:")
                # Add assistant response
                assistant_response = json.dumps(example["assistant_message"])
                messages.append(assistant_response)

                # Clean up the temporary file
                os.remove(concat_example_path)
        else:
            # Original separate behavior
            for i in range(n_shots):
                example = available_examples[i]

                # Convert first audio to bytes format
                audio1_bytes = convert_to_16kHz_bytes(example["audio1_path"])

                # Add first audio message
                messages.append(f"Here is the first audio clip:")
                messages.append({"mime_type": "audio/wav", "data": audio1_bytes})

                # Convert second audio to bytes format
                audio2_bytes = convert_to_16kHz_bytes(example["audio2_path"])

                # Add second audio message
                messages.append(f"Here is the second audio clip:")
                messages.append({"mime_type": "audio/wav", "data": audio2_bytes})

                # Get transcript info based on dataset
                if dataset_name == "pronunciation":
                    transcript1 = example.get("transcript1", example.get("word"))
                    transcript2 = example.get("transcript2", example.get("word"))
                else:
                    transcript1 = example.get("transcript1", "")
                    transcript2 = example.get("transcript2", "")

                if transcript_type != "none":
                    messages.append(f'Transcript for first audio: "{transcript1}"')
                    messages.append(f'Transcript for second audio: "{transcript2}"')

                messages.append(user_message)
                messages.append("Here is the assistant's response for this example:")
                # Add assistant response
                assistant_response = json.dumps(example["assistant_message"])
                messages.append(assistant_response)

    # Handle the test audio files
    if concat_test:
        os.makedirs("temp_audio", exist_ok=True)

        concat_test_path = os.path.join("temp_audio", f"concat_test_{time.time()}.wav")
        concatenate_audio_files(
            [audio1_path, audio2_path], concat_test_path, add_signals=True, is_test=True
        )

        # Convert audio to bytes format for Gemini
        test_audio_bytes = convert_to_16kHz_bytes(concat_test_path)

        # Add test message
        messages.append("Please analyze these audio clips:")
        messages.append({"mime_type": "audio/wav", "data": test_audio_bytes})

        os.remove(concat_test_path)
    else:
        # Convert first audio to bytes format
        audio1_bytes = convert_to_16kHz_bytes(audio1_path)

        # Add first audio message
        messages.append("Here is the first audio clip:")
        messages.append({"mime_type": "audio/wav", "data": audio1_bytes})

        # Convert second audio to bytes format
        audio2_bytes = convert_to_16kHz_bytes(audio2_path)

        # Add second audio message
        messages.append("Here is the second audio clip:")
        messages.append({"mime_type": "audio/wav", "data": audio2_bytes})

    if transcript_type == "groundtruth":
        transcript1 = get_transcript(audio1_path, transcript_type, dataset_name)
        transcript2 = get_transcript(audio2_path, transcript_type, dataset_name)

        if transcript1 and transcript2:
            messages.append(f'Transcript for first audio: "{transcript1}"')
            messages.append(f'Transcript for second audio: "{transcript2}"')

    elif transcript_type == "asr":
        transcript1 = get_asr_transcription(audio1_path)
        print(f"Transcription for first audio: {transcript1}")
        transcript2 = get_asr_transcription(audio2_path)

        messages.append(f'Transcript for first audio: "{transcript1}"')
        messages.append(f'Transcript for second audio: "{transcript2}"')

    messages.append(user_message)

    return messages


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
