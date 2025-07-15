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
from utils import (
    get_model_response,
    encode_audio_file,
    extract_json_from_response,
    load_dataset,
)
from helper import get_transcript, get_asr_transcription
from utils_speakbench_gemini import get_prompt_gemini

load_dotenv()
import audioop

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


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
    - audio_paths: List of paths to audio files (must be divisible by 3)
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

    # Check if the number of audio files is divisible by 3
    if len(audio_paths) % 3 != 0:
        raise ValueError(
            "Number of audio files must be divisible by 3 (Instruction, Audio 1, Audio 2)"
        )

    if idx != 0 and len(audio_paths) != 3:
        raise ValueError("idx setting only works for one example (3 audio files)")

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
            for i in range(len(audio_paths) // 3):
                if idx == 0:
                    required_signals.append(
                        (f"Example {i + 1}", f"example_{i + 1}.wav")
                    )
                else:
                    required_signals.append((f"Example {idx}", f"example_{idx}.wav"))

        # Generate Audio signals
        required_signals.append(("Instruction", "instruction.wav"))
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

    # Define a function to add silence
    def add_silence(duration=0.5):
        silence_frames = b"\x00" * (
            int(duration * target_sample_rate) * sampwidth * nchannels
        )
        return silence_frames

    # Create a new WAV file for the output
    with wave.open(output_path, "wb") as output_file:
        output_file.setparams(params)

        # Process triplets of audio files (example by example)
        for i in range(0, len(audio_paths), 3):
            # Add signals if needed
            if add_signals:
                # Add Example X or Test signal
                if is_test:
                    signal_filename = "test.wav"
                else:
                    if idx == 0:
                        signal_filename = f"example_{(i // 3) + 1}.wav"
                    else:
                        signal_filename = f"example_{idx}.wav"

                # Add the example signal
                output_file.writeframes(signal_segments[signal_filename])
                output_file.writeframes(add_silence())

                # Add "Instruction" signal
                output_file.writeframes(signal_segments["instruction.wav"])
                output_file.writeframes(add_silence())

            # Add instruction audio file (resampled if needed)
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
                output_file.writeframes(add_silence())

            # Add "Audio 1" signal if signals are enabled
            if add_signals:
                output_file.writeframes(signal_segments["audio_1.wav"])
                output_file.writeframes(add_silence())

            # Add audio 1 file
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
                output_file.writeframes(add_silence())

            # Add "Audio 2" signal if signals are enabled
            if add_signals:
                output_file.writeframes(signal_segments["audio_2.wav"])
                output_file.writeframes(add_silence())

            # Add audio 2 file
            with wave.open(audio_paths[i + 2], "rb") as w:
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
                output_file.writeframes(add_silence())

    return output_path


def get_prompt(
    prompt_type: str,
    instruction_path: str,
    audio1_path: str,
    audio2_path: str,
    dataset_name: str,
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
    if "speakbench" in dataset_name:
        if prompt_type == "lexical_cot":
            user_message = (
                "Please analyze which of the two recordings follows the instruction better, or tie, focusing only on lexical content, not tone or pronunciations. "
                "Respond ONLY in text and output valid JSON with keys 'reasoning' and 'label' (string, '1', '2' or 'tie')."
            )
        elif prompt_type == "paralinguistic_cot":
            user_message = (
                "Please analyze which of the two recordings follows the instruction better, or tie in terms of paralinguistic features. "
                "Respond ONLY in text and output valid JSON with keys 'reasoning' and 'label' (string, '1', '2' or 'tie')."
            )
        elif prompt_type == "speech_quality_cot":
            user_message = (
                "Please analyze which of the two recordings has a better speech quality, or tie. "
                "Respond ONLY in text and output valid JSON with keys 'reasoning' and 'label' (string, '1', '2' or 'tie')."
            )
        elif prompt_type == "no_cot":
            user_message = (
                "Please analyze which of the two recordings follows the instruction better, or tie. "
                "Respond ONLY in text and output valid JSON with key 'label' (string, '1', '2' or 'tie')."
            )
        else:
            user_message = (
                "Please analyze which of the two recordings follows the instruction better, or tie. "
                "Respond ONLY in text and output valid JSON with keys 'reasoning' and 'label' (string, '1', '2' or 'tie')."
            )
    elif "chatbotarena" in dataset_name:
        if prompt_type == "no_cot":
            user_message = (
                "Please analyze which of the two recordings follows the instruction better, or tie, in terms of content of the responses. "
                "Respond ONLY in text and output valid JSON with key 'label' (string, '1', '2' or 'tie')."
            )
        else:
            user_message = (
                "Please analyze which of the two recordings follows the instruction better, or tie, in terms of content of the responses. "
                "Respond ONLY in text and output valid JSON with keys 'reasoning' and 'label' (string, '1', '2' or 'tie')."
            )

    # Load dataset-specific few-shot examples
    if prompt_type not in ["lexical_cot", "paralinguistic_cot", "speech_quality_cot"]:
        with open("few_shots_examples.json", "r") as f:
            few_shots_examples = json.load(f)
    else:
        # For cot prompts, we use a different few-shot examples file
        with open("few_shots_examples_aspect_specific.json", "r") as f:
            few_shots_examples = json.load(f)
            few_shots_examples = few_shots_examples.get(prompt_type)

    # Check if examples exist for the specified dataset
    if not dataset_name in few_shots_examples:
        raise ValueError(f"No few-shot examples found for dataset: {dataset_name}")
    else:
        available_examples = few_shots_examples[dataset_name]
    print(available_examples[0]["instruction_path"])
    # If few-shot examples are requested, add them before the user query
    if n_shots > 0 and available_examples:
        n_shots = min(n_shots, len(available_examples))
        if two_turns:
            # Handle two-turn examples
            for i in range(n_shots):
                example = available_examples[i]
                instruction = encode_audio_file(example["instruction_path"])
                instruction_content = [
                    {
                        "type": "text",
                        "text": f"Here is the instruction for this example:",
                    },
                    {
                        "type": "input_audio",
                        "input_audio": {"data": instruction, "format": "wav"},
                    },
                ]
                if transcript_type != "none":
                    instruction_transcript = example.get("instruction_transcript")
                    instruction_content.append(
                        {
                            "type": "text",
                            "text": f'Here is the transcript for instruction: "{instruction_transcript}"',
                        }
                    )
                messages.append({"role": "user", "content": instruction_content})
                # Assistant acknowledgement
                messages.append(
                    {
                        "role": "assistant",
                        "content": f"I've heard the instruction for this example.",
                    }
                )
                # First audio turn
                example_audio1 = encode_audio_file(example["audio1_path"])
                first_content = [
                    {"type": "text", "text": f"Here is the first audio clip:"},
                    {
                        "type": "input_audio",
                        "input_audio": {"data": example_audio1, "format": "wav"},
                    },
                ]

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
                all_example_audio_paths.append(example["instruction_path"])
                all_example_audio_paths.append(example["audio1_path"])
                all_example_audio_paths.append(example["audio2_path"])

                example_result = example["assistant_message"].get("label")
                instruction_transcript = example.get("instruction_transcript")

                examples_data.append(
                    {
                        "instruction_transcript": instruction_transcript,
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
                if transcript_type != "none":
                    example_text += f'- Instruction transcript: "{example["instruction_transcript"]}"\n'
                example_text += f"- Label: {json.dumps(example['result'])}\n\n"

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
                    [
                        example["instruction_path"],
                        example["audio1_path"],
                        example["audio2_path"],
                    ],
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
                instruction_transcript = example.get("instruction_transcript")
                if transcript_type != "none":
                    content.append(
                        {
                            "type": "text",
                            "text": f'Transcript for instruction: "{instruction_transcript}"',
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
                instruction = encode_audio_file(example["instruction_path"])
                example_audio1 = encode_audio_file(example["audio1_path"])
                example_audio2 = encode_audio_file(example["audio2_path"])
                instruction_transcript = example.get("instruction_transcript")

                content = [
                    {
                        "type": "text",
                        "text": f"Here is the instruction for this example:",
                    },
                    {
                        "type": "input_audio",
                        "input_audio": {"data": instruction, "format": "wav"},
                    },
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
                            "text": f'Transcript for instruction: "{instruction_transcript}"',
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
        instruction_encoded = encode_audio_file(instruction_path)
        instruction_content = [
            {"type": "text", "text": "Here is the instruction for this test:"},
            {
                "type": "input_audio",
                "input_audio": {"data": instruction_encoded, "format": "wav"},
            },
        ]
        if transcript_type == "groundtruth" or transcript_type == "asr":
            # Get transcript for the instruction
            transcript_instruction = get_transcript(
                instruction_path, transcript_type, dataset_name
            )
            if transcript_instruction:
                instruction_content.append(
                    {
                        "type": "text",
                        "text": f'Transcript for this instruction: "{transcript_instruction}"',
                    }
                )
        messages.append({"role": "user", "content": instruction_content})
        # Assistant acknowledgement
        messages.append(
            {
                "role": "assistant",
                "content": "I've heard the instruction for this test.",
            }
        )
        audio1_encoded = encode_audio_file(audio1_path)
        first_content = [
            {"type": "text", "text": "Here is the first test audio clip:"},
            {
                "type": "input_audio",
                "input_audio": {"data": audio1_encoded, "format": "wav"},
            },
        ]

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

        second_content.append({"type": "text", "text": user_message})

        messages.append({"role": "user", "content": second_content})
    else:
        if concat_test:
            os.makedirs("temp_audio", exist_ok=True)

            concat_test_path = os.path.join(
                "temp_audio", f"concat_test_{time.time()}.wav"
            )
            concatenate_audio_files(
                [instruction_path, audio1_path, audio2_path],
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
            instruction_encoded = encode_audio_file(instruction_path)
            audio1_encoded = encode_audio_file(audio1_path)
            audio2_encoded = encode_audio_file(audio2_path)

            user_content = [
                {"type": "text", "text": "Here is the instruction for this test:"},
                {
                    "type": "input_audio",
                    "input_audio": {"data": instruction_encoded, "format": "wav"},
                },
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

        if transcript_type == "groundtruth" or transcript_type == "asr":
            if transcript_type == "groundtruth":
                transcript_instruction = get_transcript(
                    instruction_path, transcript_type, dataset_name
                )
            else:
                transcript_instruction = get_asr_transcription(instruction_path)
            user_content.append(
                {
                    "type": "text",
                    "text": f'Transcript for this instruction: "{transcript_instruction}"',
                }
            )

        user_content.append({"type": "text", "text": user_message})

        messages.append({"role": "user", "content": user_content})

    return messages


def evaluate_prompt_strategy_speakbench(
    df: pd.DataFrame,
    prompt_type: str,
    model: str,
    dataset_name: str,
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
            raise ValueError("this dataset does not support majority voting")

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
            instruction_path = row["instruction_path"]
            instruction_id = row.get("instruction_id", -1)
            instruction_text = row["instruction_text"]
            index = row.get("index", -1)
            model_a = row.get("model_a", None)
            model_b = row.get("model_b", None)
            ground_truth = row.get("label")
            if "gpt" in model:
                messages = get_prompt(
                    prompt_type=prompt_type,
                    instruction_path=instruction_path,
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
                    instruction_path=instruction_path,
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

            prediction = prediction_json.get("label", None)

            reasoning = prediction_json.get("reasoning", "")

            if prediction is None:
                print(
                    f"No prediction field in extracted JSON for {audio1_path} and {audio2_path}"
                )
                continue
            results.append(
                {
                    "index": index,
                    "audio1_path": audio1_path,
                    "audio2_path": audio2_path,
                    "ground_truth": ground_truth,
                    "prediction": prediction,
                    "reasoning": reasoning,
                    "correct": 1 if prediction == ground_truth else 0,
                    "prompt_type": prompt_type,
                    "dataset_name": dataset_name,
                    "n_shots": n_shots,
                    "transcript_type": transcript_type,
                    "concat_fewshot": concat_fewshot,
                    "concat_test": concat_test,
                    "two_turns": two_turns,
                    "aggregate_fewshot": aggregate_fewshot,
                    "instruction_id": instruction_id,
                    "model_a": model_a,
                    "model_b": model_b,
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
                "num_samples": 0,
            }
        accuracy = results_df["correct"].mean()

        metric_summary = {"accuracy": accuracy, "num_samples": len(results_df)}

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
            "num_samples": 0,
            "error": str(e),
        }
