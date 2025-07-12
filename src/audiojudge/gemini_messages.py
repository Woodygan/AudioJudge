"""
Gemini prompt generation module for AudioJudge
Handles Gemini-specific message format with mime_type and data structure
"""

import json
import os
import time
import io
from typing import List, Any, Optional
from pydub import AudioSegment
from .utils import AudioExample, concatenate_audio_files, concatenate_audio_files_with_instruction, AudioExamplePointwise, concatenate_audio_files_pointwise


def convert_audio_to_gemini_format(audio_path: str) -> bytes:
    """Convert audio file to 16kHz WAV format for Gemini API."""
    audio = AudioSegment.from_file(audio_path)
    if audio.frame_rate != 16000:
        audio = audio.set_frame_rate(16000)
    
    content = io.BytesIO()
    audio.export(content, format="wav")
    content.seek(0)
    return content.read()


def get_gemini_messages(audio1_path: str,
                       audio2_path: str,
                       system_prompt: str,
                       user_prompt: Optional[str] = None,
                       examples: Optional[List[AudioExample]] = None,
                       concatenation_method: str = "no_concatenation",
                       openai_client=None,
                       signal_folder: str = "signal_audios") -> List:
    """
    Build Gemini-format messages for audio comparison.
    
    Args:
        audio1_path: Path to the first audio file
        audio2_path: Path to the second audio file
        system_prompt: System prompt that defines the task
        user_prompt: Optional user prompt (if None, will use default)
        examples: List of AudioExample objects for in-context learning
        concatenation_method: Method for concatenating audio files
        openai_client: OpenAI client for TTS signal generation
        signal_folder: Directory for signal audio files
        
    Returns:
        List of message items for Gemini API
    """
    # Initialize messages with the system prompt
    messages = [system_prompt]
    
    # Set default user message
    if user_prompt is None:
        user_prompt = "Please provide your response according to these audio clips:"
    
    # Handle different concatenation methods
    if concatenation_method == "no_concatenation":
        # Add examples separately, test separately
        if examples:
            _add_separate_examples_gemini(messages, examples)
        _add_separate_test_audio_gemini(messages, audio1_path, audio2_path, user_prompt)
        
    elif concatenation_method == "pair_example_concatenation":
        # Each example pair concatenated, test separate
        if examples:
            _add_pair_concatenated_examples_gemini(messages, examples, openai_client, signal_folder)
        _add_separate_test_audio_gemini(messages, audio1_path, audio2_path, user_prompt)
        
    elif concatenation_method == "examples_concatenation":
        # All examples concatenated into one, test separate
        if examples:
            _add_all_examples_concatenated_gemini(messages, examples, openai_client, signal_folder)
        _add_separate_test_audio_gemini(messages, audio1_path, audio2_path, user_prompt)
        
    elif concatenation_method == "test_concatenation":
        # Examples separate, test concatenated
        if examples:
            _add_separate_examples_gemini(messages, examples)
        _add_test_concatenated_gemini(messages, audio1_path, audio2_path, user_prompt, openai_client, signal_folder)
        
    elif concatenation_method == "examples_and_test_concatenation":
        if examples:
            _add_all_examples_concatenated_gemini(messages, examples, openai_client, signal_folder)
        _add_test_concatenated_gemini(messages, audio1_path, audio2_path, user_prompt, openai_client, signal_folder)
    else:
        raise ValueError(f"Unknown concatenation method: {concatenation_method}")
    
    return messages


def _add_separate_examples_gemini(messages: List, examples: List[AudioExample]):
    """Add examples as separate audio files for Gemini."""
    for i, example in enumerate(examples):
        # Convert first audio to Gemini format
        audio1_bytes = convert_audio_to_gemini_format(example.audio1_path)
        
        # Add first audio message
        messages.append(f"Here is the first audio clip:")
        messages.append({
            "mime_type": "audio/wav",
            "data": audio1_bytes
        })
        
        # Convert second audio to Gemini format
        audio2_bytes = convert_audio_to_gemini_format(example.audio2_path)
        
        # Add second audio message
        messages.append(f"Here is the second audio clip:")
        messages.append({
            "mime_type": "audio/wav",
            "data": audio2_bytes
        })
        
        messages.append("Please analyze these audio clips:")
        messages.append("Here is the assistant's response for this example:")
        # Add assistant response
        assistant_response = example.output
        messages.append(assistant_response)


def _add_pair_concatenated_examples_gemini(messages: List, 
                                          examples: List[AudioExample], 
                                          openai_client,
                                          signal_folder: str):
    """Add examples with each pair concatenated into one audio file for Gemini."""
    # Create temp directory if it doesn't exist
    os.makedirs("temp_audio", exist_ok=True)
    
    for i, example in enumerate(examples):
        # Concatenate this example's audio pair
        concat_path = concatenate_audio_files(
            audio_paths=[example.audio1_path, example.audio2_path],
            output_path=os.path.join("temp_audio", f"example_pair_{i}_{time.time()}.wav"),
            openai_client=openai_client,
            signal_folder=signal_folder,
            is_test=False,
            idx=i+1
        )
        
        # Convert audio to Gemini format
        example_audio_bytes = convert_audio_to_gemini_format(concat_path)
        
        # Add example message
        messages.append(f"Please analyze these audio clips:")
        messages.append({
            "mime_type": "audio/wav",
            "data": example_audio_bytes
        })
        
        messages.append("Here is the assistant's response for this example:")
        # Add assistant response
        assistant_response = example.output
        messages.append(assistant_response)
        
        # Clean up the temporary file
        os.remove(concat_path)


def _add_all_examples_concatenated_gemini(messages: List, 
                                         examples: List[AudioExample],
                                         openai_client,
                                         signal_folder: str):
    """Add all examples concatenated into one audio file for Gemini."""
    # Collect all example audio paths
    all_example_audio_paths = []
    examples_data = []
    
    for i, example in enumerate(examples):
        all_example_audio_paths.append(example.audio1_path)
        all_example_audio_paths.append(example.audio2_path)
        
        examples_data.append({
            "result": example.output
        })
    
    # Create temp directory if it doesn't exist
    os.makedirs("temp_audio", exist_ok=True)
    
    # Concatenate all examples
    concat_examples_path = concatenate_audio_files(
        audio_paths=all_example_audio_paths,
        output_path=os.path.join("temp_audio", f"all_examples_{time.time()}.wav"),
        openai_client=openai_client,
        signal_folder=signal_folder,
        is_test=False,
        idx=0
    )
    
    # Convert audio to Gemini format
    examples_audio_bytes = convert_audio_to_gemini_format(concat_examples_path)
    
    # Add examples message
    messages.append("Here are some examples for reference:")
    messages.append({
        "mime_type": "audio/wav",
        "data": examples_audio_bytes
    })
    
    # Add examples data
    example_text = "Examples information:\n"
    for i, example_data in enumerate(examples_data):
        example_text += f"Example {i+1}:\n"
        example_text += f"- Expected output: {example_data['result']}\n\n"
    
    messages.append(example_text)
    
    # Clean up the temporary file
    os.remove(concat_examples_path)


def _add_separate_test_audio_gemini(messages: List, audio1_path: str, audio2_path: str, user_prompt: str):
    """Add test audio as separate files for Gemini."""
    # Convert first audio to Gemini format
    audio1_bytes = convert_audio_to_gemini_format(audio1_path)
    
    # Add first audio message
    messages.append("Here is the first audio clip:")
    messages.append({
        "mime_type": "audio/wav",
        "data": audio1_bytes
    })
    
    # Convert second audio to Gemini format
    audio2_bytes = convert_audio_to_gemini_format(audio2_path)
    
    # Add second audio message
    messages.append("Here is the second audio clip:")
    messages.append({
        "mime_type": "audio/wav",
        "data": audio2_bytes
    })
    
    messages.append(user_prompt)


def _add_test_concatenated_gemini(messages: List, 
                                 audio1_path: str, 
                                 audio2_path: str, 
                                 user_prompt: str, 
                                 openai_client,
                                 signal_folder: str):
    """Add test audio concatenated into one file for Gemini."""
    # Create temp directory if it doesn't exist
    os.makedirs("temp_audio", exist_ok=True)
    
    concat_test_path = concatenate_audio_files(
        audio_paths=[audio1_path, audio2_path],
        output_path=os.path.join("temp_audio", f"test_{time.time()}.wav"),
        openai_client=openai_client,
        signal_folder=signal_folder,
        is_test=True,
        idx=0
    )
    
    # Convert audio to Gemini format
    test_audio_bytes = convert_audio_to_gemini_format(concat_test_path)
    
    # Add test message
    messages.append("Please analyze these audio clips:")
    messages.append({
        "mime_type": "audio/wav",
        "data": test_audio_bytes
    })
    
    messages.append(user_prompt)
    
    # Clean up the temporary file
    os.remove(concat_test_path)


def get_gemini_messages_with_instruction(instruction_path: str,
                                        audio1_path: str,
                                        audio2_path: str,
                                        system_prompt: str,
                                        user_prompt: Optional[str] = None,
                                        examples: Optional[List[AudioExample]] = None,
                                        concatenation_method: str = "no_concatenation",
                                        openai_client=None,
                                        signal_folder: str = "signal_audios") -> List:
    """
    Build Gemini-format messages for audio comparison with instruction audio.
    
    Args:
        instruction_path: Path to the instruction audio file
        audio1_path: Path to the first audio file
        audio2_path: Path to the second audio file
        system_prompt: System prompt that defines the task
        user_prompt: Optional user prompt (if None, will use default)
        examples: List of AudioExample objects for in-context learning (must have instruction_path)
        concatenation_method: Method for concatenating audio files
        openai_client: OpenAI client for TTS signal generation
        signal_folder: Directory for signal audio files
        
    Returns:
        List of message items for Gemini API
    """
    # Initialize messages with the system prompt
    messages = [system_prompt]
    
    # Set default user message
    if user_prompt is None:
        user_prompt = "Please analyze these audio clips:"
    
    # Handle different concatenation methods
    if concatenation_method == "no_concatenation":
        # Add examples separately, test separately
        if examples:
            _add_separate_examples_with_instruction_gemini(messages, examples)
        _add_separate_test_audio_with_instruction_gemini(messages, instruction_path, audio1_path, audio2_path, user_prompt)
        
    elif concatenation_method == "pair_example_concatenation":
        # Each example pair concatenated, test separate
        if examples:
            _add_pair_concatenated_examples_with_instruction_gemini(messages, examples, openai_client, signal_folder)
        _add_separate_test_audio_with_instruction_gemini(messages, instruction_path, audio1_path, audio2_path, user_prompt)
        
    elif concatenation_method == "examples_concatenation":
        # All examples concatenated into one, test separate
        if examples:
            _add_all_examples_with_instruction_concatenated_gemini(messages, examples, openai_client, signal_folder)
        _add_separate_test_audio_with_instruction_gemini(messages, instruction_path, audio1_path, audio2_path, user_prompt)
        
    elif concatenation_method == "test_concatenation":
        # Examples separate, test concatenated
        if examples:
            _add_separate_examples_with_instruction_gemini(messages, examples)
        _add_test_with_instruction_concatenated_gemini(messages, instruction_path, audio1_path, audio2_path, user_prompt, openai_client, signal_folder)
        
    elif concatenation_method == "examples_and_test_concatenation":
        if examples:
            _add_all_examples_with_instruction_concatenated_gemini(messages, examples, openai_client, signal_folder)
        _add_test_with_instruction_concatenated_gemini(messages, instruction_path, audio1_path, audio2_path, user_prompt, openai_client, signal_folder)
    else:
        raise ValueError(f"Unknown concatenation method: {concatenation_method}")
    
    return messages


def _add_separate_examples_with_instruction_gemini(messages: List, examples: List[AudioExample]):
    """Add examples as separate audio files with instruction for Gemini."""
    for i, example in enumerate(examples):
        if not hasattr(example, 'instruction_path') or not example.instruction_path:
            raise ValueError(f"Example {i} missing instruction_path for instruction-based evaluation")
            
        # Convert instruction audio to Gemini format
        instruction_bytes = convert_audio_to_gemini_format(example.instruction_path)
        
        # Add instruction audio message
        messages.append(f"Here is the instruction for this example:")
        messages.append({
            "mime_type": "audio/wav",
            "data": instruction_bytes
        })
        
        # Convert first audio to Gemini format
        audio1_bytes = convert_audio_to_gemini_format(example.audio1_path)
        
        # Add first audio message
        messages.append(f"Here is the first audio clip:")
        messages.append({
            "mime_type": "audio/wav",
            "data": audio1_bytes
        })
        
        # Convert second audio to Gemini format
        audio2_bytes = convert_audio_to_gemini_format(example.audio2_path)
        
        # Add second audio message
        messages.append(f"Here is the second audio clip:")
        messages.append({
            "mime_type": "audio/wav",
            "data": audio2_bytes
        })
        
        messages.append("Please analyze these audio clips:")
        messages.append("Here is the assistant's response for this example:")
        # Add assistant response
        assistant_response = example.output
        messages.append(assistant_response)


def _add_pair_concatenated_examples_with_instruction_gemini(messages: List, 
                                            examples: List[AudioExample], 
                                            openai_client,
                                            signal_folder: str):
    """Add examples with each pair (instruction + audio1 + audio2) concatenated into one audio file for Gemini."""
    # Create temp directory if it doesn't exist
    os.makedirs("temp_audio", exist_ok=True)
    
    for i, example in enumerate(examples):
        if not hasattr(example, 'instruction_path') or not example.instruction_path:
            raise ValueError(f"Example {i} missing instruction_path for instruction-based evaluation")
            
        # Concatenate this example's audio pair
        concat_path = concatenate_audio_files_with_instruction(
            audio_paths=[example.instruction_path, example.audio1_path, example.audio2_path],
            output_path=os.path.join("temp_audio", f"example_pair_{i}_{time.time()}.wav"),
            openai_client=openai_client,
            signal_folder=signal_folder,
            is_test=False,
            idx=i+1
        )
        
        # Convert audio to Gemini format
        example_audio_bytes = convert_audio_to_gemini_format(concat_path)
        
        # Add example message
        messages.append(f"Please analyze these audio clips:")
        messages.append({
            "mime_type": "audio/wav",
            "data": example_audio_bytes
        })
        
        messages.append("Here is the assistant's response for this example:")
        # Add assistant response
        assistant_response = example.output
        messages.append(assistant_response)
        
        # Clean up the temporary file
        os.remove(concat_path)


def _add_all_examples_with_instruction_concatenated_gemini(messages: List, 
                                                         examples: List[AudioExample],
                                                         openai_client,
                                                         signal_folder: str):
    """Add all examples concatenated into one audio file with instructions for Gemini."""
    # Collect all example audio paths (pairs)
    all_example_audio_paths = []
    examples_data = []
    
    for i, example in enumerate(examples):
        if not hasattr(example, 'instruction_path') or not example.instruction_path:
            raise ValueError(f"Example {i} missing instruction_path for instruction-based evaluation")
            
        all_example_audio_paths.append(example.instruction_path)
        all_example_audio_paths.append(example.audio1_path)
        all_example_audio_paths.append(example.audio2_path)
        
        examples_data.append({
            "result": example.output
        })
    
    # Create temp directory if it doesn't exist
    os.makedirs("temp_audio", exist_ok=True)
    
    # Concatenate all examples
    concat_examples_path = concatenate_audio_files_with_instruction(
        audio_paths=all_example_audio_paths,
        output_path=os.path.join("temp_audio", f"all_examples_with_instruction_{time.time()}.wav"),
        openai_client=openai_client,
        signal_folder=signal_folder,
        is_test=False,
        idx=0
    )
    
    # Convert audio to Gemini format
    examples_audio_bytes = convert_audio_to_gemini_format(concat_examples_path)
    
    # Add examples message
    messages.append("Here are some examples for reference:")
    messages.append({
        "mime_type": "audio/wav",
        "data": examples_audio_bytes
    })
    
    # Add examples data
    example_text = "Examples information:\n"
    for i, example_data in enumerate(examples_data):
        example_text += f"Example {i+1}:\n"
        example_text += f"- Expected output: {example_data['result']}\n\n"
    
    messages.append(example_text)
    
    # Clean up the temporary file
    os.remove(concat_examples_path)


def _add_separate_test_audio_with_instruction_gemini(messages: List, 
                                                   instruction_path: str,
                                                   audio1_path: str, 
                                                   audio2_path: str, 
                                                   user_prompt: str):
    """Add test audio as separate files with instruction for Gemini."""
    # Convert instruction audio to Gemini format
    instruction_bytes = convert_audio_to_gemini_format(instruction_path)
    
    # Add instruction audio message
    messages.append("Here is the instruction for this test:")
    messages.append({
        "mime_type": "audio/wav",
        "data": instruction_bytes
    })
    
    # Convert first audio to Gemini format
    audio1_bytes = convert_audio_to_gemini_format(audio1_path)
    
    # Add first audio message
    messages.append("Here is the first audio clip:")
    messages.append({
        "mime_type": "audio/wav",
        "data": audio1_bytes
    })
    
    # Convert second audio to Gemini format
    audio2_bytes = convert_audio_to_gemini_format(audio2_path)
    
    # Add second audio message
    messages.append("Here is the second audio clip:")
    messages.append({
        "mime_type": "audio/wav",
        "data": audio2_bytes
    })
    
    messages.append(user_prompt)


def _add_test_with_instruction_concatenated_gemini(messages: List, 
                                                 instruction_path: str,
                                                 audio1_path: str, 
                                                 audio2_path: str, 
                                                 user_prompt: str, 
                                                 openai_client,
                                                 signal_folder: str):
    """Add test audio concatenated into one file with instruction for Gemini."""
    # Create temp directory if it doesn't exist
    os.makedirs("temp_audio", exist_ok=True)
    
    concat_test_path = concatenate_audio_files_with_instruction(
        audio_paths=[instruction_path, audio1_path, audio2_path],
        output_path=os.path.join("temp_audio", f"test_with_instruction_{time.time()}.wav"),
        openai_client=openai_client,
        signal_folder=signal_folder,
        is_test=True,
        idx=0
    )
    
    # Convert audio to Gemini format
    test_audio_bytes = convert_audio_to_gemini_format(concat_test_path)
    
    # Add test message
    messages.append("Please provide your response according to these audio clips:")
    messages.append({
        "mime_type": "audio/wav",
        "data": test_audio_bytes
    })
    
    messages.append(user_prompt)
    
    # Clean up the temporary file
    os.remove(concat_test_path)

def get_gemini_messages_pointwise(audio_path: str,
                                 system_prompt: str,
                                 user_prompt: Optional[str] = None,
                                 examples: Optional[List[AudioExamplePointwise]] = None,
                                 concatenation_method: str = "no_concatenation",
                                 openai_client=None,
                                 signal_folder: str = "signal_audios") -> List:
    """
    Build Gemini-format messages for pointwise audio evaluation
    
    Args:
        audio_path: Path to the audio file to evaluate
        system_prompt: System prompt that defines the task
        user_prompt: Optional user prompt (if None, will use default)
        examples: List of AudioExamplePointwise objects for in-context learning
        concatenation_method: Method for concatenating audio files ("no_concatenation" or "examples_concatenation")
        openai_client: OpenAI client for TTS signal generation
        signal_folder: Directory for signal audio files
        
    Returns:
        List of message items for Gemini API
    """
    messages = [system_prompt]
    
    # Set default user message
    if user_prompt is None:
        user_prompt = "Please provide your response according to this audio clip:"
    
    # Handle different concatenation methods
    if concatenation_method == "no_concatenation":
        # Add examples separately, test separately
        if examples:
            _add_separate_examples_pointwise_gemini(messages, examples, user_prompt)
        _add_separate_test_audio_pointwise_gemini(messages, audio_path, user_prompt)
        
    elif concatenation_method == "examples_concatenation":
        # All examples concatenated into one, test separate
        if examples:
            _add_all_examples_concatenated_pointwise_gemini(messages, examples, openai_client, signal_folder)
        _add_separate_test_audio_pointwise_gemini(messages, audio_path, user_prompt)
        
    else:
        raise ValueError(f"Unknown concatenation method for pointwise evaluation: {concatenation_method}")
    
    return messages

def _add_separate_examples_pointwise_gemini(messages: List, examples: List[AudioExamplePointwise], user_prompt: str):
    """Add pointwise examples as separate audio files for Gemini."""
    if not examples:
        return
        
    # Add introductory message for examples
    messages.append("Here are some examples for reference:")
    
    for i, example in enumerate(examples):
        # Convert audio to Gemini format
        audio_bytes = convert_audio_to_gemini_format(example.audio_path)
        
        # Add example message
        messages.append(f"Example {i+1}:")
        messages.append({
            "mime_type": "audio/wav",
            "data": audio_bytes
        })
        messages.append(user_prompt)
        
        # Add assistant response with the expected output
        messages.append("Here is the assistant's response for this example:")
        messages.append(example.output)
    
    messages.append("I understand these examples. I'll apply this understanding to analyze the new audio clip you provide.")


def _add_all_examples_concatenated_pointwise_gemini(messages: List, 
                                                   examples: List[AudioExamplePointwise],
                                                   openai_client,
                                                   signal_folder: str):
    """Add all pointwise examples concatenated into one audio file for Gemini."""
    if not examples:
        return
        
    # Collect all example audio paths
    all_example_audio_paths = []
    examples_data = []
    
    for i, example in enumerate(examples):
        all_example_audio_paths.append(example.audio_path)
        examples_data.append({
            "output": example.output
        })
    
    # Create temp directory if it doesn't exist
    os.makedirs("temp_audio", exist_ok=True)
    
    # Concatenate all examples using the single-audio concatenation function
    concat_examples_path = concatenate_audio_files_pointwise(
        audio_paths=all_example_audio_paths,
        output_path=os.path.join("temp_audio", f"all_examples_pointwise_{time.time()}.wav"),
        openai_client=openai_client,
        signal_folder=signal_folder,
        is_test=False
    )
    
    # Convert audio to Gemini format
    examples_audio_bytes = convert_audio_to_gemini_format(concat_examples_path)
    
    # Add examples message
    messages.append("Here are some examples for reference:")
    messages.append({
        "mime_type": "audio/wav",
        "data": examples_audio_bytes
    })
    
    # Add examples metadata
    example_text = "Examples information:\n"
    for i, example_data in enumerate(examples_data):
        example_text += f"Example {i+1}:\n"
        example_text += f"- Expected output: {example_data['output']}\n\n"
    
    messages.append(example_text)
    messages.append("I understand these examples. I'll apply this understanding to analyze the new audio clip you provide.")
    
    # Clean up temp file
    os.remove(concat_examples_path)


def _add_separate_test_audio_pointwise_gemini(messages: List, audio_path: str, user_prompt: str):
    """Add test audio as a separate file for pointwise evaluation for Gemini."""
    # Convert audio to Gemini format
    audio_bytes = convert_audio_to_gemini_format(audio_path)
    
    # Add test message
    messages.append("Please analyze this audio clip:")
    messages.append({
        "mime_type": "audio/wav",
        "data": audio_bytes
    })
    messages.append(user_prompt)