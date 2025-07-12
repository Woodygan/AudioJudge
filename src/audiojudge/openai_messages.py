"""
OpenAI prompt generation module for AudioJudge
Handles OpenAI-specific message format with input_audio structure
"""

import json
import os
import time
from typing import List, Dict, Any, Optional
from .utils import AudioExample, encode_audio_file, concatenate_audio_files, concatenate_audio_files_with_instruction, AudioExamplePointwise, concatenate_audio_files_pointwise


def get_openai_messages(audio1_path: str,
                       audio2_path: str,
                       system_prompt: str,
                       user_prompt: Optional[str] = None,
                       examples: Optional[List[AudioExample]] = None,
                       concatenation_method: str = "no_concatenation",
                       openai_client=None,
                       signal_folder: str = "signal_audios") -> List[Dict[str, Any]]:
    """
    Build OpenAI-format messages for audio comparison.
    
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
        List of message dictionaries for OpenAI API
    """
    messages = [{"role": "system", "content": system_prompt}]
    
    # Set default user message
    if user_prompt is None:
        user_prompt = "Please provide your response according to these audio clips:"
    
    # Handle different concatenation methods
    if concatenation_method == "no_concatenation":
        # Add examples separately, test separately
        if examples:
            _add_separate_examples_openai(messages, examples, user_prompt)
        _add_separate_test_audio_openai(messages, audio1_path, audio2_path, user_prompt)
        
    elif concatenation_method == "pair_example_concatenation":
        # Each example pair concatenated, test separate
        if examples:
            _add_pair_concatenated_examples_openai(messages, examples, openai_client, signal_folder)
        _add_separate_test_audio_openai(messages, audio1_path, audio2_path, user_prompt)
        
    elif concatenation_method == "examples_concatenation":
        # All examples concatenated into one, test separate
        if examples:
            _add_all_examples_concatenated_openai(messages, examples, openai_client, signal_folder)
        _add_separate_test_audio_openai(messages, audio1_path, audio2_path, user_prompt)
        
    elif concatenation_method == "test_concatenation":
        # Examples separate, test concatenated
        if examples:
            _add_separate_examples_openai(messages, examples, user_prompt)
        _add_test_concatenated_openai(messages, audio1_path, audio2_path, user_prompt, openai_client, signal_folder)
        
    elif concatenation_method == "examples_and_test_concatenation":
        if examples:
            _add_all_examples_concatenated_openai(messages, examples, openai_client, signal_folder)
        _add_test_concatenated_openai(messages, audio1_path, audio2_path, user_prompt, openai_client, signal_folder)
    else:
        raise ValueError(f"Unknown concatenation method: {concatenation_method}")
    
    return messages


def _add_separate_examples_openai(messages: List[Dict], examples: List[AudioExample], user_prompt: str):
    """Add examples as separate audio files."""
    for i, example in enumerate(examples):
        audio1_encoded = encode_audio_file(example.audio1_path)
        audio2_encoded = encode_audio_file(example.audio2_path)
        
        content = [
            {"type": "text", "text": f"Here is the first audio clip:"},
            {"type": "input_audio", "input_audio": {"data": audio1_encoded, "format": "wav"}},
            {"type": "text", "text": f"Here is the second audio clip:"},
            {"type": "input_audio", "input_audio": {"data": audio2_encoded, "format": "wav"}},
            {"type": "text", "text": user_prompt}
        ]
        
        messages.append({"role": "user", "content": content})
        messages.append({"role": "assistant", "content": example.output})


def _add_pair_concatenated_examples_openai(messages: List[Dict], 
                                          examples: List[AudioExample], 
                                          openai_client,
                                          signal_folder: str):
    """Add examples with each pair concatenated into one audio file."""
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
        example_audio = encode_audio_file(concat_path)
        
        content = [
            {"type": "text", "text": f"Please analyze these audio clips:"},
            {"type": "input_audio", "input_audio": {"data": example_audio, "format": "wav"}}
        ]
        
        messages.append({"role": "user", "content": content})
        messages.append({"role": "assistant", "content": example.output})
        
        # Clean up temp file
        os.remove(concat_path)


def _add_all_examples_concatenated_openai(messages: List[Dict], 
                                         examples: List[AudioExample],
                                         openai_client,
                                         signal_folder: str):
    """Add all examples concatenated into one audio file."""
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
    examples_encoded = encode_audio_file(concat_examples_path)
    
    # Create content for examples
    examples_content = [
        {"type": "text", "text": "Here are some examples for reference:"},
        {"type": "input_audio", "input_audio": {"data": examples_encoded, "format": "wav"}},
    ]
    
    # Add examples metadata
    example_text = "Examples information:\n"
    for i, example_data in enumerate(examples_data):
        example_text += f"Example {i+1}:\n"
        example_text += f"- Expected output: {example_data['result']}\n\n"
    
    examples_content.append({"type": "text", "text": example_text})
    
    messages.append({"role": "user", "content": examples_content})
    messages.append({"role": "assistant", 
                    "content": "I understand these examples. I'll apply this understanding to analyze the new audio clips you provide."})
    
    # Clean up temp file
    os.remove(concat_examples_path)


def _add_separate_test_audio_openai(messages: List[Dict], audio1_path: str, audio2_path: str, user_prompt: str):
    """Add test audio as separate files."""
    audio1_encoded = encode_audio_file(audio1_path)
    audio2_encoded = encode_audio_file(audio2_path)
    
    user_content = [
        {"type": "text", "text": "Here is the first audio clip:"},
        {"type": "input_audio", "input_audio": {"data": audio1_encoded, "format": "wav"}},
        {"type": "text", "text": "Here is the second audio clip:"},
        {"type": "input_audio", "input_audio": {"data": audio2_encoded, "format": "wav"}},
        {"type": "text", "text": user_prompt}
    ]
    
    messages.append({"role": "user", "content": user_content})


def _add_test_concatenated_openai(messages: List[Dict], 
                                 audio1_path: str, 
                                 audio2_path: str, 
                                 user_prompt: str, 
                                 openai_client,
                                 signal_folder: str):
    """Add test audio concatenated into one file."""
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
    test_encoded = encode_audio_file(concat_test_path)
    
    user_content = [
        {"type": "text", "text": user_prompt},
        {"type": "input_audio", "input_audio": {"data": test_encoded, "format": "wav"}}
    ]
    
    messages.append({"role": "user", "content": user_content})
    
    # Clean up temp file
    os.remove(concat_test_path)


def get_openai_messages_with_instruction(instruction_path: str,
                                        audio1_path: str,
                                        audio2_path: str,
                                        system_prompt: str,
                                        user_prompt: Optional[str] = None,
                                        examples: Optional[List[AudioExample]] = None,
                                        concatenation_method: str = "no_concatenation",
                                        openai_client=None,
                                        signal_folder: str = "signal_audios") -> List[Dict[str, Any]]:
    """
    Build OpenAI-format messages for audio comparison with instruction audio.
    
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
        List of message dictionaries for OpenAI API
    """
    messages = [{"role": "system", "content": system_prompt}]
    
    # Set default user message
    if user_prompt is None:
        user_prompt = "Please provide your response according to these audio clips:"
    
    # Handle different concatenation methods
    if concatenation_method == "no_concatenation":
        # Add examples separately, test separately
        if examples:
            _add_separate_examples_with_instruction_openai(messages, examples, user_prompt)
        _add_separate_test_audio_with_instruction_openai(messages, instruction_path, audio1_path, audio2_path, user_prompt)
        
    elif concatenation_method == "pair_example_concatenation":
        # Each example pair with instruction concatenated, test separate
        if examples:
            _add_pair_concatenated_examples_with_instruction_openai(messages, examples, openai_client, signal_folder)
        _add_separate_test_audio_with_instruction_openai(messages, instruction_path, audio1_path, audio2_path, user_prompt)
        
    elif concatenation_method == "examples_concatenation":
        # All examples concatenated into one, test separate
        if examples:
            _add_all_examples_with_instruction_concatenated_openai(messages, examples, openai_client, signal_folder)
        _add_separate_test_audio_with_instruction_openai(messages, instruction_path, audio1_path, audio2_path, user_prompt)
        
    elif concatenation_method == "test_concatenation":
        # Examples separate, test concatenated
        if examples:
            _add_separate_examples_with_instruction_openai(messages, examples, user_prompt)
        _add_test_with_instruction_concatenated_openai(messages, instruction_path, audio1_path, audio2_path, user_prompt, openai_client, signal_folder)
        
    elif concatenation_method == "examples_and_test_concatenation":
        if examples:
            _add_all_examples_with_instruction_concatenated_openai(messages, examples, openai_client, signal_folder)
        _add_test_with_instruction_concatenated_openai(messages, instruction_path, audio1_path, audio2_path, user_prompt, openai_client, signal_folder)
    else:
        raise ValueError(f"Unknown concatenation method: {concatenation_method}")
    
    return messages


def _add_separate_examples_with_instruction_openai(messages: List[Dict], examples: List[AudioExample], user_prompt: str):
    """Add examples as separate audio files with instruction."""
    for i, example in enumerate(examples):
        if not hasattr(example, 'instruction_path') or not example.instruction_path:
            raise ValueError(f"Example {i} missing instruction_path for instruction-based evaluation")
            
        instruction_encoded = encode_audio_file(example.instruction_path)
        audio1_encoded = encode_audio_file(example.audio1_path)
        audio2_encoded = encode_audio_file(example.audio2_path)
        
        content = [
            {"type": "text", "text": f"Here is the instruction for this example:"},
            {"type": "input_audio", "input_audio": {"data": instruction_encoded, "format": "wav"}},
            {"type": "text", "text": f"Here is the first audio clip:"},
            {"type": "input_audio", "input_audio": {"data": audio1_encoded, "format": "wav"}},
            {"type": "text", "text": f"Here is the second audio clip:"},
            {"type": "input_audio", "input_audio": {"data": audio2_encoded, "format": "wav"}},
            {"type": "text", "text": user_prompt}
        ]
        
        messages.append({"role": "user", "content": content})
        messages.append({"role": "assistant", "content": example.output})


def _add_pair_concatenated_examples_with_instruction_openai(messages: List[Dict], 
                                            examples: List[AudioExample], 
                                            openai_client,
                                            signal_folder: str):
    """Add examples with each pair with instruction (instruction + audio1 + audio2) concatenated into one audio file."""
    # Create temp directory if it doesn't exist
    os.makedirs("temp_audio", exist_ok=True)
    
    for i, example in enumerate(examples):
        if not hasattr(example, 'instruction_path') or not example.instruction_path:
            raise ValueError(f"Example {i} missing instruction_path for instruction-based evaluation")
            
        # Concatenate this example's audio pair with instruction
        concat_path = concatenate_audio_files_with_instruction(
            audio_paths=[example.instruction_path, example.audio1_path, example.audio2_path],
            output_path=os.path.join("temp_audio", f"example_pair_{i}_{time.time()}.wav"),
            openai_client=openai_client,
            signal_folder=signal_folder,
            is_test=False,
            idx=i+1
        )
        example_audio = encode_audio_file(concat_path)
        
        content = [
            {"type": "text", "text": f"Please analyze these audio clips:"},
            {"type": "input_audio", "input_audio": {"data": example_audio, "format": "wav"}}
        ]
        
        messages.append({"role": "user", "content": content})
        messages.append({"role": "assistant", "content": example.output})
        
        # Clean up temp file
        os.remove(concat_path)


def _add_all_examples_with_instruction_concatenated_openai(messages: List[Dict], 
                                                         examples: List[AudioExample],
                                                         openai_client,
                                                         signal_folder: str):
    """Add all examples concatenated into one audio file with instructions."""
    # Collect all example audio paths (pair with instructions)
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
    examples_encoded = encode_audio_file(concat_examples_path)
    
    # Create content for examples
    examples_content = [
        {"type": "text", "text": "Here are some examples for reference:"},
        {"type": "input_audio", "input_audio": {"data": examples_encoded, "format": "wav"}},
    ]
    
    # Add examples metadata
    example_text = "Examples information:\n"
    for i, example_data in enumerate(examples_data):
        example_text += f"Example {i+1}:\n"
        example_text += f"- Expected output: {example_data['result']}\n\n"
    
    examples_content.append({"type": "text", "text": example_text})
    
    messages.append({"role": "user", "content": examples_content})
    messages.append({"role": "assistant", 
                    "content": "I understand these examples. I'll apply this understanding to analyze the new audio clips you provide."})
    
    # Clean up temp file
    os.remove(concat_examples_path)


def _add_separate_test_audio_with_instruction_openai(messages: List[Dict], 
                                                   instruction_path: str,
                                                   audio1_path: str, 
                                                   audio2_path: str, 
                                                   user_prompt: str):
    """Add test audio as separate files with instruction."""
    instruction_encoded = encode_audio_file(instruction_path)
    audio1_encoded = encode_audio_file(audio1_path)
    audio2_encoded = encode_audio_file(audio2_path)
    
    user_content = [
        {"type": "text", "text": "Here is the instruction for this test:"},
        {"type": "input_audio", "input_audio": {"data": instruction_encoded, "format": "wav"}},
        {"type": "text", "text": "Here is the first audio clip:"},
        {"type": "input_audio", "input_audio": {"data": audio1_encoded, "format": "wav"}},
        {"type": "text", "text": "Here is the second audio clip:"},
        {"type": "input_audio", "input_audio": {"data": audio2_encoded, "format": "wav"}},
        {"type": "text", "text": user_prompt}
    ]
    
    messages.append({"role": "user", "content": user_content})


def _add_test_with_instruction_concatenated_openai(messages: List[Dict], 
                                                 instruction_path: str,
                                                 audio1_path: str, 
                                                 audio2_path: str, 
                                                 user_prompt: str, 
                                                 openai_client,
                                                 signal_folder: str):
    """Add test audio concatenated into one file with instruction."""
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
    test_encoded = encode_audio_file(concat_test_path)
    
    user_content = [
        {"type": "text", "text": user_prompt},
        {"type": "input_audio", "input_audio": {"data": test_encoded, "format": "wav"}}
    ]
    
    messages.append({"role": "user", "content": user_content})
    
    # Clean up temp file
    os.remove(concat_test_path)

def get_openai_messages_pointwise(audio_path: str,
                                 system_prompt: str,
                                 user_prompt: Optional[str] = None,
                                 examples: Optional[List[AudioExamplePointwise]] = None,
                                 concatenation_method: str = "no_concatenation",
                                 openai_client=None,
                                 signal_folder: str = "signal_audios") -> List[Dict[str, Any]]:
    """
    Build OpenAI-format messages for pointwise audio evaluation.
    
    Args:
        audio_path: Path to the audio file to evaluate
        system_prompt: System prompt that defines the task
        user_prompt: Optional user prompt (if None, will use default)
        examples: List of AudioExamplePointwise objects for in-context learning
        concatenation_method: Method for concatenating audio files ("no_concatenation" or "examples_concatenation")
        openai_client: OpenAI client for TTS signal generation
        signal_folder: Directory for signal audio files
        
    Returns:
        List of message dictionaries for OpenAI API
    """
    messages = [{"role": "system", "content": system_prompt}]
    
    # Set default user message
    if user_prompt is None:
        user_prompt = "Please provide your response according to this audio clip:"
    
    # Handle different concatenation methods
    if concatenation_method == "no_concatenation":
        # Add examples separately, test separately
        if examples:
            _add_separate_examples_pointwise_openai(messages, examples, user_prompt)
        _add_separate_test_audio_pointwise_openai(messages, audio_path, user_prompt)
        
    elif concatenation_method == "examples_concatenation":
        # All examples concatenated into one, test separate
        if examples:
            _add_all_examples_concatenated_pointwise_openai(messages, examples, openai_client, signal_folder)
        _add_separate_test_audio_pointwise_openai(messages, audio_path, user_prompt)
        
    else:
        raise ValueError(f"Unknown concatenation method for pointwise evaluation: {concatenation_method}")
    
    return messages


def _add_separate_examples_pointwise_openai(messages: List[Dict], examples: List[AudioExamplePointwise], user_prompt: str):
    """Add pointwise examples as separate audio files."""
    if not examples:
        return
        
    # Add introductory message for examples
    intro_content = [
        {"type": "text", "text": "Here are some examples for reference:"}
    ]
    messages.append({"role": "user", "content": intro_content})
    
    for i, example in enumerate(examples):
        audio_encoded = encode_audio_file(example.audio_path)
        
        content = [
            {"type": "text", "text": f"Example {i+1}:"},
            {"type": "input_audio", "input_audio": {"data": audio_encoded, "format": "wav"}},
            {"type": "text", "text": user_prompt}
        ]
        
        messages.append({"role": "user", "content": content})
        
        # Add assistant response with the expected output
        messages.append({
            "role": "assistant",
            "content": example.output
        })
    
    messages.append({
        "role": "assistant", 
        "content": "I understand these examples. I'll apply this understanding to analyze the new audio clip you provide."
    })


def _add_all_examples_concatenated_pointwise_openai(messages: List[Dict], 
                                                   examples: List[AudioExamplePointwise],
                                                   openai_client,
                                                   signal_folder: str):
    """Add all pointwise examples concatenated into one audio file."""
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
    examples_encoded = encode_audio_file(concat_examples_path)
    
    # Create content for examples
    examples_content = [
        {"type": "text", "text": "Here are some examples for reference:"},
        {"type": "input_audio", "input_audio": {"data": examples_encoded, "format": "wav"}},
    ]
    
    # Add examples metadata
    example_text = "Examples information:\n"
    for i, example_data in enumerate(examples_data):
        example_text += f"Example {i+1}:\n"
        example_text += f"- Expected output: {example_data['output']}\n\n"
    
    examples_content.append({"type": "text", "text": example_text})
    
    messages.append({"role": "user", "content": examples_content})
    messages.append({
        "role": "assistant", 
        "content": "I understand these examples. I'll apply this understanding to analyze the new audio clip you provide."
    })
    
    # Clean up temp file
    os.remove(concat_examples_path)


def _add_separate_test_audio_pointwise_openai(messages: List[Dict], audio_path: str, user_prompt: str):
    """Add test audio as a separate file for pointwise evaluation."""
    audio_encoded = encode_audio_file(audio_path)
    
    user_content = [
        {"type": "text", "text": "Please analyze this audio clip:"},
        {"type": "input_audio", "input_audio": {"data": audio_encoded, "format": "wav"}},
        {"type": "text", "text": user_prompt}
    ]
    
    messages.append({"role": "user", "content": user_content})