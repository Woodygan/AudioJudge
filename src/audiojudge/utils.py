"""
Utility functions and classes for AudioJudge package
"""

import json
import base64
import os
import time
import wave
import audioop
import io
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any, Union
from dataclasses import dataclass
from pydub import AudioSegment
import importlib.resources

@dataclass
class AudioExample:
    """
    Represents an in-context learning example for audio comparison.
    
    Attributes:
        audio1_path: Path to the first audio file
        audio2_path: Path to the second audio file
        output: The expected output for this example
        instruction_path: Optional path to instruction audio file
    """
    audio1_path: str
    audio2_path: str
    output: str
    instruction_path: Optional[str] = None

@dataclass
class AudioExamplePointwise:
    """
    Represents an in-context learning example for pointwise audio evaluation.
    
    Attributes:
        audio_path: Path to the audio file to evaluate
        output: The expected output/evaluation for this example
    """
    audio_path: str
    output: str




def encode_audio_file(file_path: str) -> str:
    """Encode an audio file to base64."""
    with open(file_path, "rb") as audio_file:
        encoded_string = base64.b64encode(audio_file.read()).decode('utf-8')
    return encoded_string


def convert_to_16kHz_bytes(file_path: str) -> bytes:
    """
    Convert audio file to 16kHz WAV format and return as bytes.
    This is needed for Gemini API compatibility.
    """
    try:
        # Load audio using pydub
        audio = AudioSegment.from_file(file_path)
        
        # Convert to 16kHz, mono, 16-bit
        audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
        
        # Export to bytes
        buffer = io.BytesIO()
        audio.export(buffer, format="wav")
        return buffer.getvalue()
        
    except Exception as e:
        # Fallback: just read the file as-is
        with open(file_path, "rb") as f:
            return f.read()


def concatenate_audio_files(audio_paths: List[str], 
                          output_path: str, 
                          openai_client=None,
                          signal_folder: str = "signal_audios",
                          is_test: bool = False, 
                          idx: int = 0) -> str:
    """
    Concatenate multiple audio files into a single file with spoken signals.
    
    Args:
        audio_paths: List of audio file paths to concatenate
        output_path: Path where the concatenated file should be saved
        openai_client: OpenAI client for generating TTS signals (optional)
        signal_folder: Directory to store/find signal audio files
        is_test: Whether this is for test audio (affects signal generation)
        idx: Example index for signal generation
        
    Returns:
        Path to the concatenated audio file
    """
    # Create signal folder if it doesn't exist
    os.makedirs(signal_folder, exist_ok=True)
    
    # First, determine the target sample rate
    sample_rates = []
    for audio_path in audio_paths:
        try:
            with wave.open(audio_path, 'rb') as w:
                sample_rates.append(w.getframerate())
        except Exception as e:
            print(f"Error reading {audio_path}: {e}")
    
    if sample_rates:
        from collections import Counter
        target_sample_rate = Counter(sample_rates).most_common(1)[0][0]
    else:
        target_sample_rate = 24000
        
    if idx != 0 and len(audio_paths) != 2:
        raise ValueError("idx setting only works for two audio files")
    
    # Get parameters from first file
    with wave.open(audio_paths[0], 'rb') as first_file:
        params = first_file.getparams()
        params = params._replace(framerate=target_sample_rate)
        nchannels = params.nchannels
        sampwidth = params.sampwidth
    
    # Dictionary to store generated signal files
    signal_segments = {}
    
    # Generate signal files
    required_signals = []
    
    if is_test:
        required_signals.append(("Test", "test.wav"))
    else:
        # Generate Example X signals
        for i in range((len(audio_paths) + 1) // 2):
            if idx == 0:
                required_signals.append((f"Example {i+1}", f"example_{i+1}.wav"))
            else:
                required_signals.append((f"Example {idx}", f"example_{idx}.wav"))
    
    # Generate Audio 1/Audio 2 signals
    required_signals.append(("Audio 1", "audio_1.wav"))
    required_signals.append(("Audio 2", "audio_2.wav"))
    
    # Create all signal files
    for signal_text, signal_filename in required_signals:
        signal_path = get_signal_audio_path(signal_filename, signal_folder)
        
        # Create the signal if it doesn't exist
        if not os.path.exists(signal_path) and openai_client:
            try:
                with openai_client.audio.speech.with_streaming_response.create(
                    model="tts-1",
                    voice="alloy",
                    input=signal_text,
                    response_format="wav"
                ) as response:
                    response.stream_to_file(signal_path)
            except Exception as e:
                print(f"Failed to generate signal {signal_text}: {e}")
                continue
        
        # Read and resample signal if it exists
        if os.path.exists(signal_path):
            with wave.open(signal_path, 'rb') as w:
                signal_rate = w.getframerate()
                signal_frames = w.readframes(w.getnframes())
                signal_channels = w.getnchannels()
                signal_width = w.getsampwidth()
                
                # Resample if needed
                if signal_rate != target_sample_rate:
                    signal_frames, _ = audioop.ratecv(
                        signal_frames, signal_width, signal_channels, 
                        signal_rate, target_sample_rate, None
                    )
                
                signal_segments[signal_filename] = signal_frames
    
    # Create combined audio file
    with wave.open(output_path, 'wb') as output_file:
        output_file.setparams(params)
        
        # Process pairs of audio files
        for i in range(0, len(audio_paths), 2):
            # Add signals
            # Add Example X or Test signal
            if is_test:
                signal_filename = "test.wav"
            else:
                if idx == 0:
                    signal_filename = f"example_{(i//2)+1}.wav"
                else:
                    signal_filename = f"example_{idx}.wav"
            
            # Add the signal
            if signal_filename in signal_segments:
                output_file.writeframes(signal_segments[signal_filename])
                
                # Add silence (0.5 seconds)
                silence_frames = b'\x00' * (int(0.5 * target_sample_rate) * sampwidth * nchannels)
                output_file.writeframes(silence_frames)
                
                # Add "Audio 1" signal
                if "audio_1.wav" in signal_segments:
                    output_file.writeframes(signal_segments["audio_1.wav"])
                    output_file.writeframes(silence_frames)
            
            # Add first audio file of the pair
            with wave.open(audio_paths[i], 'rb') as w:
                audio_rate = w.getframerate()
                audio_frames = w.readframes(w.getnframes())
                audio_channels = w.getnchannels()
                audio_width = w.getsampwidth()
                
                # Resample if needed
                if audio_rate != target_sample_rate:
                    audio_frames, _ = audioop.ratecv(
                        audio_frames, audio_width, audio_channels, 
                        audio_rate, target_sample_rate, None
                    )
                
                output_file.writeframes(audio_frames)
            
            # Add silence
            silence_frames = b'\x00' * (int(0.5 * target_sample_rate) * sampwidth * nchannels)
            output_file.writeframes(silence_frames)
            
            # Check if there's a second file in this pair
            if i + 1 < len(audio_paths):
                # Add "Audio 2" signal
                if "audio_2.wav" in signal_segments:
                    output_file.writeframes(signal_segments["audio_2.wav"])
                    output_file.writeframes(silence_frames)
                
                # Add second audio file
                with wave.open(audio_paths[i+1], 'rb') as w:
                    audio_rate = w.getframerate()
                    audio_frames = w.readframes(w.getnframes())
                    audio_channels = w.getnchannels()
                    audio_width = w.getsampwidth()
                    
                    if audio_rate != target_sample_rate:
                        audio_frames, _ = audioop.ratecv(
                            audio_frames, audio_width, audio_channels, 
                            audio_rate, target_sample_rate, None
                        )
                    
                    output_file.writeframes(audio_frames)
                
                # Add silence
                output_file.writeframes(silence_frames)
    
    return output_path


def concatenate_audio_files_with_instruction(audio_paths: List[str], 
                                           output_path: str, 
                                           openai_client=None,
                                           signal_folder: str = "signal_audios",
                                           is_test: bool = False, 
                                           idx: int = 0) -> str:
    """
    Concatenate multiple audio files into a single file with spoken signals.
    This version expects triplets: [instruction, audio1, audio2, instruction, audio1, audio2, ...]
    
    Args:
        audio_paths: List of audio file paths (must be divisible by 3: instruction, audio1, audio2)
        output_path: Path where the concatenated file should be saved
        openai_client: OpenAI client for generating TTS signals (optional)
        signal_folder: Directory to store/find signal audio files
        is_test: Whether this is for test audio (affects signal generation)
        idx: Example index for signal generation
        
    Returns:
        Path to the concatenated audio file
    """
    # Create signal folder if it doesn't exist
    os.makedirs(signal_folder, exist_ok=True)
    
    # Check if the number of audio files is divisible by 3
    if len(audio_paths) % 3 != 0:
        raise ValueError("Number of audio files must be divisible by 3 (Instruction, Audio 1, Audio 2)")
    
    if idx != 0 and len(audio_paths) != 3:
        raise ValueError("idx setting only works for one example (3 audio files)")
    
    # First, determine the target sample rate
    sample_rates = []
    for audio_path in audio_paths:
        try:
            with wave.open(audio_path, 'rb') as w:
                sample_rates.append(w.getframerate())
        except Exception as e:
            print(f"Error reading {audio_path}: {e}")
    
    if sample_rates:
        from collections import Counter
        target_sample_rate = Counter(sample_rates).most_common(1)[0][0]
    else:
        target_sample_rate = 24000
    
    # Get parameters from first file
    with wave.open(audio_paths[0], 'rb') as first_file:
        params = first_file.getparams()
        params = params._replace(framerate=target_sample_rate)
        nchannels = params.nchannels
        sampwidth = params.sampwidth
    
    # Dictionary to store generated signal files
    signal_segments = {}
    
    # Generate signal files
    required_signals = []
    
    if is_test:
        required_signals.append(("Test", "test.wav"))
    else:
        # Generate Example X signals
        for i in range(len(audio_paths) // 3):
            if idx == 0:
                required_signals.append((f"Example {i+1}", f"example_{i+1}.wav"))
            else:
                required_signals.append((f"Example {idx}", f"example_{idx}.wav"))
    
    # Generate signals for instruction and audio clips
    required_signals.extend([
        ("Instruction", "instruction.wav"),
        ("Audio 1", "audio_1.wav"),
        ("Audio 2", "audio_2.wav")
    ])
    
    # Create all signal files
    for signal_text, signal_filename in required_signals:
        signal_path = get_signal_audio_path(signal_filename, signal_folder)
        
        # Create the signal if it doesn't exist
        if not os.path.exists(signal_path) and openai_client:
            try:
                with openai_client.audio.speech.with_streaming_response.create(
                    model="tts-1",
                    voice="alloy",
                    input=signal_text,
                    response_format="wav"
                ) as response:
                    response.stream_to_file(signal_path)
            except Exception as e:
                print(f"Failed to generate signal {signal_text}: {e}")
                continue
        
        # Read and resample signal if it exists
        if os.path.exists(signal_path):
            with wave.open(signal_path, 'rb') as w:
                signal_rate = w.getframerate()
                signal_frames = w.readframes(w.getnframes())
                signal_channels = w.getnchannels()
                signal_width = w.getsampwidth()
                
                # Resample if needed
                if signal_rate != target_sample_rate:
                    signal_frames, _ = audioop.ratecv(
                        signal_frames, signal_width, signal_channels, 
                        signal_rate, target_sample_rate, None
                    )
                
                signal_segments[signal_filename] = signal_frames
    
    # Create combined audio file
    with wave.open(output_path, 'wb') as output_file:
        output_file.setparams(params)
        
        # Process triplets of audio files (instruction, audio1, audio2)
        for i in range(0, len(audio_paths), 3):
            # Add silence function
            silence_frames = b'\x00' * (int(0.5 * target_sample_rate) * sampwidth * nchannels)
            
            # Add Example X or Test signal
            if is_test:
                signal_filename = "test.wav"
            else:
                if idx == 0:
                    signal_filename = f"example_{(i//3)+1}.wav"
                else:
                    signal_filename = f"example_{idx}.wav"
            
            # Add the example/test signal
            if signal_filename in signal_segments:
                output_file.writeframes(signal_segments[signal_filename])
                output_file.writeframes(silence_frames)
                
                # Add "Instruction" signal
                if "instruction.wav" in signal_segments:
                    output_file.writeframes(signal_segments["instruction.wav"])
                    output_file.writeframes(silence_frames)
            
            # Add instruction audio file (resampled if needed)
            with wave.open(audio_paths[i], 'rb') as w:
                audio_rate = w.getframerate()
                audio_frames = w.readframes(w.getnframes())
                audio_channels = w.getnchannels()
                audio_width = w.getsampwidth()
                
                # Resample if needed
                if audio_rate != target_sample_rate:
                    audio_frames, _ = audioop.ratecv(
                        audio_frames, audio_width, audio_channels, 
                        audio_rate, target_sample_rate, None
                    )
                
                output_file.writeframes(audio_frames)
                output_file.writeframes(silence_frames)
            
            # Add "Audio 1" signal
            if "audio_1.wav" in signal_segments:
                output_file.writeframes(signal_segments["audio_1.wav"])
                output_file.writeframes(silence_frames)
            
            # Add audio 1 file
            with wave.open(audio_paths[i+1], 'rb') as w:
                audio_rate = w.getframerate()
                audio_frames = w.readframes(w.getnframes())
                audio_channels = w.getnchannels()
                audio_width = w.getsampwidth()
                
                if audio_rate != target_sample_rate:
                    audio_frames, _ = audioop.ratecv(
                        audio_frames, audio_width, audio_channels, 
                        audio_rate, target_sample_rate, None
                    )
                
                output_file.writeframes(audio_frames)
                output_file.writeframes(silence_frames)
            
            # Add "Audio 2" signal
            if "audio_2.wav" in signal_segments:
                output_file.writeframes(signal_segments["audio_2.wav"])
                output_file.writeframes(silence_frames)
            
            # Add audio 2 file
            with wave.open(audio_paths[i+2], 'rb') as w:
                audio_rate = w.getframerate()
                audio_frames = w.readframes(w.getnframes())
                audio_channels = w.getnchannels()
                audio_width = w.getsampwidth()
                
                if audio_rate != target_sample_rate:
                    audio_frames, _ = audioop.ratecv(
                        audio_frames, audio_width, audio_channels, 
                        audio_rate, target_sample_rate, None
                    )
                
                output_file.writeframes(audio_frames)
                output_file.writeframes(silence_frames)
    
    return output_path

def concatenate_audio_files_pointwise(audio_paths: List[str], 
                                      output_path: str, 
                                      openai_client=None,
                                      signal_folder: str = "signal_audios",
                                      is_test: bool = False, 
                                      idx: int = 0) -> str:
    """
    Concatenate multiple audio files into a single file with spoken signals for pointwise evaluation.
    Each audio file is treated as a separate example for evaluation.
    
    Args:
        audio_paths: List of audio file paths to concatenate
        output_path: Path where the concatenated file should be saved
        openai_client: OpenAI client for generating TTS signals (optional)
        signal_folder: Directory to store/find signal audio files
        is_test: Whether this is for test audio (affects signal generation)
        idx: Example index for signal generation (0 for auto-numbering)
        
    Returns:
        Path to the concatenated audio file
    """
    # Create signal folder if it doesn't exist
    os.makedirs(signal_folder, exist_ok=True)
    
    # First, determine the target sample rate
    sample_rates = []
    for audio_path in audio_paths:
        try:
            with wave.open(audio_path, 'rb') as w:
                sample_rates.append(w.getframerate())
        except Exception as e:
            print(f"Error reading {audio_path}: {e}")
    
    if sample_rates:
        from collections import Counter
        target_sample_rate = Counter(sample_rates).most_common(1)[0][0]
    else:
        target_sample_rate = 24000
        
    if idx != 0 and len(audio_paths) != 1:
        raise ValueError("idx setting only works for single audio file")
    
    # Get parameters from first file
    with wave.open(audio_paths[0], 'rb') as first_file:
        params = first_file.getparams()
        params = params._replace(framerate=target_sample_rate)
        nchannels = params.nchannels
        sampwidth = params.sampwidth
    
    # Dictionary to store generated signal files
    signal_segments = {}
    
    # Generate signal files
    required_signals = []
    
    if is_test:
        required_signals.append(("Test", "test.wav"))
    else:
        # Generate Example X signals for each audio file
        for i in range(len(audio_paths)):
            if idx == 0:
                required_signals.append((f"Example {i+1}", f"example_{i+1}.wav"))
            else:
                required_signals.append((f"Example {idx}", f"example_{idx}.wav"))
    
    # Generate Audio signal (no need for Audio 1/Audio 2 in pointwise)
    required_signals.append(("Audio", "audio.wav"))
    
    # Create all signal files
    for signal_text, signal_filename in required_signals:
        signal_path = get_signal_audio_path(signal_filename, signal_folder)
        
        # Create the signal if it doesn't exist
        if not os.path.exists(signal_path) and openai_client:
            try:
                with openai_client.audio.speech.with_streaming_response.create(
                    model="tts-1",
                    voice="alloy",
                    input=signal_text,
                    response_format="wav"
                ) as response:
                    response.stream_to_file(signal_path)
            except Exception as e:
                print(f"Failed to generate signal {signal_text}: {e}")
                continue
        
        # Read and resample signal if it exists
        if os.path.exists(signal_path):
            with wave.open(signal_path, 'rb') as w:
                signal_rate = w.getframerate()
                signal_frames = w.readframes(w.getnframes())
                signal_channels = w.getnchannels()
                signal_width = w.getsampwidth()
                
                # Resample if needed
                if signal_rate != target_sample_rate:
                    signal_frames, _ = audioop.ratecv(
                        signal_frames, signal_width, signal_channels, 
                        signal_rate, target_sample_rate, None
                    )
                
                signal_segments[signal_filename] = signal_frames
    
    # Create combined audio file
    with wave.open(output_path, 'wb') as output_file:
        output_file.setparams(params)
        
        # Process each audio file individually
        for i, audio_path in enumerate(audio_paths):
            # Add silence function
            silence_frames = b'\x00' * (int(0.5 * target_sample_rate) * sampwidth * nchannels)
            
            # Add Example X or Test signal
            if is_test:
                signal_filename = "test.wav"
            else:
                if idx == 0:
                    signal_filename = f"example_{i+1}.wav"
                else:
                    signal_filename = f"example_{idx}.wav"
            
            # Add the example/test signal
            if signal_filename in signal_segments:
                output_file.writeframes(signal_segments[signal_filename])
                output_file.writeframes(silence_frames)
                
                # Add "Audio" signal
                if "audio.wav" in signal_segments:
                    output_file.writeframes(signal_segments["audio.wav"])
                    output_file.writeframes(silence_frames)
            
            # Add the audio file (resampled if needed)
            with wave.open(audio_path, 'rb') as w:
                audio_rate = w.getframerate()
                audio_frames = w.readframes(w.getnframes())
                audio_channels = w.getnchannels()
                audio_width = w.getsampwidth()
                
                # Resample if needed
                if audio_rate != target_sample_rate:
                    audio_frames, _ = audioop.ratecv(
                        audio_frames, audio_width, audio_channels, 
                        audio_rate, target_sample_rate, None
                    )
                
                output_file.writeframes(audio_frames)
            
            # Add silence between examples (longer pause)
            silence_frames = b'\x00' * (int(0.7 * target_sample_rate) * sampwidth * nchannels)
            output_file.writeframes(silence_frames)
    
    return output_path

def get_signal_audio_path(signal_filename: str, user_signal_folder: str = "signal_audios") -> str:
    """
    Get signal audio file path with fallback strategy:
    1. Check user's local signal folder
    2. Check package's default signal folder
    3. Return path for generation if neither exists
    
    Args:
        signal_filename: Name of the signal file (e.g., "audio_1.wav")
        user_signal_folder: User's local signal folder path
        
    Returns:
        Path to the signal audio file
    """
    # 1. Check user's local folder first
    user_signal_path = os.path.join(user_signal_folder, signal_filename)
    if os.path.exists(user_signal_path):
        return user_signal_path
    
    # 2. Check package's default signal folder
    try:
        # Convert the Traversable to a string path
        package_signal_path = str(importlib.resources.files('audiojudge').joinpath(f'signal_audios/{signal_filename}'))
        if os.path.exists(package_signal_path):
            return package_signal_path
    except Exception:
        pass
    
    # 3. Return user path for generation (will be created if needed)
    return user_signal_path