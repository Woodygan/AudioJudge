"""
AudioJudge: A simple package for audio comparison using LLMs

This package provides an easy-to-use interface for comparing audio files
using large language models with optional in-context learning examples.
"""

import json
import base64
import os
import time
import wave
import audioop
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any, Union
from dataclasses import dataclass
from openai import OpenAI
import google.generativeai as genai
from .utils import AudioExample


class AudioJudge:
    """
    Main class for comparing audio files using large language models.
    
    This class provides a simple interface for audio comparison tasks with
    support for in-context learning and various audio processing options.
    """
    
    def __init__(self, 
                 openai_api_key: Optional[str] = None,
                 google_api_key: Optional[str] = None,
                 temp_dir: str = "temp_audio",
                 signal_folder: str = "signal_audios"):
        """
        Initialize the AudioJudge instance.
        
        Args:
            openai_api_key: OpenAI API key (if None, will try to get from environment)
            google_api_key: Google API key for Gemini models (if None, will try to get from environment)
            temp_dir: Directory for temporary audio files
            signal_folder: Directory for signal audio files
        """
        self.temp_dir = temp_dir
        self.signal_folder = signal_folder
        os.makedirs(temp_dir, exist_ok=True)
        os.makedirs(signal_folder, exist_ok=True)
        
        # Initialize OpenAI client
        api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        if api_key:
            self.openai_client = OpenAI(api_key=api_key)
        else:
            self.openai_client = None
            
        # Initialize Gemini
        google_key = google_api_key or os.environ.get("GOOGLE_API_KEY")
        if google_key:
            genai.configure(api_key=google_key)
            self.gemini_available = True
        else:
            self.gemini_available = False
    
    def judge_audio(self,
                   audio1_path: str,
                   audio2_path: str,
                   system_prompt: str,
                   user_prompt: Optional[str] = None,
                   instruction_path: Optional[str] = None,
                   examples: Optional[List[AudioExample]] = None,
                   model: str = "gpt-4o-audio-preview",
                   concatenation_method: str = "no_concatenation",
                   temperature: float = 0.0,
                   max_tokens: int = 800) -> Dict[str, Any]:
        """
        Judge/compare two audio files using a language model.
        
        Args:
            audio1_path: Path to the first audio file
            audio2_path: Path to the second audio file
            system_prompt: System prompt that defines the task
            user_prompt: Optional user prompt (if None, will use default)
            instruction_path: Optional path to instruction audio file
            examples: List of AudioExample objects for in-context learning
            model: Model name to use ("gpt-4o-mini-audio=preview", "gpt-4o-audio-preview", "gemini-1.5-flash", etc.)
            concatenation_method: Method for concatenating audio files:
                - "no_concatenation": Keep all audio files separate
                - "pair_example_concatenation": Concatenate each example pair into one audio file
                - "examples_concatenation": Concatenate all examples into one audio file  
                - "test_concatenation": Concatenate test audio pair into one file
                - "examples_and_test_concatenation": Concatenate all examples and test into one file
            temperature: Model temperature (0.0 for deterministic)
            max_tokens: Maximum tokens in response
            
        Returns:
            Dictionary containing the model's response and metadata
        """
        try:
            # Validate inputs
            if not os.path.exists(audio1_path):
                raise FileNotFoundError(f"Audio file not found: {audio1_path}")
            if not os.path.exists(audio2_path):
                raise FileNotFoundError(f"Audio file not found: {audio2_path}")
            if instruction_path and not os.path.exists(instruction_path):
                raise FileNotFoundError(f"Instruction audio file not found: {instruction_path}")
            
            # Build messages based on concatenation method
            messages = self._build_messages(
                audio1_path=audio1_path,
                audio2_path=audio2_path,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                instruction_path=instruction_path,
                examples=examples,
                concatenation_method=concatenation_method,
                model=model
            )
            
            # Get model response
            response = self._get_model_response(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            return {
                "success": True,
                "response": response,
                "model": model,
                "concatenation_method": concatenation_method,
                "audio1_path": audio1_path,
                "audio2_path": audio2_path,
                "instruction_path": instruction_path,
                "num_examples": len(examples) if examples else 0
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "model": model,
                "concatenation_method": concatenation_method,
                "audio1_path": audio1_path,
                "audio2_path": audio2_path,
                "instruction_path": instruction_path
            }
    
    def _build_messages(self,
                       audio1_path: str,
                       audio2_path: str,
                       system_prompt: str,
                       user_prompt: Optional[str],
                       instruction_path: Optional[str],
                       examples: Optional[List[AudioExample]],
                       concatenation_method: str,
                       model: str) -> List[Dict[str, Any]]:
        """Build the message list for the model based on concatenation method and model type."""
        # Determine which message builder to use based on model
        if "gpt" in model.lower():
            if instruction_path:
                from .openai_prompts import get_openai_messages_with_instruction
                return get_openai_messages_with_instruction(
                    instruction_path=instruction_path,
                    audio1_path=audio1_path,
                    audio2_path=audio2_path,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    examples=examples,
                    concatenation_method=concatenation_method,
                    openai_client=self.openai_client,
                    signal_folder=self.signal_folder
                )
            else:
                from .openai_prompts import get_openai_messages
                return get_openai_messages(
                    audio1_path=audio1_path,
                    audio2_path=audio2_path,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    examples=examples,
                    concatenation_method=concatenation_method,
                    openai_client=self.openai_client,
                    signal_folder=self.signal_folder
                )
        elif "gemini" in model.lower():
            if instruction_path:
                from .gemini_prompts import get_gemini_messages_with_instruction
                return get_gemini_messages_with_instruction(
                    instruction_path=instruction_path,
                    audio1_path=audio1_path,
                    audio2_path=audio2_path,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    examples=examples,
                    concatenation_method=concatenation_method,
                    openai_client=self.openai_client,
                    signal_folder=self.signal_folder
                )
            else:
                from .gemini_prompts import get_gemini_messages
                return get_gemini_messages(
                    audio1_path=audio1_path,
                    audio2_path=audio2_path,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    examples=examples,
                    concatenation_method=concatenation_method,
                    openai_client=self.openai_client,
                    signal_folder=self.signal_folder
                )
        else:
            raise ValueError(f"Unsupported model type: {model}")
    
    def _get_model_response(self,
                           model: str,
                           messages: List[Dict],
                           temperature: float,
                           max_tokens: int,
                           max_retries: int = 3) -> str:
        """Get response from the specified model."""
        for attempt in range(max_retries):
            try:
                if "gpt" in model.lower():
                    if not self.openai_client:
                        raise ValueError("OpenAI client not initialized. Please provide an API key.")
                    
                    response = self.openai_client.chat.completions.create(
                        model=model,
                        messages=messages,
                        modalities=["text"],
                        max_tokens=max_tokens,
                        temperature=temperature
                    )
                    return response.choices[0].message.content.strip()
                
                elif "gemini" in model.lower():
                    if not self.gemini_available:
                        raise ValueError("Gemini not available. Please provide a Google API key.")
                    
                    genai_model = genai.GenerativeModel(f'models/{model}')
                    response = genai_model.generate_content(messages)
                    return response.text.strip()
                
                else:
                    raise ValueError(f"Unsupported model: {model}")
                    
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                time.sleep(2 ** attempt)  # Exponential backoff
    
    def batch_judge(self,
                   audio_pairs: List[Tuple[str, str]],
                   system_prompt: str,
                   user_prompt: Optional[str] = None,
                   instruction_paths: Optional[List[str]] = None,
                   examples: Optional[List[AudioExample]] = None,
                   model: str = "gpt-4o-mini",
                   **kwargs) -> List[Dict[str, Any]]:
        """
        Judge multiple audio pairs in batch.
        
        Args:
            audio_pairs: List of (audio1_path, audio2_path) tuples
            system_prompt: System prompt for the task
            user_prompt: Optional user prompt
            instruction_paths: Optional list of instruction audio paths (same length as audio_pairs)
            examples: In-context learning examples
            model: Model to use
            **kwargs: Additional arguments passed to judge_audio
            
        Returns:
            List of results for each audio pair
        """
        results = []
        for i, (audio1_path, audio2_path) in enumerate(audio_pairs):
            instruction_path = instruction_paths[i] if instruction_paths else None
            
            result = self.judge_audio(
                audio1_path=audio1_path,
                audio2_path=audio2_path,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                instruction_path=instruction_path,
                examples=examples,
                model=model,
                **kwargs
            )
            results.append(result)
        return results