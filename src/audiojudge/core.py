"""
AudioJudge: A simple package for audio comparison using LLMs

This package provides an easy-to-use interface for comparing audio files
using large audio models with optional in-context learning examples.
"""

import json
import base64
import os
import time
import wave
import audioop
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass
from openai import OpenAI
import google.generativeai as genai
from .utils import AudioExample, AudioExamplePointwise
import hashlib
import functools
import diskcache
from .api_cache import APICache

class AudioJudge:
    """
    AudioJudge with decorator-based API caching
    """
    
    def __init__(self, 
                 openai_api_key: Optional[str] = None,
                 google_api_key: Optional[str] = None,
                 temp_dir: str = "temp_audio",
                 signal_folder: str = "signal_audios",
                 cache_dir: Optional[str] = None,
                 cache_expire_seconds: int = 60 * 60 * 24 * 30,
                 disable_cache: bool = False):
        """Initialize AudioJudge with caching"""
        self.temp_dir = temp_dir
        self.signal_folder = signal_folder
        os.makedirs(temp_dir, exist_ok=True)
        os.makedirs(signal_folder, exist_ok=True)
        
        # Initialize API cache
        self.api_cache = APICache(
            cache_dir=cache_dir,
            expire_seconds=cache_expire_seconds,
            disable_cache=disable_cache
        )
        
        # Initialize API clients
        api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        if api_key:
            from openai import OpenAI
            self.openai_client = OpenAI(api_key=api_key)
        else:
            self.openai_client = None
            
        google_key = google_api_key or os.environ.get("GOOGLE_API_KEY")
        if google_key:
            import google.generativeai as genai
            genai.configure(api_key=google_key)
            self.gemini_available = True
        else:
            self.gemini_available = False
    
    @property
    def cache_decorator(self):
        """Get the cache decorator for manual use"""
        return self.api_cache
    
    def _get_model_response(self,
                        model: str,
                        messages: List[Union[str, Dict]],  # Fixed type annotation
                        temperature: float,
                        max_tokens: int,
                        max_retries: int = 3) -> str:
        """Get response from model - this will be cached automatically"""
        
        # Apply the cache decorator directly to the actual API call
        @self.api_cache
        def _cached_model_call(model: str, 
                            messages: List[Union[str, Dict]],  
                            temperature: float, 
                            max_tokens: int) -> str:
            
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
                        try:
                            if not self.gemini_available:
                                raise ValueError("Gemini not available. Please provide a Google API key.")
                            
                            import google.generativeai as genai
                            genai_model = genai.GenerativeModel(f'models/{model}')
                            response = genai_model.generate_content(messages)
                            return response.text.strip()
                        except Exception as e:
                            raise ValueError(f"Failed to call Gemini model: {str(e)}")
                    
                    else:
                        raise ValueError(f"Unsupported model: {model}")
                        
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise e
                    time.sleep(2 ** attempt)
        
        # Call the cached function directly with the messages (including bytes)
        return _cached_model_call(model, messages, temperature, max_tokens)
        
    def judge_audio(self,
                   audio1_path: str,
                   audio2_path: str,
                   system_prompt: str,
                   user_prompt: Optional[str] = None,
                   instruction_path: Optional[str] = None,
                   examples: Optional[List[AudioExample]] = None,
                   model: str = "gpt-4o-audio-preview",
                   concatenation_method: str = "examples_and_test_concatenation",
                   temperature: float = 0.00000001,
                   max_tokens: int = 800) -> Dict[str, Any]:
        """
        Judge/compare two audio files using an audio model.
        
        Args:
            audio1_path: Path to the first audio file
            audio2_path: Path to the second audio file
            system_prompt: System prompt that defines the task
            user_prompt: Optional user prompt (if None, will use default)
            instruction_path: Optional path to instruction audio file
            examples: List of AudioExample objects for in-context learning
            model: Model name to use ("gpt-4o-mini-audio-preview", "gpt-4o-audio-preview", "gemini-1.5-flash", etc.)
            concatenation_method: Method for concatenating audio files:
                - "no_concatenation": Keep all audio files separate
                - "pair_example_concatenation": Concatenate each example pair into one audio file
                - "examples_concatenation": Concatenate all examples into one audio file  
                - "test_concatenation": Concatenate test audio pair into one file
                - "examples_and_test_concatenation": Concatenate all examples, and concatenate test
            temperature: Model temperature (0.0 is not always supported, use 0.00000001 for deterministic)
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
                from .openai_messages import get_openai_messages_with_instruction
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
                from .openai_messages import get_openai_messages
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
                from .gemini_messages import get_gemini_messages_with_instruction
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
                from .gemini_messages import get_gemini_messages
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
    def clear_cache(self):
        """Clear the API cache"""
        self.api_cache.clear_cache()
    
    def clear_none_cache(self):
        """Clear None entries from cache"""
        return self.api_cache.clear_none_cache()
    
    def get_cache_stats(self):
        """Get cache statistics"""
        return self.api_cache.get_cache_stats()
    
    def judge_audio_pointwise(self,
                   audio_path: str,
                   system_prompt: str,
                   user_prompt: Optional[str] = None,
                   examples: Optional[List[AudioExamplePointwise]] = None,
                   model: str = "gpt-4o-audio-preview",
                   concatenation_method: str = "examples_concatenation",
                   temperature: float = 0.00000001,
                   max_tokens: int = 800) -> Dict[str, Any]:
        """
        Evaluate a single audio file using an audio model (pointwise evaluation).
        
        Args:
            audio_path: Path to the audio file to evaluate
            system_prompt: System prompt that defines the evaluation task
            user_prompt: Optional user prompt (if None, will use default)
            examples: List of AudioExamplePointwise objects for in-context learning
            model: Model name to use ("gpt-4o-audio-preview", "gpt-4o-mini-audio-preview", "gemini-1.5-flash", etc.)
            concatenation_method: Method for concatenating audio files:
                - "no_concatenation": Keep all audio files separate
                - "examples_concatenation": Concatenate all examples into one file
            temperature: Model temperature (0.0 is not always supported, use 0.00000001 for deterministic)
            max_tokens: Maximum tokens in response
            
        Returns:
            Dictionary containing the model's response and metadata
        """
        try:
            # Validate inputs
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
            # Build messages based on concatenation method
            messages = self._build_messages_pointwise(
                audio_path=audio_path,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
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
                "audio_path": audio_path,
                "num_examples": len(examples) if examples else 0
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "model": model,
                "concatenation_method": concatenation_method,
                "audio_path": audio_path
            }
        
    def _build_messages_pointwise(self,
                                  audio_path: str,
                                  system_prompt: str,
                                  user_prompt: Optional[str],
                                  examples: Optional[List[AudioExamplePointwise]],
                                  concatenation_method: str,
                                  model: str) -> List[Dict[str, Any]]:
        """Build the message list for pointwise evaluation based on concatenation method and model type."""
        if "gpt" in model.lower():
            from .openai_messages import get_openai_messages_pointwise
            return get_openai_messages_pointwise(
                audio_path=audio_path,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                examples=examples,
                concatenation_method=concatenation_method,
                openai_client=self.openai_client,
                signal_folder=self.signal_folder
            )
        elif "gemini" in model.lower():
            from .gemini_messages import get_gemini_messages_pointwise
            return get_gemini_messages_pointwise(
                audio_path=audio_path,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                examples=examples,
                concatenation_method=concatenation_method,
                openai_client=self.openai_client,
                signal_folder=self.signal_folder
            )
        else:
            raise ValueError(f"Unsupported model type: {model}")