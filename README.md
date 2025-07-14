# AudioJudge 🎵

A powerful Python package for audio comparison and evaluation using Large Audio Models (LAM) with support for in-context learning and flexible audio concatenation strategies.

## Features

- **Multi-Model Support**: Works with OpenAI GPT-4o Audio and Google Gemini models (GPT-4o-audio family, Gemini-1.5/2.0/2.5-flash families)
- **Flexible Audio Comparison**: Support for both pairwise and pointwise audio evaluation
- **In-Context Learning**: Provide examples to improve model performance
- **Audio Concatenation**: Multiple strategies for combining audio files
- **Smart Caching**: Built-in API response caching to reduce costs and latency

## Installation

```bash
pip install audiojudge  # Requires Python >= 3.10
```

## Quick Start

```python
from audiojudge import AudioJudge

# Initialize with API keys
judge = AudioJudge(
    openai_api_key="your-openai-key",
    google_api_key="your-google-key"
)

# Simple pairwise comparison
result = judge.judge_audio(
    audio1_path="audio1.wav",
    audio2_path="audio2.wav",
    system_prompt="Compare these two audio clips for quality.",
    model="gpt-4o-audio-preview"
)

print(result["response"])
```

### Quick Demo
- [AudioJudge with Speaker Identification Demo](examples/audiojudge_huggingface_demo.ipynb)

## Configuration

### Environment Variables

Set your API keys as environment variables:

```bash
export OPENAI_API_KEY="your-openai-key"
export GOOGLE_API_KEY="your-google-key"
export EVAL_CACHE_DIR=".audio_cache"  # Optional
export EVAL_DISABLE_CACHE="false"     # Optional
```

### AudioJudge Parameters

```python
judge = AudioJudge(
    openai_api_key=None,           # OpenAI API key (optional if env var set)
    google_api_key=None,           # Google API key (optional if env var set)
    temp_dir="temp_audio",         # Temporary files directory for storing concatenated audios
    signal_folder="signal_audios", # TTS signal files directory used in audio concatenation
                                   # Default signal files are included in the package
                                   # Will use TTS model to generate new ones if needed
    cache_dir=None,                # API Cache directory (default: .eval_cache)
    cache_expire_seconds=2592000,  # Cache expiration (30 days)
    disable_cache=False            # Disable caching
)
```

## Core Methods

### 1. Pairwise Audio Comparison

### 1.1. Pairwise Comparison without Instruction Audio

Compare two audio files and get a model response directly:

```python
result = judge.judge_audio(
    audio1_path="speaker1.wav",
    audio2_path="speaker2.wav",
    system_prompt="Which speaker sounds more professional?",  # Define the evaluation criteria at the beginning
    user_prompt="Analyze both speakers and provide your assessment.",  # Optional specific instructions at the end
    model="gpt-4o-audio-preview",
    temperature=0.1,  # 0.0 is not supported for some api calling
    max_tokens=500    # Maximum response length
)

if result["success"]:
    print(f"Model response: {result['response']}")
else:
    print(f"Error: {result['error']}")
```

### 1.2. Pairwise Comparison with Instruction Audio

For scenarios where both audio clips are responses to the same instruction (e.g., comparing two speech-in speech-out systems):

```python
result = judge.judge_audio(
    audio1_path="system_a_response.wav",  # Response from system A
    audio2_path="system_b_response.wav",  # Response from system B
    instruction_path="original_instruction.wav",  # The instruction both systems responded to
    system_prompt="Compare which response better follows the given instruction.",
    model="gpt-4o-audio-preview"
)

print(f"Better response: {result['response']}")
```

### 2. Pointwise Audio Evaluation

Evaluate a single audio file:

```python
result = judge.judge_audio_pointwise(
    audio_path="speech.wav",
    system_prompt="Rate the speech quality from 1-10.",
    model="gpt-4o-audio-preview"
)

print(f"Quality rating: {result['response']}")
```

## In-Context Learning

Improve model performance by providing examples:

### Pairwise Examples

```python
from audiojudge.utils import AudioExample

# Create examples
examples = [
    AudioExample(
        audio1_path="example1_good.wav",
        audio2_path="example1_bad.wav",
        output="Audio 1 is better quality with clearer speech."
        # Optional: instruction_path="instruction1.wav"  # For instruction-based evaluation
    ),
    AudioExample(
        audio1_path="example2_noisy.wav",
        audio2_path="example2_clean.wav",
        output="Audio 2 is better due to less background noise."
    )
]

# Use examples in evaluation
result = judge.judge_audio(
    audio1_path="test1.wav",
    audio2_path="test2.wav",
    system_prompt="Compare audio quality and choose the better one.",
    examples=examples,
    model="gpt-4o-audio-preview"
)
```

### Pointwise Examples

```python
from audiojudge.utils import AudioExamplePointwise

examples = [
    AudioExamplePointwise(
        audio_path="high_quality.wav",
        output="9/10 - Excellent clarity and no background noise"
    ),
    AudioExamplePointwise(
        audio_path="medium_quality.wav",
        output="6/10 - Acceptable quality with minor distortions"
    )
]

result = judge.judge_audio_pointwise(
    audio_path="test_audio.wav",
    system_prompt="Rate the audio quality from 1-10 with explanation.",
    examples=examples,
    model="gpt-4o-audio-preview"
)
```

## Audio Concatenation Methods

Control how audio files are combined for model input:

### Available Methods

**For Pairwise Evaluation:**
1. **`no_concatenation`**: Keep all audio files separate
2. **`pair_example_concatenation`**: Concatenate each example pair
3. **`examples_concatenation`**: Concatenate all examples into one file
4. **`test_concatenation`**: Concatenate test audio pair
5. **`examples_and_test_concatenation`** (default): Concatenate all examples and test audio - shown as the most effective prompting strategy

**For Pointwise Evaluation:**
1. **`no_concatenation`** (default): Keep all audio files separate
2. **`examples_concatenation`**: Concatenate all examples into one file

### Example Usage

```python
# Pairwise: Keep everything separate
result = judge.judge_audio(
    audio1_path="test1.wav",
    audio2_path="test2.wav",
    system_prompt="Compare these audio clips.",
    concatenation_method="no_concatenation"
)

# Pairwise: Concatenate all for better context (recommended)
result = judge.judge_audio(
    audio1_path="test1.wav",
    audio2_path="test2.wav",
    system_prompt="Compare these audio clips.",
    examples=examples,
    concatenation_method="examples_and_test_concatenation"
)

# Pointwise: With example concatenation
result = judge.judge_audio_pointwise(
    audio_path="test.wav",
    system_prompt="Rate the audio quality from 1-10.",
    examples=pointwise_examples,
    concatenation_method="examples_concatenation"
)
```

## Instruction Audio

Use audio files as instructions for more complex tasks:

### With Examples

```python
# Examples with instruction audio
examples = [
    AudioExample(
        audio1_path="example1.wav",
        audio2_path="example2.wav",
        instruction_path="instruction_example.wav",
        output="Audio 1 follows the instruction better."
    )
]

result = judge.judge_audio(
    audio1_path="test1.wav",
    audio2_path="test2.wav",
    instruction_path="instruction.wav",
    system_prompt="Follow the audio instruction to evaluate these clips.",
    examples=examples,
    model="gpt-4o-audio-preview"
)
```

## Supported Models

### OpenAI Models
- `gpt-4o-audio-preview` (recommended)
- `gpt-4o-mini-audio-preview`

### Google Models
- `gemini-1.5-flash`
- `gemini-2.0-flash`
- `gemini-2.5-flash`

```python
# Using different models
result_gpt = judge.judge_audio(
    audio1_path="test1.wav",
    audio2_path="test2.wav",
    system_prompt="Compare quality.",
    model="gpt-4o-audio-preview"
)

result_gemini = judge.judge_audio(
    audio1_path="test1.wav",
    audio2_path="test2.wav",
    system_prompt="Compare quality.",
    model="gemini-2.0-flash"
)
```

## Caching

AudioJudge includes intelligent caching to reduce API costs and improve performance:

### Cache Management

```python
# Clear entire cache
judge.clear_cache()

# Clear only failed (None) responses
valid_entries = judge.clear_none_cache()
print(f"Kept {valid_entries} valid cache entries")

# Get cache statistics
stats = judge.get_cache_stats()
print(f"Cache entries: {stats['total_entries']}")
```

### Cache Configuration

```python
# Disable caching
judge = AudioJudge(disable_cache=True)

# Custom cache directory and expiration
judge = AudioJudge(
    cache_dir="my_audio_cache",
    cache_expire_seconds=86400  # 1 day
)
```

## Advanced Usage

### Error Handling

```python
result = judge.judge_audio(
    audio1_path="test1.wav",
    audio2_path="test2.wav",
    system_prompt="Compare these audio clips."
)

if result["success"]:
    response = result["response"]
    model_used = result["model"]
    print(f"Success with {model_used}: {response}")
else:
    error_message = result["error"]
    print(f"Evaluation failed: {error_message}")
```

### Temperature and Token Control

```python
# Deterministic output
result = judge.judge_audio(
    audio1_path="test1.wav",
    audio2_path="test2.wav",
    system_prompt="Compare quality.",
    temperature=0.000001,
    max_tokens=100
)

# More creative output
result = judge.judge_audio(
    audio1_path="test1.wav",
    audio2_path="test2.wav",
    system_prompt="Describe these audio clips creatively.",
    temperature=0.8,
    max_tokens=500
)
```

## Best Practices

### 1. System Prompt Design

```python
# Good: Specific and clear
system_prompt = """
You are an audio quality expert. Compare two audio clips and determine which has:
1. Better speech clarity
2. Less background noise  
3. More natural sound

Respond with: "Audio 1" or "Audio 2" followed by your reasoning.
"""

# Avoid: Vague instructions
system_prompt = "Which audio is better?"
```

### 2. Example Selection

```python
# Use diverse, representative examples
examples = [
    AudioExample(
        audio1_path="clear.wav", 
        audio2_path="muffled.wav", 
        output="Audio 1 - clearer speech"
    ),
    AudioExample(
        audio1_path="noisy.wav", 
        audio2_path="clean.wav", 
        output="Audio 2 - less background noise"
    ),
    AudioExample(
        audio1_path="fast.wav", 
        audio2_path="normal.wav", 
        output="Audio 2 - better pacing"
    )
]
```

### 3. Concatenation Strategy

- Use `no_concatenation` for simple cases or when preserving individual audio quality is crucial
- Use `examples_and_test_concatenation` when you have examples (recommended for best performance)
- Consider model context limits when choosing strategies

### 4. Model Selection

- **GPT-4o Audio**: Best for complex reasoning and detailed analysis
- **Gemini 2.0+**: Good for general comparisons, potentially faster and more cost-effective

## Research and Experiments

This package is based on research in audio evaluation using large audio models. The experimental code and evaluation scripts used in our research are available in the `experiments/` folder for reproducing the result.

### Example Usage

Additional usage examples can be found in the `examples/` folder, which wraps some of our experiments into the package for demonstration:

- **`examples/audiojudge_usage`**: Pairwise comparison without instruction
  - Datasets: somos, thaimos, tmhintq, pronunciation, speed, speaker evaluations
- **`examples/audiojudge_usage_with_instruction`**: Pairwise comparison with instruction audio
  - Datasets: System-level comparisons including ChatbotArena and SpeakBench
- **`examples/audiojudge_usage_pointwise`**: Pointwise evaluation
  - Datasets: somos, thaimos, tmhintq,

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For issues and questions:
- GitHub Issues: [Create an issue](https://github.com/woodygan/audiojudge/issues)