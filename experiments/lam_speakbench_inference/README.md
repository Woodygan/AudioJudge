# Experiment: LAM inference (SpeakBench)
- Generate the output cache files (e.g., audio responses) for the LAM SpeakBench inference experiments.

## Structure

### data
- `questions1_shuffle_id.json`: textual instruction 
- `questions1_kokoro_wav`: wav files of the textual instruction, synthesised by kokoroTTS
Note that this data was pushed to HuggingFace at `potsawee/speecheval-advanced-v1`

### scripts
```
inference_{model}.py
```
- `model` to run inference.

- `asr_google_generation.py`: to run ASR using GCP's ASR
- `tts_kokoro_generation.py`: to run TTS using kokoroTTS