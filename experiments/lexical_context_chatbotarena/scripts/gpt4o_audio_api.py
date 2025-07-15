import base64
import io
import os
import pdb
from datasets import load_dataset
import soundfile as sf
from pydub import AudioSegment
from openai import OpenAI

# Initialize OpenAI client with API key
# Replace with your own OpenAI API key for authentication
API_KEY = ""
client = OpenAI(api_key=API_KEY)


def get_encoded_audio_from_path(filepath):
    """
    Encodes an audio file to a Base64 string.

    Args:
        filepath (str): Path to the audio file.

    Returns:
        str: Base64-encoded string of the audio file.
    """
    with open(filepath, "rb") as audio_file:
        return base64.b64encode(audio_file.read()).decode("utf-8")


def encode_audio_with_resampling(filepath, target_sample_rate=16000):
    try:
        # Load the audio file
        audio = AudioSegment.from_file(filepath)

        # Check the current sample rate
        if audio.frame_rate != target_sample_rate:
            print(f"Resampling from {audio.frame_rate} Hz to {target_sample_rate} Hz")
            audio = audio.set_frame_rate(target_sample_rate)

        # Export the audio to a temporary file or bytes
        audio_bytes = audio.export(format="wav").read()

        # Encode as Base64
        return base64.b64encode(audio_bytes).decode("utf-8")
    except Exception as e:
        print(f"Error processing audio: {e}")
        return None


def get_encoded_audio_from_array(audio_array, sample_rate):
    """
    Encodes a NumPy audio array to a Base64 string.

    Args:
        audio_array (numpy.ndarray): Audio data as a NumPy array.
        sample_rate (int): Sample rate of the audio.

    Returns:
        str: Base64-encoded string of the audio array.
    """
    # Save the audio array to a buffer in WAV format
    buffer = io.BytesIO()
    sf.write(buffer, audio_array, samplerate=sample_rate, format="WAV")
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")


def gpt4o_mos(local_audio_file_path):
    """
    Sends an audio file to GPT-4o for quality assessment and receives a MOS score.

    Args:
        local_audio_file_path (str): Path to the local audio file.

    Returns:
        str: A floating-point MOS score (e.g., "4.5").
    """
    # Encode audio file to Base64
    encoded_audio = get_encoded_audio_from_path(local_audio_file_path)

    # Send request to GPT-4o for audio evaluation
    completion = client.chat.completions.create(
        model="gpt-4o-audio-preview-2024-12-17",
        modalities=["text", "audio"],
        audio={"voice": "alloy", "format": "wav"},
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Evaluate the quality of this audio and assign a score between 1.0 and 5.0. "
                            "Focus on clarity, fidelity, and overall listening experience. "
                            "Output strictly in floating-point format, e.g., [[4.5]]."
                        ),
                    },
                    {
                        "type": "input_audio",
                        "input_audio": {"data": encoded_audio, "format": "wav"},
                    },
                ],
            }
        ],
    )
    # Extract and return the MOS score from the response
    return completion.choices[0].message.audio.transcript.strip("[]")


def ab_testing(encoded_audio_a, encoded_audio_b):
    """
    Compares two audio files and determines which one has better quality.

    Args:
        encoded_audio_a (str): Base64-encoded string of Audio A.
        encoded_audio_b (str): Base64-encoded string of Audio B.

    Returns:
        str: Decision on which audio is better ("[[Audio A]]" or "[[Audio B]]").
    """
    # Send request to GPT-4o for A/B testing
    completion = client.chat.completions.create(
        model="gpt-4o-audio-preview-2024-12-17",
        modalities=["text", "audio"],
        audio={"voice": "alloy", "format": "wav"},
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "You are an expert in audio quality assessment. "
                            "Evaluate two audio files (Audio A and Audio B) for clarity, fidelity, and overall quality. "
                            "Decide which one is better with a strict response: [[Audio A]] or [[Audio B]]."
                        ),
                    },
                    {
                        "type": "input_audio",
                        "input_audio": {"data": encoded_audio_a, "format": "wav"},
                    },
                    {
                        "type": "input_audio",
                        "input_audio": {"data": encoded_audio_b, "format": "wav"},
                    },
                ],
            }
        ],
    )
    # Extract and return the A/B testing decision from the response
    return completion.choices[0].message.audio.transcript.strip()


def evaluate_better_audio(ds_row):
    """
    Evaluates which audio in a dataset row has better quality.

    Args:
        ds_row (dict): A single row from the Hugging Face dataset containing
                       'path_chosen' and 'path_rejected' audio data.

    Returns:
        str: Decision on which audio is better ("[[Audio A]]" or "[[Audio B]]").
    """
    chosen_audio = ds_row["path_chosen"]
    rejected_audio = ds_row["path_rejected"]

    # Encode chosen and rejected audio to Base64
    chosen_encoded = get_encoded_audio_from_array(
        chosen_audio["array"], chosen_audio["sampling_rate"]
    )
    rejected_encoded = get_encoded_audio_from_array(
        rejected_audio["array"], rejected_audio["sampling_rate"]
    )

    # Perform A/B testing and return the result
    return ab_testing(chosen_encoded, rejected_encoded)


def main():
    """
    Main function to evaluate audio quality using GPT-4o.
    Loads a dataset and evaluates the better audio for a single example.
    """
    # Load the dataset from Hugging Face
    ds = load_dataset("scb10x/tts_arena_resynthesized")

    # Example for MOS scoring
    # result = gpt4o_mos('decoded_audio.wav')
    # print("gpt-4o-audio-preview-2024-12-17 score: ", result)

    # Example for A/B testing
    # result = evaluate_better_audio(ds['train'][1])
    # print("Better audio decision:", result)


if __name__ == "__main__":
    main()
