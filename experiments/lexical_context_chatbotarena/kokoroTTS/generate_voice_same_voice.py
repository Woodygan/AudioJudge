import os
import argparse
import json
import random
from tqdm import tqdm
import numpy as np
import soundfile as sf
import librosa
from kokoro import KPipeline


def get_silence_audio(duration: float, sr: int):
    # silence_duration = 3.0  # in seconds
    # sr_16k = 16000
    # Generate silence at sr
    silence_audio = np.zeros(int(sr * duration), dtype=np.float32)
    return silence_audio


def synthesize_speech(
    pipeline,
    text: str,
    wav_path: str,  # to save the audio file
):
    # check if wav_path already exists
    if os.path.exists(wav_path):
        print(f"Already exists: {wav_path}")
        return

    # If text is empty, generate silence of 2 seconds (adjust as needed)
    if not text.strip():
        silent_duration = 2.0
        combined_audio_16k = get_silence_audio(silent_duration, 16000)

        # Save the silent audio
        sf.write(wav_path, combined_audio_16k, 16000)
        print(f"No text given. Saved {silent_duration}s of silence to {wav_path}")
        return

    # Generate audio in chunks, but do not write them individually
    generator = pipeline(text, voice="af_bella", speed=1, split_pattern=r"\n+")

    # Collect chunks of audio in a list
    audio_chunks = []
    try:
        for i, (gs, ps, audio) in enumerate(generator):
            # print(i)
            # print(gs)
            # print(ps)

            # Append this chunk to our collection
            audio_chunks.append(audio)

        # Combine all chunks into one NumPy array (at the original 24 kHz sample rate)
        if len(audio_chunks) > 1:
            combined_audio_24k = np.concatenate(audio_chunks)
        else:
            combined_audio_24k = audio_chunks[0]

        # ----- RESAMPLE from 24 kHz to 16 kHz -----
        # Librosa's resampling
        combined_audio_16k = librosa.resample(
            y=combined_audio_24k, orig_sr=24000, target_sr=16000
        )
    except OverflowError as e:
        print(f"OverflowError: {e}")
        combined_audio_16k = get_silence_audio(2.0, 16000)

    # Save the 16 kHz version to a single file
    sf.write(wav_path, combined_audio_16k, 16000)
    print(f"Saved to {wav_path}")


def experiment(
    input_path: str,
    wav_dir: str,
):
    print("-------------------------------------------")
    print("input_path:", input_path)
    print("wav_dir:", wav_dir)
    print("cuda:", os.environ.get("CUDA_VISIBLE_DEVICES"))
    print("-------------------------------------------")

    pipeline = KPipeline(lang_code="a")  # american

    with open(input_path, "r") as f:
        data = json.load(f)
    print("len(data):", len(data))

    random.shuffle(data)

    for x in tqdm(data):
        # Get the analysis result
        assert x["conversation_a"][0]["content"] == x["conversation_b"][0]["content"]
        # user turn
        user_turn_content = x["conversation_a"][0]["content"]
        voice_user_wav_path = os.path.join(wav_dir, f"{x['question_id']}-user.wav")
        synthesize_speech(pipeline, user_turn_content, voice_user_wav_path)

        # assistant turn a
        assistant_turn_a_content = x["conversation_a"][1]["content"]
        voice_assistant_a_wav_path = os.path.join(
            wav_dir, f"{x['question_id']}-assistant-a.wav"
        )
        synthesize_speech(
            pipeline, assistant_turn_a_content, voice_assistant_a_wav_path
        )

        # assistant turn b
        assistant_turn_b_content = x["conversation_b"][1]["content"]
        voice_assistant_b_wav_path = os.path.join(
            wav_dir, f"{x['question_id']}-assistant-b.wav"
        )
        synthesize_speech(
            pipeline, assistant_turn_b_content, voice_assistant_b_wav_path
        )

    print("Done.")


if __name__ == "__main__":
    # Argument parser setup
    parser = argparse.ArgumentParser(
        description="Analyze comparison between A and B from input text."
    )
    parser.add_argument(
        "--input_path",
        type=str,
        default="/data/workspace/ppotsawee/audioLM-as-judge/chatbot-arena/chatbot-arena-spoken-1turn-english-difference-voices.json",
        help="Path to the input file containing text to analyze.",
    )
    parser.add_argument(
        "--wav_dir",
        type=str,
        default="/data/workspace/ppotsawee/audioLM-as-judge/chatbot-arena/kokoroTTS/wav_af_bella",
        help="Path to the output file to save the synthesis results.",
    )
    args = parser.parse_args()
    experiment(args.input_path, args.wav_dir)
