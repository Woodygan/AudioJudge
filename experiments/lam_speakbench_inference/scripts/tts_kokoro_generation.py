import os
import argparse
import json
import random
from tqdm import tqdm
from glob import glob
import numpy as np
import soundfile as sf
import librosa
from kokoro import KPipeline

# add local path to the sys path
# import sys
# sys.path.append("/data/workspace/ppotsawee/audioLM-as-judge/chatbot-arena/kokoroTTS/")


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
        # combined_audio_16k = get_silence_audio(2.0, 16000)
        import ipdb

        ipdb.set_trace()

    # Save the 16 kHz version to a single file
    sf.write(wav_path, combined_audio_16k, 16000)
    print(f"Saved to {wav_path}")


def experiment(
    input_dir: str,
    wav_dir: str,
):
    print("-------------------------------------------")
    print("input_dir:", input_dir)
    print("wav_dir:", wav_dir)
    print("cuda:", os.environ.get("CUDA_VISIBLE_DEVICES"))
    print("-------------------------------------------")

    pipeline = KPipeline(lang_code="a")  # american

    txt_paths = glob(f"{input_dir}/*.txt")
    txt_data = {}
    for txt_path in txt_paths:
        conversation_id = int(txt_path.replace(input_dir, "").split(".")[0].strip("/"))
        with open(txt_path, "r") as f:
            txt = f.read()
        txt = txt.strip().strip('"')
        txt_data[conversation_id] = txt
    print("len(txt_data):", len(txt_data))

    for conversation_id, txt in tqdm(txt_data.items()):
        wav_path = os.path.join(wav_dir, f"{conversation_id}.wav")
        synthesize_speech(pipeline, txt, wav_path)

    print("Done.")


if __name__ == "__main__":
    # Argument parser setup
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        type=str,
        help="Path to the input dir containing files to process. (*.txt files)",
    )
    parser.add_argument(
        "--wav_dir",
        type=str,
        default="experiments/advvoiceq1/typhoon2/transcript_kokoro_wav",
        help="Path to the output file to save the synthesis results. (*.wav files)",
    )
    args = parser.parse_args()
    experiment(args.input_dir, args.wav_dir)
