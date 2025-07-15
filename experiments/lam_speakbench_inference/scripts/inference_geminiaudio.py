import os
import io
import argparse
import random
import json
from tqdm import tqdm
import asyncio
import wave
import contextlib
from google import genai
from datasets import load_dataset


# Define a context manager for opening a wave file.
@contextlib.contextmanager
def wave_file(filename, channels=1, rate=24000, sample_width=2):
    """Context manager that opens a WAV file for writing."""
    wf = wave.open(filename, "wb")
    try:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(rate)
        yield wf
    finally:
        wf.close()


# Helper async enumerator for asynchronous iterables.
async def async_enumerate(it):
    n = 0
    async for item in it:
        yield n, item
        n += 1


# Text In Speech Out
async def generate_audio(
    message: str,
    file_name: str,
):
    """
    Connects to the live session using the Gemini model,
    sends a message, and writes the returned audio data into a WAV file.
    Parameters:
        file_name (str): The output filename for the WAV file.
        message (str): The message to send to the session.
    """
    # Initialize the client with the required HTTP options.
    client = genai.Client(http_options={"api_version": "v1alpha"})

    # Define the generation configuration.
    config = {"generation_config": {"response_modalities": ["AUDIO"]}}

    MODEL = "gemini-2.0-flash-exp"

    # Connect to the live session.
    async with client.aio.live.connect(model=MODEL, config=config) as session:
        # print("> ", message, "\n")
        # Send the message and mark end-of-turn.
        await session.send(input=message, end_of_turn=True)

        # Open the WAV file for writing.
        with wave_file(file_name) as wav:
            # Get the stream of responses.
            turn = session.receive()
            async for n, response in async_enumerate(turn):
                if response.data is not None:
                    wav.writeframes(response.data)

                    # For the first response, print out the MIME type.
                    if n == 0:
                        mime_type = response.server_content.model_turn.parts[
                            0
                        ].inline_data.mime_type
                        # print("MIME Type:", mime_type)
                    # print('.', end='')  # Print progress dots.


# standard synchronous entry point that calls the asynchronous function.
def generate_audio_sync(message: str, output_file: str):
    # result = await generate_audio("Hello? Gemini are you there?")
    _ = asyncio.run(generate_audio(message, file_name=output_file))
    print("Audio saved to", output_file)


system_prompt = """You are a helpful assistant. You provide answers to user instructions. The instructions will be in the audio format. Please listen to the instruction and provide an appropriate response. If users request you to speak in a specific style or tone, please behave accordingly."""


def experiment(
    output_dir,
):
    print("-----------------------------")
    print("output_path:", output_dir)
    print("-----------------------------")

    # load dataset
    dataset = load_dataset("potsawee/speecheval-advanced-v1")["train"]
    ids = [i for i in range(len(dataset))]
    # random.shuffle(ids)

    for id in tqdm(ids):
        wav_file = f"{output_dir}/audio/{id}.wav"
        # check if the transcript file already exists
        if os.path.exists(wav_file):
            print(f"Skipping {id}")
            continue

        question_txt = dataset[id]["instruction"]
        generate_audio_sync(question_txt, wav_file)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True, help="Output Dir")
    args = parser.parse_args()
    experiment(args.output_dir)

    # usage: python inference_geminiaudio.py --output_dir experiments/advvoiceq1/gemini2flash


if __name__ == "__main__":
    main()
