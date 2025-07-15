import os
import argparse
import random
import json
from tqdm import tqdm
import io
import google.generativeai as genai
from pydub import AudioSegment

system_prompt = """You are a helpful assistant. You provide answers to user instructions. The instructions will be in the audio format. Please listen to the instruction and provide an appropriate response. If users request you to speak in a specific style or tone, please behave accordingly."""


def convert_to_16kHz_bytes(audio_path):
    # Load the audio file
    audio = AudioSegment.from_file(audio_path)

    # Check if resampling is necessary
    if audio.frame_rate != 16000:
        print(f"Resampling from {audio.frame_rate} Hz to 16 kHz...")
        audio = audio.set_frame_rate(16000)
    else:
        print("Audio is already 16 kHz.")

    # Export the audio to an in-memory buffer in WAV format
    output_buffer = io.BytesIO()
    audio.export(output_buffer, format="wav")
    output_buffer.seek(0)  # Reset the buffer to the beginning

    # Return the binary data equivalent to pathlib.Path().read_bytes()
    return output_buffer.read()


def experiment(
    model_name,
    output_dir,
):
    print("-----------------------------")
    print("model_name:", model_name)
    print("output_path:", output_dir)
    print("-----------------------------")

    # Initialize the Gemini model
    # model_name = "gemini-2.0-flash"
    GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
    print("GOOGLE_API_KEY:", GOOGLE_API_KEY)
    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel(f"models/{model_name}")

    # load dataset
    with open("data/questions1_shuffled_id.json", "r") as f:
        tmp_dataset = json.load(f)
    print("len(dataset):", len(tmp_dataset))

    ids = [i for i in range(len(tmp_dataset))]
    # random.shuffle(ids)

    for id in tqdm(ids):
        txt_file = f"{output_dir}/text/{id}.txt"
        # check if the transcript file already exists
        if os.path.exists(txt_file):
            print(f"Skipping {id}")
            continue

        # question = x['question']
        question_wav_path = f"data/questions1_kokoro_wav/{id}.kokoro.wav"
        assert os.path.exists(question_wav_path)
        audio_question = {
            "mime_type": "audio/wav",
            "data": convert_to_16kHz_bytes(question_wav_path),
        }

        # Generate the response
        response = model.generate_content(
            [
                system_prompt,
                "This is the user question in the audio format.",
                audio_question,
            ]
        )
        response = response.text

        with open(txt_file, "w") as f:
            f.write(response)
        print("Text:", response)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name", type=str, required=True, help="Specify the model name to run."
    )
    parser.add_argument("--output_dir", type=str, required=True, help="Output Dir")
    args = parser.parse_args()
    experiment(args.model_name, args.output_dir)

    # AudioIn TextOut

    # usage: python inference_geminitext.py --model_name gemini-2.0-flash --output_dir experiments/advvoiceq1/gemini2flash-api
    # usage: python inference_geminitext.py --model_name gemini-1.5-flash --output_dir experiments/advvoiceq1/gemini15flash-api


if __name__ == "__main__":
    main()
