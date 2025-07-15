import os
import argparse
import base64
import random
import json
from tqdm import tqdm
from openai import OpenAI
from gpt4o_audio_api import encode_audio_with_resampling

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

system_prompt = """You are a helpful assistant. You provide answers to user instructions. The instructions will be in the audio format. Please listen to the instruction and provide an appropriate response. If users request you to speak in a specific style or tone, please behave accordingly."""


def experiment(
    output_dir,
):
    print("-----------------------------")
    print("output_path:", output_dir)
    print("-----------------------------")

    # load dataset
    with open("data/questions1_shuffled_id.json", "r") as f:
        tmp_dataset = json.load(f)
    print("len(dataset):", len(tmp_dataset))

    ids = [i for i in range(len(tmp_dataset))]
    random.shuffle(ids)

    for id in tqdm(ids):
        txt_file = f"{output_dir}/transcript/{id}.txt"
        wav_file = f"{output_dir}/audio/{id}.wav"
        # check if the transcript file already exists
        if os.path.exists(txt_file) and os.path.exists(wav_file):
            print(f"Skipping {id}")
            continue

        # question = x['question']
        question_wav_path = f"data/questions1_kokoro_wav/{id}.kokoro.wav"
        assert os.path.exists(question_wav_path)
        encoded_audio_question = encode_audio_with_resampling(
            question_wav_path, target_sample_rate=16000
        )

        message = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "This is the user question in the audio format.",
                    },
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": encoded_audio_question,
                            "format": "wav",
                        },
                    },
                ],
            },
        ]
        completion = client.chat.completions.create(
            model="gpt-4o-audio-preview-2024-12-17",
            modalities=["text", "audio"],
            audio={"voice": "alloy", "format": "wav"},
            messages=message,
        )

        # response = completion.choices[0].message.content ---> this one is empty for audio out

        # save transcript
        transcript = completion.choices[0].message.audio.transcript
        with open(txt_file, "w") as f:
            f.write(transcript)

        # save wav file
        wav_bytes = base64.b64decode(completion.choices[0].message.audio.data)
        with open(wav_file, "wb") as f:
            f.write(wav_bytes)
        print("Generated audio:", wav_file)
        print("Transcript:", transcript)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True, help="Output Dir")
    args = parser.parse_args()
    experiment(args.output_dir)

    # usage: python inference_gpt.py --output_dir experiments/advvoiceq1/gpt4o


if __name__ == "__main__":
    main()
