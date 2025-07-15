import os
import argparse
import random
import torch
import json
from transformers import AutoModel
from tqdm import tqdm
from datasets import load_dataset
from pydub import AudioSegment
import soundfile as sf
import numpy as np

model = AutoModel.from_pretrained(
    "scb10x/llama3.1-typhoon2-audio-8b-instruct",
    torch_dtype=torch.float16,
    trust_remote_code=True,
)
model.to("cuda")

system_prompt = """You are a helpful assistant. You provide answers to user instructions. The instructions will be in the audio format. Please listen to the instruction and provide an appropriate response. If users request you to speak in a specific style or tone, please behave accordingly."""


def experiment(
    output_dir,
    randomize=False,
):
    print("-----------------------------")
    print("output_path:", output_dir)
    print("randomize:", randomize)
    print("type(randomize):", type(randomize))
    print("-----------------------------")

    # load dataset
    with open("data/questions1_shuffled_id.json", "r") as f:
        tmp_dataset = json.load(f)
    print("len(dataset):", len(tmp_dataset))

    ids = [i for i in range(len(tmp_dataset))]

    if randomize:
        random.shuffle(ids)

    for id in tqdm(ids):
        txt_file = f"{output_dir}/transcript/{id}.txt"
        wav_file = f"{output_dir}/audio/{id}.wav"
        # check if the transcript file already exists
        if os.path.exists(txt_file) and os.path.exists(wav_file):
            print(f"Skipping {id}")
            continue

        question_wav_path = f"data/questions1_kokoro_wav/{id}.kokoro.wav"
        assert os.path.exists(question_wav_path)

        # check if the sampling rate is 16kHz
        assert AudioSegment.from_wav(question_wav_path).frame_rate == 16000

        conversation = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "audio",
                        "audio_url": question_wav_path,
                    },
                    {
                        "type": "text",
                        "text": "This is the user question in the audio format. Listen and respond to this question.",
                    },
                ],
            },
        ]
        try:
            output = model.generate(
                conversation=conversation,
                max_new_tokens=500,
                do_sample=True,
                num_beams=1,
                top_p=0.9,
                repetition_penalty=1.0,
                length_penalty=1.0,
                temperature=0.7,
            )
        except RuntimeError:
            output = {
                "text": "I cannot process the audio",
                "audio": {
                    "array": np.zeros((16000 * 5), dtype=np.float32),
                    "sampling_rate": 1600,
                },
            }

        # save transcript
        transcript = output["text"]
        with open(txt_file, "w") as f:
            f.write(transcript)

        # save wav file
        sf.write(wav_file, output["audio"]["array"], output["audio"]["sampling_rate"])

        print("Generated audio:", wav_file)
        print("Transcript:", transcript)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True, help="Output Dir")
    parser.add_argument(
        "--randomize", action="store_true", help="Randomize the order of the dataset"
    )
    args = parser.parse_args()
    experiment(args.output_dir, args.randomize)

    # usage: python inference_typhoon2.py --output_dir experiments/advvoiceq1/typhoon2 --randomize


if __name__ == "__main__":
    main()
