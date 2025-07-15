import os
import argparse
import random
import torch
import json
import librosa
import numpy as np
from transformers import AutoModel
from tqdm import tqdm
from pydub import AudioSegment


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModel.from_pretrained("WillHeld/DiVA-llama-3-v0-8b", trust_remote_code=True)
model.to(device)

prompt = """Please listen to the instruction and provide an appropriate response. If users request you to speak in a specific style or tone, please behave accordingly."""


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
        txt_file = f"{output_dir}/text/{id}.txt"
        # check if the transcript file already exists
        if os.path.exists(txt_file):
            print(f"Skipping {id}")
            continue

        question_wav_path = f"data/questions1_kokoro_wav/{id}.kokoro.wav"
        assert os.path.exists(question_wav_path)

        # check if the sampling rate is 16kHz
        assert AudioSegment.from_wav(question_wav_path).frame_rate == 16000

        speech_data, _ = librosa.load(question_wav_path, sr=16_000)

        response = model.generate([speech_data], [prompt])[0]
        # save text output
        with open(txt_file, "w") as f:
            f.write(response)
        print("TextOutput:", response)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True, help="Output Dir")
    parser.add_argument(
        "--randomize", action="store_true", help="Randomize the order of the dataset"
    )
    args = parser.parse_args()
    experiment(args.output_dir, args.randomize)

    # usage: python inference_diva.py --output_dir experiments/advvoiceq1/diva --randomize


if __name__ == "__main__":
    main()
