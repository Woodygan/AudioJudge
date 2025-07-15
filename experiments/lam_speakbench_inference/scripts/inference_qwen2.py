import os
import argparse
import random
import torch
import json
import librosa
import numpy as np
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
from tqdm import tqdm
from datasets import load_dataset
from pydub import AudioSegment


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct")
model = Qwen2AudioForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-Audio-7B-Instruct"
)
model = model.to(device)


@torch.no_grad()
def run_inference(conversation):
    text = processor.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=False
    )
    audios = []
    for message in conversation:
        if isinstance(message["content"], list):
            for ele in message["content"]:
                if ele["type"] == "audio":
                    audios.append(
                        librosa.load(
                            ele["audio_url"],
                            sr=processor.feature_extractor.sampling_rate,
                        )[0]
                    )
    inputs = processor(
        text=text,
        audios=audios,
        return_tensors="pt",
        padding=True,
        sampling_rate=processor.feature_extractor.sampling_rate,
    ).to("cuda")
    generate_ids = model.generate(**inputs, max_length=6000, do_sample=False)
    generate_ids = generate_ids[:, inputs.input_ids.size(1) :]
    response = processor.batch_decode(
        generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    return response


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
        txt_file = f"{output_dir}/text/{id}.txt"
        # check if the transcript file already exists
        if os.path.exists(txt_file):
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
        response = run_inference(conversation)
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

    # usage: python inference_qwen2.py --output_dir experiments/advvoiceq1/qwen2 --randomize


if __name__ == "__main__":
    main()
