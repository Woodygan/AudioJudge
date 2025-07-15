import os
import argparse
import random
import torch
import json
import numpy as np
from glob import glob
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
model = model.to(device)


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
    transcript_paths = sorted(
        glob("experiments/advvoiceq1/asr_google/transcript/*.txt")
    )
    print("len(transcript_paths):", len(transcript_paths))

    if randomize:
        random.shuffle(transcript_paths)

    for transcript_path in tqdm(transcript_paths):
        id = int(transcript_path.split("/")[-1].replace(".txt", ""))
        output_file = f"{output_dir}/{id}.txt"
        # check if the transcript file already exists
        if os.path.exists(output_file):
            print(f"Skipping {id}")
            continue

        with open(transcript_path, "r") as f:
            instruction = f.read()

        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant. You provide answers to user instructions.",
            },
            {"role": "user", "content": instruction},
        ]

        inputs = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        )
        inputs = inputs.to(device)
        outputs = model.generate(
            inputs=inputs,
            max_new_tokens=2048,
            do_sample=False,
        )
        len_input = inputs.shape[1]
        outputs = outputs[:, len_input:]
        text = tokenizer.batch_decode(
            outputs, add_special_tokens=False, skip_special_tokens=True
        )[0]
        # save text output
        with open(output_file, "w") as f:
            f.write(text)
        print("TextOutput:", text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True, help="Output Dir")
    parser.add_argument(
        "--randomize", action="store_true", help="Randomize the order of the dataset"
    )
    args = parser.parse_args()
    experiment(args.output_dir, args.randomize)

    # usage: python inference_textllama.py --output_dir experiments/advvoiceq1/asr_google/transcript_llama


if __name__ == "__main__":
    main()
