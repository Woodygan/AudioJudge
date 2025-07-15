import os
import argparse
import random
from tqdm import tqdm
from openai import OpenAI
from datasets import load_dataset

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

system_prompt = """You are a helpful assistant. You provide answers to user instructions. Please interpret to the instruction and provide an appropriate response."""


def experiment(
    output_dir,
):
    print("-----------------------------")
    print("output_path:", output_dir)
    print("-----------------------------")

    # load dataset
    dataset = load_dataset("potsawee/speecheval-advanced-v1")["train"]

    ids = [i for i in range(len(dataset))]
    random.shuffle(ids)

    for id in tqdm(ids):
        txt_file = f"{output_dir}/text/{id}.txt"
        # check if the transcript file already exists
        if os.path.exists(txt_file):
            print(f"Skipping {id}")
            continue

        x = dataset[id]
        instruction = x["instruction"]
        message = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": instruction},
        ]
        completion = client.chat.completions.create(
            model="gpt-4o-2024-11-20", messages=message
        )

        # save response
        response = completion.choices[0].message.content
        with open(txt_file, "w") as f:
            f.write(response)
        print("Output:", response)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True, help="Output Dir")
    args = parser.parse_args()
    experiment(args.output_dir)

    # usage: python inference_gpttext.py --output_dir experiments/advvoiceq1/gpt4o_text


if __name__ == "__main__":
    main()
