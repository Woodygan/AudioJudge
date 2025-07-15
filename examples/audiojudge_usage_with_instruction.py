"""
Refactored audio evaluation system using AudioJudge framework
"""

import argparse
import json
import os
import time
import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Optional, Any, Tuple
from collections import Counter
from pathlib import Path

# Import the AudioJudge class and utilities
from audiojudge import AudioJudge, AudioExample
from dotenv import load_dotenv
from system_prompts import SYSTEM_PROMPTS
import re


def load_dataset(file_path: str) -> pd.DataFrame:
    """
    Load the dataset from a JSON file
    """
    with open(file_path, "r") as f:
        data = json.load(f)
    for item in data:
        if isinstance(item.get("match", None), str):
            item["match"] = item["match"].lower() == "true"
    df = pd.DataFrame(data)
    return df


def get_user_prompt(dataset_name: str) -> str:
    user_prompt = ""
    if "speakbench" in dataset_name:
        user_prompt = (
            "Please analyze which of the two recordings follows the instruction better, or tie. "
            "Respond ONLY in text and output valid JSON with keys 'reasoning' and 'label' (string, '1', '2' or 'tie')."
        )
    elif "chatbotarena" in dataset_name:
        user_prompt = (
            "Please analyze which of the two recordings follows the instruction better, or tie, in terms of content of the responses. "
            "Respond ONLY in text and output valid JSON with keys 'reasoning' and 'label' (string, '1', '2' or 'tie')."
        )
    return user_prompt


def main(args):
    # Initialize the AudioJudge instance
    load_dotenv()
    audio_judge = AudioJudge()
    dataset_path = os.path.join(
        "..",
        "experiments",
        "main_experiments",
        "datasets",
        f"{args.dataset_name}_dataset.json",
    )
    # Load dataset
    df = load_dataset(dataset_path)
    n_samples = min(1, len(df))
    sample_df = df[:n_samples]
    with open("../experiments/main_experiments/few_shots_examples.json", "r") as f:
        few_shots_examples = json.load(f)
    few_shots_examples = few_shots_examples[args.dataset_name][: args.n_few_shots]
    examples = []
    results = []
    for example in few_shots_examples:
        examples.append(
            AudioExample(
                audio1_path=os.path.join(
                    "..", "experiments", "main_experiments", example["audio1_path"]
                ),
                audio2_path=os.path.join(
                    "..", "experiments", "main_experiments", example["audio2_path"]
                ),
                instruction_path=os.path.join(
                    "..", "experiments", "main_experiments", example["instruction_path"]
                ),
                output=json.dumps(example["assistant_message"]),
            )
        )
    failed_samples = 0
    correct_predictions = 0
    total_predictions = 0
    for _, row in tqdm(
        sample_df.iterrows(), total=len(sample_df), desc="Processing samples"
    ):
        try:
            audio1_path = os.path.join(
                "..", "experiments", "main_experiments", row["audio1_path"]
            )
            audio2_path = os.path.join(
                "..", "experiments", "main_experiments", row["audio2_path"]
            )
            instruction_path = os.path.join(
                "..", "experiments", "main_experiments", row["instruction_path"]
            )
            user_prompt = get_user_prompt(args.dataset_name)
            system_prompt = SYSTEM_PROMPTS[args.dataset_name]["standard_cot"]
            response = audio_judge.judge_audio(
                audio1_path=audio1_path,
                audio2_path=audio2_path,
                instruction_path=instruction_path,
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                model=args.model,
                examples=examples,
                concatenation_method="examples_and_test_concatenation",
            )

            print(f"Model Response: {response['response']}")

        except Exception as e:
            print(f"Error processing {audio1_path}, {audio2_path}: {str(e)}")
            continue


def parse_args():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(description="Audio evaluation using AudioJudge")
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        choices=["chatbotarena", "speakbench508"],
        help="Name of the dataset to evaluate",
    )
    parser.add_argument(
        "--n_few_shots", type=int, default=4, help="Number of few-shot examples to use"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemini-2.5-flash-preview-04-17",
        help="Model to use for evaluation",
    )
    parser.add_argument(
        "--n_samples", type=int, default=1, help="Maximum number of samples to evaluate"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
