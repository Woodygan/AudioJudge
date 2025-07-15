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
from audiojudge import AudioJudge, AudioExamplePointwise
from dotenv import load_dotenv
from system_prompts_pointwise import SYSTEM_PROMPTS
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


def extract_json_from_response(response: str) -> Optional[Dict]:
    """Extract JSON from the model's response"""
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        json_match = re.search(r"({.*?})", response.replace("\n", " "), re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass
    return None


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
    user_prompt = (
        "Please analyze the speech quality of this recording. "
        "Respond ONLY in text and output valid JSON with key 'score' (int from 1-5)."
    )
    # Load dataset
    df = load_dataset(dataset_path)
    n_samples = min(1, len(df))
    sample_df = df[:n_samples]
    with open(
        "../experiments/main_experiments/few_shots_examples_pointwise.json", "r"
    ) as f:
        few_shots_examples = json.load(f)
    few_shots_examples = few_shots_examples[args.dataset_name][: args.n_few_shots]
    examples = []
    results = []
    for example in few_shots_examples:
        examples.append(
            AudioExamplePointwise(
                audio_path=os.path.join(
                    "..", "experiments", "main_experiments", example["audio_path"]
                ),
                output=json.dumps({"score": example["score"]}),
            )
        )
    for _, row in tqdm(
        sample_df.iterrows(), total=len(sample_df), desc="Processing samples"
    ):
        try:
            audio_path = os.path.join(
                "..", "experiments", "main_experiments", row["audio1_path"]
            )
            system_prompt = SYSTEM_PROMPTS[args.dataset_name]["standard_cot"]
            response = audio_judge.judge_audio_pointwise(
                audio_path=audio_path,
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                model=args.model,
                examples=examples,
                concatenation_method="examples_concatenation",
            )

            print(f"Response: {response}")
        except Exception as e:
            print(f"Error processing {audio_path}: {str(e)}")
            continue
    return


def parse_args():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(description="Audio evaluation using AudioJudge")
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        choices=["tmhintq", "somos", "thaimos"],
        help="Name of the dataset to evaluate",
    )
    parser.add_argument(
        "--n_few_shots", type=int, default=4, help="Number of few-shot examples to use"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-audio-preview",
        help="Model to use for evaluation",
    )
    parser.add_argument(
        "--n_samples", type=int, default=1, help="Maximum number of samples to evaluate"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)