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


def get_user_prompt(dataset_name: str) -> str:
    user_prompt = ""
    # Add dataset-specific instructions
    if dataset_name == "pronunciation":
        user_prompt = (
            "Please analyze these two recordings strictly for pronunciation details (phonemes, syllables, stress, emphasis). "
            "Ignore differences solely due to accent. Respond ONLY in text and output valid JSON with keys 'reasoning' and 'match' (boolean)."
        )
    elif dataset_name == "speaker":
        user_prompt = (
            "Please analyze if these two recordings are from the same speaker. "
            "Respond ONLY in text and output valid JSON with keys 'reasoning' and 'match' (boolean)."
        )
    elif dataset_name == "speed":
        user_prompt = (
            "Please analyze which of the two recordings has faster speech. "
            "Respond ONLY in text and output valid JSON with keys 'reasoning' and 'label' (string, either '1' or '2')."
        )
    elif (
        dataset_name == "tmhintq"
        or dataset_name == "somos"
        or dataset_name == "thaimos"
    ):
        user_prompt = (
            "Please analyze which of the two recordings is better (has better speech quality). "
            "Respond ONLY in text and output valid JSON with keys 'reasoning' and 'label' (string, either '1' or '2')."
        )


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
            if args.dataset_name in ["speed", "tmhintq", "somos", "thaimos"]:
                ground_truth = row.get("label")
                prediction_key = "label"
            else:  # pronunciation or speaker
                ground_truth = row.get("match")
                prediction_key = "match"
            audio1_path = os.path.join(
                "..", "experiments", "main_experiments", row["audio1_path"]
            )
            audio2_path = os.path.join(
                "..", "experiments", "main_experiments", row["audio2_path"]
            )
            user_prompt = get_user_prompt(args.dataset_name)
            system_prompt = SYSTEM_PROMPTS[args.dataset_name]["standard_cot"]
            response = audio_judge.judge_audio(
                audio1_path=audio1_path,
                audio2_path=audio2_path,
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                model=args.model,
                examples=examples,
                concatenation_method="examples_and_test_concatenation",
            )

            if not response["success"]:
                print(
                    f"Failed to process {audio1_path}, {audio2_path}: {response.get('error', 'Unknown error')}"
                )
                failed_samples += 1
                continue

            # Extract prediction from response
            response_text = response["response"]
            prediction_json = extract_json_from_response(response_text)

            if prediction_json is None:
                print(
                    f"Failed to extract JSON from response for {audio1_path}, {audio2_path}"
                )
                failed_samples += 1
                continue
            # print(f"Response for {audio1_path}, {audio2_path}: {response_text}")
            prediction = prediction_json.get(prediction_key, None)
            reasoning = prediction_json.get("reasoning", "")

            if prediction is None:
                print(
                    f"No '{prediction_key}' field in extracted JSON for {audio1_path}, {audio2_path}"
                )
                failed_samples += 1
                continue
            if ground_truth == prediction:
                correct_predictions += 1

            total_predictions += 1

            # Store result
            results.append(
                {
                    "audio1_path": audio1_path,
                    "audio2_path": audio2_path,
                    "ground_truth": ground_truth,
                    "prediction": prediction,
                    "reasoning": reasoning,
                    "correct": ground_truth == prediction,
                    "dataset_name": args.dataset_name,
                    "model": args.model,
                    "n_few_shots": args.n_few_shots,
                }
            )

        except Exception as e:
            print(f"Error processing {audio1_path}, {audio2_path}: {str(e)}")
            failed_samples += 1
            continue
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0

    # Print results
    print(f"\n=== Results ===")
    print(f"Dataset: {args.dataset_name}")
    print(f"Model: {args.model}")
    print(f"Few-shot examples: {args.n_few_shots}")
    print(f"Correct predictions: {correct_predictions}/{total_predictions}")
    print(f"Accuracy: {accuracy:.4f}")

    return accuracy


def parse_args():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(description="Audio evaluation using AudioJudge")
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        choices=["pronunciation", "speaker", "speed", "tmhintq", "somos", "thaimos"],
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
        "--n_samples",
        type=int,
        default=200,
        help="Maximum number of samples to evaluate",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    accuracy = main(args)
