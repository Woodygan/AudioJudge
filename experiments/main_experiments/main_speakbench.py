import argparse
import json
import base64
import os
import time
from pathlib import Path
import pandas as pd
import numpy as np
from openai import OpenAI
from tqdm import tqdm
import json
from typing import Dict, List, Tuple, Any, Optional
import re
from utils import get_prompt, load_dataset, evaluate_prompt_strategy
from generate_graph import generate_comprehensive_heatmap
from dotenv import load_dotenv
from api_cache import (
    api_cache,
    clear_cache,
    clear_none_cache,
)  # Import the caching decorator
from utils_speakbench import evaluate_prompt_strategy_speakbench
from utils_system_level import evaluate_model_comparisons


def main(args):
    """
    Main function for evaluating model comparisons against gpt4o-audio.

    Args:
        args: Command-line arguments with dataset_name, model, and other configurations
    """
    # Path to the dataset
    dataset_path = os.path.join("datasets", f"{args.dataset_name}_dataset.json")

    # Load dataset
    df = load_dataset(dataset_path)
    print(f"Loaded dataset with {len(df)} examples")

    model = args.model  # Model to use as judge
    dataset = args.dataset_name
    result_dir = os.path.join("results", f"{args.dataset_name}", f"{args.model}")
    os.makedirs(result_dir, exist_ok=True)

    # Limit the number of samples to evaluate
    n_samples = min(args.n_samples if hasattr(args, "n_samples") else 100, len(df))

    # Define configurations for testing
    n_shots_list = [0, 4] if args.n_shots is None else [args.n_shots]
    prompt_types = (
        args.prompt_types if hasattr(args, "prompt_types") else ["standard_cot"]
    )

    # Configurations
    TRANSCRIPT_TYPES = ["none"]
    FEWSHOT_CONFIGS = {
        "aggregate": {"aggregate_fewshot": True, "concat_fewshot": False},
        # "concat": {"aggregate_fewshot": False, "concat_fewshot": True},
        # "separate": {"aggregate_fewshot": False, "concat_fewshot": False}
    }
    TEST_CONFIGS = [True]  # False = separate audio files
    TWO_TURNS_CONFIGS = [False]  # Single turn by default

    all_results = []

    for prompt_type in prompt_types:
        print(f"\nEvaluating with judge prompt strategy: {prompt_type}")

        for transcript_type in TRANSCRIPT_TYPES:
            for two_turns in TWO_TURNS_CONFIGS:
                if two_turns and "gemini" in model:
                    print("Skipping two_turns for Gemini judge model")
                    continue

                for fewshot_config_name, fewshot_params in FEWSHOT_CONFIGS.items():
                    for concat_test in TEST_CONFIGS:
                        for n_shots in n_shots_list:
                            # Build configuration description
                            config_desc = (
                                f"{n_shots} shots, {transcript_type} transcript, "
                                f"{fewshot_config_name} few-shot, "
                                f"{'two_turns' if two_turns else 'single_turn'}, "
                                f"{'concat' if concat_test else 'separate'} test"
                            )
                            print(f"  With {config_desc}")

                            # Run model comparison
                            results = evaluate_model_comparisons(
                                df=df,
                                prompt_type=prompt_type,
                                model=model,
                                dataset_name=dataset,
                                n_samples=n_samples,
                                n_shots=n_shots,
                                result_dir=result_dir,
                                transcript_type=transcript_type,
                                concat_fewshot=fewshot_params["concat_fewshot"],
                                aggregate_fewshot=fewshot_params["aggregate_fewshot"],
                                concat_test=concat_test,
                                two_turns=two_turns,
                            )

                            # Add configuration info to results
                            results_with_config = {
                                "prompt_type": prompt_type,
                                "model": model,
                                "dataset_name": dataset,
                                "n_shots": n_shots,
                                "transcript_type": transcript_type,
                                "concat_fewshot": fewshot_params["concat_fewshot"],
                                "aggregate_fewshot": fewshot_params[
                                    "aggregate_fewshot"
                                ],
                                "concat_test": concat_test,
                                "two_turns": two_turns,
                                "model_results": results,
                            }

                            all_results.append(results_with_config)

    # Convert all results to a DataFrame for easier analysis
    flattened_results = []
    for config in all_results:
        config_info = {k: v for k, v in config.items() if k != "model_results"}
        for model_name, model_metrics in config["model_results"].items():
            result_entry = {**config_info, **model_metrics}
            flattened_results.append(result_entry)

    results_df = pd.DataFrame(flattened_results)

    # Save overall results
    combined_dir = os.path.join(result_dir, "combined_results")
    os.makedirs(combined_dir, exist_ok=True)

    # Save results DataFrame to CSV
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_df.to_csv(
        f"{combined_dir}/model_comparison_results_{timestamp}.csv", index=False
    )

    # Create a summary and analysis
    print("\n========== FINAL SUMMARY ==========")
    print(f"Dataset: {dataset}")
    print(f"Judge Model: {model}")
    print(f"Samples evaluated: {n_samples}")

    # Group by model and compute mean win rates
    if not results_df.empty:
        model_summary = (
            results_df.groupby("model_name")
            .agg(
                {
                    "average_win_rate": ["mean", "std", "count"],
                    "effective_win_rate": ["mean", "std"],
                    "tie_rate": ["mean"],
                }
            )
            .reset_index()
        )

        print("\nAverage Win Rates vs gpt4o-audio:")
        for _, row in model_summary.sort_values(
            ("average_win_rate", "mean"), ascending=False
        ).iterrows():
            model = row["model_name"]
            win_rate = row[("average_win_rate", "mean")]
            win_std = row[("average_win_rate", "std")]
            tie_rate = row[("tie_rate", "mean")]
            count = row[("average_win_rate", "count")]

            print(
                f"  {model}: {win_rate:.2%} ± {win_std:.2%} win rate ({tie_rate:.2%} tie rate) from {count} configurations"
            )

        # Save summary to file
        model_summary.to_csv(f"{combined_dir}/model_summary_{timestamp}.csv")

    print("\nEvaluation complete! Results saved to:", combined_dir)

    return results_df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate model comparisons against gpt4o-audio"
    )
    parser.add_argument(
        "--dataset_name", type=str, default="speakbench", help="Name of the dataset"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-audio-preview",
        help="Model to use as judge",
    )
    parser.add_argument(
        "--n_samples", type=int, default=100, help="Number of samples to evaluate"
    )
    parser.add_argument(
        "--n_shots",
        type=int,
        default=None,
        help="Number of few-shot examples to use (if None, will try 0, 2, 4)",
    )
    parser.add_argument(
        "--prompt_types",
        nargs="+",
        default=["standard_cot"],
        help="Prompt types to evaluate (e.g., standard_cot, phonetic_cot)",
    )

    args = parser.parse_args()
    main(args)
