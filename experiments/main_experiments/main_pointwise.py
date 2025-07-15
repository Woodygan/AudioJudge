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
from utils_pointwise import evaluate_prompt_strategy_pointwise

load_dotenv()
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


# Define transcript types and audio configurations
def main(args):
    # Path to the dataset
    dataset_path = os.path.join("datasets", f"{args.dataset_name}_dataset.json")

    # Load dataset
    df = load_dataset(dataset_path)
    print(f"Loaded dataset with {len(df)} examples")

    model = args.model
    dataset = args.dataset_name
    result_dir = os.path.join("results_pointwise", args.dataset_name, args.model)
    os.makedirs(result_dir, exist_ok=True)
    n_samples = min(args.n_samples if hasattr(args, "n_samples") else 500, len(df))
    if dataset == "speakbench508":
        n_samples = min(500, len(df))  # Limit to 500 samples for SpeakBench
    n_shots_list = [0, 4]  # Number of few-shot examples to test
    aggregate_fewshots = [True, False]  # Whether to aggregate few-shot examples
    transcript_types = ["none"]
    # Define prompt types to evaluate
    PROMPT_TYPES = [
        # "no_cot",
        "standard_cot",
        # "phonetic_cot",
        # "syllable_cot",
        # "stress_cot",
        # "vowel_cot",
        # "consonant_cot",
        # "comprehensive_cot",
        # "structured"
    ]
    all_results = []

    # Run evaluations for all configurations
    for prompt_type in PROMPT_TYPES:
        print(f"\nEvaluating prompt strategy: {prompt_type}")

        for n_shots in n_shots_list:
            for transcript_type in transcript_types:
                for aggregate_fewshot in aggregate_fewshots:
                    if n_shots == 0 and aggregate_fewshot:
                        # Skip aggregation for zero-shot case
                        continue
                    config_desc = (
                        f"{n_shots} shots, {transcript_type} transcript, "
                        f"{'aggregate' if aggregate_fewshot else 'separate'} few-shot"
                    )
                    print(f"  With {config_desc}")

                    results = evaluate_prompt_strategy_pointwise(
                        df=df,
                        dataset_name=dataset,
                        prompt_type=prompt_type,
                        model=model,
                        n_samples=n_samples,
                        n_shots=n_shots,
                        result_dir=result_dir,
                        transcript_type=transcript_type,
                        aggregate_fewshot=aggregate_fewshot,
                    )
                    all_results.append(results)

    # Convert results to DataFrame for easier analysis
    results_df = pd.DataFrame(all_results)

    # Save overall results
    combined_dir = os.path.join(result_dir, "combined_results")
    os.makedirs(combined_dir, exist_ok=True)

    # Save results DataFrame to CSV
    results_df.to_csv(
        f"{combined_dir}/all_results_{model.replace('-', '_')}.csv", index=False
    )

    # Create a text file for the analysis output
    analysis_path = f"{combined_dir}/analysis_{model.replace('-', '_')}.txt"

    with open(analysis_path, "w") as f:
        # Write and print prompt strategy comparison
        header = "\nPrompt strategy comparison:"
        print(header)
        f.write(f"{header}\n")

        comparison = results_df.sort_values("accuracy", ascending=False).to_string(
            index=False
        )
        print(comparison)
        f.write(f"{comparison}\n")

        # Identify and write the best strategy based on accuracy
        best_strategy = results_df.loc[results_df["accuracy"].idxmax()]
        best_summary = (
            f"\nBest strategy: {best_strategy['prompt_type']} with {best_strategy['n_shots']} shots, "
            f"{best_strategy['transcript_type']} transcript, "
            f"{'aggregate' if best_strategy['aggregate_fewshot'] else 'separate'} few-shot, "
            f"accuracy {best_strategy['accuracy']:.4f}"
        )
        print(best_summary)
        f.write(f"{best_summary}\n")

        # Report MSE metrics if available
        if "mse_overall" in best_strategy and not pd.isna(best_strategy["mse_overall"]):
            mse_summary = (
                f"\nBest strategy MSE metrics:"
                f"\n  Overall MSE: {best_strategy['mse_overall']:.4f}"
            )
            print(mse_summary)
            f.write(f"{mse_summary}\n")

        viz_msg = f"\nVisualizations saved to {result_dir}/plots/"
        print(viz_msg)
        f.write(f"{viz_msg}\n")

        # Analyze and write impact of each factor on accuracy
        analysis_header = "\nFactor Impact Analysis:"
        print(analysis_header)
        f.write(f"{analysis_header}\n")

        # Impact of transcript type
        transcript_impact = (
            results_df.groupby("transcript_type")["accuracy"]
            .mean()
            .sort_values(ascending=False)
        )
        transcript_header = "\nTranscript Type Impact:"
        print(transcript_header)
        f.write(f"{transcript_header}\n")

        for transcript_type, acc in transcript_impact.items():
            transcript_line = f"  {transcript_type.capitalize() if transcript_type != 'none' else 'No Transcript'}: {acc:.4f}"
            print(transcript_line)
            f.write(f"{transcript_line}\n")

        # Impact of few-shot aggregation
        fewshot_impact = (
            results_df.groupby("aggregate_fewshot")["accuracy"]
            .mean()
            .sort_values(ascending=False)
        )
        fewshot_header = "\nFew-shot Configuration Impact:"
        print(fewshot_header)
        f.write(f"{fewshot_header}\n")

        for aggregate_fewshot, acc in fewshot_impact.items():
            fewshot_line = f"  {'Aggregate' if aggregate_fewshot else 'Separate'} Few-shot: {acc:.4f}"
            print(fewshot_line)
            f.write(f"{fewshot_line}\n")

        # Impact of shots
        shots_impact = (
            results_df.groupby("n_shots")["accuracy"]
            .mean()
            .sort_values(ascending=False)
        )
        shots_header = "\nNumber of Shots Impact:"
        print(shots_header)
        f.write(f"{shots_header}\n")

        for n_shots, acc in shots_impact.items():
            shots_line = f"  {n_shots} shots: {acc:.4f}"
            print(shots_line)
            f.write(f"{shots_line}\n")

        # Impact of prompt type
        prompt_impact = (
            results_df.groupby("prompt_type")["accuracy"]
            .mean()
            .sort_values(ascending=False)
        )
        prompt_header = "\nPrompt Type Impact:"
        print(prompt_header)
        f.write(f"{prompt_header}\n")

        for prompt_type, acc in prompt_impact.items():
            prompt_line = f"  {prompt_type}: {acc:.4f}"
            print(prompt_line)
            f.write(f"{prompt_line}\n")

        # MSE analysis if available
        if (
            "mse_overall" in results_df.columns
            and not results_df["mse_overall"].isna().all()
        ):
            mse_header = "\nMSE Analysis (Lower is Better):"
            print(mse_header)
            f.write(f"{mse_header}\n")

            # Best strategy by MSE
            best_mse_strategy = results_df.loc[results_df["mse_overall"].idxmin()]
            mse_best_summary = (
                f"\nBest MSE strategy: {best_mse_strategy['prompt_type']} with {best_mse_strategy['n_shots']} shots, "
                f"{best_mse_strategy['transcript_type']} transcript, "
                f"{'aggregate' if best_mse_strategy['aggregate_fewshot'] else 'separate'} few-shot"
                f"\n  Overall MSE: {best_mse_strategy['mse_overall']:.4f}"
            )
            print(mse_best_summary)
            f.write(f"{mse_best_summary}\n")

            # Impact of factors on MSE
            mse_transcript_impact = (
                results_df.groupby("transcript_type")["mse_overall"]
                .mean()
                .sort_values()
            )
            mse_transcript_header = "\nTranscript Type Impact on MSE:"
            print(mse_transcript_header)
            f.write(f"{mse_transcript_header}\n")

            for transcript_type, mse in mse_transcript_impact.items():
                if not pd.isna(mse):
                    mse_transcript_line = f"  {transcript_type.capitalize() if transcript_type != 'none' else 'No Transcript'}: {mse:.4f}"
                    print(mse_transcript_line)
                    f.write(f"{mse_transcript_line}\n")

    print(f"\nAnalysis saved to {analysis_path}")


if __name__ == "__main__":
    clear_none_cache()  # Clear cache entries that returned None
    parser = argparse.ArgumentParser(
        description="Evaluate audio dataset with pairwise comparison."
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="somos",
        help="Name of the dataset to evaluate.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-audio-preview",
        help="Model to use for evaluation.",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=200,
        help="Maximum number of samples to evaluate.",
    )
    args = parser.parse_args()
    main(args)
