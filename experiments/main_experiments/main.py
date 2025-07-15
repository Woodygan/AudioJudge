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

load_dotenv()
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


def main(args):
    # Path to the dataset

    dataset_path = os.path.join("datasets", f"{args.dataset_name}_dataset.json")

    # Load dataset
    df = load_dataset(dataset_path)
    # df = df.iloc[1:]
    print(f"Loaded dataset with {len(df)} examples")

    model = args.model
    dataset = args.dataset_name
    result_dir = os.path.join(
        "results", args.dataset_name, args.model
    )  # Directory to save results
    os.makedirs(result_dir, exist_ok=True)
    if dataset == "speakbench508":
        n_samples = min(500, len(df))
    elif "chatbotarena" in dataset:
        n_samples = min(1000, len(df))
    else:
        n_samples = min(200, len(df))

    n_shots_list = [0, 2, 4]  # Including 0 (no examples)

    all_results = []

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
        # "structured",
        # "lexical_cot",
        # "paralinguistic_cot",
        # "speech_quality_cot",
    ]

    # Define majority vote combinations
    """
    MAJORITY_VOTE_CONFIGS = [
        {
            "name": "feature_specific",
            "prompt_types": ["phonetic_cot", "syllable_cot", "stress_cot", "vowel_cot", "consonant_cot"]
        },
        {
            "name": "all_cots",
            "prompt_types": ["standard_cot", "phonetic_cot", "syllable_cot", "stress_cot", "vowel_cot", 
                            "consonant_cot", "comprehensive_cot"]
        }
    ]"""
    MAJORITY_VOTE_CONFIGS = []

    # Define configurations for testing
    # TRANSCRIPT_TYPES = ["groundtruth", "asr"]
    TRANSCRIPT_TYPES = ["none"]
    FEWSHOT_CONFIGS = {
        # "aggregate": {"aggregate_fewshot": True, "concat_fewshot": False},
        # "concat": {"aggregate_fewshot": False, "concat_fewshot": True},
        "separate": {"aggregate_fewshot": False, "concat_fewshot": False}
    }
    TEST_CONFIGS = [True]  # False = separate, True = concatenated
    TWO_TURNS_CONFIGS = [False]
    # 1. Evaluate each individual prompt strategy with varying configurations
    if "speakbench" in dataset or "chatbotarena" in dataset:
        for prompt_type in PROMPT_TYPES:
            print(f"\nEvaluating prompt strategy: {prompt_type}")

            for transcript_type in TRANSCRIPT_TYPES:
                # Test with two_turns as a separate configuration
                for two_turns in TWO_TURNS_CONFIGS:
                    if two_turns:
                        if "gemini" in model:
                            raise ValueError(
                                "Gemini models are not supported for two_turn setting. Please use a compatible model."
                            )
                        if transcript_type != "none":
                            continue
                        # When two_turns is True, other configs don't matter
                        config_desc = (
                            f"{n_shots} shots, {transcript_type} transcript, two_turns"
                        )
                        print(f"  With {config_desc}")

                        for n_shots in n_shots_list:
                            results = evaluate_prompt_strategy_speakbench(
                                df=df,
                                dataset_name=dataset,
                                prompt_type=prompt_type,
                                model=model,
                                n_samples=n_samples,
                                n_shots=n_shots,
                                result_dir=result_dir,
                                transcript_type=transcript_type,
                                concat_fewshot=False,
                                aggregate_fewshot=False,
                                concat_test=False,
                                two_turns=True,
                            )
                            all_results.append(results)
                    else:
                        # When two_turns is False, test the other configurations
                        for (
                            fewshot_config_name,
                            fewshot_params,
                        ) in FEWSHOT_CONFIGS.items():
                            for concat_test in TEST_CONFIGS:
                                if transcript_type != "none" and (
                                    fewshot_config_name != "aggregate"
                                    or not concat_test
                                ):
                                    # continue
                                    pass
                                if fewshot_config_name != "aggregate" and concat_test:
                                    # continue
                                    pass
                                for n_shots in n_shots_list:
                                    # Skip few-shot configurations if n_shots is 0
                                    # if n_shots == 0 and fewshot_config_name != "separate":
                                    # continue

                                    config_desc = (
                                        f"{n_shots} shots, {transcript_type} transcript, "
                                        f"{fewshot_config_name} few-shot, "
                                        f"{'concat' if concat_test else 'separate'} test"
                                    )
                                    print(f"  With {config_desc}")
                                    results = evaluate_prompt_strategy_speakbench(
                                        df=df,
                                        dataset_name=dataset,
                                        prompt_type=prompt_type,
                                        model=model,
                                        n_samples=n_samples,
                                        n_shots=n_shots,
                                        result_dir=result_dir,
                                        transcript_type=transcript_type,
                                        concat_fewshot=fewshot_params["concat_fewshot"],
                                        aggregate_fewshot=fewshot_params[
                                            "aggregate_fewshot"
                                        ],
                                        concat_test=concat_test,
                                        two_turns=False,
                                    )
                                    all_results.append(results)
    else:
        if "gemini" in model:
            raise ValueError(
                "Gemini models are not supported for this dataset. Please use a compatible model."
            )
        for prompt_type in PROMPT_TYPES:
            print(f"\nEvaluating prompt strategy: {prompt_type}")

            for transcript_type in TRANSCRIPT_TYPES:
                # Test with two_turns as a separate configuration
                for two_turns in TWO_TURNS_CONFIGS:
                    if two_turns:
                        if transcript_type != "none":
                            continue
                        # When two_turns is True, other configs don't matter
                        config_desc = (
                            f"{n_shots} shots, {transcript_type} transcript, two_turns"
                        )
                        print(f"  With {config_desc}")

                        for n_shots in n_shots_list:
                            results = evaluate_prompt_strategy(
                                df=df,
                                prompt_type=prompt_type,
                                model=model,
                                dataset_name=dataset,
                                n_samples=n_samples,
                                n_shots=n_shots,
                                result_dir=result_dir,
                                transcript_type=transcript_type,
                                concat_fewshot=False,
                                aggregate_fewshot=False,
                                concat_test=False,
                                two_turns=True,
                            )
                            all_results.append(results)
                    else:
                        # When two_turns is False, test the other configurations
                        for (
                            fewshot_config_name,
                            fewshot_params,
                        ) in FEWSHOT_CONFIGS.items():
                            for concat_test in TEST_CONFIGS:
                                if transcript_type != "none" and (
                                    fewshot_config_name != "aggregate"
                                    or not concat_test
                                ):
                                    pass
                                    # continue
                                for n_shots in n_shots_list:
                                    # Skip few-shot configurations if n_shots is 0
                                    if (
                                        n_shots == 0
                                        and fewshot_config_name != "separate"
                                    ):
                                        continue

                                    config_desc = (
                                        f"{n_shots} shots, {transcript_type} transcript, "
                                        f"{fewshot_config_name} few-shot, "
                                        f"{'concat' if concat_test else 'separate'} test"
                                    )
                                    print(f"  With {config_desc}")

                                    results = evaluate_prompt_strategy(
                                        df=df,
                                        prompt_type=prompt_type,
                                        model=model,
                                        dataset_name=dataset,
                                        n_samples=n_samples,
                                        n_shots=n_shots,
                                        result_dir=result_dir,
                                        transcript_type=transcript_type,
                                        concat_fewshot=fewshot_params["concat_fewshot"],
                                        aggregate_fewshot=fewshot_params[
                                            "aggregate_fewshot"
                                        ],
                                        concat_test=concat_test,
                                        two_turns=False,
                                    )
                                    all_results.append(results)

        # 2. Evaluate with majority voting
        for config in MAJORITY_VOTE_CONFIGS:
            print(f"\nEvaluating majority voting with {config['name']} strategies")

            # Try different transcript types
            for transcript_type in TRANSCRIPT_TYPES:
                # Test with two_turns as a separate configuration
                for two_turns in TWO_TURNS_CONFIGS:
                    if two_turns:
                        # When two_turns is True, other configs don't matter
                        for n_shots in n_shots_list:
                            # Skip few-shot configuration if n_shots is 0
                            if n_shots == 0 and two_turns:
                                continue

                            config_desc = (
                                f"{n_shots} shots, {transcript_type} transcript, "
                                f"two_turns"
                            )
                            print(f"  With {config_desc}")

                            results = evaluate_prompt_strategy(
                                df=df,
                                prompt_type=f"majority_{config['name']}",
                                model=model,
                                dataset_name=dataset,
                                n_samples=n_samples,
                                n_shots=n_shots,
                                result_dir=result_dir,
                                majority_vote=True,
                                vote_prompt_types=config["prompt_types"],
                                transcript_type=transcript_type,
                                concat_fewshot=False,
                                aggregate_fewshot=False,
                                concat_test=False,
                                two_turns=True,
                            )
                            all_results.append(results)
                    else:
                        # When two_turns is False, test the other configurations
                        for (
                            fewshot_config_name,
                            fewshot_params,
                        ) in FEWSHOT_CONFIGS.items():
                            for concat_test in TEST_CONFIGS:
                                for n_shots in n_shots_list:
                                    # Skip few-shot configuration if n_shots is 0
                                    if (
                                        n_shots == 0
                                        and fewshot_config_name != "separate"
                                    ):
                                        continue

                                    config_desc = (
                                        f"{n_shots} shots, {transcript_type} transcript, "
                                        f"{fewshot_config_name} few-shot, "
                                        f"{'concat' if concat_test else 'separate'} test"
                                    )
                                    print(f"  With {config_desc}")

                                    results = evaluate_prompt_strategy(
                                        df=df,
                                        prompt_type=f"majority_{config['name']}",
                                        model=model,
                                        dataset_name=dataset,
                                        n_samples=n_samples,
                                        n_shots=n_shots,
                                        result_dir=result_dir,
                                        majority_vote=True,
                                        vote_prompt_types=config["prompt_types"],
                                        transcript_type=transcript_type,
                                        concat_fewshot=fewshot_params["concat_fewshot"],
                                        aggregate_fewshot=fewshot_params[
                                            "aggregate_fewshot"
                                        ],
                                        concat_test=concat_test,
                                        two_turns=False,
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
        )

        # Add configuration details based on available columns
        if "two_turns" in best_strategy and best_strategy["two_turns"]:
            best_summary += "two turns mode, "
        else:
            if (
                "aggregate_fewshot" in best_strategy
                and best_strategy["aggregate_fewshot"]
            ):
                best_summary += "aggregate few-shot, "
            elif "concat_fewshot" in best_strategy and best_strategy["concat_fewshot"]:
                best_summary += "concat few-shot, "
            else:
                best_summary += "separate few-shot, "

            best_summary += f"{'concatenated' if best_strategy['concat_test'] else 'separate'} test, "

        best_summary += f"accuracy {best_strategy['accuracy']:.4f}"
        print(best_summary)
        f.write(f"{best_summary}\n")

        generate_comprehensive_heatmap(results_df, result_dir, model.replace("-", "_"))

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

        # Impact of two turns (if available)
        if "two_turns" in results_df.columns:
            turns_impact = (
                results_df.groupby("two_turns")["accuracy"]
                .mean()
                .sort_values(ascending=False)
            )
            turns_header = "\nTwo Turns Impact:"
            print(turns_header)
            f.write(f"{turns_header}\n")

            for two_turns, acc in turns_impact.items():
                turns_line = (
                    f"  {'Two Turns' if two_turns else 'Single Turn'}: {acc:.4f}"
                )
                print(turns_line)
                f.write(f"{turns_line}\n")

        # Impact of aggregate/concat few-shot configuration
        if "aggregate_fewshot" in results_df.columns:
            # Create a combined config column for analysis
            fewshot_config_values = []
            for _, row in results_df.iterrows():
                if "aggregate_fewshot" in row and row["aggregate_fewshot"]:
                    fewshot_config_values.append("Aggregate")
                elif "concat_fewshot" in row and row["concat_fewshot"]:
                    fewshot_config_values.append("Concat")
                else:
                    fewshot_config_values.append("Separate")

            fewshot_df = results_df.copy()
            fewshot_df["fewshot_config"] = fewshot_config_values

            fewshot_impact = (
                fewshot_df.groupby("fewshot_config")["accuracy"]
                .mean()
                .sort_values(ascending=False)
            )
            fewshot_header = "\nFew-shot Configuration Impact:"
            print(fewshot_header)
            f.write(f"{fewshot_header}\n")

            for fewshot_config, acc in fewshot_impact.items():
                fewshot_line = f"  {fewshot_config} Few-shot: {acc:.4f}"
                print(fewshot_line)
                f.write(f"{fewshot_line}\n")
        else:
            # Original few-shot impact analysis
            fewshot_impact = (
                results_df.groupby("concat_fewshot")["accuracy"]
                .mean()
                .sort_values(ascending=False)
            )
            fewshot_header = "\nFew-shot Configuration Impact:"
            print(fewshot_header)
            f.write(f"{fewshot_header}\n")

            for concat_fewshot, acc in fewshot_impact.items():
                fewshot_line = f"  {'Concatenated' if concat_fewshot else 'Separate'} Few-shot: {acc:.4f}"
                print(fewshot_line)
                f.write(f"{fewshot_line}\n")

        # Impact of test configuration
        test_impact = (
            results_df.groupby("concat_test")["accuracy"]
            .mean()
            .sort_values(ascending=False)
        )
        test_header = "\nTest Configuration Impact:"
        print(test_header)
        f.write(f"{test_header}\n")

        for concat_test, acc in test_impact.items():
            test_line = (
                f"  {'Concatenated' if concat_test else 'Separate'} Test: {acc:.4f}"
            )
            print(test_line)
            print(test_line)
            f.write(f"{test_line}\n")

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

        top_prompts = list(prompt_impact.items())[:5]  # Top 5 prompt types
        for prompt_type, acc in top_prompts:
            prompt_line = f"  {prompt_type}: {acc:.4f}"
            print(prompt_line)
            f.write(f"{prompt_line}\n")

    print(f"\nAnalysis saved to {analysis_path}")


if __name__ == "__main__":
    clear_none_cache()  # Clear cache entries that returned None
    parser = argparse.ArgumentParser(
        description="Evaluate audio pronunciation dataset with various prompt strategies."
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="pronunciation",
        help="Name of the dataset to evaluate.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-audio-preview",
        help="Model to use for evaluation.",
    )
    args = parser.parse_args()
    main(args)
