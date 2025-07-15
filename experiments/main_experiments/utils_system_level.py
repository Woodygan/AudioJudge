import json
import base64
import os
import time
from pathlib import Path
import wave
import pandas as pd
import numpy as np
from openai import OpenAI
from tqdm import tqdm
import json
from typing import Counter, Dict, List, Tuple, Any, Optional
import re
from api_cache import api_cache
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import seaborn as sns
from pydub import AudioSegment
from system_prompts import SYSTEM_PROMPTS
from utils import (
    get_model_response,
    encode_audio_file,
    extract_json_from_response,
    load_dataset,
)
from helper import get_transcript, get_asr_transcription
from utils_speakbench_gemini import get_prompt_gemini
from utils_speakbench import get_prompt
from utils_speakbench_gemini import get_prompt_gemini


def evaluate_model_comparisons(
    df: pd.DataFrame,
    prompt_type: str,
    model: str,
    dataset_name: str,
    n_samples: int = 100,
    n_shots: int = 0,
    result_dir: str = "results",
    transcript_type: str = "none",
    concat_fewshot: bool = False,
    concat_test: bool = False,
    two_turns: bool = False,
    aggregate_fewshot: bool = False,
) -> Dict:
    """
    Evaluate different models against gpt4o-audio with position swapping to avoid positional bias

    Parameters:
    - df: DataFrame containing the dataset with different model outputs as columns
    - prompt_type: Type of prompt to use for the judge
    - model: Model to use as the judge
    - dataset_name: Name of the dataset
    - n_samples: Number of samples to evaluate
    - n_shots: Number of examples to use for few-shot prompting
    - result_dir: Directory to save results
    - transcript_type: Type of transcription to use ('none', 'groundtruth', 'asr')
    - concat_fewshot: Whether to concatenate few-shot example audio files
    - concat_test: Whether to concatenate test audio files
    - two_turns: Whether to send each audio file in a separate message turn
    - aggregate_fewshot: Whether to aggregate all few-shot examples into a single audio file

    Returns:
    - Dictionary with win rates for each model compared to gpt4o-audio
    """
    try:
        # Take a sample of the dataset
        sample_df = df.head(min(n_samples, len(df)))

        # Identify all model columns (any column ending with .wav that's not instruction or gpt4o-audio)
        model_columns = [
            col
            for col in sample_df.columns
            if col not in ["index", "instruction_text", "instruction_path"]
        ]

        print(f"Found {len(model_columns)} models to compare: {model_columns}")

        # Initialize results structure for each model
        model_results = {model_col: [] for model_col in model_columns}

        # Process each sample for each model
        for _, row in tqdm(
            sample_df.iterrows(),
            total=len(sample_df),
            desc=f"Evaluating models with {prompt_type} judge, {n_shots} shots",
        ):
            instruction_path = row["instruction_path"]
            index = row["index"]
            instruction_text = row["instruction_text"]
            gpt4o_audio_path = row["gpt4o-audio"]

            if gpt4o_audio_path is None:
                print(
                    f"Missing gpt4o-audio.wav for datapoint {row.get('index', 'unknown')}"
                )
                continue

            # For each model, compare with gpt4o-audio
            for model_col in model_columns:
                model_audio_path = row[model_col]
                if model_audio_path is None:
                    raise ValueError(
                        f"Missing audio for model {model_col} in datapoint {index}"
                    )

                # Position 1: model_audio as audio1, gpt4o_audio as audio2
                if "gpt" in model.lower():
                    messages_pos1 = get_prompt(
                        prompt_type=prompt_type,
                        instruction_path=instruction_path,
                        audio1_path=model_audio_path,
                        audio2_path=gpt4o_audio_path,
                        dataset_name=dataset_name,
                        n_shots=n_shots,
                        transcript_type=transcript_type,
                        concat_fewshot=concat_fewshot,
                        concat_test=concat_test,
                        two_turns=two_turns,
                        aggregate_fewshot=aggregate_fewshot,
                    )
                elif "gemini" in model.lower():
                    messages_pos1 = get_prompt_gemini(
                        prompt_type=prompt_type,
                        instruction_path=instruction_path,
                        audio1_path=model_audio_path,
                        audio2_path=gpt4o_audio_path,
                        dataset_name=dataset_name,
                        n_shots=n_shots,
                        transcript_type=transcript_type,
                        concat_fewshot=concat_fewshot,
                        concat_test=concat_test,
                        two_turns=two_turns,
                        aggregate_fewshot=aggregate_fewshot,
                    )

                response_data_pos1 = get_model_response(model, messages_pos1)

                # Position 2: gpt4o_audio as audio1, model_audio as audio2
                if "gpt" in model.lower():
                    messages_pos2 = get_prompt(
                        prompt_type=prompt_type,
                        instruction_path=instruction_path,
                        audio1_path=gpt4o_audio_path,
                        audio2_path=model_audio_path,
                        dataset_name=dataset_name,
                        n_shots=n_shots,
                        transcript_type=transcript_type,
                        concat_fewshot=concat_fewshot,
                        concat_test=concat_test,
                        two_turns=two_turns,
                        aggregate_fewshot=aggregate_fewshot,
                    )
                elif "gemini" in model.lower():
                    messages_pos2 = get_prompt_gemini(
                        prompt_type=prompt_type,
                        instruction_path=instruction_path,
                        audio1_path=gpt4o_audio_path,
                        audio2_path=model_audio_path,
                        dataset_name=dataset_name,
                        n_shots=n_shots,
                        transcript_type=transcript_type,
                        concat_fewshot=concat_fewshot,
                        concat_test=concat_test,
                        two_turns=two_turns,
                        aggregate_fewshot=aggregate_fewshot,
                    )

                response_data_pos2 = get_model_response(model, messages_pos2)

                # Skip if any response failed
                if response_data_pos1 is None or response_data_pos2 is None:
                    print(f"Failed to get response for {model_col} vs gpt4o-audio")
                    continue

                _, prediction_text_pos1 = response_data_pos1
                _, prediction_text_pos2 = response_data_pos2

                prediction_json_pos1 = extract_json_from_response(prediction_text_pos1)
                prediction_json_pos2 = extract_json_from_response(prediction_text_pos2)

                if prediction_json_pos1 is None or prediction_json_pos2 is None:
                    print(
                        f"Failed to extract JSON from response for {model_col} vs gpt4o-audio"
                    )
                    continue

                prediction_pos1 = prediction_json_pos1.get("label", None)
                prediction_pos2 = prediction_json_pos2.get("label", None)

                reasoning_pos1 = prediction_json_pos1.get("reasoning", "")
                reasoning_pos2 = prediction_json_pos2.get("reasoning", "")

                if prediction_pos1 is None or prediction_pos2 is None:
                    print(
                        f"No prediction field in extracted JSON for {model_col} vs gpt4o-audio"
                    )
                    continue

                # Interpret results - invert position 2 results since positions are swapped
                # Position 1: model_audio = 1, gpt4o_audio = 2
                # Position 2: gpt4o_audio = 1, model_audio = 2
                result_pos1 = {
                    "1": "win",  # model wins if 1 is selected
                    "2": "loss",  # model loses if 2 is selected
                    "tie": "tie",  # tie remains tie
                }.get(prediction_pos1, "unknown")

                result_pos2 = {
                    "1": "loss",  # model loses if 1 is selected
                    "2": "win",  # model wins if 2 is selected
                    "tie": "tie",  # tie remains tie
                }.get(prediction_pos2, "unknown")

                # Record both positions' results
                model_results[model_col].append(
                    {
                        "instruction_id": index,
                        "instruction_text": instruction_text,
                        "model_name": model_col.replace(".wav", ""),
                        "model_audio_path": model_audio_path,
                        "gpt4o_audio_path": gpt4o_audio_path,
                        "position1_result": result_pos1,
                        "position1_prediction": prediction_pos1,
                        "position1_reasoning": reasoning_pos1,
                        "position2_result": result_pos2,
                        "position2_prediction": prediction_pos2,
                        "position2_reasoning": reasoning_pos2,
                    }
                )

        # Compute metrics for each model
        final_results = {}
        all_results_df = pd.DataFrame()

        for model_col, results in model_results.items():
            model_name = model_col.replace(".wav", "")

            if not results:
                # No results for this model
                final_results[model_name] = {
                    "model_name": model_name,
                    "win_rate_pos1": 0.0,
                    "win_rate_pos2": 0.0,
                    "average_win_rate": 0.0,
                    "tie_rate": 0.0,
                    "samples_evaluated": 0,
                }
                continue

            # Convert results to DataFrame for easier analysis
            model_df = pd.DataFrame(results)

            # Calculate metrics
            win_count_pos1 = (model_df["position1_result"] == "win").sum()
            win_count_pos2 = (model_df["position2_result"] == "win").sum()
            tie_count_pos1 = (model_df["position1_result"] == "tie").sum()
            tie_count_pos2 = (model_df["position2_result"] == "tie").sum()

            total_samples = len(model_df)

            win_rate_pos1 = win_count_pos1 / total_samples if total_samples > 0 else 0
            win_rate_pos2 = win_count_pos2 / total_samples if total_samples > 0 else 0
            tie_rate_pos1 = tie_count_pos1 / total_samples if total_samples > 0 else 0
            tie_rate_pos2 = tie_count_pos2 / total_samples if total_samples > 0 else 0

            # Average across positions for fair comparison
            avg_win_rate = (win_rate_pos1 + win_rate_pos2) / 2
            avg_tie_rate = (tie_rate_pos1 + tie_rate_pos2) / 2

            # Calculate the effective win rate with ties (counting ties as half wins)
            effective_win_rate = avg_win_rate + (avg_tie_rate / 2)

            model_metrics = {
                "model_name": model_name,
                "win_rate_pos1": win_rate_pos1,
                "win_rate_pos2": win_rate_pos2,
                "average_win_rate": avg_win_rate,
                "effective_win_rate": effective_win_rate,
                "tie_rate": avg_tie_rate,
                "samples_evaluated": total_samples,
            }

            final_results[model_name] = model_metrics

            # Add to the combined results DataFrame
            model_df["model_name"] = model_name
            all_results_df = pd.concat([all_results_df, model_df], ignore_index=True)

        # Save detailed results
        os.makedirs(result_dir, exist_ok=True)

        # Create a descriptive filename that includes configuration
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        config_desc = (
            f"{dataset_name}_{prompt_type}_{n_shots}_shots_{transcript_type}_transcript_"
            f"{'two_turns' if two_turns else 'single_turn'}_"
            f"{model.replace('-', '_')}_{timestamp}"
        )

        # Save detailed results for each datapoint
        detailed_results_path = os.path.join(result_dir, f"detailed_{config_desc}.csv")
        all_results_df.to_csv(detailed_results_path, index=False)

        # Save summary metrics for each model
        summary_df = pd.DataFrame(list(final_results.values()))
        summary_results_path = os.path.join(result_dir, f"summary_{config_desc}.csv")
        summary_df.to_csv(summary_results_path, index=False)

        # Print summary
        print(
            f"\nResults Summary ({len(summary_df)} models evaluated against gpt4o-audio):"
        )
        print(
            f"Configuration: {prompt_type} judge, {n_shots} shots, {transcript_type} transcript"
        )
        print("\nWin Rates vs gpt4o-audio (average of both positions):")

        # Sort by average win rate for display
        for model, metrics in sorted(
            final_results.items(), key=lambda x: x[1]["average_win_rate"], reverse=True
        ):
            print(
                f"  {model}: {metrics['average_win_rate']:.2%} win rate ({metrics['tie_rate']:.2%} tie rate)"
            )

        return final_results

    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        import traceback

        traceback.print_exc()
        return {"error": str(e)}
