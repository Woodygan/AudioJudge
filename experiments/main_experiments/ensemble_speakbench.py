import pandas as pd
import numpy as np
import os
import json
import datetime
import argparse
from collections import defaultdict
from typing import Dict, List, Tuple, Set, Optional, Any


def create_output_directory():
    """
    Create a timestamped output directory within the ensemble_results folder.

    Returns:
        str: Path to the created output directory
    """
    # Create base directory if it doesn't exist
    base_dir = "ensemble_results"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    # Create timestamped directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_dir, f"detailed_ensemble_{timestamp}")
    os.makedirs(output_dir)

    return output_dir


def load_detailed_csv_files(file_paths: List[str]):
    """
    Load multiple detailed CSV files from evaluation results.

    Args:
        file_paths (list): List of paths to the detailed CSV files

    Returns:
        tuple: (List of dataframes, Set of all unique IDs, Set of all models)
    """
    dataframes = []
    all_ids = set()
    all_models = set()

    print("Loading detailed CSV files...")

    for file_path in file_paths:
        try:
            # Load the CSV file
            df = pd.read_csv(file_path)

            # Check for required columns
            required_columns = [
                "instruction_id",
                "model_name",
                "position1_prediction",
                "position2_prediction",
            ]

            if not all(col in df.columns for col in required_columns):
                print(f"Warning: Required columns not found in {file_path}. Skipping.")
                continue

            # Add to the list of dataframes
            dataframes.append(df)

            # Add IDs to the set of all IDs
            all_ids.update(df["instruction_id"].tolist())

            # Add models to the set of all models
            all_models.update(df["model_name"].tolist())

        except Exception as e:
            print(f"Error loading {file_path}: {e}")

    print(
        f"Loaded {len(dataframes)} CSV files with {len(all_ids)} unique data points across {len(all_models)} models."
    )

    return dataframes, all_ids, all_models


def convert_prediction_to_numeric(pred):
    """
    Convert string prediction to numeric value.
    '1' -> 0.0 (model wins against gpt4o)
    'tie' -> 0.5
    '2' -> 1.0 (gpt4o wins against model)

    Args:
        pred (str): Prediction string ('1', '2', or 'tie')

    Returns:
        float: Numeric value of prediction
    """
    if pred == "1":
        return 0.0
    elif pred == "tie":
        return 0.5
    elif pred == "2":
        return 1.0
    else:
        print(f"Warning: Unexpected prediction value: {pred}. Treating as missing.")
        return None


def convert_numeric_to_prediction(value):
    """
    Convert numeric value to prediction string.
    < 0.5 -> '1' (model wins)
    = 0.5 -> 'tie'
    > 0.5 -> '2' (gpt4o wins)

    Args:
        value (float): Numeric value

    Returns:
        str: Prediction string
    """
    if value < 0.5:
        return "1"
    elif value > 0.5:
        return "2"
    else:
        return "tie"


def convert_prediction_to_result(prediction, position):
    """
    Convert a prediction ('1', '2', 'tie') to a result ('win', 'loss', 'tie').

    Args:
        prediction (str): The prediction string
        position (int): Position 1 or 2 (determines how to interpret the prediction)

    Returns:
        str: Result as 'win', 'loss', or 'tie'
    """
    if position == 1:
        # Position 1: model_audio = 1, gpt4o_audio = 2
        return {
            "1": "win",  # model wins if 1 is selected
            "2": "loss",  # model loses if 2 is selected
            "tie": "tie",  # tie remains tie
        }.get(prediction, "unknown")
    else:
        # Position 2: gpt4o_audio = 1, model_audio = 2
        return {
            "1": "loss",  # model loses if 1 is selected
            "2": "win",  # model wins if 2 is selected
            "tie": "tie",  # tie remains tie
        }.get(prediction, "unknown")


def ensemble_predictions(dataframes, all_ids, all_models):
    """
    Create ensemble predictions for position 1 and position 2 separately.

    Args:
        dataframes (list): List of dataframes
        all_ids (set): Set of all unique instruction IDs
        all_models (set): Set of all model names

    Returns:
        tuple: (Ensemble predictions DataFrame, Position 1 predictions, Position 2 predictions, Coverage stats)
    """
    # Dictionary to store all predictions for each (ID, model) pair
    position1_predictions = defaultdict(list)
    position2_predictions = defaultdict(list)

    # Track coverage statistics
    coverage_stats = {
        "total_pairs": len(all_ids) * len(all_models),
        "pairs_with_predictions": 0,
        "file_coverage": {},
    }

    # Initialize file coverage counters
    for i, df in enumerate(dataframes):
        coverage_stats["file_coverage"][f"file_{i + 1}"] = 0

    # Collect all predictions from all dataframes
    for i, df in enumerate(dataframes):
        # Count covered pairs in this file
        covered_pairs = 0

        for _, row in df.iterrows():
            id_value = row["instruction_id"]
            model_name = row["model_name"]

            # Create a key for this ID-model pair
            pair_key = (id_value, model_name)

            # Get position 1 prediction
            pos1_pred = row["position1_prediction"]
            numeric_pos1 = convert_prediction_to_numeric(pos1_pred)
            if numeric_pos1 is not None:
                position1_predictions[pair_key].append(numeric_pos1)
                covered_pairs += 1

            # Get position 2 prediction
            pos2_pred = row["position2_prediction"]
            numeric_pos2 = convert_prediction_to_numeric(pos2_pred)
            if numeric_pos2 is not None:
                position2_predictions[pair_key].append(numeric_pos2)

        coverage_stats["file_coverage"][f"file_{i + 1}"] = covered_pairs

    # Calculate ensemble predictions
    ensemble_results = []
    position1_ensembles = {}
    position2_ensembles = {}

    for id_value in all_ids:
        for model_name in all_models:
            pair_key = (id_value, model_name)

            # Get predictions for this pair
            pos1_preds = position1_predictions.get(pair_key, [])
            pos2_preds = position2_predictions.get(pair_key, [])

            # Only process if we have predictions for both positions
            if pos1_preds and pos2_preds:
                # Calculate average for each position
                avg_pos1 = sum(pos1_preds) / len(pos1_preds)
                avg_pos2 = sum(pos2_preds) / len(pos2_preds)

                # Convert back to prediction strings
                ensemble_pos1 = convert_numeric_to_prediction(avg_pos1)
                ensemble_pos2 = convert_numeric_to_prediction(avg_pos2)

                # Convert to results (win/loss/tie)
                result_pos1 = convert_prediction_to_result(ensemble_pos1, 1)
                result_pos2 = convert_prediction_to_result(ensemble_pos2, 2)

                # Store the ensemble results
                position1_ensembles[pair_key] = ensemble_pos1
                position2_ensembles[pair_key] = ensemble_pos2

                # Add to results list
                ensemble_results.append(
                    {
                        "instruction_id": id_value,
                        "model_name": model_name,
                        "position1_prediction": ensemble_pos1,
                        "position1_result": result_pos1,
                        "position2_prediction": ensemble_pos2,
                        "position2_result": result_pos2,
                        "num_pos1_voters": len(pos1_preds),
                        "num_pos2_voters": len(pos2_preds),
                    }
                )

    # Update coverage statistics
    coverage_stats["pairs_with_predictions"] = len(ensemble_results)

    # Convert to DataFrame
    ensemble_df = pd.DataFrame(ensemble_results)

    return ensemble_df, position1_ensembles, position2_ensembles, coverage_stats


def calculate_model_metrics(ensemble_df):
    """
    Calculate metrics for each model based on ensemble predictions.

    Args:
        ensemble_df (pandas.DataFrame): DataFrame with ensemble predictions

    Returns:
        tuple: (DataFrame with model metrics, Dictionary with model metrics)
    """
    # Group by model_name
    model_metrics = {}

    for model_name, group in ensemble_df.groupby("model_name"):
        total_samples = len(group)

        # Position 1 metrics
        win_count_pos1 = (group["position1_result"] == "win").sum()
        tie_count_pos1 = (group["position1_result"] == "tie").sum()
        loss_count_pos1 = (group["position1_result"] == "loss").sum()

        win_rate_pos1 = win_count_pos1 / total_samples if total_samples > 0 else 0
        tie_rate_pos1 = tie_count_pos1 / total_samples if total_samples > 0 else 0
        loss_rate_pos1 = loss_count_pos1 / total_samples if total_samples > 0 else 0

        # Position 2 metrics
        win_count_pos2 = (group["position2_result"] == "win").sum()
        tie_count_pos2 = (group["position2_result"] == "tie").sum()
        loss_count_pos2 = (group["position2_result"] == "loss").sum()

        win_rate_pos2 = win_count_pos2 / total_samples if total_samples > 0 else 0
        tie_rate_pos2 = tie_count_pos2 / total_samples if total_samples > 0 else 0
        loss_rate_pos2 = loss_count_pos2 / total_samples if total_samples > 0 else 0

        # Combined metrics
        avg_win_rate = (win_rate_pos1 + win_rate_pos2) / 2
        avg_tie_rate = (tie_rate_pos1 + tie_rate_pos2) / 2
        avg_loss_rate = (loss_rate_pos1 + loss_rate_pos2) / 2

        # Effective win rate (counting ties as half wins)
        effective_win_rate = avg_win_rate + (avg_tie_rate / 2)

        model_metrics[model_name] = {
            "model_name": model_name,
            "total_samples": total_samples,
            # Position 1 metrics
            "win_count_pos1": win_count_pos1,
            "tie_count_pos1": tie_count_pos1,
            "loss_count_pos1": loss_count_pos1,
            "win_rate_pos1": win_rate_pos1,
            "tie_rate_pos1": tie_rate_pos1,
            "loss_rate_pos1": loss_rate_pos1,
            # Position 2 metrics
            "win_count_pos2": win_count_pos2,
            "tie_count_pos2": tie_count_pos2,
            "loss_count_pos2": loss_count_pos2,
            "win_rate_pos2": win_rate_pos2,
            "tie_rate_pos2": tie_rate_pos2,
            "loss_rate_pos2": loss_rate_pos2,
            # Combined metrics
            "avg_win_rate": avg_win_rate,
            "avg_tie_rate": avg_tie_rate,
            "avg_loss_rate": avg_loss_rate,
            "effective_win_rate": effective_win_rate,
        }

    # Convert to DataFrame
    metrics_df = pd.DataFrame(list(model_metrics.values()))

    # Sort by effective win rate
    metrics_df = metrics_df.sort_values(
        "effective_win_rate", ascending=False
    ).reset_index(drop=True)

    return metrics_df, model_metrics


def save_results(output_dir, ensemble_df, metrics_df, file_paths, coverage_stats):
    """
    Save ensemble results and metadata to files.

    Args:
        output_dir (str): Path to output directory
        ensemble_df (pandas.DataFrame): DataFrame with ensemble predictions
        metrics_df (pandas.DataFrame): DataFrame with model metrics
        file_paths (list): List of paths to input CSV files
        coverage_stats (dict): Dictionary with coverage statistics

    Returns:
        tuple: Paths to saved files
    """
    # Create output paths
    ensemble_path = os.path.join(output_dir, "ensemble_predictions.csv")
    metrics_path = os.path.join(output_dir, "model_metrics.csv")
    metadata_path = os.path.join(output_dir, "ensemble_metadata.json")

    # Save ensemble predictions
    ensemble_df.to_csv(ensemble_path, index=False)
    print(f"Ensemble predictions saved to {ensemble_path}")

    # Save model metrics
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Model metrics saved to {metrics_path}")

    # Prepare metadata
    metadata = {
        "timestamp": datetime.datetime.now().isoformat(),
        "input_files": [file_path for file_path in file_paths],
        "num_files": len(file_paths),
        "num_unique_instruction_ids": len(ensemble_df["instruction_id"].unique()),
        "num_models": len(ensemble_df["model_name"].unique()),
        "coverage_stats": coverage_stats,
    }

    # Save metadata to JSON
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Metadata saved to {metadata_path}")

    return ensemble_path, metrics_path, metadata_path


def print_results_summary(metrics_df):
    """
    Print a summary of model metrics.

    Args:
        metrics_df (pandas.DataFrame): DataFrame with model metrics
    """
    print("\nModel Performance Summary:")
    print("-" * 80)
    print(
        f"{'Model':<30} | {'Eff Win Rate':<12} | {'Avg Win Rate':<12} | {'Avg Tie Rate':<12}"
    )
    print("-" * 80)

    for _, row in metrics_df.iterrows():
        model_name = row["model_name"]
        effective_win_rate = row["effective_win_rate"]
        avg_win_rate = row["avg_win_rate"]
        avg_tie_rate = row["avg_tie_rate"]

        print(
            f"{model_name:<30} | {effective_win_rate:>10.2%}  | {avg_win_rate:>10.2%}  | {avg_tie_rate:>10.2%}"
        )

    print("-" * 80)
    print(f"Models sorted by effective win rate (win + tie/2)")
    print(f"Total models evaluated: {len(metrics_df)}")
    print(f"Number of samples per model: {metrics_df.iloc[0]['total_samples']}")


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Ensemble detailed evaluation results from multiple CSV files."
    )
    parser.add_argument(
        "--files",
        nargs="+",
        required=True,
        help="List of detailed CSV files to ensemble",
    )

    args = parser.parse_args()

    # Normalize and validate file paths
    file_paths = [file_path for file_path in args.files]
    valid_file_paths = [
        file_path for file_path in file_paths if os.path.exists(file_path)
    ]

    if not valid_file_paths:
        print("Error: No valid file paths provided.")
        return

    # Create output directory
    output_dir = create_output_directory()
    print(f"Created output directory: {output_dir}")

    # Load CSV files
    dataframes, all_ids, all_models = load_detailed_csv_files(valid_file_paths)

    if not dataframes:
        print("Error: No valid CSV files loaded.")
        return

    # Create ensemble predictions
    ensemble_df, pos1_ensembles, pos2_ensembles, coverage_stats = ensemble_predictions(
        dataframes, all_ids, all_models
    )

    # Calculate model metrics
    metrics_df, model_metrics = calculate_model_metrics(ensemble_df)

    # Save results
    save_results(output_dir, ensemble_df, metrics_df, valid_file_paths, coverage_stats)

    # Print summary
    print_results_summary(metrics_df)


# For easy configuration without command line arguments
def run_with_config():
    """
    Run the ensemble process with hardcoded configuration.
    Edit the variables below to match your setup.
    """
    # ======== CONFIGURATION (EDIT THESE VALUES) ========
    # List of detailed CSV files to ensemble (add or remove as needed)
    valid_file_paths = [
        "results/speakbench/gpt-4o-audio-preview/detailed_speakbench_lexical_cot_4_shots_none_transcript_single_turn_gpt_4o_audio_preview_20250613_100255.csv",
        "results/speakbench/gpt-4o-audio-preview/detailed_speakbench_paralinguistic_cot_4_shots_none_transcript_single_turn_gpt_4o_audio_preview_20250614_013126.csv",
        "results/speakbench/gpt-4o-audio-preview/detailed_speakbench_speech_quality_cot_4_shots_none_transcript_single_turn_gpt_4o_audio_preview_20250614_092118.csv",
    ]
    # =================================================

    # Create output directory
    output_dir = create_output_directory()
    print(f"Created output directory: {output_dir}")

    # Load CSV files
    dataframes, all_ids, all_models = load_detailed_csv_files(valid_file_paths)

    if not dataframes:
        print("Error: No valid CSV files loaded.")
        return

    # Create ensemble predictions
    ensemble_df, pos1_ensembles, pos2_ensembles, coverage_stats = ensemble_predictions(
        dataframes, all_ids, all_models
    )

    # Calculate model metrics
    metrics_df, model_metrics = calculate_model_metrics(ensemble_df)

    # Save results
    save_results(output_dir, ensemble_df, metrics_df, valid_file_paths, coverage_stats)

    # Print summary
    print_results_summary(metrics_df)


if __name__ == "__main__":
    # Choose one of the following methods to run:

    # Method 1: Using command line arguments
    # Example: python ensemble_detailed.py --files file1.csv file2.csv file3.csv
    # main()

    # Method 2: Using hardcoded configuration (uncomment to use)
    run_with_config()
