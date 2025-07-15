import pandas as pd
import numpy as np


def analyze_positional_bias(csv_file_path):
    """
    Analyze positional bias in a combined CSV file with position1 and position2 predictions.

    Args:
        csv_file_path: Path to the CSV file containing combined position results

    Returns:
        Dictionary with detailed bias analysis results
    """

    # Read the CSV file
    df = pd.read_csv(csv_file_path)

    print(f"Loaded CSV with {len(df)} rows")
    print(f"Columns: {list(df.columns)}")

    # Check if required columns exist
    required_columns = ["position1_prediction", "position2_prediction"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # Convert predictions to string to handle any data type issues
    df["position1_prediction"] = df["position1_prediction"].astype(str)
    df["position2_prediction"] = df["position2_prediction"].astype(str)

    # Remove any rows with NaN predictions
    df_clean = df.dropna(subset=["position1_prediction", "position2_prediction"])
    print(f"After removing NaN predictions: {len(df_clean)} rows")

    # Define bias cases: both predictions are "1" or both are "2"
    both_predict_1 = (df_clean["position1_prediction"] == "1") & (
        df_clean["position2_prediction"] == "1"
    )
    both_predict_2 = (df_clean["position1_prediction"] == "2") & (
        df_clean["position2_prediction"] == "2"
    )
    consistent_predictionsAB = (df_clean["position1_prediction"] == "1") & (
        df_clean["position2_prediction"] == "2"
    )
    consistent_predictionsBA = (df_clean["position1_prediction"] == "2") & (
        df_clean["position2_prediction"] == "1"
    )
    # Count bias cases
    both_1_count = both_predict_1.sum()
    both_2_count = both_predict_2.sum()
    consistent_AB_count = consistent_predictionsAB.sum()
    consistent_BA_count = consistent_predictionsBA.sum()
    total_cases = (
        both_1_count + both_2_count + consistent_AB_count + consistent_BA_count
    )
    print(
        f"Bias percentage where both predictions are '1': {both_1_count / len(df_clean) * 100:.2f}%)"
    )
    print(
        f"Bias percentage where both predictions are '2': {both_2_count / len(df_clean) * 100:.2f}%)"
    )
    print(both_1_count, both_2_count, consistent_AB_count, consistent_BA_count)


# Example usage:
if __name__ == "__main__":
    # Replace with your actual file path
    csv_file_path = "../main_experiments/results/speakbench/gpt-4o-audio-preview/detailed_speakbench_standard_cot_0_shots_none_transcript_single_turn_gpt_4o_audio_preview_20250518_002348.csv"

    try:
        analyze_positional_bias(csv_file_path)

    except FileNotFoundError as e:
        print(f"File not found: {e}")
        print("Please make sure the CSV file exists and the path is correct.")
    except Exception as e:
        print(f"An error occurred: {e}")
