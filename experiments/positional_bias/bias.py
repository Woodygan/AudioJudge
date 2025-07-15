import pandas as pd
import numpy as np


def analyze_audio_swap_predictions(original_file, swapped_file):
    """
    Analyze how predictions change when audio1 and audio2 paths are swapped.

    Args:
        original_file: Path to the original CSV file
        swapped_file: Path to the CSV file with swapped audio paths

    Returns:
        Dictionary with analysis results
    """

    # Read both CSV files
    df_original = pd.read_csv(original_file)
    df_swapped = pd.read_csv(swapped_file)

    print(f"Original file shape: {df_original.shape}")
    print(f"Swapped file shape: {df_swapped.shape}")

    # Create a key for matching rows based on the swapped audio paths
    # In the swapped file, audio1_path corresponds to original audio2_path and vice versa
    df_original["match_key"] = (
        df_original["audio1_path"] + "|" + df_original["audio2_path"]
    )
    df_swapped["match_key"] = (
        df_swapped["audio2_path"] + "|" + df_swapped["audio1_path"]
    )

    # Merge the dataframes on the match key
    merged_df = pd.merge(
        df_original[["match_key", "prediction", "audio1_path", "audio2_path"]],
        df_swapped[["match_key", "prediction"]],
        on="match_key",
        suffixes=("_original", "_swapped"),
        how="inner",
    )

    print(f"Successfully matched {len(merged_df)} rows")

    if len(merged_df) == 0:
        print("No matching rows found. Check if the audio paths are correctly swapped.")
        return None

    # Convert predictions to string to handle any data type issues
    merged_df["prediction_original"] = merged_df["prediction_original"].astype(str)
    merged_df["prediction_swapped"] = merged_df["prediction_swapped"].astype(str)

    # Analysis for prediction "1"
    pred_1_original = merged_df[merged_df["prediction_original"] == "1"]
    pred_1_stays_1 = pred_1_original[pred_1_original["prediction_swapped"] == "1"]

    # Calculate percentages with respect to total datapoints
    pct_1_stays_1_of_total = (len(pred_1_stays_1) / len(merged_df)) * 100
    pct_1_stays_1_of_1s = (
        (len(pred_1_stays_1) / len(pred_1_original)) * 100
        if len(pred_1_original) > 0
        else 0
    )

    # Analysis for prediction "2"
    pred_2_original = merged_df[merged_df["prediction_original"] == "2"]
    pred_2_stays_2 = pred_2_original[pred_2_original["prediction_swapped"] == "2"]

    # Calculate percentages with respect to total datapoints
    pct_2_stays_2_of_total = (len(pred_2_stays_2) / len(merged_df)) * 100
    pct_2_stays_2_of_2s = (
        (len(pred_2_stays_2) / len(pred_2_original)) * 100
        if len(pred_2_original) > 0
        else 0
    )

    # Analysis for prediction "tie"
    pred_tie_original = merged_df[merged_df["prediction_original"] == "tie"]
    pred_tie_stays_tie = pred_tie_original[
        pred_tie_original["prediction_swapped"] == "tie"
    ]

    # Calculate percentages with respect to total datapoints
    pct_tie_stays_tie_of_total = (len(pred_tie_stays_tie) / len(merged_df)) * 100
    pct_tie_stays_tie_of_ties = (
        (len(pred_tie_stays_tie) / len(pred_tie_original)) * 100
        if len(pred_tie_original) > 0
        else 0
    )

    # Overall consistency
    consistent_predictions = merged_df[
        merged_df["prediction_original"] == merged_df["prediction_swapped"]
    ]
    overall_consistency = (len(consistent_predictions) / len(merged_df)) * 100

    # Create 3x3 confusion matrix
    confusion_matrix = pd.crosstab(
        merged_df["prediction_original"],
        merged_df["prediction_swapped"],
        margins=True,
        margins_name="Total",
    )

    # Ensure all prediction types are represented
    prediction_types = ["1", "2", "tie"]
    for pred_type in prediction_types:
        if pred_type not in confusion_matrix.index:
            confusion_matrix.loc[pred_type] = 0
        if pred_type not in confusion_matrix.columns:
            confusion_matrix[pred_type] = 0

    # Reorder to have consistent layout
    confusion_matrix = confusion_matrix.reindex(
        prediction_types + ["Total"], fill_value=0
    )
    confusion_matrix = confusion_matrix.reindex(
        columns=prediction_types + ["Total"], fill_value=0
    )

    # Create percentage matrix (percentages of total datapoints)
    confusion_matrix_pct = confusion_matrix.copy()
    total_samples = len(merged_df)
    for i in confusion_matrix_pct.index[:-1]:  # Exclude 'Total' row
        for j in confusion_matrix_pct.columns[:-1]:  # Exclude 'Total' column
            confusion_matrix_pct.loc[i, j] = (
                confusion_matrix.loc[i, j] / total_samples
            ) * 100

    # Get distribution of original predictions
    original_pred_counts = merged_df["prediction_original"].value_counts()
    swapped_pred_counts = merged_df["prediction_swapped"].value_counts()

    # Results summary
    results = {
        "total_matched_pairs": len(merged_df),
        "original_pred_1_count": len(pred_1_original),
        "pred_1_stays_1_count": len(pred_1_stays_1),
        "pct_1_stays_1_of_total": pct_1_stays_1_of_total,
        "pct_1_stays_1_of_1s": pct_1_stays_1_of_1s,
        "original_pred_2_count": len(pred_2_original),
        "pred_2_stays_2_count": len(pred_2_stays_2),
        "pct_2_stays_2_of_total": pct_2_stays_2_of_total,
        "pct_2_stays_2_of_2s": pct_2_stays_2_of_2s,
        "original_pred_tie_count": len(pred_tie_original),
        "pred_tie_stays_tie_count": len(pred_tie_stays_tie),
        "pct_tie_stays_tie_of_total": pct_tie_stays_tie_of_total,
        "pct_tie_stays_tie_of_ties": pct_tie_stays_tie_of_ties,
        "overall_consistency_pct": overall_consistency,
        "confusion_matrix": confusion_matrix,
        "confusion_matrix_pct": confusion_matrix_pct,
        "original_distribution": original_pred_counts,
        "swapped_distribution": swapped_pred_counts,
    }

    # Print detailed results
    print("\n" + "=" * 60)
    print("AUDIO SWAP PREDICTION ANALYSIS RESULTS")
    print("=" * 60)

    print(f"\nTotal matched pairs: {results['total_matched_pairs']}")
    print(f"Overall consistency: {results['overall_consistency_pct']:.2f}%")

    print(f"\n--- Original Prediction Distribution ---")
    for pred, count in results["original_distribution"].items():
        pct = (count / results["total_matched_pairs"]) * 100
        print(f"'{pred}': {count} ({pct:.1f}%)")

    print(f"\n--- Prediction Consistency Analysis ---")
    print("(Percentages are calculated with respect to TOTAL datapoints)")

    print(f"\n• Prediction '1' Analysis:")
    print(f"  - Original '1' predictions: {results['original_pred_1_count']}")
    print(f"  - Still '1' after swap: {results['pred_1_stays_1_count']}")
    print(f"  - % of total that stayed '1': {results['pct_1_stays_1_of_total']:.2f}%")
    print(f"  - % of '1's that stayed '1': {results['pct_1_stays_1_of_1s']:.2f}%")

    print(f"\n• Prediction '2' Analysis:")
    print(f"  - Original '2' predictions: {results['original_pred_2_count']}")
    print(f"  - Still '2' after swap: {results['pred_2_stays_2_count']}")
    print(f"  - % of total that stayed '2': {results['pct_2_stays_2_of_total']:.2f}%")
    print(f"  - % of '2's that stayed '2': {results['pct_2_stays_2_of_2s']:.2f}%")

    print(f"\n• Prediction 'tie' Analysis:")
    print(f"  - Original 'tie' predictions: {results['original_pred_tie_count']}")
    print(f"  - Still 'tie' after swap: {results['pred_tie_stays_tie_count']}")
    print(
        f"  - % of total that stayed 'tie': {results['pct_tie_stays_tie_of_total']:.2f}%"
    )
    print(
        f"  - % of 'tie's that stayed 'tie': {results['pct_tie_stays_tie_of_ties']:.2f}%"
    )

    print(f"\n--- 3x3 CONFUSION MATRIX ---")
    print("Rows = Before Swap | Columns = After Swap")
    print("\nCounts:")
    print(results["confusion_matrix"])

    print(f"\nPercentages (% of total {results['total_matched_pairs']} datapoints):")
    # Format the percentage matrix for better display
    pct_matrix_display = results["confusion_matrix_pct"].copy()
    for i in pct_matrix_display.index[:-1]:
        for j in pct_matrix_display.columns[:-1]:
            pct_matrix_display.loc[i, j] = f"{pct_matrix_display.loc[i, j]:.1f}%"
    print(pct_matrix_display.iloc[:-1, :-1])  # Exclude totals for percentage display

    # Calculate and display diagonal consistency
    diagonal_consistency = 0
    for pred_type in ["1", "2", "tie"]:
        if (
            pred_type in results["confusion_matrix"].index
            and pred_type in results["confusion_matrix"].columns
        ):
            diagonal_consistency += results["confusion_matrix"].loc[
                pred_type, pred_type
            ]

    print(f"\nKey Insights:")
    print(f"• Diagonal sum (consistent predictions): {diagonal_consistency}")
    print(f"• Overall consistency rate: {results['overall_consistency_pct']:.1f}%")

    return results, merged_df


# Example usage:
if __name__ == "__main__":
    # Replace these with your actual file paths
    original_file_path = "../main_experiments/results/thaimos/gpt-4o-audio-preview/thaimos_standard_cot_0_shots_none_transcript_single_turn_fewshot_separate_test_separate_gpt_4o_audio_preview.csv"
    swapped_file_path = "../main_experiments/results_swapped/thaimos/gpt-4o-audio-preview/thaimos_standard_cot_0_shots_none_transcript_single_turn_fewshot_separate_test_separate_gpt_4o_audio_preview.csv"

    try:
        results, merged_data = analyze_audio_swap_predictions(
            original_file_path, swapped_file_path
        )

    except FileNotFoundError as e:
        print(f"File not found: {e}")
        print("Please make sure both CSV files exist and the paths are correct.")
    except Exception as e:
        print(f"An error occurred: {e}")
