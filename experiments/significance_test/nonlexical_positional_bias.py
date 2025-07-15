import pandas as pd
import numpy as np


def categorize_predictions(original_file, swapped_file):
    """
    Categorize predictions into stable, pos1, pos2 based on audio swap analysis.

    Args:
        original_file: Path to the original CSV file
        swapped_file: Path to the CSV file with swapped audio paths

    Returns:
        Dictionary with categorized indices and counts
    """

    # Read both CSV files
    df_original = pd.read_csv(original_file)
    df_swapped = pd.read_csv(swapped_file)

    print(f"Original file shape: {df_original.shape}")
    print(f"Swapped file shape: {df_swapped.shape}")

    # Create match keys
    df_original["match_key"] = (
        df_original["audio1_path"] + "|" + df_original["audio2_path"]
    )
    df_swapped["match_key"] = (
        df_swapped["audio2_path"] + "|" + df_swapped["audio1_path"]
    )

    # Merge dataframes
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

    # Convert predictions to string
    merged_df["prediction_original"] = merged_df["prediction_original"].astype(str)
    merged_df["prediction_swapped"] = merged_df["prediction_swapped"].astype(str)

    # Categorize predictions
    categories = {
        "stable": [],  # Consistent predictions (same after swap)
        "pos1": [],  # Position 1 bias (1->2 or 2->1, but preferring position 1)
        "pos2": [],  # Position 2 bias (1->2 or 2->1, but preferring position 2)
        "other": [],  # Any other patterns (involving ties)
    }

    for i in range(len(merged_df)):
        pred_orig = merged_df.iloc[i]["prediction_original"]
        pred_swap = merged_df.iloc[i]["prediction_swapped"]
        if pred_orig == "1" and pred_swap == "2":
            # Originally chose position 1, after swap chose position 2 = stable preference for audio content
            categories["stable"].append(i)
        elif pred_orig == "2" and pred_swap == "1":
            # Originally chose position 2, after swap chose position 1 = stable preference for audio content
            categories["stable"].append(i)
        elif pred_orig == "1" and pred_swap == "1":
            # Always chooses position 1 = position 1 bias
            categories["pos1"].append(i)
        elif pred_orig == "2" and pred_swap == "2":
            # Always chooses position 2 = position 2 bias
            categories["pos2"].append(i)
        else:
            # Everything else (involving ties or other patterns)
            categories["other"].append(i)
    return categories, merged_df


def bootstrap_position_bias(
    original_file, swapped_file, n_bootstrap=10000, random_seed=42
):
    """
    Perform bootstrap analysis for position bias in audio swap predictions.
    """
    np.random.seed(random_seed)

    # Get original categorization
    categories, merged_df = categorize_predictions(original_file, swapped_file)

    if categories is None:
        return None

    # Count original categories
    n_stable = len(categories["stable"])
    n_pos1 = len(categories["pos1"])
    n_pos2 = len(categories["pos2"])
    n_other = len(categories["other"])
    n_total = len(merged_df)

    print("\n" + "=" * 50)
    print("ORIGINAL CATEGORIZATION")
    print("=" * 50)
    print(f"Total datapoints: {n_total}")
    print(f"Stable: {n_stable} ({n_stable / n_total * 100:.1f}%)")
    print(f"Position 1 bias: {n_pos1} ({n_pos1 / n_total * 100:.1f}%)")
    print(f"Position 2 bias: {n_pos2} ({n_pos2 / n_total * 100:.1f}%)")
    print(f"Other: {n_other} ({n_other / n_total * 100:.1f}%)")

    # Original bias difference (pos1 - pos2)
    original_diff = n_pos1 - n_pos2
    print(f"\nOriginal bias difference (pos1 - pos2): {original_diff}")

    # Bootstrap resampling
    print(f"\nPerforming bootstrap with {n_bootstrap} samples...")

    bootstrap_results = {
        "n_stable": [],
        "n_pos1": [],
        "n_pos2": [],
        "n_other": [],
        "bias_diff": [],
    }

    for i in range(n_bootstrap):
        if i % 1000 == 0:
            print(f"Bootstrap iteration {i}/{n_bootstrap}")

        # Sample with replacement
        boot_indices = np.random.choice(n_total, size=n_total, replace=True)
        boot_df = merged_df.iloc[boot_indices].copy()

        # Recategorize bootstrap sample
        boot_categories = {"stable": [], "pos1": [], "pos2": [], "other": []}

        for j in range(len(boot_df)):
            pred_orig = boot_df.iloc[j]["prediction_original"]
            pred_swap = boot_df.iloc[j]["prediction_swapped"]

            if pred_orig == "1" and pred_swap == "2":
                boot_categories["stable"].append(j)
            elif pred_orig == "2" and pred_swap == "1":
                boot_categories["stable"].append(j)
            elif pred_orig == "1" and pred_swap == "1":
                boot_categories["pos1"].append(j)
            elif pred_orig == "2" and pred_swap == "2":
                boot_categories["pos2"].append(j)
            else:
                boot_categories["other"].append(j)

        # Store bootstrap counts
        boot_n_stable = len(boot_categories["stable"])
        boot_n_pos1 = len(boot_categories["pos1"])
        boot_n_pos2 = len(boot_categories["pos2"])
        boot_n_other = len(boot_categories["other"])
        boot_diff = boot_n_pos1 - boot_n_pos2

        bootstrap_results["n_stable"].append(boot_n_stable)
        bootstrap_results["n_pos1"].append(boot_n_pos1)
        bootstrap_results["n_pos2"].append(boot_n_pos2)
        bootstrap_results["n_other"].append(boot_n_other)
        bootstrap_results["bias_diff"].append(boot_diff)

    # Convert to numpy arrays
    for key in bootstrap_results:
        bootstrap_results[key] = np.array(bootstrap_results[key])

    # Calculate statistics
    print("\n" + "=" * 50)
    print("BOOTSTRAP RESULTS")
    print("=" * 50)

    metrics = ["n_stable", "n_pos1", "n_pos2", "n_other", "bias_diff"]
    original_values = [n_stable, n_pos1, n_pos2, n_other, original_diff]

    for metric, orig_val in zip(metrics, original_values):
        boot_values = bootstrap_results[metric]
        mean_val = np.mean(boot_values)
        std_val = np.std(boot_values)
        ci_lower = np.percentile(boot_values, 2.5)
        ci_upper = np.percentile(boot_values, 97.5)

        print(f"\n{metric}:")
        print(f"  Original: {orig_val}")
        print(f"  Bootstrap mean: {mean_val:.1f}")
        print(f"  Bootstrap std: {std_val:.1f}")
        print(f"  95% CI: [{ci_lower:.1f}, {ci_upper:.1f}]")

    # Test for significant bias (bias_diff != 0)
    bias_diffs = bootstrap_results["bias_diff"]
    p_value_right = np.mean(bias_diffs >= abs(original_diff))
    p_value_left = np.mean(bias_diffs <= -abs(original_diff))
    p_value = 2 * min(p_value_right, p_value_left)

    print(f"\nPosition Bias Test:")
    print(f"  Two-tailed p-value: {p_value:.4f}")
    if p_value < 0.05:
        print("  *** SIGNIFICANT position bias detected! ***")
    else:
        print("  No significant position bias detected.")

    return {
        "original_counts": {
            "stable": n_stable,
            "pos1": n_pos1,
            "pos2": n_pos2,
            "other": n_other,
            "total": n_total,
            "bias_diff": original_diff,
        },
        "bootstrap_results": bootstrap_results,
        "p_value": p_value,
        "categories": categories,
        "merged_data": merged_df,
    }


# Main execution
if __name__ == "__main__":
    # Replace these with your actual file paths
    original_file_path = "../main_experiments/results/somos/gpt-4o-audio-preview/somos_standard_cot_0_shots_none_transcript_single_turn_fewshot_separate_test_separate_gpt_4o_audio_preview.csv"
    swapped_file_path = "../main_experiments/results_swapped/somos/gpt-4o-audio-preview/somos_standard_cot_0_shots_none_transcript_single_turn_fewshot_separate_test_separate_gpt_4o_audio_preview.csv"

    try:
        # Perform bootstrap analysis
        results = bootstrap_position_bias(
            original_file_path, swapped_file_path, n_bootstrap=10000, random_seed=42
        )

        if results:
            print(f"\nAnalysis complete!")
            print(f"Stable datapoints: {results['original_counts']['stable']}")
            print(f"Position 1 bias: {results['original_counts']['pos1']}")
            print(f"Position 2 bias: {results['original_counts']['pos2']}")

    except FileNotFoundError as e:
        print(f"File not found: {e}")
        print("Please make sure both CSV files exist and the paths are correct.")
    except Exception as e:
        print(f"An error occurred: {e}")
