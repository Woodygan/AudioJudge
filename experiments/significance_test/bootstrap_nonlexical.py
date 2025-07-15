import pandas as pd
import numpy as np
from scipy import stats
import os
import argparse


def load_data(file_path, id_column, result_column):
    """
    Load data from a CSV file and extract the result column.

    Args:
        file_path (str): Path to the CSV file
        id_column (str): Name of the ID column
        result_column (str): Name of the result column

    Returns:
        pandas.DataFrame: DataFrame with ID and result columns
    """
    try:
        df = pd.read_csv(file_path)
        if id_column not in df.columns or result_column not in df.columns:
            raise ValueError(
                f"Columns {id_column} or {result_column} not found in {file_path}"
            )

        # Extract only the needed columns
        return df[[id_column, result_column]]
    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        exit(1)


def filter_common_datapoints(df_a, df_b, id_column):
    """
    Filter both dataframes to include only data points with IDs present in both datasets.

    Args:
        df_a (pandas.DataFrame): First dataset
        df_b (pandas.DataFrame): Second dataset
        id_column (str): Name of the ID column

    Returns:
        tuple: Filtered versions of df_a and df_b containing only shared data points
    """
    # Find common IDs
    common_ids = set(df_a[id_column]).intersection(set(df_b[id_column]))

    # Report on filtering
    print(f"Dataset A has {len(df_a)} data points")
    print(f"Dataset B has {len(df_b)} data points")
    print(f"Found {len(common_ids)} common data points")
    print(
        f"Filtered out {len(df_a) - len(common_ids)} from A and {len(df_b) - len(common_ids)} from B"
    )

    # Filter both dataframes to include only common IDs
    filtered_df_a = df_a[df_a[id_column].isin(common_ids)]
    filtered_df_b = df_b[df_b[id_column].isin(common_ids)]

    # Sort both dataframes by ID to ensure alignment
    filtered_df_a = filtered_df_a.sort_values(by=id_column).reset_index(drop=True)
    filtered_df_b = filtered_df_b.sort_values(by=id_column).reset_index(drop=True)

    return filtered_df_a, filtered_df_b


def bootstrap_difference_independent_two_sided(
    data_a, data_b, n_samples=10000, confidence_level=0.95
):
    """
    Perform bootstrap analysis to determine if the difference between two datasets is significant.

    Args:
        data_a (array-like): First dataset
        data_b (array-like): Second dataset
        n_samples (int): Number of bootstrap samples
        confidence_level (float): Confidence level for the interval

    Returns:
        tuple: Mean difference, confidence interval, and significance result
    """
    # Convert to numpy arrays and ensure they are numeric
    data_a = np.array(data_a, dtype=float)
    data_b = np.array(data_b, dtype=float)

    # Calculate observed difference in means
    observed_diff = np.mean(data_b) - np.mean(data_a)

    # Bootstrap to generate sampling distribution of the difference
    bootstrap_diffs = np.zeros(n_samples)

    for i in range(n_samples):
        # Sample with replacement from both datasets
        sample_a = np.random.choice(data_a, size=len(data_a), replace=True)
        sample_b = np.random.choice(data_b, size=len(data_b), replace=True)

        # Calculate and store the difference in means
        bootstrap_diffs[i] = np.mean(sample_b) - np.mean(sample_a)

    # Calculate confidence interval
    alpha = 1 - confidence_level
    lower_bound = np.percentile(bootstrap_diffs, alpha / 2 * 100)
    upper_bound = np.percentile(bootstrap_diffs, (1 - alpha / 2) * 100)

    # Determine if the difference is significant
    is_significant = (lower_bound > 0 and upper_bound > 0) or (
        lower_bound < 0 and upper_bound < 0
    )

    return observed_diff, (lower_bound, upper_bound), is_significant, bootstrap_diffs


def bootstrap_difference_paired_one_sided(
    data_a, data_b, n_samples=10000, confidence_level=0.95
):
    """
    Perform bootstrap analysis to determine if method_b is better method_a.

    Args:
        data_a (array-like): First dataset
        data_b (array-like): Second dataset
        n_samples (int): Number of bootstrap samples
        confidence_level (float): Confidence level for the interval

    Returns:
        tuple: Mean difference, p_value, and significance result
    """
    # Convert to numpy arrays and ensure they are numeric
    data_a = np.array(data_a, dtype=float)
    data_b = np.array(data_b, dtype=float)

    # Calculate observed difference in means
    observed_diff = np.mean(data_b) - np.mean(data_a)

    # Bootstrap to generate sampling distribution of the difference
    bootstrap_diffs = np.zeros(n_samples)

    # paired resampling
    diffs = data_b - data_a
    for i in range(n_samples):
        sample_diffs = np.random.choice(diffs, size=len(diffs), replace=True)
        bootstrap_diffs[i] = np.mean(sample_diffs)

    # one-sided CI
    # this is the fraction of resampled means that are not in the hypothesized direction
    # i.e. p_one_sided < 0.05  for confidence_level=0.95
    p_one_sided = np.mean(bootstrap_diffs <= 0)
    is_significant = p_one_sided < (1 - confidence_level)
    return observed_diff, p_one_sided, is_significant, bootstrap_diffs


def main():
    # Define configuration
    parser = argparse.ArgumentParser(
        description="Paired one-sided bootstrap test (B > A)"
    )
    parser.add_argument(
        "-a", "--file-a", required=True, help="Path to first CSV file (method A)"
    )
    parser.add_argument(
        "-b", "--file-b", required=True, help="Path to second CSV file (method B)"
    )
    parser.add_argument(
        "--id-column", default="audio1_path", help="Name of the ID column"
    )
    parser.add_argument(
        "--result-column", default="correct", help="Name of the result column"
    )
    parser.add_argument(
        "--method",
        default="paired_one_sided",
        choices=["independent_two_sided", "paired_one_sided"],
        help="Type of bootstrap test to perform",
    )
    parser.add_argument(
        "-n", "--n-samples", type=int, default=10000, help="Number of bootstrap samples"
    )
    parser.add_argument(
        "-c",
        "--confidence",
        type=float,
        default=0.95,
        help="Confidence level for the one-sided test",
    )

    args = parser.parse_args()

    FILE_A = args.file_a
    FILE_B = args.file_b
    N_SAMPLES = args.n_samples
    CONFIDENCE_LEVEL = args.confidence
    ID_COLUMN = args.id_column
    RESULT_COLUMN = args.result_column

    # Check if files exist
    if not os.path.exists(FILE_A):
        print(f"Error: File {FILE_A} does not exist.")
        return
    if not os.path.exists(FILE_B):
        print(f"Error: File {FILE_B} does not exist.")
        return

    # Load data
    print(f"Loading data from {FILE_A} and {FILE_B}...")
    df_a = load_data(FILE_A, ID_COLUMN, RESULT_COLUMN)
    df_b = load_data(FILE_B, ID_COLUMN, RESULT_COLUMN)

    # Filter for common data points
    print("\nFiltering for common data points...")
    filtered_df_a, filtered_df_b = filter_common_datapoints(df_a, df_b, ID_COLUMN)

    # Check if we have any common data points
    if len(filtered_df_a) == 0:
        print("Error: No common data points found between the two datasets.")
        return

    # Extract result columns
    data_a = filtered_df_a[RESULT_COLUMN].values
    data_b = filtered_df_b[RESULT_COLUMN].values

    # Verify alignment (IDs should match)
    if not all(filtered_df_a[ID_COLUMN].values == filtered_df_b[ID_COLUMN].values):
        print("Warning: IDs do not match after filtering and sorting.")
        return

    print(f"\nDataset A: {len(data_a)} samples, mean = {np.mean(data_a):.4f}")
    print(f"Dataset B: {len(data_b)} samples, mean = {np.mean(data_b):.4f}")

    if args.method == "independent_two_sided":
        # Perform bootstrap analysis --- independent, two-sided
        print(f"\nPerforming bootstrap analysis with {N_SAMPLES} samples...")
        observed_diff, ci, is_significant, bootstrap_diffs = (
            bootstrap_difference_independent_two_sided(
                data_a, data_b, n_samples=N_SAMPLES, confidence_level=CONFIDENCE_LEVEL
            )
        )

        # Calculate p-value (two-tailed)
        p_value = min(
            np.mean(bootstrap_diffs <= 0) * 2, np.mean(bootstrap_diffs >= 0) * 2
        )
        p_value = min(p_value, 1.0)  # Ensure p-value doesn't exceed 1

        # Display results
        print("\nResults:")
        print(f"Observed difference (B - A): {observed_diff:.4f}")
        print(
            f"{CONFIDENCE_LEVEL * 100}% Confidence Interval: ({ci[0]:.4f}, {ci[1]:.4f})"
        )
        print(f"Approximate p-value: {p_value:.4f}")

        if is_significant:
            print(
                f"The change is statistically significant ({CONFIDENCE_LEVEL * 100}% CI does not include zero)"
            )
        else:
            print(
                f"The change is not statistically significant ({CONFIDENCE_LEVEL * 100}% CI includes zero)"
            )

    elif args.method == "paired_one_sided":
        # Perform bootstrap analysis --- paired, one-sided
        print(f"\nPerforming bootstrap analysis with {N_SAMPLES} samples...")
        observed_diff, p_value, is_significant, bootstrap_diffs = (
            bootstrap_difference_paired_one_sided(
                data_a, data_b, n_samples=N_SAMPLES, confidence_level=CONFIDENCE_LEVEL
            )
        )
        # Display results
        print("\nResults:")
        print(f"Observed difference (B - A): {observed_diff:.4f}")
        print(f"One-sided p-value: {p_value:.4f}")
        if is_significant:
            print(
                f"Method B is significantly better than Method A (p < {1 - CONFIDENCE_LEVEL:.2f})"
            )
        else:
            print(f"No significant difference found (p >= {1 - CONFIDENCE_LEVEL:.2f})")
    else:
        print(
            f"Error: Unknown method '{args.method}'. Supported methods are 'independent_two_sided' and 'paired_one_sided'."
        )


if __name__ == "__main__":
    main()
    # example usage:
    # python bootstrap_nonlexical.py -a ../main_experiments/results/pronunciation/gpt-4o-audio-preview/pronunciation_standard_cot_0_shots_none_transcript_single_turn_fewshot_separate_test_separate_gpt_4o_audio_preview.csv -b ../main_experiments/results/pronunciation/gpt-4o-audio-preview/pronunciation_standard_cot_0_shots_groundtruth_transcript_single_turn_fewshot_separate_test_separate_gpt_4o_audio_preview.csv --id-column audio1_path --result-column correct --method paired_one_sided --n-samples 10000 --confidence 0.95
