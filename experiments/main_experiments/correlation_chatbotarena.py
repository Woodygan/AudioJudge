import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import math


def system_level_analysis_csv(path_ab, path_ba):
    """
    Analyze system-level performance from AB and BA CSV files.
    Match rows by index and discard indices with only one prediction.

    Args:
        path_ab: Path to AB comparison CSV file
        path_ba: Path to BA comparison CSV file
    """
    # Read CSV files
    data_ab = pd.read_csv(path_ab)
    data_ba = pd.read_csv(path_ba)

    print(f"Original AB data: {len(data_ab)} rows")
    print(f"Original BA data: {len(data_ba)} rows")

    # Check if index column exists, if not use row index
    if "index" in data_ab.columns:
        data_ab = data_ab.set_index("index")
    if "index" in data_ba.columns:
        data_ba = data_ba.set_index("index")

    # Find common indices between AB and BA
    common_indices = set(data_ab.index).intersection(set(data_ba.index))
    print(f"Common indices: {len(common_indices)}")

    # Filter to only common indices
    data_ab_matched = data_ab.loc[list(common_indices)].sort_index()
    data_ba_matched = data_ba.loc[list(common_indices)].sort_index()

    n = len(data_ab_matched)
    print(f"Matched pairs: {n}")

    # Get unique systems from both files
    all_models_a = set(
        data_ab_matched["model_a"].tolist() + data_ba_matched["model_a"].tolist()
    )
    all_models_b = set(
        data_ab_matched["model_b"].tolist() + data_ba_matched["model_b"].tolist()
    )
    uniq_systems = list(all_models_a.union(all_models_b))
    print(f"Unique systems: {len(uniq_systems)}")
    print(f"Systems: {uniq_systems}")

    system_preds = {k: [] for k in uniq_systems}
    system_gts = {k: [] for k in uniq_systems}

    for i in range(n):
        ab_row = data_ab_matched.iloc[i]
        ba_row = data_ba_matched.iloc[i]

        # Extract models for AB comparison
        modelA_ab = ab_row["model_a"]
        modelB_ab = ab_row["model_b"]

        # Extract models for BA comparison (should be swapped)
        modelA_ba = ba_row["model_a"]  # This should be modelB from AB
        modelB_ba = ba_row["model_b"]  # This should be modelA from AB

        # Process AB predictions
        pred_ab = ab_row["prediction"]
        if pred_ab == 1 or pred_ab == "1":  # Audio 1 (model_a) wins
            system_preds[modelA_ab].append(1.0)
            system_preds[modelB_ab].append(0.0)
        elif pred_ab == 2 or pred_ab == "2":  # Audio 2 (model_b) wins
            system_preds[modelA_ab].append(0.0)
            system_preds[modelB_ab].append(1.0)
        else:  # Tie
            system_preds[modelA_ab].append(0.5)
            system_preds[modelB_ab].append(0.5)

        # Process BA predictions (note: positions are swapped)
        pred_ba = ba_row["prediction"]
        if pred_ba == 1 or pred_ba == "1":  # Audio 1 (which is modelB from AB) wins
            system_preds[modelA_ba].append(1.0)  # modelA_ba = modelB_ab
            system_preds[modelB_ba].append(0.0)  # modelB_ba = modelA_ab
        elif pred_ba == 2 or pred_ba == "2":  # Audio 2 (which is modelA from AB) wins
            system_preds[modelA_ba].append(0.0)  # modelA_ba = modelB_ab
            system_preds[modelB_ba].append(1.0)  # modelB_ba = modelA_ab
        else:  # Tie
            system_preds[modelA_ba].append(0.5)
            system_preds[modelB_ba].append(0.5)

        # Process ground truth for AB
        gt_ab = ab_row["ground_truth"]
        if gt_ab == 1 or gt_ab == "1":  # Model A wins
            system_gts[modelA_ab].append(1.0)
            system_gts[modelB_ab].append(0.0)
        elif gt_ab == 2 or gt_ab == "2":  # Model B wins
            system_gts[modelA_ab].append(0.0)
            system_gts[modelB_ab].append(1.0)
        else:  # Tie
            system_gts[modelA_ab].append(0.5)
            system_gts[modelB_ab].append(0.5)

        # Process ground truth for BA (note: positions are swapped)
        gt_ba = ba_row["ground_truth"]
        if gt_ba == 1 or gt_ba == "1":  # Audio 1 (modelB from AB) wins
            system_gts[modelA_ba].append(1.0)  # modelA_ba = modelB_ab
            system_gts[modelB_ba].append(0.0)  # modelB_ba = modelA_ab
        elif gt_ba == 2 or gt_ba == "2":  # Audio 2 (modelA from AB) wins
            system_gts[modelA_ba].append(0.0)  # modelA_ba = modelB_ab
            system_gts[modelB_ba].append(1.0)  # modelB_ba = modelA_ab
        else:  # Tie
            system_gts[modelA_ba].append(0.5)
            system_gts[modelB_ba].append(0.5)

    # Calculate system-level win rates
    system_level_pred, system_level_gts = [], []
    count_pair_each_systems = []
    system_names = []

    print("\nSystem predictions summary:")
    for system, pred in system_preds.items():
        print(f"{system}: {len(pred)} predictions")

    for system, pred in system_preds.items():
        if len(pred) == 0:
            print(f"Skipping {system}: no predictions")
            continue
        gts = system_gts[system]
        assert len(pred) == len(gts), (
            f"Mismatch for {system}: pred={len(pred)}, gts={len(gts)}"
        )

        if not math.isnan(np.mean(pred)):
            system_level_pred.append(np.mean(pred))
            system_level_gts.append(np.mean(gts))
            count_pair_each_systems.append(len(gts))
            system_names.append(system)

    print(f"\nAvg. pairs per system: {np.mean(count_pair_each_systems):.2f}")
    print(f"Total systems analyzed: {len(system_level_gts)}")

    # Print individual system results
    print("\nSystem-level results:")
    for i, system in enumerate(system_names):
        print(
            f"{system}: GT={system_level_gts[i]:.3f}, Pred={system_level_pred[i]:.3f}, Pairs={count_pair_each_systems[i]}"
        )

    analyze_correlation_and_plot(system_level_gts, system_level_pred, system_names)

    return system_level_gts, system_level_pred, system_names


def analyze_correlation_and_plot(list_a, list_b, system_names=None):
    """
    Analyze correlation and create scatter plot.

    Args:
        list_a: Ground truth win rates
        list_b: Predicted win rates
        system_names: Optional list of system names for annotation
    """
    # Compute Spearman correlation
    spearman_corr, pvalue = stats.spearmanr(list_a, list_b)

    # Print results
    print(f"\nSpearman Correlation: {spearman_corr:.3f}")
    print("pvalue Spearman: {:.2e}".format(pvalue))

    # Scatter plot with best-fit line
    plt.figure(figsize=(10, 8))
    plt.scatter(list_a, list_b, color="blue", alpha=0.7, s=60)

    # Annotate points with system names if provided
    if system_names:
        for i, name in enumerate(system_names):
            plt.annotate(
                name,
                (list_a[i], list_b[i]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=9,
                alpha=0.8,
                ha="left",
            )

    # Add diagonal line for perfect correlation
    min_val = min(min(list_a), min(list_b))
    max_val = max(max(list_a), max(list_b))
    plt.plot(
        [min_val, max_val],
        [min_val, max_val],
        "k--",
        alpha=0.5,
        label="Perfect Correlation",
    )

    plt.xlabel("Human Judge (Win Rate)", fontsize=12, fontweight="bold")
    plt.ylabel("LLM Judge (Win Rate)", fontsize=12, fontweight="bold")
    plt.title(
        f"System-Level Correlation\nSpearman: {spearman_corr:.3f}",
        fontsize=14,
        fontweight="bold",
    )
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)

    # Make axes equal for better visualization
    plt.axis("equal")
    margin = 0.05
    plt.xlim(min_val - margin, max_val + margin)
    plt.ylim(min_val - margin, max_val + margin)

    plt.tight_layout()
    plt.show()

    return spearman_corr, pvalue


# Example usage:
if __name__ == "__main__":
    # Example call with validation
    # "results/chatbotarena/gemini-2.5-flash-preview-04-17/chatbotarena_standard_cot_0_shots_none_transcript_single_turn_fewshot_separate_test_separate_gemini_2.5_flash_preview_04_17.csv"
    # "results/chatbotarena/gpt-4o-audio-preview/chatbotarena_standard_cot_0_shots_none_transcript_single_turn_fewshot_separate_test_separate_gpt_4o_audio_preview.csv"
    path_ab = "results/chatbotarena/gpt-4o-audio-preview/chatbotarena_standard_cot_4_shots_none_transcript_single_turn_fewshot_aggregate_test_concat_gpt_4o_audio_preview.csv"
    path_ba = "results/chatbotarena_BA/gpt-4o-audio-preview/chatbotarena_BA_standard_cot_4_shots_none_transcript_single_turn_fewshot_aggregate_test_concat_gpt_4o_audio_preview.csv"

    gt_winrates, pred_winrates, systems = system_level_analysis_csv(path_ab, path_ba)
