import numpy as np
from scipy import stats
import json
import re


def extract_abc(text):
    """Extract A, B, or C from text using the same pattern as original code"""
    pattern = r"\[\[(A|B|C)\]\]"
    match = re.search(pattern, text)
    if match:
        return match.group(1)
    else:
        return "D"


def read_jsonl(file_path):
    """Read JSONL file and extract predictions"""
    data = []
    with open(file_path, "r") as file:
        for line in file:
            json_obj = json.loads(line.strip())
            data.append(json_obj)
    return data


def process_output(data):
    """Process JSONL data to extract labels"""
    labels = []
    for x in data:
        response = x["response"]
        response = response[-20:]  # Last 20 characters
        labels.append(extract_abc(response))
    return labels


def categorize_predictions(preds_ab, preds_ba):
    """
    Categorize prediction pairs into:
    - stable: consistent predictions (A->B or B->A)
    - position1: bias toward position 1 (A->A)
    - position2: bias toward position 2 (B->B)
    - ties: both predict tie (C->C)
    """
    n = min(len(preds_ab), len(preds_ba))
    preds_ab = preds_ab[:n]
    preds_ba = preds_ba[:n]

    categories = {
        "stable": [],
        "position1": [],  # A->A bias
        "position2": [],  # B->B bias
        "ties": [],
    }

    for i in range(n):
        pred_ab = preds_ab[i]
        pred_ba = preds_ba[i]

        if (pred_ab == "A" and pred_ba == "B") or (pred_ab == "B" and pred_ba == "A"):
            categories["stable"].append(i)
        elif pred_ab == "A" and pred_ba == "A":
            categories["position1"].append(i)
        elif pred_ab == "B" and pred_ba == "B":
            categories["position2"].append(i)
        elif pred_ab == "C" and pred_ba == "C":
            categories["ties"].append(i)

    return categories


def bootstrap_positional_bias(preds_ab, preds_ba, n_bootstrap=10000, random_seed=42):
    """
    Perform bootstrap test for positional bias significance

    Returns:
    - categories: dict with counts of each category
    - p_value: two-tailed p-value for testing if pos1_bias != pos2_bias
    - bootstrap_stats: array of bootstrap statistics
    """
    np.random.seed(random_seed)

    # Get original categories
    categories = categorize_predictions(preds_ab, preds_ba)

    # Count original statistics
    n_stable = len(categories["stable"])
    n_pos1 = len(categories["position1"])
    n_pos2 = len(categories["position2"])
    n_ties = len(categories["ties"])

    print(f"Original counts:")
    print(f"Stable: {n_stable}")
    print(f"Position 1 bias: {n_pos1}")
    print(f"Position 2 bias: {n_pos2}")
    print(f"Ties: {n_ties}")
    print(f"Total analyzed: {n_stable + n_pos1 + n_pos2}")

    # Original bias difference (pos1 - pos2)
    original_diff = n_pos1 - n_pos2

    print(f"\nOriginal bias difference (pos1 - pos2): {original_diff}")

    # Bootstrap sampling
    bootstrap_diffs = []
    n_total = len(preds_ab)

    for _ in range(n_bootstrap):
        # Sample with replacement
        indices = np.random.choice(n_total, size=n_total, replace=True)

        boot_preds_ab = [preds_ab[i] for i in indices]
        boot_preds_ba = [preds_ba[i] for i in indices]

        # Categorize bootstrap sample
        boot_categories = categorize_predictions(boot_preds_ab, boot_preds_ba)

        boot_pos1 = len(boot_categories["position1"])
        boot_pos2 = len(boot_categories["position2"])

        bootstrap_diffs.append(boot_pos1 - boot_pos2)

    bootstrap_diffs = np.array(bootstrap_diffs)

    # Calculate p-value (two-tailed test for bias difference != 0)
    # Under null hypothesis, pos1_bias = pos2_bias, so difference should be 0
    p_value_right = np.mean(bootstrap_diffs >= abs(original_diff))
    p_value_left = np.mean(bootstrap_diffs <= -abs(original_diff))
    p_value = 2 * min(p_value_right, p_value_left)

    print(f"\nBootstrap Results (n_bootstrap={n_bootstrap}):")
    print(f"Bootstrap mean difference: {np.mean(bootstrap_diffs):.2f}")
    print(f"Bootstrap std: {np.std(bootstrap_diffs):.2f}")
    print(
        f"95% CI: [{np.percentile(bootstrap_diffs, 2.5):.2f}, {np.percentile(bootstrap_diffs, 97.5):.2f}]"
    )
    print(f"Two-tailed p-value: {p_value:.4f}")

    if p_value < 0.05:
        print("*** SIGNIFICANT positional bias detected! ***")
    else:
        print("No significant positional bias detected.")

    return {
        "categories": categories,
        "original_diff": original_diff,
        "bootstrap_diffs": bootstrap_diffs,
        "p_value": p_value,
        "n_stable": n_stable,
        "n_pos1": n_pos1,
        "n_pos2": n_pos2,
        "n_ties": n_ties,
    }


def analyze_model_bias(ab_file, ba_file):
    """
    Analyze positional bias for a specific model
    """
    print(f"Analyzing: {ab_file} vs {ba_file}")
    print("=" * 50)

    # Load and process data
    ab_data = read_jsonl(ab_file)
    ba_data = read_jsonl(ba_file)

    ab_preds = process_output(ab_data)
    ba_preds = process_output(ba_data)

    # Perform bootstrap analysis
    results = bootstrap_positional_bias(ab_preds, ba_preds)

    return results


# Example usage with your file structure:
if __name__ == "__main__":
    result = analyze_model_bias(
        "../lexical-context-chatbot-arena/experiments/chatbot-arena-7824/audio-text-gemini2.5flash.jsonl",
        "../lexical-context-chatbot-arena/experiments/chatbot-arena-7824/audio-text-gemini2.5flash_BA.jsonl",
    )

    print(f"  Pos1 bias: {result['n_pos1']}, Pos2 bias: {result['n_pos2']}")
    print(f"  Difference: {result['original_diff']}")
    print(f"  P-value: {result['p_value']:.4f}")
    print(f"  Significant: {'Yes' if result['p_value'] < 0.05 else 'No'}")
    print()
