import json
import os
import re
import numpy as np
from scipy import stats
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")


def read_jsonl(file_path):
    """Read JSONL file and return list of JSON objects"""
    data = []
    with open(file_path, "r") as file:
        for line in file:
            json_obj = json.loads(line.strip())
            data.append(json_obj)
    print("len:", len(data))
    return data


def extract_abc(text):
    """Extract A, B, or C from text using regex pattern"""
    pattern = r"\[\[(A|B|C)\]\]"
    match = re.search(pattern, text)
    if match:
        result = match.group(1)
    else:
        result = "D"
    return result


def process_output(data):
    """Process model outputs to extract labels"""
    labels = []
    for x in data:
        response = x["response"]
        response = response[-20:]  # Look at last 20 characters
        labels.append(extract_abc(response))
    calculate_percentage(labels)
    return labels


def calculate_percentage(arr):
    """Calculate and print percentage distribution of labels"""
    total_count = len(arr)
    item_counts = {"A": 0, "B": 0, "C": 0, "D": 0}

    for item in arr:
        item_counts[item] = item_counts.get(item, 0) + 1

    percentages = {
        item: (count / total_count) * 100 for item, count in item_counts.items()
    }

    print("---------------")
    for item, percentage in percentages.items():
        print(f"{item}: {percentage:.2f}%")
    print("---------------")


def count_tokens(text):
    """Count tokens in text using GPT-2 tokenizer"""
    return len(tokenizer.tokenize(text))


def bootstrap_verbosity_bias(
    preds, gts, raw_data, n_bootstrap=10000, confidence_level=0.95
):
    """
    Perform bootstrap analysis to test statistical significance of verbosity bias.

    Args:
        preds: List of predictions ['A', 'B', 'C', 'D']
        gts: List of ground truth labels ['A', 'B', 'C']
        raw_data: Original data with conversation content
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level for intervals

    Returns:
        Dictionary with bootstrap results
    """

    # First, collect the data for tie cases only
    tie_data = []
    for i, (pred, gt) in enumerate(zip(preds, gts)):
        if gt != "C":  # Only analyze tie cases
            continue

        tokens_a = count_tokens(raw_data[i]["conversation_a"][1]["content"])
        tokens_b = count_tokens(raw_data[i]["conversation_b"][1]["content"])

        if pred in ["C", "D"]:
            choice = "tie"
        else:
            if tokens_a == tokens_b:
                choice = "tie"
            elif tokens_a > tokens_b:
                if pred == "A":
                    choice = "longer"
                else:
                    choice = "shorter"
            else:
                if pred == "A":
                    choice = "shorter"
                else:
                    choice = "longer"

        tie_data.append(
            {"choice": choice, "tokens_a": tokens_a, "tokens_b": tokens_b, "pred": pred}
        )

    print(f"Total tie cases with different lengths: {len(tie_data)}")

    # Count all choices (including ties)
    all_choices = [d["choice"] for d in tie_data]
    longer_count = all_choices.count("longer")
    shorter_count = all_choices.count("shorter")
    tie_count = all_choices.count("tie")
    total_cases = len(all_choices)

    if total_cases == 0:
        print("No cases found for bootstrap analysis")
        return None

    print(
        f"Actual: {longer_count} longer, {shorter_count} shorter, {tie_count} ties out of {total_cases} total cases"
    )

    # Bootstrap sampling
    bootstrap_longer_props = []
    bootstrap_differences = []  # (longer_count - shorter_count)

    for _ in range(n_bootstrap):
        # Resample with replacement from all choices (including ties)
        bootstrap_sample = np.random.choice(all_choices, size=total_cases, replace=True)

        boot_longer = np.sum(bootstrap_sample == "longer")
        boot_shorter = np.sum(bootstrap_sample == "shorter")
        boot_tie = np.sum(bootstrap_sample == "tie")

        # Calculate raw count differences
        bootstrap_differences.append(boot_longer - boot_shorter)

        # Also track overall proportions for completeness
        bootstrap_longer_props.append(boot_longer / total_cases)

    # Calculate statistics
    observed_difference = longer_count - shorter_count

    # Confidence intervals
    alpha = 1 - confidence_level
    ci_lower = np.percentile(bootstrap_differences, (alpha / 2) * 100)
    ci_upper = np.percentile(bootstrap_differences, (1 - alpha / 2) * 100)

    # P-value calculation (two-tailed test)
    # H0: no difference (difference = 0)
    # For two-tailed test: p-value = 2 * min(P(diff <= 0), P(diff >= 0))
    p_left = np.mean(np.array(bootstrap_differences) <= 0)
    p_right = np.mean(np.array(bootstrap_differences) >= 0)
    p_value = 2 * min(p_left, p_right)

    # Alternative p-value calculation using normal approximation
    bootstrap_std = np.std(bootstrap_differences)
    z_score = observed_difference / bootstrap_std if bootstrap_std > 0 else 0
    p_value_normal = 2 * (1 - stats.norm.cdf(abs(z_score)))

    results = {
        "total_cases": total_cases,
        "non_tie_cases": longer_count + shorter_count,
        "longer_count": longer_count,
        "shorter_count": shorter_count,
        "tie_count": tie_count,
        "observed_difference": observed_difference,
        "bootstrap_mean_difference": np.mean(bootstrap_differences),
        "bootstrap_std_difference": bootstrap_std,
        "confidence_interval": (ci_lower, ci_upper),
        "p_value_bootstrap": p_value,
        "p_value_normal": p_value_normal,
        "z_score": z_score,
        "is_significant": p_value < (1 - confidence_level),
    }

    return results


def print_bootstrap_results(results, confidence_level=0.95):
    """Print formatted bootstrap results"""
    if results is None:
        return

    print("\n" + "=" * 60)
    print("BOOTSTRAP SIGNIFICANCE TEST FOR VERBOSITY BIAS")
    print("=" * 60)

    print(f"Total tie cases with different lengths: {results['total_cases']}")
    print(f"  - Cases choosing longer: {results.get('longer_count', 0)}")
    print(f"  - Cases choosing shorter: {results.get('shorter_count', 0)}")
    print(f"  - Cases with ties/invalid: {results.get('tie_count', 0)}")

    non_tie_cases = results["non_tie_cases"]
    print(f"Non-tie cases for analysis: {non_tie_cases}")
    if non_tie_cases > 0:
        print(
            f"Cases choosing longer response: {results['longer_count']} ({results['longer_count'] / non_tie_cases:.1%})"
        )
        print(
            f"Cases choosing shorter response: {results['shorter_count']} ({results['shorter_count'] / non_tie_cases:.1%})"
        )
    print(f"Cases with ties/invalid predictions: {results['tie_count']}")

    print(f"\nObserved difference (longer - shorter): {results['observed_difference']}")
    print(f"Bootstrap mean difference: {results['bootstrap_mean_difference']:.2f}")
    print(f"Bootstrap std of difference: {results['bootstrap_std_difference']:.2f}")

    print(
        f"\n{confidence_level:.0%} Confidence Interval: [{results['confidence_interval'][0]:.2f}, {results['confidence_interval'][1]:.2f}]"
    )

    print(f"\nStatistical Test Results:")
    print(f"  Z-score: {results['z_score']:.4f}")
    print(f"  P-value (bootstrap): {results['p_value_bootstrap']:.6f}")
    print(f"  P-value (normal approx): {results['p_value_normal']:.6f}")

    alpha = 1 - confidence_level
    if results["p_value_bootstrap"] < alpha:
        print(f"  Result: SIGNIFICANT at α = {alpha:.3f} level")
        if results["observed_difference"] > 0:
            print(f"  Interpretation: Significant bias towards LONGER responses")
        else:
            print(f"  Interpretation: Significant bias towards SHORTER responses")
    else:
        print(f"  Result: NOT SIGNIFICANT at α = {alpha:.3f} level")
        print(f"  Interpretation: No significant verbosity bias detected")

    print("=" * 60)


def verbosity_bias_with_bootstrap(preds, gts, raw_data, n_bootstrap=10000):
    """Enhanced verbosity bias analysis with bootstrap testing"""

    assert len(preds) == len(gts)
    assert len(raw_data) == len(gts)

    i = 0
    tie, longer, shorter, total = 0, 0, 0, 0

    # Original analysis
    for pred, gt in zip(preds, gts):
        if gt != "C":
            i += 1
            continue
        tokens_a = count_tokens(raw_data[i]["conversation_a"][1]["content"])
        tokens_b = count_tokens(raw_data[i]["conversation_b"][1]["content"])
        if abs(tokens_a - tokens_b) < 5:
            i += 1
            continue
        if pred in ["C", "D"]:
            tie += 1
        else:
            if tokens_a > tokens_b:
                if pred == "A":
                    longer += 1
                else:
                    shorter += 1
            else:
                if pred == "A":
                    shorter += 1
                else:
                    longer += 1
        total += 1
        i += 1

    print("BASIC VERBOSITY ANALYSIS:")
    print("tie     = {:.2f}%".format(tie / total * 100))
    print("longer  = {:.2f}%".format(longer / total * 100))
    print("shorter = {:.2f}%".format(shorter / total * 100))
    print(shorter)
    # Bootstrap analysis
    bootstrap_results = bootstrap_verbosity_bias(preds, gts, raw_data, n_bootstrap)
    print_bootstrap_results(bootstrap_results)

    return bootstrap_results


def main():
    """Main function to run the complete verbosity bias analysis"""

    # Load ground truth data
    print("Loading ground truth data...")
    with open(
        "../lexical-context-chatbot-arena/data/chatbot-arena-spoken-1turn-english-difference-voices.json"
    ) as f:
        raw_data = json.load(f)

    # Process ground truth labels
    gts = []
    gt_mapping = {"model_a": "A", "model_b": "B", "tie": "C", "tie (bothbad)": "C"}
    for x in raw_data:
        gts.append(gt_mapping[x["winner"]])
    print(f"Ground truth labels loaded: {len(gts)}")

    # Initialize tokenizer (done once for efficiency)
    print("Initializing tokenizer...")
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # Load and process model predictions
    print("Loading model predictions...")
    model_predictions = process_output(
        read_jsonl(
            "../lexical-context-chatbot-arena/experiments/chatbot-arena-7824/audio-text-gemini2.5flash.jsonl"
        )
    )

    # Run bootstrap analysis
    print("\nRunning bootstrap verbosity bias analysis...")
    bootstrap_results = verbosity_bias_with_bootstrap(
        model_predictions, gts, raw_data, n_bootstrap=10000
    )

    return bootstrap_results


if __name__ == "__main__":
    main()
