import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
import json
import pandas as pd
from collections import defaultdict
from pathlib import Path
from typing import Dict


def load_model_judge_scores(csv_path: str | Path) -> Dict[str, float]:
    """
    Load model judge scores from CSV file and extract effective_win_rate.

    Args:
        csv_path: Path to the CSV file containing model statistics

    Returns:
        Dictionary mapping model names to their effective win rates (as percentages)
    """
    try:
        df = pd.read_csv(csv_path)
        # Convert effective_win_rate to percentage (multiply by 100)
        model_scores = {}
        for _, row in df.iterrows():
            model_scores[row["model_name"]] = row["effective_win_rate"] * 100

        print(f"Loaded {len(model_scores)} model scores from {csv_path}")
        return model_scores

    except FileNotFoundError:
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    except KeyError as e:
        raise KeyError(f"Required column not found in CSV: {e}")
    except Exception as e:
        raise Exception(f"Error reading CSV file: {e}")


def compute_normalized_scores(jsonl_path: str | Path) -> Dict[str, float]:
    """
    Read `annotations.jsonl` and return {model_name: avg_score}.

    Scoring per battle
    ------------------
    winner  : +1.0
    loser   : +0.0
    each tie: +0.5

    Normalization
    -------------
    avg_score = total_points / num_battles   (so 3 wins + 1 loss → 3/4 = 0.75)
    """
    jsonl_path = Path(jsonl_path).expanduser()
    if not jsonl_path.exists():
        raise FileNotFoundError(jsonl_path)

    # running tallies
    points = defaultdict(float)  # total points earned
    battles = defaultdict(int)  # how many times the model appeared

    count = 0
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                m1, m2 = rec["model1"], rec["model2"]
                pref = rec["preference"]
            except (KeyError, json.JSONDecodeError):
                continue  # skip malformed lines

            # each model fought once in this record
            battles[m1] += 1
            battles[m2] += 1

            if pref == "model1":
                points[m1] += 1.0
            elif pref == "model2":
                points[m2] += 1.0
            elif pref == "tie":
                points[m1] += 0.5
                points[m2] += 0.5
            # unknown preference token → ignore it quietly

            count += 1

    print(f"Counted {count} records in {jsonl_path}")

    # normalize
    normalized = {
        model: (points[model] / battles[model]) * 100 if battles[model] else 0.0
        for model in battles
    }
    return normalized


def main():
    # Load model judge scores from CSV file
    try:
        model_judge_scores = load_model_judge_scores(
            "results/speakbench/gpt-4o-audio-preview/summary_speakbench_standard_cot_4_shots_none_transcript_single_turn_gpt_4o_audio_preview_20250518_205840.csv"
        )
    except Exception as e:
        print(f"Error loading model judge scores: {e}")
        return

    # Compute human scores from annotations file
    try:
        human_scores = compute_normalized_scores("speakbench_annotations.jsonl")
    except FileNotFoundError:
        print("Error: speakbench_annotations.jsonl file not found!")
        print("Please ensure the annotations file is in the current directory.")
        return

    # Find common models between human and judge scores
    common = sorted(set(model_judge_scores) & set(human_scores))

    if not common:
        print("No common models found between human and judge scores!")
        return

    # Create vectors for correlation analysis
    human = np.array([human_scores[m] for m in common])
    judge = np.array([model_judge_scores[m] for m in common])

    # Calculate Spearman correlation
    rho, p = spearmanr(human, judge)
    print(f"Spearman ρ = {rho:.4f}   (p = {p:.3g})")

    # Calculate best-fit line (y = ax + b)
    a, b = np.polyfit(human, judge, 1)  # slope, intercept
    x_line = np.linspace(human.min(), human.max(), 100)
    y_line = a * x_line + b

    # Create the plot
    plt.figure(figsize=(6, 4))
    plt.scatter(human, judge, label="models", color="royalblue")
    plt.plot(
        x_line,
        y_line,
        "--",
        color="cornflowerblue",
        label=f"best fit, Spearman ρ = {rho:.3f}",
    )

    # Add model name annotations
    for m, hx, jx in zip(common, human, judge):
        if m == "asr+llama3+tts":
            plt.annotate(
                m,
                (hx - 15, jx - 1.2),
                fontsize=8,
                xytext=(3, 3),
                textcoords="offset points",
            )
        elif m == "gemini2-flash-text+tts":
            plt.annotate(
                m,
                (hx - 1, jx + 0.1),
                fontsize=8,
                xytext=(3, 3),
                textcoords="offset points",
            )
        elif m == "gemini2-flash-exp+asr+tts":
            plt.annotate(
                m, (hx, jx - 1.2), fontsize=8, xytext=(3, 3), textcoords="offset points"
            )
        elif m == "gemini2-flash-exp":
            plt.annotate(
                m,
                (hx - 16, jx - 2),
                fontsize=8,
                xytext=(3, 3),
                textcoords="offset points",
            )
        elif m == "gpt4o-text+tts":
            plt.annotate(
                m,
                (hx - 13, jx - 2),
                fontsize=8,
                xytext=(3, 3),
                textcoords="offset points",
            )
        elif m == "gpt4o-audio+asr+tts":
            plt.annotate(
                m, (hx, jx - 1.2), fontsize=8, xytext=(3, 3), textcoords="offset points"
            )
        elif m == "gpt4o-audio":
            plt.annotate(
                m, (hx - 10, jx), fontsize=8, xytext=(3, 3), textcoords="offset points"
            )
        else:
            plt.annotate(
                m, (hx, jx), fontsize=8, xytext=(3, 3), textcoords="offset points"
            )

    # Format the plot
    plt.xlabel("Human normalized score")
    plt.ylabel("Model-judge score")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    # Save the plot
    plt.savefig("human_vs_model-judge.pdf", bbox_inches="tight")
    print("Plot saved as 'human_vs_model-judge.pdf'")

    # Show the plot
    plt.show()

    # Print detailed results
    print("\nDetailed Results:")
    print("=" * 50)
    print(f"{'Model':<25} {'Human Score':<15} {'Judge Score':<15}")
    print("-" * 50)
    for m in common:
        print(f"{m:<25} {human_scores[m]:<15.2f} {model_judge_scores[m]:<15.2f}")


if __name__ == "__main__":
    main()
