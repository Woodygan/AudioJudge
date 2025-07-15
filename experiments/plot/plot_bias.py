import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Set style for publication-quality plots
plt.style.use("default")
plt.rcParams["font.size"] = 12
plt.rcParams["font.weight"] = "normal"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["axes.linewidth"] = 1.2
plt.rcParams["xtick.major.width"] = 1.2
plt.rcParams["ytick.major.width"] = 1.2

# Define colors for consistency
colors = {"tie_stable": "#95A5A6", "longer_pos1": "#3498DB", "shorter_pos2": "#E74C3C"}

import matplotlib.pyplot as plt
import numpy as np


def plot_verbosity_bias():
    """
    Plot verbosity bias from Table: Verbosity bias on ChatbotArena-Spoken
    """
    # Data from the table
    models = ["Gemini-1.5-F", "Gemini-2.0-F", "Gemini-2.5-F", "GPT-4o-Audio"]
    longer = [55.7, 54.0, 44.7, 49.3]
    shorter = [39.1, 34.2, 34.6, 37.9]

    x = np.arange(len(models))
    width = 0.35

    # Create figure with larger size
    fig, ax = plt.subplots(figsize=(12, 8))

    bars1 = ax.bar(
        x - width / 2,
        longer,
        width,
        label="Prefer Longer",
        color=colors["longer_pos1"],
        alpha=0.8,
        edgecolor="black",
        linewidth=0.5,
    )
    bars2 = ax.bar(
        x + width / 2,
        shorter,
        width,
        label="Prefer Shorter",
        color=colors["shorter_pos2"],
        alpha=0.8,
        edgecolor="black",
        linewidth=0.5,
    )

    # Customize the plot with larger fonts
    ax.set_xlabel("Models", fontweight="bold", fontsize=30)
    ax.set_ylabel("Percentage (%)", fontweight="bold", fontsize=30)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=0, fontsize=24)
    ax.tick_params(axis="y", labelsize=24)

    # Larger legend
    ax.legend(
        frameon=True,
        loc="upper right",
        fontsize=24,
        fancybox=True,
        shadow=True,
        framealpha=0.9,
    )

    ax.set_ylim(0, 70)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    # Add value labels on bars with larger font
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.1f}",
                ha="center",
                va="bottom",
                fontsize=24,
                fontweight="bold",
            )

    # Improve layout
    plt.tight_layout()
    plt.savefig(
        "verbosity_bias.pdf",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
    )
    plt.show()


def plot_positional_bias_lexical():
    """
    Plot positional bias for lexical content from Table: Positional bias on ChatbotArena-Spoken
    """
    # Data from the table
    models = ["Gemini-1.5-F", "Gemini-2.0-F", "Gemini-2.5-F", "GPT-4o-Audio"]
    pos1 = [18.3, 7.5, 7.3, 8.9]
    pos2 = [5.1, 7.6, 5.3, 4.3]

    x = np.arange(len(models))
    width = 0.35

    # Create figure with larger size
    fig, ax = plt.subplots(figsize=(12, 8))

    bars1 = ax.bar(
        x - width / 2,
        pos1,
        width,
        label="Position 1 Bias",
        color=colors["longer_pos1"],
        alpha=0.8,
        edgecolor="black",
        linewidth=0.5,
    )
    bars2 = ax.bar(
        x + width / 2,
        pos2,
        width,
        label="Position 2 Bias",
        color=colors["shorter_pos2"],
        alpha=0.8,
        edgecolor="black",
        linewidth=0.5,
    )

    # Customize the plot with larger fonts
    ax.set_xlabel("Models", fontweight="bold", fontsize=30)
    ax.set_ylabel("Percentage (%)", fontweight="bold", fontsize=30)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=0, fontsize=24)
    ax.tick_params(axis="y", labelsize=24)

    # Larger legend
    ax.legend(
        frameon=True,
        loc="upper right",
        fontsize=24,
        fancybox=True,
        shadow=True,
        framealpha=0.9,
    )

    ax.set_ylim(0, 20)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    # Add value labels on bars with larger font
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.1f}",
                ha="center",
                va="bottom",
                fontsize=24,
                fontweight="bold",
            )

    plt.tight_layout()
    plt.savefig(
        "positional_bias_lexical.pdf",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
    )
    plt.show()


def plot_positional_bias_nonlexical():
    """
    Plot positional bias for non-lexical content from Table: Positional bias analysis across non-lexical datasets
    """
    # Data from the table
    tasks = ["Speed", "SOMOS", "TMHINTQ", "ThaiMOS", "SpeakBench"]
    pos1 = [1.1, 33.5, 32.5, 15.0, 9.8]
    pos2 = [49.2, 5.0, 6.0, 15.5, 12.5]

    x = np.arange(len(tasks))
    width = 0.35

    # Create figure with larger size
    fig, ax = plt.subplots(figsize=(12, 8))

    bars1 = ax.bar(
        x - width / 2,
        pos1,
        width,
        label="Position 1 Bias",
        color=colors["longer_pos1"],
        alpha=0.8,
        edgecolor="black",
        linewidth=0.5,
    )
    bars2 = ax.bar(
        x + width / 2,
        pos2,
        width,
        label="Position 2 Bias",
        color=colors["shorter_pos2"],
        alpha=0.8,
        edgecolor="black",
        linewidth=0.5,
    )

    # Customize the plot with larger fonts
    ax.set_xlabel("Evaluation Tasks", fontweight="bold", fontsize=30)
    ax.set_ylabel("Percentage (%)", fontweight="bold", fontsize=30)
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, rotation=0, fontsize=24)
    ax.tick_params(axis="y", labelsize=24)

    # Larger legend
    ax.legend(
        frameon=True,
        loc="upper right",
        fontsize=24,
        fancybox=True,
        shadow=True,
        framealpha=0.9,
    )

    ax.set_ylim(0, 55)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    # Add value labels on bars with larger font
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.1f}",
                ha="center",
                va="bottom",
                fontsize=20,
                fontweight="bold",
            )

    plt.tight_layout()
    plt.savefig(
        "positional_bias_nonlexical.pdf",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
    )
    plt.show()


def create_all_bias_plots():
    """Generate all bias analysis plots"""
    print("Generating bias analysis visualizations...")

    plot_verbosity_bias()
    print("✓ Verbosity bias plot saved as 'verbosity_bias.pdf'")

    plot_positional_bias_lexical()
    print("✓ Positional bias (lexical) plot saved as 'positional_bias_lexical.pdf'")

    plot_positional_bias_nonlexical()
    print(
        "✓ Positional bias (non-lexical) plot saved as 'positional_bias_nonlexical.pdf'"
    )

    print("\nAll bias visualization files generated successfully!")


if __name__ == "__main__":
    create_all_bias_plots()
