import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
from typing import List


def plot_correlation(
    human_winrates: List[float],
    automated_winrates: List[float],
    xlabel: str = "Human Win Rate (%)",
    ylabel: str = "Automated Win Rate (%)",
    save_path: str = "correlation_plot.pdf",
):
    """
    Create a clean correlation plot between human and automated win rates.

    Args:
        human_winrates: List of human win rate percentages
        automated_winrates: List of automated win rate percentages
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        save_path: Path to save the plot
    """

    # Convert to numpy arrays
    human = np.array(human_winrates)
    automated = np.array(automated_winrates)

    # Calculate Spearman correlation
    rho, p_value = spearmanr(human, automated)

    # Calculate best-fit line
    slope, intercept = np.polyfit(human, automated, 1)
    x_line = np.linspace(human.min(), human.max(), 100)
    y_line = slope * x_line + intercept

    # Create the plot with larger figure size and fonts
    plt.figure(figsize=(10, 8))

    # Scatter plot
    plt.scatter(
        human,
        automated,
        s=150,
        alpha=0.7,
        color="royalblue",
        edgecolors="navy",
        linewidth=2.5,
        label="Systems",
    )

    # Best fit line
    plt.plot(x_line, y_line, "--", color="red", linewidth=2)

    # Formatting
    plt.xlabel(xlabel, fontsize=24, fontweight="bold")
    plt.ylabel(ylabel, fontsize=24, fontweight="bold")

    # Tick formatting
    plt.tick_params(axis="both", which="major", labelsize=24)

    # Grid and legend
    plt.grid(alpha=0.3, linestyle="-", linewidth=0.5)
    plt.legend(fontsize=30, frameon=True, fancybox=True, shadow=True, framealpha=0.9)

    # Add correlation text box
    textstr = f"Spearman ρ = {rho:.3f}\np-value = {p_value:.3e}"
    props = dict(boxstyle="round", facecolor="wheat", alpha=0.8)
    plt.text(
        0.05,
        0.95,
        textstr,
        transform=plt.gca().transAxes,
        fontsize=30,
        verticalalignment="top",
        bbox=props,
    )

    # Tight layout and save
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")

    print(f"Correlation plot saved as '{save_path}'")
    print(f"Spearman correlation: ρ = {rho:.4f}, p-value = {p_value:.3e}")

    plt.show()

    return rho, p_value


# Example usage
if __name__ == "__main__":
    # human_rates = [56.35, 54.73, 75.66, 56.63, 59.48, 80.25, 67.31, 57.69, 36.76, 11.90, 47.22, 20.59, 32.94]
    # automated_rates = [42.90, 25.00, 48.77, 37.65, 42.59, 50.00, 41.05, 41.98, 10.80, 0.31, 19.75, 3.70, 21.30]
    # Create the plot

    human_rates = [
        45.0,
        25.0,
        46.6,
        65.5,
        32.3,
        47.7,
        58.7,
        50.1,
        57.8,
        76.8,
        55.1,
        54.2,
        38.3,
        57.7,
        46.5,
        60.8,
        38.5,
        47.0,
        39.5,
        51.7,
    ]

    automated_rates = [
        42.5,
        14.5,
        47.5,
        69.4,
        24.6,
        39.6,
        60.3,
        59.5,
        56.2,
        77.5,
        67.0,
        47.7,
        30.7,
        64.0,
        35.2,
        60.8,
        43.0,
        46.2,
        31.1,
        51.7,
    ]

    rho, p_val = plot_correlation(
        human_rates,
        automated_rates,
        xlabel="Human Win Rate (%)",
        ylabel="Automated Win Rate (%)",
        save_path="human_vs_automated_correlation_chatbotarena.pdf",
    )
