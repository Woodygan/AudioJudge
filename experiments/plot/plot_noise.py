import matplotlib.pyplot as plt
import numpy as np

# Define the categorical x-axis labels (SNR conditions)
snr_labels = ["∞", "20", "10", "5", "1"]
x = np.arange(len(snr_labels))  # x positions for the categories

# Data for the percentage where the model's judgement remains unchanged (\cmark)
chosen_same = [92.9, 90.1, 90.1, 88.4, 85.3]
not_chosen_same = [96.6, 93.1, 92.5, 94.5, 92.7]
tie_same = [82.2, 79.5, 78.1, 74.0, 72.6]

# Create figure with larger size
plt.figure(figsize=(10, 6))

# Plot with larger markers and line width
plt.plot(
    x,
    chosen_same,
    marker="o",
    linestyle="-",
    label="Chosen (originally)",
    markersize=8,
    linewidth=2.5,
)
plt.plot(
    x,
    not_chosen_same,
    marker="s",
    linestyle="-",
    label="Not-Chosen (originally)",
    markersize=8,
    linewidth=2.5,
)
plt.plot(
    x,
    tie_same,
    marker="^",
    linestyle="-",
    label="Tie (originally)",
    markersize=8,
    linewidth=2.5,
)

# Plot dashed vertical lines at xticks
for i in range(len(x)):
    plt.axvline(x=x[i], color="gray", linestyle="--", linewidth=0.5, alpha=0.5)

# Customize with larger fonts
plt.xticks(x, snr_labels, fontsize=24)
plt.yticks(fontsize=24)
plt.xlabel("SNR (dB) [∞ indicates no noise]", fontsize=30, fontweight="bold")
plt.ylabel("Unchanged Prediction (%)", fontsize=24, fontweight="bold")

# Larger legend
plt.legend(loc="lower left", fontsize=24, frameon=True, fancybox=True, shadow=True)

# Set y-axis limits for better visualization
plt.yticks([60, 70, 80, 90, 100], fontsize=24)

# Add grid for better readability
plt.grid(True, alpha=0.3, linestyle=":")

# Tight layout and save
plt.tight_layout()
plt.savefig(
    "robustness_noise.pdf",
    bbox_inches="tight",
    dpi=300,
    facecolor="white",
    edgecolor="none",
)
plt.show()
