import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

mos_diff_avg = [0.125, 0.375, 0.625, 0.875, 1.125, 1.375, 1.809]
accuracy = [51.65, 56.03, 56.71, 64.02, 67.53, 70.40, 76.01]
consistency = [50.74, 53.47, 52.64, 57.32, 60.89, 64.94, 73.65]
inconsistency = [100 - x for x in consistency]

# Create figure with larger size
plt.figure(figsize=(10, 7))

# Larger scatter points
plt.scatter(
    accuracy, inconsistency, color="indianred", label="Data points", s=150, alpha=0.8
)

# Compute linear regression
slope, intercept, r_value, p_value, std_err = linregress(accuracy, inconsistency)

# Create fitted line
x_vals = np.array(accuracy)
y_fit = intercept + slope * x_vals

# Sort points for a clean line plot
sort_idx = np.argsort(x_vals)
x_sorted = x_vals[sort_idx]
y_sorted = y_fit[sort_idx]

plt.plot(
    x_sorted,
    y_sorted,
    "--",
    color="indianred",
    linewidth=2.5,
    label=f"slope={slope:.2f}\np={p_value:.1e}",
)

# Customize with larger fonts
plt.xlabel("Accuracy on subsets (split by ΔMOS)", fontsize=24, fontweight="bold")
plt.ylabel("Positional Bias (%)", fontsize=24, fontweight="bold")
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)

plt.ylim(22, 52)

# Larger legend
plt.legend(fontsize=24, frameon=True, fancybox=True, shadow=True, loc="upper right")

# Add grid for better readability
plt.grid(True, alpha=0.3, linestyle=":")

# Tight layout
plt.tight_layout()

# Also print out the regression statistics to the console
print(f"Slope: {slope:.5f}")
print(f"Intercept: {intercept:.5f}")
print(f"R-squared: {r_value**2:.5f}")
print(f"P-value (slope): {p_value:.5e}")
print(f"Std. Err. of slope: {std_err:.5f}")

plt.savefig(
    "position_bias_vs_accuracy.pdf",
    bbox_inches="tight",
    dpi=300,
    facecolor="white",
    edgecolor="none",
)
plt.show()
