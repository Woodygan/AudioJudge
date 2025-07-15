import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib import rcParams
from matplotlib import rc
from matplotlib import font_manager


def generate_comprehensive_heatmap(results_df, result_dir, model_name):
    """
    Create comprehensive heatmaps for pronunciation comparison results with all configurations.

    Args:
        results_df: DataFrame containing all results
        result_dir: Directory to save the output
        model_name: Name of the model to include in plot title
    """
    # Create a directory for plots if it doesn't exist
    plots_dir = os.path.join(result_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Add a configuration column to identify the type of test setup
    if "two_turns" in results_df.columns:
        # Determine if we need to create a fewshot_config column
        if "aggregate_fewshot" in results_df.columns:
            results_df["config_type"] = results_df.apply(
                lambda row: "Two Turns"
                if row["two_turns"]
                else (
                    "Aggregate"
                    if row["aggregate_fewshot"]
                    else ("Concat" if row["concat_fewshot"] else "Separate")
                ),
                axis=1,
            )
        else:
            results_df["config_type"] = results_df.apply(
                lambda row: "Two Turns"
                if row["two_turns"]
                else ("Concat" if row["concat_fewshot"] else "Separate"),
                axis=1,
            )
    else:
        # Legacy format compatibility
        if "aggregate_fewshot" in results_df.columns:
            results_df["config_type"] = results_df.apply(
                lambda row: "Aggregate"
                if row["aggregate_fewshot"]
                else ("Concat" if row["concat_fewshot"] else "Separate"),
                axis=1,
            )
        else:
            results_df["config_type"] = results_df.apply(
                lambda row: "Concat" if row["concat_fewshot"] else "Separate", axis=1
            )

    # Create comparison plots for different configurations

    # 1. Compare configuration types (Two Turns vs. Aggregate vs. Concat vs. Separate)
    for transcript_type in results_df["transcript_type"].unique():
        # For two turns vs other configs, we need to handle separately
        # because two_turns doesn't use concat_test
        config_data = []

        # Get data for each configuration and shot count
        for n_shots in sorted(results_df["n_shots"].unique()):
            row_data = {"n_shots": n_shots}

            # Get data for Two Turns if it exists
            if "Two Turns" in results_df["config_type"].unique():
                two_turns_data = results_df[
                    (results_df["transcript_type"] == transcript_type)
                    & (results_df["config_type"] == "Two Turns")
                    & (results_df["n_shots"] == n_shots)
                ]

                if len(two_turns_data) > 0:
                    row_data["Two Turns"] = two_turns_data["accuracy"].mean()
                elif n_shots == 0:
                    # For 0 shots, use separate_separate as placeholder
                    placeholder_data = results_df[
                        (results_df["transcript_type"] == transcript_type)
                        & (results_df["config_type"] == "Separate")
                        & (results_df["concat_test"] == False)
                        & (results_df["n_shots"] == 0)
                    ]
                    if len(placeholder_data) > 0:
                        row_data["Two Turns"] = placeholder_data["accuracy"].mean()

            # Get data for other configs
            for config_type in [
                ct for ct in results_df["config_type"].unique() if ct != "Two Turns"
            ]:
                for concat_test in results_df["concat_test"].unique():
                    config_data_filtered = results_df[
                        (results_df["transcript_type"] == transcript_type)
                        & (results_df["config_type"] == config_type)
                        & (results_df["concat_test"] == concat_test)
                        & (results_df["n_shots"] == n_shots)
                    ]

                    if len(config_data_filtered) > 0:
                        config_key = (
                            f"{config_type}_{('Concat' if concat_test else 'Separate')}"
                        )
                        row_data[config_key] = config_data_filtered["accuracy"].mean()
                    elif n_shots == 0 and config_type != "Separate":
                        # For 0 shots with aggregate or concat, use separate as placeholder
                        placeholder_data = results_df[
                            (results_df["transcript_type"] == transcript_type)
                            & (results_df["config_type"] == "Separate")
                            & (results_df["concat_test"] == concat_test)
                            & (results_df["n_shots"] == 0)
                        ]
                        if len(placeholder_data) > 0:
                            config_key = f"{config_type}_{('Concat' if concat_test else 'Separate')}"
                            row_data[config_key] = placeholder_data["accuracy"].mean()

            # Calculate average for this row (excluding n_shots column)
            config_vals = [
                v
                for k, v in row_data.items()
                if k != "n_shots" and isinstance(v, (int, float))
            ]
            if config_vals:
                row_data["Avg"] = sum(config_vals) / len(config_vals)

            config_data.append(row_data)

        if len(config_data) <= 1:
            continue

        # Convert to DataFrame for pivot
        config_df = pd.DataFrame(config_data)

        # Check if we have enough data to proceed
        if len(config_df.columns) <= 2:  # Only n_shots and maybe Avg
            continue

        # Calculate column averages (across shot counts)
        avg_row = {"n_shots": "Avg"}
        for col in config_df.columns:
            if col != "n_shots":
                values = [v for v in config_df[col] if isinstance(v, (int, float))]
                if values:
                    avg_row[col] = sum(values) / len(values)

        # Add average row
        config_df = pd.concat([config_df, pd.DataFrame([avg_row])], ignore_index=True)

        # Create pivot table
        pivot_df = config_df.set_index("n_shots")

        # Create the heatmap
        plt.figure(figsize=(14, 10), facecolor="#f9f9f9")

        # Use a sequential colormap
        cmap = sns.color_palette("coolwarm", as_cmap=True)

        # Get numeric values for vmin/vmax
        numeric_values = []
        for col in pivot_df.columns:
            numeric_values.extend(
                [v for v in pivot_df[col] if isinstance(v, (int, float))]
            )

        if not numeric_values:
            continue

        # Set vmin and vmax based on the data
        vmin = max(0.3, min(numeric_values) - 0.1)  # Lower bound
        vmax = min(0.9, max(numeric_values) + 0.1)  # Upper bound

        # Custom annotation formatter for percentage display
        def pct_formatter(val):
            if isinstance(val, (int, float)):
                return f"{val * 100:.1f}%"
            return ""

        # Create annotation array
        annot_array = np.empty_like(pivot_df.values, dtype=object)
        for i in range(pivot_df.shape[0]):
            for j in range(pivot_df.shape[1]):
                val = pivot_df.iloc[i, j]
                if isinstance(val, (int, float)):
                    annot_array[i, j] = pct_formatter(val)
                else:
                    annot_array[i, j] = ""

        # Plot the heatmap
        ax = sns.heatmap(
            pivot_df,
            cmap=cmap,
            center=0.5,
            vmin=vmin,
            vmax=vmax,
            annot=annot_array,
            fmt="",
            linewidths=0.5,
            annot_kws={"size": 12, "weight": "bold"},
            cbar_kws={"label": "Accuracy", "shrink": 0.8},
        )

        transcript_name = (
            transcript_type.capitalize()
            if transcript_type != "none"
            else "No Transcript"
        )
        plt.title(
            f"Configuration Comparison\n{transcript_name}\n{model_name}",
            fontsize=18,
            fontweight="bold",
        )
        plt.xlabel("Configuration Type", fontsize=16, fontweight="bold")
        plt.ylabel("Number of Shots ($k$)", fontsize=16, fontweight="bold")

        # Move x-axis to the top
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position("top")

        # Adjust x labels for better readability
        plt.xticks(rotation=45, ha="left")
        plt.yticks(rotation=0)

        # Make tick labels bold
        ax.set_xticklabels(ax.get_xticklabels(), fontweight="bold")
        ax.set_yticklabels(ax.get_yticklabels(), fontweight="bold")

        # Highlight average row and column
        # Add a thicker line for the average row (bottom row)
        ax.axhline(y=pivot_df.shape[0] - 1, color="black", linewidth=2)
        # Add a thicker line for the average column (last column)
        if "Avg" in pivot_df.columns:
            avg_col_idx = pivot_df.columns.get_loc("Avg")
            ax.axvline(x=avg_col_idx, color="black", linewidth=2)

        # Save the figure
        filename_base = f"config_comparison_heatmap_{transcript_type}"
        plt.tight_layout()
        plt.savefig(
            os.path.join(plots_dir, f"{filename_base}.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.savefig(
            os.path.join(plots_dir, f"{filename_base}.pdf"), bbox_inches="tight"
        )
        plt.close()
    for transcript_type in results_df["transcript_type"].unique():
        for config_type in results_df["config_type"].unique():
            # Two Turns is a separate config that doesn't use concat_test
            if config_type == "Two Turns":
                filtered_df = results_df[
                    (results_df["transcript_type"] == transcript_type)
                    & (results_df["config_type"] == config_type)
                ]

                if len(filtered_df) == 0:
                    continue

                # Create a meaningful title
                transcript_name = (
                    transcript_type.capitalize()
                    if transcript_type != "none"
                    else "No Transcript"
                )
                config_name = "Two Turns Mode"

                # Create pivot table for heatmap visualization
                pivot = filtered_df.pivot_table(
                    index="prompt_type",
                    columns="n_shots",
                    values="accuracy",
                    aggfunc="mean",
                ).fillna(0)

                # Calculate average for each row (prompt_type)
                if "Avg" not in pivot.columns:
                    pivot["Avg"] = pivot.mean(axis=1)

                # Calculate average for each column (n_shots)
                avg_row = pivot.mean()
                if "Avg" not in pivot.index:
                    pivot.loc["Avg"] = avg_row

                # Create the heatmap
                plt.figure(figsize=(14, 10), facecolor="#f9f9f9")

                # Use a sequential colormap for accuracy values (higher is better)
                cmap = sns.color_palette("coolwarm", as_cmap=True)

                # Set vmin and vmax based on the data
                vmin = max(0.3, filtered_df["accuracy"].min() - 0.1)  # Lower bound
                vmax = min(0.9, filtered_df["accuracy"].max() + 0.1)  # Upper bound

                # Custom annotation formatter for percentage display
                def pct_formatter(val):
                    return f"{val * 100:.1f}%"

                # Create annotation array
                annot_array = np.empty_like(pivot.values, dtype=object)
                for i in range(pivot.shape[0]):
                    for j in range(pivot.shape[1]):
                        annot_array[i, j] = pct_formatter(pivot.iloc[i, j])

                # Plot the heatmap
                ax = sns.heatmap(
                    pivot,
                    cmap=cmap,
                    center=0.5,
                    vmin=vmin,
                    vmax=vmax,
                    annot=annot_array,
                    fmt="",
                    linewidths=0.5,
                    annot_kws={"size": 12, "weight": "bold"},
                    cbar_kws={"label": "Accuracy", "shrink": 0.8},
                )

                # Configure the plot
                plt.title(
                    f"{model_name}: Pronunciation Comparison Accuracy\n{transcript_name} - {config_name}",
                    fontsize=18,
                    fontweight="bold",
                )
                plt.xlabel("Number of Shots ($k$)", fontsize=16, fontweight="bold")
                plt.ylabel("Prompt Type", fontsize=16, fontweight="bold")

                # Move x-axis to the top
                ax.xaxis.tick_top()
                ax.xaxis.set_label_position("top")

                # Remove x and y rotation
                plt.xticks(rotation=0)
                plt.yticks(rotation=0)

                # Make tick labels bold
                ax.set_xticklabels(ax.get_xticklabels(), fontweight="bold")
                ax.set_yticklabels(ax.get_yticklabels(), fontweight="bold")

                # Highlight average row and column
                # Add a thicker line for the average row (bottom row)
                ax.axhline(y=pivot.shape[0] - 1, color="black", linewidth=2)
                # Add a thicker line for the average column (last column)
                ax.axvline(x=pivot.shape[1] - 1, color="black", linewidth=2)

                # Get the colorbar and set its label with larger font size
                cbar = ax.collections[0].colorbar
                cbar.ax.set_ylabel("Accuracy", fontsize=14, fontweight="bold")

                # Save the figure
                filename_base = f"heatmap_{transcript_type}_two_turns"
                plt.tight_layout()
                plt.savefig(
                    os.path.join(plots_dir, f"{filename_base}.png"),
                    dpi=300,
                    bbox_inches="tight",
                )
                plt.savefig(
                    os.path.join(plots_dir, f"{filename_base}.pdf"), bbox_inches="tight"
                )
                plt.close()

            else:
                # Handle other configurations (Aggregate, Concat, Separate)
                for concat_test in results_df["concat_test"].unique():
                    filtered_df = results_df[
                        (results_df["transcript_type"] == transcript_type)
                        & (results_df["config_type"] == config_type)
                        & (results_df["concat_test"] == concat_test)
                    ]

                    if len(filtered_df) == 0:
                        continue

                    # Create meaningful names for the configuration
                    transcript_name = (
                        transcript_type.capitalize()
                        if transcript_type != "none"
                        else "No Transcript"
                    )
                    fewshot_name = f"{config_type} Few-shot"
                    test_name = "Concatenated Test" if concat_test else "Separate Test"

                    # Create pivot table for heatmap visualization
                    pivot = filtered_df.pivot_table(
                        index="prompt_type",
                        columns="n_shots",
                        values="accuracy",
                        aggfunc="mean",
                    ).fillna(0)

                    # Calculate average for each row (prompt_type)
                    if "Avg" not in pivot.columns:
                        pivot["Avg"] = pivot.mean(axis=1)

                    # Calculate average for each column (n_shots)
                    avg_row = pivot.mean()
                    if "Avg" not in pivot.index:
                        pivot.loc["Avg"] = avg_row

                    # Create the heatmap
                    plt.figure(figsize=(14, 10), facecolor="#f9f9f9")

                    # Use a sequential colormap for accuracy values (higher is better)
                    cmap = sns.color_palette("coolwarm", as_cmap=True)

                    # Set vmin and vmax based on the data
                    vmin = max(0.3, filtered_df["accuracy"].min() - 0.1)  # Lower bound
                    vmax = min(0.9, filtered_df["accuracy"].max() + 0.1)  # Upper bound

                    # Custom annotation formatter for percentage display
                    def pct_formatter(val):
                        return f"{val * 100:.1f}%"

                    # Create annotation array
                    annot_array = np.empty_like(pivot.values, dtype=object)
                    for i in range(pivot.shape[0]):
                        for j in range(pivot.shape[1]):
                            annot_array[i, j] = pct_formatter(pivot.iloc[i, j])

                    # Plot the heatmap
                    ax = sns.heatmap(
                        pivot,
                        cmap=cmap,
                        center=0.5,
                        vmin=vmin,
                        vmax=vmax,
                        annot=annot_array,
                        fmt="",
                        linewidths=0.5,
                        annot_kws={"size": 12, "weight": "bold"},
                        cbar_kws={"label": "Accuracy", "shrink": 0.8},
                    )

                    # Configure the plot
                    plt.title(
                        f"{model_name}: Pronunciation Comparison Accuracy\n{transcript_name} - {fewshot_name} - {test_name}",
                        fontsize=18,
                        fontweight="bold",
                    )
                    plt.xlabel("Number of Shots ($k$)", fontsize=16, fontweight="bold")
                    plt.ylabel("Prompt Type", fontsize=16, fontweight="bold")

                    # Move x-axis to the top
                    ax.xaxis.tick_top()
                    ax.xaxis.set_label_position("top")

                    # Remove x and y rotation
                    plt.xticks(rotation=0)
                    plt.yticks(rotation=0)

                    # Make tick labels bold
                    ax.set_xticklabels(ax.get_xticklabels(), fontweight="bold")
                    ax.set_yticklabels(ax.get_yticklabels(), fontweight="bold")

                    # Highlight average row and column
                    # Add a thicker line for the average row (bottom row)
                    ax.axhline(y=pivot.shape[0] - 1, color="black", linewidth=2)
                    # Add a thicker line for the average column (last column)
                    ax.axvline(x=pivot.shape[1] - 1, color="black", linewidth=2)

                    # Get the colorbar and set its label with larger font size
                    cbar = ax.collections[0].colorbar
                    cbar.ax.set_ylabel("Accuracy", fontsize=14, fontweight="bold")

                    # Save the figure
                    filename_base = f"heatmap_{transcript_type}_fewshot_{config_type.lower()}_test_{'concat' if concat_test else 'separate'}"
                    plt.tight_layout()
                    plt.savefig(
                        os.path.join(plots_dir, f"{filename_base}.png"),
                        dpi=300,
                        bbox_inches="tight",
                    )
                    plt.savefig(
                        os.path.join(plots_dir, f"{filename_base}.pdf"),
                        bbox_inches="tight",
                    )
                    plt.close()

    # 2. Compare transcript types for each configuration type
    for config_type in results_df["config_type"].unique():
        if config_type == "Two Turns":
            # For Two Turns, no concat_test dimension
            transcript_comparison = {}
            for transcript_type in results_df["transcript_type"].unique():
                filtered_df = results_df[
                    (results_df["transcript_type"] == transcript_type)
                    & (results_df["config_type"] == config_type)
                ]

                if len(filtered_df) > 0:
                    transcript_comparison[transcript_type] = filtered_df[
                        "accuracy"
                    ].mean()

            if len(transcript_comparison) <= 1:
                continue

            # Create bar chart for transcript comparison
            plt.figure(figsize=(10, 6), facecolor="#f9f9f9")

            # Sort by accuracy
            sorted_transcripts = sorted(
                transcript_comparison.items(), key=lambda x: x[1], reverse=True
            )
            x = [
                t.capitalize() if t != "none" else "No Transcript"
                for t, _ in sorted_transcripts
            ]
            y = [acc for _, acc in sorted_transcripts]

            bars = plt.bar(x, y, color=sns.color_palette("viridis", len(x)))

            # Add value labels on top of each bar
            for bar in bars:
                height = bar.get_height()
                plt.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + 0.01,
                    f"{height:.1%}",
                    ha="center",
                    va="bottom",
                    fontsize=12,
                    fontweight="bold",
                )

            plt.title(
                f"Transcript Type Comparison\n{config_type} Configuration\n{model_name}",
                fontsize=16,
                fontweight="bold",
            )
            plt.ylabel("Average Accuracy", fontsize=14, fontweight="bold")
            plt.ylim(0, max(y) + 0.1)
            plt.grid(axis="y", linestyle="--", alpha=0.7)

            # Save the figure
            filename_base = f"transcript_comparison_{config_type.lower()}"
            plt.tight_layout()
            plt.savefig(
                os.path.join(plots_dir, f"{filename_base}.png"),
                dpi=300,
                bbox_inches="tight",
            )
            plt.savefig(
                os.path.join(plots_dir, f"{filename_base}.pdf"), bbox_inches="tight"
            )
            plt.close()
        else:
            # For other config types, include concat_test dimension
            for concat_test in results_df["concat_test"].unique():
                transcript_comparison = {}
                for transcript_type in results_df["transcript_type"].unique():
                    filtered_df = results_df[
                        (results_df["transcript_type"] == transcript_type)
                        & (results_df["config_type"] == config_type)
                        & (results_df["concat_test"] == concat_test)
                    ]

                    if len(filtered_df) > 0:
                        transcript_comparison[transcript_type] = filtered_df[
                            "accuracy"
                        ].mean()

                if len(transcript_comparison) <= 1:
                    continue

                # Create bar chart for transcript comparison
                plt.figure(figsize=(10, 6), facecolor="#f9f9f9")

                # Sort by accuracy
                sorted_transcripts = sorted(
                    transcript_comparison.items(), key=lambda x: x[1], reverse=True
                )
                x = [
                    t.capitalize() if t != "none" else "No Transcript"
                    for t, _ in sorted_transcripts
                ]
                y = [acc for _, acc in sorted_transcripts]

                bars = plt.bar(x, y, color=sns.color_palette("viridis", len(x)))

                # Add value labels on top of each bar
                for bar in bars:
                    height = bar.get_height()
                    plt.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        height + 0.01,
                        f"{height:.1%}",
                        ha="center",
                        va="bottom",
                        fontsize=12,
                        fontweight="bold",
                    )

                fewshot_name = f"{config_type} Few-shot"
                test_name = "Concatenated Test" if concat_test else "Separate Test"

                plt.title(
                    f"Transcript Type Comparison\n{fewshot_name} - {test_name}\n{model_name}",
                    fontsize=16,
                    fontweight="bold",
                )
                plt.ylabel("Average Accuracy", fontsize=14, fontweight="bold")
                plt.ylim(0, max(y) + 0.1)
                plt.grid(axis="y", linestyle="--", alpha=0.7)

                # Save the figure
                filename_base = f"transcript_comparison_{config_type.lower()}_test_{'concat' if concat_test else 'separate'}"
                plt.tight_layout()
                plt.savefig(
                    os.path.join(plots_dir, f"{filename_base}.png"),
                    dpi=300,
                    bbox_inches="tight",
                )
                plt.savefig(
                    os.path.join(plots_dir, f"{filename_base}.pdf"), bbox_inches="tight"
                )
                plt.close()

    # Create a comprehensive configuration comparison heatmap
    # This will show all configurations and their relative performance
    print("\nGenerating comprehensive configuration comparison heatmap...")

    # Build a matrix of configurations x transcript types with accuracy values
    config_matrix = []

    # Two turns configurations
    if "Two Turns" in results_df["config_type"].unique():
        for transcript_type in results_df["transcript_type"].unique():
            filtered_df = results_df[
                (results_df["transcript_type"] == transcript_type)
                & (results_df["config_type"] == "Two Turns")
            ]

            if len(filtered_df) > 0:
                transcript_name = (
                    transcript_type.capitalize()
                    if transcript_type != "none"
                    else "No Transcript"
                )
                avg_accuracy = filtered_df["accuracy"].mean()
                config_matrix.append(
                    {
                        "transcript": transcript_name,
                        "config": "Two Turns",
                        "accuracy": avg_accuracy,
                    }
                )

    # Other configurations with concat_test dimension
    for config_type in [
        ct for ct in results_df["config_type"].unique() if ct != "Two Turns"
    ]:
        for concat_test in results_df["concat_test"].unique():
            for transcript_type in results_df["transcript_type"].unique():
                filtered_df = results_df[
                    (results_df["transcript_type"] == transcript_type)
                    & (results_df["config_type"] == config_type)
                    & (results_df["concat_test"] == concat_test)
                ]

                if len(filtered_df) > 0:
                    transcript_name = (
                        transcript_type.capitalize()
                        if transcript_type != "none"
                        else "No Transcript"
                    )
                    config_name = (
                        f"{config_type}_{('Concat' if concat_test else 'Separate')}"
                    )
                    avg_accuracy = filtered_df["accuracy"].mean()
                    config_matrix.append(
                        {
                            "transcript": transcript_name,
                            "config": config_name,
                            "accuracy": avg_accuracy,
                        }
                    )

    if len(config_matrix) > 0:
        config_df = pd.DataFrame(config_matrix)

        # Create pivot table for heatmap
        pivot = config_df.pivot_table(
            index="transcript", columns="config", values="accuracy"
        ).fillna(0)

        # Create the comprehensive heatmap
        plt.figure(figsize=(16, len(pivot) * 1.2 + 2), facecolor="#f9f9f9")

        # Use a sequential colormap
        cmap = sns.color_palette("coolwarm", as_cmap=True)

        # Set vmin and vmax based on the data
        vmin = max(0.3, config_df["accuracy"].min() - 0.1)
        vmax = min(0.9, config_df["accuracy"].max() + 0.1)

        # Custom annotation formatter for percentage display
        def pct_formatter(val):
            return f"{val * 100:.1f}%"

        # Create annotation array
        annot_array = np.empty_like(pivot.values, dtype=object)
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                annot_array[i, j] = pct_formatter(pivot.iloc[i, j])

        # Plot the heatmap
        ax = sns.heatmap(
            pivot,
            cmap=cmap,
            center=0.5,
            vmin=vmin,
            vmax=vmax,
            annot=annot_array,
            fmt="",
            linewidths=0.5,
            annot_kws={"size": 12, "weight": "bold"},
            cbar_kws={"label": "Accuracy", "shrink": 0.8},
        )

        # Configure the plot
        plt.title(
            f"{model_name}: Comprehensive Configuration Comparison\nTranscript Type vs. Configuration",
            fontsize=20,
            fontweight="bold",
        )
        plt.xlabel("Configuration", fontsize=16, fontweight="bold")
        plt.ylabel("Transcript Type", fontsize=16, fontweight="bold")

        # Adjust x labels for better readability
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)

        # Make tick labels bold
        ax.set_xticklabels(ax.get_xticklabels(), fontweight="bold")
        ax.set_yticklabels(ax.get_yticklabels(), fontweight="bold")

        # Save the figure
        filename_base = "comprehensive_config_comparison"
        plt.tight_layout()
        plt.savefig(
            os.path.join(plots_dir, f"{filename_base}.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.savefig(
            os.path.join(plots_dir, f"{filename_base}.pdf"), bbox_inches="tight"
        )
        plt.close()

        print(
            f"Comprehensive configuration comparison heatmap saved to {plots_dir}/{filename_base}.png"
        )

    print(f"\nAll visualizations saved to {plots_dir}/")
