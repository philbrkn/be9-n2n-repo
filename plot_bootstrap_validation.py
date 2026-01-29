import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from bootstrap import SRParams


def plot_validation_figure(all_r_full, all_r_bootstrap_std, k_index=1, label="Doubles"):
    """
    Publication-quality bootstrap validation figure.
    k_index: 1 for doubles (r_1), 2 for triples (r_2)
    """
    plt.rcParams.update(
        {
            "font.size": 11,
            "font.family": "serif",
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.dpi": 300,
        }
    )

    n_reps = len(all_r_full)
    x = np.arange(1, n_reps + 1)  # Start from 1 instead of 0
    values = all_r_full[:, k_index]
    errors = all_r_bootstrap_std[:, k_index]
    mean_val = np.mean(values)

    data_color = "#2E5090"  # Deep blue
    mean_color = "#C1403D"  # Muted red

    fig, ax = plt.subplots(figsize=(8, 5))

    # Plot with improved styling
    ax.errorbar(
        x,
        values,
        yerr=errors,
        fmt="o",
        capsize=4,
        color=data_color,
        ecolor=data_color,
        markersize=5,
        markeredgewidth=0.5,
        markeredgecolor="white",
        linewidth=1.2,
        alpha=0.85,
        label=f"{label} rate",
    )

    ax.axhline(
        mean_val,
        ls="--",
        color=mean_color,
        linewidth=1.5,
        alpha=0.8,
        label=f"Mean = {mean_val:.5f}",
    )

    ax.set_xlabel("Simulation number", fontweight="normal")
    ax.set_ylabel(f"{label} rate $r_{{{k_index}}}$", fontweight="normal")
    # ax.set_title(
    #     f"Bootstrap analysis of {label.lower()} rate for {n_reps} identical simulations",
    #     pad=15,
    # )

    # Cleaner x-axis: show every 5th tick for 50 simulations
    if n_reps == 50:
        tick_positions = np.arange(5, n_reps + 1, 5)
        ax.set_xticks(tick_positions)
        ax.set_xticks(x, minor=True)
    elif n_reps <= 20:
        ax.set_xticks(x)
    else:
        # For other ranges, show every 10th
        tick_positions = np.arange(10, n_reps + 1, 10)
        if 1 not in tick_positions:
            tick_positions = np.insert(tick_positions, 0, 1)
        ax.set_xticks(tick_positions)
        ax.set_xticks(x, minor=True)

    ax.set_xlim(0, n_reps + 1)

    # Add subtle grid
    ax.grid(True, alpha=0.2, linestyle="-", linewidth=0.5)
    ax.set_axisbelow(True)

    # Improve legend
    ax.legend(
        frameon=True, fancybox=False, edgecolor="gray", framealpha=0.95, loc="best"
    )

    # Chi-square annotation with improved styling
    chi2 = np.sum((values - mean_val) ** 2 / errors**2) / (n_reps - 1)
    ax.text(
        0.97,
        0.97,
        f"$\\chi^2_\\nu$ = {chi2:.2f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=11,
        bbox=dict(
            boxstyle="round,pad=0.5",
            facecolor="white",
            edgecolor="gray",
            alpha=0.9,
            linewidth=1,
        ),
    )

    plt.tight_layout()
    plt.savefig(f"{label}_bootstrap_validation.png", dpi=300, bbox_inches="tight")

    return fig


# def plot_validation_figure(all_r_full, all_r_bootstrap_std, k_index=1, label="Doubles"):
#     """
#     Polished validation figure for publication.
#     """
#     # Professional styling
#     plt.rcParams.update(
#         {
#             "font.family": "serif",
#             "font.size": 11,
#             "axes.grid": True,
#             "grid.alpha": 0.3,
#             "grid.linestyle": "--",
#         }
#     )
#
#     n_reps = len(all_r_full)
#     # Start x at 1 for more natural "Simulation Number"
#     x = np.arange(1, n_reps + 1)
#
#     values = all_r_full[:, k_index]
#     errors = all_r_bootstrap_std[:, k_index]
#     mean_val = np.mean(values)
#
#     fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
#
#     # Use a professional color (SlateGray or DarkSlateBlue) instead of default blue
#     main_color = "#2c3e50"
#     mean_color = "#e74c3c"  # Deep crimson for the mean line
#
#     ax.errorbar(
#         x,
#         values,
#         yerr=errors,
#         fmt="o",
#         markersize=4,
#         color=main_color,
#         ecolor=main_color,
#         elinewidth=1,
#         capsize=2,
#         capthick=1,
#         label=f"{label} rate",
#         alpha=0.8,
#     )
#
#     ax.axhline(
#         mean_val,
#         ls="--",
#         lw=1.5,
#         color=mean_color,
#         label=f"Mean: {mean_val:.4f}",
#         zorder=0,
#     )
#
#     # Fix the cluttered x-axis
#     ax.set_xticks(np.arange(0, n_reps + 1, 5))  # Label every 5th simulation
#
#     # Labels and Titles
#     ax.set_xlabel("Simulation Number", fontweight="bold")
#     ax.set_ylabel(f"{label} Rate ($r_{k_index}$)", fontweight="bold")
#     ax.set_title(f"Bootstrap Stability: {label} Rate ({n_reps} Simulations)", pad=15)
#
#     # Clean up the legend
#     ax.legend(frameon=True, loc="lower right")
#
#     # Chi-square annotation (Removed the wheat box for a cleaner look)
#     chi2 = np.sum((values - mean_val) ** 2 / errors**2) / (n_reps - 1)
#     ax.text(
#         0.02,
#         0.95,
#         f"$\chi^2_\\nu$ = {chi2:.2f}",
#         transform=ax.transAxes,
#         fontsize=12,
#         verticalalignment="top",
#         bbox=dict(facecolor="white", alpha=0.5, edgecolor="none"),
#     )
#
#     # Remove top and right spines for a modern look
#     ax.spines["top"].set_visible(False)
#     ax.spines["right"].set_visible(False)
#
#     plt.tight_layout()
#     plt.savefig(
#         f"{label}_bootstrap_validation.pdf", bbox_inches="tight"
#     )  # PDF is better for papers
#     return fig


if __name__ == "__main__":
    base_dir = Path(__file__).parent.resolve()
    output_root = base_dir / "outputs"

    PREDELAY = 4e-6
    GATE = 28e-6
    DELAY = 1000e-6
    MAX_K = 4  # r_0 through r_3

    sr = SRParams(predelay=PREDELAY, gate=GATE, delay=DELAY)

    data = np.load(output_root / "bootstrap_validation.npz")
    all_r_full = data["all_r_full"]
    all_r_bootstrap_std = data["all_r_bootstrap_std"]

    start_time = time.perf_counter()

    # Validation: compare true inter-run variance to bootstrap estimate
    true_std = np.std(all_r_full, axis=0, ddof=1)
    mean_bootstrap_std = np.mean(all_r_bootstrap_std, axis=0)

    print("\n" + "=" * 60)
    print("Bootstrap Validation Results")
    print("=" * 60)
    print(f"{'k':<4} {'True Std':>12} {'Bootstrap Std':>14} {'Ratio':>10}")
    print("-" * 44)
    for k in range(MAX_K):
        ratio = mean_bootstrap_std[k] / true_std[k] if true_std[k] > 1e-10 else np.nan
        print(
            f"{k:<4} {true_std[k]:>12.6f} {mean_bootstrap_std[k]:>14.6f} {ratio:>10.2f}"
        )

    print("\nRatio should be ~1.0 if bootstrap is valid")

    fig1 = plot_validation_figure(
        all_r_full,
        all_r_bootstrap_std,
        k_index=1,
        label="Doubles",
    )
    fig2 = plot_validation_figure(
        all_r_full,
        all_r_bootstrap_std,
        k_index=2,
        label="Triples",
    )

    print(f"Boot strapping took {time.perf_counter() - start_time:.2f} seconds.")
