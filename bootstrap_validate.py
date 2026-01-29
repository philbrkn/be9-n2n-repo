import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from bootstrap import SRParams, analyze_with_bootstrap_parallel


def plot_validation_figure(all_r_full, all_r_bootstrap_std, k_index=1, label="Doubles"):
    """
    Recreate Ridnik Fig. 1: point estimates with bootstrap error bars.

    k_index: 1 for doubles (r_1), 2 for triples (r_2)
    """
    n_reps = len(all_r_full)
    x = np.arange(n_reps)

    values = all_r_full[:, k_index]
    errors = all_r_bootstrap_std[:, k_index]
    mean_val = np.mean(values)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.errorbar(x, values, yerr=errors, fmt="o", capsize=3, label=f"{label} rate")
    ax.axhline(mean_val, ls="--", color="red", label=f"Mean = {mean_val:.5f}")

    ax.set_xlabel("Simulation number")
    ax.set_ylabel(f"{label} rate $r_{k_index}$")
    ax.set_title(
        f"Bootstrap analysis of {label} rate for {n_reps} identical simulations"
    )
    ax.legend()
    ax.set_xticks(x)

    # Add chi-square annotation
    chi2 = np.sum((values - mean_val) ** 2 / errors**2) / (n_reps - 1)
    ax.text(
        0.95,
        0.95,
        f"$\\chi^2_\\nu$ = {chi2:.2f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        bbox=dict(boxstyle="round", facecolor="wheat"),
    )

    plt.tight_layout()
    plt.savefig(f"{label}_bootstrap_validation.png")
    return fig


if __name__ == "__main__":
    base_dir = Path(__file__).parent.resolve()
    output_root = base_dir / "outputs"

    PREDELAY = 4e-6
    GATE = 28e-6
    DELAY = 1000e-6
    MAX_K = 4  # r_0 through r_3

    sr = SRParams(predelay=PREDELAY, gate=GATE, delay=DELAY)

    rep_dirs = sorted([p for p in output_root.glob("rep_*") if p.is_dir()])
    if not rep_dirs:
        raise FileNotFoundError(f"No replicate dirs found in {output_root}")

    # Collect results from each replicate
    all_r_full = []
    all_r_bootstrap_std = []

    start_time = time.perf_counter()

    for rep in rep_dirs:
        h5 = rep / "collision_track.h5"
        if not h5.exists():
            continue

        print(f"\nProcessing {rep.name}...")
        r_full, r_mean, r_std, arr = analyze_with_bootstrap_parallel(
            h5,
            sr,
            n_bootstrap=1000,
            segment_duration=10.0,
            seed=12346,
            n_workers=32,
        )

        # Pad to consistent length
        r_full_padded = (
            np.pad(r_full, (0, MAX_K - len(r_full)))
            if len(r_full) < MAX_K
            else r_full[:MAX_K]
        )
        r_std_padded = (
            np.pad(r_std, (0, MAX_K - len(r_std)))
            if len(r_std) < MAX_K
            else r_std[:MAX_K]
        )

        all_r_full.append(r_full_padded)
        all_r_bootstrap_std.append(r_std_padded)

    all_r_full = np.array(all_r_full)  # shape: (n_reps, MAX_K)
    all_r_bootstrap_std = np.array(all_r_bootstrap_std)
    np.savez(
        output_root / "bootstrap_validation.npz",
        all_r_full=all_r_full,  # (50, 4) = 200 floats
        all_r_bootstrap_std=all_r_bootstrap_std,  # (50, 4) = 200 floats
        # If we want full bootstrap distributions for Fig 2 style plots:
        # arr=arr,  # (1000, 4) = 4000 floats per replicate, maybe skip
    )

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

    # Usage
    fig1 = plot_validation_figure(
        all_r_full, all_r_bootstrap_std, k_index=1, label="Doubles"
    )
    fig2 = plot_validation_figure(
        all_r_full, all_r_bootstrap_std, k_index=2, label="Triples"
    )

    print(f"Boot strapping took {time.perf_counter() - start_time:.2f} seconds.")
