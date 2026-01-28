#!/usr/bin/env python3
"""
plot_sweep.py

Parse an existing sweep directory (produced by run_sweep/run_independent_replicates)
and generate the same style plots WITHOUT running OpenMC.

Assumptions about folder layout:
  outputs/<label>/rep_XXXX/collision_track.h5

We re-run only the lightweight SR post-processing:
  - read collision_track.h5
  - select MT=101 events
  - build detection_times
  - compute sr_counts / sr_counts_delayed
  - deconvolve to r (get_measured_multiplicity_causal)

Usage examples:
  python plot_sweep.py --outputs outputs --pattern "be_radius_*" --x-from-label float --xlabel "Be radius (cm)" --out figures/be_radius_replot.png
  python plot_sweep.py --outputs outputs --pattern "n2n_scale_*" --x-from-label float --xlabel "(n,2n) scale factor" --out figures/n2n_scale_replot.png

If your labels are like "be_radius_9.0cm" and "n2n_scale_1.10", x parsing handles both.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np

from bootstrap import analyze_with_bootstrap_parallel


@dataclass(frozen=True)
class SRParams:
    predelay: float
    gate: float
    delay: float


def list_sweep_points(outputs: Path, pattern: str) -> List[Path]:
    pts = sorted(outputs.glob(pattern))
    pts = [p for p in pts if p.is_dir() and (p / "rep_0000").exists()]
    if not pts:
        raise FileNotFoundError(
            f"No sweep points found under {outputs} matching '{pattern}'"
        )
    return pts


_num_re = re.compile(r"_(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)(?:[A-Za-z]+)?$")


def point_x(p: Path) -> float:
    m = _num_re.search(p.name)
    if not m:
        raise ValueError(f"Could not parse sweep value from: {p.name}")
    return float(m.group(1))


def plot_sensitivity_results(
    x: np.ndarray,
    r_mean_list: Sequence[np.ndarray],
    r_sem_list: Sequence[np.ndarray],
    xlabel: str,
    out: Path,
    which_r_k: int = 3,
    N_REPS: int = None,
    N_PARTICLES: int = None,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))

    means = [r[which_r_k] if len(r) > which_r_k else np.nan for r in r_mean_list]
    sems = [s[which_r_k] if len(s) > which_r_k else np.nan for s in r_sem_list]

    ax.errorbar(x, means, yerr=sems, fmt="o-", capsize=5, capthick=2, markersize=8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(f"$r_{which_r_k}$")
    ax.grid(True, alpha=0.3)

    # Fit a line to get sensitivity
    coeffs = np.polyfit(x, means, 1)
    slope = coeffs[0]
    intercept = coeffs[1]

    # Sensitivity: (dr/r) / (dXS/XS) at nominal
    r_nominal = means[2]  # this is where scale=1.0
    sem_nominal = sems[2]
    S = slope * x[2] / r_nominal  # dr/dXS normalized

    print(f"Sensitivity coefficient S = {S:.3f}")
    print(
        f"Interpretation: 1% change in (n,2n) causes {S:.2f}% change in r[{which_r_k}]"
    )
    # invert for constraint:
    # if we measure r[1] with precision delr[1] we constrain XS(n,2n) to:
    # deltaXS / XS = (1/S) * (deltar[1]/r[1])
    precision = sem_nominal / r_nominal
    constraint = 1 / S * precision
    print(
        f"A shift register measurement of r[{which_r_k}] with {precision * 100:.2f}% precision"
        f" (achievable with {N_REPS} replicates of {N_PARTICLES} particles) "
        f"can constrain the Be-9 (n,2n) cross section to ±{constraint * 100:.4f}%."
    )
    ax.plot(
        x,
        np.polyval(coeffs, x),
        "r--",
        linewidth=2,
        label=f"Linear fit (slope={slope:.4f})",
        zorder=2,
    )
    # INVERSE BAND  #
    # Horizontal band: r[1] measurement with uncertainty
    r_low = r_nominal - sem_nominal
    r_high = r_nominal + sem_nominal
    # Map to scale via inverse: scale = (r[1] - intercept) / slope
    scale_low = (r_low - intercept) / slope
    scale_high = (r_high - intercept) / slope
    # Shade the horizontal band (measurement uncertainty on r[1])
    ax.axhspan(
        r_low,
        r_high,
        alpha=0.3,
        color="green",
        label=f"r[1] = {r_nominal:.4f} ± {sem_nominal:.4f}",
    )
    # Shade the vertical band (implied constraint on scale)
    ax.axvspan(
        scale_low,
        scale_high,
        alpha=0.2,
        color="orange",
        label=f"Scale = 1.00 ± {abs(scale_high - 1.0):.3f}",
    )
    ax.legend()
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, dpi=300)
    plt.close(fig)


def plot_results(
    x: np.ndarray,
    r_mean_list: Sequence[np.ndarray],
    r_sem_list: Sequence[np.ndarray],
    # det_mean: np.ndarray,
    # det_sem: np.ndarray,
    xlabel: str,
    out: Path,
    max_r_k: int = 3,
    fit_sensitivity: bool = False,
    N_PARTICLES: int = None,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # r[0], r[1], r[2] ... up to max_r_k shown in first panels
    for idx, ax in enumerate(axes.flat[:4]):
        if idx > max_r_k:
            ax.axis("off")
            continue

        means = [r[idx] if len(r) > idx else np.nan for r in r_mean_list]
        sems = [s[idx] if len(s) > idx else np.nan for s in r_sem_list]

        ax.errorbar(x, means, yerr=sems, fmt="o-", capsize=5, capthick=2, markersize=8)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(f"$r_{idx}$")
        ax.grid(True, alpha=0.3)

        if fit_sensitivity:
            # Fit a line to get sensitivity
            coeffs = np.polyfit(x, means, 1)
            slope = coeffs[0]
            # intercept = coeffs[1]

            # Sensitivity: (dr/r) / (dXS/XS) at nominal
            r_nominal = means[2]  # this is where scale=1.0
            sem_nominal = sems[2]
            S = slope * x[2] / r_nominal  # dr/dXS normalized

            print(f"Sensitivity coefficient S = {S:.3f}")
            print(
                f"Interpretation: 1% change in (n,2n) causes {S:.2f}% change in r[{idx}]"
            )
            # invert for constraint:
            # if we measure r[1] with precision delr[1] we constrain XS(n,2n) to:
            # deltaXS / XS = (1/S) * (deltar[1]/r[1])
            precision = sem_nominal / r_nominal
            constraint = 1 / S * precision
            print(
                f"A shift register measurement of r[{idx}] with {precision * 100:.2f}% precision"
                f" (achievable with {1} replicates of {N_PARTICLES} particles) "
                f"can constrain the Be-9 (n,2n) cross section to ±{constraint * 100:.4f}%."
            )

    # detections
    # ax = axes[1, 1]
    # ax.errorbar(
    #     x, det_mean, yerr=det_sem, fmt="s-", capsize=5, capthick=2, markersize=8
    # )
    # ax.set_xlabel(xlabel)
    # ax.set_ylabel("Detections per replicate")
    # ax.grid(True, alpha=0.3)

    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, dpi=300)
    plt.close(fig)


def print_sweep_table(
    x: np.ndarray,
    r_mean_list: Sequence[np.ndarray],
    r_sem_list: Sequence[np.ndarray],
    # det_mean: np.ndarray,
    # det_sem: np.ndarray,
    xlabel: str,
    max_r_k: int = 3,
) -> None:
    """
    Print a compact table of sweep results to stdout.
    """
    header = [xlabel]
    for k in range(max_r_k + 1):
        header.append(f"r{k}")
        header.append(f"r{k}_sem")

    fmt_header = " | ".join(f"{h:>14s}" for h in header)
    print("\n" + fmt_header)
    print("-" * len(fmt_header))

    for i, xv in enumerate(x):
        # row = [f"{xv:14.6g}", f"{det_mean[i]:14.3g}±{det_sem[i]:.2g}"]
        row = [f"{xv:14.6g}"]
        for k in range(max_r_k + 1):
            r = r_mean_list[i][k] if len(r_mean_list[i]) > k else np.nan
            s = r_sem_list[i][k] if len(r_sem_list[i]) > k else np.nan
            row.append(f"{r:14.3e}")
            row.append(f"{s:14.2e}")
        print(" | ".join(row))


def plot_results_overlay(
    x: np.ndarray,
    r_mean_list: Sequence[np.ndarray],
    r_sem_list: Sequence[np.ndarray],
    # det_mean: np.ndarray,
    # det_sem: np.ndarray,
    xlabel: str,
    out_r: Path,
    out_det: Optional[Path] = None,
    max_r_k: int = 3,
) -> None:
    """
    One plot with multiple overlaid r_k curves vs x.
    Optionally write a second plot for detections.
    """
    out_r.parent.mkdir(parents=True, exist_ok=True)

    # --- r_k overlay plot ---
    plt.figure(figsize=(7.5, 5.0))

    # cmap = plt.get_cmap("plasma")

    for k in range(max_r_k + 1):
        means = np.array(
            [r[k] if len(r) > k else np.nan for r in r_mean_list], dtype=float
        )
        sems = np.array(
            [s[k] if len(s) > k else np.nan for s in r_sem_list], dtype=float
        )

        # skip if completely missing
        if np.all(~np.isfinite(means)):
            continue

        # color = cmap(k / max_r_k)
        plt.errorbar(
            x,
            means,
            yerr=sems,
            fmt="o-",
            # color=color,
            capsize=4,
            capthick=1.5,
            markersize=6,
            label=rf"$r_{k}$",
        )

    plt.yscale("log")
    plt.ylabel(r"$r_k$ (log scale)")
    plt.ylim(1e-5, 1.1)  # adjust floor to your noise level

    # plt.ylabel(r"$r_k$")
    plt.xlabel(xlabel)
    plt.grid(True, alpha=0.3)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_r, dpi=300)
    plt.close()

    # --- detections plot (optional) ---
    # if out_det is not None:
    #     out_det.parent.mkdir(parents=True, exist_ok=True)
    #     plt.figure(figsize=(7.5, 5.0))
    #     plt.errorbar(
    #         x,
    #         det_mean,
    #         yerr=det_sem,
    #         fmt="s-",
    #         capsize=4,
    #         capthick=1.5,
    #         markersize=6,
    #     )
    #     plt.xlabel(xlabel)
    #     plt.ylabel("Detections per replicate")
    #     plt.grid(True, alpha=0.3)
    #     plt.tight_layout()
    #     plt.savefig(out_det, dpi=300)
    #     plt.close()


def main() -> None:
    OUTPUT_ROOT = Path("outputs")

    # PATTERN = "be_radius_*"
    # FIG_PATH = Path("figures/be_radius_plot.png")
    # X_LABEL = "be radius"

    PATTERN = "n2n_scale_*"
    FIG_PATH = Path("figures/n2n_scale_plot.png")
    X_LABEL = "n,2n scale factor"

    # PATTERN = "rate_*"
    # FIG_PATH = Path("figures/source_rate_plot.png")
    # X_LABEL = "source rate"

    # PATTERN = "he_density_*"
    # FIG_PATH = Path("figures/he_density_plot.png")
    # X_LABEL = "he density"

    N_PARTICLES = 1e6

    PREDELAY = 4e-6
    GATE = 85e-6
    DELAY = 1000e-6

    MAX_RK = 3
    SORT = True
    SUBPLOT_PATH = FIG_PATH.with_name(FIG_PATH.stem + "_subplots" + FIG_PATH.suffix)

    sr = SRParams(predelay=PREDELAY, gate=GATE, delay=DELAY)

    points = list_sweep_points(OUTPUT_ROOT, PATTERN)
    pairs = sorted(((point_x(p), p) for p in points), key=lambda t: t[0])

    xs_arr = np.array([x for x, _ in pairs], dtype=float)
    points = [p for _, p in pairs]

    r_means: List[np.ndarray] = []
    r_sems: List[np.ndarray] = []
    # det_means: List[float] = []
    # det_sems: List[float] = []

    # Compute per point
    for p in points:
        rep_dirs = sorted([p for p in p.glob("rep_*") if p.is_dir()])
        if not rep_dirs:
            raise FileNotFoundError(f"No replicate dirs found in {p}")

        h5 = rep_dirs[0] / "collision_track.h5"
        if not h5.exists():
            # skip missing replicates (e.g., failed runs)
            continue
        sr = SRParams(predelay=sr.predelay, gate=sr.gate, delay=sr.delay)
        r_full, r_mean, r_std, arr = analyze_with_bootstrap_parallel(
            str(h5),
            sr,
            n_bootstrap=500,
            segment_duration=1.0,
            seed=12346,
            n_workers=32,
        )

        r_means.append(r_mean)
        r_sems.append(r_std)

    # det_means_arr = np.asarray(det_means, dtype=float)
    # det_sems_arr = np.asarray(det_sems, dtype=float)

    if SORT:
        order = np.argsort(xs_arr)
        xs_arr = xs_arr[order]
        # det_means_arr = det_means_arr[order]
        # det_sems_arr = det_sems_arr[order]
        r_means = [r_means[i] for i in order]
        r_sems = [r_sems[i] for i in order]

    plot_results_overlay(
        x=xs_arr,
        r_mean_list=r_means,
        r_sem_list=r_sems,
        # det_mean=det_means_arr,
        # det_sem=det_sems_arr,
        xlabel=X_LABEL,
        out_r=FIG_PATH,
        max_r_k=MAX_RK,
        # out_det=(FIG_PATH.parent / (FIG_PATH.stem + "_detections" + FIG_PATH.suffix)),
    )
    print_sweep_table(
        x=xs_arr,
        r_mean_list=r_means,
        r_sem_list=r_sems,
        # det_mean=det_means_arr,
        # det_sem=det_sems_arr,
        xlabel=X_LABEL,
        max_r_k=MAX_RK,
    )
    # plot_sensitivity_results(
    #     x=xs_arr,
    #     r_mean_list=r_means,
    #     r_sem_list=r_sems,
    #     xlabel=X_LABEL,
    #     out=Path("figures/n2n_sensitivity.png"),
    #     which_r_k=1,
    #     N_PARTICLES=5e6,
    #     N_REPS=len(all_det),
    # )
    plot_results(
        x=xs_arr,
        r_mean_list=r_means,
        r_sem_list=r_sems,
        # det_mean=det_means_arr,
        # det_sem=det_sems_arr,
        xlabel=X_LABEL,
        out=SUBPLOT_PATH,
        max_r_k=MAX_RK,
        fit_sensitivity=True,
        N_PARTICLES=N_PARTICLES,
    )


if __name__ == "__main__":
    main()
