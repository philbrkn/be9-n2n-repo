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
from typing import List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import openmc

from bootstrap import analyze_with_bootstrap_parallel
from plot_sweep_bootstrap import get_sensitivity

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 9,  # better for multi-panel
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "lines.linewidth": 1.2,
        "axes.linewidth": 0.8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
        "xtick.minor.size": 2.0,
        "ytick.minor.size": 2.0,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "mathtext.fontset": "custom",
        "mathtext.rm": "serif",
        "mathtext.it": "serif:italic",
        "mathtext.bf": "serif:bold",
        "figure.dpi": 300,
    }
)


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
    data_color = "#2E5090"  # Deep blue
    mean_color = "#C1403D"  # Muted red

    means = [r[which_r_k] if len(r) > which_r_k else np.nan for r in r_mean_list]
    sems = [s[which_r_k] if len(s) > which_r_k else np.nan for s in r_sem_list]

    ax.errorbar(
        x,
        means,
        color=data_color,
        yerr=sems,
        fmt="o-",
        capsize=5,
        capthick=2,
        markersize=8,
    )
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
    # print(
    #     f"A shift register measurement of r[{which_r_k}] with {precision * 100:.2f}% precision"
    #     f" (achievable with {N_REPS} replicates of {N_PARTICLES} particles) "
    #     f"can constrain the Be-9 (n,2n) cross section to ±{constraint * 100:.4f}%."
    # )
    ax.plot(
        x,
        np.polyval(coeffs, x),
        "r--",
        color=mean_color,
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
    det_mean: np.ndarray,
    det_sem: np.ndarray,
    xlabel: str,
    out: Path,
    max_r_k: int = 3,
    fit_sensitivity: bool = False,
    N_PARTICLES: int = None,
) -> None:
    data_color = "#2E5090"

    # fig, axes = plt.subplots(1, 4, figsize=(8.27, 2.3), constrained_layout=True)
    fig, axes = plt.subplots(2, 2, figsize=(6, 4.5), constrained_layout=True)
    axes = axes.flatten()

    # --- r0, r1, r2 panels ---
    for idx in range(3):
        ax = axes[idx]
        means = [r[idx] if len(r) > idx else np.nan for r in r_mean_list]
        sems = [s[idx] if len(s) > idx else np.nan for s in r_sem_list]

        ax.errorbar(
            x,
            means,
            yerr=sems,
            fmt="o-",
            color=data_color,
            capsize=4,
            capthick=1,
            markersize=3,
        )
        ax.set_xlabel(xlabel)
        ax.set_ylabel(f"$r_{idx}$")
        ax.grid(True, alpha=0.2, linestyle="-", linewidth=0.5)
        ax.set_axisbelow(True)

        if "density" in xlabel.lower() or "rate" in xlabel.lower():
            ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))

        if fit_sensitivity:
            coeffs = np.polyfit(x, means, 1)
            slope = coeffs[0]
            i0 = int(np.argmin(np.abs(x - 1.0)))
            r_nominal = means[i0]
            sem_nominal = sems[i0]
            x0 = x[i0]
            S = slope * x0 / r_nominal
            print(f"Sensitivity coefficient S = {S:.3f}")
            print(
                f"Interpretation: 1% change in (n,2n) causes {S:.2f}% change in r[{idx}]"
            )
            precision = sem_nominal / r_nominal
            constraint = 1 / S * precision
            print(
                f"A shift register measurement of r[{idx}] with {precision * 100:.2f}% precision"
                f" (achievable with {1} replicates of {N_PARTICLES} particles) "
                f"can constrain the {xlabel} to ±{constraint * 100:.4f}%."
            )

    # --- detector count panel ---
    ax = axes[3]
    ax.errorbar(
        x,
        det_mean,
        yerr=det_sem,
        fmt="o-",
        color=data_color,
        capsize=4,
        capthick=1,
        markersize=3,
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Detections per replicate")
    ax.grid(True, alpha=0.2, linestyle="-", linewidth=0.5)
    ax.set_axisbelow(True)

    if "density" in xlabel.lower() or "rate" in xlabel.lower():
        ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))

    # subplot labels
    for ax, label in zip(axes, ["(a)", "(b)", "(c)", "(d)"]):
        ax.text(
            0.5, -0.23, label, transform=ax.transAxes, ha="center", va="top", fontsize=9
        )

    fig.savefig(out, dpi=300)
    plt.close(fig)


def plot_detection_counts(x, det_mean, det_sem, x_label, out, fit_sensitivity=True):
    fig, ax = plt.subplots(1, 1, figsize=(3.5, 3.5), constrained_layout=True)
    ax.errorbar(
        x, det_mean, yerr=det_sem, fmt="s-", capsize=5, capthick=2, markersize=8
    )
    ax.set_xlabel(x_label)
    ax.set_ylabel("Detections per replicate")
    ax.grid(True, alpha=0.3)
    fig.savefig(out, dpi=300)
    plt.close(fig)
    if fit_sensitivity:
        # Fit a line to get sensitivity
        coeffs = np.polyfit(x, det_mean, 1)
        slope = coeffs[0]
        # intercept = coeffs[1]

        # Sensitivity: (dr/r) / (dXS/XS) at nominal
        # choose the middle one:
        i0 = int(np.argmin(np.abs(x - 1.0)))
        r_nominal = det_mean[i0]
        sem_nominal = det_sem[i0]
        x0 = x[i0]
        # S = slope * x0 / r_nominal  # dr/dXS normalized
        S = slope / r_nominal  # dr/dXS normalized

        print(f"Sensitivity coefficient S = {S:.3f}")
        print(f"Interpretation: 1% change in (n,2n) causes {S:.2f}% change in det mean")
        # invert for constraint:
        # if we measure r[1] with precision delr[1] we constrain XS(n,2n) to:
        # deltaXS / XS = (1/S) * (deltar[1]/r[1])
        precision = sem_nominal / r_nominal
        constraint = 1 / S * precision
        print(
            f"A shift register measurement of detection mean with {precision * 100:.2f}% precision"
            f" (achievable with {1} replicates of x particles) "
            f"can constrain the {x_label} to ±{constraint * 100:.4f}%."
        )


def compute_f_high_eff_and_x(
    points: List[Path],
    xs_arr: np.ndarray,
    E_CUT: float = 1.0e6,
    nominal_scale: float = 1.0,
) -> Tuple[np.ndarray, float]:
    f_high_eff_per_point = []

    for p in points:
        # find perturbed h5 for this sweep point
        xs_dir = p / "xs"
        h5_files = list(xs_dir.glob("Be9_scaled_*.h5"))
        if not h5_files:
            raise FileNotFoundError(f"No scaled Be9 h5 found in {xs_dir}")
        be9_path = h5_files[0]

        # compute f_high at each incident energy from the perturbed file
        be9 = openmc.data.IncidentNeutron.from_hdf5(str(be9_path))
        file6 = be9.reactions[16].products[0].distribution[0]
        incident_energies = np.array(file6.energy)

        f_high_per_energy = []
        for k in range(len(incident_energies)):
            tab = file6.energy_out[k]
            dE = np.diff(tab.x)
            Emid = 0.5 * (tab.x[:-1] + tab.x[1:])
            mass_bins = tab.p[:-1] * dE
            f_high_per_energy.append(mass_bins[Emid >= E_CUT].sum())
        f_high_per_energy = np.array(f_high_per_energy)

        # get reaction rate spectrum from statepoint
        rep_dirs = sorted([d for d in p.glob("rep_*") if d.is_dir()])
        sp = openmc.StatePoint(rep_dirs[0] / "statepoint.1.h5")
        tally = sp.get_tally(name="n2n_spectrum")
        energy_filter = tally.find_filter(openmc.EnergyFilter)
        energy_bins = energy_filter.bins
        energy_mids = 0.5 * (energy_bins[:, 0] + energy_bins[:, 1])
        rxn_rates = tally.get_values(scores=["(n,2n)"]).flatten()

        # interpolate f_high onto tally energy midpoints and weight
        f_high_interp = np.interp(energy_mids, incident_energies, f_high_per_energy)
        weights = rxn_rates / rxn_rates.sum()
        f_high_eff_per_point.append(np.dot(weights, f_high_interp))

    f_high_eff_arr = np.array(f_high_eff_per_point)
    nominal_idx = np.argmin(np.abs(xs_arr - nominal_scale))
    f_high_eff_nominal = f_high_eff_arr[nominal_idx]
    x_ddx = f_high_eff_arr / f_high_eff_nominal

    return x_ddx, f_high_eff_nominal


def plot_ddx_perturbation_figure(
    points: List[Path],
    xs_arr: np.ndarray,
    E_CUT: float = 1.8e6,
    nominal_scale: float = 0.0,
    out: Path = Path("figures/ddx_perturbation.png"),
):
    nominal_idx = np.argmin(np.abs(xs_arr - nominal_scale))
    nominal_point = points[nominal_idx]
    largest_idx = np.argmax(np.abs(xs_arr - nominal_scale))
    largest_point = points[largest_idx]

    # --- load all be9 files ---
    def load_file6(p, nominal_be9_path):
        xs_dir = p / "xs"
        h5_files = list(xs_dir.glob("Be9_scaled_*.h5")) if xs_dir.exists() else []
        path = str(h5_files[0]) if h5_files else nominal_be9_path
        be9 = openmc.data.IncidentNeutron.from_hdf5(path)
        return be9.reactions[16].products[0].distribution[0]

    BE9_PATH = "/home/philip/Documents/endf-b8.0-hdf5/endfb-viii.0-hdf5/neutron/Be9.h5"
    nominal_be9_path = str(
        next((nominal_point / "xs").glob("Be9_scaled_*.h5"), Path(BE9_PATH))
    )
    file6_nom = load_file6(nominal_point, nominal_be9_path)
    inc_energies = np.array(file6_nom.energy)
    e_idx = np.argmin(np.abs(inc_energies - 14.1e6))

    # --- reaction rate spectrum from nominal statepoint ---
    # rep_dirs = sorted([d for d in nominal_point.glob("rep_*") if d.is_dir()])
    # sp = openmc.StatePoint(rep_dirs[0] / "statepoint.1.h5")
    # tally = sp.get_tally(name="n2n_spectrum")
    # energy_filter = tally.find_filter(openmc.EnergyFilter)
    # energy_bins = energy_filter.bins
    # energy_mids = 0.5 * (energy_bins[:, 0] + energy_bins[:, 1])
    # rxn_rates = tally.get_values(scores=["(n,2n)"]).flatten()
    # --- f_high_eff at nominal for annotation ---
    # f_high_per_energy = []
    # for k in range(len(inc_energies)):
    #     tab = file6_nom.energy_out[k]
    #     dE = np.diff(tab.x)
    #     Emid = 0.5 * (tab.x[:-1] + tab.x[1:])
    #     mass_bins = tab.p[:-1] * dE
    #     f_high_per_energy.append(mass_bins[Emid >= E_CUT].sum())
    # f_high_interp = np.interp(energy_mids, inc_energies, f_high_per_energy)
    # weights = rxn_rates / rxn_rates.sum()
    # f_high_eff = np.dot(weights, f_high_interp)

    data_color = "#2E5090"  # Deep blue
    mean_color = "#C1403D"  # Muted red
    fig, axes = plt.subplots(1, 2, figsize=(8.27, 3.0), constrained_layout=True)

    # --- panel a: family ---
    ax = axes[0]
    non_nominal = [i for i in range(len(points)) if i != nominal_idx]
    colors = plt.cm.RdBu(np.linspace(0.1, 0.9, len(non_nominal)))
    color_iter = iter(colors)
    for i, (p, label) in enumerate(zip(points, xs_arr)):
        file6 = load_file6(p, nominal_be9_path)
        tab = file6.energy_out[e_idx]
        if i == nominal_idx:
            ax.semilogy(
                tab.x / 1e6,
                tab.p * 1e6,
                color="black",
                lw=2,
                label=f"{label:+.2f} (nominal)",
                zorder=5,
            )
        else:
            ax.semilogy(
                tab.x / 1e6,
                tab.p * 1e6,
                color=next(color_iter),
                lw=1,
                ls="--",
                label=f"{label:+.2f}",
            )
    ax.axvline(
        E_CUT / 1e6, color="k", ls=":", lw=1, label=f"$E_{{cut}}$={E_CUT / 1e6:.1f} MeV"
    )
    ax.set_xlabel("E' (MeV)")
    ax.set_ylabel("p(E'|E) (per MeV)")
    ax.set_title(f"$E_i$ = {inc_energies[e_idx] / 1e6:.1f} MeV")
    ax.legend(fontsize=7, title="δ")
    ax.grid(True, alpha=0.3)
    # ax = axes[0]
    # for p, label in zip(points, xs_arr):
    #     file6 = load_file6(p, nominal_be9_path)
    #     tab = file6.energy_out[e_idx]
    #     ax.semilogy(tab.x / 1e6, tab.p * 1e6, label=f"{label:+.2f}")
    # ax.axvline(E_CUT / 1e6, color="k", ls=":", lw=1)
    # ax.set_xlabel("E' (MeV)")
    # ax.set_ylabel("p(E'|E) (per MeV)")
    # ax.set_title(f"$E_i$ = {inc_energies[e_idx] / 1e6:.1f} MeV")
    # ax.legend(fontsize=7, title="δ")
    # ax.grid(True, alpha=0.3)

    # --- panel b: delta p largest perturbation ---
    ax = axes[1]
    tab0 = file6_nom.energy_out[e_idx]
    E0, p0 = tab0.x.copy(), tab0.p.copy()
    file6_pert = load_file6(largest_point, nominal_be9_path)
    p1 = file6_pert.energy_out[e_idx].p.copy()
    ax.plot(E0 / 1e6, (p1 - p0) * 1e6, color="k")
    ax.axhline(0, color="k", lw=0.5)
    ax.axvline(
        E_CUT / 1e6,
        color=mean_color,
        ls=":",
        lw=1,
        label=f"$E_{{cut}}$ = {E_CUT / 1e6:.1f} MeV",
    )
    ax.set_xlabel("E' (MeV)")
    ax.set_ylabel("Δp(E'|E) (per MeV)")
    ax.set_title(f"δ = {xs_arr[largest_idx]:+.2f}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # # --- panel c: reaction rate spectrum ---
    # ax = axes[2]
    # rates_norm = rxn_rates / rxn_rates.sum()
    # ax.plot(energy_mids / 1e6, weights, color="k", label="reaction rate weights")
    # ax.plot(energy_mids / 1e6, f_high_interp, color="b", label="$f_{high}(E_i)$")
    # ax.fill_between(
    #     energy_mids / 1e6,
    #     weights * f_high_interp,
    #     alpha=0.3,
    #     label=f"integrand, $f_{{high,eff}}$ = {f_high_eff:.3f}",
    # )
    # ax.axvline(E_CUT / 1e6, color="r", ls=":", lw=1)
    # ax.set_xlabel("Incident energy (MeV)")
    # ax.set_ylabel("")
    # ax.legend(fontsize=8)
    # ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out, dpi=300)
    plt.close(fig)


def main() -> None:
    OUTPUT_ROOT = Path("outputs")
    # OUTPUT_ROOT = Path("outputs/be_rad_ddx_sweep_11.0/")
    # OUTPUT_ROOT = Path("outputs/10e6_particles/")

    FIG_DIR = Path("figures")
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    PATTERN = "ddx_scale_*"
    FIG_PATH = Path("figures/ddx_scale_plot.png")
    X_LABEL = "ddx scale factor"

    # --- cache settings (minimal) ---
    CACHE_PATH = Path("figures") / f"cache_{PATTERN.rstrip('*')}.npz"
    USE_CACHE = False

    # PARAMETERS
    N_PARTICLES = 1e8
    PREDELAY = 4e-6
    GATE = 28e-6
    DELAY = 1000e-6
    MAX_RK = 5
    SORT = True
    SUBPLOT_PATH = FIG_PATH.with_name(FIG_PATH.stem + "_subplots" + FIG_PATH.suffix)

    sr = SRParams(predelay=PREDELAY, gate=GATE, delay=DELAY)
    points = list_sweep_points(OUTPUT_ROOT, PATTERN)
    pairs = sorted(((point_x(p), p) for p in points), key=lambda t: t[0])
    points = [p for _, p in pairs]

    if USE_CACHE and CACHE_PATH.exists():
        d = np.load(CACHE_PATH, allow_pickle=True)
        xs_arr = d["xs_arr"]
        det_means_arr = d["det_means_arr"]
        det_sems_arr = d["det_sems_arr"]
        r_means = list(d["r_means"])
        r_sems = list(d["r_sems"])
        # rxn_rates_all = d["rxn_rates_all"]
        MAX_RK = int(d["MAX_RK"])
        SORT = bool(d["SORT"])
        X_LABEL = str(d["X_LABEL"])
        N_PARTICLES = float(d["N_PARTICLES"])
        print(f"Loaded cache: {CACHE_PATH}")
    else:
        xs_arr = np.array([x for x, _ in pairs], dtype=float)

        r_means: List[np.ndarray] = []
        r_sems: List[np.ndarray] = []
        det_means: List[float] = []
        det_sems: List[float] = []
        rxn_rates_all = []

        for p in points:
            rep_dirs = sorted([p for p in p.glob("rep_*") if p.is_dir()])
            if not rep_dirs:
                raise FileNotFoundError(f"No replicate dirs found in {p}")

            h5 = rep_dirs[0] / "collision_track.h5"
            if not h5.exists():
                continue

            sr = SRParams(predelay=sr.predelay, gate=sr.gate, delay=sr.delay)
            r_full, r_mean, r_std, arr, det_mean, det_sem, dets = (
                analyze_with_bootstrap_parallel(
                    str(h5),
                    sr,
                    n_bootstrap=200,
                    segment_duration=1.0,
                    seed=12346,
                    n_workers=32,
                )
            )

            r_means.append(r_mean)
            r_sems.append(r_std)
            det_means.append(det_mean)
            det_sems.append(det_sem)
            print(
                f"[{p.name}] reps={len(dets)} detections={det_mean:.1f}±{det_sem:.1f}"
            )

        det_means_arr = np.asarray(det_means, dtype=float)
        det_sems_arr = np.asarray(det_sems, dtype=float)

        if SORT:
            order = np.argsort(xs_arr)
            xs_arr = xs_arr[order]
            det_means_arr = det_means_arr[order]
            det_sems_arr = det_sems_arr[order]
            r_means = [r_means[i] for i in order]
            r_sems = [r_sems[i] for i in order]

        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            CACHE_PATH,
            xs_arr=xs_arr,
            det_means_arr=det_means_arr,
            det_sems_arr=det_sems_arr,
            r_means=np.array(r_means, dtype=object),
            r_sems=np.array(r_sems, dtype=object),
            rxn_rates_all=np.array(rxn_rates_all, dtype=object),
            MAX_RK=MAX_RK,
            SORT=SORT,
            X_LABEL=X_LABEL,
            N_PARTICLES=N_PARTICLES,
        )
        print(f"Saved cache: {CACHE_PATH}")

    x_ddx, f_high_eff_nominal = compute_f_high_eff_and_x(
        points,
        xs_arr,
        E_CUT=1.8e6,
        nominal_scale=0.0,
    )
    print(x_ddx, f_high_eff_nominal)

    # plot_ddx_perturbation_figure(points, xs_arr, E_CUT=1.8e6, nominal_scale=0.0)

    get_sensitivity(
        x=x_ddx,
        r_mean_list=r_means,
        r_sem_list=r_sems,
        det_mean=det_means_arr,
        det_sem=det_sems_arr,
        max_r_k=MAX_RK,
    )
    # plot_results(
    #     x=x_ddx,
    #     r_mean_list=r_means,
    #     r_sem_list=r_sems,
    #     det_mean=det_means_arr,
    #     det_sem=det_sems_arr,
    #     xlabel=X_LABEL,
    #     out=SUBPLOT_PATH,
    #     max_r_k=MAX_RK,
    #     fit_sensitivity=True,
    #     N_PARTICLES=N_PARTICLES,
    # )
    # det_path = f"figures/{X_LABEL}_detection_counts.png"
    # plot_detection_counts(x_ddx, det_means_arr, det_sems_arr, X_LABEL, out=det_path)


if __name__ == "__main__":
    import multiprocessing as mp

    mp.set_start_method("fork", force=True)
    main()
