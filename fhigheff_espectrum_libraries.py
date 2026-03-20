#!/usr/bin/env python3
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import openmc

from bootstrap import analyze_with_bootstrap_parallel
from plot_sweep_bootstrap import get_sensitivity


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


def compute_f_high_for_library(
    h5_path: Path, E_CUT: float = 1.8e6
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (incident_energies, f_high_per_energy) for one library."""
    be9 = openmc.data.IncidentNeutron.from_hdf5(str(h5_path))
    file6 = be9.reactions[16].products[0].distribution[0]
    incident_energies = np.array(file6.energy)
    f_high = []
    for k in range(len(incident_energies)):
        tab = file6.energy_out[k]
        dE = np.diff(tab.x)
        Emid = 0.5 * (tab.x[:-1] + tab.x[1:])
        mass_bins = tab.p[:-1] * dE
        f_high.append(mass_bins[Emid >= E_CUT].sum())
    return incident_energies, np.array(f_high)


def plot_spectrum_nominal(
    statepoint_path: Path,
    be9_h5_dir_endf: Path,
    tendl_dir: Path,
    endf7_dir: Path,
    xs_arr: np.ndarray,
    E_CUT: float = 1.8e6,
    nominal_scale: float = 1.0,
) -> Tuple[np.ndarray, float]:
    # wy is this blue
    libraries = {
        "ENDF/B-VII.1": tendl_dir,
        "TENDL-2025": endf7_dir,
        "ENDF/B-VIII.0": be9_h5_dir_endf,
        # "JEFF-3.3": jeff_h5_path,  # if you parse the channels
    }

    # reaction rate spectrum from nominal statepoint (same for all)
    sp = openmc.StatePoint(statepoint_path)
    tally = sp.get_tally(name="n2n_spectrum")
    energy_filter = tally.find_filter(openmc.EnergyFilter)
    energy_bins = energy_filter.bins
    energy_mids = 0.5 * (energy_bins[:, 0] + energy_bins[:, 1])
    rxn_rates = tally.get_values(scores=["(n,2n)"]).flatten()
    weights = rxn_rates / rxn_rates.sum()

    for name, h5_path in libraries.items():
        inc_e, f_high = compute_f_high_for_library(h5_path)
        f_high_interp = np.interp(energy_mids, inc_e, f_high)
        f_high_eff = np.dot(weights, f_high_interp)
        print(f"{name}: f_high_eff = {f_high_eff:.4f}")
    # # PLOT
    # fig, ax = plt.subplots(figsize=(5, 3.5), constrained_layout=True)
    # plt.plot(energy_mids, rxn_rates, color="k", label="Energy spectrum")
    # ax.axvline(
    #     f_high_eff * energy_mids[-1] / f_high_per_energy[-1],
    #     color="k",
    #     ls=":",
    #     lw=1,
    #     label=f"$E_{{cut}}$ = {E_CUT / 1e6:.1f} MeV",
    # )
    # ax.grid(True, alpha=0.3)
    # ax.set_xlabel("E' (MeV)")
    # ax.set_ylabel("Reaction rate")
    # plt.savefig("figures/energy_spectrum.png")


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
    print(nominal_idx, nominal_point, largest_idx, largest_point)
    smallest_idx = 4
    smallest_point = points[smallest_idx]

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

    # --- panel b: delta p largest perturbation ---
    ax = axes[1]
    tab0 = file6_nom.energy_out[e_idx]
    E0, p0 = tab0.x.copy(), tab0.p.copy()
    file6_pert = load_file6(largest_point, nominal_be9_path)
    p1 = file6_pert.energy_out[e_idx].p.copy()
    ax.plot(E0 / 1e6, (p1 - p0) * 1e6, color=colors[largest_idx], label="δ = -0.20")
    ax.axhline(0, color="k", lw=0.5)
    ax.axvline(
        E_CUT / 1e6,
        color="k",
        ls=":",
        lw=1,
        label=f"$E_{{cut}}$ = {E_CUT / 1e6:.1f} MeV",
    )
    # smalelst:
    file6_pert = load_file6(smallest_point, nominal_be9_path)
    p1 = file6_pert.energy_out[e_idx].p.copy()
    ax.plot(E0 / 1e6, (p1 - p0) * 1e6, color=colors[-1], label="δ = +0.20")

    ax.set_xlabel("E' (MeV)")
    ax.set_ylabel("Δp(E'|E) (per MeV)")
    # ax.set_title(f"δ = {xs_arr[largest_idx]:+.2f}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

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
    USE_CACHE = True

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

    BE9_PATH = "/home/philip/Documents/endf-b8.0-hdf5/endfb-viii.0-hdf5/neutron/Be9.h5"
    TENDL_PATH = "TENDL-2019_Be9.h5"
    ENDF7_PATH = "ENDFB-7.1-NNDC_Be9.h5"
    statepoint_path = Path("outputs/ddx_scale_0.00/rep_0000/statepoint.1.h5")
    plot_spectrum_nominal(statepoint_path, BE9_PATH, TENDL_PATH, ENDF7_PATH, xs_arr)
    plot_ddx_perturbation_figure(points, xs_arr, E_CUT=1.8e6, nominal_scale=0.0)

    get_sensitivity(
        x=x_ddx,
        r_mean_list=r_means,
        r_sem_list=r_sems,
        det_mean=det_means_arr,
        det_sem=det_sems_arr,
        max_r_k=MAX_RK,
    )


if __name__ == "__main__":
    import multiprocessing as mp

    mp.set_start_method("fork", force=True)
    main()
