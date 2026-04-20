from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import openmc
from scipy.optimize import curve_fit

from run_replicates import ReplicateConfig
from utils.build_complex_input import create_geometry


def run_dieaway_simulation(input_dir, output_dir):
    """Runs a pulsed-source simulation to measure die-away."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = ReplicateConfig(
        n_replicates=1,
        particles_per_rep=1_000_000,
        base_seed=12345,
        gate=28e-6,
        predelay=4e-6,
        delay=1000e-6,
        rate=3e4,
    )

    # 1. Create Geometry
    geo = create_geometry(output_dir, cfg)

    # 2. Create Settings for a PULSE (t=0)
    settings = openmc.Settings()
    settings.particles = cfg.particles_per_rep
    settings.batches = 1
    settings.run_mode = "fixed source"
    settings.output = {"path": str(output_dir), "tallies": False}

    # Force specific cell IDs for recording tracks (from your geo function)
    settings.collision_track = {
        "cell_ids": geo["he3_cell_ids"],
        "max_collisions": int(cfg.particles_per_rep),
    }

    # === THE KEY CHANGE: Pulsed Source at t=0 ===
    settings.source = openmc.IndependentSource(
        space=openmc.stats.Point((0, 0, 0)),
        energy=openmc.stats.Normal(mean_value=14.1e6, std_dev=5.0e4),
        angle=openmc.stats.Isotropic(),
        time=openmc.stats.Uniform(0.0, 1e-6),  # All neutrons start at t=0
    )
    settings.export_to_xml(path=str(output_dir / "settings.xml"))

    # Copy materials/geometry to run dir
    # for f in ["materials.xml", "geometry.xml"]:
    #     (input_dir / f).rename(output_dir / f)  # or copy

    # 3. Run OpenMC
    openmc.run(cwd=str(output_dir), path_input=str(output_dir))

    return output_dir


def analyze_dieaway(run_dir):
    """Fits the exponential decay to find Tau."""
    run_dir = Path(run_dir)

    # Load data
    track_file = run_dir / "collision_track.h5"
    tracks = openmc.read_collision_track_hdf5(str(track_file))

    # Safest bet for He3 detectors: Take all events in the track file
    # because we only asked for tracks in He3 cells, and He3 x-section is dominated by absorption.
    times = tracks["time"]

    # Histogram the data
    # Use log-spaced bins or fine linear bins
    hist, bin_edges = np.histogram(times, bins=200, range=(1e-6, 1000e-6))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Select the "tail" region to fit (ignore the messy buildup at the start)
    # Visual inspection is best, but usually 50us to 500us is the sweet spot.
    fit_mask = (bin_centers > 20e-6) & (bin_centers < 500e-6) & (hist > 0)

    x_data = bin_centers[fit_mask]
    y_data = hist[fit_mask]

    # Fit function: A * exp(-t / tau)
    # Linear fit on log data is more robust: ln(y) = ln(A) - t/tau
    def log_line(t, a, tau):
        return a - (t / tau)

    popt, _ = curve_fit(
        log_line, x_data, np.log(y_data), p0=[np.log(max(y_data)), 100e-6]
    )

    tau = popt[1]

    print(f"Calculated Die-Away Time (Tau): {tau * 1e6:.2f} microseconds")

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
    # Plot to verify
    plt.semilogy(bin_centers * 1e6, hist, label="Simulated Data", color=data_color)
    plt.semilogy(
        x_data * 1e6,
        np.exp(log_line(x_data, *popt)),
        label=f"Fit (Tau={tau * 1e6:.1f} us)",
        alpha=0.8,
        color=mean_color,
        linestyle="--",
    )
    plt.xlabel("Time (us)")
    plt.ylabel("Counts")

    plt.legend(
        frameon=True, fancybox=False, edgecolor="gray", framealpha=0.95, loc="best"
    )
    plt.grid(True, alpha=0.2, linestyle="-", linewidth=0.5)
    # plt.set_axisbelow(True)
    plt.savefig("dieaway.png", dpi=300)

    return tau


base_dir = Path(__file__).parent.parent.resolve()
input_dir = base_dir / "inputs"
output_dir = base_dir / "outputs"
run_dieaway_simulation(input_dir, output_dir)
analyze_dieaway(output_dir)
