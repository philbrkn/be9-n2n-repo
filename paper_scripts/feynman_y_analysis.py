import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import openmc
from matplotlib.ticker import MaxNLocator
from scipy.optimize import curve_fit

from utils.analyze_sr import (
    SRParams,
    sr_histograms_twoptr,
)


def plot_inter_arrival_times(detection_times, plot_dir):
    # compute difference between each detection time:
    dt = np.diff(detection_times)
    t = detection_times[:-1]

    plt.figure(figsize=(12, 4))
    plt.scatter(t, dt, s=0.1, alpha=0.3)
    plt.yscale("log")
    plt.xlim(10, 15)
    plt.ylim(1e-6, 1e-3)
    plt.xlabel("Time (s)")
    plt.ylabel("Time until next count (s)")
    plt.savefig(plot_dir / "inter_arrrival_times.png", dpi=300)


def plot_frequency_of_counts_histogram(rplusa_dist, max_counts=20):
    b_k = rplusa_dist / rplusa_dist.sum()

    # plt.figure(figsize=(10, 5))
    # plt.bar(range(len(b_k[:40])), b_k[:40])
    # plt.xlabel("Counts (k) in gate width")
    # plt.ylabel("Frequency of counts (b_k)")
    # plt.savefig(plot_dir / "frequency_of_counts.png")
    k = np.arange(len(rplusa_dist))
    total = rplusa_dist.sum()
    mean = (k * rplusa_dist).sum() / total
    variance = ((k - mean) ** 2 * rplusa_dist).sum() / total

    y = b_k[:max_counts]
    x = np.arange(len(y))
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        np.repeat(x, 2)[1:],  # horizontal connections
        np.repeat(y, 2)[:-1],
        color="black",
        linewidth=1.5,
    )
    ax.vlines(x=0, ymin=0, ymax=y[0], color="black", linewidth=1.5)
    # ---- mean line ----
    ax.axvline(mean, color="black", linestyle="--", linewidth=1.2)

    # ---- annotation ----
    ax.text(
        0.98,
        0.95,
        rf"$\mu = {mean:.2f}$" "\n" rf"$\sigma^2 = {variance:.2f}$",
        transform=ax.transAxes,
        ha="right",
        va="top",
    )

    # This guarantees integer ticks only while letting matplotlib choose spacing.
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    ax.set_xlabel("Counts (k) in gate width")
    ax.set_ylabel("Frequency of counts (b_k)")

    plt.savefig(plot_dir / "frequency_of_counts.png", dpi=300)


def feynmann_y_analysis(
    detection_times,
    gate_widths,
    predelay=4e-6,
    delay=1000e-6,
    plot_dir=None,
):
    y_arr = []
    for g in gate_widths:
        rplusa_dist, a_dist = sr_histograms_twoptr(
            detection_times, predelay, g, delay, cap=64
        )

        k = np.arange(len(rplusa_dist))
        total = rplusa_dist.sum()
        mean = (k * rplusa_dist).sum() / total
        variance = ((k - mean) ** 2 * rplusa_dist).sum() / total
        # Feynman-Y = variance/mean - 1 (excess variance)
        Y = variance / mean - 1 if mean > 0 else 0
        y_arr.append(Y)

    # plot
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
    plt.figure(figsize=(8, 5))
    plt.semilogx(
        gate_widths,
        y_arr,
        "o-",
        color=data_color,
    )
    plt.xlabel("Gate width (μs)")
    plt.ylabel("Variance/mean -1")
    plt.grid(True, alpha=0.3)
    plt.axhline(0, color=mean_color, alpha=0.5, linestyle="--", label="Poisson limit")

    plt.savefig(plot_dir / "feyman-y.png", dpi=300)

    return y_arr


def feynman_model(g, Y_inf, tau):
    x = g / tau
    return Y_inf * (1 - (1 - np.exp(-x)) / x)


if __name__ == "__main__":
    base_dir = Path(__file__).parent.resolve()
    input_dir = base_dir / "inputs"
    output_root = base_dir / "outputs"
    plot_dir = base_dir / "figures" / "feynmann-y"
    plot_dir.mkdir(parents=True, exist_ok=True)

    GATE = 128e-6
    PREDELAY = 4e-6
    DELAY = 1000e-6

    # col_track_file = output_root / "rep_0000" / "collision_track.h5"
    col_track_file = output_root / "standard_rep_0000_1e8p" / "collision_track.h5"

    sr = SRParams(predelay=PREDELAY, gate=GATE, delay=DELAY)

    start_time = time.perf_counter()

    # Load detection times
    ct = openmc.read_collision_track_hdf5(str(col_track_file))

    absorption_events = ct[ct["event_mt"] == 101]
    detection_times = np.sort(absorption_events["time"])

    # this is a bit of a useless plot with so much data:
    plot_inter_arrival_times(detection_times, plot_dir)

    gate_widths = [2e-6 * (2**n) for n in range(12)]  # 2, 4, 8, ... 1024 μs
    Y_arr = feynmann_y_analysis(
        detection_times,
        gate_widths,
        predelay=sr.predelay,
        delay=sr.delay,
        plot_dir=plot_dir,
    )

    popt, pcov = curve_fit(feynman_model, gate_widths, Y_arr, p0=[0.35, 70e-6])
    Y_inf_fit, tau_fit = popt
    perr = np.sqrt(np.diag(pcov))

    print(f"Y_∞ = {Y_inf_fit:.3f} ± {perr[0]:.3f}")
    print(f"τ = {tau_fit * 1e6:.1f} ± {perr[1] * 1e6:.1f} μs")

    # plot histograms:
    # rplusa_dist, a_dist = sr_histograms_twoptr(
    #     detection_times, sr.predelay, sr.gate, sr.delay, cap=64
    # )
    # plot_frequency_of_counts_histogram(rplusa_dist)

    print(f"Analysis took {time.perf_counter() - start_time:.2f}s")
