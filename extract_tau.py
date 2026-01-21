#!/usr/bin/env python3
"""
Proper tau extraction from detector response tally
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import openmc
from scipy.optimize import curve_fit


def fit_single_exponential(t, y, yerr, t_fit_start=10e-6, t_fit_end=200e-6):
    mask = (t >= t_fit_start) & (t <= t_fit_end)
    t_fit = t[mask]
    y_fit = y[mask]
    sigma = yerr[mask].copy()

    good = sigma > 0
    sigma[~good] = np.min(sigma[good]) if np.any(good) else 1.0

    def model(t, A, tau, B):
        return A * np.exp(-t / tau) + B

    B_guess = np.mean(y_fit[-10:])
    A_guess = np.max(y_fit) - B_guess

    signal = np.maximum(y_fit - B_guess, 1e-30)
    slope = np.polyfit(t_fit, np.log(signal), 1)[0]
    tau_guess = -1.0 / slope if slope < 0 else 50e-6

    p0 = [A_guess, tau_guess, B_guess]

    popt, pcov = curve_fit(
        model,
        t_fit,
        y_fit,
        p0=p0,
        sigma=sigma,
        absolute_sigma=True,
        bounds=([0, 1e-6, 0], [np.inf, 500e-6, np.inf]),
    )
    perr = np.sqrt(np.diag(pcov))

    residuals = y_fit - model(t_fit, *popt)
    chi2 = np.sum((residuals / sigma) ** 2)
    dof = len(t_fit) - len(popt)
    chi2_reduced = chi2 / dof if dof > 0 else np.nan

    A, tau, B = popt
    A_err, tau_err, B_err = perr

    return {
        "tau": tau,
        "tau_err": tau_err,
        "A": A,
        "A_err": A_err,
        "B": B,
        "B_err": B_err,
        "chi2_reduced": chi2_reduced,
        "fit_range": (t_fit_start, t_fit_end),
        "model": model,
        "popt": popt,
        "pcov": pcov,
    }


def fit_double_exponential(t, y, yerr, t_fit_start=10e-6, t_fit_end=300e-6):
    mask = (t >= t_fit_start) & (t <= t_fit_end)
    t_fit = t[mask]
    y_fit = y[mask]
    sigma = yerr[mask].copy()

    good = sigma > 0
    sigma[~good] = np.min(sigma[good]) if np.any(good) else 1.0

    def model(t, A1, tau1, A2, tau2, B):
        return A1 * np.exp(-t / tau1) + A2 * np.exp(-t / tau2) + B

    B_guess = np.mean(y_fit[-10:])

    # crude initial guesses using y_fit (not counts)
    A1_guess = np.max(y_fit) - B_guess
    A2_guess = A1_guess / 5.0

    # simple tau guesses
    tau1_guess, tau2_guess = 20e-6, 100e-6
    p0 = [A1_guess, tau1_guess, A2_guess, tau2_guess, B_guess]

    popt, pcov = curve_fit(
        model,
        t_fit,
        y_fit,
        p0=p0,
        sigma=sigma,
        absolute_sigma=True,
        bounds=([0, 1e-6, 0, 1e-6, 0], [np.inf, 200e-6, np.inf, 500e-6, np.inf]),
        maxfev=20000,
    )
    perr = np.sqrt(np.diag(pcov))

    # sort so tau1 < tau2, and reorder popt/perr accordingly
    A1, tau1, A2, tau2, B = popt
    if tau1 > tau2:
        # swap (A1,tau1) with (A2,tau2)
        popt = np.array([A2, tau2, A1, tau1, B], dtype=float)

        # reorder covariance accordingly: indices [0,1,2,3,4] -> [2,3,0,1,4]
        idx = [2, 3, 0, 1, 4]
        pcov = pcov[np.ix_(idx, idx)]
        perr = np.sqrt(np.diag(pcov))

    residuals = y_fit - model(t_fit, *popt)
    chi2 = np.sum((residuals / sigma) ** 2)
    dof = len(t_fit) - len(popt)
    chi2_reduced = chi2 / dof if dof > 0 else np.nan

    A1, tau1, A2, tau2, B = popt
    return {
        "tau1": tau1,
        "tau1_err": perr[1],
        "tau2": tau2,
        "tau2_err": perr[3],
        "A1": A1,
        "A2": A2,
        "B": B,
        "B_err": perr[4],
        "chi2_reduced": chi2_reduced,
        "fit_range": (t_fit_start, t_fit_end),
        "model": model,
        "popt": popt,
        "pcov": pcov,
    }


def plot_fits(t, counts, single_fit, double_fit, filename="tau_extraction.png"):
    """Create comprehensive diagnostic plots"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    t_us = t * 1e6  # Convert to microseconds

    # 1. Linear scale with both fits
    ax = axes[0, 0]
    ax.plot(t_us, counts, "k.", markersize=2, alpha=0.5, label="Data")

    if single_fit:
        t_plot = np.linspace(t[0], t[-1], 500)
        ax.plot(
            t_plot * 1e6,
            single_fit["model"](t_plot, *single_fit["popt"]),
            "r-",
            linewidth=2,
            label=f"Single: τ={single_fit['tau'] * 1e6:.1f}±{single_fit['tau_err'] * 1e6:.1f} μs",
        )

    if double_fit:
        t_plot = np.linspace(t[0], t[-1], 500)
        ax.plot(
            t_plot * 1e6,
            double_fit["model"](t_plot, *double_fit["popt"]),
            "b-",
            linewidth=2,
            label=f"Double: τ1={double_fit['tau1'] * 1e6:.1f}, τ2={double_fit['tau2'] * 1e6:.1f} μs",
        )

    ax.set_xlabel("Time (μs)")
    ax.set_ylabel("Counts")
    ax.set_title("Detector Response - Linear Scale")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Log scale
    ax = axes[0, 1]
    ax.semilogy(t_us, counts, "k.", markersize=2, alpha=0.5, label="Data")

    if single_fit:
        t_plot = np.linspace(t[0], t[-1], 500)
        ax.semilogy(
            t_plot * 1e6,
            single_fit["model"](t_plot, *single_fit["popt"]),
            "r-",
            linewidth=2,
            label="Single exp fit",
        )

    if double_fit:
        t_plot = np.linspace(t[0], t[-1], 500)
        ax.semilogy(
            t_plot * 1e6,
            double_fit["model"](t_plot, *double_fit["popt"]),
            "b-",
            linewidth=2,
            label="Double exp fit",
        )

    ax.set_xlabel("Time (μs)")
    ax.set_ylabel("Counts (log)")
    ax.set_title("Detector Response - Log Scale")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Residuals - Single exponential
    if single_fit:
        ax = axes[1, 0]
        mask = (t >= single_fit["fit_range"][0]) & (t <= single_fit["fit_range"][1])
        t_fit = t[mask]
        residuals = counts[mask] - single_fit["model"](t_fit, *single_fit["popt"])
        uncertainties = np.sqrt(counts[mask])
        normalized_residuals = residuals / uncertainties

        ax.plot(t_fit * 1e6, normalized_residuals, "r.", markersize=3)
        ax.axhline(0, color="k", linestyle="--", linewidth=1)
        ax.axhline(2, color="gray", linestyle=":", linewidth=1)
        ax.axhline(-2, color="gray", linestyle=":", linewidth=1)
        ax.set_xlabel("Time (μs)")
        ax.set_ylabel("Normalized Residuals (σ)")
        ax.set_title(f"Single Exp: χ²/dof = {single_fit['chi2_reduced']:.2f}")
        ax.grid(True, alpha=0.3)

    # 4. Residuals - Double exponential
    if double_fit:
        ax = axes[1, 1]
        mask = (t >= double_fit["fit_range"][0]) & (t <= double_fit["fit_range"][1])
        t_fit = t[mask]
        residuals = counts[mask] - double_fit["model"](t_fit, *double_fit["popt"])
        uncertainties = np.sqrt(counts[mask])
        normalized_residuals = residuals / uncertainties

        ax.plot(t_fit * 1e6, normalized_residuals, "b.", markersize=3)
        ax.axhline(0, color="k", linestyle="--", linewidth=1)
        ax.axhline(2, color="gray", linestyle=":", linewidth=1)
        ax.axhline(-2, color="gray", linestyle=":", linewidth=1)
        ax.set_xlabel("Time (μs)")
        ax.set_ylabel("Normalized Residuals (σ)")
        ax.set_title(f"Double Exp: χ²/dof = {double_fit['chi2_reduced']:.2f}")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    print(f"Plot saved to {filename}")


# =============================================================================
# Main analysis
# =============================================================================
if __name__ == "__main__":
    settings = openmc.Settings()
    settings.batches = 10
    settings.particles = 100000

    # Pulsed source configuration
    BURST_DURATION = 1e-6  # 1 μs burst
    MEASUREMENT_TIME = 500e-6  # Measure die-away for 500 μs
    T_TOTAL = BURST_DURATION + MEASUREMENT_TIME

    # Source: neutron burst at t ≈ 0
    source = openmc.IndependentSource(
        space=openmc.stats.Point((0, 0, 0)),
        energy=openmc.stats.Normal(mean_value=14.1e6, std_dev=5.0e4),
        angle=openmc.stats.Isotropic(),
        time=openmc.stats.Uniform(0.0, BURST_DURATION),  # Concentrated burst
    )

    settings.source = source
    settings.run_mode = "fixed source"
    settings.export_to_xml(path="inputs/settings.xml")
    # =============================================================================
    # Tallies - HIGH RESOLUTION TIME BINS
    # =============================================================================
    # Now time bins cover the measurement period with good resolution
    time_bins = np.linspace(0, T_TOTAL, 501)  # ~1 μs per bin
    time_filter = openmc.TimeFilter(time_bins)

    geom = openmc.Geometry.from_xml(
        path="inputs/geometry.xml", materials="inputs/materials.xml"
    )
    he3_cell = [c for c in geom.get_all_cells().values() if c.name == "He3_detector"][0]
    he3_cell_id = he3_cell.id
    tallies = openmc.Tallies()

    # Detector response vs time (key for tau extraction)
    det_response = openmc.Tally(name="detector_response")
    he3_filter = openmc.CellFilter([he3_cell])
    det_response.filters = [he3_filter, time_filter]
    det_response.scores = ["absorption"]
    tallies.append(det_response)
    tallies.export_to_xml(path="inputs/tallies.xml")

    print("\nSimulation parameters:")
    print(f"  Particles: {settings.particles:,}")
    print(f"  Burst duration: {BURST_DURATION * 1e6:.2f} μs")
    print(f"  Measurement time: {MEASUREMENT_TIME * 1e6:.2f} μs")
    print(f"  Time bin width: {(T_TOTAL / 500) * 1e6:.3f} μs")

    base_dir = Path(__file__).parent.resolve()
    input_dir = base_dir / "inputs"
    output_dir = base_dir / "outputs"
    openmc.run(output=False, cwd=output_dir, path_input=input_dir)

    # Load results
    sp = openmc.StatePoint(f"{output_dir}/statepoint.{settings.batches}.h5")

    # Get detector response
    det_resp = sp.get_tally(name="detector_response")
    det_df = det_resp.get_pandas_dataframe()

    t_low = det_df["time low [s]"].values
    t_high = det_df["time high [s]"].values
    t_centers = (t_low + t_high) / 2

    y = det_df["mean"].values
    yerr = det_df["std. dev."].values  # OpenMC uncertainty of mean estimator

    counts = det_df["mean"].values

    # Scale counts to get actual rate (counts per time bin)
    dt = t_high[0] - t_low[0]

    print("\n" + "=" * 70)
    print("TAU EXTRACTION FROM DETECTOR RESPONSE")
    print("=" * 70)

    # Fit single exponential
    print("\n1. SINGLE EXPONENTIAL FIT")
    print("-" * 70)
    single_fit = fit_single_exponential(
        t_centers, y, yerr, t_fit_start=20e-6, t_fit_end=200e-6
    )

    if single_fit:
        print(
            f"  τ = {single_fit['tau'] * 1e6:.2f} ± {single_fit['tau_err'] * 1e6:.2f} μs"
        )
        print(f"  A = {single_fit['A']:.2e}")
        print(f"  B (background) = {single_fit['B']:.2e}")
        print(f"  χ²/dof = {single_fit['chi2_reduced']:.3f}")
        if single_fit["chi2_reduced"] > 2:
            print("  ⚠ Poor fit (χ²/dof > 2), try double exponential")

    # Fit double exponential
    print("\n2. DOUBLE EXPONENTIAL FIT")
    print("-" * 70)
    double_fit = fit_double_exponential(
        t_centers, y, yerr, t_fit_start=20e-6, t_fit_end=300e-6
    )
    if double_fit:
        print(
            f"  τ₁ (fast) = {double_fit['tau1'] * 1e6:.2f} ± {double_fit['tau1_err'] * 1e6:.2f} μs"
        )
        print(
            f"  τ₂ (slow) = {double_fit['tau2'] * 1e6:.2f} ± {double_fit['tau2_err'] * 1e6:.2f} μs"
        )
        print(f"  A₁ = {double_fit['A1']:.2e}")
        print(f"  A₂ = {double_fit['A2']:.2e}")
        print(f"  B = {double_fit['B']:.2e}")
        print(f"  χ²/dof = {double_fit['chi2_reduced']:.3f}")

    # Create diagnostic plots
    plot_fits(t_centers, counts, single_fit, double_fit)

    # Model selection
    print("\n3. MODEL SELECTION")
    print("-" * 70)
    if single_fit and double_fit:
        if double_fit["chi2_reduced"] < single_fit["chi2_reduced"] * 0.8:
            print("  ✓ Double exponential provides significantly better fit")
            print(
                f"    Use: τ₁ = {double_fit['tau1'] * 1e6:.1f} μs, τ₂ = {double_fit['tau2'] * 1e6:.1f} μs"
            )
        else:
            print("  ✓ Single exponential is adequate")
            print(f"    Use: τ = {single_fit['tau'] * 1e6:.1f} μs")

    sp.close()
