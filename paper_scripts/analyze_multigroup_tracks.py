# analyze_replicates.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import openmc

# from track_analyzing import Ps_truth_from_tracks, leakage_count_per_source_track
from predicted_multiplicity import r_pred_from_Ps
from utils.analyze_sr import get_measured_multiplicity_causal, sr_histograms_twoptr


@dataclass(frozen=True)
class SRParams:
    predelay: float
    gate: float
    delay: float


def list_rep_dirs(output_root: Path) -> List[Path]:
    return sorted([p for p in Path(output_root).glob("rep_*") if p.is_dir()])


def _cell_ids_from_geometry_xml(rep_dir: Path) -> tuple[int, list[int]]:
    """Return (be_cell_id, he3_cell_id) by cell name from this replicate's geometry.xml."""
    geom = openmc.Geometry.from_xml(
        path=str(rep_dir / "geometry.xml"), materials=str(rep_dir / "materials.xml")
    )
    cells = list(geom.get_all_cells().values())

    be = [c for c in cells if c.name == "beryllium"]
    # he3 = [c for c in cells if c.name in ("he3_tube_")]
    # he3 = [c.id for c in cells if "he3_tube_" in (c.name or "")]
    # he3[0].id
    he3_ids = [c.id for c in cells if (c.name or "").startswith("he3_tube_")]

    if not be:
        raise RuntimeError(f"No cell named 'beryllium' in {rep_dir}/geometry.xml")
    if not he3_ids:
        raise RuntimeError(f"No cell named 'He3_detector' in {rep_dir}/geometry.xml")

    return be[0].id, he3_ids


def analyze_rep_collision_track(
    rep_dir: Path,
    sr: SRParams,
    max_r_k: int = 4,
) -> Tuple[Optional[np.ndarray], Optional[int]]:
    """
    Returns (r, detections) for this replicate from collision_track.h5
    r is padded to at least max_r_k+1
    """
    h5 = rep_dir / "collision_track.h5"
    if not h5.exists():
        return None, None

    ct = openmc.read_collision_track_hdf5(str(h5))
    absorption = ct[ct["event_mt"] == 101]
    detection_times = np.sort(absorption["time"])
    detections = int(detection_times.size)

    rplusa_dist, a_dist = sr_histograms_twoptr(
        detection_times, sr.predelay, sr.gate, sr.delay, cap=64
    )

    r = get_measured_multiplicity_causal(rplusa_dist, a_dist)
    if len(r) < (max_r_k + 1):
        r = np.pad(r, (0, (max_r_k + 1) - len(r)))
    return r, detections


def get_eps_denom_and_Ps(
    tracks_path: Path,
    be_cell_id: int,
    vmax: Optional[int] = None,
) -> tuple[int, np.ndarray, np.ndarray]:
    """
    Estimate per-leaked-neutron detection efficiency epsilon from tracks.h5:

    For each source history i:
      v_i = # neutrons that leave Be at least once (your definition)

    MLE under Binomial(n_i | v_i, epsilon): epsilon_hat = sum(n_i)/sum(v_i)
    Returns (epsilon_hat, v_list)
    """
    # this takes a while
    tracks = openmc.Tracks(str(tracks_path))  # Keep as generator
    v_list = []
    leaked_energies = []
    for tr in tracks:
        leakers = 0
        for ptype, states in tr.particle_tracks:
            if getattr(ptype, "name", str(ptype)) != "NEUTRON":
                continue

            cell_ids = states["cell_id"]
            E = states["E"]

            # outside after first entry into be_cell_id
            # idx_in = np.where(cell_ids == be_cell_id)[0]
            # if idx_in.size == 0:
            #     continue
            #
            # first_in = idx_in[0]
            #
            # # find first index after first_in where particle is no longer in be_cell_id
            # out_rel = np.where(cell_ids[first_in + 1 :] != be_cell_id)[0]
            # # if np.any(cell_ids[first_in + 1 :] != be_cell_id):
            # if out_rel.size > 0:
            #     leakers += 1
            #     first_out = first_in + 1 + out_rel[0]
            #     leaked_energies.append(float(E[first_out]))  # scalar
            hdpe_id = 5
            entered_be = np.any(cell_ids == be_cell_id)
            if not entered_be:
                continue

            ever_in_hdpe = np.any(cell_ids == hdpe_id)
            if ever_in_hdpe:
                leakers += 1
                idx_hdpe = np.where(cell_ids == hdpe_id)[0][0]
                leaked_energies.append(float(E[idx_hdpe]))

        v_list.append(leakers)

    v_arr = np.asarray(v_list, dtype=np.int64)
    v_sum = int(v_arr.sum())

    vmax_eff = int(v_arr.max()) if (vmax is None and v_arr.size) else int(vmax or 0)
    counts = np.bincount(v_arr, minlength=vmax_eff + 1)
    Ps = counts / counts.sum() if counts.sum() > 0 else counts.astype(float)

    # bimodal analysis
    leaked_energies = np.asarray(leaked_energies, dtype=float)
    plot_leaked_energy(leaked_energies)

    THRESHOLD = 9e6
    f_fast = (leaked_energies > THRESHOLD).mean() if leaked_energies.size else 0.0
    f_thermal = 1 - f_fast

    print(f"Fast fraction: {f_fast:.3f}")
    print(f"Thermal fraction: {f_thermal:.3f}")

    return v_sum, Ps, v_arr, f_fast, f_thermal


def plot_leaked_energy(leaked_energies, figname=None):
    hist, edges = np.histogram(leaked_energies, bins=np.logspace(4, 7.5, 500))
    bin_centers = (edges[1:] + edges[:-1]) / 2

    THRESHOLD = 9e6
    plt.figure(figsize=(10, 6))
    plt.semilogx(bin_centers / 1e6, hist, "o-", label="Leakage spectrum")
    plt.axvline(
        THRESHOLD / 1e6,
        color="red",
        linestyle="--",
        label=f"Threshold = {THRESHOLD / 1e6:.1f} MeV",
    )
    plt.xlabel("Neutron Energy at Leakage (MeV)")
    plt.ylabel("Count")
    plt.title("Bimodal Leakage Energy Spectrum")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("leakage_energy_spectrum.png", dpi=300)

    print(f"\nPeak 1 (thermal): {bin_centers[hist[:50].argmax()] / 1e6:.2f} MeV")
    print(f"Peak 2 (fast): {bin_centers[50:][hist[50:].argmax()] / 1e6:.2f} MeV")


def compute_replicate_stats(
    output_root: Path,
    sr: SRParams,
    max_r_k: int = 4,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[np.ndarray], List[int]]:
    rep_dirs = list_rep_dirs(output_root)
    if not rep_dirs:
        raise FileNotFoundError(f"No rep_* dirs found in {output_root}")

    all_r: List[np.ndarray] = []
    all_det: List[int] = []

    for rep in rep_dirs:
        r, det = analyze_rep_collision_track(rep, sr=sr, max_r_k=max_r_k)
        if det is not None:
            all_det.append(det)
        if r is not None:
            all_r.append(r)

    if not all_r:
        raise RuntimeError(f"No successful SR deconvolutions in {output_root}")

    max_len = max(len(r) for r in all_r)
    r_padded = np.array([np.pad(r, (0, max_len - len(r))) for r in all_r])

    r_mean = r_padded.mean(axis=0)
    r_std = r_padded.std(axis=0, ddof=0)
    r_sem = r_std / np.sqrt(len(all_r))

    return r_mean, r_std, r_sem, all_r, all_det


def get_group_efficiencies_from_tracks(
    tracks_path: Path,
    be_cell_id: int,
    he3_cell_ids: set[int],
    energy_threshold: float = 9e6,
) -> tuple[float, float]:
    """
    Compute group efficiencies using only particle tracks.
    A neutron is "detected" if it enters an He-3 cell after leaking from Be.
    """
    tracks = openmc.Tracks(str(tracks_path))

    n_leaked_fast = 0
    n_leaked_thermal = 0
    n_detected_fast = 0
    n_detected_thermal = 0

    for tr in tracks:
        for ptype, states in tr.particle_tracks:
            if getattr(ptype, "name", str(ptype)) != "NEUTRON":
                continue

            cell_ids = states["cell_id"]
            E = states["E"]

            # Find first entry into Be
            idx_in = np.where(cell_ids == be_cell_id)[0]
            if idx_in.size == 0:
                continue
            first_in = idx_in[0]

            # Find first exit from Be
            out_rel = np.where(cell_ids[first_in + 1 :] != be_cell_id)[0]
            if out_rel.size == 0:
                continue  # Never leaked

            first_out = first_in + 1 + out_rel[0]
            E_leak = float(E[first_out])

            # Count as leaked
            if E_leak > energy_threshold:
                n_leaked_fast += 1
                is_fast = True
            else:
                n_leaked_thermal += 1
                is_fast = False

            # Check if this neutron entered any He-3 cell later
            final = states[-1]
            detected = final["cell_id"] in he3_cell_ids

            if detected:
                if is_fast:
                    n_detected_fast += 1
                else:
                    n_detected_thermal += 1

    return n_detected_fast, n_leaked_fast, n_detected_thermal, n_leaked_thermal


def measure_tau_from_leakage(
    tracks_path, collision_tracks_path, be_cell_id, he3_cell_ids
):
    """
    Measure τ from when neutron LEAVES Be, not when source emits
    """
    # Build map: particle_id -> leakage_time
    tracks = openmc.Tracks(str(tracks_path))
    he3_set = set(he3_cell_ids)
    delta_t_list = []

    for tr in tracks:
        for ptype, states in tr.particle_tracks:
            if getattr(ptype, "name", str(ptype)) != "NEUTRON":
                continue

            cell_ids = states["cell_id"]
            times = states["time"]

            idx_in_be = np.where(cell_ids == be_cell_id)[0]
            if idx_in_be.size == 0:
                continue

            last_in_be = idx_in_be[-1]  # Last index where particle is in Be
            t_leak = float(times[last_in_be])

            # Find first entry into He-3 AFTER leaving Be
            idx_after = last_in_be + 1
            if idx_after >= len(cell_ids):
                continue

            later_cells = cell_ids[idx_after:]
            later_times = times[idx_after:]

            he3_entries = np.where(np.isin(later_cells, list(he3_set)))[0]

            if he3_entries.size > 0:
                first_he3 = he3_entries[0]
                t_detect = float(later_times[first_he3])
                delta_t = t_detect - t_leak

                # Physical sanity: minimum ~1 μs to cross 6 cm of HDPE
                if delta_t > 1e-6:
                    delta_t_list.append(delta_t)

    # Histogram and fit
    delta_t = np.array(delta_t_list)
    print(f"Found {len(delta_t)} leakage→detection events")
    print(f"  Mean delay: {np.mean(delta_t) * 1e6:.1f} μs")
    print(f"  Median delay: {np.median(delta_t) * 1e6:.1f} μs")

    # Histogram
    bins = np.linspace(0, 300e-6, 300)
    counts, edges = np.histogram(delta_t, bins=bins)
    t_centers = (edges[1:] + edges[:-1]) / 2

    # Fit exponential - FIT IN LOG SPACE
    mask = (t_centers > 4e-6) & (t_centers < 32e-6) & (counts > 5)

    t_fit = t_centers[mask]
    c_fit = counts[mask]

    # LINEAR FIT in log space: log(c) = log(A) - t/tau
    log_c = np.log(c_fit)

    # Weighted linear fit (weight by counts to handle low-count bins)
    weights = np.sqrt(c_fit)
    slope, intercept = np.polyfit(t_fit, log_c, 1, w=weights)

    tau_intrinsic = -1.0 / slope  # tau = -1/slope
    A = np.exp(intercept)

    # Estimate uncertainty
    residuals = log_c - (slope * t_fit + intercept)
    sigma_slope = np.sqrt(
        np.sum((residuals * weights) ** 2) / (len(t_fit) - 2)
    ) / np.sqrt(np.sum(weights**2 * (t_fit - t_fit.mean()) ** 2))
    tau_err = tau_intrinsic**2 * sigma_slope

    print(
        f"\nτ_intrinsic (from leakage) = {tau_intrinsic * 1e6:.1f} ± {tau_err * 1e6:.1f} μs"
    )
    print(f"A = {A:.2e}")
    print(f"Slope = {slope:.3e} (should be negative)")

    # Plot
    plt.figure(figsize=(10, 6))
    plt.semilogy(t_centers * 1e6, counts + 1, "o", alpha=0.5, label="Data")

    # Plot fit
    t_plot = np.linspace(0, 300e-6, 500)
    fit_curve = A * np.exp(-t_plot / tau_intrinsic)
    plt.plot(
        t_plot * 1e6,
        fit_curve,
        "r--",
        linewidth=2,
        label=f"Fit: τ = {tau_intrinsic * 1e6:.1f} μs",
    )

    plt.axvline(20, color="gray", linestyle=":", alpha=0.5, label="Fit range start")
    plt.axvline(150, color="gray", linestyle=":", alpha=0.5, label="Fit range end")
    plt.xlabel("Time from leakage (μs)")
    plt.ylabel("Count")
    plt.title(f"Die-away from leakage: τ = {tau_intrinsic * 1e6:.1f} μs")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig("tau_from_leakage.png", dpi=150)

    return tau_intrinsic, delta_t


def main() -> None:
    OUTPUT_ROOT = Path("outputs/reps_with_tracksh5")

    # SR params (match your run)
    TAU = 40e-6
    sr = SRParams(predelay=4e-6, gate=28e-6, delay=1000e-6)
    MAX_RK = 3

    r_mean, r_std, r_sem, all_r, all_det = compute_replicate_stats(
        OUTPUT_ROOT, sr=sr, max_r_k=MAX_RK
    )

    det_arr = np.asarray(all_det, dtype=float)
    print(f"Replicates found: {len(list_rep_dirs(OUTPUT_ROOT))}")
    print(f"SR-success reps:  {len(all_r)}")
    print(
        f"Detections/rep:   {det_arr.mean():.1f} ± {det_arr.std(ddof=0) / np.sqrt(det_arr.size):.1f} (SEM)"
    )

    print(f"\n{'idx':>3} | {'mean':>12} | {'std':>12} | {'SEM':>12}")
    print("-" * 50)
    for i in range(min(len(r_mean), MAX_RK + 1)):
        print(f"{i:>3} | {r_mean[i]:>12.6f} | {r_std[i]:>12.6f} | {r_sem[i]:>12.6f}")

    # Detection efficiency:
    # (A) "absolute" efficiency per *source history* requires knowing particles per rep
    #     abs_eff_rep = detections / particles_per_rep

    # (B) per-leaked-neutron epsilon from tracks.h5 (preferred for your Ps->r_pred model)
    rep_dirs = list_rep_dirs(OUTPUT_ROOT)[:1]

    eps_per_rep = []

    for i, rep in enumerate(rep_dirs):
        openmc.reset_auto_ids()
        tracks_path = rep / "tracks.h5"
        collision_tracks_path = rep / "collision_track.h5"

        if not tracks_path.exists():
            continue
        be_id, he3_id = _cell_ids_from_geometry_xml(rep)

        # measure_tau_from_leakage(tracks_path, collision_tracks_path, be_id, he3_id)
        get_eps_denom_and_Ps(tracks_path, be_id)
        # n_detected_fast, n_leaked_fast, n_detected_thermal, n_leaked_thermal = (
        #     get_group_efficiencies_from_tracks(
        #         tracks_path,
        #         be_id,
        #         he3_id,
        #         energy_threshold=9e6,
        #     )
        # )
        eps_fast = n_detected_fast / n_leaked_fast if n_leaked_fast > 0 else 0.0
        eps_thermal = (
            n_detected_thermal / n_leaked_thermal if n_leaked_thermal > 0 else 0.0
        )

        print(f"ε_fast = {eps_fast:.4f} ({n_detected_fast}/{n_leaked_fast})")
        print(
            f"ε_thermal = {eps_thermal:.4f} ({n_detected_thermal}/{n_leaked_thermal})"
        )

        # v_sum, Ps_truth, v_arr, f_fast, f_thermal = get_eps_denom_and_Ps(
        #     tracks_path,
        #     be_id,
        # )

    # Final Pooled Stats
    # print("Ps_truth:")
    # for i, v in enumerate(Ps_truth):
    #     print(f"  [{i}] {v:.6e}")

    # This is your Srinivasan Model calculation
    r_predicted_fast = r_pred_from_Ps(Ps_fast, eps_fast, sr.predelay, sr.gate, tau_fast)
    r_predicted_thermal = r_pred_from_Ps(
        Ps_thermal, eps_thermal, sr.predelay, sr.gate, tau_thermal
    )

    r_predicted = r_fast + r_thermal

    # print(f"\nFinal Pooled Epsilon: {eps_pooled:.6f}")
    print(f"{'idx':>3} | {'r_measured':>15} | {'r_predicted':>15} | {'Error %':>10}")
    print("-" * 55)
    for k in range(min(len(r_mean), len(r_predicted), MAX_RK + 1)):
        err = (
            (r_mean[k] - r_predicted[k]) / r_predicted[k] * 100
            if r_predicted[k] != 0
            else 0
        )
        print(f"{k:>3} | {r_mean[k]:>15.6e} | {r_predicted[k]:>15.6e} | {err:>9.2f}%")


if __name__ == "__main__":
    main()
