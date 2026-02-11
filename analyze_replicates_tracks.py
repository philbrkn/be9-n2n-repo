# analyze_replicates.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

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

    # for cell in geom.get_all_cells().values():
    #     print(f"Cell ID = {cell.id:4d} | name = {cell.name}")
    hdpe_id = 5
    be = [c for c in cells if c.name == "beryllium"]
    cd = [c for c in cells if c.name == "cadmium_lining"]
    # he3 = [c for c in cells if c.name in ("he3_tube_")]
    # he3 = [c.id for c in cells if "he3_tube_" in (c.name or "")]
    # he3[0].id
    he3_ids = [c.id for c in cells if (c.name or "").startswith("he3_tube_")]

    if not be:
        raise RuntimeError(f"No cell named 'beryllium' in {rep_dir}/geometry.xml")
    if not he3_ids:
        raise RuntimeError(f"No cell named 'He3_detector' in {rep_dir}/geometry.xml")

    return be[0].id, he3_ids, cd[0].id, hdpe_id


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
    cd_cell_id: int,
    hdpe_id: int,
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
    print(f"Total source particles: {len(tracks)}")
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
            entered_be = np.any(cell_ids == be_cell_id)
            if not entered_be:
                continue

            ever_in_hdpe = np.any(cell_ids == hdpe_id)
            if ever_in_hdpe:
                leakers += 1
                idx_hdpe = np.where(cell_ids == hdpe_id)[0][0]
                leaked_energies.append(float(E[idx_hdpe]))
            # first_in = idx_in[0]
            #
            # # find first index after first_in where particle is no longer in be_cell_id
            # out_rel = np.where(cell_ids[first_in + 1 :] != be_cell_id)[0]
            # # if np.any(cell_ids[first_in + 1 :] != be_cell_id):
            # if out_rel.size > 0:
            #     first_out = first_in + 1 + out_rel[0]
            #     next_cell = cell_ids[first_out]
            #     if next_cell == cd_cell_id:
            #         leakers += 1
            #         leaked_energies.append(float(E[first_out]))  # scalar

        v_list.append(leakers)

    v_arr = np.asarray(v_list, dtype=np.int64)
    v_sum = int(v_arr.sum())

    vmax_eff = int(v_arr.max()) if (vmax is None and v_arr.size) else int(vmax or 0)
    counts = np.bincount(v_arr, minlength=vmax_eff + 1)
    Ps = counts / counts.sum() if counts.sum() > 0 else counts.astype(float)

    # bimodal analysis
    leaked_energies = np.asarray(leaked_energies, dtype=float)
    hist, edges = np.histogram(leaked_energies, bins=np.logspace(5, 7.5, 100))
    THRESHOLD = 5e6
    f_fast = (leaked_energies > THRESHOLD).mean() if leaked_energies.size else 0.0
    f_thermal = 1 - f_fast

    # print(f"Fast fraction: {f_fast:.3f}")
    # print(f"Thermal fraction: {f_thermal:.3f}")

    return v_sum, Ps, v_arr


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


def compute_predicted_with_uncertainty(rep_dirs, sr, tau, max_rk, all_det):
    """Compute r_predicted for each replicate, then get mean/SEM."""

    r_pred_all = []

    for i, rep in enumerate(rep_dirs):
        openmc.reset_auto_ids()
        tracks_path = rep / "tracks.h5"
        if not tracks_path.exists():
            continue

        be_id, he3_id, cd_id, hdpe_id = _cell_ids_from_geometry_xml(rep)
        v_sum, Ps_rep, v_arr = get_eps_denom_and_Ps(tracks_path, be_id, cd_id, hdpe_id)

        det_rep = all_det[i]
        eps_rep = det_rep / v_sum

        # Predict r from THIS replicate's Ps and epsilon
        r_pred_rep = r_pred_from_Ps(Ps_rep, eps_rep, sr.predelay, sr.gate, tau)
        r_pred_all.append(r_pred_rep)

    # Pad and compute stats
    max_len = max(len(r) for r in r_pred_all)
    r_pred_padded = np.array([np.pad(r, (0, max_len - len(r))) for r in r_pred_all])

    r_pred_mean = r_pred_padded.mean(axis=0)
    r_pred_std = r_pred_padded.std(axis=0, ddof=1)
    r_pred_sem = r_pred_std / np.sqrt(len(r_pred_all))

    return r_pred_mean, r_pred_sem, r_pred_all


def main() -> None:
    OUTPUT_ROOT = Path("outputs/reps_with_tracksh5")

    GATE = 28e-6
    PREDELAY = 4e-6
    DELAY = 1000e-6
    # SR params (match your run)
    TAU = 70e-6
    MAX_RK = 3

    sr = SRParams(predelay=PREDELAY, gate=GATE, delay=DELAY)

    # col_track_file = output_root / "standard_rep_0000_1e8p" / "collision_track.h5"
    # col_track_file = OUTPUT_ROOT / "collision_track.h5"
    # r_full, r_mean, r_std, arr = analyze_with_bootstrap_parallel(
    #     col_track_file,
    #     sr,
    #     n_bootstrap=200,
    #     segment_duration=1.0,
    #     seed=12345,
    #     n_workers=32,
    #     chunk_size=1,
    # )

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

    rep_dirs = list_rep_dirs(OUTPUT_ROOT)  # [:5]

    # with uncerstainties
    r_pred_mean, r_pred_sem, r_pred_all = compute_predicted_with_uncertainty(
        rep_dirs, sr, TAU, MAX_RK, all_det
    )

    print(
        f"{'k':>3} | {'r_meas':>12} | {'σ_meas':>10} | {'r_pred':>12} | {'σ_pred':>10} | {'Δ/σ':>8}"
    )
    print("-" * 70)
    for k in range(MAX_RK + 1):
        delta = r_mean[k] - r_pred_mean[k]
        sigma_combined = np.sqrt(r_sem[k] ** 2 + r_pred_sem[k] ** 2)
        n_sigma = delta / sigma_combined if sigma_combined > 0 else 0
        print(
            f"{k:>3} | {r_mean[k]:>12.4e} | {r_sem[k]:>10.2e} | {r_pred_mean[k]:>12.4e} | {r_pred_sem[k]:>10.2e} | {n_sigma:>8.1f}"
        )

    # num_total = 0
    # den_total = 0
    # total_counts = None  # pooled histogram for Ps(v)
    #
    # eps_per_rep = []
    #
    # for i, rep in enumerate(rep_dirs):
    #     openmc.reset_auto_ids()
    #     tracks_path = rep / "tracks.h5"
    #     if not tracks_path.exists():
    #         continue
    #     be_id, he3_id, cd_id, hdpe_id = _cell_ids_from_geometry_xml(rep)
    #
    #     v_sum, Ps_truth, v_arr = get_eps_denom_and_Ps(
    #         tracks_path, be_id, cd_id, hdpe_id
    #     )
    #
    #     det_rep = all_det[i]
    #     eps_per_rep.append(det_rep / v_sum)
    #
    #     num_total += det_rep
    #     den_total += v_sum
    #
    #     # pool Ps counts
    #     c = np.bincount(v_arr)
    #     if total_counts is None:
    #         total_counts = c.astype(np.int64)
    #     else:
    #         if c.size > total_counts.size:
    #             total_counts = np.pad(total_counts, (0, c.size - total_counts.size))
    #         elif c.size < total_counts.size:
    #             c = np.pad(c, (0, total_counts.size - c.size))
    #         total_counts += c
    #     print(
    #         f"[{rep.name}] eps_hat = {det_rep} / {v_sum} = {det_rep / v_sum:.4f} (from {v_sum} leakers)"
    #     )
    #
    # # Final Pooled Stats
    # eps_pooled = (num_total / den_total) if den_total > 0 else 0.0
    # Ps_truth = total_counts / total_counts.sum() if total_counts is not None else None
    #
    # print("Ps_truth:")
    # for i, v in enumerate(Ps_truth):
    #     print(f"  [{i}] {v:.6e}")
    #
    # # This is your Srinivasan Model calculation
    # r_predicted = r_pred_from_Ps(Ps_truth, eps_pooled, sr.predelay, sr.gate, TAU)
    #
    # print(f"\nFinal Pooled Epsilon: {eps_pooled:.6f}")
    # print(f"{'idx':>3} | {'r_measured':>15} | {'r_predicted':>15} | {'Error %':>10}")
    # print("-" * 55)
    # for k in range(min(len(r_mean), len(r_predicted), MAX_RK + 1)):
    #     err = (
    #         (r_mean[k] - r_predicted[k]) / r_predicted[k] * 100
    #         if r_predicted[k] != 0
    #         else 0
    #     )
    #     print(f"{k:>3} | {r_mean[k]:>15.6e} | {r_predicted[k]:>15.6e} | {err:>9.2f}%")


if __name__ == "__main__":
    main()
