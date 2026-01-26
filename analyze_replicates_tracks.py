# analyze_replicates.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import openmc

# from track_analyzing import Ps_truth_from_tracks, leakage_count_per_source_track
from track_analyzing import r_pred_from_Ps
from utils.analyze_sr import (
    get_measured_multiplicity_causal,
    sr_counts,
    sr_counts_delayed,
)


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

    rplusa_counts = sr_counts(detection_times, sr.predelay, sr.gate)
    a_counts = sr_counts_delayed(detection_times, sr.predelay, sr.gate, sr.delay)

    rplusa_dist = np.bincount(rplusa_counts)
    a_dist = np.bincount(a_counts, minlength=len(rplusa_dist))

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
    tracks = openmc.Tracks(str(tracks_path))  # Keep as generator

    v_list = []
    for tr in tracks:
        leakers = 0
        for ptype, states in tr.particle_tracks:
            if getattr(ptype, "name", str(ptype)) != "NEUTRON":
                continue

            cell_ids = states["cell_id"]

            # has it ever been inside Be?
            # A neutron that goes Be → air cavity → back into Be → absorbed in Be would count as
            # a leaker, even though it never escaped the assembly.
            # was_in_be = np.any(cell_ids == be_cell_id)
            # if not was_in_be:
            #     continue
            # # did it ever go outside after being inside?
            # if np.any(cell_ids != be_cell_id):
            #     leakers += 1

            # outside after first entry into be_cell_id
            idx_in = np.where(cell_ids == be_cell_id)[0]
            if idx_in.size == 0:
                continue
            first_in = idx_in[0]
            if np.any(cell_ids[first_in + 1 :] != be_cell_id):
                leakers += 1

        v_list.append(leakers)

    v_arr = np.asarray(v_list, dtype=np.int64)
    v_sum = int(v_arr.sum())

    vmax_eff = int(v_arr.max()) if (vmax is None and v_arr.size) else int(vmax or 0)
    counts = np.bincount(v_arr, minlength=vmax_eff + 1)
    Ps = counts / counts.sum() if counts.sum() > 0 else counts.astype(float)

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


def main() -> None:
    OUTPUT_ROOT = Path("outputs")

    # SR params (match your run)
    TAU = 69e-6
    sr = SRParams(predelay=4e-6, gate=85e-6, delay=1000e-6)
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
    rep_dirs = list_rep_dirs(OUTPUT_ROOT)[:5]

    num_total = 0
    den_total = 0
    total_counts = None  # pooled histogram for Ps(v)

    eps_per_rep = []

    for i, rep in enumerate(rep_dirs):
        openmc.reset_auto_ids()
        tracks_path = rep / "tracks.h5"
        if not tracks_path.exists():
            continue
        be_id, he3_id = _cell_ids_from_geometry_xml(rep)
        v_sum, Ps_truth, v_arr = get_eps_denom_and_Ps(tracks_path, be_id)

        det_rep = all_det[i]
        eps_per_rep.append(det_rep / v_sum)

        num_total += det_rep
        den_total += v_sum

        # pool Ps counts
        c = np.bincount(v_arr)
        if total_counts is None:
            total_counts = c.astype(np.int64)
        else:
            if c.size > total_counts.size:
                total_counts = np.pad(total_counts, (0, c.size - total_counts.size))
            elif c.size < total_counts.size:
                c = np.pad(c, (0, total_counts.size - c.size))
            total_counts += c
        print(
            f"[{rep.name}] eps_hat = {det_rep} / {v_sum} = {det_rep / v_sum:.4f} (from {v_sum} leakers)"
        )

    # Final Pooled Stats
    eps_pooled = (num_total / den_total) if den_total > 0 else 0.0
    Ps_truth = total_counts / total_counts.sum() if total_counts is not None else None

    print("Ps_truth:")
    for i, v in enumerate(Ps_truth):
        print(f"  [{i}] {v:.6e}")

    # This is your Srinivasan Model calculation
    r_predicted = r_pred_from_Ps(Ps_truth, eps_pooled, sr.predelay, sr.gate, TAU)

    print(f"\nFinal Pooled Epsilon: {eps_pooled:.6f}")
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
