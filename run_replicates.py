# run_replicates.py
# Runs N independent OpenMC replicates into outputs/rep_XXXX/
# Uses base XML in ./inputs but writes per-replicate settings.xml and runs in per-rep dir.

from __future__ import annotations

import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import openmc

from utils.analyze_sr import (
    get_measured_multiplicity_causal,
    sr_counts,
    sr_counts_delayed,
)


@dataclass(frozen=True)
class ReplicateConfig:
    n_replicates: int = 10
    particles_per_rep: int = 100_000
    base_seed: Optional[int] = 12345

    gate: float = 32e-6
    predelay: float = 4e-6
    delay: float = 1000e-6

    rate: float = 3e4  # neutrons/sec for source time window T=N/rate

    max_collisions: Optional[int] = None
    max_collision_track_files: int = 100


def run_independent_replicates(
    input_dir: Path,
    output_root: Path,
    cfg: ReplicateConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[np.ndarray], List[int]]:
    input_dir = Path(input_dir)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    # Load geometry once to get He3 cell id
    geom = openmc.Geometry.from_xml(
        path=str(input_dir / "geometry.xml"),
        materials=str(input_dir / "materials.xml"),
    )
    he3_cell = [c for c in geom.get_all_cells().values() if c.name == "He3_detector"][0]
    he3_cell_id = he3_cell.id

    if cfg.base_seed is None:
        base_seed = int(time.time() * 1000) % 2**31
    else:
        base_seed = cfg.base_seed

    all_r: List[np.ndarray] = []
    all_detections: List[int] = []

    for i in range(cfg.n_replicates):
        rep_dir = output_root / f"rep_{i:04d}"
        rep_dir.mkdir(parents=True, exist_ok=True)

        # Copy constant XMLs into each rep dir
        for name in ("materials.xml", "geometry.xml", "tallies.xml"):
            src = input_dir / name
            if src.exists():
                shutil.copy2(src, rep_dir / name)

        # Load settings template and override per-rep values
        settings = openmc.Settings.from_xml(path=str(input_dir / "settings.xml"))
        settings.seed = base_seed + i * 1000
        settings.particles = cfg.particles_per_rep
        settings.batches = 1
        settings.run_mode = "fixed source"
        settings.output = {"path": ".", "tallies": False}

        max_coll = (
            cfg.max_collisions
            if cfg.max_collisions is not None
            else cfg.particles_per_rep
        )
        settings.collision_track = {
            "cell_ids": [he3_cell_id],
            "max_collisions": int(max_coll),
            "max_collision_track_files": int(cfg.max_collision_track_files),
        }

        T = cfg.particles_per_rep / cfg.rate
        settings.source = openmc.IndependentSource(
            space=openmc.stats.Point((0, 0, 0)),
            energy=openmc.stats.Normal(mean_value=14.1e6, std_dev=5.0e4),
            angle=openmc.stats.Isotropic(),
            time=openmc.stats.Uniform(0.0, T),
        )

        settings.export_to_xml(path=str(rep_dir / "settings.xml"))

        # Run using rep_dir for both cwd and input
        openmc.run(output=False, cwd=str(rep_dir), path_input=str(rep_dir))

        # Read collision track for this replicate
        col_track_file = rep_dir / "collision_track.h5"
        collision_tracks = openmc.read_collision_track_hdf5(str(col_track_file))

        # Keep your current choice; validate MT mapping separately
        absorption_events = collision_tracks[collision_tracks["event_mt"] == 101]
        detection_times = np.sort(absorption_events["time"])
        all_detections.append(int(len(detection_times)))

        # SR analysis
        rplusa_counts = sr_counts(detection_times, cfg.predelay, cfg.gate)
        a_counts = sr_counts_delayed(detection_times, cfg.predelay, cfg.gate, cfg.delay)

        rplusa_dist = np.bincount(rplusa_counts)
        a_dist = np.bincount(a_counts, minlength=len(rplusa_dist))

        try:
            r = get_measured_multiplicity_causal(rplusa_dist, a_dist)
            all_r.append(r)
        except Exception as e:
            print(f"[rep {i:04d}] deconvolution failed: {e}")

    if not all_r:
        raise RuntimeError("No successful replicates (all deconvolutions failed).")

    max_len = max(len(r) for r in all_r)
    r_padded = np.array([np.pad(r, (0, max_len - len(r))) for r in all_r])

    r_mean = r_padded.mean(axis=0)
    r_std = r_padded.std(axis=0, ddof=0)
    r_sem = r_std / np.sqrt(len(all_r))

    return r_mean, r_std, r_sem, all_r, all_detections


if __name__ == "__main__":
    base_dir = Path(__file__).parent.resolve()
    input_dir = base_dir / "inputs"
    output_root = base_dir / "outputs"

    cfg = ReplicateConfig(
        n_replicates=10,
        particles_per_rep=100_000,
        base_seed=12345,
        gate=32e-6,
        predelay=4e-6,
        delay=1000e-6,
        rate=3e4,
    )

    r_mean, r_std, r_sem, all_r, all_det = run_independent_replicates(
        input_dir=input_dir,
        output_root=output_root,
        cfg=cfg,
    )

    print("\n" + "=" * 60)
    print(f"FINAL RESULTS ({cfg.n_replicates} independent replicates)")
    print("=" * 60)
    print(f"Mean detections per run: {np.mean(all_det):.0f} +/- {np.std(all_det):.0f}")
    print(f"\n{'idx':>3} | {'mean':>12} | {'std':>12} | {'SEM':>12}")
    print("-" * 50)
    for i in range(min(5, len(r_mean))):
        print(f"{i:>3} | {r_mean[i]:>12.6f} | {r_std[i]:>12.6f} | {r_sem[i]:>12.6f}")

