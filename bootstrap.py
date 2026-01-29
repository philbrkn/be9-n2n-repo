from __future__ import annotations

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import Optional

import numpy as np
import openmc

from utils.analyze_sr import (
    get_measured_multiplicity_causal,
    sr_counts,
    sr_counts_delayed,
)

# Worker globals
_SEGMENTS = None
_SR = None
_SEG_DUR = None
_SEED = None


@dataclass(frozen=True)
class SRParams:
    predelay: float
    gate: float
    delay: float


@dataclass(frozen=True)
class ReplicateConfig:
    n_replicates: int = 10
    particles_per_rep: int = 100_000
    base_seed: Optional[int] = 12345

    be_radius: float = 9
    be_density: float = 1.85
    he_density: float = 0.0005

    n_tubes: int = 20
    # (later) he3_radius: float = 1.5, he3_radial_pos: float = 15.0

    gate: float = 85e-6
    predelay: float = 4e-6
    delay: float = 1000e-6
    rate: float = 3e4  # neutrons/sec for source time window T=N/rate

    max_collisions: Optional[int] = None


def analyze_with_bootstrap_histograms(
    collision_track_path: Path,
    sr: SRParams,
    n_bootstrap: int = 200,
    segment_duration: float = 10.0,
    progress_every: int = 10,
    seed: int | None = 12345,
    gap_between_blocks: float | None = None,
):
    """

    segment_duration: seconds per segment (should be >> tau, << total time)
    tau is around 1e-4 seconds, time is 3e4 seconds
    """
    rng = np.random.default_rng(seed)

    # Load the file
    ct = openmc.read_collision_track_hdf5(str(collision_track_path))
    absorption_events = ct[ct["event_mt"] == 101]
    detection_times = np.sort(absorption_events["time"])

    # run time
    t_min = float(detection_times.min())
    t_max = float(detection_times.max())
    total_time = t_max - t_min
    print(f"Total detection time {total_time:.2f} seconds")
    # n segments: how many time chunks the experiment is divided into
    # n bootstrap: how many times we resample those segments to estimate uncertainty
    n_segments = int(ceil(total_time / segment_duration))  # use ceil to keep tail
    print(f"Using {n_segments} segments of {segment_duration:.2f} s")

    # Segment edges: force last edge to include last event
    segment_edges = t_min + np.arange(n_segments + 1) * segment_duration
    segment_edges[-1] = t_max + 1e-12
    # print(f"segment_edges first 5: {segment_edges[:5]}")

    if gap_between_blocks is None:
        gap_between_blocks = 0.0

    block_r = []
    block_a = []

    # build per block histograms
    for i in range(n_segments):
        t_start, t_end = segment_edges[i], segment_edges[i + 1]
        mask = (detection_times >= t_start) & (detection_times < t_end)
        seg = detection_times[mask]

        if seg.size < 2:
            # keep as zeros to avoid bias
            rplusa_dist = np.zeros(1, dtype=np.int64)
            a_dist = np.zeros(1, dtype=np.int64)
        else:
            rplusa = sr_counts(seg, sr.predelay, sr.gate)
            a = sr_counts_delayed(seg, sr.predelay, sr.gate, sr.delay)
            rplusa_dist = np.bincount(rplusa).astype(np.int64)
            a_dist = np.bincount(a).astype(np.int64)

        L = max(len(rplusa_dist), len(a_dist))
        if len(rplusa_dist) < L:
            rplusa_dist = np.pad(rplusa_dist, (0, L - len(rplusa_dist)))
        if len(a_dist) < L:
            a_dist = np.pad(a_dist, (0, L - len(a_dist)))

        block_r.append(rplusa_dist)
        block_a.append(a_dist)
        if progress_every and ((i + 1) % progress_every == 0):
            print(f"segmenting {i + 1}/{n_segments}")

    # bootstrap
    n_blocks = len(block_r)
    failures = 0
    boots = []
    for b in range(n_bootstrap):
        idx = rng.integers(0, n_blocks, size=n_blocks)  # sample blocks with replacement

        Lb = 1
        for j in idx:
            Lb = max(Lb, len(block_r[j]), len(block_a[j]))

        Rb = np.zeros(Lb, dtype=np.int64)
        Ab = np.zeros(Lb, dtype=np.int64)

        for j in idx:
            r_d = block_r[j]
            a_d = block_a[j]
            Rb[: len(r_d)] += r_d
            Ab[: len(a_d)] += a_d

        try:
            boots.append(get_measured_multiplicity_causal(Rb, Ab))
        except Exception:
            failures += 1
            continue

        if progress_every and ((b + 1) % progress_every == 0):
            print(f"bootstrap {b + 1}/{n_bootstrap}")

    print(f"There were {failures} deconvolution failures during bootstrapping.")

    L = max(len(r) for r in boots)
    arr = np.array([np.pad(r, (0, L - len(r))) for r in boots], dtype=float)

    r_mean = arr.mean(axis=0)
    r_std = arr.std(axis=0, ddof=1)
    r_sem = r_std / np.sqrt(arr.shape[0])

    # point estimate: sum all blocks once, then invert
    # L_full = max(len(x) for x in block_r + block_a)
    # R_full = np.zeros(L_full, dtype=np.int64)
    # A_full = np.zeros(L_full, dtype=np.int64)
    # for r_d, a_d in zip(block_r, block_a):
    #     R_full[: len(r_d)] += r_d
    #     A_full[: len(a_d)] += a_d
    # r_full = get_measured_multiplicity_causal(R_full, A_full)

    rplusa_counts_full = sr_counts(detection_times, sr.predelay, sr.gate)
    a_counts_full = sr_counts_delayed(detection_times, sr.predelay, sr.gate, sr.delay)
    rplusa_dist_full = np.bincount(rplusa_counts_full)
    a_dist_full = np.bincount(a_counts_full, minlength=len(rplusa_dist))
    r_full = get_measured_multiplicity_causal(rplusa_dist_full, a_dist_full)

    return r_full, r_mean, r_std, r_sem


def analyze_with_bootstrap(
    collision_track_path: Path,
    sr: SRParams,
    n_bootstrap: int = 200,
    segment_duration: float = 10.0,
    progress_every: int = 10,
    seed: int | None = 12345,
):
    """
    LIST-mode block bootstrap:
      1) split detection event times into contiguous time blocks
      2) bootstrap by resampling blocks with replacement
      3) create synthetic pulse train by concatenating selected blocks
      4) compute SR histograms and deconvolve for each replicate

    segment_duration: seconds per segment (should be >> tau, << total time)
    tau is around 1e-4 seconds, time is 3e4 seconds
    """
    rng = np.random.default_rng(seed)

    # Load the file
    ct = openmc.read_collision_track_hdf5(str(collision_track_path))
    absorption_events = ct[ct["event_mt"] == 101]
    detection_times = np.sort(absorption_events["time"])

    # run time
    t_min = float(detection_times.min())
    t_max = float(detection_times.max())
    total_time = t_max - t_min
    n_segments = int(ceil(total_time / segment_duration))  # use ceil to keep tail
    # n segments: how many time chunks the experiment is divided into
    # n bootstrap: how many times we resample those segments to estimate uncertainty
    print(f"Total detection time {total_time:.2f} seconds")
    print(f"Using {n_segments} segments of {segment_duration:.2f} s")

    # Segment edges: force last edge to include last event
    segment_edges = t_min + np.arange(n_segments + 1) * segment_duration
    segment_edges[-1] = t_max + 1e-12
    # print(f"segment_edges first 5: {segment_edges[:5]}")

    # Use searchsorted for O(n log n) instead of O(n * n_segments)
    segment_indices = (
        np.searchsorted(segment_edges[:-1], detection_times, side="right") - 1
    )
    segment_indices = np.clip(segment_indices, 0, n_segments - 1)
    # Extract pulse trains for each segment (times relative to segment start)
    segments = []
    for i in range(n_segments):
        mask = segment_indices == i
        seg_times = detection_times[mask] - segment_edges[i]
        segments.append(seg_times)
        if progress_every and ((i + 1) % 1000 == 0):
            print(f"segment {i + 1}/{n_segments}")

    print(f"Starting bootstrap for {n_bootstrap} bootstraps.")
    # Bootstrap: resample segments and concatenate pulse trains
    boots = []
    failures = 0
    for b in range(n_bootstrap):
        # Sample N segments with replacement
        idx = rng.integers(0, n_segments, size=n_segments)

        # Chain segments into a single synthetic pulse train
        # Each segment is offset by its position in the chain
        synthetic_train = []
        current_offset = 0.0

        for j in idx:
            seg = segments[j]
            if len(seg) > 0:
                synthetic_train.append(seg + current_offset)
            current_offset += segment_duration

        if len(synthetic_train) == 0:
            failures += 1
            continue

        synthetic_times = np.concatenate(synthetic_train)
        synthetic_times = np.sort(synthetic_times)  # Ensure sorted

        # Now run SR analysis on the concatenated pulse train
        # (This is what the paper describes)
        rplusa_counts = sr_counts(synthetic_times, sr.predelay, sr.gate)
        a_counts = sr_counts_delayed(synthetic_times, sr.predelay, sr.gate, sr.delay)

        rplusa_dist = np.bincount(rplusa_counts)
        a_dist = np.bincount(a_counts, minlength=len(rplusa_dist))

        try:
            r = get_measured_multiplicity_causal(rplusa_dist, a_dist)
            boots.append(r)
        except Exception:
            failures += 1
            continue

        if progress_every and ((b + 1) % progress_every == 0):
            print(f"bootstrap {b + 1}/{n_bootstrap}")

    print(f"There were {failures} deconvolution failures during bootstrapping.")

    L = max(len(r) for r in boots)
    arr = np.array([np.pad(r, (0, L - len(r))) for r in boots], dtype=float)

    r_mean = arr.mean(axis=0)
    r_std = arr.std(axis=0, ddof=1)

    # point estimate: sum all blocks once, then invert
    # L_full = max(len(x) for x in block_r + block_a)
    # R_full = np.zeros(L_full, dtype=np.int64)
    # A_full = np.zeros(L_full, dtype=np.int64)
    # for r_d, a_d in zip(block_r, block_a):
    #     R_full[: len(r_d)] += r_d
    #     A_full[: len(a_d)] += a_d
    # r_full = get_measured_multiplicity_causal(R_full, A_full)

    rplusa_counts_full = sr_counts(detection_times, sr.predelay, sr.gate)
    a_counts_full = sr_counts_delayed(detection_times, sr.predelay, sr.gate, sr.delay)
    rplusa_dist_full = np.bincount(rplusa_counts_full)
    a_dist_full = np.bincount(a_counts_full, minlength=len(rplusa_dist))
    r_full = get_measured_multiplicity_causal(rplusa_dist_full, a_dist_full)

    return r_full, r_mean, r_std, arr


def _init_worker(segments, sr, segment_duration, seed):
    global _SEGMENTS, _SR, _SEG_DUR, _SEED
    _SEGMENTS = segments
    _SR = sr
    _SEG_DUR = segment_duration
    _SEED = seed


def _worker_bootstrap(b: int):
    rng = np.random.default_rng(_SEED + b)
    n_seg = len(_SEGMENTS)
    idx = rng.integers(0, n_seg, size=n_seg)

    parts = []
    offset = 0.0
    for j in idx:
        seg = _SEGMENTS[j]
        if seg.size:
            parts.append(seg + offset)
        offset += _SEG_DUR  # No sort needed since offsets are monotonic

    if not parts:
        return None

    times = np.concatenate(parts)  # Already sorted

    rplusa = sr_counts(times, _SR.predelay, _SR.gate)
    a = sr_counts_delayed(times, _SR.predelay, _SR.gate, _SR.delay)
    rplusa_dist = np.bincount(rplusa)
    a_dist = np.bincount(a, minlength=len(rplusa_dist))

    try:
        return get_measured_multiplicity_causal(rplusa_dist, a_dist)
    except Exception:
        return None


def analyze_with_bootstrap_parallel(
    collision_track_path: Path,
    sr: SRParams,
    n_bootstrap: int = 200,
    segment_duration: float = 10.0,
    seed: int | None = 12345,
    n_workers: int | None = None,
):
    """
    LIST-mode block bootstrap with parallel execution.
    """
    if n_workers is None:
        n_workers = mp.cpu_count()

    # Load detection times
    ct = openmc.read_collision_track_hdf5(str(collision_track_path))
    absorption_events = ct[ct["event_mt"] == 101]
    detection_times = np.sort(absorption_events["time"])

    t_min = float(detection_times.min())
    t_max = float(detection_times.max())
    total_time = t_max - t_min
    n_segments = int(ceil(total_time / segment_duration))

    events_per_segment = len(detection_times) / n_segments
    print(f"There are {events_per_segment:.2f} events per segment (should be >1000)")
    print(f"Total detection time {total_time:.2f} seconds")

    # Vectorized segmentation (much faster than loop)
    segment_edges = t_min + np.arange(n_segments + 1) * segment_duration
    segment_edges[-1] = t_max + 1e-12
    cuts = np.searchsorted(detection_times, segment_edges)
    segments = [
        detection_times[cuts[i] : cuts[i + 1]] - segment_edges[i]
        for i in range(n_segments)
    ]
    print(f"Using {n_segments} segments of {segment_duration:.2f} s")
    # segments = []
    # for i in range(n_segments):
    #     seg = detection_times[cuts[i] : cuts[i + 1]] - segment_edges[i]
    #     segments.append(seg)
    #     if (i + 1) % 500 == 0:
    #         print(f"Completed {i + 1}/{n_segments} segment constructions")

    # Parallel bootstrap
    print(f"Starting parallel bootstrap with {n_workers} workers...")
    boots = []
    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_init_worker,
        initargs=(segments, sr, segment_duration, seed),
    ) as ex:
        results = ex.map(_worker_bootstrap, range(n_bootstrap), chunksize=20)
        boots = [r for r in results if r is not None]

    print(f"Completed {len(boots)}/{n_bootstrap} bootstrap samples")

    # Compute statistics
    L = max(len(r) for r in boots)
    arr = np.array([np.pad(r, (0, L - len(r))) for r in boots], dtype=float)
    r_mean = arr.mean(axis=0)
    r_std = arr.std(axis=0, ddof=1)

    # Point estimate from full data
    rplusa_full = sr_counts(detection_times, sr.predelay, sr.gate)
    a_full = sr_counts_delayed(detection_times, sr.predelay, sr.gate, sr.delay)
    r_full = get_measured_multiplicity_causal(
        np.bincount(rplusa_full),
        np.bincount(a_full, minlength=len(np.bincount(rplusa_full))),
    )
    return r_full, r_mean, r_std, arr


if __name__ == "__main__":
    base_dir = Path(__file__).parent.resolve()
    input_dir = base_dir / "inputs"
    output_root = base_dir / "outputs"

    cfg = ReplicateConfig(
        n_replicates=1,
        particles_per_rep=100_000_000,
        base_seed=12346,
        gate=85e-6,
        predelay=4e-6,
        delay=1000e-6,
        rate=3e4,
    )

    col_track_file = output_root / "rep_0000" / "collision_track.h5"
    sr = SRParams(predelay=cfg.predelay, gate=cfg.gate, delay=cfg.delay)
    r_full, r_mean, r_std, arr = analyze_with_bootstrap_parallel(
        col_track_file,
        sr,
        n_bootstrap=200,
        segment_duration=10.0,
        seed=12346,
        n_workers=32,
    )
    print("\nidx |   r_full | boot_mean | boot_std")
    print("-" * 45)
    K = 6
    for k in range(min(K, len(r_mean), len(r_full))):
        print(f"{k:>3} | {r_full[k]:>8.5f} | {r_mean[k]:>9.5f} | {r_std[k]:>8.5f}")
