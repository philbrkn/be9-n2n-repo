from __future__ import annotations

import multiprocessing as mp
import os
import threading
import time
from concurrent.futures import ProcessPoolExecutor
from math import ceil
from multiprocessing import shared_memory
from pathlib import Path

import psutil

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
import openmc

from utils.analyze_sr import (
    SRParams,
    get_measured_multiplicity_causal,
    # sr_counts,
    # sr_counts_delayed,
    sr_histograms,
    sr_histograms_twoptr,
)

# Worker globals
_SHM = None
_OFFSETS = None  # (start_idx, end_idx) for each segment
_SR = None
_SEG_DUR = None
_SEED = None
_DTYPE = None
_SHAPE = None


def warmup_numba():
    dummy = np.array([0.0, 1.0, 2.0], dtype=np.float64)
    sr_histograms_twoptr(dummy, 0.1, 0.1, 0.5, 64)


_proc = psutil.Process(os.getpid())


def fmt_bytes(n):
    for u in ["B", "KiB", "MiB", "GiB", "TiB"]:
        if n < 1024:
            return f"{n:,.1f} {u}"
        n /= 1024
    return f"{n:,.1f} PiB"


def log_mem(tag=""):
    mi = _proc.memory_info()
    rss = mi.rss
    vms = mi.vms
    # children can be empty early on
    children = _proc.children(recursive=True)
    child_rss = 0
    child_max = 0
    for c in children:
        try:
            cr = c.memory_info().rss
            child_rss += cr
            child_max = max(child_max, cr)
        except psutil.Error:
            pass
    print(
        f"[MEM] {tag:>18} | parent RSS {fmt_bytes(rss)} VMS {fmt_bytes(vms)} "
        f"| children {len(children)} RSS(sum) {fmt_bytes(child_rss)} max {fmt_bytes(child_max)}",
        flush=True,
    )


def start_mem_sampler(interval=2.0):
    stop = threading.Event()

    def _run():
        while not stop.is_set():
            log_mem("periodic")
            time.sleep(interval)

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return stop


def _init_worker_shm(shm_name, shape, dtype, offsets, sr, seg_dur, seed):
    global _SHM, _OFFSETS, _SR, _SEG_DUR, _SEED, _DTYPE, _SHAPE
    _SHM = shared_memory.SharedMemory(name=shm_name)
    _SHAPE = shape
    _DTYPE = dtype
    _OFFSETS = offsets
    _SR = sr
    _SEG_DUR = seg_dur
    _SEED = seed
    warmup_numba()  # Add this line


def _worker_bootstrap_shm(b: int):
    all_times = np.ndarray(_SHAPE, dtype=_DTYPE, buffer=_SHM.buf)

    rng = np.random.default_rng(_SEED + b)
    n_seg = _OFFSETS.shape[0]  # make _OFFSETS a (n_seg,2) int64 array
    idx = rng.integers(0, n_seg, size=n_seg)

    # pass 1: total length
    total = 0
    for j in idx:
        s, e = _OFFSETS[j]
        total += e - s
    if total == 0:
        return None

    # allocate once
    times = np.empty(total, dtype=all_times.dtype)

    # pass 2: fill
    pos = 0
    offset = 0.0
    for j in idx:
        s, e = _OFFSETS[j]
        m = e - s
        if m:
            times[pos : pos + m] = all_times[s:e] + offset
            pos += m
        offset += _SEG_DUR

    rplusa_dist, a_dist = sr_histograms_twoptr(
        times, _SR.predelay, _SR.gate, _SR.delay, cap=64
    )

    # return get_measured_multiplicity_causal(rplusa_dist, a_dist)
    r = get_measured_multiplicity_causal(rplusa_dist, a_dist)
    return r, int(times.size)


def _wlog(tag):
    """
    usage:
    if b == 0:
        _wlog("after bincount a")
    """
    p = psutil.Process(os.getpid())
    rss = p.memory_info().rss / 1024**2
    print(f"[WORKER {os.getpid()}] {tag:<20} RSS={rss:,.1f} MiB", flush=True)


def analyze_with_bootstrap_parallel(
    collision_track_path: Path,
    sr: SRParams,
    n_bootstrap: int = 200,
    segment_duration: float = 10.0,
    seed: int | None = 12345,
    n_workers: int | None = None,
    chunk_size: int | None = None,
):
    """
    LIST-mode block bootstrap with parallel execution.
    """
    start_time = time.perf_counter()
    if n_workers is None:
        n_workers = mp.cpu_count()

    # Load detection times
    ct = openmc.read_collision_track_hdf5(str(collision_track_path))

    absorption_events = ct[ct["event_mt"] == 101]
    detection_times = np.sort(absorption_events["time"])
    n_det_full = int(detection_times.size)

    # Point estimate from full data
    # rplusa_full = sr_counts(detection_times, sr.predelay, sr.gate)
    # a_full = sr_counts_delayed(detection_times, sr.predelay, sr.gate, sr.delay)
    rplusa_dist_full, a_dist_full = sr_histograms(
        detection_times, sr.predelay, sr.gate, sr.delay, cap=64
    )
    r_full = get_measured_multiplicity_causal(rplusa_dist_full, a_dist_full)

    del ct, absorption_events  # Free memory

    t_min = float(detection_times.min())
    t_max = float(detection_times.max())
    total_time = t_max - t_min
    n_segments = int(ceil(total_time / segment_duration))

    print(f"Events per segment: {len(detection_times) / n_segments:.2f}")
    print(f"Using {n_segments} segments of {segment_duration:.2f} s")

    # Vectorized segmentation (much faster than loop)
    segment_edges = t_min + np.arange(n_segments + 1) * segment_duration
    segment_edges[-1] = t_max + 1e-12
    cuts = np.searchsorted(detection_times, segment_edges)

    # Normalize times within each segment and flatten back
    normalized = np.empty_like(detection_times)
    for i in range(n_segments):
        s, e = cuts[i], cuts[i + 1]
        normalized[s:e] = detection_times[s:e] - segment_edges[i]

    offsets = np.column_stack([cuts[:-1], cuts[1:]]).astype(np.int64)

    # Create shared memory
    shm = shared_memory.SharedMemory(create=True, size=normalized.nbytes)
    shm_array = np.ndarray(normalized.shape, dtype=normalized.dtype, buffer=shm.buf)
    shm_array[:] = normalized

    del detection_times, normalized  # Free original arrays

    # Parallel bootstrap
    if chunk_size is None:
        chunk_size = max(1, n_bootstrap // n_workers)
    print(
        f"Starting parallel bootstrap with {n_workers} workers and chunk size {chunk_size}..."
    )

    # TIMING
    print(f"bootstrapping pre loop took {time.perf_counter() - start_time:.3f} seconds")
    start_time = time.perf_counter()

    # START PARALLELIZER
    try:
        with ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=_init_worker_shm,
            initargs=(
                shm.name,
                shm_array.shape,
                shm_array.dtype,
                offsets,
                sr,
                segment_duration,
                seed,
            ),
        ) as ex:
            results = ex.map(
                _worker_bootstrap_shm, range(n_bootstrap), chunksize=chunk_size
            )
            # boots = [r for r in results if r is not None]
            results = [r for r in results if r is not None]
            boots = [r for (r, _) in results]
            dets = np.array([d for (_, d) in results], dtype=float)

    finally:
        shm.close()
        shm.unlink()

    # TIMING
    print(f"bootstrapping loop took {time.perf_counter() - start_time:.3f} seconds")

    print(f"Completed {len(boots)}/{n_bootstrap} bootstrap samples")

    # Compute statistics
    L = max(len(r) for r in boots)
    arr = np.array([np.pad(r, (0, L - len(r))) for r in boots], dtype=float)
    r_mean = arr.mean(axis=0)
    r_std = arr.std(axis=0, ddof=1)
    det_mean = float(dets.mean())
    det_sem = float(dets.std(ddof=1))  # bootstrap SE
    return r_full, r_mean, r_std, arr, det_mean, det_sem, dets


if __name__ == "__main__":
    base_dir = Path(__file__).parent.resolve()
    input_dir = base_dir / "inputs"
    output_root = base_dir / "outputs"

    GATE = 28e-6
    PREDELAY = 4e-6
    DELAY = 1000e-6

    start_time = time.perf_counter()
    # col_track_file = output_root / "rep_0000" / "collision_track.h5"
    # col_track_file = output_root / "standard_rep_0000_1e8p" / "collision_track.h5"
    col_track_file = output_root / "tendl" / "rep_0000" / "collision_track.h5"

    sr = SRParams(predelay=PREDELAY, gate=GATE, delay=DELAY)
    r_full, r_mean, r_std, arr, det_mean, det_sem, _ = analyze_with_bootstrap_parallel(
        col_track_file,
        sr,
        n_bootstrap=200,
        segment_duration=1.0,
        seed=12345,
        n_workers=32,
        chunk_size=1,
    )
    print(f"det mean {det_mean} det sem {det_sem}")
    print("\nidx |   r_full | boot_mean | boot_std")
    print("-" * 45)
    K = 6
    for k in range(min(K, len(r_mean), len(r_full))):
        print(f"{k:>3} | {r_full[k]:>8.5f} | {r_mean[k]:>9.5f} | {r_std[k]:>8.5f}")

    print(f"Bootstrapping took {time.perf_counter() - start_time}")
