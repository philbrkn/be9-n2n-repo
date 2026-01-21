import time
from pathlib import Path

import numpy as np
import openmc


def get_measured_multiplicity_causal(rplusa_dist, a_dist):
    """
    Given pulse triggred distribution p (R+A) and delayed distribution q (A),
    compute real SR multiplicity distrbiution using:
    from Krick and Swansen 84:
        [r_k] = [U_kn]^-1 [p_k]
        U_kn = q_(k-n) if k>=n, else =0
    """
    rplusa_dist = np.asarray(rplusa_dist, dtype=float)
    a_dist = np.asarray(a_dist, dtype=float)

    p = rplusa_dist / rplusa_dist.sum()
    q = a_dist / a_dist.sum()

    N = len(p)
    r = np.zeros(N, float)
    if q[0] == 0:
        raise ValueError("q[0] must be nonzero for causal deconvolution.")
    for k in range(N):
        s = 0.0
        # sum_{i=1..k} q[i] * r[k-i]
        for i in range(1, k + 1):
            s += q[i] * r[k - i]
        r[k] = (p[k] - s) / q[0]
    return r


def sr_counts(times, predelay, gate):
    """Pulse-triggered SR: for each pulse at t, count pulses in (t+predelay, t+predelay+gate]."""
    t = np.asarray(times)
    left = np.searchsorted(t, t + predelay, side="right")
    right = np.searchsorted(t, t + predelay + gate, side="right")
    return (right - left).astype(np.int64)


def sr_counts_delayed(times, predelay, gate, delay):
    """Delayed trigger (accidentals): same, but window is shifted by delay."""
    t = np.asarray(times)
    left = np.searchsorted(t, t + delay + predelay, side="right")
    right = np.searchsorted(t, t + delay + predelay + gate, side="right")
    return (right - left).astype(np.int64)


def run_independent_replicates(
    n_replicates=20,
    particles_per_rep=100000,
    base_seed=None,
    gate=54e-6,
    predelay=4e-6,
    delay=1000e-6,
    input_path=None,
    output_path=None,
):
    """
    Run multiple independent simulations and collect SR results.
    """
    all_r = []
    all_detections = []
    if base_seed is None:
        base_seed = int(time.time() * 1000) % 2**31

    geom = openmc.Geometry.from_xml(
        path=f"{input_path}/geometry.xml", materials=f"{input_path}/materials.xml"
    )
    he3_cell = [c for c in geom.get_all_cells().values() if c.name == "He3_detector"][0]
    he3_cell_id = he3_cell.id

    RATE = 3e4  # neutrons/second

    for i in range(n_replicates):
        # print(f"\n=== Replicate {i + 1}/{n_replicates} ===")

        # Update settings with new seed
        settings = openmc.Settings.from_xml(path=f"{input_path}/settings.xml")
        settings.seed = base_seed + i * 1000  # different seed each time
        settings.particles = particles_per_rep
        settings.batches = 1
        settings.collision_track = {
            "cell_ids": [he3_cell_id],
            # "reactions": [103],
            "max_collisions": int(0.5 * particles_per_rep),
            "max_collision_track_files": 100,
        }
        T = particles_per_rep / RATE
        settings.source = openmc.IndependentSource(
            space=openmc.stats.Point((0, 0, 0)),
            energy=openmc.stats.Normal(mean_value=14.1e6, std_dev=5.0e4),
            angle=openmc.stats.Isotropic(),
            time=openmc.stats.Uniform(0.0, T),
        )
        settings.export_to_xml(path=f"{input_path}/settings.xml")

        # print(f"  Replicate {i}: seed={settings.seed}")
        # Run OpenMC
        openmc.run(output=False, cwd=output_path, path_input=input_path)

        # Read results
        col_track_file = f"{output_path}/collision_track.h5"
        collision_tracks = openmc.read_collision_track_hdf5(col_track_file)
        absorption_events = collision_tracks[collision_tracks["event_mt"] == 101]
        detection_times = np.sort(absorption_events["time"])

        all_detections.append(len(detection_times))

        # SR analysis
        rplusa_counts = sr_counts(detection_times, predelay, gate)
        a_counts = sr_counts_delayed(detection_times, predelay, gate, delay)

        rplusa_dist = np.bincount(rplusa_counts)
        a_dist = np.bincount(a_counts, minlength=len(rplusa_dist))

        try:
            r = get_measured_multiplicity_causal(rplusa_dist, a_dist)
            all_r.append(r)
            # print(f"  Detections: {len(detection_times)}, r[0]={r[0]:.4f}, r[1]={r[1]:.4f}")
        except Exception as e:
            print(f"  Failed: {e}")
            continue

    # Compute statistics
    max_len = max(len(r) for r in all_r)
    r_padded = np.array([np.pad(r, (0, max_len - len(r))) for r in all_r])

    r_mean = r_padded.mean(axis=0)
    r_std = r_padded.std(axis=0)
    r_sem = r_std / np.sqrt(len(all_r))

    return r_mean, r_std, r_sem, all_r, all_detections


if __name__ == "__main__":
    base_dir = Path(__file__).parent.resolve()
    input_dir = base_dir / "inputs"
    output_dir = base_dir / "outputs"
    # SR parameters
    GATE = 32e-6
    PREDELAY = 4e-6
    DELAY = 1000e-6

    r_mean, r_std, r_sem, all_r, all_det = run_independent_replicates(
        n_replicates=10,
        particles_per_rep=100000,
        gate=GATE,
        predelay=PREDELAY,
        delay=DELAY,
        input_path=input_dir,
        output_path=output_dir,
        base_seed=12345,
    )

    print("\n" + "=" * 60)
    print("FINAL RESULTS (20 independent replicates)")
    print("=" * 60)
    print(f"Mean detections per run: {np.mean(all_det):.0f} +/- {np.std(all_det):.0f}")
    print(f"\n{'idx':>3} | {'mean':>12} | {'std':>12} | {'SEM':>12}")
    print("-" * 50)
    for i in range(min(5, len(r_mean))):
        print(f"{i:>3} | {r_mean[i]:>12.6f} | {r_std[i]:>12.6f} | {r_sem[i]:>12.6f}")
