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


if __name__ == "__main__":
    SOURCE_RATE = 3e4  # maximum value from srinivasan paper
    GATE = 54e-6
    PREDELAY = 4e-6
    DELAY = 1000e-6
    N_source = 1000000

    collision_tracks = openmc.read_collision_track_hdf5("outputs/collision_track.h5")
    absorption_events = collision_tracks[collision_tracks["event_mt"] == 101]
    detection_times = np.sort(absorption_events["time"])
    print(f"Total detections: {len(detection_times)}")
    efficiency = len(absorption_events) / N_source
    print(f"Detection efficiency: {efficiency:.4f}")

    # 1. Check what fields are available
    # first_collision = collision_tracks[0]
    # print(f"Fields: {first_collision.dtype.names}")
    # # Extract the MT numbers from every collision in the list
    # all_mts = [event["event_mt"] for event in collision_tracks]
    # unique_mts, counts = np.unique(all_mts, return_counts=True)
    # print("Unique Reaction MTs found:")
    # for mt, count in zip(unique_mts, counts):
    #     print(f"  MT {mt}: {count} occurrences")

    # shift register analysis
    rplusa_counts = sr_counts(detection_times, PREDELAY, GATE)
    a_counts = sr_counts_delayed(detection_times, PREDELAY, GATE, DELAY)
    rplusa_dist = np.bincount(rplusa_counts)
    a_dist = np.bincount(a_counts, minlength=len(rplusa_dist))
    # print("R+A Coincidence distribution:")
    # for n, count in enumerate(rplusa_dist):
    #     print(f"  {n} coincidences: {count}")
    # print("\nA (Accidental) distribution:")
    # for n, count in enumerate(a_dist):
    #     print(f"  {n} coincidences: {count}")

    # Measured SR real distribution from pulse train
    r_measured = get_measured_multiplicity_causal(rplusa_dist, a_dist)

    print(f"\n{'idx':>3} | {'r_measured':>12}")
    print("-" * 19)

    for i, rm in enumerate(r_measured):
        print(f"{i:>3} | {rm:>12.3e}")

    # Test different gate widths
    gate_values = [
        20e-6,
        32e-6,
        40e-6,
        54e-6,
        240e-6,
        360e-6,
    ]  # fast, weighted avg, slow

    for gate in gate_values:
        rplusa_counts = sr_counts(detection_times, PREDELAY, gate)
        a_counts = sr_counts_delayed(detection_times, PREDELAY, gate, DELAY)

        rplusa_dist = np.bincount(rplusa_counts)
        a_dist = np.bincount(a_counts, minlength=len(rplusa_dist))

        r_measured = get_measured_multiplicity_causal(rplusa_dist, a_dist)

        print(f"\nGate = {gate * 1e6:.0f} us:")
        print(f"  r[0] = {r_measured[0]:.4f}")
        print(f"  r[1] = {r_measured[1]:.4f}")
        print(f"  r[2] = {r_measured[2]:.6f}")
