# analyze_sr.py
# Pure analysis utilities

import numpy as np


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

