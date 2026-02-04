# analyze_sr.py
# Pure analysis utilities

from dataclasses import dataclass

import numba
import numpy as np


@dataclass(frozen=True)
class SRParams:
    predelay: float
    gate: float
    delay: float


@numba.njit
def sr_histograms_twoptr(times, predelay, gate, delay, cap):
    n = times.size
    rplusa_hist = np.zeros(cap + 1, dtype=np.int64)
    a_hist = np.zeros(cap + 1, dtype=np.int64)

    # R+A window pointers
    L1 = 0
    R1 = 0

    # A (delayed) window pointers
    L2 = 0
    R2 = 0

    for i in range(n):
        t = times[i]

        # R+A window: (t + predelay, t + predelay + gate]
        target_L1 = t + predelay
        target_R1 = t + predelay + gate

        # Advance L1 until times[L1] > target_L1
        while L1 < n and times[L1] <= target_L1:
            L1 += 1
        # Advance R1 until times[R1] > target_R1
        while R1 < n and times[R1] <= target_R1:
            R1 += 1

        c1 = R1 - L1
        if c1 > cap:
            c1 = cap
        rplusa_hist[c1] += 1

        # A window: (t + delay + predelay, t + delay + predelay + gate]
        target_L2 = t + delay + predelay
        target_R2 = t + delay + predelay + gate

        while L2 < n and times[L2] <= target_L2:
            L2 += 1
        while R2 < n and times[R2] <= target_R2:
            R2 += 1

        c2 = R2 - L2
        if c2 > cap:
            c2 = cap
        a_hist[c2] += 1

    return rplusa_hist, a_hist


@numba.njit
def _search_right(a, x):
    """
    returns first index where times[idx]>x
    """
    lo, hi = 0, a.size
    while lo < hi:
        mid = (lo + hi) >> 1
        if a[mid] <= x:
            lo = mid + 1
        else:
            hi = mid
    return lo


@numba.njit
def sr_histograms(times, predelay, gate, delay, cap):
    # cap is the maximum count represented; overflow goes into cap
    n = times.size
    rplusa_hist = np.zeros(cap + 1, dtype=np.int64)
    a_hist = np.zeros(cap + 1, dtype=np.int64)

    for i in range(n):
        t = times[i]

        # R+A
        left = _search_right(times, t + predelay)
        right = _search_right(times, t + predelay + gate)
        c1 = right - left
        if c1 > cap:
            c1 = cap
        rplusa_hist[c1] += 1

        # A (delayed)
        left = _search_right(times, t + delay + predelay)
        right = _search_right(times, t + delay + predelay + gate)
        c2 = right - left
        if c2 > cap:
            c2 = cap
        a_hist[c2] += 1

    return rplusa_hist, a_hist


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
