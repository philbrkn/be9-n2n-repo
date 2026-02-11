from math import comb, exp

import numpy as np
import openmc

from utils.analyze_sr import (
    get_measured_multiplicity_causal,
    sr_histograms_twoptr,
)


def leakage_count_per_source_track(track, be_cell_id):
    """
    Count neutrons that ever leave the Be cell in this source history.
    definition: for each neutron track, if it has any state with cell_id != be_cell_id
    after having been in be_cell_id, then count it as a leaker.

    this will slightly overcount if neutron is born outside of Be
    future: make sure in history that first cell was in Be
    """
    leakers = 0
    for ptype, states in track.particle_tracks:
        if getattr(ptype, "name", str(ptype)) != "NEUTRON":
            continue

        cell_ids = states["cell_id"]
        # has it ever been inside Be?
        was_in_be = np.any(cell_ids == be_cell_id)
        if not was_in_be:
            continue
        # did it ever go outside after being inside?
        if np.any(cell_ids != be_cell_id):
            leakers += 1
    return leakers


def Ps_truth_from_tracks(tracks, be_cell_id, vmax=None):
    """
    iterate through the tracks to build Ps distribution
    """
    # construct a list of how many leaked
    v_list = [leakage_count_per_source_track(tr, be_cell_id) for tr in tracks]
    vmax = max(v_list) if vmax is None else vmax
    # do a pivot table of sorts
    counts = np.bincount(v_list, minlength=vmax + 1)
    Ps = counts / counts.sum()
    return Ps, np.asarray(v_list)


def emission_times_poisson(n, source_rate, rng=None):
    """
    to make synthetic pulse train to put independent openmc histories
    on a global clock

    we assume the temporal distribution of source neutron prod. is
    Poissonian in nature, which should be experimentally established first.
    source rate should not exceed 3e4 neutrons per second
    """
    # rng = np.random.default_rng() if rng is None else rng
    # draws i.i.d. inter-arrival times Δt with mean 1/λ.
    dt = np.random.exponential(1.0 / source_rate, size=n)
    return np.cumsum(dt)


def detector_event_times_from_tracks(tracks, det_cell_id):
    """
    Return global detection times (sorted) using definition:
    count one pulse for each neutron whose final state is in detector cell.
    """
    det_times = []
    for i, tr in enumerate(tracks):
        # t_emit = source_times[i]
        for ptype, states in tr.particle_tracks:
            if getattr(ptype, "name", str(ptype)) != "NEUTRON":
                continue

            final = states[-1]
            if final["cell_id"] == det_cell_id:
                # det_times.append(t_emit + final["time"])
                det_times.append(final["time"])
    det_times = np.sort(np.asarray(det_times))
    return det_times


def get_detection_matrix_D(vmax, epsilon):
    """
    Dkn = (n k) epsilon^k (1-epsilon)^(n-k)  k <= n else 0
    rows n=0...vmax, cols v=0...vmax
    """
    D = np.zeros((vmax + 1, vmax + 1))
    for v in range(vmax + 1):
        for n in range(v + 1):
            D[n, v] = comb(v, n) * (epsilon**n) * ((1 - epsilon) ** (v - n))

    return D


def get_sr_response_matrix_Lambda(vmax, F):
    """
    Lambdakn
    """
    Lambda = np.zeros((vmax + 1, vmax + 1))
    for n in range(vmax + 1):
        for k in range(n):  # k < n
            inner_sum = 0
            for j in range(n - k):
                inner_sum += comb(k + j, j) * (1 - F) ** j
            Lambda[k, n] = F**k / n * inner_sum

    return Lambda


def r_pred_from_Ps(Ps, epsilon, predelay, gate_width, tau):
    """
    compute r_pred from truth Ps using sri paper
    """
    Ps = np.asarray(Ps, dtype=float)
    vmax = len(Ps) - 1
    D = get_detection_matrix_D(vmax, epsilon)
    F = exp(-predelay / tau) * (1 - exp(-gate_width / tau))
    Lambda = get_sr_response_matrix_Lambda(vmax, F)
    # for weighing matrix we want a diagonal matirx with increasing values
    W = np.diag(np.arange(0, vmax + 1))

    r = Lambda @ W @ D @ Ps
    # need to normalize columns of r to sum to 1
    r /= r.sum()
    return r


if __name__ == "__main__":
    tracks = openmc.Tracks("outputs/tracks.h5")
    SOURCE_RATE = 3e4  # maximum value from srinivasan paper
    GATE = 28e-6
    PREDELAY = 4e-6
    DELAY = 1000e-6

    DETECTOR_EFFICIENCY = 0.3344
    TAU = 68e-6

    ### DEBUG ###
    for i, tr in enumerate(tracks[:5]):
        for ptype, states in tr.particle_tracks:
            if getattr(ptype, "name", str(ptype)) != "NEUTRON":
                continue
            print(
                f"Track {i}: birth time = {states[0]['time']:.6e}, final time = {states[-1]['time']:.6e}"
            )
    ## DEBUG END ##

    n_times = len(tracks)
    np.random.seed(42)

    geom = openmc.Geometry.from_xml()
    be_cell = [c for c in geom.get_all_cells().values() if c.name == "beryllium"][0]
    be_cell_id = be_cell.id
    he3_cell = [c for c in geom.get_all_cells().values() if c.name == "He3_detector"][0]
    he3_cell_id = he3_cell.id

    # TRUTH Ps leaking out of beryllium
    Ps_truth, vlist = Ps_truth_from_tracks(tracks, be_cell_id)

    # synthetic pulse train (dont need)
    # pulse_train = emission_times_poisson(n_times, SOURCE_RATE)
    detection_times = detector_event_times_from_tracks(tracks, he3_cell_id)

    # shift register analysis
    rplusa_dist, a_dist = sr_histograms_twoptr(
        detection_times, PREDELAY, GATE, DELAY, cap=64
    )
    # Measured SR real distribution from pulse train
    r_measured = get_measured_multiplicity_causal(rplusa_dist, a_dist)

    # Predicted r from Ps_truth
    # need detector efficiency and tau
    r_predicted = r_pred_from_Ps(Ps_truth, DETECTOR_EFFICIENCY, PREDELAY, GATE, TAU)
    print("Ps_truth:")
    for i, v in enumerate(Ps_truth):
        print(f"  [{i}] {v:.6e}")

    print(f"\n{'idx':>3} | {'r_measured':>15} | {'r_predicted':>15}")
    print("-" * 42)

    for i, (rm, rp) in enumerate(zip(r_measured, r_predicted)):
        print(f"{i:>3} | {rm:>15.3e} | {rp:>15.3e}")
