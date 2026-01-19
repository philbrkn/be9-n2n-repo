import numpy as np
import openmc

tracks = openmc.Tracks("tracks.h5")

source_rate = 3e4  # maximum value from srinivasan paper
GATE = 42e-6
PREDELAY = 4e-6
DELAY = 1000e-6
np.random.seed(42)

# inspect one track
track = tracks[1]
print(track.identifier)
print(len(track.particle_tracks))
# particle tracks stores secondary particles if they exist (like n,2n)
for ptype, states in track.particle_tracks:
    print(f"Particle: {ptype}, {len(states)} states")
    print(f"Fields available: {states.dtype.names}")
    print(states[0])  # first state (birth)
    print(states[1])  # last state (termination/escape)
    print(states[2])  # last state (termination/escape)
    print(states[-1])  # last state (termination/escape)
# fields: r (position), u (direction), E (energy), time, wgt, cell_id, material_id

# FIND OUT WHICH TYPES OF PARTICLES:
# seen = set()
# for track in tracks:
#     for ptype, _ in track.particle_tracks:
#         # normalize ptype to a clean Python string
#         if isinstance(ptype, (bytes, np.bytes_)):
#             p = ptype.decode().strip()
#         else:
#             p = str(ptype).strip()
#         seen.add(p)
# print("Particle types seen:", seen)

# # len() counts how many neutrons (primary+secondaries) exist in each track
# progeny_counts = [len(t.particle_tracks) for t in tracks]
# mult_dist = np.bincount(progeny_counts)
# print(mult_dist)
# M_L_from_tracks = np.mean(progeny_counts)
# print(f"M_L from tracking: {M_L_from_tracks:.4f}")

# need the escape times for SR
# for track in tracks[:5]:  # look at first 5
#     print(f"\nSource particle {track.identifier}:")
#     for i, (ptype, states) in enumerate(track.particle_tracks):
#         final = states[-1]
#         print(
#             f"  Neutron {i}: E={final['E'] / 1e6:.2f} MeV, t={final['time'] * 1e9:.2f} ns"
#         )

# === in a shift register experiment: === #
# 1. neutron gen emits 14 MeV neutrons at random times (poisson process, 3e4 n/s)
# 2. each source neutron creats burst of 1-6 escaping neutrons (mult dist)
# 3. these hit detectors and create pulse train
# 4. SR finds coincidences within time gate

# assign emission times to each source particle
n_sources = len(tracks)
time_between_neutrons_dist = np.random.exponential(1.0 / source_rate, n_sources)
# cumulative sum gives absolute emission times
# if inter arrival = [25, 40, 30 microsec]
# then source times = [25, 65, 95 microsec]
source_times = np.cumsum(time_between_neutrons_dist)

geom = openmc.Geometry.from_xml()
he3_cell = [c for c in geom.get_all_cells().values() if c.name == "He3_detector"][0]
he3_cell_id = he3_cell.id
# print(he3_cell_id)


# Look for ANY state in He-3, not just final state
# for track in tracks[:20]:  # first 20 tracks
#     for ptype, states in track.particle_tracks:
#         if ptype.name == "NEUTRON":
#             # Check all states in He-3
#             for j, state in enumerate(states):
#                 # print(type(state["cell_id"]), type(he3_cell_id))
#                 if state["cell_id"] == he3_cell_id:
#                     is_last = j == len(states) - 1
#                     # print(
#                     #     f"In He3: state {j}/{len(states)}, E={state['E']:.2f} eV, wgt={state['wgt']}, last={is_last}"
#                     # )

# collect all escape times on a global timeline
detection_times = []
# track_files = glob.glob("track_*.h5")
# print(f"Processing {len(track_files)} track files...")

for i, track in enumerate(tracks):
    t_source = source_times[i]  # corresponds to i track in experiment
    for ptype, states in track.particle_tracks:
        # has to be neutron, but i think all are neutrons anyway
        if ptype.name == "NEUTRON":
            final_state = states[-1]
            # print(final_state)
            if final_state["cell_id"] == he3_cell_id:
                detection_times.append(t_source + final_state["time"])
                # print(
                #     f"Detection: E={final_state['E']:.5f} eV, t={final_state['time'] * 1e6:.5f} μs, wgt={final_state['wgt']}"
                # )


# shift register analysis
def sr_multiplicity_counts(times, predelay, gate):
    """
    For each trigger at times[i], count events in (t+predelay, t+predelay+gate].
    Returns array m_i (multiplicity per trigger).
    """
    t = np.asarray(times)
    left = np.searchsorted(t, t + predelay, side="right")
    right = np.searchsorted(t, t + predelay + gate, side="right")
    return (right - left).astype(np.int64)


def sr_accidental_counts(times, predelay, gate, delay):
    t = np.asarray(times)
    left = np.searchsorted(t, t + delay + predelay, side="right")
    right = np.searchsorted(t, t + delay + predelay + gate, side="right")
    return (right - left).astype(np.int64)


# detection_times = np.sort(detection_times)
# print(f"total escaping neutrons: {len(detection_times)}")
#
# coinc_per_trigger = sr_multiplicity_counts(
#     detection_times, predelay=PREDELAY, gate=GATE
# )
# coinc_distribution = np.bincount(coinc_per_trigger)
# print("R+A Coincidence distribution:")
# for n, count in enumerate(coinc_distribution):
#     print(f"  {n} coincidences: {count}")
#
# accidental_per_n = sr_accidental_counts(
#     detection_times, predelay=PREDELAY, gate=GATE, delay=DELAY
# )
# accidental_distribution = np.bincount(
#     accidental_per_n, minlength=len(coinc_distribution)
# )
# print("\nA (Accidental) distribution:")
# for n, count in enumerate(accidental_distribution):
#     print(f"  {n} coincidences: {count}")
# print(
#     f"\napproximate Real coincidences (R): {coinc_per_trigger.sum() - accidental_per_n.sum()}"
# )
