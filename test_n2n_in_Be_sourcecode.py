import pickle

import numpy as np
import openmc

# def get_n2n_secondary_energies(tracks, be_cell_id):
#     """
#     For each source history, detect if an n,2n reaction occurred in Be9
#     and collect the outgoing energies of the two product neutrons.
#
#     Strategy:
#       - Primary neutron is particle_tracks[0]
#       - Secondary neutrons spawned during the primary's transport appear
#         as subsequent NEUTRON entries in particle_tracks
#       - An n,2n event in Be means the primary was inside Be when it spawned
#         secondaries, and there are exactly 2 neutrons born from it
#       - The outgoing energy of each secondary = states["E"][0] (birth energy)
#
#     Returns:
#         List of (E1, E2) tuples for each identified n,2n event [eV]
#     """
#     n2n_pairs = []
#
#     for track in tracks:
#         # Separate primary from secondaries
#         # particle_tracks[0] is always the source neutron
#         ptype0, states0 = track.particle_tracks[0]
#         if getattr(ptype0, "name", str(ptype0)) != "NEUTRON":
#             continue
#
#         # Collect all secondary neutron tracks (index 1 onward)
#         secondary_neutrons = []
#         for ptype, states in track.particle_tracks[1:]:
#             if getattr(ptype, "name", str(ptype)) != "NEUTRON":
#                 continue
#             secondary_neutrons.append(states)
#
#         # n,2n produces exactly 2 neutrons; the primary is absorbed,
#         # so we expect 2 secondaries born in Be
#         if len(secondary_neutrons) != 2:
#             continue
#
#         # Confirm both secondaries were born inside Be
#         born_in_be = all(s["cell_id"][0] == be_cell_id for s in secondary_neutrons)
#         if not born_in_be:
#             continue
#
#         # Extract birth energies
#         E1 = secondary_neutrons[0]["E"][0]
#         E2 = secondary_neutrons[1]["E"][0]
#         n2n_pairs.append((E1, E2))
#
#     return n2n_pairs


def build_parent_map(track):
    """
    For each particle index j, find which particle index i spawned it
    by finding which state of i has position exactly matching j's birth pos.
    Returns dict: {child_index: parent_index}
    """
    parent_map = {}
    for j, (src_j, (ptype_j, states_j)) in enumerate(
        zip(track.sources, track.particle_tracks)
    ):
        if j == 0:
            continue
        birth_pos_j = np.array(src_j.r)
        for i, (src_i, (ptype_i, states_i)) in enumerate(
            zip(track.sources, track.particle_tracks)
        ):
            if i >= j:
                continue
            for k in range(len(states_i)):
                pos = np.array(
                    [
                        states_i["r"][k]["x"],
                        states_i["r"][k]["y"],
                        states_i["r"][k]["z"],
                    ]
                )
                if np.linalg.norm(birth_pos_j - pos) == 0.0:
                    parent_map[j] = i
                    break
            if j in parent_map:
                break
    return parent_map


def get_n2n_secondary_energies_v2(tracks_list, be_cell_id):
    n2n_pairs = []

    for track in tracks_list:
        ptype0, states0 = track.particle_tracks[0]
        if getattr(ptype0, "name", str(ptype0)) != "NEUTRON":
            continue

        parent_map = build_parent_map(track)

        # find all neutrons whose direct parent is particle 0
        children_of_primary = [
            j
            for j, parent in parent_map.items()
            if parent == 0
            and getattr(
                track.particle_tracks[j][0], "name", str(track.particle_tracks[j][0])
            )
            == "NEUTRON"
        ]

        if len(children_of_primary) != 2:
            continue

        # both must be born in Be
        if not all(
            track.particle_tracks[j][1]["cell_id"][0] == be_cell_id
            for j in children_of_primary
        ):
            continue

        E_incident = states0["E"][-2] if len(states0) >= 2 else states0["E"][0]
        E1 = track.particle_tracks[children_of_primary[0]][1]["E"][0]
        E2 = track.particle_tracks[children_of_primary[1]][1]["E"][0]
        n2n_pairs.append((E_incident, E1, E2))

    return n2n_pairs


def _pos(state_r):
    """Convert structured array position record to plain float array."""
    return np.array([state_r["x"], state_r["y"], state_r["z"]])


def get_n2n_secondary_energies(tracks_list, be_cell_id):
    """
    In OpenMC, (n,2n) with integral yield=2 modifies the primary in-place
    and creates 1 secondary clone at the same position with the same energy.

    Strategy: for each secondary neutron born in Be, find the primary state
    at the same position. The state BEFORE that point is E_in; the state
    AT that point (after collision) is the outgoing energy. The secondary
    has the same outgoing energy (it's a clone).
    """
    import numpy as np
    import openmc

    results = []

    for track in tracks_list:
        ptracks = track.particle_tracks
        sources = track.sources

        if len(ptracks) < 2:
            continue

        primary_type, primary_states = ptracks[0]
        if primary_type != openmc.ParticleType.NEUTRON:
            continue

        for i in range(1, len(ptracks)):
            p_type, p_states = ptracks[i]
            if p_type != openmc.ParticleType.NEUTRON:
                continue

            first = p_states[0]
            if first["cell_id"] != be_cell_id:
                continue

            birth_r = first["r"]
            E_sec = first["E"]

            # Walk along primary states. Find the state whose position
            # matches the secondary's birth position. The energy at
            # that state is AFTER the collision (= outgoing neutron 1).
            # The energy at the PREVIOUS state is E_in.
            for j in range(1, len(primary_states)):
                r_p = primary_states[j]["r"]
                d = np.sqrt(np.sum((r_p - birth_r) ** 2))
                if d < 1e-6:
                    E_in = primary_states[j - 1]["E"]
                    E_primary_out = primary_states[j]["E"]
                    # Confirm it looks like (n,2n): secondary energy
                    # should equal the primary's post-collision energy
                    # (they're clones)
                    if abs(E_primary_out - E_sec) / max(E_sec, 1.0) < 1e-6:
                        results.append([E_in, E_primary_out, E_sec])
                    break

    return results


def inspect_tracks(tracks_list, be_cell_id, max_tracks=50):
    """Dump structure of tracks to understand how (n,2n) appears."""
    import openmc

    for t_idx, track in enumerate(tracks_list[:max_tracks]):
        ptracks = track.particle_tracks
        sources = track.sources

        # Only look at tracks with multiple neutron secondaries
        neutron_indices = []
        for i, (p_type, p_states) in enumerate(ptracks):
            if p_type == openmc.ParticleType.NEUTRON:
                neutron_indices.append(i)

        if len(neutron_indices) < 3:  # primary + at least 2 secondaries
            continue

        print(
            f"\n=== Track {t_idx} | {len(ptracks)} sub-tracks | "
            f"{len(neutron_indices)} neutrons ==="
        )

        for i, (p_type, p_states) in enumerate(ptracks):
            src = sources[i]
            first = p_states[0]
            last = p_states[-1]
            print(
                f"  [{i}] {p_type!s:>10} | "
                f"nstates={len(p_states):3d} | "
                f"birth_cell={first['cell_id']:4d} | "
                f"birth_mat={first['material_id']:4d} | "
                f"E_birth={first['E']:12.1f} eV | "
                f"E_last={last['E']:12.1f} eV | "
                f"birth_r=({first['r'][0]:.4f}, {first['r'][1]:.4f}, {first['r'][2]:.4f}) | "
                f"src_E={src.E:12.1f} eV"
            )

        # Flag which sub-tracks are in the Be cell
        print(f"  Be cell id = {be_cell_id}")
        for i, (p_type, p_states) in enumerate(ptracks):
            cells_visited = set(p_states["cell_id"])
            if be_cell_id in cells_visited:
                print(f"    sub-track [{i}] visits Be cell")

    print("\n--- Done inspecting ---")


def diagnose_n2n(tracks_list, be_cell_id):
    """
    (n,2n) in OpenMC: primary is modified in-place, 1 secondary is cloned.
    Match each secondary's birth position to a primary track state.
    The primary state BEFORE = E_in, the state AT that point = E_out_primary,
    and the secondary's birth energy = E_out_secondary.
    """
    import numpy as np
    import openmc

    def dist(r1, r2):
        return np.sqrt(
            (float(r1["x"]) - float(r2["x"])) ** 2
            + (float(r1["y"]) - float(r2["y"])) ** 2
            + (float(r1["z"]) - float(r2["z"])) ** 2
        )

    events = []

    for t_idx, track in enumerate(tracks_list):
        ptracks = track.particle_tracks
        primary_type, primary_states = ptracks[0]
        if primary_type != openmc.ParticleType.NEUTRON:
            continue

        for i in range(1, len(ptracks)):
            p_type, p_states = ptracks[i]
            if p_type != openmc.ParticleType.NEUTRON:
                continue
            if p_states[0]["cell_id"] != be_cell_id:
                continue

            birth_r = p_states[0]["r"]
            E_sec = p_states[0]["E"]

            # Search primary track for matching position
            for j in range(1, len(primary_states)):
                d = dist(primary_states[j]["r"], birth_r)
                if d < 1e-4:  # relaxed tolerance
                    E_in = primary_states[j - 1]["E"]
                    E_primary_post = primary_states[j]["E"]
                    clone = abs(E_primary_post - E_sec) / max(E_sec, 1.0) < 1e-4
                    events.append(
                        {
                            "track": t_idx,
                            "sub": i,
                            "E_in": E_in,
                            "E_primary_post": E_primary_post,
                            "E_sec": E_sec,
                            "clone": clone,
                            "dist": d,
                        }
                    )
                    break

    # Print
    matched = [e for e in events if True]
    print(f"Secondaries in Be: searched all, matched {len(matched)} to primary track\n")
    print(
        f"{'track':>6} {'sub':>4} | {'E_in MeV':>12} | {'E_pri_post MeV':>14} | "
        f"{'E_sec MeV':>12} | {'clone?':>6} | {'dist':>10}"
    )
    print("-" * 85)
    for ev in matched[:40]:
        print(
            f"{ev['track']:6d} {ev['sub']:4d} | "
            f"{ev['E_in'] / 1e6:12.4f} | "
            f"{ev['E_primary_post'] / 1e6:14.4f} | "
            f"{ev['E_sec'] / 1e6:12.4f} | "
            f"{'YES' if ev['clone'] else 'NO':>6} | "
            f"{ev['dist']:10.2e}"
        )

    # Summary
    clones = sum(1 for e in matched if e["clone"])
    print(
        f"\nTotal matched: {len(matched)}, clones: {clones}, different: {len(matched) - clones}"
    )

    # Also print unmatched secondaries for debugging
    return events


if __name__ == "__main__":
    # tracks = openmc.Tracks("outputs/reps_with_tracksh5/rep_0001/tracks.h5")
    # run once to cache
    # with open("tracks_cache.pkl", "wb") as f:
    #     pickle.dump(list(tracks), f)
    # subsequent runs
    with open("tracks_cache.pkl", "rb") as f:
        tracks_list = pickle.load(f)

    geom = openmc.Geometry.from_xml(
        path="outputs/reps_with_tracksh5/rep_0001/geometry.xml",
        materials="outputs/reps_with_tracksh5/rep_0001/materials.xml",
    )
    be_cell = [c for c in geom.get_all_cells().values() if c.name == "beryllium"][0]
    be_cell_id = be_cell.id
    diagnose_n2n(tracks_list, be_cell_id=be_cell_id)
    # n2n_pairs = get_n2n_secondary_energies(tracks_list, be_cell_id=be_cell_id)
    #
    # n2n_pairs = np.array(n2n_pairs)
    # E_in = n2n_pairs[:, 0] / 1e6
    # E1 = n2n_pairs[:, 1] / 1e6
    # E2 = n2n_pairs[:, 2] / 1e6
    #
    # print(f"n,2n events identified: {len(n2n_pairs)}")
    # print(
    #     f"\n{'E_in (MeV)':>12}  {'E1 (MeV)':>12}  {'E2 (MeV)':>12}  {'E1+E2 (MeV)':>12}"
    # )
    # print("-" * 56)
    # for ein, e1, e2 in zip(E_in[:20], E1[:20], E2[:20]):
    #     print(f"{ein:>12.4f}  {e1:>12.4f}  {e2:>12.4f}  {e1 + e2:>12.4f}")
    #
    # print(f"\nE1 mean: {E1.mean():.3f} MeV,  E2 mean: {E2.mean():.3f} MeV")
