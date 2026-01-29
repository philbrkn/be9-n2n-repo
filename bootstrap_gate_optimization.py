from __future__ import annotations

from pathlib import Path

import numpy as np

from bootstrap import SRParams, analyze_with_bootstrap_parallel

if __name__ == "__main__":
    base_dir = Path(__file__).parent.resolve()
    output_root = base_dir / "outputs"
    col_track_file = output_root / "standard_rep_0000_1e8p" / "collision_track.h5"

    die_away = 70e-6  # measured value
    # gate_multipliers = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0])
    gate_multipliers = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 1.5])
    gate_widths = gate_multipliers * die_away

    results = []

    for g in gate_widths:
        print(f"\n--- Gate: {g * 1e6:.1f} μs ({g / die_away:.1f}τ) ---")

        sr = SRParams(predelay=4e-6, gate=g, delay=1000e-6)
        r_full, r_mean, r_std, arr = analyze_with_bootstrap_parallel(
            col_track_file,
            sr,
            n_bootstrap=200,
            segment_duration=10.0,
            seed=12346,
            n_workers=32,
        )
        # print("\nidx |   r_full | boot_mean | boot_std")
        # print("-" * 45)
        # K = 6
        # for k in range(min(K, len(r_mean), len(r_full))):
        #     print(f"{k:>3} | {r_full[k]:>8.5f} | {r_mean[k]:>9.5f} | {r_std[k]:>8.5f}")
        # Store what matters
        results.append(
            {
                "gate": g,
                "gate_over_tau": g / die_away,
                "r_full": r_full[:4].copy(),
                "r_std": r_std[:4].copy(),
                "rel_unc_r1": r_std[1] / abs(r_full[1])
                if abs(r_full[1]) > 1e-10
                else np.inf,
                "rel_unc_r2": r_std[2] / abs(r_full[2])
                if abs(r_full[2]) > 1e-10
                else np.inf,
            }
        )

    # Save
    np.save(output_root / "gate_optimization_results.npy", results)

    # Print summary
    print("\n" + "=" * 70)
    print("Gate Optimization Results")
    print("=" * 70)
    print(
        f"{'Gate (μs)':<12} {'Gate/τ':<10} {'r_1':<12} {'σ(r_1)':<12} {'Rel Unc %':<12}"
    )
    print("-" * 58)
    for r in results:
        print(
            f"{r['gate'] * 1e6:<12.1f} {r['gate_over_tau']:<10.1f} {r['r_full'][1]:<12.5f} "
            f"{r['r_std'][1]:<12.5f} {r['rel_unc_r1'] * 100:<12.2f}"
        )

    # Find optimum
    best = min(results, key=lambda x: x["rel_unc_r1"])
    print(f"\nOptimal gate: {best['gate'] * 1e6:.1f} μs ({best['gate_over_tau']:.1f}τ)")
