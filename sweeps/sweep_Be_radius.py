from run_replicates import ReplicateConfig
from sweeps.sweep_n2n_scale import setup_scaled_xs
from utils.config import REPO_ROOT

# from sweeps.sweep_ddx_scale import setup_scaled_ddx
from utils.sweep import run_sweep

if __name__ == "__main__":
    input_dir = REPO_ROOT / "inputs"
    # results = run_sweep(
    #     input_dir=input_dir,
    #     output_root=output_root,
    #     base_cfg=base_cfg,
    #     param_name="be_radius",
    #     values=radii,
    #     label_fmt="be_radius_{:.1f}cm",
    # )

    # 9.0 is nominal
    radii = [7.0, 8.0, 10.0, 11.0]
    # radii = [3.0, 4.0, 5.0, 6.0]

    PARTICLES_PER_REP = 100_000_000
    for r in radii:
        # output_root = base_dir / "outputs" / f"be_rad_ddx_sweep_{r}"
        output_root = REPO_ROOT / "outputs" / f"fixed_outer_be_rad_n2n_sweep_{r}"
        cfg = ReplicateConfig(
            n_replicates=1,
            particles_per_rep=PARTICLES_PER_REP,
            gate=28e-6,
            predelay=4e-6,
            delay=1000e-6,
            rate=3e4,
            base_seed=123456,
            max_collisions=0.5 * PARTICLES_PER_REP,
            be_radius=r,
        )

        scales = [0.8, 0.9, 1.0, 1.1, 1.2]
        # scales = [-0.2, -0.1, 0.0, 0.1, 0.2]

        results = run_sweep(
            input_dir=input_dir,
            output_root=output_root,
            base_cfg=cfg,
            values=scales,
            # label_fmt="ddx_scale_{:.2f}",
            label_fmt="n2n_scale_{:.2f}",
            setup_hook=setup_scaled_xs,
            param_name=None,
        )
