from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional

import numpy as np
import openmc

from run_replicates import ReplicateConfig, run_independent_replicates

SetupHook = Callable[[Any, Path, Path], None]
# signature: setup(value, input_dir, scale_dir) -> None


def run_sweep(
    *,
    input_dir: Path,
    output_root: Path,
    base_cfg: ReplicateConfig,
    values: Iterable[Any],
    label_fmt: Optional[str] = None,
    setup_hook: Optional[SetupHook] = None,
    param_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Generic 1D sweep runner:
      - creates outputs/<label>/rep_XXXX/
      - replaces base_cfg.<param_name> with each value
      - optional setup hook per point (cross sections, env vars, etc.)
      - collects detections + r stats into a uniform results dict
    """
    input_dir = Path(input_dir)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    results: Dict[str, Any] = {
        "param_name": param_name,
        "values": [],
        "detections_mean": [],
        "detections_std": [],
        "r_mean": [],
        "r_std": [],
        "r_sem": [],
        "all_r": [],
        "all_det": [],
    }

    for v in values:
        openmc.reset_auto_ids()
        start_time = time.perf_counter()

        label = label_fmt.format(v) if label_fmt else f"{param_name}_{v}"
        scale_dir = output_root / label
        scale_dir.mkdir(parents=True, exist_ok=True)

        if setup_hook is not None:
            setup_hook(v, scale_dir)

        cfg = replace(base_cfg, **{param_name: v}) if param_name else base_cfg

        r_mean, r_std, r_sem, all_r, all_det = run_independent_replicates(
            input_dir=input_dir,
            output_root=scale_dir,
            cfg=cfg,
        )

        results["values"].append(v)
        results["detections_mean"].append(float(np.mean(all_det)))
        results["detections_std"].append(float(np.std(all_det)))
        results["r_mean"].append(r_mean.copy())
        results["r_std"].append(r_std.copy())
        results["r_sem"].append(r_sem.copy())
        results["all_r"].append(all_r)
        results["all_det"].append(all_det)

        run_time = time.perf_counter() - start_time
        print(
            f"[{param_name}={v}] mean detections: {np.mean(all_det):.1f}  std: {np.std(all_det):.1f}"
            f". time: {run_time:.2f}s"
        )

    return results
