# paper_scripts

Frozen one-off scripts for paper figures and substudies. **Not** the active
workflow — the live sweeps moved to `sweeps/` at the repo root. See the main
README.

## How to run

From the repo root:

```bash
python -m paper_scripts.<script_name>
```

The `-m` invocation puts the repo root on `sys.path` so `from run_replicates
import ...` and `from utils.X import ...` resolve.

## What's here

| File / dir | What it does |
|-----------|--------------|
| `sweep_Be_density.py`, `sweep_He_density.py`, `sweep_source_rates.py` | One-off sensitivity sweeps not currently active |
| `bootstrap_validate.py`, `bootstrap_gate_optimization.py` | Variance-by-bootstrap diagnostics |
| `plot_sweep.py`, `plot_bootstrap_gate_optim.py`, `plot_bootstrap_validation.py`, `_plot_*.py` | Paper-figure makers |
| `analyze_replicates_tracks.py`, `analyze_multigroup_tracks.py` | Per-replicate event-by-event diagnostics |
| `feynman_y_analysis.py`, `predicted_multiplicity.py`, `measure_dieaway.py`, `get_fisher_matrix.py`, `fhigheff_espectrum_libraries.py`, `test_n2n_in_Be_sourcecode.py` | One-off analyses |
| `perkins/` | Substudy on outgoing-angular and secondary-energy perturbations |
| `tendl/` | TENDL vs ENDF library comparison |
| `data/` | Snapshots of nuclear data, cached arrays, paper figures |

## Caveats

These scripts were single-use during paper writing. Some `__main__` blocks
have hardcoded sweep values, output subdirectories, or commented-out
alternatives. Read the file before re-running. The `perkins/` and `tendl/`
subdirectories still contain absolute paths like `/home/philip/...` — if you
re-run them and something breaks, fix the path locally or replace with
`utils.config.get_be9_h5()` / `get_cross_sections_xml()`.
