# n2n

OpenMC-based simulation of an (n,2n) multiplicity assay. A 14.1 MeV DT neutron
source irradiates a beryllium target surrounded by a polyethylene-moderated
He-3 detector array; collision tracks are post-processed with shift-register
logic to recover the singles/doubles/triples distribution.

The core question the repo is set up to answer: **how sensitive is the measured
multiplicity to the Be9 (n,2n) cross section and its outgoing energy
distribution (DDx)?** That's done by running sweeps that scale the underlying
nuclear data and re-running the whole detector simulation at each point.

## Quick start (tl;dr)

```bash
# 1. Create the env
conda env create -f environment.yml
conda activate n2n-env

# 2. Get ENDF/B-VIII.0 cross sections (see "Nuclear data setup" below)
# 3. Point the code at them:
cp config.yaml my_config.yaml      # optional
# edit my_config.yaml paths.cross_sections_xml and paths.be9_h5
# OR just export the env vars — see below

# 4. Run a single replicate (sanity check)
python run_replicates.py --config config.yaml

# 5. Run the main (n,2n) scale sweep
python -m sweeps.sweep_n2n_scale

# 6. Post-process the sweep into a figure
python -m sweeps.plot_sweep_bootstrap
```

## Install

```bash
conda env create -f environment.yml
conda activate n2n-env
```

OpenMC is only reliably installable via conda-forge; `requirements.txt` exists
as a secondary option for people who already have OpenMC built.

## Nuclear data setup (required)

OpenMC does not ship cross-section data. You need an ENDF/B-VIII.0 HDF5
library, unpacked somewhere on disk, and the code needs to know two paths:

- `cross_sections.xml` — the library index OpenMC reads to resolve nuclide names
- `Be9.h5` — the individual Be9 file; the DDx/(n,2n) sweeps open this directly
  and write perturbed copies into each sweep-point directory

### Download

From <https://openmc.org/data/> (or the anl.gov mirror), grab:

```
endfb-viii.0-hdf5.tar.xz   # ~1.5 GB compressed, ~3 GB extracted
```

Extract wherever you want:

```bash
cd ~/nuclear_data
wget https://anl.box.com/shared/static/uhbxlrx7hvxqw27psymfbhi7bx7s6u6a.xz \
     -O endfb-viii.0-hdf5.tar.xz
tar -xf endfb-viii.0-hdf5.tar.xz
```

Should now have:

```
~/nuclear_data/endfb-viii.0-hdf5/
  cross_sections.xml
  neutron/Be9.h5
  neutron/He3.h5
  ...
```

### Point the code at the library

**Option A — environment variables** (recommended; leaves `config.yaml`
portable between machines):

```bash
export OPENMC_CROSS_SECTIONS=$HOME/nuclear_data/endfb-viii.0-hdf5/cross_sections.xml
export ENDF_BE9_H5=$HOME/nuclear_data/endfb-viii.0-hdf5/neutron/Be9.h5
```

Put these in your shell rc file (`~/.bashrc`, `~/.zshrc`) so they're always
set.

**Option B — hardcode in `config.yaml`**:

```yaml
paths:
  cross_sections_xml: /home/yourname/nuclear_data/endfb-viii.0-hdf5/cross_sections.xml
  be9_h5: /home/yourname/nuclear_data/endfb-viii.0-hdf5/neutron/Be9.h5
```

Values in `config.yaml` take precedence over the env vars. If neither is set
when a script actually needs the path, will throw an error.

### Alternative libraries

If you want to compare against TENDL-2019 or ENDF/B-VII.1, the same pattern
works — just point the two settings at a different library. The existing
TENDL-vs-ENDF comparison lives under `paper_scripts/tendl/` (frozen, will need
path touch-ups).

## The main workflow: sweep → post-process

Most things of use for the paper are in the `sweeps/` directory. Run from the repo root
as a module so imports resolve:

```bash
python -m sweeps.sweep_n2n_scale    # scale Be9 (n,2n) cross section by {0.8 .. 1.2}
python -m sweeps.sweep_ddx_scale    # perturb Be9 outgoing energy distribution
python -m sweeps.sweep_Be_radius    # outer loop: one of the above, per Be radius
```

Each sweep point produces `outputs/<label>/rep_XXXX/collision_track.h5`.

Post-processing:

```bash
python -m sweeps.plot_sweep_bootstrap   # figures for (n,2n) and Be-radius sweeps
python -m sweeps.postproc_ddx           # figures for DDx sweeps, plus pdf-panels
```

Edit the `main()` block in each to change which sweep directory to read
(`OUTPUT_ROOT`), which pattern to glob for (`PATTERN`), and where to write the
figure.

## Config file

`config.yaml` at the repo root controls `run_replicates.py` directly. The
sweep scripts build their own `ReplicateConfig` instances in code — edit the
`__main__` block of `sweeps/sweep_n2n_scale.py` etc. to change particle counts,
seeds, or scale ranges for a sweep.

Known bug: `material.be_density` is wired into `ReplicateConfig` but
currently ignored by geometry construction (beryllium density is hardcoded at
1.85 g/cm3 inside `utils/build_complex_input.py`). Don't treat
`material.be_density` as functional until that's wired up.

## Repo layout

| Path                | Purpose                                                                                      |
| ------------------- | -------------------------------------------------------------------------------------------- |
| `run_replicates.py` | Library + CLI: runs N OpenMC replicates, reads collision tracks, deconvolves to multiplicity |
| `bootstrap.py`      | Parallel bootstrap resampling of detection times on a single track file                      |
| `config.yaml`       | Parameters for `run_replicates.py`                                                           |
| `sweeps/`           | **Active workflow.** (n,2n), DDx, and Be-radius sweeps + post-processing                     |
| `utils/`            | Reusable library: geometry builder, SR analysis (numba), config loader, generic sweep driver |
| `inputs/`           | Static OpenMC XML templates (starting point for each run)                                    |
| `outputs/`          | Per-run output directories (gitignored)                                                      |
| `figures/`          | Plot outputs (gitignored)                                                                    |
| `paper_scripts/`    | Frozen one-off scripts for paper figures. See `paper_scripts/README.md`                      |
