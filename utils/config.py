"""YAML config loading for run_replicates and paper_scripts.

Precedence for path fields: value in config.yaml > corresponding env var.
Raises if neither is set at the moment the path is actually needed.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "config.yaml"


@dataclass
class Paths:
    cross_sections_xml: Optional[str]
    be9_h5: Optional[str]


def _read_yaml(path: Path) -> Dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def load_raw(path: Optional[Path] = None) -> Dict[str, Any]:
    """Return the parsed YAML dict without constructing a ReplicateConfig."""
    return _read_yaml(Path(path) if path else DEFAULT_CONFIG_PATH)


def load_config(path: Optional[Path] = None):
    """Load config.yaml and return (ReplicateConfig, Paths, output_subdir).

    Imported lazily to avoid a circular import with run_replicates.
    """
    from run_replicates import ReplicateConfig

    data = load_raw(path)
    sim = data.get("simulation", {})
    geom = data.get("geometry", {})
    mat = data.get("material", {})
    sr = data.get("sr_analysis", {})
    paths = data.get("paths", {})

    cfg = ReplicateConfig(
        n_replicates=int(sim.get("n_replicates", 10)),
        particles_per_rep=int(sim.get("particles_per_rep", 100_000)),
        base_seed=sim.get("base_seed"),
        max_collisions=sim.get("max_collisions"),
        be_radius=float(geom.get("be_radius", 9.0)),
        n_tubes=int(geom.get("n_tubes", 20)),
        source_z=float(geom.get("source_z", 0.0)),
        be_density=float(mat.get("be_density", 1.85)),
        he_density=float(mat.get("he_density", 0.0005)),
        gate=float(sr.get("gate", 85e-6)),
        predelay=float(sr.get("predelay", 4e-6)),
        delay=float(sr.get("delay", 1000e-6)),
        rate=float(sr.get("rate", 3e4)),
    )
    p = Paths(
        cross_sections_xml=paths.get("cross_sections_xml"),
        be9_h5=paths.get("be9_h5"),
    )
    output_subdir = str(sim.get("output_subdir", "default"))
    return cfg, p, output_subdir


def get_cross_sections_xml(cfg_path: Optional[Path] = None) -> str:
    """Resolve the ENDF cross_sections.xml path: config.yaml first, then env."""
    raw = load_raw(cfg_path)
    v = raw.get("paths", {}).get("cross_sections_xml")
    if v:
        return v
    env = os.environ.get("OPENMC_CROSS_SECTIONS")
    if env:
        return env
    raise RuntimeError(
        "cross_sections_xml is unset in config.yaml and $OPENMC_CROSS_SECTIONS "
        "is empty. Set one of them before running."
    )


def get_be9_h5(cfg_path: Optional[Path] = None) -> str:
    """Resolve the path to a Be9.h5 data file (used by DDx/n2n perturbation scripts)."""
    raw = load_raw(cfg_path)
    v = raw.get("paths", {}).get("be9_h5")
    if v:
        return v
    env = os.environ.get("ENDF_BE9_H5")
    if env:
        return env
    raise RuntimeError(
        "be9_h5 is unset in config.yaml and $ENDF_BE9_H5 is empty. "
        "Set one of them before running Be9 perturbation scripts."
    )
