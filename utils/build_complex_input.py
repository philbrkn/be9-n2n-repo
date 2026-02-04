#!/usr/bin/env python3
# build_inputs.py
# Writes the base OpenMC XML inputs once into ./inputs

from pathlib import Path
from typing import Any, Dict

import numpy as np
import openmc

# os.environ["OPENMC_CROSS_SECTIONS"] = os.path.abspath(
#     "Documents/endf-b8.0-hdf5/endfb-viii.0-hdf5/cross_sections.xml"
# )


def create_geometry(input_dir: Path, cfg) -> Dict[str, Any]:
    # === MATERIALS === #
    be = openmc.Material(name="beryllium")
    be.add_nuclide("Be9", 1.0)  # 1,0 is composition value (100% berylium-9)
    be.set_density("g/cm3", 1.85)  # physical density of solid beryllium at room temp
    # testing graphite instead of be:
    # be.add_element("C", 1.0)  # 1,0 is composition value (100% berylium-9)
    # be.set_density("g/cm3", 2.25)  # physical density of solid beryllium at room temp
    # be.add_s_alpha_beta("c_Graphite")

    hdpe = openmc.Material(name="polyethylene")
    hdpe.add_element("C", 1.0)
    hdpe.add_element("H", 2.0)
    hdpe.set_density("g/cm3", 0.95)
    hdpe.add_s_alpha_beta("c_H_in_CH2")  # thermal scattering

    he3 = openmc.Material(name="He3")
    he3.add_nuclide("He3", 1.0)
    he3.set_density("g/cm3", float(cfg.he_density))

    cd = openmc.Material(name="Cd")
    cd.add_element("Cd", 1.0)
    cd.set_density("g/cm3", 8.65)

    air = openmc.Material(name="air")
    air.add_element("N", 0.78)
    air.add_element("O", 0.21)
    air.add_element("Ar", 0.01)
    air.set_density("g/cm3", 0.001225)

    materials = openmc.Materials([be, hdpe, he3, cd, air])  # collection object
    materials.export_to_xml(path=str(input_dir / "materials.xml"))

    # === GEOMETRY PARAMETERS === #
    # Beryllium assembly
    BE_RADIUS = float(cfg.be_radius)  # cm
    BE_HALF_HEIGHT = 32.5  # cm
    # central beam hole for DT target:
    BEAM_HOLE_RADIUS = 1.0  # cm

    # he3 detectors
    N_TUBES = int(cfg.n_tubes)
    HE3_RADIUS = 1.5
    HE3_RADIAL_POS = 15  # distance from origin to each cylinder

    # outer boundary
    OUTER_RADIUS = 20.0  # cm total
    OUTER_HALF_HEIGHT = 40.0  # cm

    CD_THICKNESS = 0.1  # 1mm lining

    # Derived dimensions
    CD_RADIUS = BE_RADIUS + CD_THICKNESS

    # === SURFACES === #

    # Central beam hole
    beam_hole = openmc.ZCylinder(r=BEAM_HOLE_RADIUS)

    # Be surfaces
    be_cyl = openmc.ZCylinder(r=BE_RADIUS)
    be_top = openmc.ZPlane(z0=BE_HALF_HEIGHT)
    be_bot = openmc.ZPlane(z0=-BE_HALF_HEIGHT)

    # Cd surface
    cd_cyl = openmc.ZCylinder(r=CD_RADIUS)

    # outer boundary surface
    outer_cyl = openmc.ZCylinder(r=OUTER_RADIUS, boundary_type="vacuum")
    outer_top = openmc.ZPlane(z0=OUTER_HALF_HEIGHT, boundary_type="vacuum")
    outer_bot = openmc.ZPlane(z0=-OUTER_HALF_HEIGHT, boundary_type="vacuum")

    # === REGIONS === #

    # be region: cylinder with hole, between be_bot and be_top
    be_region = -be_cyl & +beam_hole & +be_bot & -be_top  # Be with hole

    # Cd lining: shell around Be (radial)
    cd_region = +be_cyl & -cd_cyl & -outer_top & +outer_bot

    # Outer boundary region
    outer_region = -outer_cyl & +outer_bot & -outer_top

    # Air cavity
    air_region = (
        -be_cyl
        & +beam_hole
        & +outer_bot
        & -outer_top  # cylinder from beam hole to Cd
        & ~(-be_cyl & +be_bot & -be_top)  # exclude Be volume
    )
    # === CELLS == #

    cells = []

    # Beam hole (void)
    beam_cell = openmc.Cell(name="beam_hole")
    beam_cell.region = -beam_hole & +outer_bot & -outer_top
    # beam_cell.fill = None  # void
    cells.append(beam_cell)

    # Air cavity
    air_cell = openmc.Cell(name="air_cavity", fill=air, region=air_region)
    cells.append(air_cell)

    # Beryllium (with central hole)
    be_cell = openmc.Cell(name="beryllium")
    be_cell.fill = be
    be_cell.region = be_region
    cells.append(be_cell)

    # Cadmium lining (wraps Be)
    cd_cell = openmc.Cell(name="cadmium_lining")
    cd_cell.fill = cd
    cd_cell.region = cd_region
    cells.append(cd_cell)

    # He3 detector tubes
    HE3_BASE_ID = 100  # He-3 tubes will be 100..100+N_TUBES-1
    he3_cells = []
    he3_cyls = []
    he3_cell_ids = []

    for i in range(N_TUBES):
        theta = 2.0 * np.pi * i / N_TUBES
        x = HE3_RADIAL_POS * np.cos(theta)
        y = HE3_RADIAL_POS * np.sin(theta)

        cyl = openmc.ZCylinder(x0=x, y0=y, r=HE3_RADIUS)
        he3_cyls.append(cyl)

        he3_region = (-cyl) & (+outer_bot) & (-outer_top)

        cid = HE3_BASE_ID + i
        he3_cell_ids.append(cid)
        cell = openmc.Cell(
            cell_id=cid, name=f"he3_tube_{i:02d}", fill=he3, region=he3_region
        )
        he3_cells.append(cell)
        cells.append(cell)

    # HDPE moderator (fills everything)
    hdpe_region = outer_region & +cd_cyl
    for cyl in he3_cyls:
        hdpe_region &= +cyl

    hdpe_cell = openmc.Cell(fill=hdpe, region=hdpe_region, name="hdpe_cell")
    cells.append(hdpe_cell)

    # === CREATE GEOMETRY === #
    universe = openmc.Universe(cells=cells)
    geometry = openmc.Geometry(universe)
    geometry.export_to_xml(path=str(input_dir / "geometry.xml"))

    return {
        "he3_cell_ids": he3_cell_ids,
        "materials": {
            "be": be,
            "hdpe": hdpe,
            "he3": he3,
            "cd": cd,
            "air": air,
        },
        "cells": {
            "be": be_cell,
            "he3": he3_cells,
        },
    }


def create_noBe_geometry(input_dir: Path) -> None:
    # === MATERIALS === #

    hdpe = openmc.Material(name="polyethylene")
    hdpe.add_element("C", 1.0)
    hdpe.add_element("H", 2.0)
    hdpe.set_density("g/cm3", 0.95)
    hdpe.add_s_alpha_beta("c_H_in_CH2")  # thermal scattering

    he3 = openmc.Material(name="He3")
    he3.add_nuclide("He3", 1.0)
    he3.set_density("g/cm3", 0.0005)  # this is a rough value

    cd = openmc.Material(name="Cd")
    cd.add_element("Cd", 1.0)
    cd.set_density("g/cm3", 8.65)

    air = openmc.Material(name="air")
    air.add_element("N", 0.78)
    air.add_element("O", 0.21)
    air.add_element("Ar", 0.01)
    air.set_density("g/cm3", 0.001225)

    materials = openmc.Materials([hdpe, he3, cd, air])  # collection object
    materials.export_to_xml(path=str(input_dir / "materials.xml"))

    # === GEOMETRY PARAMETERS === #
    # Beryllium assembly
    BE_RADIUS = 9.0  # cm

    # central beam hole for DT target:
    BEAM_HOLE_RADIUS = 1.0  # cm

    # he3 detectors
    N_TUBES = 20
    HE3_RADIUS = 1.5
    HE3_RADIAL_POS = 15  # distance from origin to each cylinder

    # outer boundary
    OUTER_RADIUS = 20.0  # cm total
    OUTER_HALF_HEIGHT = 40.0  # cm

    CD_THICKNESS = 0.1  # 1mm lining

    # Derived dimensions
    CD_RADIUS = BE_RADIUS + CD_THICKNESS

    # === SURFACES === #

    # Central beam hole
    beam_hole = openmc.ZCylinder(r=BEAM_HOLE_RADIUS)

    # Be surfaces
    air_cyl = openmc.ZCylinder(r=BE_RADIUS)

    # Cd surface
    cd_cyl = openmc.ZCylinder(r=CD_RADIUS)

    # outer boundary surface
    outer_cyl = openmc.ZCylinder(r=OUTER_RADIUS, boundary_type="vacuum")
    outer_top = openmc.ZPlane(z0=OUTER_HALF_HEIGHT, boundary_type="vacuum")
    outer_bot = openmc.ZPlane(z0=-OUTER_HALF_HEIGHT, boundary_type="vacuum")

    # === REGIONS === #

    # Cd lining: shell around Be (radial)
    cd_region = +air_cyl & -cd_cyl & -outer_top & +outer_bot

    # Outer boundary region
    outer_region = -outer_cyl & +outer_bot & -outer_top

    # Air cavity
    air_region = (
        -air_cyl & +beam_hole & +outer_bot & -outer_top  # cylinder from beam hole to Cd
    )
    # === CELLS == #

    cells = []

    # Beam hole (void)
    beam_cell = openmc.Cell(name="beam_hole")
    beam_cell.region = -beam_hole & +outer_bot & -outer_top
    # beam_cell.fill = None  # void
    cells.append(beam_cell)

    # Air cavity
    air_cell = openmc.Cell(name="air_cavity", fill=air, region=air_region)
    cells.append(air_cell)

    # Cadmium lining (wraps Be)
    cd_cell = openmc.Cell(name="cadmium_lining")
    cd_cell.fill = cd
    cd_cell.region = cd_region
    cells.append(cd_cell)

    # He3 detector tubes
    HE3_BASE_ID = 100  # He-3 tubes will be 100..100+N_TUBES-1
    he3_cells = []
    he3_cyls = []
    he3_cell_ids = []

    for i in range(N_TUBES):
        theta = 2.0 * np.pi * i / N_TUBES
        x = HE3_RADIAL_POS * np.cos(theta)
        y = HE3_RADIAL_POS * np.sin(theta)

        cyl = openmc.ZCylinder(x0=x, y0=y, r=HE3_RADIUS)
        he3_cyls.append(cyl)

        he3_region = (-cyl) & (+outer_bot) & (-outer_top)

        cid = HE3_BASE_ID + i
        he3_cell_ids.append(cid)
        cell = openmc.Cell(
            cell_id=cid, name=f"he3_tube_{i:02d}", fill=he3, region=he3_region
        )
        he3_cells.append(cell)
        cells.append(cell)

    # HDPE moderator (fills everything)
    hdpe_region = outer_region & +cd_cyl
    for cyl in he3_cyls:
        hdpe_region &= +cyl

    hdpe_cell = openmc.Cell(fill=hdpe, region=hdpe_region)
    cells.append(hdpe_cell)

    # === CREATE GEOMETRY === #
    universe = openmc.Universe(cells=cells)
    geometry = openmc.Geometry(universe)
    geometry.export_to_xml(path=str(input_dir / "geometry.xml"))

    return {
        "he3_cell_ids": he3_cell_ids,
        "materials": {
            "hdpe": hdpe,
            "he3": he3,
            "cd": cd,
            "air": air,
        },
        "cells": {
            "he3": he3_cells,
        },
    }


def build_complex_inputs(input_dir: Path) -> None:
    """
    Build and export materials.xml, geometry.xml, settings.xml (template),
    tallies.xml into input_dir.
    """

    input_dir = Path(input_dir)
    input_dir.mkdir(parents=True, exist_ok=True)

    openmc.reset_auto_ids()

    geo = create_geometry(input_dir)

    # =============================================================================
    # Settings
    # =============================================================================
    settings = openmc.Settings()
    settings.batches = 1
    settings.particles = 100_000  # template default; replicate runner will override
    settings.run_mode = "fixed source"

    # IMPORTANT: collision_track is set in the replicate runner (needs He3 cell id)
    # IMPORTANT: source is set in the replicate runner (depends on RATE, T)

    # settings.track = [(1, 1, i) for i in range(1, settings.particles)]  # (batch, gen, part)

    # settings.seed = 12346
    settings.output = {"path": ".", "tallies": False}
    settings.export_to_xml(path=str(input_dir / "settings.xml"))

    # =============================================================================
    # Plots: Create a plot object
    # =============================================================================
    plots_list = []

    # Radial cross-section at z=0
    p_xy = openmc.Plot()
    p_xy.filename = "geom_xy"
    p_xy.basis = "xy"
    p_xy.origin = (0, 0, 0)  # z=0 slice
    p_xy.width = (60.0, 60.0)
    p_xy.pixels = (1000, 1000)
    p_xy.color_by = "material"
    p_xy.colors = {
        geo["materials"]["be"]: "lightgrey",
        geo["materials"]["hdpe"]: "lightblue",
        geo["materials"]["he3"]: "orange",
        geo["materials"]["cd"]: "red",
        geo["materials"]["air"]: "lightyellow",
    }
    plots_list.append(p_xy)

    # Longitudinal bisection through axis (xz plane at y=0)
    p_xz = openmc.Plot()
    p_xz.filename = "geom_xz"
    p_xz.basis = "xz"
    p_xz.origin = (0, 0, 0)  # y=0 slice
    p_xz.width = (60.0, 100.0)  # taller to see full height; tune as needed
    p_xz.pixels = (1000, 1600)
    p_xz.color_by = "material"
    p_xz.colors = p_xy.colors
    plots_list.append(p_xz)

    plots = openmc.Plots(plots_list)
    plots.export_to_xml(path=str(input_dir / "plots.xml"))
    openmc.plot_geometry(path_input=str(input_dir))


if __name__ == "__main__":
    base_dir = Path(__file__).parent.resolve()
    build_complex_inputs(base_dir / "inputs")
