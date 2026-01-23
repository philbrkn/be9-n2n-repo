#!/usr/bin/env python3
# build_inputs.py
# Writes the base OpenMC XML inputs once into ./inputs

from pathlib import Path

import numpy as np
import openmc

# os.environ["OPENMC_CROSS_SECTIONS"] = os.path.abspath(
#     "Documents/endf-b8.0-hdf5/endfb-viii.0-hdf5/cross_sections.xml"
# )


def build_inputs(input_dir: Path) -> None:
    """
    Build and export materials.xml, geometry.xml, settings.xml (template),
    tallies.xml into input_dir.
    """

    input_dir = Path(input_dir)
    input_dir.mkdir(parents=True, exist_ok=True)

    openmc.reset_auto_ids()

    # =============================================================================
    # Materials
    # =============================================================================
    be = openmc.Material(name="beryllium")
    be.add_nuclide("Be9", 1.0)  # 1,0 is composition value (100% berylium-9)
    be.set_density("g/cm3", 1.85)  # physical density of solid beryllium at room temp

    # polyethylene (ch2)
    hdpe = openmc.Material(name="polyethylene")
    hdpe.add_element("C", 1.0)
    hdpe.add_element("H", 2.0)
    hdpe.set_density("g/cm3", 0.95)
    hdpe.add_s_alpha_beta("c_H_in_CH2")  # thermal scattering

    # helium-3 detector gas (at WHAT PRESSURE)
    he3 = openmc.Material(name="He3")
    he3.add_nuclide("He3", 1.0)
    he3.set_density("g/cm3", 0.0005)  # this is a rough value
    # TODO confirm if this is sensitive to change

    # cadmium lining
    cd = openmc.Material(name="Cd")
    cd.add_element("Cd", 1.0)
    cd.set_density("g/cm3", 8.65)

    materials = openmc.Materials([be, hdpe, he3, cd])  # collection object
    materials.export_to_xml(path=str(input_dir / "materials.xml"))

    # =============================================================================
    # Geometry: Be cylinder (Basu et al. dimensions)
    # Be cylinder: 65 cm long, 18 cm diameter (9 cm radius)
    # HDPE moderator with embedded He3 tubes
    # outer boundary is a vacuum
    # =============================================================================

    # METHOD WITH VOID
    BE_RADIUS = 9.0  # cm
    BE_HALF_HEIGHT = 32.5  # cm
    POLY_THICKNESS = 15.0  # cm of moderator
    HE3_INNER_R = 12.0  # He-3 annulus inner radius
    HE3_OUTER_R = 13.0  # He-3 annulus outer radius (1 cm thick)
    OUTER_RADIUS = 25.0  # cm total
    OUTER_HALF_HEIGHT = 40.0  # cm
    # Cd lining
    CD_THICKNESS = 0.1  # 1mm lining
    CD_RADIUS = BE_RADIUS + CD_THICKNESS

    # Be surfaces
    be_radius = openmc.ZCylinder(r=BE_RADIUS)
    be_top = openmc.ZPlane(z0=BE_HALF_HEIGHT)
    be_bot = openmc.ZPlane(z0=-BE_HALF_HEIGHT)
    # Cadmium
    cd_radius = openmc.ZCylinder(r=CD_RADIUS)
    # He-3 annulus
    he3_inner = openmc.ZCylinder(r=HE3_INNER_R)
    he3_outer = openmc.ZCylinder(r=HE3_OUTER_R)
    # Outer boundary
    outer_cyl = openmc.ZCylinder(r=OUTER_RADIUS, boundary_type="vacuum")
    outer_top = openmc.ZPlane(z0=OUTER_HALF_HEIGHT, boundary_type="vacuum")
    outer_bot = openmc.ZPlane(z0=-OUTER_HALF_HEIGHT, boundary_type="vacuum")

    # --- Cells ---

    # 1. Define the Be Cell
    be_cell = openmc.Cell(name="beryllium")
    be_cell.fill = be
    be_cell.region = -be_radius & -be_top & +be_bot

    # 2. Cadmium lining cell
    cd_cell = openmc.Cell(name="cadmium_lining")
    cd_cell.fill = cd
    cd_cell.region = +be_radius & -cd_radius & -be_top & +be_bot

    # 2. Inner polyethylene (between Be and He-3)
    poly_inner_cell = openmc.Cell(name="poly_inner")
    poly_inner_cell.fill = hdpe
    poly_inner_cell.region = (
        -he3_inner  # Inside the detector ring
        & -outer_top
        & +outer_bot
        & ~(-cd_radius & -be_top & +be_bot)  # SUBTRACT the Be+Cd volume
    )

    # 3. He-3 detector annulus
    he3_cell = openmc.Cell(name="He3_detector")
    he3_cell.fill = he3
    he3_cell.region = (
        +he3_inner
        & -he3_outer  # annular region
        & -outer_top
        & +outer_bot  # within axial bounds
    )

    # 4. Outer polyethylene (outside He-3)
    poly_outer_cell = openmc.Cell(name="poly_outer")
    poly_outer_cell.fill = hdpe
    poly_outer_cell.region = (
        +he3_outer
        & -outer_cyl  # outside He-3, inside boundary
        & -outer_top
        & +outer_bot
    )

    universe = openmc.Universe(
        cells=[be_cell, cd_cell, poly_inner_cell, he3_cell, poly_outer_cell]
    )
    geometry = openmc.Geometry(universe)
    geometry.export_to_xml(path=str(input_dir / "geometry.xml"))

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
    p_xy.filename = "geom_xy_z0"
    p_xy.basis = "xy"
    p_xy.origin = (0, 0, 0)  # z=0 slice
    p_xy.width = (60.0, 60.0)
    p_xy.pixels = (1000, 1000)
    p_xy.color_by = "material"
    p_xy.colors = {be: "lightgrey", hdpe: "lightblue", cd: "red", he3: "orange"}
    plots_list.append(p_xy)

    # Longitudinal bisection through axis (xz plane at y=0)
    p_xz = openmc.Plot()
    p_xz.filename = "geom_xz_y0"
    p_xz.basis = "xz"
    p_xz.origin = (0, 0, 0)  # y=0 slice
    p_xz.width = (60.0, 100.0)  # taller to see full height; tune as needed
    p_xz.pixels = (1000, 1600)
    p_xz.color_by = "material"
    p_xz.colors = {be: "lightgrey", hdpe: "lightblue", cd: "red", he3: "orange"}
    plots_list.append(p_xz)

    plots = openmc.Plots(plots_list)
    plots.export_to_xml(path=str(input_dir / "plots.xml"))
    openmc.plot_geometry(path_input=str(input_dir))

    # =============================================================================
    # Tallies: Leakage current on Be surface
    # =============================================================================

    # Energy filter to separate 14 MeV (uncollided/elastic) from (n,2n) neutrons
    # Paper mentions two groups: ~14 MeV and 1-3 MeV
    # energy_bins = np.array([0, 1.0e6, 3.0e6, 6.0e6, 10.0e6, 14.0e6, 15.0e6])
    energy_bins = [0, 3.0e6, 15.0e6]  # simplified: low (<3 MeV) and high (>3 MeV)
    energy_filter = openmc.EnergyFilter(energy_bins)

    # time filter:
    # 14 MeV neutron speed: ~5.2 cm/ns
    # 1 MeV neutron speed: ~1.4 cm/ns
    # For 9 cm radius cylinder, transit time ~2-10 ns
    time_bins = np.linspace(0, 500e-6, 501)  # 0 to 500 micros in 1 ns bins
    time_filter = openmc.TimeFilter(time_bins)

    tallies = openmc.Tallies()
    # Tally 1: Leakage vs time (the key response function)
    det_response = openmc.Tally(name="detector_response")
    #  map every click of the detector to a specific nanosecond
    he3_filter = openmc.CellFilter([he3_cell])
    # Reuse that same variable
    det_response.filters = [he3_filter, time_filter]
    # neutron is detected by the reaction n+3He->H+3H, counts as absorption
    det_response.scores = ["absorption"]
    # this is the dieaway time of the whole system (how long signal lingers afterb burst)
    tallies.append(det_response)

    # Total He-3 absorption (detection efficiency)
    # if Be produces 1.71 neutrons (ML), but the tally shows 0.17,
    # the detection efficiency epsilon is 10%
    det_total = openmc.Tally(name="detector_total")
    det_total.filters = [he3_filter]
    det_total.scores = ["absorption"]
    tallies.append(det_total)

    # Be leakage (to compare with Step 1)
    # sanity check: measures the neutrons at the surface of beryllium before
    # they hit the moderator, can measure how many neutrons are lost
    be_surface_filter = openmc.SurfaceFilter([be_radius, be_top, be_bot])
    be_leakage = openmc.Tally(name="be_leakage")
    be_leakage.filters = [be_surface_filter]
    be_leakage.scores = ["current"]
    tallies.append(be_leakage)

    # (n,2n) in Be
    n2n_tally = openmc.Tally(name="n2n_rate")
    n2n_tally.filters = [openmc.CellFilter([be_cell])]
    n2n_tally.scores = ["(n,2n)"]
    tallies.append(n2n_tally)
    tallies.export_to_xml(path=str(input_dir / "tallies.xml"))


if __name__ == "__main__":
    base_dir = Path(__file__).parent.resolve()
    build_inputs(base_dir / "inputs")
