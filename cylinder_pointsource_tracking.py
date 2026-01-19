#!/usr/bin/env python3
"""
Basic Be cylinder with 14 MeV point source.

Goal: Get M_L (leakage multiplication) and compare to literature values.
reference experiment is on page 130 Srinivasan.
"""

import numpy as np
import openmc

# os.environ["OPENMC_CROSS_SECTIONS"] = os.path.abspath(
#     "Documents/endf-b8.0-hdf5/endfb-viii.0-hdf5/cross_sections.xml"
# )
# =============================================================================
# Materials
# =============================================================================
be = openmc.Material(name="beryllium")
be.add_nuclide("Be9", 1.0)  # 1,0 is composition value (100% berylium-9)
be.set_density("g/cm3", 1.85)  # physical density of solid beryllium at room temp

materials = openmc.Materials([be])  # collection object
materials.export_to_xml()  # generates materials.xml file

# =============================================================================
# Geometry: Be cylinder (Basu et al. dimensions)
# 65 cm long, 18 cm diameter (9 cm radius)
# Point source at center
# =============================================================================

# # Surfaces (defining the physical size)
# # Setting boundary_type="vacuum" tells OpenMC to kill any neutron that touches it
be_radius = openmc.ZCylinder(r=9.0, boundary_type="vacuum")
be_top = openmc.ZPlane(z0=32.5, boundary_type="vacuum")
be_bot = openmc.ZPlane(z0=-32.5, boundary_type="vacuum")
# Be cell
# Region: inside the cylinder (-) and between the top(-) and bottom(+) planes
be_cell = openmc.Cell(name="beryllium")
be_cell.fill = be
be_cell.region = -be_radius & -be_top & +be_bot
universe = openmc.Universe(cells=[be_cell])
geometry = openmc.Geometry(universe)
geometry.export_to_xml()
# # === METHOD WITH VOID
# BE_RADIUS = 9.0  # cm
# BE_HEIGHT = 65.0  # cm (half-height = 32.5)
#
# # Be surfaces
# be_radius = openmc.ZCylinder(r=BE_RADIUS)
# be_top = openmc.ZPlane(z0=BE_HEIGHT / 2)
# be_bot = openmc.ZPlane(z0=-BE_HEIGHT / 2)
#
# # Outer void (needed to tally surface current)
# outer_cyl = openmc.ZCylinder(r=BE_RADIUS + 1, boundary_type="vacuum")
# outer_top = openmc.ZPlane(z0=BE_HEIGHT / 2 + 1, boundary_type="vacuum")
# outer_bot = openmc.ZPlane(z0=-BE_HEIGHT / 2 - 1, boundary_type="vacuum")
#
# # 1. Define the Be Cell
# be_cell = openmc.Cell(name="beryllium")
# be_cell.fill = be
# be_cell.region = -be_radius & -be_top & +be_bot
#
# # 2. Define the Void Cell (The "Shell")
# # Instead of using ~, just define it as:
# # "Inside the outer box" AND "Outside the inner cylinder"
# void_cell = openmc.Cell(name="void")
# void_cell.fill = None
# void_cell.region = (-outer_cyl & -outer_top & +outer_bot) & (
#     +be_radius | +be_top | -be_bot
# )
#
# universe = openmc.Universe(cells=[be_cell, void_cell])

# =============================================================================
# Source: 14 MeV point source at center, isotropic
# =============================================================================
point = openmc.stats.Point((0, 0, 0))
# Gaussian around 14.1 MeV (D-T), sigma ~ 50 keV typical
energy_dist = openmc.stats.Normal(mean_value=14.1e6, std_dev=5.0e4)
angle_dist = openmc.stats.Isotropic()
source = openmc.IndependentSource(space=point, energy=energy_dist, angle=angle_dist)

# =============================================================================
# Settings
# =============================================================================
settings = openmc.Settings()
settings.run_mode = "fixed source"
settings.source = source
settings.batches = 1
# more particles means smoother results and smaller error bars
settings.particles = 1000  # 1M total histories
# no inactive batches. every neutron counts toward final answer

# TRACKING HERE
# to track particles 1-100 in batch 1 only
settings.track = [(1, 1, i) for i in range(1, 1000)]  # (batch, gen, particle)

settings.export_to_xml()

# =============================================================================
# Tallies: Leakage current on Be surface
# =============================================================================


# Energy filter to separate 14 MeV (uncollided/elastic) from (n,2n) neutrons
# Paper mentions two groups: ~14 MeV and 1-3 MeV
energy_bins = np.array([0, 1.0e6, 3.0e6, 6.0e6, 10.0e6, 14.0e6, 15.0e6])
energy_filter = openmc.EnergyFilter(energy_bins)

# Surface filter for leakage
surface_filter = openmc.SurfaceFilter([be_radius, be_top, be_bot])

# Tally 1: Total leakage current (all surfaces)
# at the exact outer boundary of Be cyl, get one single number
# numerator for Leakage multiplication formula
leakage_tally = openmc.Tally(name="leakage_current")
leakage_tally.filters = [surface_filter]
leakage_tally.scores = ["current"]

# Tally 2: Leakage spectrum
# also at the boundary, several different buckets lined up based on energy
# if 14 MeV neutron crosses, goes in the Fast bucket, 2Mev in slow
# get a list of numbers (a histogram)
leakage_spectrum = openmc.Tally(name="leakage_spectrum")
leakage_spectrum.filters = [surface_filter, energy_filter]
leakage_spectrum.scores = ["current"]

# Tally 3: (n,2n) reaction rate in Be (for verification)
# counts collisions where one neutron hits Be atom and 2 neutrons come out
n2n_tally = openmc.Tally(name="n2n_rate")
n2n_tally.filters = [openmc.CellFilter([be_cell])]
n2n_tally.scores = ["(n,2n)"]

# Tally 4: Absorption in Be (neutron losses)
# bc neutrons out = source +n2n - absorption
absorption_tally = openmc.Tally(name="absorption")
absorption_tally.filters = [openmc.CellFilter([be_cell])]
absorption_tally.scores = ["absorption"]

tallies = openmc.Tallies([leakage_tally, leakage_spectrum, n2n_tally, absorption_tally])
tallies.export_to_xml()

# =============================================================================
# Run
# =============================================================================
print("Running OpenMC...")
openmc.run()

# =============================================================================
# Process results
# =============================================================================
# sp = openmc.StatePoint(glob.glob('statepoint.*.h5')[-1])
sp = openmc.StatePoint("statepoint.100.h5")

# Extract leakage
leakage = sp.get_tally(name="leakage_current")
leakage_summed = leakage.summation(filter_type=openmc.SurfaceFilter, remove_filter=True)
leakage_mean = leakage_summed.mean.flatten()[0]
leakage_std = leakage_summed.std_dev.flatten()[0]

# Extract (n,2n) rate
n2n = sp.get_tally(name="n2n_rate")
n2n_mean = n2n.mean.flatten()[0]
n2n_std = n2n.std_dev.flatten()[0]

# Extract absorption
absorption = sp.get_tally(name="absorption")
abs_mean = absorption.mean.flatten()[0]
abs_std = absorption.std_dev.flatten()[0]

# Extract spectrum using summation
spectrum = sp.get_tally(name="leakage_spectrum")
spectrum_summed = spectrum.summation(
    filter_type=openmc.SurfaceFilter, remove_filter=True
)
spectrum_df = spectrum_summed.get_pandas_dataframe()


print("\n" + "=" * 60)
print("RESULTS (per source neutron)")
print("=" * 60)
print(f"Leakage current (M_L):     {leakage_mean:.4f} +/- {leakage_std:.4f}")
print(f"(n,2n) reactions:          {n2n_mean:.4f} +/- {n2n_std:.4f}")
print(f"Absorption in Be:          {abs_mean:.4f} +/- {abs_std:.4f}")
print(f"Neutron balance check:     {1 + n2n_mean - abs_mean:.4f} (should ~ M_L)")
print(f"Theoretical maximum multiplication:     {(n2n_mean / 1.0) + 1}")

print("\nLeakage spectrum:")
print(spectrum_df[["energy low [eV]", "energy high [eV]", "mean", "std. dev."]])

sp.close()
