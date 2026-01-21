#!/usr/bin/env python3
"""
Basic Be cylinder with 14 MeV point source and time filter
get the time distribution of leaking neutrons
"""

from pathlib import Path

import numpy as np
import openmc

openmc.reset_auto_ids()

# os.environ["OPENMC_CROSS_SECTIONS"] = os.path.abspath(
#     "Documents/endf-b8.0-hdf5/endfb-viii.0-hdf5/cross_sections.xml"
# )
# Create a folder for inputs if it doesn't exist
base_dir = Path(__file__).parent.resolve()
input_dir = base_dir / "inputs"
output_dir = base_dir / "outputs"

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
materials.export_to_xml(path=f"{input_dir}/materials.xml")

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
geometry.export_to_xml(path=f"{input_dir}/geometry.xml")

# =============================================================================
# Settings
# =============================================================================
settings = openmc.Settings()
settings.batches = 1
settings.particles = 1000000  # 1M total histories
N = settings.particles * settings.batches
# RATE = 1e8  # neutrons/second
RATE = 3e4  # neutrons/second
T = N / RATE

# Source: 14 MeV point source at center, isotropic
# Gaussian around 14.1 MeV (D-T), sigma ~ 50 keV typical
source = openmc.IndependentSource(
    space=openmc.stats.Point((0, 0, 0)),
    energy=openmc.stats.Normal(mean_value=14.1e6, std_dev=5.0e4),
    angle=openmc.stats.Isotropic(),
    time=openmc.stats.Uniform(0.0, T),
)
settings.source = source
settings.run_mode = "fixed source"

# settings.track = [(1, 1, i) for i in range(1, settings.particles)]  # (batch, gen, part)
# Assuming be_cell and poly_inner_cell were defined earlier
settings.collision_track = {
    "cell_ids": [he3_cell.id],
    # "reactions": [103],
    "max_collisions": 1000000,
    "max_collision_track_files": 100,
}
# OpenMC RNG seed (reproducible result)
# settings.seed = 12346
settings.output = {
    "path": "outputs",  # This moves statepoints, summary, etc.
    "tallies": False,  # Optional: set to False to stop generating tallies.out
}
settings.export_to_xml(path=f"{input_dir}/settings.xml")

# Create a plot object
# plot = openmc.Plot()
# plot.filename = "geometry_debug"
# plot.origin = (0, 0, 0)
# plot.width = (60.0, 60.0)  # Total width/height in cm
# plot.pixels = (1000, 1000)  # Resolution
# plot.color_by = "material"
# # plot.colors = {be: "lightgrey", hdpe: "blue", cadmium: "red", he3: "orange"}
# plot.colors = {be: "lightgrey", hdpe: "blue", he3: "orange"}
# # Create a Plots collection and export
# plots = openmc.Plots([plot])
# plots.export_to_xml(path=f"{input_dir}/settings.xml")
# # Generate the plot (.png file)
# openmc.plot_geometry()

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
tallies.export_to_xml(path=f"{input_dir}/tallies.xml")

# =============================================================================
# Run
# =============================================================================
print("Running OpenMC...")
openmc.run(output=False, cwd=output_dir, path_input=input_dir)

# =============================================================================
# Process results
# =============================================================================
sp = openmc.StatePoint(f"{output_dir}/statepoint.{settings.batches}.h5")

# Total detection (efficiency)
det_total_tally = sp.get_tally(name="detector_total")
detection_efficiency = det_total_tally.mean.flatten()[0]

# Be leakage
be_leak = sp.get_tally(name="be_leakage")
be_leak_summed = be_leak.summation(filter_type=openmc.SurfaceFilter, remove_filter=True)
M_L = be_leak_summed.mean.flatten()[0]

# (n,2n) rate
n2n = sp.get_tally(name="n2n_rate")
n2n_rate = n2n.mean.flatten()[0]

print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)
print(f"Be leakage (M_L):           {M_L:.4f}")
print(f"(n,2n) rate:                {n2n_rate:.4f}")
print(f"He-3 detection efficiency:  {detection_efficiency:.4f}")
print("  (fraction of source neutrons detected)")

# Detector response vs time
det_resp = sp.get_tally(name="detector_response")
det_df = det_resp.get_pandas_dataframe()

t_low = det_df["time low [s]"].values
t_high = det_df["time high [s]"].values
t_centers = (t_low + t_high) / 2 * 1e6  # convert to μs
response = det_df["mean"].values

# Verify integral
print(f"Response integral:          {response.sum():.4f} (should = efficiency)")
print(T)

# Fit exponential to extract die-away time
# Response should follow: R(t) ∝ exp(-t/τ)
# Find where response drops to 1/e of max
max_idx = np.argmax(response)
max_val = response[max_idx]

# Find 1/e point after the peak
threshold = max_val / np.e
after_peak = response[max_idx:]
t_after_peak = t_centers[max_idx:]

try:
    # Find first crossing below threshold
    below_threshold = np.where(after_peak < threshold)[0]
    if len(below_threshold) > 0:
        tau_idx = below_threshold[0]
        tau_estimate = t_after_peak[tau_idx] - t_centers[max_idx]
        print(f"\nEstimated die-away time τ:  {tau_estimate:.1f} μs")
    else:
        print("\nCould not estimate die-away time (response doesn't decay enough)")
except:
    print("\nCould not estimate die-away time")

sp.close()

# =============================================================================
# Plot
# =============================================================================
# fig, axes = plt.subplots(1, 2, figsize=(12, 5))
#
# # Linear scale
# axes[0].plot(t_centers, response, "b-", linewidth=0.5)
# axes[0].set_xlabel("Time (μs)")
# axes[0].set_ylabel("He-3 absorption rate (per source neutron per μs)")
# axes[0].set_title("Detector Response Function")
# axes[0].set_xlim(0, 200)
# axes[0].grid(True, alpha=0.3)
#
# # Log scale (to see exponential decay)
# axes[1].semilogy(t_centers, response, "b-", linewidth=0.5)
# axes[1].set_xlabel("Time (μs)")
# axes[1].set_ylabel("He-3 absorption rate (log scale)")
# axes[1].set_title("Detector Response - Log Scale")
# axes[1].set_xlim(0, 300)
# axes[1].grid(True, alpha=0.3)
#
# plt.tight_layout()
# plt.savefig("detector_response.png", dpi=150)
# plt.show()
#
# print("\nPlot saved to detector_response.png")
