#!/usr/bin/env python3
"""
Basic Be cylinder with 14 MeV point source and time filter
get the time distribution of leaking neutrons
"""

import matplotlib.pyplot as plt
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

# METHOD WITH VOID
BE_RADIUS = 9.0  # cm
BE_HEIGHT = 65.0  # cm (half-height = 32.5)

# Be surfaces
be_radius = openmc.ZCylinder(r=BE_RADIUS)
be_top = openmc.ZPlane(z0=BE_HEIGHT / 2)
be_bot = openmc.ZPlane(z0=-BE_HEIGHT / 2)

# Outer void (needed to tally surface current)
outer_cyl = openmc.ZCylinder(r=BE_RADIUS + 1, boundary_type="vacuum")
outer_top = openmc.ZPlane(z0=BE_HEIGHT / 2 + 1, boundary_type="vacuum")
outer_bot = openmc.ZPlane(z0=-BE_HEIGHT / 2 - 1, boundary_type="vacuum")

# 1. Define the Be Cell
be_cell = openmc.Cell(name="beryllium")
be_cell.fill = be
be_cell.region = -be_radius & -be_top & +be_bot

# 2. Define the Void Cell (The "Shell")
# Instead of using ~, just define it as:
# "Inside the outer box" AND "Outside the inner cylinder"
void_cell = openmc.Cell(name="void")
void_cell.fill = None
void_cell.region = (-outer_cyl & -outer_top & +outer_bot) & (
    +be_radius | +be_top | -be_bot
)

universe = openmc.Universe(cells=[be_cell, void_cell])

# =============================================================================
# Source: 14 MeV point source at center, isotropic
# =============================================================================
point = openmc.stats.Point((0, 0, 0))
# Gaussian around 14.1 MeV (D-T), sigma ~ 50 keV typical
energy_dist = openmc.stats.Normal(mean_value=14.1e6, std_dev=5.0e4)
angle_dist = openmc.stats.Isotropic()
time_dist = openmc.stats.Discrete([0.0], [1.0])  # all start at t=0
source = openmc.IndependentSource(
    space=point, energy=energy_dist, angle=angle_dist, time=time_dist
)

# =============================================================================
# Settings
# =============================================================================
settings = openmc.Settings()
settings.run_mode = "fixed source"
settings.source = source
settings.batches = 100
# more particles means smoother results and smaller error bars
settings.particles = 10000  # 1M total histories
# no inactive batches. every neutron counts toward final answer
settings.export_to_xml()

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
# Use fine bins from 0 to 100 ns to capture the distribution
time_bins = np.linspace(0, 100e-9, 101)  # 0 to 100 ns in 1 ns bins
time_filter = openmc.TimeFilter(time_bins)

# Surface filter for leakage
surface_filter = openmc.SurfaceFilter([be_radius, be_top, be_bot])

tallies = openmc.Tallies()
# Tally 1: Leakage vs time (the key response function)
leakage_time = openmc.Tally(name="leakage_vs_time")
leakage_time.filters = [surface_filter, time_filter]
leakage_time.scores = ["current"]
tallies.append(leakage_time)

# Tally 2: Leakage vs time AND energy (separate fast/slow neutrons)
leakage_time_energy = openmc.Tally(name="leakage_vs_time_energy")
leakage_time_energy.filters = [surface_filter, time_filter, energy_filter]
leakage_time_energy.scores = ["current"]
tallies.append(leakage_time_energy)

# Tally 3: Total leakage (for normalization check)
leakage_total = openmc.Tally(name="leakage_total")
leakage_total.filters = [surface_filter]
leakage_total.scores = ["current"]
tallies.append(leakage_total)

# Tally 4: (n,2n) rate (same as before)
n2n_tally = openmc.Tally(name="n2n_rate")
n2n_tally.filters = [openmc.CellFilter([be_cell])]
n2n_tally.scores = ["(n,2n)"]
tallies.append(n2n_tally)
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

# Total leakage (sanity check)
total = sp.get_tally(name="leakage_total")
total_summed = total.summation(filter_type=openmc.SurfaceFilter, remove_filter=True)
M_L = total_summed.mean.flatten()[0]
print(f"\nTotal leakage M_L = {M_L:.4f}")

# (n,2n) rate
n2n = sp.get_tally(name="n2n_rate")
print(f"(n,2n) rate = {n2n.mean.flatten()[0]:.4f}")

# Leakage vs time
leakage_t = sp.get_tally(name="leakage_vs_time")
leakage_t_summed = leakage_t.summation(
    filter_type=openmc.SurfaceFilter, remove_filter=True
)
time_df = leakage_t_summed.get_pandas_dataframe()

# Extract time bin centers and values
t_low = time_df["time low [s]"].values
t_high = time_df["time high [s]"].values
t_centers = (t_low + t_high) / 2 * 1e9  # convert to ns
leakage_vs_t = time_df["mean"].values

print(f"\nTime distribution integral: {leakage_vs_t.sum():.4f} (should = M_L)")

# Leakage vs time and energy
leakage_te = sp.get_tally(name="leakage_vs_time_energy")
leakage_te_summed = leakage_te.summation(
    filter_type=openmc.SurfaceFilter, remove_filter=True
)
te_df = leakage_te_summed.get_pandas_dataframe()

# Separate low and high energy
low_E_mask = te_df["energy high [eV]"] <= 3.0e6
high_E_mask = te_df["energy low [eV]"] >= 3.0e6

low_E_df = te_df[low_E_mask].copy()
high_E_df = te_df[high_E_mask].copy()

t_centers_low = (
    (low_E_df["time low [s]"].values + low_E_df["time high [s]"].values) / 2
) * 1e9
t_centers_high = (
    (high_E_df["time low [s]"].values + high_E_df["time high [s]"].values) / 2
) * 1e9

leakage_low_E = low_E_df["mean"].values
leakage_high_E = high_E_df["mean"].values

print(f"Low energy (<3 MeV) integral:  {leakage_low_E.sum():.4f}")
print(f"High energy (>3 MeV) integral: {leakage_high_E.sum():.4f}")

sp.close()

# =============================================================================
# Plot results
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot 1: Total leakage vs time
axes[0].bar(t_centers, leakage_vs_t, width=1.0, alpha=0.7, label="All energies")
axes[0].set_xlabel("Time (ns)")
axes[0].set_ylabel("Leakage current per source neutron per ns")
axes[0].set_title("Neutron Leakage Time Distribution")
axes[0].set_xlim(0, 50)
axes[0].legend()

# Plot 2: Separated by energy
axes[1].bar(
    t_centers_low, leakage_low_E, width=1.0, alpha=0.7, label="<3 MeV (n,2n products)"
)
axes[1].bar(
    t_centers_high, leakage_high_E, width=1.0, alpha=0.7, label=">3 MeV (source)"
)
axes[1].set_xlabel("Time (ns)")
axes[1].set_ylabel("Leakage current per source neutron per ns")
axes[1].set_title("Leakage by Energy Group")
axes[1].set_xlim(0, 50)
axes[1].legend()

plt.tight_layout()
plt.savefig("leakage_time_distribution.png", dpi=150)

print("\nPlot saved to leakage_time_distribution.png")
