import openmc.data

# Load the Be9 data from your environment's library
# Ensure your OPENMC_CROSS_SECTIONS env variable is set
# lib = openmc.data.IncidentNeutron.from_hdf5(
#     openmc.config["cross_sections"].find_nuclide("Be9")
# )
# print(f"Reactions for {lib.name}:")
# for mt, reaction in lib.reactions.items():
#     print(f"MT {mt}: {reaction.description}")
#
BE9_PATH = "/home/philip/Documents/endf-b8.0-hdf5/endfb-viii.0-hdf5/neutron/Be9.h5"


# Load the evaluation
be9 = openmc.data.IncidentNeutron.from_hdf5(BE9_PATH)
rx = be9.reactions[16]
prod = rx.products[0]
# openmc.data.AngleEnergy type
dist = prod.distribution
# openmc.data.CorrelatedAngleEnergy type
file6 = dist[0]
# has parameters: breakpoints. interpolation, energy, energy_out, mu
# energy: iterable of float
# energy_out iterable of openmc.stats.Univariate.tabular, distribution of outgoing energies for each incoming
# energy_out[i] represents p(E'|E_i)
# mu[i][j] represents p(mu|E_i, E'_j)
# mu iterable of iterable of openmc.stats.Univariate.tabular, distribution of scattering cosine for each incoming/outoing energy
print(file6.energy_out[0].integral())
