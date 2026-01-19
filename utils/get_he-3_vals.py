"""
get helium 3 values for openmc
"""

pressure_atm = 4  # atm
temp = 300  # K


pressure_Pa = 101325 * pressure_atm
# https://pubchem.ncbi.nlm.nih.gov/compound/Helium-3
molar_mass = 3.0160293220  # g/mol

# calc mass density from ideal gas law
R = 8.314462618  # J/(mol K)
rho = pressure_Pa * molar_mass / (R * temp)
# density in g/cm3
rho_gcm3 = rho / 1e6
print(rho_gcm3)

avogrado = 6.0221408e23
atomic_density = rho * avogrado / molar_mass
N_atom_bcm = atomic_density / 10e24
