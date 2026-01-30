import openmc

BE9_PATH = "/home/philip/Documents/endf-b8.0-hdf5/endfb-viii.0-hdf5/neutron/Be9.h5"


# Load the evaluation
be9 = openmc.data.IncidentNeutron.from_hdf5(BE9_PATH)

# Check if covariance data exists
# MT=16 is (n,2n)
print(be9.reactions[16])
