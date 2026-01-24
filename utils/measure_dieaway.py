import openmc
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from pathlib import Path

# Import your existing geometry builder
from build_complex_input import create_geometry 

def run_dieaway_simulation(input_dir, output_dir, particles=100_000):
    """Runs a pulsed-source simulation to measure die-away."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Create Geometry
    geo = create_geometry(input_dir)
    
    # 2. Create Settings for a PULSE (t=0)
    settings = openmc.Settings()
    settings.particles = particles
    settings.batches = 1
    settings.run_mode = "fixed source"
    settings.output = {"path": str(output_dir), "tallies": False}
    
    # Force specific cell IDs for recording tracks (from your geo function)
    settings.collision_track = {
        "cell_ids": geo["he3_cell_ids"], 
        "max_collisions": int(particles)
    }

    # === THE KEY CHANGE: Pulsed Source at t=0 ===
    settings.source = openmc.IndependentSource(
        space=openmc.stats.Point((0, 0, 0)),
        energy=openmc.stats.Normal(mean_value=14.1e6, std_dev=5.0e4),
        angle=openmc.stats.Isotropic(),
        time=openmc.stats.Uniform(0.0, 1e-6) # All neutrons start at t=0
    )
    settings.export_to_xml(path=str(output_dir / "settings.xml"))
    
    # Copy materials/geometry to run dir
    for f in ["materials.xml", "geometry.xml"]:
        (input_dir / f).rename(output_dir / f) # or copy

    # 3. Run OpenMC
    openmc.run(cwd=str(output_dir), path_input=str(output_dir))
    
    return output_dir

def analyze_dieaway(run_dir):
    """Fits the exponential decay to find Tau."""
    run_dir = Path(run_dir)
    
    # Load data
    track_file = run_dir / "collision_track.h5"
    tracks = openmc.read_collision_track_hdf5(str(track_file))
    
    # Filter for absorption (MT=101 for n,gamma in He3 is standard, 
    # but check your library. Usually 102 is capture, 103 is (n,p). 
    # For He3(n,p)T, the MT number is typically 103. 
    # OpenMC might map capture to 101/102 generic. 
    # Let's assume you verified MT=101 or simply take all collisions in the He3 cells).
    
    # Safest bet for He3 detectors: Take all events in the track file 
    # because we only asked for tracks in He3 cells, and He3 x-section is dominated by absorption.
    times = tracks['time']
    
    # Histogram the data
    # Use log-spaced bins or fine linear bins
    hist, bin_edges = np.histogram(times, bins=100, range=(1e-6, 1000e-6))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Select the "tail" region to fit (ignore the messy buildup at the start)
    # Visual inspection is best, but usually 50us to 500us is the sweet spot.
    fit_mask = (bin_centers > 50e-6) & (bin_centers < 600e-6) & (hist > 0)
    
    x_data = bin_centers[fit_mask]
    y_data = hist[fit_mask]
    
    # Fit function: A * exp(-t / tau)
    # Linear fit on log data is more robust: ln(y) = ln(A) - t/tau
    def log_line(t, a, tau):
        return a - (t / tau)
        
    popt, _ = curve_fit(log_line, x_data, np.log(y_data), p0=[np.log(max(y_data)), 100e-6])
    
    tau = popt[1]
    
    print(f"Calculated Die-Away Time (Tau): {tau*1e6:.2f} microseconds")
    
    # Plot to verify
    plt.semilogy(bin_centers * 1e6, hist, label='Simulated Data')
    plt.semilogy(x_data * 1e6, np.exp(log_line(x_data, *popt)), 'r--', label=f'Fit (Tau={tau*1e6:.1f} us)')
    plt.xlabel('Time (us)')
    plt.ylabel('Counts')
    plt.savefig("dieaway.png")
    plt.show()
    
    return tau



base_dir = Path(__file__).parent.parent.resolve()
input_dir = base_dir / "inputs"
output_dir = base_dir / "outputs"
run_dieaway_simulation(input_dir, output_dir)
analyze_dieaway("outputs")