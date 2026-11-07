import h5py
import numpy as np

def analyze_cathode_crossings(fname, label):
    with h5py.File(fname, 'r') as f:
        segments = f['segments'][:]
        
    event_ids = segments['event_id'] if 'event_id' in segments.dtype.names else segments['eventID']
    traj_ids = segments['traj_id'] if 'traj_id' in segments.dtype.names else segments['trackID']
    
    # x is the drift direction
    x_start = segments['x_start']
    x_end = segments['x_end']
    x_mid = segments['x']
    dE = segments['dE']
    dx = segments['dx']
    
    # Cathode gap is along x: [-0.15875, 0.15875] cm
    cathode_min = -0.15875
    cathode_max = 0.15875
    
    # A segment is inside the cathode if its middle point is within the gap
    in_cathode_mask = (x_mid >= cathode_min) & (x_mid <= cathode_max)
    
    total_de = np.sum(dE)
    total_dx = np.sum(dx)
    
    cathode_de = np.sum(dE[in_cathode_mask])
    cathode_dx = np.sum(dx[in_cathode_mask])
    
    print(f"=== Cathode Gap Analysis for {label} ===")
    print(f"Total Segments: {len(segments):,}")
    print(f"Segments inside cathode gap: {np.sum(in_cathode_mask):,} ({np.mean(in_cathode_mask)*100:.3f}%)")
    print(f"Total dx: {total_dx:.2f} cm")
    print(f"dx inside cathode gap: {cathode_dx:.2f} cm ({cathode_dx/total_dx*100:.3f}%)")
    print(f"Total dE: {total_de:.2f} MeV")
    print(f"dE inside cathode gap: {cathode_de:.2f} MeV ({cathode_de/total_de*100:.3f}%)")
    print()

analyze_cathode_crossings('/sdf/data/neutrino/cyifan/diffsim_input/true_through_muon_edep_10cm_vol1cm.h5', 'Target (True Tracks)')
analyze_cathode_crossings('/sdf/group/neutrino/pgranger/lads-data/linear_guess_segments.h5', 'Guess (Linear Approximation)')
