"""
create_pos_demo_h5.py
Creates small h5 demo files where all muon dEdx is replaced by the mean value.
This isolates position fitting by ensuring sim and target have identical dEdx.

Outputs:
  data/pos_demo_sim.h5   — from linear_guess_segments.h5, dEdx=mean
  data/pos_demo_tgt.h5   — from true_through_muon_edep_10cm_vol1cm.h5, dEdx=mean
"""
import h5py
import numpy as np
import os

SIM_IN = '/sdf/group/neutrino/pgranger/lads-data/linear_guess_segments.h5'
TGT_IN = '/sdf/data/neutrino/cyifan/diffsim_input/true_through_muon_edep_10cm_vol1cm.h5'
N_EVENTS = 20   # first 20 events → typically ~5-15 usable muon batches
MEAN_DEDX = 1.885  # MeV/cm, typical MIP muon in LAr
os.makedirs('data', exist_ok=True)

def copy_with_uniform_dedx(src_path, dst_path, event_ids, mean_dedx):
    with h5py.File(src_path, 'r') as f:
        segs  = f['segments'][:]
        trajs = f['trajectories'][:]

    # Select events
    seg_mask  = np.isin(segs['event_id'],  event_ids)
    traj_mask = np.isin(trajs['event_id'], event_ids)
    segs  = segs[seg_mask].copy()
    trajs = trajs[traj_mask].copy()

    print(f"  {src_path}: {len(segs)} segments from {len(event_ids)} events")

    # Replace dEdx with mean for ALL segments (muon and non-muon alike)
    segs['dEdx'] = mean_dedx
    segs['dE']   = (mean_dedx * segs['dx']).astype(segs['dE'].dtype)
    # Recompute n_electrons (W_ion = 23.6 eV per ion pair, 1 MeV = 1e6 eV)
    W_ION = 23.6e-6  # MeV per ion pair
    segs['n_electrons'] = np.round(segs['dE'] / W_ION).astype(segs['n_electrons'].dtype)

    with h5py.File(dst_path, 'w') as f:
        f.create_dataset('segments',     data=segs,  compression='gzip', compression_opts=4)
        f.create_dataset('trajectories', data=trajs, compression='gzip', compression_opts=4)
    print(f"  Written: {dst_path}  ({os.path.getsize(dst_path)/1e6:.1f} MB)")

# Use the same event IDs from both files
with h5py.File(SIM_IN, 'r') as f:
    all_evs = np.unique(f['segments']['event_id'])
event_ids = all_evs[:N_EVENTS]
print(f"Selected {len(event_ids)} events: {event_ids[:5]} ... {event_ids[-5:]}")
print(f"Mean dEdx set to: {MEAN_DEDX} MeV/cm")

copy_with_uniform_dedx(SIM_IN, 'data/pos_demo_sim.h5', event_ids, MEAN_DEDX)
copy_with_uniform_dedx(TGT_IN, 'data/pos_demo_tgt.h5', event_ids, MEAN_DEDX)
print("Done.")
