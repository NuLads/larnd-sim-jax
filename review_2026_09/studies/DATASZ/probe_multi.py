"""DATASZ probe (3 Nov, CPU): what the production argv actually loads for each (data_sz, max_nbatch, max_batch_len,
data_seed): batches, tracks, length, and the set of tracks used (first row index of each used trajectory), saved to
probe_sets.npz so overlap between configurations can be checked. data_sz 0 = the whole pool (no event selection).
usage: python3 probe_multi.py nev:nb:len:seed [...]"""
import sys, logging, numpy as np
logging.basicConfig(level=logging.WARNING)
from optimize.dataio import TracksDataset
out = {}
for spec in sys.argv[1:]:
    nev, nb, bl, seed = spec.split(':'); nev, nb, bl, seed = int(nev), int(nb), float(bl), int(seed)
    d = TracksDataset(filename='/sdf/group/neutrino/pgranger/lads-data/linear_guess_segments.h5', nevents=nev, max_nbatch=nb,
                      random_nevents=True, data_seed=seed, track_len_sel=2, max_abs_costheta_sel=0.966, min_abs_segz_sel=15.,
                      track_z_bound=28, max_batch_len=bl, print_input=False, chopped=True, pad=True,
                      electron_sampling_resolution=0.01, live_selection=False)
    used = np.array(sorted(int(d.trajectory_row_indices[t][0]) for b in d.batch_traj_indices for t in b))
    ntr = [len(b) for b in d.batch_traj_indices]
    tl = np.array([d.tracks_struct[r]['dx'].sum() for r in d.trajectory_row_indices])
    ev = np.unique([d.tracks_struct[r[0]]['eventID'] for r in d.trajectory_row_indices])
    out[spec] = used
    print(f"{spec}: batches {len(ntr)}, tracks used {len(used)} ({np.mean(ntr):.1f}/batch, min {min(ntr)} max {max(ntr)}), "
          f"length used {d.tot_data_length:.0f} cm | selected: {len(ev)} events, {len(tl)} trajectories, {tl.sum():.0f} cm, "
          f"longest {tl.max():.1f} cm, > len {int((tl > bl).sum())}", flush=True)
np.savez('probe_sets.npz', **{k.replace(':', '_').replace('.', 'p'): v for k, v in out.items()})
