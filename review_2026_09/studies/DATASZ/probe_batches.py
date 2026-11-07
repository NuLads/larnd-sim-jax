"""DATASZ probe (3 Nov, CPU): what the production argv actually loads, as a function of --data_sz, --max_nbatch and
--max_batch_len (the 'requested is not what you get' family). Builds only the simulation TracksDataset (no fit).
usage: python3 probe_batches.py <data_sz> <max_nbatch> <max_batch_len> [data_seed]"""
import sys, logging, numpy as np
logging.basicConfig(level=logging.WARNING)
from optimize.dataio import TracksDataset
nev, nb, bl = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]); seed = int(sys.argv[4]) if len(sys.argv) > 4 else 3
d = TracksDataset(filename='/sdf/group/neutrino/pgranger/lads-data/linear_guess_segments.h5', nevents=nev, max_nbatch=nb,
                  random_nevents=True, data_seed=seed, track_len_sel=2, max_abs_costheta_sel=0.966, min_abs_segz_sel=15.,
                  track_z_bound=28, max_batch_len=bl, print_input=False, chopped=True, pad=True,
                  electron_sampling_resolution=0.01, live_selection=False)
ntr = [len(b) for b in d.batch_traj_indices]
tl = np.array([d.tracks_struct[r]['dx'].sum() for r in d.trajectory_row_indices])
print(f"data_sz {nev} max_nbatch {nb} len {bl} seed {seed}: batches {len(ntr)}, tracks used {sum(ntr)} "
      f"(per batch {np.mean(ntr):.1f}, min {min(ntr)} max {max(ntr)}), length used {d.tot_data_length:.0f} cm | "
      f"selected trajectories {len(tl)}, total {tl.sum():.0f} cm, longest {tl.max():.1f} cm, > len {int((tl > bl).sum())}")
