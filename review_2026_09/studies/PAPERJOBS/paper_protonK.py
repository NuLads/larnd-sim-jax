"""R101 follow-up: beam width K on the stopping-proton batch of paper_protonval.py (same data, same seed): expected hits
per slot of the probabilistic model for several K, against the 1e4-throw frequencies of protonval.npz."""
import sys, json, time
from paper_common import *
from larndsim.fee_jax import get_adc_values_average_noise_chunked, digitize, get_average_hit_values
PROTON = '/sdf/data/neutrino/cyifan/diffsim_input/true_proton_edep_2cm_range_0.1-cm.h5'
KS = [int(k) for k in sys.argv[1].split(',')]; OUT = sys.argv[2]
truth, response = params()
ds = TracksDataset(filename=PROTON, nevents=600, max_nbatch=1, random_nevents=True, data_seed=1, track_len_sel=2.,
                   max_abs_costheta_sel=0.966, min_abs_segz_sel=15., track_z_bound=28., max_batch_len=200., print_input=False,
                   chopped=True, pad=True, electron_sampling_resolution=0.01, live_selection=False, use_dedx_density=False, dedx_density_mode='histogram')
fields = ds.get_track_fields(); tracks = jnp.asarray(ds[0])
wfs, up = simulate_wfs(truth, response, tracks, fields); up = np.asarray(up); wfs = jnp.asarray(np.asarray(wfs)[up >= 0])
res = {}
for K in KS:
    t0 = time.time(); p = truth.replace(fee_paths_scaling=K)
    lp, ch = get_adc_values_average_noise_chunked(p, wfs, chunk_size=128)
    Et, Eadc, lam = (np.asarray(x) for x in get_average_hit_values(lp, digitize(p, ch)))
    res[K] = lam
    print(f'K={K}: expected hits {lam.sum():.1f}; per slot ' + ' '.join(f'{lam[:, n].sum():.1f}' for n in range(5)) + f'  ({time.time()-t0:.0f}s)', flush=True)
np.savez_compressed(OUT, **{f'lam_K{K}': v for K, v in res.items()})
