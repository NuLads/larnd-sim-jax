"""Review R101 (20 Oct 2026): the validation of Fig. prob_vs_stoch (probabilistic electronics model against stochastic
throws of the SAME waveforms) repeated on stopping protons. One 200-cm batch of true stopping-proton segments
(cyifan/diffsim_input/true_proton_edep_2cm_range_0.1-cm.h5, the target file of the proton fits), production detector
parameters, production electronics: model = get_adc_values_average_noise_chunked (K = 100) + get_average_hit_values;
stochastic = fee_jax.get_adc_values, N throws; plus M pairs of single throws (two noise seeds) for the scale of the
noise. Same accumulators as fig_prob_vs_stoch.py, for the first NH hit slots.
usage: python3 paper_protonval.py --n_throws 10000 --chunk 50 --pairs 50 --out /abs/path.npz"""
import argparse, json, time
from paper_common import *
from larndsim.fee_jax import get_adc_values_average_noise_chunked, digitize, get_average_hit_values, get_adc_values
PROTON = '/sdf/data/neutrino/cyifan/diffsim_input/true_proton_edep_2cm_range_0.1-cm.h5'
ap = argparse.ArgumentParser(); ap.add_argument('--n_throws', type=int, default=10000); ap.add_argument('--chunk', type=int, default=50)
ap.add_argument('--pairs', type=int, default=50); ap.add_argument('--nh', type=int, default=3); ap.add_argument('--len', type=float, default=200.)
ap.add_argument('--out', required=True); a = ap.parse_args()
truth, response = params()
ds = TracksDataset(filename=PROTON, nevents=600, max_nbatch=1, random_nevents=True, data_seed=1, track_len_sel=2.,
                   max_abs_costheta_sel=0.966, min_abs_segz_sel=15., track_z_bound=28., max_batch_len=a.len, print_input=False,
                   chopped=True, pad=True, electron_sampling_resolution=0.01, live_selection=False,
                   use_dedx_density=False, dedx_density_mode='histogram')
fields = ds.get_track_fields(); tracks = jnp.asarray(ds[0]); L = track_length_cm(tracks, fields)
t0 = time.time(); wfs, up = simulate_wfs(truth, response, tracks, fields); up = np.asarray(up); valid = up >= 0
wfs = jnp.asarray(np.asarray(wfs)[valid]); npx = int(valid.sum())
print(f'protons: {L:.1f} cm of track, {npx} active pixels, simulate_wfs {time.time()-t0:.1f}s', flush=True)
t0 = time.time(); lp, ch = get_adc_values_average_noise_chunked(truth, wfs, chunk_size=128)
Et, Eadc, lam = (np.asarray(x) for x in get_average_hit_values(lp, digitize(truth, ch)))
print(f'model: {time.time()-t0:.1f}s, expected hits {lam.sum():.1f}', flush=True)
NH = a.nh
cnt, st, st2, sa, sa2 = (np.zeros((npx, NH)) for _ in range(5)); tot = np.zeros(npx)
run = jax.jit(lambda w, k: get_adc_values(truth, w, k)); key = jax.random.key(2026); t0 = time.time()
nchunks = a.n_throws // a.chunk
for i in range(nchunks):
    key, sub = jax.random.split(key)
    q, tk = run(jnp.tile(wfs, (a.chunk, 1)), sub)
    adc = np.asarray(digitize(truth, q)).reshape(a.chunk, npx, -1); qq = np.asarray(q).reshape(a.chunk, npx, -1)
    tk = np.asarray(tk).reshape(a.chunk, npx, -1); hit = qq > 0; tot += hit.sum(axis=(0, 2))
    for n in range(NH):
        h = hit[:, :, n]
        cnt[:, n] += h.sum(0); st[:, n] += np.where(h, tk[:, :, n], 0).sum(0); st2[:, n] += np.where(h, tk[:, :, n] ** 2, 0).sum(0)
        sa[:, n] += np.where(h, adc[:, :, n], 0).sum(0); sa2[:, n] += np.where(h, adc[:, :, n] ** 2, 0).sum(0)
    if i == 0 or (i + 1) % 40 == 0: print(f'{(i+1)*a.chunk} throws, {time.time()-t0:.0f}s', flush=True)
N = nchunks * a.chunk
key, sub = jax.random.split(key); T = 2 * a.pairs
q, tk = run(jnp.tile(wfs, (T, 1)), sub)
p_adc = np.asarray(digitize(truth, q)).reshape(T, npx, -1)[:, :, :NH]; p_q = np.asarray(q).reshape(T, npx, -1)[:, :, :NH]
p_tk = np.asarray(tk).reshape(T, npx, -1)[:, :, :NH]
np.savez_compressed(a.out, lam=lam[:, :NH], Et=Et[:, :NH], Eadc=Eadc[:, :NH], lam_all=lam, cnt=cnt, st=st, st2=st2, sa=sa, sa2=sa2,
                    tot=tot, N=N, t_sampling=float(truth.t_sampling), pairs_hit=p_q > 0, pairs_tick=p_tk, pairs_adc=p_adc, M=a.pairs,
                    track_len_cm=L, n_pixels=npx)
print(f'wrote {a.out}: N={N}, hits/throw {tot.sum()/N:.1f}, model {lam.sum():.1f} ({100*(lam.sum()/(tot.sum()/N)-1):+.2f}%)', flush=True)
