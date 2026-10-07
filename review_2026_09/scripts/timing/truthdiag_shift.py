"""
Truth-point closure diagnostic for the probabilistic FEE model vs the stochastic target.

Everything (calibration, dE/dx, geometry) at truth: guess and target are simulated from the SAME
target-file track batches with the SAME physical parameters. Only the FEE stage differs:
  target = stochastic get_adc_values (reset+uncorrelated noise, threshold)   [LUTSimulation]
  guess  = probabilistic get_adc_values_average_noise_*                     [LUTProbabilisticSimulation]
Loss = ProbabilisticLossStrategy (PPP NLL) with production defaults (llhd).

Outputs per batch (npz): per-tick observed/expected hit counts & charge (E1), and NLL on
lifetime / long_diff grids for every target realisation (E2).
Imports the library read-only from the repo; modifies nothing there.
"""
import os, sys, time, argparse, json
import numpy as np

REPO = '/sdf/group/neutrino/pgranger/larnd-sim-jax'
sys.path.insert(0, REPO + '/src'); sys.path.insert(0, REPO)
os.chdir(REPO)

import jax, jax.numpy as jnp
jax.config.update('jax_platform_name', 'gpu')

from optimize.dataio import TracksDataset, TgtTracksDataset, DataLoader
from optimize.strategies import LUTProbabilisticSimulation, ProbabilisticLossStrategy
from larndsim.consts_jax import build_params_class, load_detector_properties, load_lut
from larndsim.sim_jax import simulate_wfs, simulate_stochastic
from larndsim.losses_jax import adc2charge, llhd_loss

p = argparse.ArgumentParser()
p.add_argument('--b0', type=int, default=0)
p.add_argument('--b1', type=int, default=2)
p.add_argument('--nseeds', type=int, default=10)
p.add_argument('--lt_grid', default='0.85,0.8875,0.925,0.9625,1.0,1.0375,1.075,1.1125,1.15')
p.add_argument('--ld_grid', default='0.7,0.8,0.9,1.0,1.1,1.2,1.3')
p.add_argument('--out', required=True)
p.add_argument('--max_nbatch', type=int, default=100)
p.add_argument('--tiny_noise', type=float, default=1.0,
               help='noise [e-] used for the "noise-off" limit of the probabilistic model (sanity 1)')
p.add_argument('--drift_check', type=int, default=0)
args = p.parse_args()

# ---------------------------------------------------------------- production configuration
INPUT_SIM = '/sdf/group/neutrino/pgranger/lads-data/linear_guess_segments.h5'
INPUT_TGT = '/sdf/data/neutrino/cyifan/diffsim_input/true_through_muon_edep_10cm_vol1cm.h5'
TRUTH = dict(Ab=0.83488135, eField=0.52151894, tran_diff=1.00276338e-05,
             long_diff=5.81418228e-06, lifetime=2406.44659703)   # production seed-0 targets
PARAMS = ['Ab', 'eField', 'tran_diff', 'long_diff', 'lifetime']

t_start = time.time()
ds_sim = TracksDataset(filename=INPUT_SIM, nevents=600, max_nbatch=args.max_nbatch, random_nevents=True,
                       data_seed=1, track_len_sel=2., max_abs_costheta_sel=0.966, min_abs_segz_sel=15.,
                       track_z_bound=28., max_batch_len=400, print_input=False, chopped=True, pad=True,
                       electron_sampling_resolution=0.01, live_selection=False,
                       use_dedx_density=False, dedx_density_mode='histogram')
ds_tgt = TgtTracksDataset(filename=INPUT_TGT, dataset_sim=ds_sim, chopped=True, pad=True,
                          electron_sampling_resolution=0.01, print_input=False)
assert len(ds_sim) == len(ds_tgt)
fields = ds_tgt.get_track_fields()
print(f'[DATA] {len(ds_tgt)} batches, fields={fields}, load {time.time()-t_start:.1f}s', flush=True)

Params = build_params_class(PARAMS)
ref = load_detector_properties(Params, 'src/larndsim/detector_properties/module0.yaml',
                               'src/larndsim/pixel_layouts/multi_tile_layout-2.4.16_v4.yaml')
ref = ref.replace(electron_sampling_resolution=0.01, number_pix_neighbors=4, signal_length=150, time_window=150)
response, ref = load_lut('src/larndsim/detector_properties/response_44.npy', ref)
ref = ref.replace(diffusion_in_current_sim=False, mc_diff=False, use_dedx_density=False, dedx_density_mode='histogram')
truth = ref.replace(**TRUTH)
truth_nonoise = truth.replace(RESET_NOISE_CHARGE=0., UNCORRELATED_NOISE_CHARGE=0.)
truth_tiny = truth.replace(RESET_NOISE_CHARGE=float(args.tiny_noise), UNCORRELATED_NOISE_CHARGE=float(args.tiny_noise))
print('[PARAMS] truth', {k: float(getattr(truth, k)) for k in PARAMS},
      'noise', truth.RESET_NOISE_CHARGE, truth.UNCORRELATED_NOISE_CHARGE,
      'thr', truth.DISCRIMINATION_THRESHOLD, 't_sampling', truth.t_sampling,
      'MAX_ADC_VALUES', truth.MAX_ADC_VALUES, 'time_interval', truth.time_interval, flush=True)

sim = LUTProbabilisticSimulation(response)
loss = ProbabilisticLossStrategy(loss_fn=llhd_loss)   # production: loss_fn_kw = {} -> defaults


def target_hits(params, wfs, up, seed):
    adcs, x, y, z, ticks, hp, ev, pix = simulate_stochastic(params, wfs, up, rngseed=seed)
    return dict(adcs=adcs, pixel_x=x, pixel_y=y, pixel_z=z, ticks=ticks, hit_prob=hp, event=ev, pixel_id=pix)



# ============ SHIFT SCAN: where does each likelihood form put the timing optimum? (true tracks, true params) ============
# For hits drawn (a) from the probabilistic model itself and (b) from the stochastic FEE, shift the model's per-tick
# hit and charge distributions by delta (fractional ticks, linear interpolation) and evaluate:
#   TW0: log p(t_obs - delta) + log N(q_obs | q(t_obs - delta), 0.5 ke)                    [exact point-wise]
#   TW2: logsumexp_{d=-2..2} [log p(t_obs+d - delta) + log w_d + log N(q_obs | q(t_obs+d - delta), 0.5 ke)]  [production]
# delta* = argmax. A correctly specified likelihood gives delta* = 0 on self-consistent hits.
from optimize.strategies import LUTProbabilisticSamplingSimulation
sampler = LUTProbabilisticSamplingSimulation(response)
DELTAS = np.round(np.arange(-0.6, 0.61, 0.1), 2)
w = np.exp(-0.5 * (np.arange(-2, 3) / 1.0) ** 2); w /= w.sum(); logw = np.log(w)
SIGQ = 0.5
EDGES = [0, 8, 16, 24, 32]
acc = {(src_, form): np.zeros((len(EDGES) - 1, len(DELTAS))) for src_ in ('selfcons', 'stoch') for form in ('TW0', 'TW2')}
nh = {src_: np.zeros(len(EDGES) - 1) for src_ in ('selfcons', 'stoch')}
def occ(pid):                                        # occurrence index of each hit within its pixel, in array order
    k = np.zeros(len(pid), int); seen = {}
    for j, p in enumerate(pid):
        k[j] = seen.get(p, 0); seen[p] = k[j] + 1
    return k
for i in range(args.b0, args.b1):
    tracks = jnp.asarray(ds_tgt[i])
    wfs, up = simulate_wfs(truth, response, tracks, fields)
    pred = sim.predict(truth, tracks, fields, None)
    lp = np.asarray(pred['hit_prob'], np.float64); upx = np.asarray(pred['unique_pixels'])
    qd = np.asarray(adc2charge(pred['adcs_distrib'], truth), np.float64)
    P = np.exp(lp); P[upx < 0] = 0.0; Ntk = P.shape[2]
    order = np.argsort(upx); ups = upx[order]
    for s in range(args.nseeds):
        hitsets = {}
        sc = sampler.predict(truth, tracks, fields, (i + 1) + 100003 * s)
        hitsets['selfcons'] = (np.asarray(sc['hit_pixels']), np.asarray(sc['ticks']).astype(int), np.asarray(adc2charge(sc['adcs'], truth)), np.asarray(sc['pixel_z']))
        tg = target_hits(truth, wfs, up, (i + 1) + 100003 * s)
        hitsets['stoch'] = (np.asarray(tg['pixel_id']), np.asarray(tg['ticks']).astype(int), np.asarray(adc2charge(tg['adcs'], truth)), np.asarray(tg['pixel_z']))
        for src_, (pid, tt, qq, zz) in hitsets.items():
            ok = (pid >= 0) & (tt >= 3) & (tt < Ntk - 3)
            k = occ(pid)
            idx = np.clip(np.searchsorted(ups, pid), 0, len(ups) - 1)
            good = ok & (ups[idx] == pid) & (k < P.shape[1])
            pix = order[idx[good]]; kk = k[good]; t0 = tt[good]; q0 = qq[good]
            D = 30.58975 - np.abs(zz[good]); bi = np.clip(np.digitize(D, EDGES) - 1, 0, len(EDGES) - 2)
            for b in range(len(EDGES) - 1): nh[src_][b] += (bi == b).sum()
            for jd, dl in enumerate(DELTAS):
                fl = int(np.floor(-dl)); f = -dl - fl            # model evaluated at t - dl = t + fl + f
                def ev(tq):
                    a = np.clip(tq + fl, 0, Ntk - 1); b_ = np.clip(tq + fl + 1, 0, Ntk - 1)
                    p = (1 - f) * P[pix, kk, a] + f * P[pix, kk, b_]
                    q = (1 - f) * qd[pix, kk, a] + f * qd[pix, kk, b_]
                    return p, q
                p0, qm0 = ev(t0)
                s0 = np.log(np.maximum(p0, 1e-300)) - 0.5 * ((q0 - qm0) / SIGQ) ** 2
                terms = []
                for jj, d in enumerate(range(-2, 3)):
                    p_, q_ = ev(t0 + d)
                    terms.append(np.log(np.maximum(p_, 1e-300)) + logw[jj] - 0.5 * ((q0 - q_) / SIGQ) ** 2)
                s2 = np.logaddexp.reduce(np.stack(terms, 1), axis=1)
                for b in range(len(EDGES) - 1):
                    m = bi == b
                    acc[(src_, 'TW0')][b, jd] += s0[m].sum(); acc[(src_, 'TW2')][b, jd] += s2[m].sum()
    print(f'[batch {i}] done', flush=True)
def argmax_parab(L):
    j = int(np.argmax(L)); j = min(max(j, 1), len(L) - 2)
    y0, y1, y2 = L[j - 1], L[j], L[j + 1]; den = (y0 - 2 * y1 + y2)
    return DELTAS[j] + (0.1 * 0.5 * (y0 - y2) / den if den != 0 else 0.0)
print('delta* (ticks; + = model must be shifted LATER to match hits; 1 tick = 158.7 um)')
for src_ in ('selfcons', 'stoch'):
    for form in ('TW0', 'TW2'):
        L = acc[(src_, form)]
        print(f'  {src_:8s} {form}: all-drift delta* {argmax_parab(L.sum(0)):+.3f} | by drift ' +
              '  '.join(f'{EDGES[b]}-{EDGES[b+1]}cm: {argmax_parab(L[b]):+.3f}' for b in range(len(EDGES) - 1)) + f'   (hits/bin {nh[src_].astype(int).tolist()})')
np.savez(args.out, deltas=DELTAS, edges=EDGES, **{f'{a}_{b}': v for (a, b), v in acc.items()})
