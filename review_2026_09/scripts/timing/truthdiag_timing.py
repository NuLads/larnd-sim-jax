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


# ======================= TIMING CLOSURE AT TRUTH (per pixel, per hit index) =======================
# For every pixel and hit index k: model expected tick E[t | hit k] from the probabilistic FEE vs the observed
# tick of the k-th stochastic target hit, same true tracks, same true parameters. Residual in ticks and um
# (1 tick = t_sampling * vdrift), binned by drift distance. A coherent offset of ~0.1-0.3 tick would explain
# the free fits' coherent +25..60 um drift shift.
VD = 0.1587; TS = float(truth.t_sampling)                  # cm/us, us/tick
UM_PER_TICK = TS * VD * 1e4
recs = []
for i in range(args.b0, args.b1):
    tracks = jnp.asarray(ds_tgt[i])
    wfs, up = simulate_wfs(truth, response, tracks, fields)
    pred = sim.predict(truth, tracks, fields, None)
    lp = np.asarray(pred['hit_prob']); upx = np.asarray(pred['unique_pixels'])
    pr = np.exp(lp); pr[upx < 0] = 0.0
    Ntk = pr.shape[2]; tk = np.arange(Ntk)
    psum = pr.sum(axis=2)                                    # (Npix, Nval) P(hit k exists)
    tmean = (pr * tk[None, None, :]).sum(axis=2) / np.maximum(psum, 1e-12)
    tvar = (pr * (tk[None, None, :] - tmean[..., None]) ** 2).sum(axis=2) / np.maximum(psum, 1e-12)
    cdf = np.cumsum(pr, axis=2) / np.maximum(psum[..., None], 1e-12)      # model CDF of the k-th hit tick
    order = np.argsort(upx); ups = upx[order]
    for s in range(args.nseeds):
        tg = target_hits(truth, wfs, up, (i + 1) + 100003 * s)
        pid = np.asarray(tg['pixel_id']); tt = np.asarray(tg['ticks']).astype(float); z = np.asarray(tg['pixel_z'])
        ok = pid >= 0
        pid, tt, z = pid[ok], tt[ok], z[ok]
        o = np.lexsort((tt, pid)); pid, tt, z = pid[o], tt[o], z[o]
        k = np.zeros(len(pid), int)
        for j in range(1, len(pid)):
            k[j] = k[j - 1] + 1 if pid[j] == pid[j - 1] else 0
        idx = np.searchsorted(ups, pid); idx = np.clip(idx, 0, len(ups) - 1)
        good = (ups[idx] == pid) & (k < pr.shape[1])
        pix = order[idx[good]]; kk = k[good]
        ti = np.clip(tt[good].astype(int), 0, Ntk - 1)
        c_hi = cdf[pix, kk, ti]; c_lo = np.where(ti > 0, cdf[pix, kk, np.maximum(ti - 1, 0)], 0.0)
        pit = 0.5 * (c_lo + c_hi)                                            # mid-PIT (discrete ticks)
        tmed = np.argmax(cdf[pix, kk, :] >= 0.5, axis=1).astype(float)
        recs.append(np.stack([np.full(good.sum(), i), np.full(good.sum(), s), kk, tt[good], tmean[pix, kk], np.sqrt(tvar[pix, kk]), psum[pix, kk], z[good], pit, tmed], 1))
    print(f'[batch {i}] pixels {int((upx>=0).sum())}  hits(seed0) {len(pid)}', flush=True)
R = np.concatenate(recs)
np.savez(args.out, rec=R, um_per_tick=UM_PER_TICK, cols='batch,seed,k,t_obs,t_exp,t_sd,p_exist,pixel_z,pit,t_med')
# summary: first hits with P(exist) > 0.95, residual by drift distance (from pixel_z, anodes at +-30.58975)
D = 30.58975 - np.abs(R[:, 7]); r = R[:, 3] - R[:, 4]
for lab, m in (('k=0, P>0.95', (R[:, 2] == 0) & (R[:, 6] > 0.95)), ('k=1, P>0.95', (R[:, 2] == 1) & (R[:, 6] > 0.95)), ('all k', R[:, 6] > 0)):
    print(f'== {lab}: n={m.sum()}  mean resid {r[m].mean():+.4f} +- {r[m].std(ddof=1)/np.sqrt(m.sum()):.4f} ticks  = {r[m].mean()*UM_PER_TICK:+.1f} um')
    for lo in range(0, 30, 4):
        mm = m & (D >= lo) & (D < lo + 4)
        if mm.sum() > 50:
            print(f'     D {lo:2d}-{lo+4:2d} cm: n={mm.sum():6d}  resid {r[mm].mean():+.4f} +- {r[mm].std(ddof=1)/np.sqrt(mm.sum()):.4f} ticks  ({r[mm].mean()*UM_PER_TICK:+6.1f} um)')
