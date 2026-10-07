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
p.add_argument('--ab_grid', default='0.994,0.997,1.0,1.003,1.006')
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
loss_nq = ProbabilisticLossStrategy(loss_fn=llhd_loss, sigma_charge=1e5)   # charge information ~removed (100 ke width; keeps joint log-prob far from the -18.4 clamp)

lt_grid = [float(x) for x in args.lt_grid.split(',')]
ld_grid = [float(x) for x in args.ld_grid.split(',')]


def target_hits(params, wfs, up, seed):
    adcs, x, y, z, ticks, hp, ev, pix = simulate_stochastic(params, wfs, up, rngseed=seed)
    return dict(adcs=adcs, pixel_x=x, pixel_y=y, pixel_z=z, ticks=ticks, hit_prob=hp, event=ev, pixel_id=pix)


def expected_per_tick(pred, params):
    lp = pred['hit_prob']                               # (Npix, Nval, Ntk)
    valid = (pred['unique_pixels'] >= 0)[:, None, None]
    pr = jnp.where(valid, jnp.exp(lp), 0.0)
    q = adc2charge(pred['adcs_distrib'], params)        # ke-
    return np.asarray(pr.sum(axis=(0, 1))), np.asarray((pr * q).sum(axis=(0, 1)))


def observed_per_tick(tg, params, ntk):
    t = np.asarray(tg['ticks']).astype(int)
    q = np.asarray(adc2charge(tg['adcs'], params))
    ok = (t >= 0) & (t < ntk)
    return (np.bincount(t[ok], minlength=ntk)[:ntk].astype(np.float64),
            np.bincount(t[ok], weights=q[ok], minlength=ntk)[:ntk])


def nll(params, pred, tg, L=None):
    v, aux = (L or loss).compute(params, pred, tg)
    return float(v), {k: float(aux[k]) for k in ('log_likelihood_tick', 'no_match_penalty',
                                                 'expected_total_hits', 'matched_hits')}



ab_grid = [float(x) for x in args.ab_grid.split(',')]
TICK0, TS = 22.4, 0.1
NB = 19; EDG = np.round(np.linspace(0, 190, NB + 1) / TS + TICK0).astype(int)

def cond_charge(params, pred, tg):
    """For each target hit: model expected charge at the observed tick (posterior-weighted over slots
    and the +-2 tick window exactly as the loss's time kernel, WITHOUT the charge term), and the
    observed charge. Returns per-drift-bin sums of (q_obs - q_model), q_obs, count."""
    tpid = tg['pixel_id']; up_ = pred['unique_pixels']
    idx = jnp.clip(jnp.searchsorted(up_, tpid), 0, up_.shape[0] - 1)
    ok = (up_[idx] == tpid) & (tpid >= 0)
    lp = pred['hit_prob']; ad = pred['adcs_distrib']; nt = lp.shape[-1]; nsl = lp.shape[1]
    tt = tg['ticks'].astype(int)
    d = jnp.arange(-loss.time_window, loss.time_window + 1)
    wt = jnp.clip(tt[:, None] + d[None, :], 0, nt - 1)
    wlp = lp[idx[:, None, None], jnp.arange(nsl)[None, :, None], wt[:, None, :]] + loss.log_time_weights[None, None, :]
    wq = adc2charge(ad[idx[:, None, None], jnp.arange(nsl)[None, :, None], wt[:, None, :]], params)
    w = jax.nn.softmax(wlp.reshape(wlp.shape[0], -1), axis=1).reshape(wlp.shape)
    qm = jnp.sum(w * wq, axis=(1, 2))
    qo = adc2charge(tg['adcs'], params)
    tt_, qm_, qo_, ok_ = np.asarray(tt), np.asarray(qm), np.asarray(qo), np.asarray(ok)
    b = np.digitize(tt_, EDG) - 1
    m = ok_ & (b >= 0) & (b < NB)
    return np.stack([np.bincount(b[m], weights=(qo_ - qm_)[m], minlength=NB)[:NB],
                     np.bincount(b[m], weights=((qo_ - qm_) ** 2)[m], minlength=NB)[:NB],
                     np.bincount(b[m], weights=qo_[m], minlength=NB)[:NB],
                     np.bincount(b[m], minlength=NB)[:NB].astype(float)])

rec = {k: [] for k in ['batch', 'nll2_on', 'nll2_off', 'nll2nq_on', 'nll2nq_off', 'cq_on', 'cq_off', 'cq_off_tiny']}
for i in range(args.b0, min(args.b1, len(ds_tgt))):
    tb = time.time()
    tracks = jnp.asarray(np.asarray(ds_tgt[i]).reshape(-1, len(fields)))
    wfs, up = simulate_wfs(truth, response, tracks, fields)
    tg_on = [target_hits(truth, wfs, up, (i + 1) + 100003 * s) for s in range(args.nseeds)]
    tg_off = target_hits(truth_nonoise, wfs, up, i + 1)
    pred = sim.predict(truth, tracks, fields, None)
    pred_t = sim.predict(truth_tiny, tracks, fields, None)
    rec['cq_on'].append(np.array([cond_charge(truth, pred, t) for t in tg_on]))
    rec['cq_off'].append(cond_charge(truth, pred, tg_off))
    rec['cq_off_tiny'].append(cond_charge(truth_tiny, pred_t, tg_off))
    del pred_t
    A_on = np.zeros((len(ab_grid), len(lt_grid), args.nseeds)); A_off = np.zeros((len(ab_grid), len(lt_grid)))
    B_on = np.zeros_like(A_on); B_off = np.zeros_like(A_off)
    for ia, fa in enumerate(ab_grid):
        for il, fl in enumerate(lt_grid):
            pp = truth.replace(Ab=TRUTH['Ab'] * fa, lifetime=TRUTH['lifetime'] * fl)
            pr = pred if (fa == 1.0 and fl == 1.0) else sim.predict(pp, tracks, fields, None)
            for s, t in enumerate(tg_on):
                A_on[ia, il, s] = nll(pp, pr, t)[0]; B_on[ia, il, s] = nll(pp, pr, t, loss_nq)[0]
            A_off[ia, il] = nll(pp, pr, tg_off)[0]; B_off[ia, il] = nll(pp, pr, tg_off, loss_nq)[0]
            del pr
    rec['batch'].append(i); rec['nll2_on'].append(A_on); rec['nll2_off'].append(A_off)
    rec['nll2nq_on'].append(B_on); rec['nll2nq_off'].append(B_off)
    c = rec['cq_on'][-1].sum(0)
    print(f'[BATCH {i}] {time.time()-tb:.1f}s  cond. charge resid (on) mean {c[0].sum()/c[3].sum():+.4f} ke/hit', flush=True)
    np.savez(args.out, ab_grid=np.array(ab_grid), lt_grid=np.array(lt_grid), edges_tick=EDG,
             truth=json.dumps(TRUTH), **{k: np.array(v) for k, v in rec.items()})
print(f'[DONE] batches {rec["batch"]} total {time.time()-t_start:.1f}s', flush=True)
