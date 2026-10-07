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
p.add_argument('--drift_check', type=int, default=1)
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


def nll(params, pred, tg):
    v, aux = loss.compute(params, pred, tg)
    return float(v), {k: float(aux[k]) for k in ('log_likelihood_tick', 'no_match_penalty',
                                                 'expected_total_hits', 'matched_hits')}


# ---------------------------------------------------------------- sanity 2: tick <-> drift time
drift_check = None
if args.drift_check:
    tr0 = np.asarray(ds_tgt[args.b0]).reshape(-1, len(fields))
    iz, izs, ize = fields.index('z'), fields.index('z_start'), fields.index('z_end')
    ne = fields.index('n_electrons') if 'n_electrons' in fields else None
    row = tr0[np.argmax(tr0[:, fields.index('dE')])].copy()
    plane = int(row[fields.index('pixel_plane')]) if 'pixel_plane' in fields else 0
    borders = np.asarray(truth.tpc_borders)
    res = []
    for pl in range(borders.shape[0]):
        z_an, z_ca = borders[pl, 2, 0], borders[pl, 2, 1]
        sgn = np.sign(z_ca - z_an)
        for d in [1., 3., 6., 10., 15., 20., 25., 29.]:
            r = row.copy()
            zc = z_an + sgn * d
            r[iz] = zc; r[izs] = zc - 0.005; r[ize] = zc + 0.005
            # put x/y safely inside this plane's tile
            ix, iy = fields.index('x'), fields.index('y')
            xc, yc = borders[pl, 0].mean(), borders[pl, 1].mean()
            for k, c in (('x', xc), ('y', yc)):
                for suf in ('', '_start', '_end'):
                    r[fields.index(k + suf)] = c
            tt = jnp.asarray(r[None, :])
            w, u = simulate_wfs(truth, response, tt, fields)
            w = np.asarray(w)
            if w.size == 0 or w.sum() == 0:
                continue
            s = w.sum(axis=0)
            qc = np.cumsum(s)
            t50 = int(np.searchsorted(qc, 0.5 * qc[-1]))
            res.append((pl, d, float(d / float(jax.device_get(__import__('larndsim.drifting_jax', fromlist=['get_vdrift']).get_vdrift(truth)))),
                        int(np.argmax(s)), t50))
    drift_check = np.array(res, dtype=float)
    print('[DRIFT-CHECK] plane, d[cm], d/v[us], argmax tick, median-charge tick\n', drift_check, flush=True)

# ---------------------------------------------------------------- main loop
out = dict(lt_grid=np.array(lt_grid), ld_grid=np.array(ld_grid), truth=json.dumps(TRUTH),
           drift_check=drift_check if drift_check is not None else np.zeros((0, 5)))
rec = {k: [] for k in ['batch', 'exp_n', 'exp_q', 'exp_n_tiny', 'exp_q_tiny', 'obs_n_on', 'obs_q_on',
                       'obs_n_off', 'obs_q_off', 'exp_q_lt', 'exp_n_lt', 'exp_q_ld', 'exp_n_ld',
                       'nll_lt_on', 'nll_lt_off', 'nll_ld_on', 'nll_ld_off', 'aux_lt_on', 'aux_lt_off',
                       'nll_lt_on_tinymodel_off', 'frac_tgt_pix_in_guess', 'n_track_rows', 'npix']}

for i in range(args.b0, min(args.b1, len(ds_tgt))):
    tb = time.time()
    tracks = jnp.asarray(np.asarray(ds_tgt[i]).reshape(-1, len(fields)))
    wfs, up = simulate_wfs(truth, response, tracks, fields)
    ntk = wfs.shape[1] - 1           # probabilistic tick axis length (Nticks-1)
    # targets: production uses rngkey=i+1 (seed index 0); others offset
    tg_on = [target_hits(truth, wfs, up, (i + 1) + 100003 * s) for s in range(args.nseeds)]
    tg_off = target_hits(truth_nonoise, wfs, up, i + 1)

    pred = sim.predict(truth, tracks, fields, None)
    en, eq = expected_per_tick(pred, truth)
    pred_tiny = sim.predict(truth_tiny, tracks, fields, None)
    en_t, eq_t = expected_per_tick(pred_tiny, truth_tiny)

    on = [observed_per_tick(t, truth, ntk) for t in tg_on]
    off = observed_per_tick(tg_off, truth, ntk)

    # sanity 3: every target hit pixel exists in the guess pixel list
    gp = np.asarray(pred['unique_pixels'])
    tp_all = np.concatenate([np.asarray(t['pixel_id']) for t in tg_on] + [np.asarray(tg_off['pixel_id'])])
    frac = float(np.isin(tp_all, gp[gp >= 0]).mean()) if tp_all.size else np.nan

    nll_lt_on, nll_lt_off, aux_on, aux_off, eq_lt, en_lt, nll_tiny_off = [], [], [], [], [], [], []
    for f in lt_grid:
        pp = truth.replace(lifetime=TRUTH['lifetime'] * f)
        pr = pred if f == 1.0 else sim.predict(pp, tracks, fields, None)
        a, b = expected_per_tick(pr, pp)
        en_lt.append(a); eq_lt.append(b)
        r_on = [nll(pp, pr, t) for t in tg_on]
        nll_lt_on.append([r[0] for r in r_on]); aux_on.append([[r[1][k] for k in sorted(r[1])] for r in r_on])
        r_off = nll(pp, pr, tg_off)
        nll_lt_off.append(r_off[0]); aux_off.append([r_off[1][k] for k in sorted(r_off[1])])
        # tiny-noise model vs noise-off target (is the model consistent in its noiseless limit?)
        ppt = truth_tiny.replace(lifetime=TRUTH['lifetime'] * f)
        prt = pred_tiny if f == 1.0 else sim.predict(ppt, tracks, fields, None)
        nll_tiny_off.append(nll(ppt, prt, tg_off)[0])
        del pr, prt
    nll_ld_on, nll_ld_off, eq_ld, en_ld = [], [], [], []
    for f in ld_grid:
        pp = truth.replace(long_diff=TRUTH['long_diff'] * f)
        pr = pred if f == 1.0 else sim.predict(pp, tracks, fields, None)
        a, b = expected_per_tick(pr, pp)
        en_ld.append(a); eq_ld.append(b)
        nll_ld_on.append([nll(pp, pr, t)[0] for t in tg_on])
        nll_ld_off.append(nll(pp, pr, tg_off)[0])
        del pr

    rec['batch'].append(i); rec['exp_n'].append(en); rec['exp_q'].append(eq)
    rec['exp_n_tiny'].append(en_t); rec['exp_q_tiny'].append(eq_t)
    rec['obs_n_on'].append(np.array([o[0] for o in on])); rec['obs_q_on'].append(np.array([o[1] for o in on]))
    rec['obs_n_off'].append(off[0]); rec['obs_q_off'].append(off[1])
    rec['exp_q_lt'].append(np.array(eq_lt)); rec['exp_n_lt'].append(np.array(en_lt))
    rec['exp_q_ld'].append(np.array(eq_ld)); rec['exp_n_ld'].append(np.array(en_ld))
    rec['nll_lt_on'].append(np.array(nll_lt_on)); rec['nll_lt_off'].append(np.array(nll_lt_off))
    rec['nll_ld_on'].append(np.array(nll_ld_on)); rec['nll_ld_off'].append(np.array(nll_ld_off))
    rec['aux_lt_on'].append(np.array(aux_on)); rec['aux_lt_off'].append(np.array(aux_off))
    rec['nll_lt_on_tinymodel_off'].append(np.array(nll_tiny_off))
    rec['frac_tgt_pix_in_guess'].append(frac)
    rec['n_track_rows'].append(int(tracks.shape[0])); rec['npix'].append(int((gp >= 0).sum()))

    Qon = np.mean([o[1].sum() for o in on]); Qoff = off[1].sum()
    print(f'[BATCH {i}] {time.time()-tb:.1f}s npix={int((gp>=0).sum())} ntk={ntk} '
          f'hits on/exp={np.mean([o[0].sum() for o in on]):.1f}/{en.sum():.1f} '
          f'Q on/exp={Qon:.1f}/{eq.sum():.1f} ({Qon/eq.sum()-1:+.4%}) | '
          f'noise-off: hits {off[0].sum():.0f} vs tiny-model {en_t.sum():.1f}, '
          f'Q {Qoff:.1f} vs tiny {eq_t.sum():.1f} ({Qoff/eq_t.sum()-1:+.4%}) pixfrac={frac:.4f}', flush=True)
    print(f'   nll_lt_on[seed0]={np.round(np.array(nll_lt_on)[:,0]-np.array(nll_lt_on)[:,0].min(),2).tolist()}'
          f'  nll_lt_off={np.round(np.array(nll_lt_off)-min(nll_lt_off),2).tolist()}', flush=True)

    np.savez(args.out, **out, **{k: np.array(v) for k, v in rec.items()})

print(f'[DONE] batches {rec["batch"]} total {time.time()-t_start:.1f}s', flush=True)
