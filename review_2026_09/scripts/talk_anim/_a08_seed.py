"""a08 helper: reproduce the fit's hit seed for sample 8, batch 35 with the PRODUCTION method (FitParams._seed_drift_from_hits
called on a stub holding the batch's rebuilt contexts), validate it against the run's own log line and pos_residual_iter[35],
then evaluate the production probabilistic model at the seed (initial calibration, seeded geometry, flat prior-centre dE/dx,
which is what the fit's first forward pass sees). Saves $T/data/a08_seed.pkl. Env = inner_sci_full_KN10N8C75.sh + C75."""
import os, sys, glob, pickle, time, logging
for k, v in dict(LARND_CHAIN_DRIFT_W='0', LARND_CHAIN_FRAME='drift', LARND_CHAIN_FREEZE_DRIFT='1', LARND_DRIFT_FROM_HITS='1',
                 LARND_FREEZE_STRAIGHT_CL='0.50', LARND_HITS_SEED_E1='1', LARND_SPLINE_KMIN='3', LARND_CHAIN_COARSE_ITER='7500').items():
    os.environ[k] = v
import numpy as np
from rebuild import FitRebuild
S = '/sdf/scratch/neutrino/pgranger/fit_result_staging'; T = '/sdf/scratch/neutrino/pgranger/talk_animations'
B, EV, MAIN = 35, 2.0, (2.0, 4.0)
t0 = time.time()
R = FitRebuild(glob.glob(f'{S}/sci_full_KN10N8C75/history_iter10000*.pkl')[0]); print('loaded', round(time.time() - t0), flush=True)
import jax.numpy as jnp
import optimize.fit_params as fp
_h = logging.StreamHandler(sys.stdout); _h.setFormatter(logging.Formatter('LOG %(message)s')); fp.logger.addHandler(_h); fp.logger.setLevel(logging.INFO)
from larndsim.losses_jax import adc2charge
from optimize.strategies import LUTProbabilisticSimulation
print('flags', fp._DRIFT_FROM_HITS, fp._FREEZE_STRAIGHT_CL, fp._CHAIN_COARSE_ITER, fp._N_EP, fp._SPLINE_KMIN, fp._SEED_DROP_UPPER, flush=True)
b = R.batch(B); ctxs = b['ctxs']; f, col = R.sfields, R.col
print('rebuild checks', b['checks'], flush=True)
tg = R.target(B)
class Stub: pass
s = Stub()
tp_raw = R.ck['true_positions'][B]
s._batch_true_positions = {B: {tuple(map(float, c.track_id)): np.asarray(tp_raw.get(tuple(map(float, c.track_id)), tp_raw.get(str(tuple(map(float, c.track_id)))))) for c in ctxs}}
s._batch_chain_contexts = {B: ctxs}
s._chain_cache = {B: [{'angles': jnp.zeros(2 * fp._spline_K(c.total_len, R.knot_cm), jnp.float32), 'opt_state': None} for c in ctxs]}
s._chain_optimizer = Stub(); s._chain_optimizer.init = lambda a: None
print('tracks in batch', len(ctxs), 'coeff len of main', [len(st['angles']) for c, st in zip(ctxs, s._chain_cache[B]) if tuple(c.track_id) == MAIN], flush=True)
fp.GradientDescentFitter._seed_drift_from_hits(s, B, tg['pixel_x'], tg['pixel_y'], tg['pixel_z'], tg['event'], tg['adcs'], tg['pixel_id'], apply_seed=True)
print('straight set', s._straight_freeze[B], flush=True)
seed = {tuple(c.track_id): np.asarray(st['angles']) for c, st in zip(ctxs, s._chain_cache[B])}
print('main seed coefs', seed[MAIN], 'saved at 10000', b['coeffs'][MAIN], flush=True)

def warp(coefd):
    meta = fp._build_chain_meta_spline(ctxs, R.knot_cm)
    out, _ = fp._chain_warp_spline(jnp.asarray(b['guess']), [jnp.asarray(coefd[tuple(c.track_id)]) for c in ctxs], meta, col)
    return np.asarray(out).copy()
def posres(arr):                                     # same quantity as _compute_pos_residual_batch, on the rows
    r = []
    for c in ctxs:
        tru = s._batch_true_positions[B][tuple(map(float, c.track_id))]; idx = np.asarray(c.idxs)
        r.append(np.linalg.norm(arr[idx][:, [col['x'], col['y'], col['z']]] - tru, axis=1))
    return 1e4 * float(np.concatenate(r).mean())
full = warp(seed)
coarse = {}
for k, a in seed.items():
    n = len(a); K = n // 2; ns = fp._spline_nsine(K); kc = fp._coarse_kc(ns); m = np.ones(n, np.float32); m[kc:ns] = 0; m[K + kc:K + ns] = 0; coarse[k] = a * m
cz = warp(coarse)
zero = {k: 0 * a for k, a in seed.items()}
tr = np.load(f'{T}/data/traces.npz', allow_pickle=True); nm = [str(n) for n in tr['names']]; i = nm.index('KN10N8C75H')
print('pos residual batch-35 (um): straight', round(posres(b['guess']), 1), 'zero-coef warp', round(posres(warp(zero)), 1), 'full seed', round(posres(full), 1), 'seed upper modes dropped', round(posres(cz), 1),
      '| trace at iter 35 (after first Adam step, upper modes zeroed)', 1e4 * float(tr['pos_residual_iter'][i][35]), 'batch', tr['pos_residual_batch'][i][35], flush=True)
print('checkpoint', 'at 10000 mine-from-coeffs', round(posres(b['fitted']), 1), 'trace nearest visit', 1e4 * float(tr['pos_residual_iter'][i][9935]), tr['pos_residual_batch'][i][9935], flush=True)
print('adam first step', 'chain_lr 1e-2: each free coefficient moves by at most ~100 um', flush=True)

# dE/dx at the first forward pass: flat prior centre (_get_or_init_dedx_state); calibration: initial
loc = float(R.ck['config'].dedx_student_loc); print('dedx_student_loc', loc, flush=True)
pid = np.asarray(R.ck['batch_parent_ids'][B]); di, Ei, xi = f.index('dEdx'), f.index('dE'), col['dx']
arr = full.copy(); ded = np.where(pid >= 0, loc, arr[:, di]); arr[:, di] = ded; arr[:, Ei] = ded * arr[:, xi]
evi = f.index('eventID'); rows = np.where((b['guess'][:, evi] == EV) & (b['guess'][:, xi] > 0))[0]
main = [c for c in ctxs if tuple(map(float, c.track_id)) == MAIN][0]
P = R.truth_params.replace(**R.calib_init)
def pack(pred, P):
    hp = np.exp(np.asarray(pred['hit_prob'])); q = np.asarray(adc2charge(pred['adcs_distrib'], P)); upx = np.asarray(pred['unique_pixels']); ok = upx >= 0
    tt = np.arange(hp.shape[2], dtype=float); lam = hp.sum(2); p0 = hp[:, 0, :]; l0 = np.maximum(lam[:, 0], 1e-12)
    m0 = (p0 * tt).sum(1) / l0; s0 = np.sqrt(np.maximum((p0 * tt ** 2).sum(1) / l0 - m0 ** 2, 0))
    return dict(pid=upx[ok], x=np.asarray(pred['pixel_x'])[ok], y=np.asarray(pred['pixel_y'])[ok], nhit=lam.sum(1)[ok], q=(hp * q).sum((1, 2))[ok], lam0=lam[ok, 0], t0=m0[ok], t0sd=s0[ok])
t1 = time.time(); sim = LUTProbabilisticSimulation(R.response)
pred = sim.predict(P, jnp.asarray(arr[rows]), f, None); pk = pack(pred, P); print('predict', round(time.time() - t1), 's', pk['nhit'].sum(), pk['q'].sum(), flush=True)
u0 = fp._dir_from_angles(main.theta0_i, main.phi0_i); e1, e2 = fp._transverse_frame(u0); idx = np.asarray(main.idxs); s_cm = np.asarray(main.alpha_mid) * main.total_len
nom = np.asarray(main.x0_fixed)[None, :] + s_cm[:, None] * np.asarray(u0)[None, :]
dev = lambda A: 1e4 * np.stack([(A[idx][:, [col['x'], col['y'], col['z']]] - nom) @ np.asarray(e1), (A[idx][:, [col['x'], col['y'], col['z']]] - nom) @ np.asarray(e2)], 1)
out = dict(pred=pk, dev_um=dev(arr), dev_um_coarse=dev(cz), s_cm=s_cm, dedx_main=arr[idx, di], calib=dict(R.calib_init), coef=seed[MAIN], coef_coarse=coarse[MAIN],
           straight=(sorted(s._straight_freeze[B]), [tuple(c.track_id) for c in ctxs]), dedx_loc=loc)
pickle.dump(out, open(f'{T}/data/a08_seed.pkl', 'wb')); print('saved', round(time.time() - t0), flush=True)

# diagnostic: what removes the flat first-hit offset at the seed? same seeded geometry and dE/dx, calibration changed
Zp = np.load(f'{T}/data/a08_pack.npz', allow_pickle=True); Dpid, Dt0, tm = Zp['D_pid'], Zp['D_t0'], Zp['tmask']
for name, cal in (('initial', dict(R.calib_init)), ('all true', dict(R.calib_true)), ('only E true', {**R.calib_init, 'eField': R.calib_true['eField']}),
                  ('E, Ab true', {**R.calib_init, 'eField': R.calib_true['eField'], 'Ab': R.calib_true['Ab']}), ('fit@20000', {p: float(Zp['iter20000_calib'][i]) for i, p in enumerate(['Ab', 'eField', 'tran_diff', 'long_diff', 'lifetime'])})):
    Pq = R.truth_params.replace(**cal); sim = LUTProbabilisticSimulation(R.response); pk = pack(sim.predict(Pq, jnp.asarray(arr[rows]), f, None), Pq)
    lk = np.where(np.isin(Dpid, pk['pid']), pk['t0'][np.clip(np.searchsorted(pk['pid'], Dpid), 0, len(pk['pid']) - 1)], np.nan)
    r = (Dt0 - lk)[tm]; print(f'SEEDGEOM calib {name}: first-hit residual mean {np.nanmean(r):.2f} sd {np.nanstd(r):.2f} n {np.isfinite(r).sum()} | charge {pk["q"].sum():.0f}', flush=True)
