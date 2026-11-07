"""Shared data pack for the example event (sample 8, batch 35, event 2, main track (2,4), the paper's event display).
Geometry, dE/dx and calibration come from the real saved checkpoints of KN10N8C75 (iter 10000), KN10N8C75FR (14000),
KN10N8C75H (20000). The model prediction at each state comes from the production probabilistic simulation.
'start' is the straight-line input at the initial calibration; the fit's own iteration-0 geometry is hit-seeded and was not saved."""
import os, sys, glob, json, pickle, time, numpy as np
sys.path.insert(0, 'review_2026_09/scripts/eventdisplay')
from rebuild import FitRebuild, load_ckpt
S = '/sdf/scratch/neutrino/pgranger/fit_result_staging'
ck = lambda n: glob.glob(f'{S}/sci_full_{n}/history_iter*.pkl')[0]
B, EV, MAIN = 35, 2.0, (2.0, 4.0)
t0 = time.time()
R = FitRebuild(ck('KN10N8C75H')); print('loaded', time.time() - t0, flush=True)
import jax.numpy as jnp
from larndsim.losses_jax import adc2charge
from optimize.strategies import LUTProbabilisticSimulation
f, col = R.sfields, R.col
tg = R.target(B); m = tg['event'] == int(EV)
data = {k: v[m] for k, v in tg.items()}; data['q'] = np.asarray(adc2charge(jnp.asarray(data['adcs']), R.truth_params))
out = dict(fields=list(f), tfields=list(R.tfields), col=col, batch=B, event=EV, main=MAIN, seed=R.seed, data=data, calib_true=R.calib_true, calib_init=R.calib_init)
PARAMS = list(R.calib_true)
def pack(pred, P):
    hp = np.exp(np.asarray(pred['hit_prob'])); q = np.asarray(adc2charge(pred['adcs_distrib'], P)); upx = np.asarray(pred['unique_pixels']); ok = upx >= 0
    tt = np.arange(hp.shape[2], dtype=float); lam = hp.sum(2); p0 = hp[:, 0, :]; l0 = np.maximum(lam[:, 0], 1e-12)
    m0 = (p0 * tt).sum(1) / l0; s0 = np.sqrt(np.maximum((p0 * tt ** 2).sum(1) / l0 - m0 ** 2, 0))
    return dict(pid=upx[ok], x=np.asarray(pred['pixel_x'])[ok], y=np.asarray(pred['pixel_y'])[ok], nhit=lam.sum(1)[ok], q=(hp * q).sum((1, 2))[ok],
                lam0=lam[ok, 0], t0=m0[ok], t0sd=s0[ok], hit_prob=hp[ok].astype(np.float32), q_tick=q[ok].astype(np.float32))
states = {}
main_ctx = None
for name, run, it in (('start', 'KN10N8C75H', 0), ('iter10000', 'KN10N8C75', 10000), ('iter14000', 'KN10N8C75FR', 14000), ('iter20000', 'KN10N8C75H', 20000)):
    R.ckpt_path = ck(run); R.ck = load_ckpt(R.ckpt_path)
    b = R.batch(B)
    evi = f.index('eventID'); rows = np.where((b['guess'][:, evi] == EV) & (b['guess'][:, col['dx']] > 0))[0]
    ctxs = [c for c in b['ctxs'] if c.track_id[0] == EV]
    main = [c for c in ctxs if tuple(map(float, c.track_id)) == MAIN][0]
    arr = b['guess'] if name == 'start' else b['fitted']
    calib = dict(R.calib_init) if name == 'start' else {p: float(np.ravel(R.ck[p + '_iter'])[-1000:].mean()) for p in PARAMS}
    P = R.truth_params.replace(**calib)
    t1 = time.time(); sim = LUTProbabilisticSimulation(R.response)
    pred = sim.predict(P, jnp.asarray(arr[rows]), f, None); pk = pack(pred, P); print(name, 'predict', round(time.time() - t1, 1), 's', pk['hit_prob'].shape, flush=True)
    fp = R.fp; u0 = fp._dir_from_angles(main.theta0_i, main.phi0_i); e1, e2 = fp._transverse_frame(u0)
    idx = np.asarray(main.idxs); s_cm = np.asarray(main.alpha_mid) * main.total_len; nom = np.asarray(main.x0_fixed)[None, :] + s_cm[:, None] * np.asarray(u0)[None, :]
    xyz = lambda a: a[idx][:, [col['x'], col['y'], col['z']]]
    dev = 1e4 * np.stack([(xyz(arr) - nom) @ np.asarray(e1), (xyz(arr) - nom) @ np.asarray(e2)], 1)
    pid = np.asarray(R.ck['batch_parent_ids'][B])[idx]
    states[name] = dict(iter=it, run=run, calib=calib, rows_tracks=arr[rows].astype(np.float32), pred=pk, dev_um=dev, s_cm=s_cm, dedx_main=arr[idx, f.index('dEdx')],
                        coef=np.asarray(b['coeffs'][tuple(main.track_id)]) if name != 'start' else None)
    if main_ctx is None:
        tid = tuple(map(float, main.track_id)); tp = R.ck['true_positions'][B]; true_pos = np.asarray(tp.get(tid, tp.get(str(tid))))
        tde = np.asarray(R.ck['true_dedx_matched_cache'][B])
        main_ctx = dict(track_id=tid, total_len=float(main.total_len), x0=np.asarray(main.x0_fixed), u0=np.asarray(u0), e1=np.asarray(e1), e2=np.asarray(e2), idx=idx, s_cm=s_cm,
                        true_pos=true_pos, dev_true_um=1e4 * np.stack([(true_pos - nom) @ np.asarray(e1), (true_pos - nom) @ np.asarray(e2)], 1), dedx_true=tde[pid])
        out['event_tracks'] = {tuple(map(float, c.track_id)): float(c.total_len) for c in ctxs}
        out['true_rows'] = np.asarray(b['true'])[np.asarray(b['true'])[:, list(R.tfields).index('eventID')] == EV].astype(np.float32)
    # checkpoint scalar curves are not needed here: see traces.npz
out['states'] = states; out['main_ctx'] = main_ctx
pickle.dump(out, open('/sdf/scratch/neutrino/pgranger/talk_animations/data/hero.pkl', 'wb'), protocol=4)
print('saved', os.path.getsize('/sdf/scratch/neutrino/pgranger/talk_animations/data/hero.pkl') / 1e6, 'MB', time.time() - t0)
print('data hits', len(data['q']), 'q', data['q'].sum(), {k: (s['pred']['nhit'].sum(), s['pred']['q'].sum()) for k, s in states.items()})
