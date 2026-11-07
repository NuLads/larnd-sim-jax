"""CPU, offline, descriptive: which tracks the hits call straight (e2 frozen at 0, LARND_FREEZE_STRAIGHT_CL) at several
levels, against their TRUE drift-direction shape (rms of the e2 truth projection) and their KN10S1 @ 1e4 outcome.
A track called straight keeps e2 = 0 for the whole fit, so its true e2 shape is its floor in the drift direction."""
import os, sys
for k, v in dict(LARND_DRIFT_FROM_HITS='1', LARND_HITS_SEED_E1='1', LARND_FREEZE_STRAIGHT_CL='0.50', LARND_CHAIN_FREEZE_DRIFT='1',
                 LARND_CHAIN_FRAME='drift', LARND_SPLINE_KMIN='3').items(): os.environ[k] = v
sys.path.insert(0, '/sdf/group/neutrino/pgranger/larnd-sim-jax/review_2026_09/scripts/eventdisplay')
import glob, json, types, logging, numpy as np, optax, jax.numpy as jnp
logging.disable(logging.INFO)
from rebuild import FitRebuild
D = '/sdf/scratch/neutrino/pgranger/review_2026_09/POLISH/joint'
R = FitRebuild(sorted(glob.glob('/sdf/group/neutrino/pgranger/larnd-sim-jax/fit_result/sci_full_KN10S1/history_iter10000_*.pkl'))[0])
fp, ck = R.fp, R.ck
o = {(x['batch'], round(x['L'], 3)): x for x in json.load(open(f'{D}/per_track_KN10S1_10000.json'))}
CLS = (0.05, 0.1, 0.2, 0.3, 0.5, 0.7)   # called straight if chi2 p-value of the e2 modes > 1 - CL: LOWER CL = fewer tracks frozen
rows = []
for i in range(100):
    bt = R.batch(i, check=False); ctxs = bt['ctxs']; tp = ck['true_positions'][i]; tg = R.target(i); F = {}
    for cl in CLS:
        fp._FREEZE_STRAIGHT_CL = cl
        obj = types.SimpleNamespace(_batch_chain_contexts={i: ctxs}, _chain_optimizer=optax.adam(1e-2), _batch_true_positions={i: {}},
                                    _chain_cache={i: [{'angles': jnp.zeros(2 * fp._spline_K(c.total_len, R.knot_cm), jnp.float32)} for c in ctxs]})
        fp.GradientDescentFitter._seed_drift_from_hits(obj, i, tg['pixel_x'], tg['pixel_y'], tg['pixel_z'], tg['event'], tg['adcs'], tg['pixel_id'], apply_seed=True)
        F[cl] = obj._straight_freeze[i]
    fp._FREEZE_STRAIGHT_CL = 0.5
    for ti, c in enumerate(ctxs):
        tr = tp.get(tuple(map(float, c.track_id)), tp.get(str(tuple(map(float, c.track_id)))))
        if tr is None: continue
        K = fp._spline_K(c.total_len, R.knot_cm); u0 = fp._dir_from_angles(c.theta0_i, c.phi0_i); e1, e2 = fp._transverse_frame(u0)
        L = float(c.total_len); al = np.asarray(c.alpha_mid, float); tr = np.asarray(tr, float); m = min(len(al), len(tr)); al, tr = al[:m], tr[:m]
        d = tr - (np.asarray(c.x0_fixed)[None, :] + al[:, None] * L * u0[None, :]); B = fp._spline_basis(al, K)
        t2 = np.linalg.lstsq(B, d @ e2, rcond=None)[0]; x = o[(i, round(L, 3))]
        rows.append(dict(batch=i, L=L, true_e2_um=1e4 * float(np.sqrt(np.mean((B @ t2) ** 2))), r_z=x['r_z'], r_mean=x['r_mean'],
                         frozen_fit=bool(x['frozen']), **{f'frozen_{cl}': ti in F[cl] for cl in CLS}))
    print(f'batch {i}: {len(rows)} tracks', flush=True)
json.dump(rows, open(f'{D}/freeze_cl_KN10S1.json', 'w'), indent=1)
a = {k: np.array([r[k] for r in rows], float) for k in rows[0]}
print(f'{len(rows)} tracks; frozen in the fit {int(a["frozen_fit"].sum())}, replay at 0.50 {int(a["frozen_0.5"].sum())} '
      f'(agree {int(np.sum(a["frozen_fit"] == a["frozen_0.5"]))}); true e2 shape median {np.median(a["true_e2_um"]):.0f} um')
for cl in CLS:
    f = a[f'frozen_{cl}'] > 0; t = a['true_e2_um']
    print(f'  CL {cl:.2f}: frozen {f.sum():4d}; true e2 shape of the frozen median {np.median(t[f]):4.0f} um, > 300 um {np.sum(f & (t > 300)):3d}, '
          f'> 500 um {np.sum(f & (t > 500)):3d}; KN10S1 outliers among them {np.sum(f & (a["r_mean"] > 200)):3d}, r_z > 200 {np.sum(f & (a["r_z"] > 200)):3d}')
