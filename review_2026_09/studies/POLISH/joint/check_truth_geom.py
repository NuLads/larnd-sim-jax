"""CPU check of joint_polish2.py --geom truth: per batch, mean distance to the true positions of the fitted tracks and of
the truth-projected spline tracks (the representation floor), and the mean coherent drift (x) offset of each."""
import sys, numpy as np
sys.path.insert(0, '/sdf/group/neutrino/pgranger/larnd-sim-jax/review_2026_09/scripts/eventdisplay')
from rebuild import FitRebuild
import jax.numpy as jnp, glob
ck_path = sorted(glob.glob('/sdf/group/neutrino/pgranger/larnd-sim-jax/fit_result/sci_full_KN10S1/history_iter10000_*.pkl'))[0]
R = FitRebuild(ck_path); fp, ck = R.fp, R.ck; fields = R.sfields
di, Ei, xi = fields.index('dEdx'), fields.index('dE'), fields.index('dx')
exec(open('/sdf/scratch/neutrino/pgranger/review_2026_09/POLISH/joint/joint_polish2.py').read().split('def truth_geom(i):')[1].split('\ndef batch_fns')[0].join(['def truth_geom(i):', '']))
for i in [0, 1, 48]:
    bt = R.batch(i, check=True); tw, nkeep = truth_geom(i); tp = ck['true_positions'][i]; rf, rt, dxf, dxt = [], [], [], []
    for c in bt['ctxs']:
        tr = tp.get(tuple(map(float, c.track_id)), tp.get(str(tuple(map(float, c.track_id)))))
        if tr is None: continue
        tr = np.asarray(tr); idx = np.asarray(c.idxs)[:len(tr)]; cols = [R.col['x'], R.col['y'], R.col['z']]
        pf, pt = bt['fitted'][idx][:, cols], tw[idx][:, cols]
        rf.append(np.linalg.norm(pf - tr, axis=1)); rt.append(np.linalg.norm(pt - tr, axis=1)); dxf.append(pf[:, 0] - tr[:, 0]); dxt.append(pt[:, 0] - tr[:, 0])
    f = lambda L: 1e4 * float(np.concatenate(L).mean())
    print(f'batch {i}: tracks {len(bt["ctxs"])} kept-fitted {nkeep} | mean |r| fitted {f(rf):.1f} um, truth-proj {f(rt):.1f} um | '
          f'mean x offset fitted {f(dxf):+.1f} um, truth-proj {f(dxt):+.1f} um | rebuild check {bt["checks"].get("pos_mean_fit_um", -1):.1f} um | '
          f'dEdx cols equal {np.allclose(tw[:, di], bt["fitted"][:, di])}', flush=True)
