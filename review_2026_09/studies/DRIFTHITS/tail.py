import sys, glob, pickle, numpy as np
sys.path.insert(0, '/sdf/group/neutrino/pgranger/larnd-sim-jax')
from optimize.fit_params import _dir_from_angles, _transverse_frame, _DRIFT_AXIS
run = sys.argv[1]
fs = sorted(glob.glob(f'fit_result/sci_full_{run}/history_iter*.pkl'), key=lambda x: int(x.split('iter')[-1].split('_')[0]))
f = [x for x in fs if int(x.split('iter')[-1].split('_')[0]) <= 10000][-1]; ck = pickle.load(open(f, 'rb'))
R = []
for b, ctxs in ck['chain_contexts'].items():
    for c, st in zip(ctxs, ck['chain_cache'][b]):
        u0 = _dir_from_angles(c['theta0_i'], c['phi0_i']); e1, e2 = _transverse_frame(u0)
        if not (abs(e1[2]) < 1e-6 and np.linalg.norm(np.cross(_DRIFT_AXIS, u0)) > 1e-3): continue
        ang = np.asarray(st['angles'], float); K = len(ang) // 2
        tr = np.asarray(ck['true_positions'][b][str(tuple(c['track_id']))], float); al = np.asarray(c['alpha_mid'], float)[:len(tr)]; tr = tr[:len(al)]
        x0 = np.asarray(c['x0_fixed'], float); L = c['total_len']
        B = np.sin(np.outer(al, np.pi * np.arange(1, K + 1))); dev = (tr - x0 - al[:, None] * L * u0) @ e2
        b2t = np.linalg.lstsq(B, dev, rcond=None)[0]; resid_rep = np.sqrt(np.mean((B @ b2t - dev)**2)) * 1e4
        ag = np.linspace(0, 1, 101); Bg = np.sin(np.outer(ag, np.pi * np.arange(1, K + 1)))
        e = np.sqrt(np.mean((Bg @ (ang[K:] - b2t))**2)) * 1e4
        zmid = x0[2] + 0.5 * L * u0[2]
        R.append((e, abs(u0[2]), L, K, np.sqrt(np.mean((Bg @ b2t)**2))*1e4, resid_rep, abs(zmid), np.abs(ang[K:]).max()*1e4, np.abs(ang[:K]).max()*1e4))
R = np.array(R); e = R[:, 0]; hi = e > np.percentile(e, 85)
names = ['err', '|u0z|', 'L', 'K', 'truth dev rms', 'repr resid', '|zmid|', 'max|b2| fit', 'max|b1| fit']
print(f"{run}: n={len(e)} p50 {np.median(e):.0f} p85 {np.percentile(e,85):.0f} p95 {np.percentile(e,95):.0f} max {e.max():.0f}; tail (top15%) share of sum(err^2): {(e[hi]**2).sum()/(e**2).sum():.2f}")
for j, n in enumerate(names): print(f"  {n:14s} tail median {np.median(R[hi, j]):9.2f}   bulk median {np.median(R[~hi, j]):9.2f}   spearman w/ err {np.corrcoef(np.argsort(np.argsort(R[:,j])), np.argsort(np.argsort(e)))[0,1]:+.2f}")
