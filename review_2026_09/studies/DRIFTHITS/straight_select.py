"""Offline: if we FREEZE e2 at the straight line for tracks the hits call 'straight' (chi2 test of the hit-derived
e2 modes vs zero NOT significant at level CL), how accurate is that frozen geometry? (sample 8, K=3, DSREP8 contexts)"""
import sys, pickle, numpy as np
from scipy.stats import chi2
sys.path.insert(0, '/sdf/group/neutrino/pgranger/larnd-sim-jax')
from optimize.fit_params import _dir_from_angles, _transverse_frame, _DRIFT_AXIS
ck = pickle.load(open(sys.argv[1], 'rb')); tdir = sys.argv[2]
rows = []
for b in range(100):
    try: f = np.load(f'{tdir}/batch{b}_target.npz')
    except FileNotFoundError: continue
    H = np.stack([f['pixel_x'], f['pixel_y'], f['pixel_z']], 1).astype(float); ev = f['event']; q = f['adcs'].astype(float); pid = f['pixel_id']
    for c, st in zip(ck['chain_contexts'][b], ck['chain_cache'][b]):
        u0 = _dir_from_angles(c['theta0_i'], c['phi0_i']); e1, e2 = _transverse_frame(u0)
        if not (abs(e1[2]) < 1e-6 and np.linalg.norm(np.cross(_DRIFT_AXIS, u0)) > 1e-3): continue
        K = len(np.asarray(st['angles'])) // 2; L = c['total_len']; x0 = np.asarray(c['x0_fixed'], float)
        tr = np.asarray(ck['true_positions'][b][str(tuple(c['track_id']))], float); al = np.asarray(c['alpha_mid'], float)[:len(tr)]; tr = tr[:len(al)]
        Bt = np.sin(np.outer(al, np.pi * np.arange(1, K + 1))); bt2 = np.linalg.lstsq(Bt, (tr - x0 - al[:, None] * L * u0) @ e2, rcond=None)[0]
        ag = np.linspace(0, 1, 101); Bg = np.sin(np.outer(ag, np.pi * np.arange(1, K + 1)))
        err_straight = np.sqrt(np.mean((Bg @ bt2) ** 2)) * 1e4
        m = ev == int(c['track_id'][0]); P, w, I = H[m], q[m], pid[m]
        a = ((P - x0) @ u0) / L; d = P - x0 - a[:, None] * L * u0
        s = (a > .01) & (a < .99) & (np.linalg.norm(d - (d @ u0)[:, None] * u0, axis=1) < 1.0)
        P, w, I = P[s], w[s], I[s]
        if len(np.unique(I)) < 3 * K + 2: rows.append((err_straight, np.nan, L, abs(u0[2]))); continue
        up, inv = np.unique(I, return_inverse=True); ws = np.bincount(inv, w)
        P = np.stack([np.bincount(inv, w * P[:, j]) / ws for j in range(3)], 1); w = ws
        a = ((P - x0) @ u0) / L; dz = (P - x0 - a[:, None] * L * u0)[:, 2]
        B = np.sin(np.outer(a, np.pi * np.arange(1, K + 1)))
        A = np.hstack([np.ones((len(a), 1)), a[:, None], B * e2[2]]); sw = np.sqrt(w / w.mean()); keep = np.ones(len(a), bool)
        for _ in range(4):
            coef = np.linalg.lstsq(A[keep] * sw[keep, None], dz[keep] * sw[keep], rcond=None)[0]
            r = dz - A @ coef; sig = 1.4826 * np.median(np.abs(r[keep])) + 1e-4; nk = np.abs(r) < 3 * sig
            if nk.sum() < 3 * K + 2 or (nk == keep).all(): break
            keep = nk
        Aw = A[keep] * sw[keep, None]; cov = sig ** 2 * np.linalg.pinv(Aw.T @ Aw)
        T = float(coef[2:] @ np.linalg.pinv(cov[2:, 2:]) @ coef[2:])
        rows.append((err_straight, chi2.sf(T, K), L, abs(u0[2])))
R = np.array(rows); e = R[:, 0]; pv = R[:, 1]
print(f'drift-frame tracks {len(R)}; straight-line e2 error over ALL: median {np.median(e):.0f} mean {e.mean():.0f} p95 {np.percentile(e,95):.0f} um')
for cl in (0.5, 0.8, 0.9, 0.99, 0.999):
    fr = np.isfinite(pv) & (pv > 1 - cl)          # NOT significant at level cl -> called straight -> frozen
    ee = e[fr]
    print(f'  freeze if not significant at CL {cl:5.3f}: frozen {fr.mean():.2f} of tracks; their straight-line error median {np.median(ee):5.0f} mean {ee.mean():5.0f} p90 {np.percentile(ee,90):5.0f} p95 {np.percentile(ee,95):5.0f} max {ee.max():6.0f} um')
