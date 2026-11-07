"""Offline: quality of hit-derived spline seeds (e1 and e2) vs the straight line, by track curvature class.
usage: seed_eval.py CKPT TARGET_DIR batches"""
import sys, pickle, numpy as np
sys.path.insert(0, '/sdf/group/neutrino/pgranger/larnd-sim-jax')
from optimize.fit_params import _dir_from_angles, _transverse_frame, _DRIFT_AXIS
ck = pickle.load(open(sys.argv[1], 'rb')); rows = []
for b in [int(x) for x in sys.argv[3].split(',')]:
    f = np.load(sys.argv[2] + f'/batch{b}_target.npz')
    H = np.stack([f['pixel_x'], f['pixel_y'], f['pixel_z']], 1).astype(float); ev = f['event']; q = f['adcs'].astype(float); pid = f['pixel_id']
    for c, st in zip(ck['chain_contexts'][b], ck['chain_cache'][b]):
        u0 = _dir_from_angles(c['theta0_i'], c['phi0_i']); e1, e2 = _transverse_frame(u0)
        if not (abs(e1[2]) < 1e-6 and np.linalg.norm(np.cross(_DRIFT_AXIS, u0)) > 1e-3): continue
        K = len(np.asarray(st['angles'])) // 2
        tid = tuple(float(np.ravel(np.asarray(x))[0]) for x in np.ravel(np.asarray(c['track_id'], dtype=object))) if np.size(c['track_id']) == 2 else tuple(c['track_id']); tr = np.asarray(ck['true_positions'][b][str(tid)], float); al = np.asarray(c['alpha_mid'], float)[:len(tr)]; tr = tr[:len(al)]
        x0 = np.asarray(c['x0_fixed'], float); L = c['total_len']
        Bt = np.sin(np.outer(al, np.pi * np.arange(1, K + 1))); dt = tr - x0 - al[:, None] * L * u0
        bt1 = np.linalg.lstsq(Bt, dt @ e1, rcond=None)[0]; bt2 = np.linalg.lstsq(Bt, dt @ e2, rcond=None)[0]
        m = ev == int(tid[0]); P, w, I = H[m], q[m], pid[m]
        a = ((P - x0) @ u0) / L; d = P - x0 - a[:, None] * L * u0
        s = (a > .01) & (a < .99) & (np.linalg.norm(d, axis=1) < 1.0)
        P, w, I = P[s], w[s], I[s]
        if len(np.unique(I)) < 3 * K + 2: continue
        up, inv = np.unique(I, return_inverse=True); ws = np.bincount(inv, w)
        P = np.stack([np.bincount(inv, w * P[:, j]) / ws for j in range(3)], 1); w = ws
        a = ((P - x0) @ u0) / L; d = P - x0 - a[:, None] * L * u0
        ag = np.linspace(0, 1, 101); Bg = np.sin(np.outer(ag, np.pi * np.arange(1, K + 1)))
        rms = lambda bb: float(np.sqrt(np.mean((Bg @ bb) ** 2)) * 1e4)
        res = dict(curv=max(rms(bt1), rms(bt2)), K=K)
        for comp, y, bt in (('e1', d @ e1, bt1), ('e2', d @ e2, bt2)):
            B = np.sin(np.outer(a, np.pi * np.arange(1, K + 1)))
            A = np.hstack([np.ones((len(a), 1)), a[:, None], B]); sw = np.sqrt(w / w.mean()); keep = np.ones(len(a), bool)
            for _ in range(4):
                coef, *_ = np.linalg.lstsq(A[keep] * sw[keep, None], y[keep] * sw[keep], rcond=None)
                r = y - A @ coef; sig = 1.4826 * np.median(np.abs(r[keep])) + 1e-4; nk = np.abs(r) < 3 * sig
                if nk.sum() < 3 * K + 2 or (nk == keep).all(): break
                keep = nk
            # coefficient uncertainties from the weighted LS
            Aw = A[keep] * sw[keep, None]; cov = sig ** 2 * np.linalg.pinv(Aw.T @ Aw); sk = np.sqrt(np.diag(cov))[2:]
            bc = coef[2:]
            gate = np.where(np.abs(bc) > 2 * sk, bc, 0.0)
            shrink = bc * np.clip(1 - sk ** 2 / np.maximum(bc ** 2, 1e-30), 0, 1)
            from scipy.stats import chi2
            T = float(bc @ np.linalg.pinv(cov[2:, 2:]) @ bc)
            for lv, nm in ((0.99, 'tg99'), (0.999, 'tg999'), (0.9999, 'tg9999')):
                res[comp + '_' + nm] = rms((bc if T > chi2.ppf(lv, K) else 0 * bc) - bt); res[comp + '_' + nm + '_on'] = T > chi2.ppf(lv, K)
            for thr in (300, 400, 500, 700):
                res[comp + f'_amp{thr}'] = rms((bc if rms(bc) > thr else 0 * bc) - bt); res[comp + f'_amp{thr}_on'] = rms(bc) > thr
            res[comp + '_straight'] = rms(bt); res[comp + '_raw'] = rms(bc - bt); res[comp + '_gate'] = rms(gate - bt); res[comp + '_shrink'] = rms(shrink - bt)
        rows.append(res)
import collections
cv = np.array([r['curv'] for r in rows]); print(f"tracks {len(rows)}; true deviation rms (max of e1,e2) median {np.median(cv):.0f} um")
for lab, sel in (('all', cv > -1), ('curved >500um', cv > 500), ('straightish <250um', cv < 250)):
    print(f"--- {lab} (n={sel.sum()})  median shape error [um]")
    for comp in ('e1', 'e2'):
        print('   ' + comp + ': ' + '  '.join(f"{v}: {np.median([r[comp + '_' + v] for r, s in zip(rows, sel) if s]):6.0f}" for v in ('straight', 'raw', 'tg999', 'amp300', 'amp400', 'amp500', 'amp700')))
    print('   frac ON amp400 e1 / amp700 e2: ' + f"{np.mean([r['e1_amp400_on'] for r, s in zip(rows, sel) if s]):.2f} / {np.mean([r['e2_amp700_on'] for r, s in zip(rows, sel) if s]):.2f}")
    print('   frac switched ON (tg999) e1/e2: ' + f"{np.mean([r['e1_tg999_on'] for r, s in zip(rows, sel) if s]):.2f} / {np.mean([r['e2_tg999_on'] for r, s in zip(rows, sel) if s]):.2f}")
    print('   frac tracks where gate beats straight (e2): ' + f"{np.mean([r['e2_gate'] < r['e2_straight'] for r, s in zip(rows, sel) if s]):.2f}" + '  (e1): ' + f"{np.mean([r['e1_gate'] < r['e1_straight'] for r, s in zip(rows, sel) if s]):.2f}")
