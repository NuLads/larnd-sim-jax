"""CPU, offline: can the truth-free hit offset (hit_flag.py) REPAIR a misplaced track instead of cutting it? Per track,
fit the charge-weighted in-plane offset of its own hits vs position along the track with the spline's sine basis
(endpoints anchored, as in the fit), move the fitted path by it, and compare the truth residual before and after.
usage: repair_test.py RUN SAMPLE ITER"""
import sys, glob, json, numpy as np
sys.path.insert(0, '/sdf/group/neutrino/pgranger/larnd-sim-jax/review_2026_09/scripts/eventdisplay')
from rebuild import FitRebuild
run, s, it = sys.argv[1], sys.argv[2], sys.argv[3]
D = '/sdf/scratch/neutrino/pgranger/review_2026_09/POLISH/joint'
R = FitRebuild(sorted(glob.glob(f'/sdf/group/neutrino/pgranger/larnd-sim-jax/fit_result/sci_full_{run}{s}/history_iter{it}_*.pkl'))[0])
fp, ck = R.fp, R.ck; cols = [R.col['x'], R.col['y'], R.col['z']]; ev_c = R.sfields.index('eventID'); out = []
for i in range(len(ck['true_positions'])):
    bt = R.batch(i, check=False); f = np.asarray(bt['fitted']); tg = R.target(i); tp = ck['true_positions'][i]
    hx = np.stack([tg['pixel_x'], tg['pixel_y'], tg['pixel_z']], 1).astype(float); hq = np.asarray(tg['adcs'], float); hev = np.asarray(tg['event'])
    evs = [e for e in np.unique(f[:, ev_c]) if e >= 0]
    trk = [(c, f[np.asarray(c.idxs)][:, cols], float(f[np.asarray(c.idxs)[0], ev_c])) for c in bt['ctxs']]
    for le, e in enumerate(evs):
        H = hx[hev == le]; Q = hq[hev == le]; T = [(c, P) for c, P, ee in trk if ee == e]
        if len(H) == 0 or not T: continue
        dist = np.stack([np.min(np.linalg.norm(H[:, None, :] - P[None, ::10, :], axis=2), axis=1) for _, P in T], 1)
        own = np.argmin(dist, 1); dmin = dist[np.arange(len(H)), own]
        for k, (c, P) in enumerate(T):
            tr = tp.get(tuple(map(float, c.track_id)), tp.get(str(tuple(map(float, c.track_id)))))
            m = (own == k) & (dmin < 1.0)
            if tr is None or m.sum() < 8: continue
            tr = np.asarray(tr, float); n = min(len(tr), len(P)); h, q = H[m], Q[m]
            u = P[-1] - P[0]; L = np.linalg.norm(u); u = u / max(L, 1e-9)
            nxy = np.array([-u[1], u[0], 0.]); nxy /= max(np.linalg.norm(nxy), 1e-9)
            j = np.argmin(np.linalg.norm(h[:, None, :] - P[None, ::10, :], axis=2), axis=1) * 10
            off = (h - P[j]) @ nxy; al = np.clip(((h - P[0]) @ u) / max(L, 1e-9), 0, 1)
            K = fp._spline_K(c.total_len, R.knot_cm); B = np.sin(np.outer(al, np.pi * np.arange(1, K + 1)))
            sw = np.sqrt(q / q.mean()); b = np.linalg.lstsq(B * sw[:, None], off * sw, rcond=None)[0]
            alP = np.clip(((P - P[0]) @ u) / max(L, 1e-9), 0, 1); corr = np.sin(np.outer(alP, np.pi * np.arange(1, K + 1))) @ b
            Pn = P + corr[:, None] * nxy[None, :]
            r0 = 1e4 * float(np.linalg.norm(P[:n] - tr[:n], axis=1).mean()); r1 = 1e4 * float(np.linalg.norm(Pn[:n] - tr[:n], axis=1).mean())
            sl = [float(np.sum(q[(al >= a0) & (al < a0 + .25)] * off[(al >= a0) & (al < a0 + .25)]) / np.sum(q[(al >= a0) & (al < a0 + .25)]))
                  for a0 in (0, .25, .5, .75) if np.sum((al >= a0) & (al < a0 + .25)) >= 4]
            out.append(dict(batch=i, n=n, r0=r0, r1=r1, flag=1e4 * float(np.max(np.abs(sl))) if sl else float('nan'),
                            corr_max_um=1e4 * float(np.abs(corr).max())))
json.dump(out, open(f'{D}/repair_{run}{s}_{it}.json', 'w'), indent=1)
a = {k: np.array([o[k] for o in out], float) for k in out[0]}; w = a['n']
print(f'{run}{s}@{it}: {len(out)} tracks')
fl = np.isfinite(a['flag'])
for name, m in [('all', fl), ('outliers |r0|>200', fl & (a['r0'] > 200)), ('rest', fl & (a['r0'] <= 200))] + \
               [(f'flag top {p}%', fl & (a['flag'] > np.quantile(a['flag'][fl], 1 - p / 100))) for p in (5, 10, 20)]:
    print(f'  {name:20s} n {m.sum():4d}  mean |r| before {np.average(a["r0"][m], weights=w[m]):6.1f}  after repair {np.average(a["r1"][m], weights=w[m]):6.1f} um  '
          f'(median {np.median(a["r0"][m]):.0f} -> {np.median(a["r1"][m]):.0f}; improved on {np.mean(a["r1"][m] < a["r0"][m]):.2f})')
for p in (10, 20):
    t = np.quantile(a['flag'][fl], 1 - p / 100); r = np.where(a['flag'] > t, a['r1'], a['r0'])
    print(f'  repair only the flag top {p}%: segment-weighted mean |r| all tracks {np.average(a["r0"][fl], weights=w[fl]):.1f} -> {np.average(r[fl], weights=w[fl]):.1f} um')
