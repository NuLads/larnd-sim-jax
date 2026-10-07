import sys, pickle, numpy as np
sys.path.insert(0, '/sdf/group/neutrino/pgranger/larnd-sim-jax')
from optimize.fit_params import _dir_from_angles, _transverse_frame
ck = pickle.load(open(sys.argv[1], 'rb')); out = {k: [] for k in ('all', 'first', 'qmean', 'median', 'last')}; npx=[]
for b in [int(x) for x in sys.argv[3].split(',')]:
    f = np.load(sys.argv[2] + f'/batch{b}_target.npz'); H = np.stack([f['pixel_x'], f['pixel_y'], f['pixel_z']], 1).astype(float); ev = f['event']; q = f['adcs']; pid = f['pixel_id']; tk = f['ticks']
    for c in ck['chain_contexts'][b]:
        tid = tuple(c['track_id']); tr = np.asarray(ck['true_positions'][b][str(tid)], float)
        u0 = _dir_from_angles(c['theta0_i'], c['phi0_i']); e1, e2 = _transverse_frame(u0); x0 = np.asarray(c['x0_fixed'], float); L = c['total_len']
        if abs(u0[2]) > 0.5: continue
        al = np.asarray(c['alpha_mid'], float)[:len(tr)]; o = np.argsort(al); te2 = ((tr - x0 - al[:, None] * L * u0) @ e2)
        m = ev == int(tid[0]); P = H[m]; Q = q[m]; I = pid[m]; T = tk[m]
        a = ((P - x0) @ u0) / L; d = P - x0 - a[:, None] * L * u0; s = (a > .01) & (a < .99) & (np.linalg.norm(d, axis=1) < 1)
        P, Q, I, T, a, d = P[s], Q[s], I[s], T[s], a[s], d[s]
        r = d @ e2 - np.interp(a, al[o], te2[o]); out['all'] += list(r - np.median(r))
        rr = {k: [] for k in ('first', 'qmean', 'median', 'last')}
        for p in np.unique(I):
            k = I == p; j = np.argsort(T[k]); rk = r[k][j]; w = Q[k][j]
            rr['first'].append(rk[0]); rr['last'].append(rk[-1]); rr['qmean'].append((rk * w).sum() / w.sum()); rr['median'].append(np.median(rk))
        npx.append(len(np.unique(I)))
        for kk in rr: v = np.array(rr[kk]); out[kk] += list(v - np.median(v))
for k, v in out.items(): v = np.array(v); print(f"{k:7s} n={len(v):5d} MADsig {1.4826*np.median(np.abs(v))*1e4:6.0f} um  rms {v.std()*1e4:6.0f}")
print('pixels/track median', np.median(npx), 'hits/pixel', len(out['all'])/len(out['first']))
