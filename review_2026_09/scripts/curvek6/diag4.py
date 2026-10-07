import sys, pickle, numpy as np
sys.path.insert(0, '/sdf/group/neutrino/pgranger/larnd-sim-jax')
from optimize.fit_params import _dir_from_angles, _transverse_frame
ck = pickle.load(open(sys.argv[1], 'rb')); ac=[]; rows=[]
for b in [int(x) for x in sys.argv[3].split(',')]:
    f = np.load(sys.argv[2] + f'/batch{b}_target.npz'); H = np.stack([f['pixel_x'], f['pixel_y'], f['pixel_z']], 1).astype(float); ev = f['event']; q = f['adcs']; pid = f['pixel_id']
    for c in ck['chain_contexts'][b]:
        tid = tuple(c['track_id']); tr = np.asarray(ck['true_positions'][b][str(tid)], float)
        u0 = _dir_from_angles(c['theta0_i'], c['phi0_i']); e1, e2 = _transverse_frame(u0); x0 = np.asarray(c['x0_fixed'], float); L = c['total_len']
        al = np.asarray(c['alpha_mid'], float)[:len(tr)]; o = np.argsort(al)
        m = ev == int(tid[0]); P = H[m]; Q = q[m]; I = pid[m]
        a = ((P - x0) @ u0) / L; d = P - x0 - a[:, None] * L * u0; s = (a > .01) & (a < .99) & (np.linalg.norm(d, axis=1) < 1)
        P, Q, I = P[s], Q[s], I[s]; up, inv = np.unique(I, return_inverse=True); ws = np.bincount(inv, Q)
        P = np.stack([np.bincount(inv, Q * P[:, j]) / ws for j in range(3)], 1)
        a = ((P - x0) @ u0) / L
        # truth z at the pixel's xy position: nearest truth midpoint in xy (avoids alpha mismatch)
        tz = np.array([tr[np.argmin(((tr[:, :2] - p[:2])**2).sum(1)), 2] for p in P])
        r = P[:, 2] - tz; j = np.argsort(a); r = r[j] - np.median(r)
        if len(r) > 20: ac.append(np.corrcoef(r[:-1], r[1:])[0, 1])
        # smooth component: running mean over 15 pixels
        sm = np.convolve(r, np.ones(15)/15, 'valid')
        rows.append((abs(u0[2]), len(r), 1.4826*np.median(np.abs(r))*1e4, sm.std()*1e4, np.corrcoef(r, a[j])[0,1]))
rows = np.array(rows); print('lag-1 autocorr median', np.median(ac))
print('|u0z|  npix  MADsig_um  smooth15_rms_um  corr(r,alpha)'); [print(f"{x[0]:.2f} {x[1]:4.0f} {x[2]:6.0f} {x[3]:6.0f} {x[4]:+.2f}") for x in rows[:20]]
print('median smooth rms', np.median(rows[:,3]), ' expected if white:', np.median(rows[:,2])/np.sqrt(15))
