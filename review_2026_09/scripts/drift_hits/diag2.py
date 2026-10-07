import sys, pickle, numpy as np
sys.path.insert(0, '/sdf/group/neutrino/pgranger/larnd-sim-jax')
from optimize.fit_params import _dir_from_angles, _transverse_frame
ck = pickle.load(open(sys.argv[1], 'rb'))
for b in [int(x) for x in sys.argv[3].split(',')]:
    f = np.load(sys.argv[2] + f'/batch{b}_target.npz'); H = np.stack([f['pixel_x'], f['pixel_y'], f['pixel_z']], 1).astype(float); ev = f['event']; q = f['adcs']
    for c in ck['chain_contexts'][b]:
        tid = tuple(c['track_id']); tr = np.asarray(ck['true_positions'][b][str(tid)], float)
        u0 = _dir_from_angles(c['theta0_i'], c['phi0_i']); e1, e2 = _transverse_frame(u0); x0 = np.asarray(c['x0_fixed'], float); L = c['total_len']
        al = np.asarray(c['alpha_mid'], float)[:len(tr)]
        P = H[ev == int(tid[0])]; a = ((P - x0) @ u0) / L; d = P - x0 - a[:, None] * L * u0
        s = (a > .01) & (a < .99) & (np.linalg.norm(d, axis=1) < 1)
        o = np.argsort(al); tz = np.interp(a[s], al[o], ((tr - x0 - al[:, None] * L * u0) @ e2)[o])  # truth e2-offset at hit alpha
        r = (d[s] @ e2) - tz; t_e2 = tz
        fr = abs(u0[2])
        print(f"b{b} {tid} L={L:5.1f} |u0z|={fr:.2f} e2xy={np.hypot(*e2[:2]):.2f} n={s.sum():4d}  hit-truth e2 resid: med {np.median(r)*1e4:+6.0f}um  MADsig {1.4826*np.median(np.abs(r-np.median(r)))*1e4:5.0f}um  truth e2 dev rms {t_e2.std()*1e4:5.0f}um  corr(r,a) {np.corrcoef(r,a[s])[0,1]:+.2f}")
