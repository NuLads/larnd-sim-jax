"""Classic crossing-muon lifetime estimate from the TARGET HITS (no forward model, no fitted geometry):
dQ/dx of 2-cm track pieces vs drift time, truncated mean per drift-time bin, exponential slope -> tau.
Only uses: hits (pixel x/y, timing z, ADC), the nominal track lines, vdrift, t0 = 0.
usage: tau_std.py CKPT TARGET_DIR [trunc_frac=0.7] [piece_cm=2]"""
import sys, pickle, numpy as np
sys.path.insert(0, '/sdf/group/neutrino/pgranger/larnd-sim-jax')
from optimize.fit_params import _dir_from_angles
ZA, VD = 30.58975, 0.1587
def adc2q(adc): return (adc / 256 * (1282.71484375 - 284.27734375) + 284.27734375 - 580) / 4e-3 * 1e-3   # ke
def pieces(ck, tdir, piece=2.0, batches=None):
    out = []                                           # (track key, drift time [us], dQ/dx [ke/cm], |u0z|)
    for b, ctxs in ck['chain_contexts'].items():
        if batches is not None and b not in batches: continue
        try: f = np.load(f'{tdir}/batch{b}_target.npz')
        except FileNotFoundError: continue
        H = np.stack([f['pixel_x'], f['pixel_y'], f['pixel_z']], 1).astype(float); ev = f['event']; Q = adc2q(f['adcs'].astype(float))
        for c in ctxs:
            u0 = _dir_from_angles(c['theta0_i'], c['phi0_i']); x0 = np.asarray(c['x0_fixed'], float); L = float(c['total_len'])
            m = ev == int(c['track_id'][0]); P, q = H[m], Q[m]
            a = (P - x0) @ u0; d = P - x0 - a[:, None] * u0
            s = (a > 0) & (a < L) & (np.linalg.norm(d, axis=1) < 1.0)
            a, q, Pz = a[s], q[s], P[s, 2]
            nb = int(L // piece)
            if nb < 1: continue
            for k in range(nb):
                lo = k * piece; mm = (a >= lo) & (a < lo + piece)
                if mm.sum() == 0: continue
                zmid = x0[2] + (lo + piece / 2) * u0[2]
                if abs(zmid) > ZA - 0.5 or abs(zmid) < 0.5: continue          # skip anode/cathode edges
                out.append((b, str(tuple(c['track_id'])), (ZA - abs(zmid)) / VD, q[mm].sum() / piece, abs(u0[2])))
    return out
def tau_fit(rows, trunc=0.7, nbins=9, tmax=190.0):
    t = np.array([r[2] for r in rows]); y = np.array([r[3] for r in rows])
    edges = np.linspace(0, tmax, nbins + 1); tc, mu, er = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        v = np.sort(y[(t >= lo) & (t < hi)])
        if len(v) < 50: continue
        k = max(1, int(trunc * len(v))); w = v[:k]
        tc.append(t[(t >= lo) & (t < hi)].mean()); mu.append(w.mean()); er.append(w.std(ddof=1) / np.sqrt(k))
    tc, mu, er = map(np.array, (tc, mu, er))
    A = np.vstack([np.ones_like(tc), -tc]).T; W = (mu / er) ** 2                 # ln mu = ln A - t/tau, weights 1/var(ln mu)
    cov = np.linalg.inv(A.T @ (A * W[:, None])); p = cov @ A.T @ (W * np.log(mu))
    return 1 / p[1], np.sqrt(cov[1, 1]) / p[1] ** 2, tc, mu, er
if __name__ == '__main__':
    ck = pickle.load(open(sys.argv[1], 'rb')); trunc = float(sys.argv[3]) if len(sys.argv) > 3 else 0.7; piece = float(sys.argv[4]) if len(sys.argv) > 4 else 2.0
    rows = pieces(ck, sys.argv[2], piece)
    tau, etau, tc, mu, er = tau_fit(rows, trunc)
    truth = float(np.ravel(ck['lifetime_target'][0])[0])
    print(f"pieces {len(rows)}  tau_std = {tau:7.1f} +- {etau:5.1f} us   truth {truth:.1f}   err {100*(tau/truth-1):+.2f} +- {100*etau/truth:.2f}%")
    print('  bins t[us]: ' + ' '.join(f'{x:5.0f}' for x in tc)); print('  trunc mean dQ/dx [ke/cm]: ' + ' '.join(f'{x:5.1f}' for x in mu))
