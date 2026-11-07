"""ENDFIT option (b) check (24 Oct): batch sum of the endpoints' drift-distance shifts, sum_j w_j c_j, from a checkpoint
(chain_cache + chain_contexts; same weights as fit_params._endpoint_drift_weights, frozen tracks included: their e2
endpoint coefficients stay 0). With LARND_ENDPOINT_DRIFT_ZERO_MEAN=1 every batch must be ~0 relative to the scale of
the individual terms; without it, not. Also prints the batch-mean drift shift (um) and the rms per-endpoint shift.
usage (in the container, PYTHONPATH = the live tree): python3 check_zero_mean.py <history.pkl> [...]"""
import sys, pickle, numpy as np
from optimize.fit_params import _dir_from_angles, _transverse_frame
ZB = np.array([[30.59, 0.159], [-30.59, -0.159]])        # module0 tpc_borders[:, 2]: z of [anode, cathode] (cm)
for f in sys.argv[1:]:
    d = pickle.load(open(f, 'rb')); cc, cx = d['chain_cache'], d['chain_contexts']
    print(f'{f.split("/")[-2]}  (total_iter {d.get("total_iter")})')
    for bi in sorted(cc):
        terms = []
        for st, c in zip(cc[bi], cx[bi]):
            a = np.asarray(st['angles'], np.float64); K = len(a) // 2
            u0 = _dir_from_angles(c['theta0_i'], c['phi0_i']); _, e2 = _transverse_frame(u0)
            for j, z in ((2 * K - 2, float(c['x0_fixed'][2])), (2 * K - 1, float(c['x0_fixed'][2] + c['total_len'] * u0[2]))):
                t = int(np.argmin(np.abs(ZB.mean(1) - z)))
                terms.append(float(e2[2]) * np.sign(z - ZB[t, 0]) * a[j])
        t = np.array(terms); s = t.sum(); rms = np.sqrt(np.mean(t ** 2)) if len(t) else 0.
        print(f'  batch {bi:3d}: {len(t) // 2:2d} tracks  sum w.c {s:+.3e} cm  mean drift shift {1e4 * t.mean():+8.3f} um  '
              f'rms per endpoint {1e4 * rms:7.2f} um  |sum| / (rms sqrt n) {abs(s) / max(rms * np.sqrt(len(t)), 1e-30):.1e}')
