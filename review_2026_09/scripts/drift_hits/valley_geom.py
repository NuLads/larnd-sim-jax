"""What geometry change accompanies the lifetime move? Paired, track-by-track: ARM - REF geometry (same tracks).
usage: valley_geom.py REF ARM [ARM...]   e.g. GEOFRZ8 GTRU8 DSREP8
Reports, over drift-frame tracks: RMS of the e1 / e2 displacement change, the signed change of DRIFT DISTANCE
(+ = farther from anode) vs drift distance, and the change of the track's drift-direction slope |dz/ds|."""
import sys, glob, pickle, numpy as np
sys.path.insert(0, '/sdf/group/neutrino/pgranger/larnd-sim-jax')
from optimize.fit_params import _dir_from_angles, _transverse_frame, _DRIFT_AXIS
ZA = 30.58975
def load(tag):
    for t in (f'fit_result/sci_full_{tag}', f'fit_result/sci_full{tag}'):
        fs = sorted(glob.glob(f'{t}/history_iter*.pkl'), key=lambda x: int(x.split('iter')[-1].split('_')[0]))
        f10 = [x for x in fs if int(x.split('iter')[-1].split('_')[0]) <= 10000]
        if f10 or fs: return pickle.load(open((f10 or fs)[-1], 'rb'))   # frozen-geometry runs may only keep a later file
def geom(d):
    out = {}
    for b, ctxs in d['chain_contexts'].items():
        for c, st in zip(ctxs, d['chain_cache'][b]):
            out[(b, str(tuple(c['track_id'])))] = (c, np.asarray(st['angles'], float))
    return out
ref = geom(load(sys.argv[1]))
ag = np.linspace(0.05, 0.95, 37)
for arm in sys.argv[2:]:
    g = geom(load(arm)); rows = []; prof = []
    for k, (c, a_ref) in ref.items():
        if k not in g: continue
        a_arm = g[k][1]
        if a_arm.shape != a_ref.shape: continue
        u0 = _dir_from_angles(c['theta0_i'], c['phi0_i']); e1, e2 = _transverse_frame(u0)
        if not (abs(e1[2]) < 1e-6 and np.linalg.norm(np.cross(_DRIFT_AXIS, u0)) > 1e-3): continue
        K = len(a_ref) // 2; L = c['total_len']; x0 = np.asarray(c['x0_fixed'], float)
        B = np.sin(np.outer(ag, np.pi * np.arange(1, K + 1))); dB = np.cos(np.outer(ag, np.pi * np.arange(1, K + 1))) * (np.pi * np.arange(1, K + 1)) / L
        db1, db2 = a_arm[:K] - a_ref[:K], a_arm[K:] - a_ref[K:]
        nom = x0[None, :] + ag[:, None] * L * u0[None, :]
        pos_ref = nom + (B @ a_ref[:K])[:, None] * e1 + (B @ a_ref[K:])[:, None] * e2
        dz = (B @ db2) * e2[2]                                   # change of z (e1z = 0)
        ddrift = -np.sign(pos_ref[:, 2]) * dz                    # + = farther from anode
        D = ZA - np.abs(pos_ref[:, 2])                           # drift distance
        slope_ref = u0[2] + (dB @ a_ref[K:]) * e2[2]             # dz/ds of the reference track
        dslope = (dB @ db2) * e2[2]
        dabs = np.sign(slope_ref) * dslope                       # change of |dz/ds| (+ = steeper in drift)
        rows.append((np.sqrt(np.mean((B @ db1) ** 2)) * 1e4, np.sqrt(np.mean((B @ db2) ** 2)) * 1e4, ddrift.mean() * 1e4, dabs.mean() * 1e3, abs(u0[2]), D.mean()))
        prof.append((D, ddrift * 1e4, dabs * 1e3))
    R = np.array(rows)
    print(f"{arm} - {sys.argv[1]}: {len(R)} drift-frame tracks")
    print(f"   RMS shape change  e1 {np.median(R[:,0]):6.1f} um (median)  e2 {np.median(R[:,1]):6.1f} um")
    print(f"   mean d(drift distance) {R[:,2].mean():+6.2f} +- {R[:,2].std(ddof=1)/np.sqrt(len(R)):.2f} um   mean d|dz/ds| {R[:,3].mean():+7.3f} +- {R[:,3].std(ddof=1)/np.sqrt(len(R)):.3f} mrad")
    edges = np.arange(0, 32, 4.0); Dall = np.concatenate([p[0] for p in prof]); dd = np.concatenate([p[1] for p in prof]); ds = np.concatenate([p[2] for p in prof])
    line = '   by drift distance [cm]: ' + '  '.join(f"{lo:.0f}-{lo+4:.0f}: {dd[(Dall>=lo)&(Dall<lo+4)].mean():+5.1f}um/{ds[(Dall>=lo)&(Dall<lo+4)].mean():+6.2f}mrad" for lo in edges[:-1] if ((Dall>=lo)&(Dall<lo+4)).sum() > 50)
    print(line)
    # slope change vs track inclination
    for lo, hi in ((0, .1), (.1, .3), (.3, .6), (.6, 1.0)):
        m = (R[:,4] >= lo) & (R[:,4] < hi)
        if m.sum() > 10: print(f"   |u0z| {lo:.1f}-{hi:.1f}: n={m.sum():4d}  mean d|dz/ds| {R[m,3].mean():+7.3f} +- {R[m,3].std(ddof=1)/np.sqrt(m.sum()):.3f} mrad   mean d(drift) {R[m,2].mean():+6.2f} um")
