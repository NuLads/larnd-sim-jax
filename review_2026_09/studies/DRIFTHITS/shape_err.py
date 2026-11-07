"""Per-run drift-shape error: RMS over the track of (fitted - truth-projected) e2 offset, drift-frame tracks.
usage: shape_err.py RUN [RUN...]   (fit_result/sci_full_<RUN>)"""
import sys, glob, pickle, numpy as np
sys.path.insert(0, '/sdf/group/neutrino/pgranger/larnd-sim-jax')
from optimize.fit_params import _dir_from_angles, _transverse_frame, _DRIFT_AXIS
for run in sys.argv[1:]:
    fs = sorted(glob.glob(f'fit_result/sci_full_{run}/history_iter*.pkl'), key=lambda x: int(x.split('iter')[-1].split('_')[0]))
    f = [x for x in fs if int(x.split('iter')[-1].split('_')[0]) <= 10000][-1]; ck = pickle.load(open(f, 'rb'))
    E, E0, Ea = [], [], []
    for b, ctxs in ck['chain_contexts'].items():
        for c, st in zip(ctxs, ck['chain_cache'][b]):
            u0 = _dir_from_angles(c['theta0_i'], c['phi0_i']); e1, e2 = _transverse_frame(u0)
            if not (abs(e1[2]) < 1e-6 and np.linalg.norm(np.cross(_DRIFT_AXIS, u0)) > 1e-3): continue
            ang = np.asarray(st['angles'], float); K = len(ang) // 2
            tr = np.asarray(ck['true_positions'][b][str(tuple(c['track_id']))], float); al = np.asarray(c['alpha_mid'], float)[:len(tr)]; tr = tr[:len(al)]
            x0 = np.asarray(c['x0_fixed'], float); L = c['total_len']
            B = np.sin(np.outer(al, np.pi * np.arange(1, K + 1))); b2t = np.linalg.lstsq(B, (tr - x0 - al[:, None] * L * u0) @ e2, rcond=None)[0]
            ag = np.linspace(0, 1, 101); Bg = np.sin(np.outer(ag, np.pi * np.arange(1, K + 1)))
            E.append(np.sqrt(np.mean((Bg @ (ang[K:] - b2t))**2)) * 1e4); E0.append(np.sqrt(np.mean((Bg @ b2t)**2)) * 1e4)
            Ea.append(np.mean(Bg @ (ang[K:] - b2t)) * 1e4)
    print(f"{run:16s} {f.split('/')[-1][:18]} ntrk {len(E)}  shape err median {np.median(E):6.1f} um  mean {np.mean(E):6.1f}  | straight-line {np.median(E0):6.1f} | signed mean offset {np.mean(Ea):+6.1f}")
