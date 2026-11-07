"""ENDFIT (b) diagnosis (25 Oct): is the partial endpoint recovery in EZS shrinkage toward the given (smeared) point
(prior- or convergence-limited) or a data floor? Per endpoint and transverse axis: b = true - given (the smear), a =
fitted endpoint coefficient (ideal a = b). Slope of a on b, scatter about the line, at the fit end (10000) and after
the continuation (14000); EZT (b = 0) gives the pure data-driven displacement.
usage (in the container, PYTHONPATH = the live tree): python3 shrink.py"""
import zlib, glob, pickle, numpy as np
from optimize.fit_params import _dir_from_angles, _transverse_frame
FR = '/sdf/group/neutrino/pgranger/larnd-sim-jax/fit_result'


def pts(run, it, smear_mm):
    d = pickle.load(open(sorted(glob.glob(f'{FR}/sci_full_{run}/history_iter{it}_*.pkl'))[0], 'rb'))
    A, B = {1: [], 2: []}, {1: [], 2: []}
    for bi in d['chain_cache']:
        for st, c in zip(d['chain_cache'][bi], d['chain_contexts'][bi]):
            a = np.asarray(st['angles'], np.float64); K = len(a) // 2
            u0 = _dir_from_angles(c['theta0_i'], c['phi0_i']); e1, e2 = _transverse_frame(u0)
            g = np.random.default_rng(zlib.crc32(f"{tuple(c['track_id'])}|0".encode()))
            sm = (g.normal(0., .1 * smear_mm, 3), g.normal(0., .1 * smear_mm, 3)) if smear_mm else (np.zeros(3),) * 2
            for k in (0, 1):
                for ax, e, j in ((1, e1, K - 2 + k), (2, e2, 2 * K - 2 + k)):
                    B[ax].append(-sm[k] @ e); A[ax].append(a[j])
    return {ax: (np.array(A[ax]), np.array(B[ax])) for ax in (1, 2)}


print('run     iter  axis   n    slope a~b   scatter um   rms(a-b) um   rms a um   corr')
for s in (1, 2, 4):
    for run, it, sm in ((f'EZS{s}', 10000, 1.), (f'EZS{s}FR', 14000, 1.), (f'EZT{s}', 10000, 0.), (f'EZT{s}FR', 14000, 0.)):
        for ax, (a, b) in pts(run, it, sm).items():
            if sm:
                sl = float(a @ b / (b @ b)); sc = 1e4 * np.std(a - sl * b); cr = np.corrcoef(a, b)[0, 1]
            else:
                sl, sc, cr = np.nan, 1e4 * np.std(a), np.nan
            print(f'{run:7s} {it:5d}  e{ax}   {len(a):4d}   {sl:9.3f}   {sc:9.1f}   {1e4 * np.sqrt(np.mean((a - b) ** 2)):10.1f}   '
                  f'{1e4 * np.sqrt(np.mean(a ** 2)):8.1f}   {cr:5.2f}')
