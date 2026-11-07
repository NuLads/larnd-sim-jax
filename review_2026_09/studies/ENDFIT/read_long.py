"""ENDFIT (e) readout (rule: REVIEW.md 25 Oct "ENDFIT (e)", written before any result). EZS<s>L vs KN10N<s>L @ 44000
(long continuation of <X>FR @ 14000), s = 1, 2, 4, paired by sample.
  - errors % = mean of the last 1000 iterations / target - 1 (read_ens.read), at 44000 and, from the same history, in
    the 1000-iteration windows ending at 14000, 16000, 20000, ..., 44000 (trajectory of the paired shift);
  - endpoint slope a~b per axis (shrink.py convention) at 14000 (<X>FR), at the kept checkpoints (ENDFIT/long_ckpt/)
    and at 44000;
  - verdict by the pre-registered rule.
usage (in the container, PYTHONPATH = the live tree): python3 read_long.py"""
import glob, zlib, pickle, numpy as np
from optimize.fit_params import _dir_from_angles, _transverse_frame
exec(open('/sdf/scratch/neutrino/pgranger/review_2026_09/KNOT/read_ens.py').read().split('\nS = list')[0])
LK = '/sdf/scratch/neutrino/pgranger/review_2026_09/ENDFIT/long_ckpt'
S, Q, END = (1, 2, 4), P + ['pos'], 44000
W = [14000, 16000] + list(range(20000, END + 1, 4000))


def hist(run):
    fs = sorted(glob.glob(f'{FR}/sci_full_{run}/history_iter{END}_*.pkl'))
    if not fs: raise SystemExit(f'{run}: no checkpoint at {END} -> block not complete, no readout')
    return pickle.load(open(fs[0], 'rb'))


def window(d, it):
    r = {}
    for p in P:
        a = np.array([f1(x) for x in d[f'{p}_iter']]); a = a[:len(a) - (END - it)]; t = f1(d[f'{p}_target'][0])
        r[p] = 100 * (a[-1000:].mean() / t - 1)
    pr = np.array([f1(x) for x in d['pos_residual_iter']]); r['pos'] = 1e4 * float(np.nanmean(pr[:len(pr) - (END - it)][-1000:]))
    return r


def slope(d, smear_mm=1.0):
    A, B = {1: [], 2: []}, {1: [], 2: []}
    for bi in d['chain_cache']:
        for st, c in zip(d['chain_cache'][bi], d['chain_contexts'][bi]):
            a = np.asarray(st['angles'], np.float64); K = len(a) // 2
            u0 = _dir_from_angles(c['theta0_i'], c['phi0_i']); e1, e2 = _transverse_frame(u0)
            g = np.random.default_rng(zlib.crc32(f"{tuple(c['track_id'])}|0".encode()))
            sm = (g.normal(0., .1 * smear_mm, 3), g.normal(0., .1 * smear_mm, 3))
            for k in (0, 1):
                for ax, e, j in ((1, e1, K - 2 + k), (2, e2, 2 * K - 2 + k)):
                    B[ax].append(-sm[k] @ e); A[ax].append(a[j])
    out = {}
    for ax in (1, 2):
        a, b = np.array(A[ax]), np.array(B[ax])
        out[ax] = (float(a @ b / (b @ b)), 1e4 * float(np.sqrt(np.mean((a - b) ** 2))))
    return out


H = {(a, s): hist(f'{a}{s}L') for a in ('EZS', 'KN10N') for s in S}
assert all(int(d['total_iter']) == END for d in H.values()), {k: d['total_iter'] for k, d in H.items()}
R = {k: window(d, END) for k, d in H.items()}
print(f"{'run':9s} " + ' '.join(f'{q:>9s}' for q in Q) + f'   (@ {END})')
for s in S:
    for a in ('KN10N', 'EZS'):
        print(f'{a}{s}L    '[:9] + ' ' + ' '.join(f'{R[a, s][q]:+9.2f}' for q in Q))
print(f'\npaired shift EZS - KN10N @ {END} (s = 1 / 2 / 4; mean)')
M = {}
for q in Q:
    v = np.array([R['EZS', s][q] - R['KN10N', s][q] for s in S]); M[q] = v
    print(f'  {q:9s} ' + ' / '.join(f'{x:+7.2f}' for x in v) + f'   mean {v.mean():+7.2f}')
print('\ntrajectory of the paired mean shift (1000-iteration windows ending at it)')
print(f"{'it':>6s} " + ' '.join(f'{q:>9s}' for q in Q))
for it in W:
    v = {q: np.mean([window(H['EZS', s], it)[q] - window(H['KN10N', s], it)[q] for s in S]) for q in Q}
    print(f'{it:6d} ' + ' '.join(f'{v[q]:+9.2f}' for q in Q))
print('\nendpoint slope a~b (rms a-b um), e1 | e2, per sample')
SL = {}
for it in W:
    row = []
    for s in S:
        if it == 14000:
            fs = sorted(glob.glob(f'{FR}/sci_full_EZS{s}FR/history_iter14000_*.pkl'))
        elif it == END:
            fs = ['END']
        else:
            fs = sorted(glob.glob(f'{LK}/sci_full_EZS{s}L/history_iter{it}_*.pkl'))
        if not fs:
            row.append('      (no checkpoint)       '); continue
        o = slope(H['EZS', s] if fs[0] == 'END' else pickle.load(open(fs[0], 'rb'))); SL[it, s] = o
        row.append(f'{o[1][0]:.2f} ({o[1][1]:4.0f}) | {o[2][0]:.2f} ({o[2][1]:4.0f})')
    print(f'{it:6d}  ' + '   '.join(row))
conv = all(SL[END, s][ax][0] >= 0.9 for s in S for ax in (1, 2))
lt, ld = M['lifetime'].mean(), M['long_diff'].mean()
if conv and abs(lt) <= 3 and abs(ld) <= 5: v = 'CONVERGED ENDS RECOVER'
elif conv and lt >= 8: v = 'CONVERGED ENDS DO NOT RECOVER'
elif not conv: v = 'STILL CONVERGING'
else: v = 'PARTIAL RECOVERY'
print(f'\nconverged (slope >= 0.9 on e1 and e2, all samples): {conv}; mean lifetime shift {lt:+.2f}, long_diff {ld:+.2f} -> {v}')
