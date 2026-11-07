"""ENDFIT (f) readout (rule: REVIEW.md 25 Oct "ENDFIT (f)", written before any result): the two long-continuation
controls of (e), paired by sample with KN10N<s>L @ 44000, s = 1, 2, 4.
  - EXS<s>L: 1 mm smear, ends HELD (not fitted). Does the long run alone remove the 14000 bias?
  - EZT<s>L: true ends, fitted, per-batch zero-mean drift constraint. Does the anchor hold over 30000 more iterations?
Errors and windows as read_long.py (mean of the last 1000 iterations / target - 1). For EZT also the size of the fitted
endpoint coefficients (truth = 0): rms per axis, and the mean drift-distance shift (+ = away from the anode; EFT0,
unconstrained, had +32..+44 um).
usage (container, PYTHONPATH = live tree): python3 read_ctrl.py            [test: RC_END=14000 RC_SUF=FR RC_REF=FR]"""
import os
exec(open('/sdf/scratch/neutrino/pgranger/review_2026_09/ENDFIT/read_long.py').read().split('\nH = {')[0])
END = int(os.environ.get('RC_END', '44000')); SUF = os.environ.get('RC_SUF', 'L'); REF = os.environ.get('RC_REF', 'L')
W = [w for w in [14000, 16000] + list(range(20000, 44001, 4000)) if w <= END]
ZB = np.array([[30.59, 0.159], [-30.59, -0.159]])


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


def ends(d):
    """fitted endpoint coefficients (cm) of a true-endpoint run: rms per axis, mean drift-distance shift."""
    A1, A2, DL = [], [], []
    for bi in d['chain_cache']:
        for st, c in zip(d['chain_cache'][bi], d['chain_contexts'][bi]):
            a = np.asarray(st['angles'], np.float64); K = len(a) // 2
            u0 = _dir_from_angles(c['theta0_i'], c['phi0_i']); e1, e2 = _transverse_frame(u0)
            x0 = np.asarray(c['x0_fixed'], float); xs = (x0, x0 + c['total_len'] * u0)
            for k in (0, 1):
                z = xs[k][2]; t = int(np.argmin(np.abs(ZB.mean(1) - z)))
                A1.append(a[K - 2 + k]); A2.append(a[2 * K - 2 + k]); DL.append(float(e2[2]) * np.sign(z - ZB[t, 0]) * a[2 * K - 2 + k])
    um = lambda v: 1e4 * float(np.sqrt(np.mean(np.square(v))))
    return um(A1), um(A2), 1e4 * float(np.mean(DL))


ARMS = ('EXS', 'EZT')
H = {(a, s): hist(f'{a}{s}{SUF}') for a in ARMS for s in S}
H.update({('KN10N', s): hist(f'KN10N{s}{REF}') for s in S})
assert all(int(d['total_iter']) == END for d in H.values()), {k: d['total_iter'] for k, d in H.items()}
R = {k: window(d, END) for k, d in H.items()}
print(f"{'run':9s} " + ' '.join(f'{q:>9s}' for q in Q) + f'   (@ {END})')
for s in S:
    for a in ('KN10N',) + ARMS:
        print(f'{a}{s}{SUF if a != "KN10N" else REF}    '[:9] + ' ' + ' '.join(f'{R[a, s][q]:+9.2f}' for q in Q))
M = {}
for a in ARMS:
    print(f'\npaired shift {a} - KN10N @ {END} (s = 1 / 2 / 4; mean)')
    for q in Q:
        v = np.array([R[a, s][q] - R['KN10N', s][q] for s in S]); M[a, q] = v
        print(f'  {q:9s} ' + ' / '.join(f'{x:+7.2f}' for x in v) + f'   mean {v.mean():+7.2f}')
    print(f'  trajectory of the paired mean shift (1000-iteration windows ending at it)')
    print(f"  {'it':>6s} " + ' '.join(f'{q:>9s}' for q in Q))
    for it in W:
        v = {q: np.mean([window(H[a, s], it)[q] - window(H['KN10N', s], it)[q] for s in S]) for q in Q}
        print(f'  {it:6d} ' + ' '.join(f'{v[q]:+9.2f}' for q in Q))
print('\nEZT fitted endpoint coefficients (truth = 0): rms e1 / rms e2 / mean drift shift (+ = away from anode), um')
for s in S:
    fs = sorted(glob.glob(f'{FR}/sci_full_EZT{s}FR/history_iter14000_*.pkl'))
    o0 = ends(pickle.load(open(fs[0], 'rb'))) if fs else (np.nan,) * 3; o1 = ends(H['EZT', s])
    print(f'  EZT{s}: @14000 {o0[0]:5.0f} / {o0[1]:5.0f} / {o0[2]:+6.1f}   @{END} {o1[0]:5.0f} / {o1[1]:5.0f} / {o1[2]:+6.1f}')
lx, dx = M['EXS', 'lifetime'].mean(), M['EXS', 'long_diff'].mean()
vx = ('HELD ENDS STAY BIASED' if lx >= 8 else 'LONG RUN ALONE RECOVERS' if abs(lx) <= 3 and abs(dx) <= 5 else 'PARTIAL')
lz, mz = M['EZT', 'lifetime'].mean(), M['EZT', 'lifetime'].max()
vz = 'ANCHOR HOLDS LONG' if abs(lz) <= 3 and mz <= 6 else 'FREEDOM COST GROWS' if lz >= 5 else 'MARGINAL'
print(f'\nEXS (held smear): mean lifetime shift {lx:+.2f}, long_diff {dx:+.2f} -> {vx}')
print(f'EZT (true ends fitted): mean lifetime shift {lz:+.2f}, max {mz:+.2f} -> {vz}')
