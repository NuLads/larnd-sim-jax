"""ENDFIT (g) readout (rule: REVIEW.md 25 Oct "ENDFIT (g)", written before any result): the (f) / (e) test repeated with
an INDEPENDENT per-track smear (LARND_ENDPOINT_SMEAR_KEY=entry, 1 mm), samples s = 1, 2, 4, paired with KN10N<s>:
  - EXP<s>: ends held; EZP<s>: ends fitted + per-batch zero-mean drift constraint;
  - each run: 1e4 fit -> production continuation <X>FR @ 14000 -> long continuation <X>L @ 44000.
Errors as read_long.py (mean of the last 1000 iterations / target - 1), at 14000 (<X>FR vs KN10N<s>FR: the production
end point) and 44000 (<X>L vs KN10N<s>L), with the trajectory; endpoint slope a~b of EZP from the smear STORED in the
checkpoint (training_history['endpoint_smear']), forward and reverse (b~a).
usage (container, PYTHONPATH = live tree): python3 read_g.py"""
exec(open('/sdf/scratch/neutrino/pgranger/review_2026_09/ENDFIT/read_long.py').read().split('\nH = {')[0])
import os
ARMS = tuple(os.environ.get('RG_ARMS', 'EXP,EZP').split(','))   # RG_ARMS=EXS,EZS: test on the legacy-smear runs


def hist_at(run, it):
    fs = sorted(glob.glob(f'{FR}/sci_full_{run}/history_iter{it}_*.pkl'))
    if not fs: raise SystemExit(f'{run}: no checkpoint at {it} -> block not complete, no readout')
    return pickle.load(open(fs[0], 'rb'))


def win(d, it, end):
    r = {}
    for p in P:
        a = np.array([f1(x) for x in d[f'{p}_iter']]); a = a[:len(a) - (end - it)]; t = f1(d[f'{p}_target'][0])
        r[p] = 100 * (a[-1000:].mean() / t - 1)
    pr = np.array([f1(x) for x in d['pos_residual_iter']]); r['pos'] = 1e4 * float(np.nanmean(pr[:len(pr) - (end - it)][-1000:]))
    return r


def slope_stored(d):
    sm = d.get('endpoint_smear'); A, B = {1: [], 2: []}, {1: [], 2: []}
    if sm is None: print('   (no stored smear: legacy (ev, tr) re-draw -- test runs only)')
    for bi in d['chain_cache']:
        for st, c in zip(d['chain_cache'][bi], d['chain_contexts'][bi]):
            a = np.asarray(st['angles'], np.float64); K = len(a) // 2
            u0 = _dir_from_angles(c['theta0_i'], c['phi0_i']); e1, e2 = _transverse_frame(u0)
            if sm is not None:
                s = sm[bi][tuple(c['track_id'])]
            else:
                g = np.random.default_rng(zlib.crc32(f"{tuple(c['track_id'])}|0".encode())); s = (g.normal(0., .1, 3), g.normal(0., .1, 3))
            for k in (0, 1):
                for ax, e, j in ((1, e1, K - 2 + k), (2, e2, 2 * K - 2 + k)):
                    B[ax].append(-np.asarray(s[k]) @ e); A[ax].append(a[j])
    out = {}
    for ax in (1, 2):
        a, b = np.array(A[ax]), np.array(B[ax])
        out[ax] = (float(a @ b / (b @ b)), float(a @ b / (a @ a)), 1e4 * float(np.sqrt(np.mean((a - b) ** 2))), 1e4 * float(np.sqrt(np.mean(b * b))))
    return out


Hs = {(a, s): hist_at(f'{a}{s}FR', 14000) for a in ARMS + ('KN10N',) for s in S}
Hl = {(a, s): hist_at(f'{a}{s}L', END) for a in ARMS + ('KN10N',) for s in S}
assert all(int(d['total_iter']) == 14000 for d in Hs.values()) and all(int(d['total_iter']) == END for d in Hl.values())
M = {}
for tag, H, end in (('14000 (<X>FR, production end point)', Hs, 14000), (f'{END} (<X>L)', Hl, END)):
    R = {k: win(d, end, end) for k, d in H.items()}
    print(f'\n=== @ {tag}')
    print(f"{'run':9s} " + ' '.join(f'{q:>9s}' for q in Q))
    for s in S:
        for a in ('KN10N',) + ARMS:
            print(f'{a}{s}      '[:9] + ' ' + ' '.join(f'{R[a, s][q]:+9.2f}' for q in Q))
    for a in ARMS:
        print(f'paired shift {a} - KN10N (s = 1 / 2 / 4; mean)')
        for q in Q:
            v = np.array([R[a, s][q] - R['KN10N', s][q] for s in S]); M[end, a, q] = v
            print(f'  {q:9s} ' + ' / '.join(f'{x:+7.2f}' for x in v) + f'   mean {v.mean():+7.2f}')
print('\ntrajectory of the paired mean shift, lifetime | long_diff (1000-iteration windows ending at it)')
for it in W:
    row = []
    for a in ARMS:
        v = {q: np.mean([win(Hl[a, s], it, END)[q] - win(Hl['KN10N', s], it, END)[q] for s in S]) for q in ('lifetime', 'long_diff')}
        row.append(f'{a} {v["lifetime"]:+7.2f} | {v["long_diff"]:+7.2f}')
    print(f'  {it:6d}  ' + '    '.join(row))
print(f'\n{ARMS[1]} endpoint slope a~b: fwd / rev (rms a-b; rms b) um, e1 | e2')
for s in S:
    for it, d in ((14000, Hs[ARMS[1], s]), (END, Hl[ARMS[1], s])):
        o = slope_stored(d)
        print(f'  {ARMS[1]}{s} @{it:6d}  ' + '  |  '.join(f'{o[ax][0]:.2f} / {o[ax][1]:.2f} ({o[ax][2]:4.0f}; {o[ax][3]:4.0f})' for ax in (1, 2)))
lx, dx = M[END, ARMS[0], 'lifetime'].mean(), M[END, ARMS[0], 'long_diff'].mean()
v1 = ('CONFIRMED' if abs(lx) <= 3 and abs(dx) <= 5 else 'NOT CONFIRMED' if lx >= 6 or dx >= 10 else 'WEAK')
g = M[END, ARMS[0], 'lifetime'] - M[END, ARMS[1], 'lifetime']; gd = M[END, ARMS[0], 'long_diff'] - M[END, ARMS[1], 'long_diff']
v2 = 'FITTING HELPS' if (g > 0).all() and (gd > 0).all() and g.mean() >= 1 else 'NO RESOLVED GAIN'
print(f'\n1. EXP (held, independent smear) @ {END}: lifetime {lx:+.2f}, long_diff {dx:+.2f} -> {v1}')
print(f'2. EXP - EZP @ {END}: lifetime ' + ' / '.join(f'{x:+.2f}' for x in g) + f' (mean {g.mean():+.2f}), long_diff '
      + ' / '.join(f'{x:+.2f}' for x in gd) + f' (mean {gd.mean():+.2f}) -> {v2}')
