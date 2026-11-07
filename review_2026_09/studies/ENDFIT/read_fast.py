"""ENDFIT (h) readout (rule: REVIEW.md 26 Oct "ENDFIT (h)", written before any result): FAST continuation <X>H @ 20000
(calibration LR 1e-2, 6000 iterations) vs the LONG one <X>L @ 44000 (1e-3, 30000), both from <X>FR @ 14000;
X = EXS<s> (1 mm smear, ends held) and KN10N<s> (true ends), s = 1, 2, 4.
Errors as read_long.py (mean of the last 1000 iterations / target - 1).
usage (container, PYTHONPATH = live tree): python3 read_fast.py"""
exec(open('/sdf/scratch/neutrino/pgranger/review_2026_09/ENDFIT/read_long.py').read().split('\nH = {')[0])
RUNS = ('KN10N', 'EXS'); EH, EL = 20000, 44000


def hist_at(run, it):
    fs = sorted(glob.glob(f'{FR}/sci_full_{run}/history_iter{it}_*.pkl'))
    if not fs: raise SystemExit(f'{run}: no checkpoint at {it} -> block not complete, no readout')
    return pickle.load(open(fs[0], 'rb'))


def win(d, it, end, n=1000):
    r, sd = {}, {}
    for p in P:
        a = np.array([f1(x) for x in d[f'{p}_iter']]); a = a[:len(a) - (end - it)]; t = f1(d[f'{p}_target'][0])
        x = 100 * (a[-n:] / t - 1); r[p] = x.mean(); sd[p] = x.std()
    pr = np.array([f1(x) for x in d['pos_residual_iter']]); r['pos'] = 1e4 * float(np.nanmean(pr[:len(pr) - (end - it)][-n:]))
    return r, sd


HH = {(a, s): hist_at(f'{a}{s}H', EH) for a in RUNS for s in S}
HL = {(a, s): hist_at(f'{a}{s}L', EL) for a in RUNS for s in S}
assert all(int(d['total_iter']) == EH for d in HH.values()) and all(int(d['total_iter']) == EL for d in HL.values())
RH = {k: win(d, EH, EH)[0] for k, d in HH.items()}; RL = {k: win(d, EL, EL)[0] for k, d in HL.items()}
R0 = {k: win(d, 14000, EL)[0] for k, d in HL.items()}
print(f"{'run':10s} " + ' '.join(f'{q:>9s}' for q in Q) + '   (FR @ 14000 / H @ 20000 / L @ 44000)')
for s in S:
    for a in RUNS:
        for tag, R in (('FR', R0), ('H', RH), ('L', RL)):
            print(f'{a}{s}{tag}      '[:10] + ' ' + ' '.join(f'{R[a, s][q]:+9.2f}' for q in Q))
print('\npaired shift EXS - KN10N (s = 1 / 2 / 4; mean)')
M = {}
for tag, R in (('14000', R0), ('H 20000', RH), ('L 44000', RL)):
    for q in ('lifetime', 'long_diff', 'tran_diff', 'Ab', 'pos'):
        v = np.array([R['EXS', s][q] - R['KN10N', s][q] for s in S]); M[tag, q] = v
        print(f'  {tag:8s} {q:9s} ' + ' / '.join(f'{x:+7.2f}' for x in v) + f'   mean {v.mean():+7.2f}')
print('\ntrajectory, fast run: paired mean shift lifetime | long_diff, and the KN10N lifetime jitter (sd within the window, %)')
for it in range(15000, EH + 1, 1000):
    v = {q: np.mean([win(HH['EXS', s], it, EH)[0][q] - win(HH['KN10N', s], it, EH)[0][q] for s in S]) for q in ('lifetime', 'long_diff')}
    j = np.mean([win(HH['KN10N', s], it, EH)[1]['lifetime'] for s in S])
    print(f'  {it:6d}  {v["lifetime"]:+7.2f} | {v["long_diff"]:+7.2f}   jitter {j:.2f}')
D = {(a, s): (RH[a, s]['lifetime'] - RL[a, s]['lifetime'], RH[a, s]['long_diff'] - RL[a, s]['long_diff']) for a in RUNS for s in S}
print('\nH @ 20000 - L @ 44000, per run (lifetime, long_diff): ' + '  '.join(f'{a}{s} {D[a, s][0]:+.2f}, {D[a, s][1]:+.2f}' for a in RUNS for s in S))
lx, dx = M['H 20000', 'lifetime'].mean(), M['H 20000', 'long_diff'].mean()
v1 = 'FAST RECOVERS' if abs(lx) <= 3 and abs(dx) <= 5 else 'FAST DOES NOT RECOVER' if lx >= 8 else 'PARTIAL'
v2 = 'SAME END STATE' if all(abs(x) <= 3 and abs(y) <= 5 for x, y in D.values()) else 'DIFFERENT END STATE'
print(f'\n1. EXS - KN10N @ 20000: lifetime {lx:+.2f}, long_diff {dx:+.2f} -> {v1}')
print(f'2. fast vs long, all 6 runs within 3 (lifetime) / 5 (long_diff) points -> {v2}')
