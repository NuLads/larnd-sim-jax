"""ENDFIT (h2) readout (rule: REVIEW.md 26 Oct "ENDFIT (h2)", written before any result): <X>H2 @ 20000 = the (h) fast
continuation with the chain LR budget matched to the long run's (1e-2 held 14000-17000, -> 6e-4 at 20000), vs <X>H
(chain 2e-3 -> 6e-4) and <X>L @ 44000, all from <X>FR @ 14000; X = EXS<s> (1 mm smear, ends held), KN10N<s> (true
ends), s = 1, 2, 4. Errors as read_long.py (mean of the last 1000 iterations / target - 1); nuisance relaxation
c = <Y - FR, L - FR> / |L - FR|^2 per run (real tracks only), Y = H, H2.
usage (container, PYTHONPATH = live tree): python3 read_h2.py        [test: RH2_SUF=H, the (h) runs as stand-ins]"""
import os
exec(open('/sdf/scratch/neutrino/pgranger/review_2026_09/ENDFIT/read_long.py').read().split('\nH = {')[0])
SUF = os.environ.get('RH2_SUF', 'H2'); RUNS = ('KN10N', 'EXS'); EH, EL = 20000, 44000


def ck(run, it):
    fs = sorted(glob.glob(f'{FR}/sci_full_{run}/history_iter{it}_*.pkl'))
    if not fs: raise SystemExit(f'{run}: no checkpoint at {it} -> block not complete, no readout')
    return pickle.load(open(fs[0], 'rb'))


def win(d, it, end, n=1000):
    r = {}
    for p in P:
        a = np.array([f1(x) for x in d[f'{p}_iter']]); a = a[:len(a) - (end - it)]; t = f1(d[f'{p}_target'][0])
        r[p] = 100 * (a[-n:] / t - 1).mean()
    pr = np.array([f1(x) for x in d['pos_residual_iter']]); r['pos'] = 1e4 * float(np.nanmean(pr[:len(pr) - (end - it)][-n:]))
    return r


def nuis(d):
    g = {}
    for bi in d['chain_cache']:
        for st, c in zip(d['chain_cache'][bi], d['chain_contexts'][bi]):
            t = tuple(int(x) for x in c['track_id'])
            if t != (-1, -1): g[bi, t] = np.asarray(st['angles'], np.float64)
    return g, {bi: np.asarray(v['log_dedx'], np.float64).ravel() for bi, v in d['dedx_cache'].items()}


def relax(f, y, l):
    out = []
    for a, b, c in zip(f, y, l):
        ks = sorted(set(a) & set(b) & set(c), key=str); assert len(ks) == len(a) == len(b) == len(c)
        va, vb, vc = (np.concatenate([x[k] for k in ks]) for x in (a, b, c)); dl = vc - va
        out.append(float((vb - va) @ dl / (dl @ dl)))
    return out


C = {}
for a in RUNS:
    for s in S:
        C[a, s] = {'FR': ck(f'{a}{s}FR', 14000), 'H': ck(f'{a}{s}H', EH), 'H2': ck(f'{a}{s}{SUF}', EH), 'L': ck(f'{a}{s}L', EL)}
        assert int(C[a, s]['H2']['total_iter']) == EH and int(C[a, s]['L']['total_iter']) == EL
END = {'FR': EL, 'H': EH, 'H2': EH, 'L': EL}; AT = {'FR': 14000, 'H': EH, 'H2': EH, 'L': EL}
SRC = {'FR': 'L', 'H': 'H', 'H2': 'H2', 'L': 'L'}   # the 14000 window is read from the L history (as read_fast.py)
R = {(a, s, t): win(C[a, s][SRC[t]], AT[t], END[t]) for a in RUNS for s in S for t in END}
print(f"{'run':11s} " + ' '.join(f'{q:>9s}' for q in Q) + f'   (FR @ 14000 / H, {SUF} @ 20000 / L @ 44000)')
for s in S:
    for a in RUNS:
        for t in ('FR', 'H', 'H2', 'L'):
            print(f'{a}{s}{t}        '[:11] + ' ' + ' '.join(f'{R[a, s, t][q]:+9.2f}' for q in Q))
M = {}
print('\npaired shift EXS - KN10N (s = 1 / 2 / 4; mean)')
for t in ('FR', 'H', 'H2', 'L'):
    for q in ('lifetime', 'long_diff', 'tran_diff', 'Ab', 'pos'):
        v = np.array([R['EXS', s, t][q] - R['KN10N', s, t][q] for s in S]); M[t, q] = v
        print(f'  {t:3s} {q:9s} ' + ' / '.join(f'{x:+7.2f}' for x in v) + f'   mean {v.mean():+7.2f}')
print(f'\ntrajectory, {SUF}: paired mean shift lifetime | long_diff, and the KN10N lifetime jitter (sd in the window, %)')
for it in range(15000, EH + 1, 1000):
    v = {q: np.mean([win(C['EXS', s]['H2'], it, EH)[q] - win(C['KN10N', s]['H2'], it, EH)[q] for s in S]) for q in ('lifetime', 'long_diff')}
    j = []
    for s in S:
        a = np.array([f1(x) for x in C['KN10N', s]['H2']['lifetime_iter']]); a = a[:len(a) - (EH - it)][-1000:]; j.append(100 * a.std() / a.mean())
    print(f'  {it:6d}  {v["lifetime"]:+7.2f} | {v["long_diff"]:+7.2f}   jitter {np.mean(j):.2f}')
print('\nnuisance relaxation c (geometry | log dE/dx), H vs ' + SUF)
CG = {}
for a in RUNS:
    for s in S:
        n = {t: nuis(C[a, s][t]) for t in ('FR', 'H', 'H2', 'L')}
        cH, cH2 = relax(n['FR'], n['H'], n['L']), relax(n['FR'], n['H2'], n['L']); CG[a, s] = cH2[0]
        print(f'  {a}{s:<2d}  H {cH[0]:.2f} | {cH[1]:.2f}    {SUF} {cH2[0]:.2f} | {cH2[1]:.2f}')
D = {(a, s): (R[a, s, 'H2']['lifetime'] - R[a, s, 'L']['lifetime'], R[a, s, 'H2']['long_diff'] - R[a, s, 'L']['long_diff']) for a in RUNS for s in S}
print(f'\n{SUF} @ 20000 - L @ 44000 (lifetime, long_diff): ' + '  '.join(f'{a}{s} {D[a, s][0]:+.2f}, {D[a, s][1]:+.2f}' for a in RUNS for s in S))
print('KN10N position residual um, H / ' + SUF + ' / L: ' + '  '.join(f's{s} {R["KN10N", s, "H"]["pos"]:.1f} / {R["KN10N", s, "H2"]["pos"]:.1f} / {R["KN10N", s, "L"]["pos"]:.1f}' for s in S))
lx, dx = M['H2', 'lifetime'].mean(), M['H2', 'long_diff'].mean()
v1 = 'FAST RECOVERS' if abs(lx) <= 3 and abs(dx) <= 5 else 'STILL SHORT' if lx >= 5 else 'PARTIAL'
dl = M['H2', 'lifetime'] - M['H', 'lifetime']
v2 = 'GEOMETRY LR HELPS' if (dl < 0).all() and dl.mean() <= -2 else 'NO RESOLVED GAIN'
cg = np.array(list(CG.values()))
v3 = 'GEOMETRY LR-BUDGET-LIMITED' if (cg >= 0.7).all() else 'GEOMETRY VISIT-LIMITED' if (cg < 0.5).all() else 'MIXED'
print(f'\n1. EXS - KN10N @ 20000 ({SUF}): lifetime {lx:+.2f}, long_diff {dx:+.2f} -> {v1}')
print(f'2. change vs H, lifetime: ' + ' / '.join(f'{x:+.2f}' for x in dl) + f' (mean {dl.mean():+.2f}) -> {v2}')
print(f'3. geometry relaxation c ({SUF}), min {cg.min():.2f} max {cg.max():.2f} -> {v3}')
