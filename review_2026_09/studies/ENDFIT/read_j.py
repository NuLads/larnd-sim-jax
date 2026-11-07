"""ENDFIT (j) readout (rule: REVIEW.md 27 Oct "ENDFIT (j)", written before any result): is the FAST continuation enough
for REALISTIC endpoint errors (independent per-track 1 mm smear, LARND_ENDPOINT_SMEAR_KEY=entry, ends held)?
  EXP<s>H  = (h) fast continuation of EXP<s>FR @ 14000 -> 20000 (calibration LR 1e-2 held to 16500, then x0.805/epoch);
  EXP<s>HH = the same with the calibration LR held to 20000, then x0.805/epoch, -> 22000;  s = 1, 2, 4.
Reference: each run's own long continuation EXP<s>L @ 44000. Floor: the true-end fast runs' distance to their long
runs (KN10N<s>H @ 20000 vs KN10N<s>L @ 44000). Errors as read_fast.py (mean of the last 1000 iterations / target - 1).
usage (container, PYTHONPATH = live tree): python3 read_j.py      [test: RJ_BASE=EXS RJ_ARMS=H:20000,G:20000]"""
import os
exec(open('/sdf/scratch/neutrino/pgranger/review_2026_09/ENDFIT/read_long.py').read().split('\nH = {')[0])
BASE = os.environ.get('RJ_BASE', 'EXP')
ARMS = [(x.split(':')[0], int(x.split(':')[1])) for x in os.environ.get('RJ_ARMS', 'H:20000,HH:22000').split(',')]


def ck(run, it):
    fs = sorted(glob.glob(f'{FR}/sci_full_{run}/history_iter{it}_*.pkl'))
    if not fs: raise SystemExit(f'{run}: no checkpoint at {it} -> block not complete, no readout')
    d = pickle.load(open(fs[0], 'rb')); assert int(d['total_iter']) == it, (run, d['total_iter']); return d


def win(d, it, end, n=1000):
    r = {}
    for p in P:
        a = np.array([f1(x) for x in d[f'{p}_iter']]); a = a[:len(a) - (end - it)]; t = f1(d[f'{p}_target'][0])
        r[p] = 100 * (a[-n:].mean() / t - 1)
    pr = np.array([f1(x) for x in d['pos_residual_iter']]); r['pos'] = 1e4 * float(np.nanmean(pr[:len(pr) - (end - it)][-n:]))
    return r


R = {}
for s in S:
    dl = ck(f'{BASE}{s}L', 44000); R[BASE, s, 'FR'] = win(dl, 14000, 44000); R[BASE, s, 'L'] = win(dl, 44000, 44000)
    R[BASE, s, 'L20'] = win(dl, 20000, 44000)
    for a, e in ARMS:
        R[BASE, s, a] = win(ck(f'{BASE}{s}{a}', e), e, e)
    R['KN10N', s, 'H'] = win(ck(f'KN10N{s}H', 20000), 20000, 20000); R['KN10N', s, 'L'] = win(ck(f'KN10N{s}L', 44000), 44000, 44000)
TAGS = ['FR', 'L20'] + [a for a, _ in ARMS] + ['L']
print(f"{'run':10s} " + ' '.join(f'{q:>9s}' for q in Q) + '   (FR @ 14000, L20 = long run @ 20000, arms @ their end, L @ 44000)')
for s in S:
    for t in TAGS:
        print(f'{BASE}{s}{t}          '[:10] + ' ' + ' '.join(f'{R[BASE, s, t][q]:+9.2f}' for q in Q))
    for t in ('H', 'L'):
        print(f'KN10N{s}{t}          '[:10] + ' ' + ' '.join(f'{R["KN10N", s, t][q]:+9.2f}' for q in Q))


def dist(base, tag, q):
    return np.array([abs(R[base, s, tag][q] - R[base, s, 'L'][q]) for s in S])


fl, fd = dist('KN10N', 'H', 'lifetime'), dist('KN10N', 'H', 'long_diff')
print(f'\n|X - own L @ 44000| per sample (s = 1 / 2 / 4; mean), lifetime | long_diff')
print(f'  floor KN10N H  ' + ' / '.join(f'{x:5.2f}' for x in fl) + f'  mean {fl.mean():5.2f} | ' + ' / '.join(f'{x:5.2f}' for x in fd) + f'  mean {fd.mean():5.2f}')
V = {}
for t in ['FR', 'L20'] + [a for a, _ in ARMS]:
    dl_, dd_ = dist(BASE, t, 'lifetime'), dist(BASE, t, 'long_diff')
    print(f'  {BASE} {t:10s} ' + ' / '.join(f'{x:5.2f}' for x in dl_) + f'  mean {dl_.mean():5.2f} | ' + ' / '.join(f'{x:5.2f}' for x in dd_) + f'  mean {dd_.mean():5.2f}')
    V[t] = (dl_.mean(), dd_.mean())
print('\npaired shift BASE - KN10N (s = 1 / 2 / 4; mean)')
for tag, kt in (('H', 'H'), ('L', 'L')):
    if (BASE, 1, tag) not in R: continue
    for q in ('lifetime', 'long_diff', 'pos'):
        v = np.array([R[BASE, s, tag][q] - R['KN10N', s, kt][q] for s in S])
        print(f'  {tag:3s} {q:9s} ' + ' / '.join(f'{x:+7.2f}' for x in v) + f'   mean {v.mean():+7.2f}')
print('\nVERDICTS (SUFFICIENT: mean |X - L| <= 2.5 lifetime and <= 3 long_diff; NOT SUFFICIENT: >= 4 or >= 6; else MARGINAL)')
for a, e in ARMS:
    l, d = V[a]
    v = 'SUFFICIENT' if l <= 2.5 and d <= 3 else 'NOT SUFFICIENT' if l >= 4 or d >= 6 else 'MARGINAL'
    print(f'  {BASE}{a} @ {e}: |X - L| lifetime {l:.2f}, long_diff {d:.2f} (floor {fl.mean():.2f} / {fd.mean():.2f}) -> {v}')
