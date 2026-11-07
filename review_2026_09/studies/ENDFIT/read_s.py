"""ENDFIT (s) readout: the TRUE-END check of the longer coarse phase C75 (LARND_CHAIN_COARSE_ITER=7500) against TRUTH (rules:
REVIEW.md 30 Oct "ENDFIT (s)", written before any result; the (o) rule). Samples KN10N3, 5, 6, 7, 8, 9 (true endpoints);
arms at H @ 20000: P = production re-run of 29 Oct (the paired reference, (o)), CF5000 = KN10N<s>CFH ((o)), C75 =
KN10N<s>C75H. Error % = last-1000 mean / target - 1; pos = mean pos_residual over the last 1000 iterations (um, vs truth).
usage (container): python3 read_s.py   [RS_DRY=1: the CF5000 arm stands in for C75, to test the script]"""
import glob, os, pickle, numpy as np
FR = '/sdf/group/neutrino/pgranger/larnd-sim-jax/fit_result'
P = ['Ab', 'eField', 'tran_diff', 'long_diff', 'lifetime']
f1 = lambda x: np.nan if x is None else float(np.mean(x))
DRY = os.environ.get('RS_DRY') == '1'
S = [3, 5, 6, 7, 8, 9]


def read(run, it=20000):
    fs = sorted(glob.glob(f'{FR}/sci_full_{run}/history_iter{it}_*.pkl'))
    if not fs: raise SystemExit(f'{run}: no checkpoint at {it} -> set not complete, no readout')
    d = pickle.load(open(fs[0], 'rb')); assert int(d['total_iter']) == it, run
    r = {p: 100 * (np.array([f1(x) for x in d[f'{p}_iter']])[-1000:].mean() / f1(d[f'{p}_target'][0]) - 1) for p in P}
    r['pos'] = 1e4 * float(np.nanmean(np.array([f1(x) for x in d['pos_residual_iter']])[-1000:]))
    r['env'] = {k: d.get('env_overrides', {}).get(k) for k in ('LARND_CHAIN_COARSE_ITER', 'LARND_ENDPOINT_SMEAR_MM')}
    return r


A = ('P', 'CF5000', 'C75')
RUN = {'P': lambda s: f'KN10N{s}PH', 'CF5000': lambda s: f'KN10N{s}CFH', 'C75': lambda s: f'KN10N{s}CFH' if DRY else f'KN10N{s}C75H'}
R = {s: {a: read(RUN[a](s)) for a in A} for s in S}
want = {'P': (None, '', '0'), 'CF5000': ('5000',), 'C75': ('5000',) if DRY else ('7500',)}
for s in S:
    for a in A:
        e = R[s][a]['env']
        assert e['LARND_ENDPOINT_SMEAR_MM'] in (None, '', '0', '0.0'), (s, a, e)
        assert e['LARND_CHAIN_COARSE_ITER'] in want[a], (s, a, e)
print('config check OK on 6 x 3 runs (true ends; coarse iter P none, CF 5000, C75 7500)' + (' -- DRY RUN' if DRY else ''))
Q = P + ['pos']
print('\n   ' + ' '.join(f'{q:>23s}' for q in Q) + '   (P / CF5000 / C75; error % vs truth, pos um)')
for s in S:
    print(f'{s:2d} ' + ' '.join('/'.join(f'{R[s][a][q]:+6.2f}' if q != 'pos' else f'{R[s][a][q]:6.1f}' for a in A) for q in Q))
ms = lambda v: (v.mean(), v.std(ddof=1) / np.sqrt(len(v)))
worse = {'CF5000': [], 'C75': []}
print('\nmean |err| P / CF5000 / C75 (pos: mean um), paired d vs P +- SE, and C75 - CF5000')
for q in Q:
    v = {a: np.array([abs(R[s][a][q]) if q != 'pos' else R[s][a][q] for s in S]) for a in A}
    line = f'  {q:10s} ' + ' / '.join(f'{v[a].mean():6.2f}' for a in A)
    for a in ('CF5000', 'C75'):
        m, se = ms(v[a] - v['P']); w = q != 'pos' and m > max(2 * se, 0.5)
        if w: worse[a].append(q)
        line += f'   {a}-P {m:+6.2f} +- {se:4.2f}' + (' WORSE' if w else '      ')
    m, se = ms(v['C75'] - v['CF5000']); line += f'   C75-CF5000 {m:+6.2f} +- {se:4.2f}'
    print(line)
pp = {a: np.array([R[s][a]['pos'] for s in S]) for a in A}
for a in ('CF5000', 'C75'):
    cl = int((pp[a] < pp['P']).sum())
    print(f"{a}: positions below P {cl}/6 (gain {float((pp['P'] - pp[a]).mean()):+.1f} um) -> POSITION "
          f"{'CONFIRMED' if cl >= 5 else 'not confirmed'}; CALIBRATION {'WORSE in ' + ', '.join(worse[a]) if worse[a] else 'NOT WORSE'}"
          f" -> {'CONFIRMED' if cl >= 5 and not worse[a] else 'NOT CONFIRMED'}")
cl = int((pp['C75'] < pp['P']).sum())
print(f"VERDICT: C75 on true ends {'CONFIRMED' if cl >= 5 and not worse['C75'] else 'NOT CONFIRMED'}  [control: CF5000 should "
      f"reproduce (o): CONFIRMED]")
