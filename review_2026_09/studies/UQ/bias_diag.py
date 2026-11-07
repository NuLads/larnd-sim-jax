"""Bias diagnostics from existing runs (descriptive, post hoc, 4 Nov):
(a) is the end point at the calibration's conditional minimum? E1 Newton step d at the 20000 end points: e + d = error of
    the minimum with every nuisance (geometry, dE/dx) held where the fit left it.
(b) true vs realistic (1 mm smeared) ends, same samples: KN10N<s>C75H vs EXP<s>C75H."""
import sys, re, glob, os; sys.path.insert(0, '/sdf/scratch/neutrino/pgranger/review_2026_09/UQ')
import numpy as np
from uqt_common import P, end, e1, ckpt, D
def row(lab, x, s=None):
    m = x.mean(0); se = x.std(0, ddof=1) / np.sqrt(len(x))
    print(f'  {lab:34s} ' + ' '.join(f'{a:+7.2f}+-{b:5.2f}' for a, b in zip(m, se)))
print('(a) E1 at the end points (mean +- SEM over fits, %):  ' + '  '.join(f'{p:>13s}' for p in P))
for lab, names, runs in (('Set D, EXP17-28C75H (n=12)', [f'D_EXP{s}' for s in range(17, 29)], [f'EXP{s}C75H' for s in range(17, 29)]),
                         ('T1 20k, EXP1-6C75H (n=6)', [f'T1_EXP{s}_20k' for s in range(1, 7)], [f'EXP{s}C75H' for s in range(1, 7)])):
    X = [e1(n, truth=end(r)['truth']) for n, r in zip(names, runs)]
    e = np.array([x['e'] for x in X]); d = np.array([x['d'] for x in X]); E = np.array([end(r)['e'] for r in runs])
    print(lab); row('error at the dump point e', e); row('Newton step to cond. min. d', d); row('cond. minimum e + d', e + d)
    row('(end-point error, last-1000 mean)', E)
    print('  median |z|: ' + ' '.join(f'{v:.1f}' for v in np.median(np.abs([x['z'] for x in X]), 0)))
print('\n(a2) DATASZ dumps: conditional minimum e + d by arm')
for arm, ss, run in (('V025', (17, 19, 21), 'V025s{}H'), ('V050', (17, 19, 21), 'V050s{}H'), ('V200', (17, 18, 19), 'V200s{}H')):
    X = []
    for s in ss:
        dd = ckpt(run.format(s), {'V025': 5000, 'V050': 10000, 'V200': 40000}[arm])
        tr = np.array([float(np.ravel(dd[f'{p}_target'][0])[0]) for p in P]); X.append(e1(f'D_{arm}s{s}', truth=tr))
    e = np.array([x['e'] for x in X]); d = np.array([x['d'] for x in X])
    print(arm); row('e', e); row('d', d); row('e + d', e + d)
print('\n(b) true vs realistic ends, same samples')
pairs = []
for s in range(1, 17):
    try:
        a = ckpt(f'KN10N{s}C75H', 20000); b = ckpt(f'EXP{s}C75H', 20000)
    except SystemExit:
        continue
    sa = re.search(r'--data_seed (\d+)', ' '.join(a['provenance']['argv'])).group(1); sb = re.search(r'--data_seed (\d+)', ' '.join(b['provenance']['argv'])).group(1)
    if sa != sb: print('  seed mismatch', s, sa, sb); continue
    pairs.append((end(f'KN10N{s}C75H')['e'], end(f'EXP{s}C75H')['e']))
T = np.array([p[0] for p in pairs]); R = np.array([p[1] for p in pairs])
print(f'  n = {len(pairs)} paired samples')
row('true ends (KN10N C75H)', T); row('realistic ends (EXP C75H)', R); row('realistic - true (paired)', R - T)
