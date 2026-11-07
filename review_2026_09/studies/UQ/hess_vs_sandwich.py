"""Hessian vs Huber sandwich vs observed spread (descriptive, 6 Nov; Pierre: why don't they work?). The 30 calibration-only
E1 dumps at production end points (T1_EXP1-6_20k, D_EXP17-28, D_F29-40): conditional Hessian width sig (nuisances held at
the fit), sandwich width sig_sw (per-batch score scatter), vs the SD of the polished estimate m over the 30 samples.
usage (container): python3 hess_vs_sandwich.py"""
import sys, numpy as np
sys.path.insert(0, '/sdf/scratch/neutrino/pgranger/review_2026_09/UQ')
from uqt_common import P, end
import uq_lib as Q
ROWS = [(s, f'T1_EXP{s}_20k') for s in range(1, 7)] + [(s, f'D_EXP{s}') for s in range(17, 29)] + [(s, f'D_F{s}') for s in range(29, 41)]
SG, SW, M = [], [], []
for s, dn in ROWS:
    x = end(f'EXP{s}C75H'); d = Q.load(f'{Q.__file__.rsplit("/", 1)[0]}/dumps/{dn}.npz'); e = Q.estimators(d)['E1']
    SG.append(100 * e['sig']); SW.append(100 * e['sig_sw']); M.append(100 * (d['phys'] / x['truth'] - 1) + 100 * e['d'])
SG, SW, M = map(np.array, (SG, SW, M)); sd = M.std(0, ddof=1)
print(f'n = {len(M)}; widths in %, per parameter: Hessian (median over fits) | sandwich (median) | observed SD of m | SD / Hessian, SD / sandwich')
for j, p in enumerate(P):
    print(f'  {p:10s} {np.median(SG[:, j]):7.4f} | {np.median(SW[:, j]):7.4f} | {sd[j]:7.4f} | {sd[j] / np.median(SG[:, j]):6.1f}x  {sd[j] / np.median(SW[:, j]):6.1f}x')
print('  sandwich / Hessian (median): ' + ' '.join(f'{p} {np.median(SW[:, j] / SG[:, j]):.2f}' for j, p in enumerate(P)))
