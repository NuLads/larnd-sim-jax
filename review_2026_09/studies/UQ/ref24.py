"""FRESH TEST reference (4 Nov; REVIEW.md "FRESH TEST"): frozen bias / SD of the pooled 24 single-fit end points
EXP1-12C75H (training) + EXP17-28C75H (Set A), computed before any FRESH job was submitted. Writes ref24.txt.
usage (container): python3 ref24.py"""
import numpy as np
from scipy import stats
from uqt_common import P, end
S = list(range(1, 13)) + list(range(17, 29))
E = np.array([end(f'EXP{s}C75H', check=s)['e'] for s in S])
b, sd = E.mean(0), E.std(0, ddof=1)
lo, hi = np.sqrt(stats.f.ppf([0.025, 0.975], 11, 23))
with open('ref24.txt', 'w') as f:
    f.write('# n = 24 (EXP1-12C75H + EXP17-28C75H), error % = last-1000 mean / truth - 1; params ' + ' '.join(P) + '\n')
    f.write('bias ' + ' '.join(f'{v:.6f}' for v in b) + '\n'); f.write('sd ' + ' '.join(f'{v:.6f}' for v in sd) + '\n')
    f.write(f'sd_ratio_band_F11_23 {lo:.4f} {hi:.4f}\n')
print(open('ref24.txt').read())
