"""Bias-correction table (item 4 of the 4 Nov bias plan; descriptive, no rule). Production end points EXP<s>C75H @ 20000:
24 reference (s = 1-12, 17-28) + 12 FRESH (s = 29-40), realistic 1 mm ends. Correction c = pooled mean error; a corrected
single fit e - c then has error SD sd * sqrt(1 + 1/n). Out-of-sample check: correct the 12 fresh fits with the n = 24
bias only and compare RMS before / after; leave-one-out RMS on the 36.
usage (container): python3 correction_table.py"""
import sys, numpy as np
sys.path.insert(0, '/sdf/scratch/neutrino/pgranger/review_2026_09/UQ')
from uqt_common import P, end
S24 = list(range(1, 13)) + list(range(17, 29)); SF = list(range(29, 41))
E24 = np.array([end(f'EXP{s}C75H')['e'] for s in S24]); EF = np.array([end(f'EXP{s}C75H', check=s)['e'] for s in SF])
E = np.vstack([E24, EF]); n = len(E)
c, sd = E.mean(0), E.std(0, ddof=1); se = sd / np.sqrt(n)
rms = lambda X: np.sqrt((X ** 2).mean(0))
loo = np.array([E[i] - np.delete(E, i, 0).mean(0) for i in range(n)])
print(f'correction table: n = {n} production end points (24 reference + 12 fresh), error in %')
print(f"  {'':10s} {'bias c':>8s} {'+- SE':>6s} {'c/SE':>6s} {'single SD':>9s} {'corr. SD':>8s} | {'RMS raw':>7s} {'RMS LOO':>7s} | fresh: {'RMS raw':>7s} {'RMS corr24':>10s}")
for j, p in enumerate(P):
    print(f'  {p:10s} {c[j]:+8.3f} {se[j]:6.3f} {c[j] / se[j]:6.1f} {sd[j]:9.3f} {sd[j] * np.sqrt(1 + 1 / n):8.3f} | '
          f'{rms(E)[j]:7.3f} {rms(loo)[j]:7.3f} | {"":6s} {rms(EF)[j]:7.3f} {rms(EF - E24.mean(0))[j]:10.3f}')
print('  bias c in natural units: c / 100 x truth; the correction is valid for this configuration only (KN10, 400 cm/batch, '
      'production C75 pipeline, 1 mm smeared ends, simulation with the same physics model).')
