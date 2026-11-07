"""Uncertainty and bias of the polished estimate m (descriptive; follows POLISH RESULT 5 Nov). m = end-point iterate @ 20000
+ calibration-only E1 Newton step, on every production end point EXP<s>C75H with such a dump: D_EXP17-28 (s = 17-28),
T1_EXP<s>_20k (s = 1-6, if calibration-only) and D_F29-40 (s = 29-40, the POLISH test). Same table as correction_table.py
for e and m on the same samples; out-of-sample: the 12 fresh m corrected with the bias of m from the other samples.
usage (container): python3 polish_table.py"""
import sys, numpy as np
sys.path.insert(0, '/sdf/scratch/neutrino/pgranger/review_2026_09/UQ')
from uqt_common import P, end, e1
ROWS = [(s, f'T1_EXP{s}_20k') for s in range(1, 7)] + [(s, f'D_EXP{s}') for s in range(17, 29)] + [(s, f'D_F{s}') for s in range(29, 41)]
S, E, M = [], [], []
for s, dn in ROWS:
    x = end(f'EXP{s}C75H'); y = e1(dn, x['truth'])
    if not (y['calib_only'] and y['it'] in (20000, 20100) and np.abs(y['e'] - x['last']).max() < 1e-3):
        print(f'  skip {dn}: calib_only {y["calib_only"]}, it {y["it"]}, at end point {np.abs(y["e"] - x["last"]).max() < 1e-3}'); continue
    S.append(s); E.append(x['e']); M.append(y['e'] + y['d'])
S, E, M = np.array(S), np.array(E), np.array(M); n = len(S); fr = S >= 29
rms = lambda X: np.sqrt((X ** 2).mean(0))
print(f'n = {n} end points with a calibration-only dump at 20000 (s = {", ".join(map(str, S))}); error in %')
print(f"  {'':10s} | {'e bias':>7s} {'SD':>6s} {'RMS':>6s} | {'m bias':>7s} {'+-SE':>6s} {'SD':>6s} {'RMS':>6s} {'LOO corr':>8s} | fresh m: {'RMS':>6s} {'corr(other)':>11s}")
for j, p in enumerate(P):
    loo = np.array([M[i, j] - np.delete(M[:, j], i).mean() for i in range(n)])
    print(f'  {p:10s} | {E[:, j].mean():+7.3f} {E[:, j].std(ddof=1):6.3f} {rms(E[:, j]):6.3f} | {M[:, j].mean():+7.3f} '
          f'{M[:, j].std(ddof=1) / np.sqrt(n):6.3f} {M[:, j].std(ddof=1):6.3f} {rms(M[:, j]):6.3f} {rms(loo):8.3f} | '
          f'{"":8s} {rms(M[fr, j]):6.3f} {rms(M[fr, j] - M[~fr, j].mean()):11.3f}')
