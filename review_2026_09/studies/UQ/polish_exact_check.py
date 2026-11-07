"""POLISH adoption check (6 Nov): the production estimator KNOT/polish_estimate.py applies the Newton step exactly through the
sigmoid map; the validated readout (read_polish.py) used e + d (first order). Same tests on the exact form: fresh 12
(primary long_diff, one-sided Wilcoxon) and the n = 30 table. Inputs: UQ/polish_exact/EXP<s>.json."""
import json, numpy as np
from scipy import stats
P = ['Ab', 'eField', 'tran_diff', 'long_diff', 'lifetime']
S = list(range(1, 7)) + list(range(17, 41))
J = {s: json.load(open(f'/sdf/scratch/neutrino/pgranger/review_2026_09/UQ/polish_exact/EXP{s}.json')) for s in S}
E = np.array([[J[s]['sim_error_pct']['end_point'][p] for p in P] for s in S]); M = np.array([[J[s]['sim_error_pct']['polished'][p] for p in P] for s in S])
fr = np.array([s >= 29 for s in S]); rms = lambda X: np.sqrt((X ** 2).mean(0))
print('fresh 12, exact form: per parameter RMS e -> m, |m| < |e|, Wilcoxon p(better), p(worse)')
for j, p in enumerate(P):
    a, b = np.abs(E[fr, j]), np.abs(M[fr, j])
    print(f'  {p:10s} {rms(E[fr, j]):6.3f} -> {rms(M[fr, j]):6.3f} | {int((b < a).sum())}/12 | {stats.wilcoxon(a - b, alternative="greater").pvalue:.3f} {stats.wilcoxon(a - b, alternative="less").pvalue:.3f}')
print('n = 30, exact form: m bias +- SE | SD | RMS')
for j, p in enumerate(P):
    x = M[:, j]; print(f'  {p:10s} {x.mean():+7.3f} +- {x.std(ddof=1) / np.sqrt(len(x)):.3f} | {x.std(ddof=1):6.3f} | {rms(x):6.3f}')
