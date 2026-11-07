"""POLISH (4 Nov; rules: REVIEW.md "POLISH"): is the conditional minimum m (the iterate at 20000 + the calibration-only
Newton step of a held-epoch E1 dump) a better final estimate than the production end point e (last-1000 mean)? Tested on
the 12 UNSEEN FRESH end points EXP<s>C75H @ 20000, s = 29-40, dumps D_F<s> (UQ/run_dump.sbatch,
LARND_JOINT_NEWTON_DUMP_CALIB=1, as Set D). Per parameter: one-sided Wilcoxon signed-rank test of |e| - |m| > 0 (paired); primary = long_diff.
usage (container): python3 read_polish.py [--dry]     --dry: Set D (EXP17-28C75H / D_EXP<s>), hypothesis-forming, script test only"""
import sys, numpy as np
from scipy import stats
sys.path.insert(0, '/sdf/scratch/neutrino/pgranger/review_2026_09/UQ')
from uqt_common import P, end, e1
DRY = '--dry' in sys.argv
ROWS = [(f'EXP{s}C75H', f'D_EXP{s}', s) for s in range(17, 29)] if DRY else [(f'EXP{s}C75H', f'D_F{s}', s) for s in range(29, 41)]
E, M = [], []
for run, dn, s in ROWS:
    x = end(run, check=s); y = e1(dn, x['truth'])
    assert y['calib_only'] and y['it'] in (20000, 20100), (dn, y['calib_only'], y['it'])
    assert np.abs(y['e'] - x['last']).max() < 1e-3, (dn, 'dump is not at the end-point iterate')
    E.append(x['e']); M.append(y['e'] + y['d'])
E, M = np.array(E), np.array(M); n = len(E)
print(f"POLISH{' DRY RUN on Set D (hypothesis-forming, not the test)' if DRY else ''}: n = {n}; config checks OK "
      '(seed, coarse 7500, smear 1.0 / entry; calibration-only dumps at the end-point iterate)')
res = {}
for j, p in enumerate(P):
    a, b = np.abs(E[:, j]), np.abs(M[:, j])
    pg = stats.wilcoxon(a - b, alternative='greater').pvalue      # m better
    pl = stats.wilcoxon(a - b, alternative='less').pvalue         # m worse
    res[p] = (pg, pl)
    tag = ('  PRIMARY: ' + ('IMPROVES' if pg < 0.05 else 'not improved')) if p == 'long_diff' else ''
    print(f'  {p:10s} e mean {E[:, j].mean():+6.2f} SD {E[:, j].std(ddof=1):5.2f} RMS {np.sqrt((E[:, j] ** 2).mean()):5.2f} | '
          f'm mean {M[:, j].mean():+6.2f} SD {M[:, j].std(ddof=1):5.2f} RMS {np.sqrt((M[:, j] ** 2).mean()):5.2f} | '
          f'|m| < |e| on {int((b < a).sum())}/{n}; Wilcoxon p(better) {pg:.3f}, p(worse) {pl:.3f}' + tag)
conf = res['long_diff'][0] < 0.05
safe = all(res[p][1] >= 0.05 for p in P)
sec = lambda p: 'IMPROVES' if res[p][0] < 0.05 else ('WORSE' if res[p][1] < 0.05 else 'no detected change')
print(f"POLISH VERDICT: {'CONFIRMED' if conf else 'NOT CONFIRMED'}; {'SAFE' if safe else 'NOT SAFE (a parameter gets worse)'}; "
      f"secondary: tran_diff {sec('tran_diff')}, lifetime {sec('lifetime')}")
