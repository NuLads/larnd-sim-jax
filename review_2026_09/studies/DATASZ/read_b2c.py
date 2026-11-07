"""B200C (4 Nov; rules: REVIEW.md "B200C"): confirmation of the T6 hint that 200 cm batches lower the spread, on 12 UNSEEN
samples. Arm B200s<s>H (DATASZ make_ds.py, 200 x 200 cm, r = 2, w = 1250), s = 29-40; reference = the production end
points EXP<s>C75H of the FRESH TEST (100 x 400 cm), paired by sample. Errors and config checks = read_ds.read (last 10
epochs at 20000 r). Primary, per p in {long_diff, lifetime}: one-sided Pitman-Morgan test of var(B200) < var(ref) for paired
samples (corr(x - y, x + y) < 0, t with n - 2 dof). usage (container): python3 read_b2c.py [--dry]
--dry: the T6 samples s = 17-24 (B200s<s>H vs EXP<s>C75H), to test the script; not the confirmation."""
import sys, numpy as np
from scipy import stats
sys.argv = [sys.argv[0]] + [a for a in sys.argv[1:] if a != '--dry'] if '--dry' not in sys.argv else [sys.argv[0], '--dry']
DRY = '--dry' in sys.argv
sys.path.insert(0, '/sdf/scratch/neutrino/pgranger/review_2026_09/DATASZ')
import read_ds as R
P = R.P
SEEDS = range(17, 25) if DRY else range(29, 41)


def pitman_morgan(x, y):
    """one-sided p for var(x) < var(y), paired; returns ratio SD_x / SD_y, corr(x, y), t, p"""
    n = len(x); r, _ = stats.pearsonr(x - y, x + y)
    t = r * np.sqrt((n - 2) / (1 - r ** 2))
    return np.std(x, ddof=1) / np.std(y, ddof=1), stats.pearsonr(x, y)[0], t, stats.t.cdf(t, n - 2)


B = [R.read(f'B200s{s}H', 200, 200, 600) for s in SEEDS]
F = [R.read(f'EXP{s}C75H', 100, 400, 600) for s in SEEDS]
EB = np.array([[x[p] for p in P] for x in B]); EF = np.array([[x[p] for p in P] for x in F])
n = len(EB)
print(f"B200C{' DRY RUN on the T6 samples s = 17-24 (not the confirmation)' if DRY else ''}: n = {n}, config checks OK")
print(f"  track length used: B200 {np.mean([x['cm'] for x in B]):.0f} cm, reference {np.mean([x['cm'] for x in F]):.0f} cm")
red, safe = {}, True
for j, p in enumerate(P):
    x, y = EB[:, j], EF[:, j]; d = x - y; m, se = d.mean(), d.std(ddof=1) / np.sqrt(n)
    ratio, rho, t, pv = pitman_morgan(x, y)
    pF = stats.f.cdf(ratio ** 2, n - 1, n - 1)
    away = abs(m) > 2 * se and abs(x.mean()) > abs(y.mean())      # a shift that makes |mean error| larger
    safe &= not away
    tag = ''
    if p in ('long_diff', 'lifetime'):
        red[p] = pv < 0.05; tag = '  PRIMARY: ' + ('REDUCED' if red[p] else 'not reduced')
    print(f'  {p:10s} SD {x.std(ddof=1):.3f} vs {y.std(ddof=1):.3f} (ratio {ratio:.2f}, corr {rho:+.2f}; Pitman-Morgan one-sided p {pv:.3f}, '
          f'F p {pF:.3f}) | mean {x.mean():+.3f} vs {y.mean():+.3f}, paired delta {m:+.3f} +- {se:.3f}'
          + ('  <- SHIFT AWAY FROM ZERO' if away else '') + tag)
    print(f'  {"":10s} RMS error {np.sqrt((x ** 2).mean()):.3f} vs {np.sqrt((y ** 2).mean()):.3f}')
dp = np.array([x['pos'] for x in B]) - np.array([x['pos'] for x in F])
print(f'  position paired delta {dp.mean():+.1f} +- {dp.std(ddof=1) / np.sqrt(n):.1f} um (secondary; one-sided p for < 0: '
      f'{stats.t.cdf(dp.mean() / (dp.std(ddof=1) / np.sqrt(n)), n - 1):.3f})')
k = sum(red.values())
v = 'CONFIRMED' if k == 2 else ('PARTLY CONFIRMED (' + [q for q in red if red[q]][0] + ' only)' if k == 1 else 'NOT CONFIRMED')
print(f"B200C VERDICT: {v}; {'SAFE (no shift away from zero)' if safe else 'NOT SAFE (a parameter shifts away from zero)'}")
