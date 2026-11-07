"""FRESH TEST (4 Nov; rules: REVIEW.md "FRESH TEST"). 12 fresh samples, data_seed 29-40, production pipeline EXP<s>C75 ->
A = EXP<s>C75H @ 20000, plus three H-stage-only re-runs EXP<s>C75H2/H3/H4 (prodfast again from the same EXP<s>C75FR @ 14000).
F1: is the pooled n = 24 reference (ref24.txt) a calibrated uncertainty on unseen single fits? pulls
    q = (e_A - bias24) / (sd24 sqrt(1 + 1/24)) ~ t_23.
F2: does averaging the 4 H-stage end points of a sample shrink the error spread? within-sample SD sigma_w (36 dof),
    implied gain g = 1 - sqrt(1 - (3/4) sigma_w^2 / sd24^2).
usage (container): python3 read_f.py [f1|f2|both]
[F_DRY=1: EXP17-28C75H stand in for A (F1); s = 17-24 with H, H2, RH, H2 as the 4 replicas (F2); no bitwise check on RH]"""
import os, sys, numpy as np
from scipy import stats
from uqt_common import P, ckpt, end, f1, tq
DRY = os.environ.get('F_DRY') == '1'
WHAT = sys.argv[1] if len(sys.argv) > 1 else 'both'
R = {k: np.array([float(x) for x in v.split()]) for k, v in
     (l.split(' ', 1) for l in open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ref24.txt')) if not l.startswith('#'))}
B24, SD24, (LO, HI) = R['bias'], R['sd'], R['sd_ratio_band_F11_23']
N24 = 24
tag = '  -- DRY RUN' if DRY else ''

if WHAT in ('f1', 'both'):
    S = list(range(17, 29)) if DRY else list(range(29, 41))
    E = np.array([end(f'EXP{s}C75H', check=s)['e'] for s in S]); n = len(S)
    print(f'F1: config check OK on {n} end points (seed, coarse 7500, smear 1.0 / entry){tag}')
    print('  s  ' + ' '.join(f'{p:>10s}' for p in P))
    for s, e in zip(S, E):
        print(f' {s:2d}  ' + ' '.join(f'{v:+10.3f}' for v in e))
    q = (E - B24) / (SD24 * np.sqrt(1 + 1 / N24))
    sdt, mt = E.std(0, ddof=1), E.mean(0); ratio = sdt / SD24
    cal = (ratio >= LO) & (ratio <= HI); tol = 2 * np.sqrt(1 / n + 1 / N24); btr = np.abs(mt - B24) < tol * SD24
    print(f"\n  {'':10s} {'bias24':>8s} {'mean':>8s} {'transfers':>9s} | {'sd24':>7s} {'SD':>7s} {'ratio':>6s} {'calibrated':>10s} | |q|<1 |q|<2")
    for j, p in enumerate(P):
        c1, c2 = tq(q[:, j])
        print(f"  {p:10s} {B24[j]:+8.3f} {mt[j]:+8.3f} {str(bool(btr[j])):>9s} | {SD24[j]:7.3f} {sdt[j]:7.3f} {ratio[j]:6.2f} {str(bool(cal[j])):>10s} | {c1:5.2f} {c2:5.2f}")
    c1, c2 = tq(q.ravel()); ok = 0.50 <= c1 <= 0.82 and c2 >= 0.83
    print(f'  bands: ratio [{LO:.2f}, {HI:.2f}] (95 % of F(11, 23)); bias tolerance {tol:.3f} sd24')
    print(f'  pooled over {q.size} pulls: |q| < 1 {c1:.2f} (expected {2 * stats.t.cdf(1, 23) - 1:.2f}, pass 0.50-0.82), '
          f'|q| < 2 {c2:.2f} (expected {2 * stats.t.cdf(2, 23) - 1:.2f}, pass >= 0.83) -> {"PASS" if ok else "FAIL"}')
    v = 'CONFIRMED' if cal.all() and ok else 'NOT CONFIRMED'
    print(f"F1 VERDICT: pooled-24 uncertainty {v} (NOT calibrated: {', '.join(p for p, c in zip(P, cal) if not c) or 'none'}; "
          f"biases that do not transfer: {', '.join(p for p, b in zip(P, btr) if not b) or 'none'}){tag}\n")

if WHAT in ('f2', 'both'):
    S = list(range(17, 25)) if DRY else list(range(29, 41))
    REPS = ('H', 'H2', 'RH', 'H2') if DRY else ('H', 'H2', 'H3', 'H4')
    E = np.array([[end(f'EXP{s}C75{r}', check=s)['e'] for r in REPS] for s in S])   # (n, 4, 5)
    # every H-only re-run must equal its A history bit for bit through 14000 (same FR start) and differ after it
    for s in S:
        a = ckpt(f'EXP{s}C75H', 20000)
        for r in REPS[1:]:
            if DRY and r == 'RH':
                continue
            h = ckpt(f'EXP{s}C75{r}', 20000)
            for p in P:
                xa = np.array([f1(x) for x in a[f'{p}_iter'][:14001]]); xh = np.array([f1(x) for x in h[f'{p}_iter'][:14001]])
                assert np.array_equal(xa, xh), (s, r, p, 'differs from A before 14001: not an H-only re-run of the same FR')
    n, m = E.shape[0], E.shape[1]
    print(f'F2: config check OK on {n} x {m} end points; every H-only re-run equals A through 14000{tag}')
    print('  s  ' + ' '.join(f'{p:>30s}' for p in P) + '   (the 4 H-stage end points, error %)')
    for s, e in zip(S, E):
        print(f' {s:2d}  ' + ' '.join('/'.join(f'{v:+6.2f}' for v in e[:, j]).rjust(30) for j in range(5)))
    dof = n * (m - 1)
    sw = np.sqrt(((E - E.mean(1, keepdims=True)) ** 2).sum((0, 1)) / dof)
    sw_lo, sw_hi = sw * np.sqrt(dof / stats.chi2.ppf(0.975, dof)), sw * np.sqrt(dof / stats.chi2.ppf(0.025, dof))
    g = lambda x: 1 - np.sqrt(np.clip(1 - (1 - 1 / m) * x ** 2 / SD24 ** 2, 0, None))
    G, Glo, Ghi = g(sw), g(sw_lo), g(sw_hi)
    cls = np.where(Glo >= 0.10, 'USEFUL', np.where(Ghi < 0.10, 'NO GAIN', 'MARGINAL'))
    t2h = np.array([0.119, 0.006, 0.310, 3.053, 3.236]); t2f = np.array([0.127, 0.013, 0.616, 2.838, 1.540])
    print(f"\n  {'':10s} {'sigma_w':>8s} {'95 % CI':>16s} {'T2 H-only':>9s} {'T2 full':>8s} | {'sd24':>7s} {'gain g':>7s} {'g range':>14s} {'class':>9s}")
    for j, p in enumerate(P):
        print(f'  {p:10s} {sw[j]:8.3f} {sw_lo[j]:7.3f}-{sw_hi[j]:<8.3f} {t2h[j]:9.3f} {t2f[j]:8.3f} | {SD24[j]:7.3f} {100 * G[j]:6.1f}% '
              f'{100 * Glo[j]:5.1f}-{100 * Ghi[j]:4.1f} %  {cls[j]:>9s}')
    # descriptive: direct spreads of the averaged vs the single end point over the n samples, paired bootstrap
    A, M = E[:, 0], E.mean(1)
    rng = np.random.default_rng(0); bs = []
    for _ in range(4000):
        i = rng.integers(0, n, n); bs.append(M[i].std(0, ddof=1) / A[i].std(0, ddof=1))
    lo, hi = np.percentile(np.array(bs), [2.5, 97.5], 0)
    print('  descriptive, SD(average of 4) / SD(A) over the samples: '
          + ' '.join(f'{p} {r:.2f} [{a:.2f}, {b:.2f}]' for p, r, a, b in zip(P, M.std(0, ddof=1) / A.std(0, ddof=1), lo, hi)))
    print('  descriptive, mean(average) - mean(A): ' + ' '.join(f'{v:+.3f}' for v in M.mean(0) - A.mean(0)))
    sa = np.sqrt(np.clip(SD24 ** 2 - (1 - 1 / m) * sw ** 2, 1e-12, None))
    qa = (M - B24) / (sa * np.sqrt(1 + 1 / N24)); a1, a2 = tq(qa.ravel())
    print(f'  descriptive, the averages against bias24 and sigma_avg = sqrt(sd24^2 - (3/4) sigma_w^2) = '
          + ' '.join(f'{v:.3g}' for v in sa) + f': pooled |q| < 1 {a1:.2f}, < 2 {a2:.2f}')
    print(f"F2 VERDICT: averaging 4 H-stage runs " + ', '.join(f'{p} {c}' for p, c in zip(P, cls)) + tag)
