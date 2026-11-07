"""UQ TEST T2 (R2: replica-based convergence) on the fresh samples s = 17..24 (Set R). Rules: REVIEW.md "UQ TEST of the
recommendations". A = EXP<s>C75H, B = full-pipeline replica EXP<s>C75RH, H2 = H-stage-only replica EXP<s>C75H2 (from A's
FR @ 14000). sigma_opt = SD(B - A) / sqrt(2) (ddof 1); rho = sigma_opt / SD_train; converged if rho < 0.5. k-pair estimates
(k < 8) use the RMS of the k differences / sqrt(2) (replica differences have zero mean by symmetry; k = 1: |B - A| / sqrt(2)).
usage (container): python3 read_t2.py   [T2_DRY=1: pre-C75 pairs KN10N<s>H / KN10N<s>PH, s = 3, 5..9, stand in; H2 = B]"""
import itertools, os, numpy as np
from uqt_common import P, SD, end
DRY = os.environ.get('T2_DRY') == '1'
S = [3, 5, 6, 7, 8, 9] if DRY else list(range(17, 25))
nm = (lambda s: (f'KN10N{s}H', f'KN10N{s}PH', f'KN10N{s}PH')) if DRY else (lambda s: (f'EXP{s}C75H', f'EXP{s}C75RH', f'EXP{s}C75H2'))
A, B, H2 = (np.array([end(nm(s)[k], check=None if DRY else s)['e'] for s in S]) for k in range(3))
n = len(S)
print('T2' + (' -- DRY RUN (pre-C75 pairs; H2 = B)' if DRY else f': config check OK on {n} x 3 end points (seed, coarse 7500, smear 1.0 / entry)'))
print('  s  ' + ' '.join(f'{p:>22s}' for p in P) + '   (A / B / H2, error %)')
for i, s in enumerate(S):
    print(f' {s:2d}  ' + ' '.join(f'{A[i, j]:+6.2f}/{B[i, j]:+6.2f}/{H2[i, j]:+6.2f}' for j in range(5)))
dB = B - A; dH = H2 - A
so = dB.std(0, ddof=1) / np.sqrt(2); rho = so / SD; conv = rho < 0.5
soH = dH.std(0, ddof=1) / np.sqrt(2)
PRE = {'Ab': 'borderline', 'eField': 'not', 'tran_diff': 'converged', 'long_diff': 'not', 'lifetime': 'borderline'}
print(f"\n  {'':10s} {'sigma_opt':>9s} {'SD_train':>8s} {'rho':>5s} {'class':>13s} {'(pre-C75)':>11s} | {'SD_A':>6s} | single-pair agree | {'sig_opt^H':>9s} {'H/full':>6s}")
agree1 = []
for j, p in enumerate(P):
    a1 = np.mean((np.abs(dB[:, j]) / np.sqrt(2) / SD[j] < 0.5) == conv[j]); agree1.append(a1)
    print(f"  {p:10s} {so[j]:9.3f} {SD[j]:8.3f} {rho[j]:5.2f} {'converged' if conv[j] else 'NOT converged':>13s} {PRE[p]:>11s} | "
          f"{A[:, j].std(ddof=1):6.3f} | {a1:17.2f} | {soH[j]:9.3f} {soH[j] / so[j]:6.2f}")
one_ok = all(a >= 7 / 8 for a in agree1)
kneed = None
for k in range(1, n + 1):
    ok = True
    for j in range(5):
        cl = [np.sqrt(np.mean(dB[list(c), j] ** 2) / 2) / SD[j] < 0.5 for c in itertools.combinations(range(n), k)]
        ok &= np.mean(np.array(cl) == conv[j]) >= 0.8
    if ok:
        kneed = k; break
print(f"  one pair suffices: {'CONFIRMED' if one_ok else 'NOT'} (agreement " + ' '.join(f'{a:.2f}' for a in agree1)
      + f"); smallest k with >= 80 % agreement on every parameter (all subsets): {kneed}")
sel = [j for j in range(5) if rho[j] >= 0.5] or list(range(5)); r = soH[sel] / so[sel]
vH = 'SUFFICES' if np.all(r >= 0.7) else ('INSUFFICIENT' if np.any(r < 0.5) else 'PARTIAL')
print(f"  H-only replica on {', '.join(P[j] for j in sel)}: sigma_opt^H / sigma_opt " + ' '.join(f'{v:.2f}' for v in r) + f' -> H-ONLY {vH}')
print(f"T2 VERDICT: classes " + ', '.join(f"{p} {'conv' if c else 'NOT'}" for p, c in zip(P, conv))
      + f"; one pair {'suffices' if one_ok else 'does NOT suffice (k = ' + str(kneed) + ')'}; H-only replica {vH}" + ('  -- DRY RUN' if DRY else ''))
