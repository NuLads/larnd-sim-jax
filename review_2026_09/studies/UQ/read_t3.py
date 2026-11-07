"""UQ TEST T3 (R3: is the training-ensemble SD a calibrated uncertainty, and do its biases transfer?) on the 12 FRESH
samples EXP<s>C75H, s = 17..28 (Set A). Rules: REVIEW.md "UQ TEST of the recommendations". Pulls q = (e - bias_train) /
(SD_train sqrt(1 + 1/12)) ~ t_11. usage (container): python3 read_t3.py   [T3_DRY=1: KN10N1-12C75H stand in, no config check]"""
import glob, os, numpy as np
from uqt_common import P, BIAS, SD, NTR, D, end, e1, tq
DRY = os.environ.get('T3_DRY') == '1'
S = list(range(1, 13)) if DRY else list(range(17, 29))
E = np.array([end(f'KN10N{s}C75H' if DRY else f'EXP{s}C75H', check=None if DRY else s)['e'] for s in S])
print('T3' + (' -- DRY RUN (true-end KN10N1-12C75H stand in)' if DRY else ': config check OK on 12 fresh end points (seed, coarse 7500, smear 1.0 / entry)'))
print('  s  ' + ' '.join(f'{p:>10s}' for p in P))
for s, e in zip(S, E):
    print(f' {s:2d}  ' + ' '.join(f'{v:+10.3f}' for v in e))
q = (E - BIAS) / (SD * np.sqrt(1 + 1 / NTR))
sdt = E.std(0, ddof=1); mt = E.mean(0); ratio = sdt / SD
cal = (ratio >= 0.54) & (ratio <= 1.86); btr = np.abs(mt - BIAS) < 0.82 * SD
print(f"\n  {'':10s} {'bias_train':>10s} {'mean_test':>10s} {'transfers':>9s} | {'SD_train':>8s} {'SD_test':>8s} {'ratio':>6s} {'calibrated':>10s} | cover |q|<1 |q|<2")
for j, p in enumerate(P):
    c1, c2 = tq(q[:, j])
    print(f"  {p:10s} {BIAS[j]:+10.3f} {mt[j]:+10.3f} {str(bool(btr[j])):>9s} | {SD[j]:8.3f} {sdt[j]:8.3f} {ratio[j]:6.2f} {str(bool(cal[j])):>10s} | {c1:10.2f} {c2:5.2f}")
c1, c2 = tq(q.ravel()); ok_cov = 0.50 <= c1 <= 0.82 and c2 >= 0.83
print(f'  pooled over {q.size} pulls: |q| < 1 {c1:.2f} (expected 0.66, pass 0.50-0.82), |q| < 2 {c2:.2f} (expected 0.93, pass >= 0.83) -> {"PASS" if ok_cov else "FAIL"}')
# descriptive comparisons: the current metric (sigma_E1 alone, no bias), and E1 (+) the pre-C75 replica scatter
t1 = sorted(glob.glob(f'{D}/T1_EXP*_20k.npz'))
sE1 = np.median([e1(os.path.basename(f)[:-4])['sig'] for f in t1], 0) if t1 else np.array([0.030, 0.0006, 0.37, 0.86, 0.75])
so_pre = np.array([0.072, 0.007, 0.21, 1.90, 1.24])
for lab, sg, b in (('sigma_E1 alone, no bias (current metric)', sE1, 0 * BIAS), ('sigma_E1 alone, bias-corrected', sE1, BIAS),
                   ('E1 (+) pre-C75 sigma_opt, bias-corrected', np.sqrt(sE1 ** 2 + so_pre ** 2), BIAS)):
    qq = (E - b) / sg; a1, a2 = tq(qq.ravel())
    print(f'  descriptive, {lab}: pull SD per parameter ' + ' '.join(f'{v:.1f}' for v in qq.std(0, ddof=1))
          + f' | pooled |q| < 1 {a1:.2f}, < 2 {a2:.2f}')
print(f"  (sigma_E1 used: {'median of ' + str(len(t1)) + ' T1 20000 dumps' if t1 else 'pilot value'}: " + ' '.join(f'{v:.3g}' for v in sE1) + ')')
v = 'CONFIRMED' if cal.all() and ok_cov else 'NOT CONFIRMED'
print(f"T3 VERDICT: R3 {v} (calibrated: {', '.join(p for p, c in zip(P, cal) if c) or 'none'}; NOT calibrated: "
      f"{', '.join(p for p, c in zip(P, cal) if not c) or 'none'}; biases transfer: {', '.join(p for p, b in zip(P, btr) if b) or 'none'}; "
      f"do not: {', '.join(p for p, b in zip(P, btr) if not b) or 'none'})" + ('  -- DRY RUN' if DRY else ''))
