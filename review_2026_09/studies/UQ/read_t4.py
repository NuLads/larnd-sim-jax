"""UQ TEST T4 (R4: shortcut sigma = k x sigma_E1) on the 12 fresh end points (Set D dumps D_EXP<s>, s = 17..28). Rules:
REVIEW.md "UQ TEST of the recommendations". k_p = SD_train,p / median sigma_E1,p over the six training end points
(T1_EXP<s>_20k, s = 1..6); per fit sigma_i = k sigma_E1,i; r_i = (e_i - bias_train) / sigma_i; CALIBRATED if SD(r) is in
[0.54, 1.86]; pooled coverage of q = r / sqrt(1 + 1/12) as T3. usage (container): python3 read_t4.py
[T4_DRY=1: pilot dump C3 stands in for every dump, KN10N1-12C75H for the end points, no config check]"""
import os, numpy as np
from uqt_common import P, BIAS, SD, NTR, end, e1, tq
DRY = os.environ.get('T4_DRY') == '1'
tr = np.array([e1('C3' if DRY else f'T1_EXP{s}_20k')['sig'] for s in range(1, 7)])
k = SD / np.median(tr, 0)
S = list(range(1, 13)) if DRY else list(range(17, 29))
fins = [end(f'KN10N{s}C75H' if DRY else f'EXP{s}C75H', check=None if DRY else s) for s in S]
X = [e1('C3' if DRY else f'D_EXP{s}', f['truth']) for s, f in zip(S, fins)]
if not DRY:
    assert all(x['calib_only'] and x['it'] == 20100 for x in X)
E = np.array([f['e'] for f in fins]); sE = np.array([x['sig'] for x in X]); Z = np.array([x['z'] for x in X])
print('T4' + (' -- DRY RUN' if DRY else ': config check OK on 12 fresh end points; 12 calib-only dumps at 20100'))
print('  training sigma_E1 (%) median of 6: ' + ' '.join(f'{p} {v:.4g}' for p, v in zip(P, np.median(tr, 0)))
      + ' | spread (max/min) ' + ' '.join(f'{v:.2f}' for v in tr.max(0) / tr.min(0)))
print('  k = SD_train / sigma_E1: ' + ' '.join(f'{p} {v:.2f}' for p, v in zip(P, k)))
cv = sE.std(0, ddof=1) / sE.mean(0)
print('  test sigma_E1 (%) mean ' + ' '.join(f'{v:.4g}' for v in sE.mean(0)) + ' | CV ' + ' '.join(f'{v:.3f}' for v in cv)
      + f" -> {'CV < 10 % everywhere: k x E1 = the ensemble SD in this configuration' if np.all(cv < 0.10) else 'CV >= 10 % somewhere: E1 carries per-fit information'}")
r = (E - BIAS) / (k * sE); q = r / np.sqrt(1 + 1 / NTR)
sr = r.std(0, ddof=1); cal = (sr >= 0.54) & (sr <= 1.86)
for j, p in enumerate(P):
    c1, c2 = tq(q[:, j])
    print(f'  {p:10s} SD(r) {sr[j]:5.2f} -> {"calibrated" if cal[j] else "NOT calibrated"} | cover |q|<1 {c1:.2f} <2 {c2:.2f} | '
          f'end-point E1 |z| median {np.median(np.abs(Z[:, j])):.1f}, max {np.abs(Z[:, j]).max():.1f}')
c1, c2 = tq(q.ravel()); ok = 0.50 <= c1 <= 0.82 and c2 >= 0.83
print(f'  pooled over {q.size}: |q| < 1 {c1:.2f}, |q| < 2 {c2:.2f} -> {"PASS" if ok else "FAIL"}; test end points with max|z| > 3: '
      f'{int((np.abs(Z).max(1) > 3).sum())} of {len(S)}')
print(f"T4 VERDICT: R4 {'CONFIRMED' if cal.all() and ok else 'NOT CONFIRMED'} (NOT calibrated: "
      f"{', '.join(p for p, c in zip(P, cal) if not c) or 'none'})" + ('  -- DRY RUN' if DRY else ''))
