"""Readout of the joint calibration + dE/dx step on the 16 KN10C end points @ 1.4e4 (rule: REVIEW.md 21 Oct, stated before
any result), beside the 20 Oct 5-parameter polish of the same 16 (POLISH/polish_KN10C_s*.json)."""
import json, numpy as np
D = '/sdf/scratch/neutrino/pgranger/review_2026_09/POLISH'
P = ['Ab', 'eField', 'tran_diff', 'long_diff', 'lifetime']
J = [json.load(open(f'{D}/joint/joint_KN10C{s}.json'))['summary'] for s in range(1, 17)]
C = [json.load(open(f'{D}/polish_KN10C_s{s}.json'))['summary'] for s in range(1, 17)]
nv = sum(x['accepted'] for x in J)
print(f'step VALID {nv}/16 (rule >= 14); best kappa per sample: ' + ' '.join(str(x['best_kappa']) for x in J)
      + f'; samples with negative-curvature batches: {sum(x["n_batches_neg"] > 0 for x in J)}')
se = lambda v: np.std(v, ddof=1) / np.sqrt(len(v))
verdict = {}
for i, k in enumerate(P):
    e0 = np.array([x['err0'][k] for x in J]); e1 = np.array([x['err1'][k] for x in J]); ec = np.array([x['err1'][k] for x in C])
    r = np.std(e1, ddof=1) / np.std(e0, ddof=1); dm = (abs(e1.mean()) - abs(e0.mean())) / se(e0); verdict[k] = (r, dm)
    print(f'  {k:9s} mean, SD before {e0.mean():+.3f}, {np.std(e0, ddof=1):.3f} -> joint {e1.mean():+.3f}, {np.std(e1, ddof=1):.3f} '
          f'(SD x{r:.2f}; |mean| {dm:+.2f} SE)  | 5-param {ec.mean():+.3f}, {np.std(ec, ddof=1):.3f} (x{np.std(ec, ddof=1) / np.std(e0, ddof=1):.2f})'
          f'  | mean joint step {np.mean([x["du_pct"][i] for x in J]):+.3f}%')
(rl, ml), (rd, md) = verdict['lifetime'], verdict['long_diff']
print('verdict:', 'IMPROVES' if rl <= 0.9 and rd <= 0.9 and ml <= 1 and md <= 1 else 'WORSENS' if rl >= 1.1 or rd >= 1.1 or ml > 2 or md > 2 else 'NEUTRAL')
for i, k in enumerate(P):
    e0 = np.array([x['err0'][k] for x in J]); sm = np.array([x['sigma_marg_pct'][i] for x in J]); sc = np.array([x['sigma_cond_pct'][i] for x in J])
    pm, pcn = np.sqrt(np.mean((e0 / sm) ** 2)), np.sqrt(np.mean((e0 / sc) ** 2))
    print(f'  pull {k:9s} RMS (theta_0 - truth)/sigma_marg {pm:.2f} ({"CALIBRATED" if 0.7 <= pm <= 1.4 else "NOT calibrated"}); with sigma_cond {pcn:.2f}; '
          f'median sigma marg/cond {np.median(sm):.3f}/{np.median(sc):.3f}% vs observed SD {np.std(e0, ddof=1):.3f}%')
