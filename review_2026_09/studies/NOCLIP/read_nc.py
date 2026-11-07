"""NOCLIP readout (4 Nov; rules: REVIEW.md "NOCLIP"). A = EXP<s>C75H (clip 100), HNC = EXP<s>C75HNC (clip off, same FR @ 14000),
s = 17-28; E1 dumps D_EXP<s> (A) and D_NC<s> (HNC). d = calibration Newton step of the unclipped objective (%).
usage (container): python3 read_nc.py   [NC_DRY=1: HNC := H2 (s = 17-24 only) and D_NC := D_EXP, no log check]"""
import os, re, sys, glob, numpy as np
sys.path.insert(0, '/sdf/scratch/neutrino/pgranger/review_2026_09/UQ')
from uqt_common import P, ckpt, end, e1, f1
DRY = os.environ.get('NC_DRY') == '1'
S = list(range(17, 25)) if DRY else list(range(17, 29))
HN = 'H2' if DRY else 'HNC'
LOGS = '/sdf/scratch/neutrino/pgranger/review_2026_09/KNOT/logs'
tag = '  -- DRY RUN' if DRY else ''
A = {s: end(f'EXP{s}C75H', check=s) for s in S}; N = {s: end(f'EXP{s}C75{HN}', check=s) for s in S}
for s in S:
    a, h = ckpt(f'EXP{s}C75H', 20000), ckpt(f'EXP{s}C75{HN}', 20000)
    for p in P:
        xa = np.array([f1(x) for x in a[f'{p}_iter'][:14001]]); xh = np.array([f1(x) for x in h[f'{p}_iter'][:14001]])
        assert np.array_equal(xa, xh), (s, p, 'HNC differs from A before 14001')
        xl = np.array([f1(x) for x in h[f'{p}_iter']]); assert np.all(np.isfinite(xl)), (s, p, 'NaN')
    if not DRY:
        lg = sorted(glob.glob(f'{LOGS}/EXP{s}C75HNC-*.out')); assert lg, f'no log for EXP{s}C75HNC'
        t = open(lg[-1]).read()
        assert re.search(r'Will clip gradient norm at (1e\+?12|1000000000000\.0)\b', t), (s, 'clip 1e12 not in the log')
        assert 'Will clip gradient norm at 100.0' not in t, (s, 'clip 100 in the log')
        assert 'LARND_CALIB_ADAM_RESET: calibration Adam moments re-initialised' in t, (s, 'no Adam reset line')
print(f'NOCLIP: {len(S)} pairs; config check OK (seed, coarse, smear / entry; HNC = A through 14000; no NaN'
      + ('' if DRY else '; clip 1e12 and Adam reset in every log') + ')' + tag)
dA = np.array([e1(f'D_EXP{s}', truth=A[s]['truth'])['d'] for s in S])
dN = np.array([e1(f'D_EXP{s}' if DRY else f'D_NC{s}', truth=N[s]['truth'])['d'] for s in S])
eA = np.array([A[s]['e'] for s in S]); eN = np.array([N[s]['e'] for s in S])
ms = lambda x: (x.mean(0), x.std(0, ddof=1) / np.sqrt(len(x)))
(mA, sA), (mN, sN), (mD, sD) = ms(dA), ms(dN), ms(dN - dA)
print(f"\n  {'':10s} {'d_A':>14s} {'d_NC':>14s} {'Dd = d_NC - d_A':>16s} {'gap closed':>10s}  class")
cls = {}
for j, p in enumerate(P):
    toward = np.sign(mD[j]) == -np.sign(mA[j])
    if abs(mN[j]) < 2 * sN[j] and toward and abs(mD[j]) > 2 * sD[j]: c = 'CLIP IS THE CAUSE'
    elif abs(mD[j]) < 2 * sD[j] and abs(mN[j]) > 2 * sN[j]: c = 'LAG'
    else: c = 'PARTIAL'
    cls[p] = c
    fr = mD[j] / -mA[j] if mA[j] != 0 else np.nan
    print(f'  {p:10s} {mA[j]:+7.3f}+-{sA[j]:5.3f} {mN[j]:+7.3f}+-{sN[j]:5.3f} {mD[j]:+8.3f}+-{sD[j]:6.3f} {fr:10.2f}  '
          + (c if p in ('long_diff', 'lifetime') else '(descriptive)'))
(m1, s1) = ms(eN - eA)
print('\n  descriptive, end-point shift e(HNC) - e(A), mean +- SE: ' + ' '.join(f'{p} {a:+.3f}+-{b:.3f}' for p, a, b in zip(P, m1, s1)))
s8 = [s for s in S if s <= 24]
if not DRY:
    H2 = np.array([end(f'EXP{s}C75H2', check=s)['e'] for s in s8]); iA = [S.index(s) for s in s8]
    (m2, s2) = ms(eN[iA] - 0.5 * (eA[iA] + H2))
    print('  descriptive, e(HNC) - mean(A, H2), s = 17-24: ' + ' '.join(f'{p} {a:+.3f}+-{b:.3f}' for p, a, b in zip(P, m2, s2)))
print('  descriptive, mean error A / HNC: ' + ' '.join(f'{p} {a:+.2f} / {b:+.2f}' for p, a, b in zip(P, eA.mean(0), eN.mean(0))))
print('  descriptive, RMS error A / HNC:  ' + ' '.join(f'{p} {a:.2f} / {b:.2f}' for p, a, b in zip(P, np.sqrt((eA ** 2).mean(0)), np.sqrt((eN ** 2).mean(0)))))
pos = lambda r: 1e4 * np.mean([f1(x) for x in ckpt(r, 20000)['pos_residual_iter'][-1000:]]) if 'pos_residual_iter' in ckpt(r, 20000) else np.nan
pa = np.array([pos(f'EXP{s}C75H') for s in S]); pn = np.array([pos(f'EXP{s}C75{HN}') for s in S])
print(f'  descriptive, position (um) A {np.nanmean(pa):.1f}, HNC {np.nanmean(pn):.1f}, paired {np.nanmean(pn - pa):+.1f}+-{np.nanstd(pn - pa, ddof=1) / np.sqrt(len(S)):.1f}')
print('NOCLIP VERDICT: ' + ', '.join(f'{p} {cls[p]}' for p in ('long_diff', 'lifetime')) + tag)
