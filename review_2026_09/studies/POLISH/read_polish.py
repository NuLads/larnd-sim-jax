"""Readout of the 16 Newton-polish runs (rule: REVIEW.md 20 Oct, stated before any result)."""
import json, numpy as np
D = [json.load(open(f'polish_KN10C_s{s}.json')) for s in range(1, 17)]
P = D[0]['params']; S = [d['summary'] for d in D]
pd = [min(s['eig']) > 0 for s in S]; dec = [s['dL_actual'] < 0 for s in S]
print(f'H positive definite {sum(pd)}/16; data loss decreases at theta_1 {sum(dec)}/16; both {sum(a and b for a, b in zip(pd, dec))}/16')
print('per sample dL actual / predicted: ' + ' '.join(f'{s["dL_actual"]:+.1f}/{s["dL_pred"]:+.1f}' for s in S))
print('min eig per sample: ' + ' '.join(f'{min(s["eig"]):.2e}' for s in S))
E0 = np.array([[s['err0'][k] for k in P] for s in S]); E1 = np.array([[s['err1'][k] for k in P] for s in S])
se = lambda x: x.std(ddof=1) / np.sqrt(len(x))
print(f'{"param":10s} {"mean0":>8s} {"SD0":>7s} {"mean1":>8s} {"SD1":>7s}  SD ratio  |mean| change in SE0   step mean +- SD (%)')
for j, k in enumerate(P):
    m0, s0, m1, s1 = E0[:, j].mean(), E0[:, j].std(ddof=1), E1[:, j].mean(), E1[:, j].std(ddof=1)
    st = np.array([s['step_pct'][j] for s in S])
    print(f'{k:10s} {m0:+8.3f} {s0:7.3f} {m1:+8.3f} {s1:7.3f}  {s1 / s0:8.2f}  {(abs(m1) - abs(m0)) / se(E0[:, j]):+8.2f}            {st.mean():+.3f} +- {st.std(ddof=1):.3f}')
for tag in ('sigma_cond_pct', 'sigma_sandwich_centred_pct', 'sigma_sandwich_uncentred_pct'):
    sig = np.array([s[tag] for s in S]); pull = E0 / sig
    print(f'{tag:30s} RMS pull ' + '  '.join(f'{k} {np.sqrt(np.mean(pull[:, j] ** 2)):.2f}' for j, k in enumerate(P)))
    print(f'{"":30s} median sigma ' + '  '.join(f'{k} {np.median(sig[:, j]):.3f}' for j, k in enumerate(P)))
print('observed SD of theta_0 errors: ' + '  '.join(f'{k} {E0[:, j].std(ddof=1):.3f}' for j, k in enumerate(P)))
C = np.array([s['corr'] for s in S]).mean(0)
print('mean correlation (cond H^-1), lifetime row: ' + '  '.join(f'{k} {C[P.index("lifetime"), j]:+.2f}' for j, k in enumerate(P)))
print('mean correlation, long_diff row: ' + '  '.join(f'{k} {C[P.index("long_diff"), j]:+.2f}' for j, k in enumerate(P)))
