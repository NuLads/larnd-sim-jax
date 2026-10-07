"""fig19: K6 (drift from truth) vs production, lifetime error and attenuation-rate offset d(1/tau) vs true tau, per sample."""
import json, csv, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
c = json.load(open('/sdf/scratch/neutrino/pgranger/review_2026_09/CURVEK6/curve.json'))
S = {'1200': [1, 3, 7], '1800': [1, 3, 7], '3200': [1, 3, 7], '2406': [1, 2, 4]}
col = {1: 'C0', 2: 'C4', 3: 'C1', 4: 'C5', 7: 'C2'}
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
for T, v in c.items():
    t = float(T)
    for s, k, p in zip(S[T], v['k6'], v['prod']):
        for y, m, lab in ((k, 'o', 'K6 drift-from-truth'), (p, 'x', 'production')):
            di = -(y / 100) / (1 + y / 100) / t * 1e5
            ax[0].plot(t, y, m, color=col[s]); ax[1].plot(t, di, m, color=col[s])
for s in (1, 3, 7):
    ks = [(float(T), c[T]['k6'][S[T].index(s)]) for T in c if s in S[T]]; ks.sort()
    ax[1].plot([a for a, _ in ks], [-(y / 100) / (1 + y / 100) / a * 1e5 for a, y in ks], '-', color=col[s], alpha=.5, label=f'sample {s} (K6)')
ax[0].axhline(0, color='k', lw=.5); ax[1].axhline(0, color='k', lw=.5)
ax[0].set(xlabel='true lifetime [µs]', ylabel='lifetime error [%]', title='lifetime error (o K6, x production)')
ax[1].set(xlabel='true lifetime [µs]', ylabel='Δ(1/τ) [1e-5 / µs]', title='attenuation-rate offset: ~constant per sample')
ax[1].legend(fontsize=8); fig.tight_layout(); fig.savefig('figs/fig19_k6_curve.png', dpi=130)
