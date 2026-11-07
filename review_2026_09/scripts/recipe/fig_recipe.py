"""fig20: lifetime and long_diff error per track sample for each configuration, production -> truth-free recipe."""
import sys, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0, '/sdf/scratch/neutrino/pgranger/review_2026_09/GEOFRZ')
import compare_arms as ca
ARMS = [('DSREP', [1, 2, 3, 4, 5, 6, 7, 8, 9], 'production\n(straight start)', 'C3'),
        ('GTRU', [1, 2, 4], 'truth start\n(free)', 'C1'),
        ('HITSEED', [1, 2, 4], 'hit seed\n(free)', 'C4'),
        ('STRFRZ90', [1, 2, 4], 'hit seed +\nfreeze CL 0.9', 'C8'),
        ('STRFRZ50', [1, 2, 3, 4, 5, 6, 7, 8, 9], 'RECIPE\nhit seed +\nfreeze CL 0.5', 'C2'),
        ('DRIFTTRUK6', [1, 2, 4], 'truth drift\nfrozen (K6)', 'C7')]
fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), sharey=False)
for k, par in enumerate(['lifetime', 'long_diff']):
    ax = axes[k]
    for i, (arm, ss, lab, col) in enumerate(ARMS):
        v = np.array([ca.read(arm, s)[par] for s in ss])
        ax.plot(np.full(len(v), i) + np.linspace(-0.12, 0.12, len(v)), v, 'o', color=col, ms=6, alpha=0.85)
        ax.errorbar(i + 0.25, v.mean(), v.std(ddof=1) / np.sqrt(len(v)), fmt='s', color='k', ms=5, capsize=3)
        ax.text(i + 0.3, v.mean(), f' {v.mean():+.1f}', va='center', fontsize=8)
    ax.axhline(0, color='k', lw=0.6); ax.axhspan(-5, 5, color='g', alpha=0.06)
    ax.set_xticks(range(len(ARMS))); ax.set_xticklabels([a[2] for a in ARMS], fontsize=8)
    ax.set_ylabel(f'{par} error vs truth [%]'); ax.set_title(f'{par}: per track sample (dots), mean ± SE (squares)')
fig.suptitle('Truth-free calibration recipe (100 batches × 400 cm, τ = 2406 µs; simulation closure)', fontsize=11)
fig.tight_layout(); fig.savefig('/sdf/group/neutrino/pgranger/larnd-sim-jax/review_2026_09/figs/fig20_recipe_summary.png', dpi=130)
print('saved')
