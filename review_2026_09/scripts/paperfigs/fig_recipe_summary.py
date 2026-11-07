"""Fig. recipe (recipe_summary), paper version (12 Oct 2026): lifetime and longitudinal-diffusion error per track sample for
the initialisation and drift-pinning configurations, all at 1e4 iterations (no continuation). The straight-start fit is
no longer shown (Pierre: mentioned in the text only). Readout: compare_arms.read (mean of the last 1e3 of 1e4 iterations).
Arms: GTRU (true starting trajectories, fitted freely), HITSEED (hit-derived start, free), STRFRZ90 / STRFRZ50 (hit-derived
start, drift held for tracks not significant at 90% / 50%; STRFRZ50 = the procedure), DRIFTTRUK6 (drift coordinate held at
the truth, 6 modes; reference)."""
import sys, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0, '/sdf/scratch/neutrino/pgranger/review_2026_09/GEOFRZ')
import compare_arms as ca
OUT = '/sdf/group/neutrino/pgranger/NuLads_draft/figures/recipe_summary'
ARMS = [('GTRU', [1, 2, 4], 'true start,\nfree'),
        ('HITSEED', [1, 2, 4], 'hit-derived\nstart, free'),
        ('STRFRZ90', [1, 2, 4], 'hit-derived,\nheld at 90%'),
        ('STRFRZ50', list(range(1, 17)), 'procedure:\nheld at 50%'),
        ('DRIFTTRUK6', [1, 2, 4], 'drift held\nat truth')]
COL = ['#999999', '#56B4E9', '#009E73', '#D55E00', '#000000']
plt.rcParams.update({'font.size': 8, 'axes.labelsize': 8, 'xtick.labelsize': 6.5, 'ytick.labelsize': 7, 'axes.linewidth': 0.6})
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5))
for k, (par, name) in enumerate((('lifetime', r'lifetime $\tau$'), ('long_diff', r'longitudinal diffusion $D_L$'))):
    ax = axes[k]
    for i, (arm, ss, lab) in enumerate(ARMS):
        v = np.array([ca.read(arm, s)[par] for s in ss])
        ax.plot(np.full(len(v), i) + np.linspace(-0.13, 0.13, len(v)), v, 'o', color=COL[i], ms=3.5, alpha=0.8, lw=0)
        ax.errorbar(i + 0.27, v.mean(), v.std(ddof=1) / np.sqrt(len(v)), fmt='s', color='k', ms=3.5, capsize=2, lw=0.8)
        print(f'{par:9s} {arm:11s} n={len(v):2d} mean {v.mean():+6.2f} +- {v.std(ddof=1)/np.sqrt(len(v)):.2f}')
    ax.axhline(0, color='k', lw=0.6)
    ax.set_xticks(range(len(ARMS))); ax.set_xticklabels([a[2] for a in ARMS])
    ax.set_ylabel(f'{name} error (%)'); ax.grid(alpha=0.25, lw=0.5, axis='y')
    ax.text(0.97, 0.95, f'({"ab"[k]})', transform=ax.transAxes, va='top', ha='right')
fig.tight_layout(w_pad=1.5)
fig.savefig(OUT + '.pdf', bbox_inches='tight'); fig.savefig(OUT + '.png', dpi=250, bbox_inches='tight'); print('saved', OUT)
