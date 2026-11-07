"""Fig. bias_vs_variance, paper version (12 Oct 2026): the input-quality ladder of make_ladder_plots.py (fig30), same data
(one-dimensional likelihood scans of each detector parameter, the others at truth, 21 batches, fit_result/loss_profile),
without the internal title and without the `guess` column (the obsolete straight-line input file).
  (a) position of the likelihood minimum, lifetime and longitudinal diffusion, with its bootstrap uncertainty over batches;
  (b) that uncertainty on a log axis.
usage (repo root): python review_2026_09/scripts/paperfigs/fig_bias_vs_variance.py"""
import os, sys
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0, '/sdf/group/neutrino/pgranger/larnd-sim-jax')
os.chdir('/sdf/group/neutrino/pgranger/larnd-sim-jax')
from analyze_quality_ladder import analyse
OUT = '/sdf/group/neutrino/pgranger/NuLads_draft/figures/bias_vs_variance'
COND = ['true', 'pos50', 'pos170', 'pos400', 'pos880', 'dedx0.75', 'dedx0.5', 'dedx0.25', 'dedx0.0']
LAB = ['true', '50 $\\mu$m', '170 $\\mu$m', '400 $\\mu$m', '880 $\\mu$m', '$f$ = 0.75', '$f$ = 0.5', '$f$ = 0.25', '$f$ = 0']
TAB = {t: analyse(t, 0) for t in COND}
POSC, DDXC = '#08519c', '#a63603'
plt.rcParams.update({'font.size': 8, 'axes.labelsize': 8, 'legend.fontsize': 7, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
                     'axes.linewidth': 0.6})
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7))
xs = np.arange(len(COND))
for p, colr, lab in [('lifetime', '#0072B2', 'lifetime $\\tau$'), ('long_diff', '#E69F00', 'longitudinal diffusion $D_L$')]:
    m = np.array([TAB[t][p]['pct'] for t in COND]); e = np.array([TAB[t][p]['boot'] for t in COND])
    axes[0].errorbar(xs, m, yerr=e, color=colr, lw=0, elinewidth=1.2, marker='o', ms=4, capsize=2, label=lab)
    axes[1].plot(xs, e, color=colr, lw=1.3, marker='s', ms=4, label=lab)
    print(p, ' '.join(f'{t}={v:+.1f}+-{w:.1f}' for t, v, w in zip(COND, m, e)))
for k, ax in enumerate(axes):
    ax.set_xticks(xs); ax.set_xticklabels(LAB, rotation=45, ha='right')
    for t, tl in enumerate(ax.get_xticklabels()):
        tl.set_color('k' if t == 0 else POSC if t <= 4 else DDXC)
    ax.axvspan(0.5, 4.5, color='#9ecae1', alpha=.16, zorder=0); ax.axvspan(4.5, 8.5, color='#fdc07a', alpha=.16, zorder=0)
    ax.grid(alpha=0.25, lw=0.5)
    ax.text(0.03 if k == 0 else 0.97, 0.95 if k == 0 else 0.05, f'({"ab"[k]})', transform=ax.transAxes,
            va='top' if k == 0 else 'bottom', ha='left' if k == 0 else 'right')
axes[0].axhline(0, color='#666666', lw=0.8)
axes[0].set_ylabel('likelihood minimum, error (%)')
axes[1].set_ylabel('uncertainty of the minimum (points)'); axes[1].set_yscale('log')
axes[0].legend(loc='lower left', frameon=False)
for ax in axes:
    yl = ax.get_ylim()
    yt = yl[1] - 0.04 * (yl[1] - yl[0]) if ax is axes[0] else yl[1] / 1.25
    ax.text(2.5, yt, 'random rigid offsets', fontsize=6.5, color=POSC, va='top', ha='center')
    ax.text(6.5, yt, '$dE/dx$ blended to its mean', fontsize=6.5, color=DDXC, va='top', ha='center')
fig.tight_layout(w_pad=1.6)
fig.savefig(OUT + '.pdf', bbox_inches='tight'); fig.savefig(OUT + '.png', dpi=250, bbox_inches='tight'); print('saved', OUT)
