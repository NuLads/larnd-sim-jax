"""Fig 85 — the ensemble at n = 15, the fig-73 view redone with double the samples, plus the
calibration-init arms shown as what they are: a SECOND source of scatter on top of sample luck.

Blue = the cold replicas (one per data seed, identical config). Orange = the calibration-init
variants on replicas 3 and 8 (same data, same target, only the calibration START differs).
Black diamond = replica mean +- SE. Red star = the campaign's historical sample (BT2).

The point of putting them on one axis: moving the calibration START spans as much range
(+24.6 down to -59.3 on lifetime) as changing the DATA does. Both are 'where the optimizer
happened to land', not properties of the estimator.
"""
import glob, pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import sliding_window_view

OUT = 'plots/noise_report'
C = dict(blue='#0072B2', orange='#E69F00', green='#009E73', red='#D55E00', ink='#222222')
plt.rcParams.update({'font.size': 10, 'axes.grid': True, 'grid.alpha': .22,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.spines.left': False, 'figure.dpi': 130, 'savefig.bbox': 'tight'})
P = ['lifetime', 'long_diff', 'Ab', 'tran_diff', 'eField']
W = 200


def sm(v, w=W):
    v = np.asarray(v, float)
    return np.median(sliding_window_view(v, w), axis=1) if len(v) > w else v


def read(tag):
    fs = glob.glob(f'fit_result/sci_full_{tag}/history_iter10000_*.pkl')
    if not fs:
        return None
    h = pickle.load(open(fs[0], 'rb'))
    return {p: 100 * (np.median(sm(np.ravel(h[p + '_iter'])[1:])[-300:])
                      / float(np.ravel(h[p + '_target'])[0]) - 1) for p in P}


SEEDS = [s for s in range(1, 17)]
REP = {s: r for s in SEEDS if (r := read(f'DSREP{s}'))}
CIN = {f'{k}{s}': r for s in (3, 8) for k in ('CINITT', 'CINITH', 'CINITL', 'CINITA')
       if (r := read(f'{k}{s}'))}
OLD = read('BT2')

TITLES = {
    'lifetime': ('LIFETIME', 'n=%d:  +%.2f ± %.2f%%   (sd %.1f)',
                 'bias confirmed on fresh samples; seed 13 at −0.6%'),
    'long_diff': ('LONGITUDINAL DIFFUSION', 'n=%d:  +%.1f ± %.1f%%   (sd %.1f)',
                  'the larger bias of the two'),
    'Ab': ('A$_b$', 'n=%d:  %+.3f ± %.3f%%   (sd %.2f)',
           'centred on truth: the "+0.27%" was the sample'),
    'tran_diff': ('TRANSVERSE DIFFUSION', 'n=%d:  %+.2f ± %.2f%%   (sd %.1f)',
                  'small, now established at 3.4σ'),
    'eField': ('E FIELD', 'n=%d:  %+.4f ± %.4f%%   (sd %.3f)', 'tiny but systematic'),
}

fig = plt.figure(figsize=(16.4, 9.4))
gs = fig.add_gridspec(2, 3, hspace=.45, wspace=.18)
rng = np.random.default_rng(3)

for k, p in enumerate(P):
    a = fig.add_subplot(gs[k // 3, k % 3])
    v = np.array([REP[s][p] for s in sorted(REP)])
    ci = np.array([CIN[t][p] for t in sorted(CIN)])
    m, se, sd = v.mean(), v.std(ddof=1) / np.sqrt(len(v)), v.std(ddof=1)

    a.axvline(0, color=C['green'], lw=2.2, zorder=1)
    a.scatter(v, .55 + rng.uniform(-.09, .09, len(v)), s=52, color=C['blue'],
              alpha=.85, edgecolors='white', lw=.6, zorder=3, label='cold replicas (data seed varies)')
    a.scatter(ci, .18 + rng.uniform(-.05, .05, len(ci)), s=48, color=C['orange'], marker='s',
              alpha=.85, edgecolors='white', lw=.6, zorder=3,
              label='calibration-init variants (reps 3, 8)')
    a.errorbar(m, .82, xerr=se, fmt='D', color=C['ink'], ms=8, capsize=4, zorder=4,
               label='replica mean ± SE')
    if OLD:
        a.scatter([OLD[p]], [.95], marker='*', s=200, color=C['red'], zorder=4,
                  label="campaign's historical sample")
    a.annotate('truth', (0, 1.30), ha='center', color=C['green'], fontweight='bold', fontsize=9)
    ttl, stat, note = TITLES[p]
    a.set_title(f'{ttl}\n' + stat % (len(v), m, se, sd) + f'\n{note}',
                fontsize=9.5, loc='left', pad=10, linespacing=1.6)
    a.set_yticks([])
    a.set_ylim(0, 1.42)
    a.set_xlabel(f'{p} error at the pinned truth (%)', fontsize=9)

ax = fig.add_subplot(gs[1, 2])
ax.axis('off')
h, l = fig.axes[0].get_legend_handles_labels()
ax.legend(h, l, frameon=False, fontsize=10, loc='center')
ax.text(.5, .12, 'Same target and truth throughout.\nBlue varies the DATA; orange varies only\n'
                 'the calibration STARTING POINT —\nand spans just as much range.',
        ha='center', fontsize=9.5, color='#444444', transform=ax.transAxes)

fig.suptitle('Fig 85 — the ensemble at n = %d (fig 73 redone with double the samples), with the calibration-init arms on the same axis.'
             % len(REP), fontsize=12.5, y=.98)
fig.savefig(f'{OUT}/fig85_ensemble15.png')
print('wrote fig85, n =', len(REP), 'cinit =', len(CIN))
