"""Figures 28-30: the geometry/dEdx quality ladder for the likelihood scans.

Two controlled degradations of the TRUE segment file (see optimize/scripts/make_quality_ladder.py):
  POSITION  rigid per-trajectory 3-D offset of 50 / 170 / 400 / 880 um RMS
  DEDX      dEdx blended toward its global mean, dEdx' = mean + f*(dEdx - mean), f = .75/.5/.25/0
plus the two pre-existing endpoints, `true` (undegraded) and `guess` (the straight-line file).

House style follows make_noise_report_plots.py. Degradation level is a MAGNITUDE, so each ladder
gets a single-hue sequential ramp (light = mild, dark = severe) rather than categorical hues; the
parameter panels use the Okabe-Ito categorical order, validated for CVD separation.
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from analyze_quality_ladder import load_run, analyse, NOM, ORDER, TAGS

OUT = 'plots/noise_report'
os.makedirs(OUT, exist_ok=True)
PL = {'Ab': 'A$_b$ (recomb.)', 'eField': 'E field', 'lifetime': 'lifetime',
      'tran_diff': 'tran. diff.', 'long_diff': 'long. diff.'}
C = dict(blue='#0072B2', orange='#E69F00', green='#009E73', red='#D55E00',
         purple='#CC79A7', grey='#666666', ink='#222222')
plt.rcParams.update({'font.size': 9, 'axes.grid': True, 'grid.alpha': .25,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.dpi': 130, 'savefig.bbox': 'tight'})

POS = [('pos50', '50 µm'), ('pos170', '170 µm'), ('pos400', '400 µm'), ('pos880', '880 µm')]
DDX = [('dedx0.75', 'f = 0.75'), ('dedx0.5', 'f = 0.5'),
       ('dedx0.25', 'f = 0.25'), ('dedx0.0', 'f = 0 (const)')]
# sequential ramps: one hue, light -> dark, monotone in lightness
RAMP_POS = ['#9ecae1', '#5aa3d0', '#2b7bba', '#08519c']
RAMP_DDX = ['#fdc07a', '#f79646', '#dd6b1f', '#a63603']

TAB = {t: analyse(t, 0) for t in TAGS}


def curve(tag, p):
    x, Cm = load_run(tag, 0)[p]
    y = Cm.sum(0)
    return 100.0 * (x - NOM[p]) / NOM[p], y - y.min()


# ---------------------------------------------------------------- fig28: the scan curves
fig, axes = plt.subplots(2, 2, figsize=(10.2, 6.4))
for row, p in enumerate(['lifetime', 'long_diff']):
    for col, (rungs, ramp, ttl) in enumerate(
            [(POS, RAMP_POS, 'position error (rigid per-track offset)'),
             (DDX, RAMP_DDX, 'dE/dx degradation (blend toward the mean)')]):
        ax = axes[row, col]
        xt, yt = curve('true', p)
        ax.plot(xt, yt, color=C['ink'], lw=1.8, ls='--', zorder=5)
        v = TAB['true'][p]['pct']
        ax.axvline(v, color=C['ink'], lw=1.0, ls=':', zorder=5)
        for (tag, lab), colr in zip(rungs, ramp):
            xx, yy = curve(tag, p)
            ax.plot(xx, yy, color=colr, lw=1.7, marker='o', ms=3.4, zorder=3)
            ax.axvline(TAB[tag][p]['pct'], color=colr, lw=1.0, ls=':', zorder=2)
        ax.axvline(0, color=C['grey'], lw=1.4)
        # Zoom hard. The grid step is many sigma wide (see vertex5's docstring), so an autoscaled
        # panel is dominated by the steep walls and every well looks identical; the whole story
        # lives within ~2 grid steps of the minimum.
        lim, ymax = {'lifetime': ((-32, 42), 340.0),
                     'long_diff': ((-50, 90), 430.0)}[p]
        ax.set_xlim(*lim); ax.set_ylim(0, ymax)
        ax.set_xlabel(f'{PL[p]}: offset from truth (%)')
        if col == 0:
            ax.set_ylabel('Δ NLL  (summed over 21 batches)')
        if row == 0:
            ax.set_title(ttl, fontsize=9.5)
        h = [Line2D([], [], color=C['ink'], ls='--', lw=1.8, label='true geometry')]
        h += [Line2D([], [], color=c, lw=1.7, marker='o', ms=3.4, label=l)
              for (_, l), c in zip(rungs, ramp)]
        ax.legend(handles=h, fontsize=7.4, frameon=False, loc='upper center', ncol=1)
fig.suptitle('Fig 28 — how the likelihood well moves under controlled degradation\n'
             'dotted verticals = fitted minimum · grey vertical = truth', fontsize=10.5)
fig.tight_layout(rect=[0, 0, 1, .94])
fig.savefig(f'{OUT}/fig28_ladder_scans.png'); plt.close(fig)

# ---------------------------------------------------------------- fig29: the calibration curve
fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2))
SER = [('lifetime', C['blue']), ('long_diff', C['orange']), ('tran_diff', C['green']),
       ('Ab', C['red']), ('eField', C['purple'])]
for ax, (rungs, xlabs, ttl) in zip(axes, [
        (['true'] + [t for t, _ in POS], ['0\n(true)', '50', '170', '400', '880'],
         'position error RMS (µm)'),
        (['true'] + [t for t, _ in DDX], ['1.0\n(true)', '0.75', '0.5', '0.25', '0.0'],
         'dE/dx retained fraction  f')]):
    xs = np.arange(len(rungs))
    for p, colr in SER:
        m = np.array([TAB[t][p]['pct'] for t in rungs])
        e = np.array([TAB[t][p]['boot'] for t in rungs])
        ax.errorbar(xs, m, yerr=e, color=colr, lw=1.8, marker='o', ms=5,
                    capsize=3, label=PL[p], zorder=3)
    # statistical resolution of the dataset, from the NLL curvature at the undegraded point
    for p, colr in SER[:2]:
        s = TAB['true'][p]['fisher']
        ax.axhspan(-s, s, color=colr, alpha=.10, zorder=0)
    ax.axhline(0, color=C['grey'], lw=1.4)
    ax.set_xticks(xs); ax.set_xticklabels(xlabs)
    ax.set_xlabel(ttl); ax.set_title(ttl.split('(')[0].strip(), fontsize=9.5)
    ax.set_ylim(-30, 50)
axes[0].set_ylabel('minimum offset from truth (%)')
# legend goes in the LEFT panel: its lower-left is empty, whereas the right panel's is where
# long. diffusion actually goes
axes[0].legend(fontsize=8, frameon=False, ncol=2, loc='lower left')
fig.suptitle('Fig 29 — the calibration curve: where the likelihood minimum sits vs input quality\n'
             'shaded bands = 1σ statistical resolution of this dataset (lifetime, long. diff.)',
             fontsize=10.5)
fig.tight_layout(rect=[0, 0, 1, .90])
fig.savefig(f'{OUT}/fig29_ladder_calibration.png'); plt.close(fig)

# ---------------------------------------------------------------- fig30: bias vs variance
fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.0), sharey=False)
COND = ['true'] + [t for t, _ in POS] + [t for t, _ in DDX] + ['guess']
LAB = ['true', '50µm', '170µm', '400µm', '880µm',
       'f.75', 'f.50', 'f.25', 'f.00', 'guess']
FAM = [C['grey']] + [RAMP_POS[-1]] * 4 + [RAMP_DDX[-1]] * 4 + [C['purple']]
xs = np.arange(len(COND))
for p, colr in [('lifetime', C['blue']), ('long_diff', C['orange'])]:
    m = np.array([TAB[t][p]['pct'] for t in COND])
    e = np.array([TAB[t][p]['boot'] for t in COND])
    axes[0].errorbar(xs, m, yerr=e, color=colr, lw=0, elinewidth=1.6, marker='o',
                     ms=6, capsize=3, label=PL[p])
    axes[1].plot(xs, e, color=colr, lw=1.8, marker='s', ms=5.5, label=PL[p])
for ax in axes:
    ax.set_xticks(xs); ax.set_xticklabels(LAB, rotation=45, ha='right', fontsize=8)
    for t, c in zip(xs, FAM):
        ax.get_xticklabels()[t].set_color(c)
    ax.axvspan(0.5, 4.5, color=RAMP_POS[0], alpha=.16, zorder=0)
    ax.axvspan(4.5, 8.5, color=RAMP_DDX[0], alpha=.16, zorder=0)
axes[0].axhline(0, color=C['grey'], lw=1.4)
axes[0].set_ylabel('minimum offset from truth (%)')
axes[0].set_title('(a) where the minimum sits — dE/dx shifts it', fontsize=9.5)
axes[1].set_ylabel('bootstrap uncertainty (% points)')
axes[1].set_title('(b) how well it is determined — position blurs it', fontsize=9.5)
axes[1].set_yscale('log')
for ax in axes:
    ax.legend(fontsize=8, frameon=False)
    ax.text(2.5, ax.get_ylim()[1], ' position ladder', fontsize=7.6, color=RAMP_POS[-1],
            va='top', ha='center')
    ax.text(6.5, ax.get_ylim()[1], ' dE/dx ladder', fontsize=7.6, color=RAMP_DDX[-1],
            va='top', ha='center')
fig.suptitle('Fig 30 — the two degradation axes do different things:\n'
             'dE/dx error displaces the minimum (bias); position error widens it (variance)',
             fontsize=10.5)
fig.tight_layout(rect=[0, 0, 1, .89])
fig.savefig(f'{OUT}/fig30_bias_vs_variance.png'); plt.close(fig)

print('wrote fig28_ladder_scans.png, fig29_ladder_calibration.png, fig30_bias_vs_variance.png')
for p in ['lifetime', 'long_diff']:
    print(f'{p:10s} grid systematic (|5pt-3pt|):',
          ' '.join(f'{t}={TAB[t][p]["grid_sys"]:.2f}' for t in ['true', 'dedx0.5', 'guess']))
