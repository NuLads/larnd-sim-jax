"""Figures 31-33: the dE/dx mean axis, the completed ladder, and the fit-side arms.

fig31  the dominant bias channel -- lifetime response to a dE/dx MEAN error, and the resulting
       decomposition of the straight-line guess file's +15.2%.
fig32  the completed ladder: every degradation axis on one footing.
fig33  fit-side arms: the shuffle pair (open question 5) and the 4x-data run.

House style follows make_noise_report_plots.py / make_ladder_plots.py (Okabe-Ito, validated for
CVD separation). Degradation LEVEL is a magnitude -> single-hue sequential ramps; distinct
PARAMETERS -> the categorical order.
"""
import io, os, contextlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from analyze_quality_ladder import analyse, NOM, ORDER
# robust_convergence runs its whole campaign analysis at import time and prints tables; silence it.
with contextlib.redirect_stdout(io.StringIO()):
    import robust_convergence as RC
import pickle

OUT = 'plots/noise_report'
os.makedirs(OUT, exist_ok=True)
PL = {'Ab': 'A$_b$ (recomb.)', 'eField': 'E field', 'lifetime': 'lifetime',
      'tran_diff': 'tran. diff.', 'long_diff': 'long. diff.'}
C = dict(blue='#0072B2', orange='#E69F00', green='#009E73', red='#D55E00',
         purple='#CC79A7', grey='#666666', ink='#222222')
plt.rcParams.update({'font.size': 9, 'axes.grid': True, 'grid.alpha': .25,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.dpi': 130, 'savefig.bbox': 'tight'})

T = {t: analyse(t, 0) for t in
     ['true', 'pos880', 'chord0.5', 'chord1.0', 'dedx0.40b', 'dedx0.0',
      'combo', 'dmeanp2', 'dmeanm2', 'reseg', 'guess']}
GUESS_MEAN_DEFICIT = -0.440          # measured: guess 1.87704 vs true 1.88533 (length-weighted)

# ---------------------------------------------------------------- fig31
fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.3))
ax = axes[0]
xs = np.array([-2.0, 0.0, 2.0])
ys = np.array([T['dmeanm2']['lifetime']['pct'], T['true']['lifetime']['pct'],
               T['dmeanp2']['lifetime']['pct']])
es = np.array([T['dmeanm2']['lifetime']['boot'], T['true']['lifetime']['boot'],
               T['dmeanp2']['lifetime']['boot']])
ax.errorbar(xs, ys, yerr=es, color=C['blue'], lw=2, marker='o', ms=7, capsize=4, zorder=3,
            label='measured (scan)')
# local slope from the negative arm, which is the side the guess file sits on
slope = (ys[0] - ys[1]) / (xs[0] - xs[1])
xf = np.linspace(-2.2, 0.2, 50)
ax.plot(xf, ys[1] + slope * xf, ls='--', lw=1.2, color=C['grey'], zorder=1,
        label=f'local slope {slope:.1f}% per 1% of mean')
pred = ys[1] + slope * GUESS_MEAN_DEFICIT
ax.axvline(GUESS_MEAN_DEFICIT, color=C['red'], lw=1.4, ls=':')
ax.plot([GUESS_MEAN_DEFICIT], [pred], marker='*', ms=15, color=C['red'], zorder=5,
        label=f'guess file mean deficit ({GUESS_MEAN_DEFICIT:.2f}%) $\\rightarrow$ {pred:+.1f}%')
g = T['guess']['lifetime']
ax.axhspan(g['pct'] - g['boot'], g['pct'] + g['boot'], color=C['red'], alpha=.13, zorder=0)
ax.axhline(g['pct'], color=C['red'], lw=1.2, ls='-.',
           label=f"guess file OBSERVED ({g['pct']:+.1f} ± {g['boot']:.1f}%)")
ax.axhline(0, color=C['grey'], lw=1.2); ax.axvline(0, color=C['grey'], lw=1.2)
ax.set_xlabel('dE/dx mean error (%)'); ax.set_ylabel('lifetime minimum offset (%)')
ax.set_title(f'(a) lifetime is ~{abs(slope):.0f}× levered on the dE/dx mean', fontsize=9.5)
ax.legend(fontsize=7.4, frameon=False, loc='upper right')

# (b) all axes on one footing
ax = axes[1]
CONDS = [('position 880 µm\n(random)', 'pos880', C['green']),
         ('chord-cut 1011 µm\n(systematic)', 'chord1.0', C['green']),
         ('dE/dx spread\nf = 0.40', 'dedx0.40b', C['orange']),
         ('dE/dx spread\nf = 0 (const)', 'dedx0.0', C['orange']),
         ('dE/dx mean\n−0.44% (guess)', None, C['blue']),
         ('dE/dx mean\n−2%', 'dmeanm2', C['blue'])]
lab, val, err, col = [], [], [], []
for name, key, c in CONDS:
    lab.append(name); col.append(c)
    if key is None:
        val.append(pred - ys[1]); err.append(0.0)
    else:
        val.append(T[key]['lifetime']['pct'] - T['true']['lifetime']['pct'])
        err.append(T[key]['lifetime']['boot'])
y = np.arange(len(lab))
ax.barh(y, val, xerr=err, color=col, edgecolor='white', linewidth=.7, capsize=3)
ax.set_yticks(y); ax.set_yticklabels(lab, fontsize=8)
ax.invert_yaxis(); ax.axvline(0, color=C['grey'], lw=1.2)
ax.set_xlabel('lifetime bias induced, relative to the undegraded scan (% points)')
ax.set_title('(b) only the dE/dx mean moves the answer', fontsize=9.5)
ax.legend(handles=[Line2D([], [], color=C['green'], lw=7, label='geometry'),
                   Line2D([], [], color=C['orange'], lw=7, label='dE/dx spread'),
                   Line2D([], [], color=C['blue'], lw=7, label='dE/dx mean')],
          fontsize=8, frameon=False, loc='upper right')
fig.suptitle('Fig 31 — the dE/dx MEAN is the dominant bias channel, and it explains the guess file',
             fontsize=10.5)
fig.tight_layout(rect=[0, 0, 1, .93])
fig.savefig(f'{OUT}/fig31_dedx_mean_leverage.png'); plt.close(fig)

# ---------------------------------------------------------------- fig32
ORDER_C = ['true', 'reseg', 'pos880', 'chord0.5', 'chord1.0', 'dedx0.40b', 'dedx0.0',
           'combo', 'dmeanm2', 'dmeanp2', 'guess']
NICE = {'true': 'true', 'reseg': 're-segmented', 'pos880': 'pos 880µm', 'chord0.5': 'chord 0.5',
        'chord1.0': 'chord 1.0', 'dedx0.40b': 'dEdx f=0.40', 'dedx0.0': 'dEdx f=0',
        'combo': 'chord1 + f=0.40', 'dmeanm2': 'mean −2%', 'dmeanp2': 'mean +2%', 'guess': 'guess'}
FAM = {'true': C['grey'], 'reseg': C['purple'], 'pos880': C['green'], 'chord0.5': C['green'],
       'chord1.0': C['green'], 'dedx0.40b': C['orange'], 'dedx0.0': C['orange'],
       'combo': C['red'], 'dmeanm2': C['blue'], 'dmeanp2': C['blue'], 'guess': C['ink']}
fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.3), sharey=True)
xs = np.arange(len(ORDER_C))
for ax, p in zip(axes, ['lifetime', 'long_diff']):
    v = np.array([T[t][p]['pct'] for t in ORDER_C])
    e = np.array([T[t][p]['boot'] for t in ORDER_C])
    ax.bar(xs, v, yerr=e, color=[FAM[t] for t in ORDER_C], edgecolor='white',
           linewidth=.7, capsize=3)
    ax.axhline(0, color=C['grey'], lw=1.2)
    ax.set_xticks(xs); ax.set_xticklabels([NICE[t] for t in ORDER_C], rotation=45,
                                          ha='right', fontsize=8)
    for t, xx in zip(ORDER_C, xs):
        ax.get_xticklabels()[xx].set_color(FAM[t])
    ax.set_title(PL[p], fontsize=9.5)
axes[0].set_ylabel('minimum offset from truth (%)')
axes[0].legend(handles=[Line2D([], [], color=FAM[k], lw=7, label=l) for k, l in
                        [('true', 'undegraded'), ('reseg', 'segmentation'), ('pos880', 'geometry'),
                         ('dedx0.0', 'dE/dx spread'), ('dmeanm2', 'dE/dx mean'),
                         ('combo', 'combined'), ('guess', 'guess file')]],
               fontsize=7.6, frameon=False, ncol=2, loc='lower left')
fig.suptitle('Fig 32 — the completed quality ladder: every axis on one footing\n'
             'geometry (green) and segmentation (purple) are null; only dE/dx moves the minimum',
             fontsize=10.5)
fig.tight_layout(rect=[0, 0, 1, .90])
fig.savefig(f'{OUT}/fig32_ladder_complete.png'); plt.close(fig)

# ---------------------------------------------------------------- fig33
RUNS = [('ANNEALLONG\n(reference)', 'fit_result/sci_full_ANNEALLONG', C['grey']),
        ('SHUFOFF2\nsequential', 'fit_result/sci_full_SHUFOFF2', C['blue']),
        ('SHUFON2\nshuffled', 'fit_result/sci_full_SHUFON2', C['orange']),
        ('NB400FIX\n4× data', 'fit_result/sci_full_NB400FIX', C['green'])]
res = []
for name, d, c in RUNS:
    with contextlib.redirect_stdout(io.StringIO()):
        r = RC.analyse(name, d, True)
    res.append((name, r, c))
fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.0))
P3 = ['lifetime', 'long_diff', 'tran_diff']
for ax, p in zip(axes[:2], P3[:2]):
    for k, (name, r, c) in enumerate(res):
        vals = r['rows'][p]
        ax.errorbar([k], [np.mean(vals)], yerr=[np.std(vals)], color=c, marker='o', ms=8,
                    capsize=5, lw=0, elinewidth=2)
        ax.scatter([k] * len(vals), vals, color=c, alpha=.45, s=22, zorder=3)
    ax.axhline(0, color=C['grey'], lw=1.2)
    ax.set_xticks(range(len(res)))
    ax.set_xticklabels([n for n, _, _ in res], fontsize=7.6)
    ax.set_title(PL[p], fontsize=9.5)
axes[0].set_ylabel('error vs truth (%)  — dots = individual seeds')
ax = axes[2]
pos = [np.mean(r['pos']) for _, r, _ in res]
ax.bar(range(len(res)), pos, color=[c for _, _, c in res], edgecolor='white', linewidth=.7)
ax.set_xticks(range(len(res))); ax.set_xticklabels([n for n, _, _ in res], fontsize=7.6)
ax.set_ylabel('position residual (µm)'); ax.set_title('geometry', fontsize=9.5)
for k, v in enumerate(pos):
    ax.text(k, v, f'{v:.0f}', ha='center', va='bottom', fontsize=8)
fig.suptitle('Fig 33 — fit-side arms. Shuffling changes nothing (open question 5). 4× data tightens\n'
             'long. diffusion but starves per-track geometry. NOTE: only the last three share\n'
             'chain_decay_rate=0.999; ANNEALLONG used 0.9997, so its geometry is NOT comparable.',
             fontsize=9.8)
fig.tight_layout(rect=[0, 0, 1, .86])
fig.savefig(f'{OUT}/fig33_fit_arms.png'); plt.close(fig)

print('wrote fig31_dedx_mean_leverage.png, fig32_ladder_complete.png, fig33_fit_arms.png')
print(f'  local slope {slope:.2f} %lifetime per 1% dEdx mean; guess deficit {GUESS_MEAN_DEFICIT}% '
      f'-> predicted {pred:+.2f}%, observed {T["guess"]["lifetime"]["pct"]:+.2f}%')
for name, r, _ in res:
    print(f'  {name.splitlines()[0]:11s} lifetime {np.mean(r["rows"]["lifetime"]):+6.2f}±'
          f'{np.std(r["rows"]["lifetime"]):5.2f}  long_diff sd {np.std(r["rows"]["long_diff"]):5.2f}'
          f'  pos {np.mean(r["pos"]):.0f} um')
