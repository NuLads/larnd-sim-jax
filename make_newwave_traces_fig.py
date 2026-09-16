"""Figs 86/87 — the actual fit iterations behind figs 84/85.

86: the 15 cold replicas, one line per data seed, lifetime + long_diff + loss + position.
    Every seed runs the SAME schedule on different tracks: identical overshoot-then-truncated-
    relaxation shape, different landing heights. The spread is not noise around a path — it is
    where each relaxation happened to be frozen.

87: the calibration-init arms on replicas 3 and 8, lifetime traces coloured by START. The
    trajectories separate immediately and never re-merge: the transient carries the memory of
    where calibration began all the way to the endpoint.
"""
import glob, pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import sliding_window_view

OUT = 'plots/noise_report'
plt.rcParams.update({'font.size': 9.5, 'axes.grid': True, 'grid.alpha': .25,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.dpi': 130, 'savefig.bbox': 'tight'})
GREEN, GREY = '#009E73', '#888888'
W = 200


def sm(v, w=W):
    v = np.asarray(v, float)
    return np.median(sliding_window_view(v, w), axis=1) if len(v) > w else v


def load(tag):
    fs = glob.glob(f'fit_result/sci_full_{tag}/history_iter10000_*.pkl')
    if not fs:
        return None
    h = pickle.load(open(fs[0], 'rb'))
    d = {'loss': sm(np.asarray(h['losses_iter'], float)),
         'pos': sm(np.ravel(h['pos_residual_iter']) * 1e4)}
    for p in ['lifetime', 'long_diff']:
        t = float(np.ravel(h[p + '_target'])[0])
        d[p] = sm(100 * (np.ravel(h[p + '_iter'])[1:] / t - 1))
    d['lt0'] = 100 * (np.ravel(h['lifetime_iter'])[0]
                      / float(np.ravel(h['lifetime_target'])[0]) - 1)
    return d


# ---------------- FIG 86: the 15 cold replicas ----------------
SEEDS = [s for s in range(1, 17) if load(f'DSREP{s}')]
cmap = plt.cm.viridis(np.linspace(0, .92, len(SEEDS)))
fig, ax = plt.subplots(1, 4, figsize=(15.5, 3.9))
PAN = [('lifetime', 'lifetime error (%)', (-12, 48)),
       ('long_diff', 'long. diffusion error (%)', (-25, 60)),
       ('pos', 'position residual (µm)', (80, 700)),
       ('loss', 'loss', (8300, 12500))]
for a, (k, lab, yl) in zip(ax, PAN):
    for col, s in zip(cmap, SEEDS):
        d = load(f'DSREP{s}')
        x = np.linspace(0, 10, len(d[k]))
        a.plot(x, d[k], color=col, lw=1.0, alpha=.85)
        a.plot(x[-1], d[k][-1], 'o', color=col, ms=3.5)
    if k in ('lifetime', 'long_diff'):
        a.axhline(0, color=GREEN, lw=1.6)
    a.set_ylim(*yl)
    a.set_xlabel('iteration (×1000)')
    a.set_ylabel(lab)
ax[0].set_title('every seed: overshoot → truncated relaxation', fontsize=9.5)
ax[3].set_title('the loss converges cleanly on all seeds', fontsize=9.5)
sm_ = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(min(SEEDS), max(SEEDS)))
cb = fig.colorbar(sm_, ax=ax[3], pad=.02)
cb.set_label('data seed', fontsize=8.5)
fig.suptitle('Fig 86 — the actual iterations of the %d cold replicas (one line per data seed, identical schedule). '
             'The landing SPREAD is not noise about a common path:\nevery seed traces the same overshoot-and-relax shape, '
             'and the anneal freezes each one at a different point along its relaxation.' % len(SEEDS),
             fontsize=10.5, y=1.08)
fig.tight_layout()
fig.savefig(f'{OUT}/fig86_replica_traces.png')
print('wrote fig86, n =', len(SEEDS))

# ---------------- FIG 87: calibration-init arms ----------------
ARMS = [('CINITL', 'start low −31%', '#0072B2'), ('DSREP', 'start nominal −8.6%', '#222222'),
        ('CINITT', 'start AT TRUTH', '#009E73'), ('CINITH', 'start high +14%', '#E69F00')]
# the all-shifted arm is NOT here: its A_b init was out of range and clamped at the bound, so it
# is not a valid 'where did calibration start' variant -- fig 88 studies it as its own experiment.
fig, ax = plt.subplots(1, 3, figsize=(14, 4.1))
for a, s in zip(ax[:2], (3, 8)):
    for pre, lab, col in ARMS:
        d = load(f'{pre}{s}')
        if d is None:
            continue
        x = np.linspace(0, 10, len(d['lifetime']))
        a.plot(x, d['lifetime'], color=col, lw=1.3, label=f'{lab} → {d["lifetime"][-1]:+.1f}%')
        a.plot(0, d['lt0'], '<', color=col, ms=7, clip_on=False)
        a.plot(x[-1], d['lifetime'][-1], 'o', color=col, ms=5)
    a.axhline(0, color=GREEN, lw=1.6)
    a.set_ylim(-36, 62)
    a.set_xlabel('iteration (×1000)')
    a.set_ylabel('lifetime error (%)')
    a.set_title(f'replica {s}: same data, same target,\nonly the calibration START differs', fontsize=9.5)
    a.legend(frameon=False, fontsize=7.6, loc='upper right')
# third panel: zoom on the last 3000 iterations, both replicas
for pre, lab, col in ARMS:
    for s, ls in ((3, '-'), (8, '--')):
        d = load(f'{pre}{s}')
        if d is None:
            continue
        x = np.linspace(0, 10, len(d['lifetime']))
        m = x > 6.5
        ax[2].plot(x[m], d['lifetime'][m], ls, color=col, lw=1.3)
ax[2].axhline(0, color=GREEN, lw=1.6)
ax[2].set_ylim(0, 28)
ax[2].set_xlabel('iteration (×1000)')
ax[2].set_ylabel('lifetime error (%)')
ax[2].set_title('endpoints, zoomed (solid rep 3, dashed rep 8):\nthe arms never re-merge', fontsize=9.5)
fig.suptitle('Fig 87 — the iterations behind fig 84: where calibration STARTS (◀ markers on the left axis) separates the trajectories\n'
             'within the first few hundred iterations, and they never reconverge — the transient carries init memory to the endpoint.',
             fontsize=10.5, y=1.07)
fig.tight_layout()
fig.savefig(f'{OUT}/fig87_cinit_traces.png')
print('wrote fig87')
