"""Figs 80/81 — what correlates with what: loss vs dE/dx error vs position error vs the
calibration soft plane, across ~20 converged endpoints and along the trajectories.

FIG 80 (endpoints): every completed run of the replica family with the IDENTICAL objective
(mean_w=2500, no drift penalty): pass 1 on all 8 seeds, the multi-pass chains and gentle-relax
arms on seeds 1/3/7/8, plus GTRU8/HOT8/WU3000 (schedule/init variants on seed 8). Loss is centred
per seed (Δ vs that seed's pass-1) because absolute loss is sample-dependent. LATE8/WU6000
(catastrophic, +55/+59%) and DW* (different objective) excluded.

FIG 81 (trajectories, seed 8): phase planes loss-vs-quantity along every arm of the richest seed.
The expected discovery: position and dE/dx collapse onto one monotone band (they ARE the loss),
while lifetime loops and lands anywhere on the flat valley floor at fixed loss.
"""
import glob, pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import sliding_window_view

OUT = 'plots/noise_report'
plt.rcParams.update({'font.size': 9, 'axes.grid': True, 'grid.alpha': .25,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.dpi': 130, 'savefig.bbox': 'tight'})
C = dict(p1='#222222', p2='#0072B2', p3='#E69F00', p4='#D55E00',
         rlx='#009E73', rlxp='#CC79A7', var='#888888', green='#009E73')
SEEDCOL = {1: '#0072B2', 3: '#E69F00', 7: '#CC79A7', 8: '#D55E00',
           2: '#888888', 4: '#888888', 5: '#888888', 6: '#888888'}


def load(tag):
    fs = glob.glob(f'fit_result/sci_full_{tag}/history_iter10000_*.pkl')
    if not fs:
        return None
    h = pickle.load(open(fs[0], 'rb'))
    d = {'loss': np.asarray(h['losses_iter'], float),
         'pos': np.ravel(h['pos_residual_iter']) * 1e4,
         'dedx': np.ravel(h['dedx_mae_truth_iter'])}
    for p in ['lifetime', 'long_diff']:
        t = float(np.ravel(h[p + '_target'])[0])
        d[p] = 100 * (np.ravel(h[p + '_iter'])[1:] / t - 1)
    return d


def endpoint(tag):
    d = load(tag)
    if d is None:
        return None
    e = {}
    for k, v in d.items():
        n = len(v)
        e[k] = float(np.median(v[-max(200, n // 20):]))
    return e


# (tag, seed, marker)  -- marker encodes arm type
RUNS = [(f'DSREP{s}', s, 'o') for s in range(1, 9)]
RUNS += [(f'PASS2R{s}', s, 's') for s in (1, 3, 7, 8)]
RUNS += [(f'PASS3R{s}', s, '^') for s in (1, 3, 7, 8)]
RUNS += [('PASS4R8', 8, 'v'), ('RLXR3', 3, 'D'), ('RLXR7', 7, 'D'), ('RLXR8', 8, 'D'),
         ('RLXP38', 8, 'P'), ('GTRU8', 8, '*'), ('HOT8', 8, 'X'), ('WU3000', 8, 'X')]

pts = []
base = {}
for tag, s, m in RUNS:
    e = endpoint(tag)
    if e is None:
        continue
    if tag.startswith('DSREP'):
        base[s] = e['loss']
    pts.append((tag, s, m, e))

VARS = [('dloss', 'Δloss vs seed pass-1'), ('pos', 'position (µm)'),
        ('dedx', 'dE/dx MAE'), ('lifetime', 'lifetime err (%)'),
        ('long_diff', 'long. diff err (%)')]
for tag, s, m, e in pts:
    e['dloss'] = e['loss'] - base[s]

fig, axes = plt.subplots(4, 4, figsize=(12.5, 11.5))
for i in range(4):          # row: VARS[i+1]
    for j in range(4):      # col: VARS[j]
        ax = axes[i][j]
        if j > i:
            ax.axis('off')
            continue
        kx, lx = VARS[j]
        ky, ly = VARS[i + 1]
        xs, ys = [], []
        for tag, s, m, e in pts:
            ax.scatter(e[kx], e[ky], c=SEEDCOL[s], marker=m, s=55 if m == '*' else 32,
                       alpha=.85, lw=.4, edgecolors='white', zorder=3)
            xs.append(e[kx]); ys.append(e[ky])
        r = np.corrcoef(xs, ys)[0, 1]
        rs = np.corrcoef(np.argsort(np.argsort(xs)), np.argsort(np.argsort(ys)))[0, 1]
        ax.set_title(f'r = {r:+.2f}   (rank {rs:+.2f})', fontsize=8.5,
                     color='#B00020' if abs(rs) > .6 else '#444444')
        if i == 3:
            ax.set_xlabel(lx, fontsize=8.5)
        if j == 0:
            ax.set_ylabel(ly, fontsize=8.5)

leg1 = [plt.Line2D([], [], marker='o', ls='', color=SEEDCOL[s], label=f'seed {s}')
        for s in (1, 3, 7, 8)] + [plt.Line2D([], [], marker='o', ls='', color='#888888',
                                             label='seeds 2/4/5/6 (pass 1 only)')]
leg2 = [plt.Line2D([], [], marker=m, ls='', color='#444444', label=l) for m, l in
        [('o', 'pass 1'), ('s', 'pass 2'), ('^', 'pass 3'), ('v', 'pass 4'),
         ('D', 'gentle relax'), ('P', 'gentle from p3'), ('*', 'geom @ truth'),
         ('X', 'schedule variant')]]
fig.legend(handles=leg1 + leg2, ncol=7, frameon=False, loc='lower center',
           bbox_to_anchor=(.5, -0.015), fontsize=8)
fig.suptitle('Fig 80 — endpoint correlations across all comparable runs (n = %d). Δloss, position and dE/dx form a tight\n'
             'triangle (r = +0.80…+0.89): they are one axis, "fit quality". Lifetime correlates with NOTHING — not the loss (−0.22),\n'
             'not the nuisance errors, not even long. diffusion at endpoints (+0.18). The loss does not know where lifetime is.' % len(pts),
             fontsize=11, y=0.985)
fig.tight_layout(rect=(0, .03, 1, .96))
fig.savefig(f'{OUT}/fig80_endpoint_correlations.png')
print('wrote fig80, n =', len(pts))

# ---- FIG 81: trajectory phase planes on seed 8 ----
ARMS8 = [('DSREP8', C['p1'], 'pass 1'), ('PASS2R8', C['p2'], 'pass 2'),
         ('PASS3R8', C['p3'], 'pass 3'), ('PASS4R8', C['p4'], 'pass 4'),
         ('RLXR8', C['rlx'], 'gentle relax'), ('RLXP38', C['rlxp'], 'gentle from p3')]
W = 300


def sm(v):
    v = np.asarray(v, float)
    return np.median(sliding_window_view(v, W), axis=1) if len(v) > W else v


fig, ax = plt.subplots(1, 4, figsize=(14.5, 3.9))
PANELS = [('pos', 'position residual (µm)', (80, 420)),
          ('dedx', 'dE/dx MAE (vs matched truth)', (0.05, 0.075)),
          ('lifetime', 'lifetime error (%)', (-5, 42)),
          ('long_diff', 'long. diff. error (%)', (-10, 55))]
for a, (k, lab, yl) in zip(ax, PANELS):
    for tag, col, l in ARMS8:
        d = load(tag)
        if d is None:
            continue
        n = min(len(d['loss']), len(d[k]))
        x, y = sm(d['loss'][:n]), sm(d[k][:n])
        a.plot(x, y, color=col, lw=1.1, alpha=.9, label=l)
        a.plot(x[-1], y[-1], 'o', color=col, ms=5, zorder=4)
    a.set_xlim(8250, 10500)
    a.set_ylim(*yl)
    a.invert_xaxis()          # optimization moves left->right toward lower loss
    a.set_xlabel('loss (→ improving)')
    a.set_ylabel(lab, fontsize=8.5)
    if k in ('lifetime', 'long_diff'):
        a.axhline(0, color=C['green'], lw=1.2)
ax[0].legend(frameon=False, fontsize=7.5, loc='upper left')
fig.suptitle('Fig 81 — phase planes on seed 8 (all arms, dots = endpoints): position and dE/dx are single-valued functions of the loss\n'
             '(every arm rides the same curve), while lifetime and long. diffusion loop and land anywhere at the same loss — the flat valley, seen directly.',
             fontsize=10.5, y=1.06)
fig.tight_layout()
fig.savefig(f'{OUT}/fig81_phase_planes.png')
print('wrote fig81')
