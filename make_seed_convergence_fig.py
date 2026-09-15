"""Figs 79a/79b — FINAL convergence, one panel per tested data seed (replicas 1-8).

Each panel chains every arm run on that seed on a cumulative-iteration axis:
  pass 1 (DSREP, cold)  ->  pass 2  ->  pass 3  ->  pass 4      (solid, re-anneal chain)
  RLX branch from the pass-1 endpoint (gentle schedule)          (dashed)
  RLXP38 branch from rep 8's pass-3 endpoint                     (dashed)

79a: lifetime error vs truth. 79b: loss (same layout) — loss descends monotonically across
passes while lifetime scatters: the soft-manifold picture, seed by seed.
"""
import glob, pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import sliding_window_view

OUT = 'plots/noise_report'
C = dict(p1='#222222', p2='#0072B2', p3='#E69F00', p4='#D55E00',
         rlx='#009E73', rlxp='#CC79A7', green='#009E73', grey='#888888')
plt.rcParams.update({'font.size': 9, 'axes.grid': True, 'grid.alpha': .25,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.dpi': 130, 'savefig.bbox': 'tight'})
W = 200


def sm(v, w=W):
    v = np.asarray(v, float)
    return np.median(sliding_window_view(v, w), axis=1) if len(v) > w else v


def load(tag):
    fs = glob.glob(f'fit_result/sci_full_{tag}/history_iter10000_*.pkl')
    if not fs:
        return None
    h = pickle.load(open(fs[0], 'rb'))
    t = float(np.ravel(h['lifetime_target'])[0])
    return {'lt': 100 * (np.ravel(h['lifetime_iter'])[1:] / t - 1),
            'loss': np.asarray(h['losses_iter'], float)}


# per seed: the re-anneal chain (list of (tag, color, label)) and branches [(tag, start_offset_k, color, label)]
CHAINS = {s: [('DSREP%d' % s, C['p1'], 'pass 1 (cold)')] for s in range(1, 9)}
for s in (1, 3, 7, 8):
    CHAINS[s] += [('PASS2R%d' % s, C['p2'], 'pass 2'), ('PASS3R%d' % s, C['p3'], 'pass 3')]
CHAINS[8] += [('PASS4R8', C['p4'], 'pass 4')]
BRANCH = {3: [('RLXR3', 1, C['rlx'], 'gentle relax')],
          7: [('RLXR7', 1, C['rlx'], 'gentle relax')],
          8: [('RLXR8', 1, C['rlx'], 'gentle relax'),
              ('RLXP38', 3, C['rlxp'], 'gentle from pass 3')]}


def draw(key, ylab, title, fname, ylim=None):
    fig, axes = plt.subplots(2, 4, figsize=(14, 6), sharex=False)
    handles = {}
    for s, ax in zip(range(1, 9), axes.ravel()):
        off = 0
        for tag, col, lab in CHAINS[s]:
            d = load(tag)
            if d is None:
                continue
            y = sm(d[key])
            x = (off + np.linspace(0, 10, len(y)))
            ln, = ax.plot(x, y, color=col, lw=1.2)
            handles.setdefault(lab, ln)
            off += 10
        for tag, at, col, lab in BRANCH.get(s, []):
            d = load(tag)
            if d is None:
                continue
            y = sm(d[key])
            x = (10 * at + np.linspace(0, 10, len(y)))
            ln, = ax.plot(x, y, color=col, lw=1.2, ls='--')
            handles.setdefault(lab, ln)
        for b in range(10, off, 10):
            ax.axvline(b, color=C['grey'], lw=.6, ls=':')
        if key == 'lt':
            ax.axhline(0, color=C['green'], lw=1.2)
        if ylim:
            ax.set_ylim(*ylim)
        ax.set_title(f'data seed {s}', fontsize=9.5)
        if s > 4:
            ax.set_xlabel('cumulative iteration (×1000)')
        if s in (1, 5):
            ax.set_ylabel(ylab)
    fig.legend(handles.values(), handles.keys(), ncol=len(handles), frameon=False,
               loc='lower center', bbox_to_anchor=(.5, -0.04), fontsize=9)
    fig.suptitle(title, fontsize=11, y=1.005)
    fig.tight_layout()
    fig.savefig(f'{OUT}/{fname}')
    print('wrote', f'{OUT}/{fname}')


draw('lt', 'lifetime error (%)',
     'Fig 79a — final convergence of every tested data seed: lifetime. Solid = the re-anneal chain '
     '(each 10k-block restarts calibration at nominal);\ndashed = gentle-relaxation branches. '
     'Green = truth. Loss descends throughout (fig 79b) while lifetime scatters upward.',
     'fig79a_seed_convergence_lifetime.png', ylim=(-12, 45))

draw('loss', 'loss (rolling median)',
     'Fig 79b — final convergence of every tested data seed: loss, same layout as 79a. '
     'Every pass/branch lowers the loss (restart spikes aside) —\nthe objective improves '
     'monotonically across arms while lifetime (79a) does not follow.',
     'fig79b_seed_convergence_loss.png', ylim=(8100, 12000))
