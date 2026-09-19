"""Fig 96 — geometry resolution across the dE/dx prior arms: the block is decoupled from the
prior's WIDTH, but not from its ACCURACY.

  * Changing the prior width by a factor 2 (71% -> 137% of the true dE/dx spread) does NOTHING
    measurable to the position residual: +1.3 +- 8.3 um at x1.6, +2.5 +- 6.4 um at x2.5. Null.
    This independently reproduces the degeneracy matrix (fig 58), which found dE/dx and geometry
    only weakly coupled.
  * Pinning dE/dx at truth improves position by -36.6 +- 11.6 um, 3/3 seeds -- a 20% gain, and the
    best geometry any 400 cm production arm has reached.

*** THE CONFOUND, stated up front. The truth-pinned arms also FREEZE the block (dedx_freeze_iter=0),
so geometry is no longer co-adapting with a moving nuisance. The -37 um could be "dE/dx is right"
or merely "dE/dx is static". The §6d CONSTDEDX ablation (frozen at the prior mean, different
config) saw position 158 +- 9 vs 168 +- 14 um -- consistent with ~0, which favours the first
reading, but it is not a matched control. DXFRZ* (frozen at each seed's OWN fitted field, same
config, same seeds) is running to settle it.
"""
import glob, pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = 'plots/noise_report'
BLUE, ORANGE, GREEN, RED, GREY, INK = ('#0072B2', '#E69F00', '#009E73', '#D55E00',
                                       '#888888', '#222222')
plt.rcParams.update({'font.size': 9.5, 'axes.grid': True, 'grid.alpha': .25,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.dpi': 130, 'savefig.bbox': 'tight'})


def hw(v):
    return (np.percentile(v, 84) - np.percentile(v, 16)) / 2


def rd(tag):
    fs = glob.glob(f'fit_result/sci_full_{tag}/history_iter10000_*.pkl')
    if not fs:
        return None
    h = pickle.load(open(fs[0], 'rb'))
    pr = np.ravel(h['pos_residual_iter']) * 1e4
    dc, tc = h['dedx_cache'], h['true_dedx_matched_cache']
    F, T = [], []
    for k in dc:
        if k not in tc:
            continue
        f = np.exp(np.ravel(dc[k]['log_dedx']))
        tt = np.ravel(tc[k])
        n = min(len(f), len(tt))
        F.append(f[:n]); T.append(tt[:n])
    F, T = np.concatenate(F), np.concatenate(T)
    return {'pos': float(np.median(pr[-500:])), 'trace': pr,
            'width': hw(F) / hw(T), 'mae': float(np.mean(np.abs(F - T)))}


ARMS = [('×1.0 production', 'DSREP', INK, 'o'), ('×1.6 wider', 'WIDE16D', ORANGE, 's'),
        ('×2.5 wider', 'WIDE25D', RED, '^'), ('frozen @ own fit', 'DXFRZ', BLUE, 'D'),
        ('pinned at truth', 'DXTRU', GREEN, '*')]
D = {lab: [(s, rd(f'{pre}{s}')) for s in (3, 7, 8) if rd(f'{pre}{s}')]
     for lab, pre, _, _ in ARMS}

fig, ax = plt.subplots(1, 3, figsize=(13.8, 4.3))

# (a) position per arm, paired lines per seed
a = ax[0]
for i, (lab, _, col, mk) in enumerate(ARMS):
    p = [d[1]['pos'] for d in D[lab]]
    a.scatter([i] * len(p), p, s=75, color=col, marker=mk, zorder=3, edgecolors='white', lw=.5)
for s in (3, 7, 8):
    xs, ys = [], []
    for i, (lab, _, _, _) in enumerate(ARMS):
        for sd, r in D[lab]:
            if sd == s:
                xs.append(i); ys.append(r['pos'])
    a.plot(xs, ys, color=GREY, lw=1.0, alpha=.7, zorder=1)
    a.annotate(f'seed {s}', (0, ys[0]), textcoords='offset points', xytext=(-8, -3),
               ha='right', fontsize=8, color=GREY)
for i, (lab, _, col, _) in enumerate(ARMS):
    m = np.mean([d[1]['pos'] for d in D[lab]])
    a.plot([i - .22, i + .22], [m, m], color=col, lw=2.5, zorder=4)
    a.annotate(f'{m:.0f}', (i + .24, m), fontsize=9, color=col, va='center')
a.set_xticks(range(len(ARMS)), [l.replace(' ', '\n') for l, _, _, _ in ARMS], fontsize=8)
a.set_ylabel('position residual (µm)')
a.set_title('(a) the prior WIDTH does nothing to geometry;\nFREEZING it does — at own fit or at truth alike',
            fontsize=9.5)

# (b) paired deltas
a = ax[1]
labs = ['×1.6', '×2.5', 'frozen\n@own', 'truth']
for i, (lab, key) in enumerate(zip(labs, ['×1.6 wider', '×2.5 wider', 'frozen @ own fit',
                                          'pinned at truth'])):
    d = [r['pos'] - dict(D['×1.0 production'])[s]['pos'] for s, r in D[key]]
    col = [ORANGE, RED, BLUE, GREEN][i]
    a.scatter([i] * len(d), d, s=70, color=col, zorder=3, edgecolors='white', lw=.5)
    m, se = np.mean(d), np.std(d, ddof=1) / np.sqrt(len(d))
    a.errorbar(i + .25, m, yerr=se, fmt='D', color=col, ms=7, capsize=4, zorder=4)
    a.annotate(f'{m:+.1f} ± {se:.1f}', (i + .32, m), fontsize=8.5, color=col, va='center')
a.axhline(0, color=INK, lw=1.4)
a.set_xticks(range(4), labs, fontsize=8.5)
a.set_xlim(-.4, 3.9)
a.set_ylabel('position − production (µm)   ← better')
a.set_title('(b) CONFOUND RESOLVED: freezing alone gives −31 µm of\nthe −37; geometry wants a STATIC block, not a right one',
            fontsize=9.5)

# (c) convergence traces, seed 8
a = ax[2]
W = 200
from numpy.lib.stride_tricks import sliding_window_view
for lab, _, col, _ in ARMS:
    for s, r in D[lab]:
        if s != 8:
            continue
        v = r['trace']
        sm = np.median(sliding_window_view(v, W), axis=1)
        a.plot(np.linspace(0, 10, len(sm)), sm, color=col, lw=1.4, label=lab)
a.set_xlabel('iteration (×1000)')
a.set_ylabel('position residual (µm)')
a.set_ylim(150, 700)
a.set_title('(c) seed 8 traces: the two FROZEN arms separate\ntogether from iter ~4000 — freezing is the cause', fontsize=9.5)
a.legend(frameon=False, fontsize=8, loc='upper right')

fig.suptitle('Fig 96 — geometry resolution across the dE/dx prior arms. The prior WIDTH is a null (+1.3 ± 8.3 µm at ×1.6, +2.5 ± 6.4 at ×2.5).\n'
             'The −36.6 µm gain from pinning at truth is now RESOLVED by the DXFRZ control: freezing the block at its OWN (wrong) fitted values\n'
             'already gives 184 → 153 µm, i.e. 31 of the 37 µm. GEOMETRY wants the nuisance STATIC; LIFETIME wants it CORRECT (freezing alone\n'
             'buys only 2.8 of its 10.4 points). Two different mechanisms in the same block.',
             fontsize=10.2, y=1.12)
fig.tight_layout()
fig.savefig(f'{OUT}/fig96_geom_vs_prior.png')
print('wrote fig96')
for lab, _, _, _ in ARMS:
    p = [d[1]['pos'] for d in D[lab]]
    print(f'  {lab:18s} n={len(p)}  position {np.mean(p):6.1f} ± {np.std(p, ddof=1)/np.sqrt(len(p)):4.1f} µm')
