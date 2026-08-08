"""Fig 51 (study 1, per seed): A_b and the two diffusions, same layout as fig 50.

Rows are parameters, columns are seeds. Seeds keep fig 50's ordering -- by the per-seed lifetime
LEVER ARM, easiest first -- purely so the two figures can be read side by side. That ordering has
no meaning for these three parameters, and whether it turns out to have none is itself the question
this figure answers: if A_b or the diffusions also degraded on the low-lever-arm seeds, the effect
would be something about those targets in general rather than about lifetime's signal specifically.

E field is omitted: it sits at -0.00 +- 0.00% in every arm and every seed, and a panel of six flat
lines carries no information.
"""
import glob, os, pickle, re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = 'plots/noise_report'
C = dict(blue='#0072B2', orange='#E69F00', green='#009E73', red='#D55E00',
         purple='#CC79A7', grey='#666666', ink='#222222')
plt.rcParams.update({'font.size': 9, 'axes.grid': True, 'grid.alpha': .25,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.dpi': 130, 'savefig.bbox': 'tight'})
ARMS = [('S1A0', 'baseline  0.91 / const', C['ink']),
        ('S1A1', 'calib 0.80  (less travel)', C['blue']),
        ('S1A2', 'calib 0.97  (more travel)', C['green']),
        ('S1A3', 'dE/dx → 0.1×', C['orange']),
        ('S1A4', 'dE/dx → 0.01×', C['red'])]
PARS = [('Ab', 'A$_b$', 1.0), ('tran_diff', 'transverse diffusion', 4.0),
        ('long_diff', 'longitudinal diffusion', 10.0)]     # (key, label, y half-range)
T_DRIFT = 0.087 * 2200.0
W = 200


def smooth(v, w=W):
    v = np.asarray(v, float)
    if len(v) <= w:
        return v
    from numpy.lib.stride_tricks import sliding_window_view
    return np.median(sliding_window_view(v, w), axis=1)


D, TAU = {}, {}
for tag, _, _ in ARMS:
    for f in sorted(glob.glob(f'fit_result/sci_ceiling_{tag}/history_iter10000_*.pkl')):
        s = int(re.search(r'seed(\d+)', f).group(1))
        h = pickle.load(open(f, 'rb'))
        for p, _, _ in PARS:
            t = float(np.ravel(h[p + '_target'])[0])
            D[(tag, s, p)] = 100 * (np.ravel(h[p + '_iter'])[1:] / t - 1)
        TAU[s] = float(np.ravel(h['lifetime_target'])[0])
SEEDS = sorted(TAU)
SWING = {s: 100 * (1 - np.exp(-T_DRIFT / TAU[s])) for s in SEEDS}
ORDER = sorted(SEEDS, key=lambda s: -SWING[s])

fig, axes = plt.subplots(len(PARS), len(ORDER), figsize=(23.5, 10.4), sharex=True)
for r, (p, plab, yr) in enumerate(PARS):
    for c, s in enumerate(ORDER):
        a = axes[r, c]
        a.axhline(0, color=C['red'], lw=1.4, ls='--', zorder=1)
        fin = {}
        for tag, lab, col in ARMS:
            v = D[(tag, s, p)]
            a.plot(smooth(v), color=col, lw=1.5, label=lab if (r == 0 and c == 0) else None)
            fin[tag] = float(np.median(v[int(len(v) * .8):]))
        a.set_ylim(-yr, yr)
        if r == 0:
            a.set_title(f'seed {s}   τ={TAU[s]:.0f}   swing {SWING[s]:.1f}%', fontsize=9.0)
        if c == 0:
            a.set_ylabel(f'{plab}\nerror vs truth (%)', fontsize=8.6)
        if r == len(PARS) - 1:
            a.set_xlabel('iteration', fontsize=8.4)
        sp = np.ptp(list(fin.values()))
        a.text(.97, .04, f'spread across arms\n{sp:.2f} pts', transform=a.transAxes,
               ha='right', va='bottom', fontsize=6.6, color=C['grey'])
        if SWING[s] <= 6.5:
            for spn in a.spines.values():
                spn.set_edgecolor(C['red']); spn.set_linewidth(1.6)
axes[0, 0].legend(fontsize=6.8, frameon=True, framealpha=.9, loc='lower right')

fig.suptitle('Fig 51 — study 1 per seed, A$_b$ and the diffusions (same seed order as fig 50, by '
             'LIFETIME lever arm; red frames = low lever arm).\nThe schedule dependence that '
             'dominates lifetime is almost absent here: all five arms overlap, and the '
             'low-lever-arm seeds are not\nspecially bad. That is the control — the lever-arm '
             'effect is specific to lifetime, not a property of those targets in general.',
             fontsize=10.2)
fig.tight_layout(rect=[0, 0, 1, .91])
os.makedirs(OUT, exist_ok=True)
fig.savefig(f'{OUT}/fig51_study1_per_seed_allpars.png'); plt.close(fig)
print('wrote', f'{OUT}/fig51_study1_per_seed_allpars.png')

print('\nspread ACROSS ARMS at each seed (max-min of the five final values, points)')
print(f'{"seed":>5s} {"swing":>6s} ' + ''.join(f'{pl:>24s}' for _, pl, _ in PARS) + f'{"lifetime":>12s}')
for s in ORDER:
    row = ''
    for p, _, _ in PARS:
        v = [float(np.median(D[(t, s, p)][int(len(D[(t, s, p)]) * .8):])) for t, _, _ in ARMS]
        row += f'{np.ptp(v):24.2f}'
    lf = []
    for tag, _, _ in ARMS:
        f = sorted(glob.glob(f'fit_result/sci_ceiling_{tag}/history_iter10000_*seed{s}.pkl'))[0]
        h = pickle.load(open(f, 'rb'))
        t = float(np.ravel(h['lifetime_target'])[0])
        vv = 100 * (np.ravel(h['lifetime_iter'])[1:] / t - 1)
        lf.append(float(np.median(vv[int(len(vv) * .8):])))
    print(f'{s:5d} {SWING[s]:5.1f}% ' + row + f'{np.ptp(lf):12.2f}')
print('\nmean |error| by lever-arm subset (all arms pooled)')
HI = [s for s in SEEDS if SWING[s] > 6.5]; LO = [s for s in SEEDS if SWING[s] <= 6.5]
for p, plab, _ in PARS:
    h_ = np.mean([abs(np.median(D[(t, s, p)][int(len(D[(t, s, p)]) * .8):]))
                  for t, _, _ in ARMS for s in HI])
    l_ = np.mean([abs(np.median(D[(t, s, p)][int(len(D[(t, s, p)]) * .8):]))
                  for t, _, _ in ARMS for s in LO])
    print(f'  {plab:24s} high-lever {h_:6.2f}   low-lever {l_:6.2f}   ratio {l_/h_:.2f}')
