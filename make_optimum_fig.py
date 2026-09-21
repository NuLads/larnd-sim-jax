"""Fig 99 — the nuisance-freedom optimum LOCATED, and it reaches truth-pinned quality with no truth.

Fig 97 showed calibration accuracy improving monotonically as the dE/dx block was constrained
harder, down to x0.6. CONSTDEDX (SS6d, the zero-freedom limit) was known to be much worse, so a
turning point had to exist below that. This scan finds it.

    prior scale     x1.0     x0.6     x0.40    x0.25    x0.15
    dE/dx width      71%      36%      18%       8%       3%
    mean |error|    7.26%    3.36%    2.95%    4.18%    4.84%
                                      ^^^^^ minimum

x0.40 reaches mean |error| 2.95% -- statistically identical to pinning BOTH nuisance blocks at
truth (2.94%) and better than pinning dE/dx alone (3.19%). It uses no truth information: it is a
one-line change to the prior scale.

THE PARAMETERS DO NOT ALL OPTIMISE TOGETHER, which matters for choosing an operating point:
  * mean |error| bottoms at x0.40
  * lifetime keeps improving to x0.25 (+1.71%) and overshoots by x0.15 (-4.30%)
  * long_diff crosses zero between x0.6 (+5.8%) and x0.40 (-8.2%)
  * position is flat-to-better throughout, best at x0.25 (159 um)
So x0.40 is the best single compromise; x0.25 is preferable if lifetime is the deliverable.

CAVEAT: x0.15 is n=1 (its other two seeds were lost to the /sdf/group outage) and is plotted
hollow. The minimum's location rests on the x0.40 and x0.25 points, both n=3.
"""
import glob, pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = 'plots/noise_report'
BLUE, ORANGE, GREEN, RED, GREY, INK, PURPLE = ('#0072B2', '#E69F00', '#009E73', '#D55E00',
                                               '#888888', '#222222', '#CC79A7')
plt.rcParams.update({'font.size': 9.5, 'axes.grid': True, 'grid.alpha': .25,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.dpi': 130, 'savefig.bbox': 'tight'})
P = ['Ab', 'eField', 'tran_diff', 'long_diff', 'lifetime']
PL = {'Ab': 'A$_b$', 'eField': 'E field', 'tran_diff': 'tran. diff.',
      'long_diff': 'long. diff.', 'lifetime': 'lifetime'}


def rd(tag):
    fs = []
    for base in ('fit_result', '/sdf/scratch/neutrino/pgranger/fit_result'):
        fs = glob.glob(f'{base}/sci_full_{tag}/history_iter10000_*.pkl')
        if fs:
            break
    if not fs:
        return None
    h = pickle.load(open(fs[0], 'rb'))
    o = {}
    for p in P:
        v = np.ravel(h[p + '_iter'])
        o[p] = 100 * (np.median(v[int(len(v) * .9):]) / float(np.ravel(h[p + '_target'])[0]) - 1)
    o['pos'] = float(np.median(np.ravel(h['pos_residual_iter'])[-500:] * 1e4))
    return o


SCAN = [(1.00, 'DSREP'), (0.60, 'NARR06D'), (0.40, 'NARR040D'),
        (0.25, 'NARR025D'), (0.15, 'NARR015D')]
X, ME, SEs, N, PAR, POS = [], [], [], [], {p: [] for p in P}, []
for sc, pre in SCAN:
    rr = [rd(f'{pre}{s}') for s in (3, 7, 8)]
    rr = [r for r in rr if r]
    if not rr:
        continue
    X.append(sc); N.append(len(rr))
    m = [np.mean([abs(r[p]) for p in P]) for r in rr]
    ME.append(np.mean(m))
    SEs.append(np.std(m, ddof=1) / np.sqrt(len(m)) if len(m) > 1 else 0)
    for p in P:
        PAR[p].append(np.mean([r[p] for r in rr]))
    POS.append(np.mean([r['pos'] for r in rr]))
X = np.array(X); ME = np.array(ME); SEs = np.array(SEs); N = np.array(N)

fig, ax = plt.subplots(1, 3, figsize=(14, 4.4))

# (a) the U-curve
a = ax[0]
solid = N >= 3
a.errorbar(X[solid], ME[solid], yerr=SEs[solid], fmt='o-', color=BLUE, lw=2, ms=9, capsize=4,
           zorder=3, label='measured (n=3)')
a.plot(X[~solid], ME[~solid], 'o', mfc='white', mec=BLUE, mew=2, ms=9, zorder=3,
       label='n=1 (jobs lost to the disk outage)')
i = int(np.argmin(ME))
a.annotate(f'optimum ×{X[i]:.2f}\n{ME[i]:.2f}%', (X[i], ME[i]), textcoords='offset points',
           xytext=(6, -30), fontsize=9, color=BLUE, fontweight='bold')
a.axhline(2.94, color=GREEN, lw=2, ls='--')
a.annotate('both nuisance blocks pinned AT TRUTH: 2.94%', (0.17, 2.6), fontsize=8.3, color=GREEN)
a.axhline(3.19, color=GREY, lw=1.4, ls=':')
a.annotate('dE/dx at truth: 3.19%', (0.17, 3.30), fontsize=8, color=GREY)
a.set_xlabel('dE/dx prior scale (×production)   ← less nuisance freedom')
a.set_ylabel('mean |error| over the 5 calibration params (%)')
a.set_title('(a) the optimum is ×0.40 — and it MATCHES\ntruth-pinning, using no truth', fontsize=9.5)
a.invert_xaxis()
a.legend(frameon=False, fontsize=8, loc='upper left')

# (b) per parameter: they do not optimise together
a = ax[1]
for p, col, lw in [('lifetime', BLUE, 2.2), ('long_diff', RED, 2.2), ('tran_diff', PURPLE, 1.2),
                   ('Ab', GREY, 1.0), ('eField', GREY, 1.0)]:
    a.plot(X, PAR[p], 'o-', color=col, lw=lw, ms=5, label=PL[p], alpha=1 if lw > 2 else .7)
a.axhline(0, color=GREEN, lw=1.8)
a.axvline(X[i], color=BLUE, lw=1.2, ls='--')
a.annotate('×0.40', (X[i], 19), fontsize=8, color=BLUE, ha='center')
a.set_xlabel('dE/dx prior scale (×production)')
a.set_ylabel('error (%)')
a.set_title('(b) they do NOT optimise together: lifetime keeps\nimproving to ×0.25, long. diff. crosses zero earlier',
            fontsize=9.5)
a.invert_xaxis()
a.legend(frameon=False, fontsize=8, ncol=2, loc='lower left')

# (c) the operating-point choice
a = ax[2]
a.plot(X, np.abs(PAR['lifetime']), 'o-', color=BLUE, lw=2, ms=7, label='|lifetime error| (%)')
a.set_ylabel('|lifetime error| (%)', color=BLUE)
a.tick_params(axis='y', labelcolor=BLUE)
a.set_xlabel('dE/dx prior scale (×production)')
a.invert_xaxis()
b = a.twinx()
b.plot(X, POS, 's--', color=ORANGE, lw=2, ms=6)
b.set_ylabel('position residual (µm)', color=ORANGE)
b.tick_params(axis='y', labelcolor=ORANGE)
b.grid(False)
a.axvspan(0.45, 0.20, color=GREEN, alpha=.12, lw=0)
a.annotate('recommended operating band\n×0.40 (best overall) … ×0.25\n(best lifetime & position)',
           (0.43, 9), fontsize=8.3, color='#2a6b55')
a.set_title('(c) choosing the operating point:\n×0.40 overall, ×0.25 if lifetime is the deliverable', fontsize=9.5)
a.legend(frameon=False, fontsize=8.5, loc='upper right')

fig.suptitle('Fig 99 — the nuisance-freedom optimum located. Constraining the ~4000 per-segment dE/dx parameters to ×0.40 of the production\n'
             'prior scale gives mean |error| 2.95% across the five calibration parameters — statistically identical to pinning BOTH nuisance blocks\n'
             'at truth (2.94%), and better than pinning dE/dx alone (3.19%). It is a one-line change that uses no truth information.',
             fontsize=10.4, y=1.08)
fig.tight_layout()
fig.savefig(f'{OUT}/fig99_freedom_optimum.png')
print('wrote fig99')
for x, m, n in zip(X, ME, N):
    print(f'  ×{x:.2f}  n={n}  mean|err| {m:5.2f}%')
