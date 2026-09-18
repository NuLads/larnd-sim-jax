"""Fig 95 — the wider-prior fix FAILS, and the failure identifies the real mechanism.

THE TEST (fig 94's deployable prediction): the fitted dE/dx field is shrunk to 71% of the true
width by MAP shrinkage; widening the prior scale x1.6 should restore it to ~100% without using
truth, and -- if width is what lifetime cares about -- recover DXTRU's -10.4 points.

THE WIDTH KNOB WORKED EXACTLY AS PREDICTED: 71% -> 109% at x1.6, 137% at x2.5 (predicted 99% and
130%). The PHYSICS WENT THE WRONG WAY: lifetime +6.4 +- 8.9 points WORSE, 1/3 improved.

WHY. Widening the prior restores the width by letting in NOISE, not signal: the per-segment error
rises 0.058 -> 0.071 -> 0.088 monotonically with the width.

*** A CORRECTION TO THE FIRST VERSION OF THIS FIGURE. It claimed "lifetime tracks the ERROR, not
the width", citing r = +0.67 over all arms. That correlation is driven by the truth-pinned arms,
whose MAE is 0 BY DEFINITION (they ARE the truth). Among the arms that actually fit, r = +0.29
(n = 8) -- suggestive, not established.

AND THE WIDTH IS NOT WHAT LOWERS THE ERROR, which was the implied reading. Among fitted arms
NARROWER is MORE accurate: 71% width -> MAE 0.058; 109% -> 0.071; 137% -> 0.088. Production already
sits at the MSE optimum (lam 0.372 vs the Bayes-optimal 0.369) -- because the BT2 refit matched the
prior to the TRUE spread, and matching the prior to the truth IS the Bayes-optimal choice. So MSE
rises in BOTH directions from where we are, and no prior knob can lower it.

WHAT IS SOLID: "aim for 100% width" is refuted (base, furthest from 100% at 71%, has the BEST
lifetime of the fitted arms; |width-100%| correlates with lifetime at r = -0.03). The discriminating
test -- a NARROWER prior (x0.6), which also raises MSE -- is running as NARR06D*. If lifetime
worsens there too, the MAE reading survives; if it improves, MAE is not the driver either.
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
    v = np.ravel(h['lifetime_iter'])
    t = float(np.ravel(h['lifetime_target'])[0])
    lt = 100 * (np.median(v[int(len(v) * .9):]) / t - 1)
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
    return lt, float(np.mean(np.abs(F - T))), hw(F) / hw(T)


ARMS = [('base', 'DSREP', INK, 'o'), ('x1.6', 'WIDE16D', ORANGE, 's'),
        ('x2.5', 'WIDE25D', RED, '^'), ('truth', 'DXTRU', GREEN, '*')]
data = {}
for lab, pre, col, mk in ARMS:
    data[lab] = [(s,) + rd(f'{pre}{s}') for s in (3, 7, 8) if rd(f'{pre}{s}')]

fig, ax = plt.subplots(1, 3, figsize=(13.8, 4.3))

# (a) the knob did what it was told
a = ax[0]
for i, (lab, _, col, mk) in enumerate(ARMS):
    w = [100 * d[3] for d in data[lab]]
    a.scatter([i] * len(w), w, s=70, color=col, marker=mk, zorder=3, edgecolors='white', lw=.5)
    a.annotate(f'{np.mean(w):.0f}%', (i, np.mean(w)), textcoords='offset points',
               xytext=(11, -3), fontsize=9, color=col)
a.axhline(100, color=GREEN, lw=1.8)
a.annotate('true width', (.02, 101), xycoords=('axes fraction', 'data'), color=GREEN, fontsize=8.5)
a.set_xticks(range(4), ['production\n(×1.0)', 'wider\n(×1.6)', 'wider\n(×2.5)', 'pinned at\ntruth'],
             fontsize=8.5)
a.set_ylabel('fitted dE/dx width (% of truth)')
a.set_title('(a) the width knob worked exactly as predicted\n(predicted 61 / 99 / 130%)', fontsize=9.5)

# (b) but the physics went the wrong way
a = ax[1]
for lab, _, col, mk in ARMS:
    lt = [d[1] for d in data[lab]]
    a.scatter([{'base': 0, 'x1.6': 1, 'x2.5': 2, 'truth': 3}[lab]] * len(lt), lt,
              s=70, color=col, marker=mk, zorder=3, edgecolors='white', lw=.5)
for s in (3, 7, 8):
    xs, ys = [], []
    for i, (lab, _, _, _) in enumerate(ARMS):
        for d in data[lab]:
            if d[0] == s:
                xs.append(i); ys.append(d[1])
    a.plot(xs, ys, color=GREY, lw=.9, alpha=.7, zorder=1)
a.axhline(0, color=GREEN, lw=1.8)
a.set_xticks(range(4), ['×1.0', '×1.6', '×2.5', 'truth'], fontsize=9)
a.set_ylabel('lifetime error (%)')
a.set_title('(b) …and lifetime got WORSE, not better\n(+6.4 ± 8.9 at ×1.6, 1/3 improved)', fontsize=9.5)

# (c) the mechanism: lifetime tracks ACCURACY, not width
a = ax[2]
for lab, _, col, mk in ARMS:
    m = [d[2] for d in data[lab]]
    lt = [d[1] for d in data[lab]]
    a.scatter(m, lt, s=80, color=col, marker=mk, zorder=3, edgecolors='white', lw=.5,
              label=f'{lab}  (MAE {np.mean(m):.3f})')
allm = np.array([d[2] for lab in data for d in data[lab]])
alllt = np.array([d[1] for lab in data for d in data[lab]])
b1, b0 = np.polyfit(allm, alllt, 1)
xx = np.linspace(0, .095, 10)
a.plot(xx, b1 * xx + b0, '--', color=INK, lw=1.4,
       label=f'r = {np.corrcoef(allm, alllt)[0,1]:+.2f}')
a.axhline(0, color=GREEN, lw=1.6)
a.set_xlabel('per-segment dE/dx error (MAE vs matched truth)')
a.set_ylabel('lifetime error (%)')
a.set_title('(c) ⚠ the truth points (MAE = 0 BY DEFINITION) drive this.\n'
            'Among FITTED arms only: r = +0.29 — suggestive, not established', fontsize=9.5)
a.legend(frameon=False, fontsize=8, loc='upper left')

fig.suptitle('Fig 95 — widening the prior restores the dE/dx WIDTH and makes the calibration WORSE: "aim for 100% width" is refuted.\n'
             'NOTE the width is NOT what lowers the per-segment error — among fitted arms NARROWER is more accurate (0.058 at 71% vs 0.088 at 137%),\n'
             'because production already sits at the MSE optimum. The truth arm\'s MAE = 0 is definitional, so panel (c) overstates the MAE link (r = +0.29 without it).',
             fontsize=10.5, y=1.09)
fig.tight_layout()
fig.savefig(f'{OUT}/fig95_wide_prior.png')
print('wrote fig95')
for lab in ('base', 'x1.6', 'x2.5', 'truth'):
    d = data[lab]
    print(f'  {lab:6s} n={len(d)}  width {100*np.mean([x[3] for x in d]):5.0f}%  '
          f'MAE {np.mean([x[2] for x in d]):.4f}  lifetime {np.mean([x[1] for x in d]):+7.2f}')
