"""Fig 101 — in-depth review: what actually improves the fits, across every convergence metric.

Panel (a) is the one that matters. The x0.40 prior optimum was FOUND on seeds 3/7/8 and reported
as 7.26% -> 2.95% mean |error|. Run on five seeds it was NOT tuned on, the gain is
-0.34 +- 0.65 points (3/5). The original number was largely REGRESSION TO THE MEAN: seeds 3/7/8
happen to have unusually bad baselines (7.26% vs 3.42% for the fresh five), so almost any
intervention improves them.

Panels (b)-(d) show the full metric set across the constraint ladder, which is what makes the
mechanism legible: constraining the dE/dx block trades the dE/dx field's OWN accuracy (dxMAE, which
is minimised at production) against the calibration's (minimised near x0.40), while the data
likelihood monotonically WORSENS -- the objective disagrees with every physics metric.
"""
import glob, pickle, warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

OUT = 'plots/noise_report'
BLUE, ORANGE, GREEN, RED, GREY, INK, PURPLE = ('#0072B2', '#E69F00', '#009E73', '#D55E00',
                                               '#888888', '#222222', '#CC79A7')
plt.rcParams.update({'font.size': 9.5, 'axes.grid': True, 'grid.alpha': .25,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.dpi': 130, 'savefig.bbox': 'tight'})
P = ['Ab', 'eField', 'tran_diff', 'long_diff', 'lifetime']
hw = lambda v: (np.percentile(v, 84) - np.percentile(v, 16)) / 2


def rd(tag):
    g = []
    for b in ('fit_result', '/sdf/scratch/neutrino/pgranger/fit_result'):
        g = glob.glob(f'{b}/sci_full_{tag}/history_iter10000_*.pkl')
        if g:
            break
    if not g:
        return None
    h = pickle.load(open(g[0], 'rb'))
    o = {}
    for p in P:
        v = np.ravel(h[p + '_iter'])
        t = float(np.ravel(h[p + '_target'])[0])
        o[p] = 100 * (np.median(v[int(len(v) * .9):]) / t - 1)
    o['pos'] = float(np.median(np.ravel(h['pos_residual_iter'])[-500:] * 1e4))
    o['dedx'] = float(np.median(np.ravel(h['dedx_mae_truth_iter'])[-500:]))
    A = h['aux_iter'][-2000:]
    o['lik'] = float(np.median([abs(a['log_likelihood_tick']) + abs(a['no_match_penalty'])
                                + abs(a['expected_total_hits']) for a in A]))
    o['me'] = np.mean([abs(o[p]) for p in P])
    return o


fig, ax = plt.subplots(1, 4, figsize=(17.5, 4.4))

# (a) tuned vs fresh — the headline correction
a = ax[0]
TUNED, FRESH = (3, 7, 8), (1, 2, 4, 5, 6)
for grp, col, lab, off in ((TUNED, RED, 'seeds it was TUNED on', -0.12),
                           (FRESH, BLUE, 'FRESH seeds', 0.12)):
    b = [rd(f'DSREP{s}')['me'] for s in grp]
    x = [rd(f'NARR040D{s}')['me'] for s in grp]
    for i in range(len(b)):
        a.plot([0 + off, 1 + off], [b[i], x[i]], '-', color=col, lw=1, alpha=.45)
    a.plot([0 + off, 1 + off], [np.mean(b), np.mean(x)], 'o-', color=col, lw=2.8, ms=10,
           label=f'{lab}  ({np.mean(x)-np.mean(b):+.2f})')
a.set_xticks([0, 1], ['production\n(×1.0)', 'the recipe\n(×0.40)'], fontsize=9)
a.set_xlim(-.45, 1.45)
a.set_ylabel('mean |error| over the 5 calibration params (%)')
a.set_title('(a) ⚠ THE HEADLINE LARGELY EVAPORATES.\n−4.31 ± 1.48 on tuned seeds, −0.34 ± 0.65 on fresh',
            fontsize=9.5)
a.legend(frameon=False, fontsize=8.5, loc='upper right')

# (b) the constraint ladder, all calibration params
LAD = [('×2.5', 'WIDE25D'), ('×1.6', 'WIDE16D'), ('×1.0', 'DSREP'), ('×0.60', 'NARR06D'),
       ('×0.40', 'NARR040D'), ('×0.25', 'NARR025D'), ('×0.15', 'NARR015D')]
X = np.arange(len(LAD))
D = {lab: [rd(f'{pre}{s}') for s in (3, 7, 8)] for lab, pre in LAD}
a = ax[1]
for p, col, lw in (('lifetime', BLUE, 2.2), ('long_diff', RED, 2.2),
                   ('tran_diff', PURPLE, 1.2), ('Ab', GREY, 1.0), ('eField', GREY, 1.0)):
    a.plot(X, [np.mean([r[p] for r in D[l]]) for l, _ in LAD], 'o-', color=col, lw=lw, ms=5,
           label=p, alpha=1 if lw > 2 else .7)
a.axhline(0, color=GREEN, lw=1.8)
a.set_xticks(X, [l for l, _ in LAD], fontsize=8)
a.set_ylabel('error (%)')
a.set_title('(b) the constraint ladder: only the soft plane moves.\nA$_b$/E field flat throughout',
            fontsize=9.5)
a.legend(frameon=False, fontsize=7.5, ncol=2)

# (c) the three-way trade
a = ax[2]
a.plot(X, [np.mean([r['dedx'] for r in D[l]]) for l, _ in LAD], 'o-', color=ORANGE, lw=2, ms=6)
a.set_ylabel('dE/dx MAE (own accuracy)', color=ORANGE)
a.tick_params(axis='y', labelcolor=ORANGE)
a.set_xticks(X, [l for l, _ in LAD], fontsize=8)
b_ = a.twinx()
b_.plot(X, [np.mean([r['me'] for r in D[l]]) for l, _ in LAD], 's--', color=BLUE, lw=2, ms=6)
b_.set_ylabel('calibration mean |error| (%)', color=BLUE)
b_.tick_params(axis='y', labelcolor=BLUE)
b_.grid(False)
a.annotate('dE/dx best\nhere', (2, np.mean([r['dedx'] for r in D['×1.0']])), fontsize=8,
           color=ORANGE, textcoords='offset points', xytext=(-30, -18))
a.annotate('calibration\nbest here', (4, np.mean([r['dedx'] for r in D['×0.40']])), fontsize=8,
           color=BLUE, textcoords='offset points', xytext=(6, 14))
a.set_title('(c) the trade: the dE/dx field is most accurate at ×1.0,\nthe CALIBRATION near ×0.40',
            fontsize=9.5)

# (d) the objective disagrees with everything
a = ax[3]
a.plot(X, [np.mean([r['lik'] for r in D[l]]) for l, _ in LAD], 'o-', color=RED, lw=2.2, ms=6)
a.set_ylabel('data likelihood (lower = better fit)', color=RED)
a.tick_params(axis='y', labelcolor=RED)
a.set_xticks(X, [l for l, _ in LAD], fontsize=8)
b2 = a.twinx()
b2.plot(X, [np.mean([r['pos'] for r in D[l]]) for l, _ in LAD], 's--', color=GREEN, lw=2, ms=6)
b2.set_ylabel('position residual (µm)', color=GREEN)
b2.tick_params(axis='y', labelcolor=GREEN)
b2.grid(False)
a.set_title('(d) the objective gets steadily WORSE as the physics\nimproves — loss-blindness, again',
            fontsize=9.5)

fig.suptitle('Fig 101 — in-depth review. (a) The ×0.40 recipe was validated on the seeds that produced it; on five fresh seeds the gain is −0.34 ± 0.65\n'
             'points (3/5), not −4.31. Seeds 3/7/8 simply had bad baselines (7.26% vs 3.42%), so the original number was mostly regression to the mean.\n'
             '(b)–(d) What the ladder does do is real but narrow: it moves ONLY the soft plane, trading the dE/dx field’s own accuracy for the calibration’s,\n'
             'while the data likelihood worsens monotonically — every physics gain here is invisible to, and anti-correlated with, the objective.',
             fontsize=10.2, y=1.12)
fig.tight_layout()
fig.savefig(f'{OUT}/fig101_review.png')
print('wrote fig101')
