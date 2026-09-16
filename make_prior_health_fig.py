"""Fig 89 — is the dE/dx prior still hurting us? A zero-cost audit of the 16 cold replicas.

Everything here is read from checkpoints already on disk (`dedx_cache` = the fitted per-segment
field, `true_dedx_matched_cache` = its arc-length-matched truth, `aux_iter` = the exact loss
decomposition). No fitting, no new jobs.

THE THREE QUESTIONS AND THEIR ANSWERS:
  (a) Is the refit prior still over-shrinking?  Barely. It recovers 71.1% of the true per-segment
      spread, against 32% for the old w=5 prior and the 66% the likelihood itself wants -- we are
      now slightly ABOVE the data's own optimum, i.e. on the safe side.
  (b) Could the residual shrinkage explain the SPREAD of the lifetime bias?  No. The recovered
      fraction is 0.711 +- 0.002 (sd 0.010, 1.4% relative) across 16 independent samples while
      lifetime ranges -0.6 to +16.7%. A constant cannot drive a variable. r = +0.14.
  (c) Could it explain the MEAN bias?  Partly, and this is the one live question. Constancy does
      NOT rule this out -- a fixed shrinkage gives a fixed offset. Reading our 71% onto the §6f
      spread ladder (which measured the conditional lifetime optimum at f = 1, .75, .5, .25, 0)
      predicts ~+1.8 points of the +8.85%, i.e. ~20%. That is the number a dE/dx-frozen-at-truth
      fit would confirm or kill.
"""
import glob, pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = 'plots/noise_report'
BLUE, ORANGE, GREEN, RED, GREY = '#0072B2', '#E69F00', '#009E73', '#D55E00', '#888888'
plt.rcParams.update({'font.size': 9.5, 'axes.grid': True, 'grid.alpha': .25,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.dpi': 130, 'savefig.bbox': 'tight'})


def hw(v):
    return (np.percentile(v, 84) - np.percentile(v, 16)) / 2


rows = []
for s in range(1, 17):
    fs = glob.glob(f'fit_result/sci_full_DSREP{s}/history_iter10000_*.pkl')
    if not fs:
        continue
    h = pickle.load(open(fs[0], 'rb'))
    dc, tc = h['dedx_cache'], h['true_dedx_matched_cache']
    F, T = [], []
    for k in dc:
        if k not in tc:
            continue
        f = np.exp(np.ravel(dc[k]['log_dedx']))
        t = np.ravel(tc[k])
        n = min(len(f), len(t))
        F.append(f[:n]); T.append(t[:n])
    F, T = np.concatenate(F), np.concatenate(T)
    d = {'seed': s, 'recov': hw(F) / hw(T)}
    for p in ['lifetime', 'long_diff']:
        v = np.ravel(h[p + '_iter']); tt = float(np.ravel(h[p + '_target'])[0])
        d[p] = 100 * (np.median(v[int(len(v) * .9):]) / tt - 1)
    a = h['aux_iter'][-1]
    tot = sum(abs(a[k]) for k in ['log_likelihood_tick', 'no_match_penalty',
                                  'expected_total_hits', 'dedx_prior', 'dedx_mean_penalty'])
    d['share'] = 100 * abs(a['dedx_prior']) / tot
    rows.append(d)

R = np.array([r['recov'] for r in rows])
L = np.array([r['lifetime'] for r in rows])
S = np.array([r['share'] for r in rows])

fig, ax = plt.subplots(1, 3, figsize=(13.6, 4.2))

# (a) where the recovered spread sits
a = ax[0]
a.scatter(R, np.arange(len(R)), s=45, color=BLUE, zorder=3, edgecolors='white', lw=.5)
for x, lab, col in [(0.32, 'old prior (w=5)\n32%', RED), (0.66, "data's optimum\n66%", GREEN),
                    (1.00, 'truth\n100%', GREY)]:
    a.axvline(x, color=col, lw=1.8, ls='--' if x != 1 else ':')
    a.annotate(lab, (x, len(R) * .62), color=col, fontsize=8.5, ha='center',
               rotation=90, va='center')
a.set_xlim(0.25, 1.05)
a.set_yticks([])
a.set_xlabel('recovered fraction of the true per-segment spread')
a.set_title(f'(a) the refit prior works: {R.mean():.3f} ± {R.std(ddof=1)/np.sqrt(len(R)):.3f}\n'
            "slightly ABOVE the data's own optimum — the safe side", fontsize=9.5)

# (b) does it predict the lifetime bias?
a = ax[1]
a.scatter(R, L, s=55, color=BLUE, zorder=3, edgecolors='white', lw=.5)
a.axhline(0, color=GREEN, lw=1.5)
r = np.corrcoef(R, L)[0, 1]
a.set_xlabel('recovered spread fraction')
a.set_ylabel('lifetime error (%)')
a.set_title(f'(b) but it cannot explain the SPREAD of the bias:\n'
            f'r = {r:+.2f} — the prior state is the same in every fit', fontsize=9.5)
a.annotate(f'recovered spread varies by only\n±{100*R.std(ddof=1)/R.mean():.1f}% (sd) across replicas,\n'
           f'while lifetime spans {L.min():+.1f} … {L.max():+.1f}%',
           (.03, .80), xycoords='axes fraction', fontsize=8.5, color='#444444')

# (c) the ladder reading — the one live question
a = ax[2]
f_lad = np.array([1.0, 0.75, 0.50, 0.25, 0.0])
lt_lad = np.array([2.2, 3.8, 5.4, 6.8, 7.0])
a.plot(f_lad, lt_lad, 'o-', color=ORANGE, lw=1.8, ms=6, label='§6f spread ladder\n(conditional optimum)')
pred = np.interp(R.mean(), f_lad[::-1], lt_lad[::-1])
a.axvline(R.mean(), color=BLUE, lw=1.8)
a.plot([R.mean()], [pred], '*', color=BLUE, ms=17, zorder=4)
a.annotate(f'we are here: f = {R.mean():.2f}\n→ predicts {pred:.1f}%, i.e. ~{pred-lt_lad[0]:.1f} pts\nof the observed +{L.mean():.1f}% (~{100*(pred-lt_lad[0])/L.mean():.0f}%)',
           (.06, .58), xycoords='axes fraction', fontsize=8.5, color=BLUE)
a.axhline(L.mean(), color=RED, lw=1.6, ls='--')
a.annotate(f'observed ensemble bias +{L.mean():.1f}%', (.97, L.mean()),
           xycoords=('axes fraction', 'data'), ha='right', va='bottom',
           fontsize=8.5, color=RED)
a.set_xlabel('dE/dx spread retained (f)')
a.set_ylabel('lifetime error (%)')
a.set_ylim(0, 10.5)
a.set_title('(c) the one live question: a CONSTANT shrinkage\nstill gives a constant offset', fontsize=9.5)
a.legend(frameon=False, fontsize=8, loc='lower left')

fig.suptitle('Fig 89 — dE/dx prior health check on the 16 cold replicas, from checkpoints already on disk (no new jobs).\n'
             'The refit prior is doing its job (71% of true spread recovered vs 32% before) and cannot explain the bias SPREAD;\n'
             'what it can still plausibly own is ~20% of the MEAN offset — one frozen-dE/dx fit would settle it.',
             fontsize=10.5, y=1.07)
fig.tight_layout()
fig.savefig(f'{OUT}/fig89_prior_health.png')
print(f'wrote fig89 | recovered {R.mean():.4f} ± {R.std(ddof=1)/np.sqrt(len(R)):.4f} '
      f'| prior share of |loss| {S.mean():.1f}% | r(recov,lifetime) {r:+.2f} | ladder predicts {pred:.2f}%')
