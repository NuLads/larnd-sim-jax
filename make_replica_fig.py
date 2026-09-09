"""Fig 73 — the data-replica ensemble: the bias question answered, and the answer cuts both ways.

THE STUDY. Truth PINNED at the seed-0 draw; ONLY the track sample varies (data_seed 1-8, 600
random events each from the 93,533-event pool, 0.33% pairwise overlap, verified before launch).
8 independent BT2 fits. Enabled by fixing a silent IO defect: the load-time pre-trim ran before
event selection, so data_seed had been a NO-OP for the entire campaign -- every one of its ~120
fits shares ONE physical track sample (the file's first ~40.8k cm; topology checked ordinary).

WHAT THE ENSEMBLE SEPARATES. mean - truth = intrinsic bias of the procedure at this truth;
spread = the data fluctuation; and the campaign's fixed sample can be located within the replica
distribution.

THE VERDICT, parameter by parameter:
  A_b        +0.061 +- 0.042  (1.5 sigma)  -> NULL. The campaign's celebrated "6/6-positive
             +0.27%" A_b bias does NOT reproduce on random samples: the fixed sample sits 3.5
             sigma off the replica distribution. THE A_B BIAS WAS THE SAMPLE.
  E field    -0.036 +- 0.006  (5.9 sigma)  -> real, tiny, matches the fixed sample. Intrinsic.
  lifetime   +8.57 +- 1.60    (5.4 sigma, 8/8 positive)  -> a LARGE intrinsic bias at this truth
             -- while the fixed sample gave -3.0% at the SAME truth, a 2.6-sigma low outlier
             under the replica distribution (p ~ 1% two-sided). The campaign's small lifetime
             errors were substantially the luck of one sample.
  long.diff  +14.0 +- 3.4     (4.1 sigma, 8/8 positive)  -> large intrinsic bias, direction
             consistent with the campaign's +7.4 mean.
  tran.diff  +1.09 +- 0.49    (2.2 sigma)  -> suggestive positive bias, below decisive.

AND THE SPREADS INDICT THE ERROR BUDGET: lifetime's sample-to-sample spread is 4.5%, x4.5 the
dEdx-marginalised within-sample floor. The floor is the per-sample statistical limit; the spread
includes how the bias itself moves with sample composition (drift-depth make-up, i.e. the lever
arm). Fig 72's "lifetime is nearly statistics-limited" survives only in the narrow per-sample
sense: ACROSS samples, composition-driven bias variation dominates the budget.

CAVEATS, honestly: one truth point (Arm B at a second truth is the confirming step); and the fixed
sample being 2.6-sigma low is uncomfortable enough that a further systematic difference between
"first events" and "random events" cannot be fully excluded -- though the obvious one (topology)
was checked and refuted: first events are ordinary 50-280 cm muons, median event length 99 cm
file-wide.
"""
import glob, os, pickle, re, io, contextlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
with contextlib.redirect_stdout(io.StringIO()):
    import robust_convergence as R

OUT = 'plots/noise_report'
C = dict(blue='#0072B2', orange='#E69F00', green='#009E73', red='#D55E00',
         purple='#CC79A7', grey='#888888', ink='#222222')
plt.rcParams.update({'font.size': 9, 'axes.grid': True, 'grid.alpha': .25,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.dpi': 130, 'savefig.bbox': 'tight'})
P = R.P
PL = {'Ab': 'A$_b$', 'eField': 'E field', 'lifetime': 'lifetime',
      'tran_diff': 'tran. diff.', 'long_diff': 'long. diff.'}
FLOOR = {'Ab': 0.048, 'eField': 0.0005, 'tran_diff': 0.45, 'long_diff': 0.9, 'lifetime': 1.0}
W = 200


def sm(v, w=W):
    v = np.asarray(v, float)
    from numpy.lib.stride_tricks import sliding_window_view
    return np.median(sliding_window_view(v, w), axis=1) if len(v) > w else v


def read(path):
    h = pickle.load(open(path, 'rb'))
    d = {}
    for p in P:
        t = float(np.ravel(h[p + '_target'])[0])
        d[p] = 100 * (np.median(sm(np.ravel(h[p + '_iter'])[1:])[-300:]) / t - 1)
    return d


REP = {ds: read(glob.glob(f'fit_result/sci_full_DSREP{ds}/history_iter10000_*.pkl')[0])
       for ds in range(1, 9)}
OLD = read(glob.glob('fit_result/sci_full_BT2/history_iter10000_*seed0.pkl')[0])

fig, ax = plt.subplots(1, 3, figsize=(16.6, 5.6))

# ---------------- (a) all parameters ----------------
a = ax[0]
for j, p in enumerate(P):
    v = np.array([REP[d][p] for d in range(1, 9)])
    m, se = v.mean(), v.std(ddof=1) / np.sqrt(8)
    for x in v:
        a.plot(x, j, 'o', ms=7, color=C['blue'], alpha=.55)
    a.errorbar(m, j, xerr=se, fmt='s', ms=11, color=C['ink'], capsize=5, zorder=5)
    a.plot(OLD[p], j, '*', ms=17, color=C['red'], zorder=6)
    sg = abs(m) / se
    a.text(1.01, j, f'{m:+.2f}±{se:.2f} ({sg:.1f}σ)', transform=a.get_yaxis_transform(),
           fontsize=7.6, va='center')
a.axvline(0, color=C['green'], lw=2)
a.set_yticks(range(5)); a.set_yticklabels([PL[p] for p in P], fontsize=9); a.invert_yaxis()
a.set_xscale('symlog', linthresh=1)
a.set_xlabel('signed error at the PINNED truth (%)')
a.set_title('(a) 8 replicas (blue), ensemble mean±SE (black), and the\n'
            'campaign\'s fixed sample (red star). Truth = green line.', fontsize=9.3)

# ---------------- (b) lifetime detail ----------------
a = ax[1]
v = np.array([REP[d]['lifetime'] for d in range(1, 9)])
a.hist(v, bins=np.arange(-6, 20, 2), color=C['blue'], alpha=.6, edgecolor='white')
a.axvline(v.mean(), color=C['ink'], lw=2.4)
a.annotate(f'  ensemble {v.mean():+.1f}±{v.std(ddof=1)/np.sqrt(8):.1f}%\n  (8/8 positive)',
           (v.mean(), 2.6), fontsize=8.4)
a.axvline(OLD['lifetime'], color=C['red'], lw=2.4, ls='--')
a.annotate(f'the campaign\'s sample\n{OLD["lifetime"]:+.1f}% — a 2.6σ low\noutlier (p ≈ 1%)',
           (OLD['lifetime'] - .5, 1.8), fontsize=8.0, color=C['red'], ha='right')
a.axvline(0, color=C['green'], lw=2)
a.annotate(' truth', (0, 2.95), fontsize=8, color=C['green'])
a.set_xlabel('lifetime signed error (%)')
a.set_ylabel('replicas')
a.set_title('(b) LIFETIME: a +8.6% intrinsic bias on random samples —\n'
            'the campaign\'s small errors were substantially one sample\'s luck.', fontsize=9.3)

# ---------------- (c) spread vs the floors ----------------
a = ax[2]
xs = np.arange(5)
sp = [np.array([REP[d][p] for d in range(1, 9)]).std(ddof=1) for p in P]
fl = [FLOOR[p] for p in P]
a.bar(xs - .19, fl, width=.38, color=C['green'], edgecolor='white',
      label='within-sample floor (dE/dx-marg, fig 72)')
a.bar(xs + .19, sp, width=.38, color=C['orange'], edgecolor='white',
      label='sample-to-sample spread (this study)')
for i in range(5):
    a.text(i + .19, sp[i] * 1.15, f'×{sp[i]/fl[i]:.1f}', ha='center', fontsize=8.2)
a.set_yscale('log')
a.set_xticks(xs); a.set_xticklabels([PL[p] for p in P], fontsize=8, rotation=18, ha='right')
a.set_ylabel('% of the fitted value')
a.set_title('(c) the spread EXCEEDS the per-sample floor ×2.5–35:\n'
            'sample-composition bias variation dominates the error budget.', fontsize=9.3)
a.legend(fontsize=7.2, frameon=False)

fig.suptitle('Fig 73 — the data-replica ensemble (truth pinned, 8 independent track samples; '
             'enabled by fixing data_seed, a silent no-op campaign-wide).\nThe A$_b$ "bias" was '
             'the sample (null at +0.06 ± 0.04); lifetime carries a REAL +8.6 ± 1.6% bias at this '
             'truth on random samples, and the campaign\'s\nfixed sample sits at the 2.6σ low tail '
             '— its small lifetime errors were substantially luck. Sample-composition variation, '
             'not per-sample statistics,\nnow dominates the error budget.', fontsize=10.0)
fig.tight_layout(rect=[0, 0, 1, .84])
os.makedirs(OUT, exist_ok=True)
fig.savefig(f'{OUT}/fig73_replica_ensemble.png'); plt.close(fig)
print('wrote', f'{OUT}/fig73_replica_ensemble.png')
