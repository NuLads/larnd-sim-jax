"""Fig 61 — G20K PRELIMINARY: 20 000 iterations with both schedules stretched x2. INCOMPLETE RUN.

READ THE STATUS LINE FIRST. These are traces at ~15 000 of 20 000 iterations, seeds 0-2 only. The
calibration learning rate is still live, so every calibration number here can still move. Nothing
below is a result; it is a progress read the user asked for explicitly.

THE COMPARISON IS NOT ITERATION-MATCHED, AND THAT IS THE POINT. G20K stretches both schedules by
exactly a factor two in the exponent:  chain_decay 0.9997 -> 0.99985 = 0.9997^0.5,  calib decay
0.91 -> 0.9539 = 0.91^0.5. So G20K at iteration 2N carries the SAME learning rates BT2 had at
iteration N. Comparing the two at equal iteration therefore compares a cooler optimiser against a
hotter one and tells you nothing.

WHICH GIVES A SHARP NULL HYPOTHESIS, drawn as the dashed curve in the position panel: BT2's own
trace re-plotted at 2x its iteration axis. That is what G20K would look like if the SCHEDULE were
the only thing that mattered -- twice the iterations, each learning-rate level visited twice as
long, ending at 20 000 exactly where BT2 ended at 10 000, gaining nothing. G20K falls BELOW that
curve only if the extra steps are buying real descent, which is the hypothesis: fig 56 measured
geometry as optimiser-limited (alpha* = 0.988), not schedule-limited.

Calibration parameters are % error against each seed's OWN target (--seed draws the target); the
targets were verified identical seed-for-seed between the two arms before differencing.
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
         purple='#CC79A7', grey='#666666', ink='#222222')
plt.rcParams.update({'font.size': 9, 'axes.grid': True, 'grid.alpha': .25,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.dpi': 130, 'savefig.bbox': 'tight'})
SEEDS = [0, 1, 2]
W = 200
TRUE_MEAN = 1.88693
PL = {'Ab': 'A$_b$ (recombination)', 'eField': 'E field', 'lifetime': 'lifetime',
      'tran_diff': 'transverse diffusion', 'long_diff': 'longitudinal diffusion'}


def smooth(v, w=W):
    v = np.asarray(v, float)
    if len(v) <= w:
        return v
    from numpy.lib.stride_tricks import sliding_window_view
    return np.median(sliding_window_view(v, w), axis=1)


def load(tag, s):
    fs = glob.glob(f'fit_result/sci_full_{tag}/history_iter*seed{s}.pkl')
    f = max(fs, key=lambda p: int(re.search(r'iter(\d+)', p).group(1)))
    h = pickle.load(open(f, 'rb'))
    d = {'_n': int(re.search(r'iter(\d+)', f).group(1))}
    for p in R.P:
        t = float(np.ravel(h[p + '_target'])[0])
        d[p] = 100 * (np.ravel(h[p + '_iter'])[1:] / t - 1)
    d['mae'] = np.asarray([x for x in h.get('dedx_mae_truth_iter', []) if x == x], float)
    d['meandedx'] = np.asarray([x for x in h.get('mean_dedx_iter', []) if x == x], float)
    d['pos'] = np.ravel(h['pos_residual_iter']) * 1e4
    d['loss'] = np.asarray(h['losses_iter'], float)
    d['st'] = np.asarray(h['step_time'], float)
    return d


D = {t: {s: load(t, s) for s in SEEDS} for t in ('G20K', 'BT2')}
NG = min(D['G20K'][s]['_n'] for s in SEEDS)
print(f'G20K at {[D["G20K"][s]["_n"] for s in SEEDS]} of 20000; common {NG}')

PANELS = [('Ab', PL['Ab'], '% error', None), ('eField', PL['eField'], '% error', None),
          ('tran_diff', PL['tran_diff'], '% error', None),
          ('long_diff', PL['long_diff'], '% error', None),
          ('lifetime', PL['lifetime'], '% error', None),
          ('mae', 'per-segment dE/dx MAE vs TRUTH', 'MAE', None),
          ('meandedx', 'length-weighted ⟨dE/dx⟩', 'MeV/cm', TRUE_MEAN),
          ('pos', 'position residual  ← THE POINT OF THE RUN', 'µm', None),
          ('loss', 'loss (PPP negative log-likelihood)', 'NLL', None)]

fig, axes = plt.subplots(3, 3, figsize=(16.6, 11.4), sharex=True)
for k, (key, title, ylab, hline) in enumerate(PANELS):
    a = axes.ravel()[k]
    for tag, col, lab in [('BT2', C['ink'], 'BT2 — 10 000 it, baseline schedules'),
                          ('G20K', C['blue'], 'G20K — 20 000 it, schedules stretched ×2')]:
        cs = [smooth(D[tag][s][key]) for s in SEEDS if len(D[tag][s][key]) > W]
        for c in cs:
            a.plot(c, color=col, lw=.6, alpha=.32)
        n = min(len(c) for c in cs)
        a.plot(np.median([c[:n] for c in cs], axis=0), color=col, lw=2.3,
               label=lab if k == 0 else None)
    if key == 'pos':  # the null hypothesis: BT2 re-plotted on a 2x iteration axis
        cs = [smooth(D['BT2'][s]['pos']) for s in SEEDS]
        n = min(len(c) for c in cs)
        m = np.median([c[:n] for c in cs], axis=0)
        a.plot(2 * np.arange(n), m, color=C['red'], lw=2.2, ls='--',
               label='BT2 stretched ×2 = "schedule is all that matters"')
        a.legend(fontsize=7.0, frameon=True, framealpha=.92, loc='upper right')
    if hline is not None:
        a.axhline(hline, color=C['red'], lw=1.6, ls='--')
        a.annotate(f' TRUE {hline:.4f}', (0, hline), fontsize=7.4, color=C['red'], va='bottom')
    elif ylab == '% error':
        a.axhline(0, color=C['red'], lw=1.4, ls='--')
    a.axvline(NG, color=C['grey'], lw=1.4, ls=':')
    if k == 0:
        a.annotate(' G20K is\n here now', (NG, 0), textcoords='offset points', xytext=(3, 22),
                   fontsize=7.0, color=C['grey'])
    a.set_xlim(0, 20000)
    a.set_title(title, fontsize=9.5)
    a.set_ylabel(ylab, fontsize=8.6)
    if key in ('pos', 'loss'):
        a.set_yscale('log')
    if key == 'long_diff':
        a.set_ylim(-25, 35)
    if key == 'lifetime':
        a.set_ylim(-12, 12)
    if key == 'meandedx':
        a.set_ylim(1.855, 1.895)
    if k >= 6:
        a.set_xlabel('iteration')
axes.ravel()[0].legend(fontsize=7.4, frameon=True, framealpha=.92, loc='upper right')

fig.suptitle('Fig 61 — G20K, PRELIMINARY at ~15 000 of 20 000 iterations, seeds 0–2, seed-matched '
             'against BT2. The calibration LR is still live:\nnothing here is final. Read the '
             'position panel against the DASHED curve, not against BT2 — because G20K stretches '
             'both schedules ×2,\nG20K at iteration 2N carries the learning rates BT2 had at N, so '
             'the dashed curve is the null "extra iterations buy nothing".', fontsize=10.1)
fig.tight_layout(rect=[0, 0, 1, .925])
os.makedirs(OUT, exist_ok=True)
fig.savefig(f'{OUT}/fig61_g20k_prelim.png'); plt.close(fig)
print('wrote', f'{OUT}/fig61_g20k_prelim.png')

# ---- the numbers ----
print(f'\nposition (median over seeds {SEEDS}, 200-it rolling median)')
pg = [smooth(D['G20K'][s]['pos']) for s in SEEDS]
pb = [smooth(D['BT2'][s]['pos']) for s in SEEDS]
for it in (3000, 5000, 8000, 10000, 14000):
    g = np.median([c[it - W // 2] for c in pg]) if it - W // 2 < min(len(c) for c in pg) else np.nan
    h = it // 2  # schedule-matched BT2 iteration
    b = np.median([c[h - W // 2] for c in pb])
    braw = (np.median([c[it - W // 2] for c in pb])
            if it - W // 2 < min(len(c) for c in pb) else np.nan)
    print(f'  G20K it {it:6d}: {g:6.1f} um |  null (BT2 at {h:5d}) {b:6.1f} '
          f'({100*(g/b-1):+5.1f}%) |  BT2 same it {braw:6.1f}')
def at(c, i):
    return c[min(i, len(c) - 1)]


print('\nper-seed at the common iteration')
for s in SEEDS:
    print(f'  seed {s}: G20K {at(smooth(D["G20K"][s]["pos"]), NG-W//2-1):6.1f} um   '
          f'null {at(smooth(D["BT2"][s]["pos"]), NG//2-W//2):6.1f}   '
          f'BT2 final {np.median(smooth(D["BT2"][s]["pos"])[-500:]):6.1f}')
print('\ncalibration |error| at the common iteration (NOT converged — LR still live)')
for p in R.P:
    g = np.median([abs(at(smooth(D['G20K'][s][p]), NG - W // 2 - 1)) for s in SEEDS])
    b = np.median([abs(at(smooth(D['BT2'][s][p]), NG // 2 - W // 2)) for s in SEEDS])
    bf = np.median([abs(np.median(smooth(D['BT2'][s][p])[-500:])) for s in SEEDS])
    print(f'  {p:11s} G20K {g:7.3f}%   null {b:7.3f}%   BT2 final {bf:7.3f}%')
print('\nwhere G20K crosses BT2 final position')
bf = np.median([np.median(smooth(D['BT2'][s]['pos'])[-500:]) for s in SEEDS])
gm = np.median([smooth(D['G20K'][s]['pos'])[:min(len(smooth(D['G20K'][x]['pos'])) for x in SEEDS)]
                for s in SEEDS], axis=0)
x = np.where(gm < bf)[0]
print(f'  BT2 final {bf:.1f} um; G20K reaches it at iteration {x[0]+W//2 if len(x) else -1}')
st = np.median([np.median(D['G20K'][s]['st']) for s in SEEDS])
print(f'\nstep time {st*1e3:.0f} ms -> 20 000 iters = {20000*st/3600:.1f} h')
