"""Fig 61 — G20K FINAL: 20 000 iterations with both schedules stretched x2. Seeds 0-2, complete.

BOTH ARMS ARE CONVERGED AND READ AT THE SAME SCHEDULE PHASE, which is what makes the endpoints
comparable at all -- and is exactly the discipline trap 1b (fig 62) exists to enforce. The x2
stretch is in the exponent, so G20K's terminal LR after 200 epochs is 0.9539^200 = 7.95e-5 against
BT2's 0.91^100 = 8.02e-5: the same number. End-of-run parameter spreads are 0.0001-0.016 in both.
Neither arm is sampled mid-oscillation.

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

READ THE PAIRED DIFFERENCES, NOT THE RATIO OF MEDIANS. With n = 3 the median is nearly the middle
sample, and ratios of medians manufacture headline numbers the per-seed data does not support:
A_b reads "+73%" and lifetime "+88%" as median ratios while their paired differences are
+0.027 +- 0.085 and +0.718 +- 0.888 -- both consistent with zero. Panel titles quote the paired
difference for that reason. This is the same signed-mean/median trap section 7 catalogues.
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

PANELS = [('Ab', PL['Ab'] + '   (paired +0.03 ± 0.09 → null)', '% error', None),
          ('eField', PL['eField'] + '   (paired −0.000 ± 0.012 → null)', '% error', None),
          ('tran_diff', PL['tran_diff'] + '  ← |err| −1.42 ± 0.40, 3/3', '% error', None),
          ('long_diff', PL['long_diff'] + '   (paired −1.23 ± 4.02 → null)', '% error', None),
          ('lifetime', PL['lifetime'] + '  (paired +0.72 ± 0.89 = not established)', '% error', None),
          ('mae', 'per-segment dE/dx MAE vs TRUTH', 'MAE', None),
          ('meandedx', 'length-weighted ⟨dE/dx⟩', 'MeV/cm', TRUE_MEAN),
          ('pos', 'position residual  ← −57.2 ± 0.3 µm, 3/3 seeds', 'µm', None),
          ('loss', 'loss (PPP NLL)  ← −667 ± 92, the objective itself', 'NLL', None)]

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
    a.axvline(10000, color=C['grey'], lw=1.2, ls=':')
    if k == 0:
        a.annotate(' BT2 ends\n here', (10000, 0), textcoords='offset points', xytext=(3, 22),
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

fig.suptitle('Fig 61 — G20K FINAL, 20 000 iterations, seeds 0–2, seed-matched against BT2 and read '
             'at the SAME terminal LR (7.95e−5 vs 8.02e−5).\nPosition −57.2 ± 0.3 µm on 3/3 seeds '
             '(−34.7%) and the loss −8.7%: the extra iterations buy a genuinely better optimum, not '
             'a better-looking metric.\nRead the position panel against the DASHED null — BT2 '
             'replotted on a doubled axis — because the ×2 schedule stretch means equal iteration '
             'is NOT equal LR.', fontsize=10.1)
fig.tight_layout(rect=[0, 0, 1, .925])
os.makedirs(OUT, exist_ok=True)
fig.savefig(f'{OUT}/fig61_g20k.png'); plt.close(fig)
print('wrote', f'{OUT}/fig61_g20k.png')

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
