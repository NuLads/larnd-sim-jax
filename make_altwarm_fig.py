"""Fig 62 — the warm-start control, which RETRACTS the alternation claim of fig 59.

THE TEST. ALT froze BT2's geometry and re-fitted calibration COLD, reaching A_b 0.147 against BT2's
0.340 (fig 59). Is 0.147 the optimum of the frozen-geometry objective, or where the cold start
happened to land? ALTW approaches the SAME objective from the other direction: calibration
initialised at BT2's own fitted values. Converge back to ~0.15 and 0.15 is real; stay near 0.34 and
the cold start got lucky.

VERIFIED FIRST, because the whole comparison rests on it: ALTW's iteration-0 calibration equals
BT2's final values to all printed digits on all five parameters and both seeds, and both arms
inherited byte-identical geometry (860/864 tracks match BT2 and each other, same 4 mismatches). The
two arms differ ONLY in the calibration initialisation.

THE ANSWER IS NEITHER, AND THE RUN INVALIDATES ITS OWN PREMISE TWICE OVER.

  (b) The warm start is ERASED inside ~200 iterations. Only calibration is warm-started -- there is
      no init-from for the dEdx block, so it restarts near the prior centre in both arms and
      immediately drags A_b with it, the charge-normalisation coupling of section 6u. ALTW leaves
      its init (+0.48) and is at +1.27 by iteration 300, while ALT arrives from -4.18. By iteration
      ~300 the two arms are in the SAME region and neither remembers where it started. This is not
      a two-directions test; it is two runs of the same cold-dEdx transient.

  (c) NEITHER ARM IS CONVERGED at readout. The calibration LR decays 0.91 per EPOCH, so at 3 000
      iterations (30 epochs) it is 0.059 of peak -- against 8.0e-5 for BT2 at 10 000, a factor 700.
      A_b's end-of-run spread is 0.12-0.17 in ALT and ALTW and 0.001 in BT2. Their endpoints are
      snapshots of an oscillation.

WHICH MEANS FIG 59's HEADLINE WAS A UNIT MISMATCH IN DISGUISE: 0.340 vs 0.147 compared BT2's FROZEN
endpoint against a mid-oscillation sample of a still-moving run. The improvement is not established.
Panel (e) is the corroborating evidence -- all three arms sit within 0.4% on the objective actually
being minimised, so nothing here is a meaningful descent.
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
S = [0, 1]
W = 200
PL = {'Ab': 'A$_b$', 'eField': 'E field', 'lifetime': 'lifetime',
      'tran_diff': 'tran. diff.', 'long_diff': 'long. diff.'}
ARMS = [('BT2', C['ink'], 'BT2 — joint, 10 000 it'),
        ('ALT', C['green'], 'ALT — frozen geom, COLD start'),
        ('ALTW', C['purple'], 'ALTW — frozen geom, WARM start')]


def sm(v, w=W):
    v = np.asarray(v, float)
    from numpy.lib.stride_tricks import sliding_window_view
    return np.median(sliding_window_view(v, w), axis=1) if len(v) > w else v


def load(tag, s):
    fs = glob.glob(f'fit_result/sci_full_{tag}/history_iter*seed{s}.pkl')
    f = max(fs, key=lambda p: int(re.search(r'iter(\d+)', p).group(1)))
    h = pickle.load(open(f, 'rb'))
    d = {}
    for p in R.P:
        t = float(np.ravel(h[p + '_target'])[0])
        d[p] = 100 * (np.ravel(h[p + '_iter']) / t - 1)
    d['mae'] = np.asarray([x for x in h.get('dedx_mae_truth_iter', []) if x == x], float)
    d['loss'] = np.asarray(h['losses_iter'], float)
    return d


D = {t: {s: load(t, s) for s in S} for t, _, _ in ARMS}
fin = lambda t, s, p: np.median(sm(D[t][s][p])[-300:])

fig, ax = plt.subplots(2, 3, figsize=(16.6, 9.0))

# ---------------- (a) the two-directions test ----------------
a = ax[0, 0]
for tag, col, lab in ARMS:
    for s in S:
        c = sm(D[tag][s]['Ab'])
        a.plot(c, color=col, lw=1.9 if s == 0 else 1.9, alpha=.85 if s == 0 else .45,
               label=lab if s == 0 else None)
a.axhline(0, color=C['red'], lw=1.3, ls='--')
a.set_xlim(0, 3200); a.set_ylim(-1.2, 1.6)
a.set_xlabel('iteration'); a.set_ylabel('A$_b$ % error')
a.set_title('(a) THE TEST: do the two directions meet? They do NOT.\n'
            'ALTW ends at 0.38 |err|, ALT at 0.14, BT2 at 0.34.', fontsize=9.3)
a.legend(fontsize=7.0, frameon=False, loc='lower right')

# ---------------- (b) the warm start is erased ----------------
a = ax[0, 1]
for tag, col, lab in [('ALT', C['green'], 'ALT (cold)'), ('ALTW', C['purple'], 'ALTW (warm)')]:
    for s in S:
        a.plot(D[tag][s]['Ab'][:400], color=col, lw=1.7, alpha=.85 if s == 0 else .45,
               label=lab if s == 0 else None)
for s in S:
    a.plot(0, D['BT2'][s]['Ab'][-1], marker='*', ms=15, color=C['ink'], zorder=5,
           label='BT2 final = ALTW init' if s == 0 else None)
a.axvspan(200, 400, color=C['red'], alpha=.09)
a.annotate('by ~300 both arms are in the\nSAME region — the init is forgotten',
           (215, -3.2), fontsize=7.2, color=C['red'])
a.set_xlim(0, 400)
a.set_xlabel('iteration'); a.set_ylabel('A$_b$ % error')
a.set_title('(b) WHY IT CANNOT WORK: the dE/dx block is NOT warm-started,\n'
            'restarts near the prior centre, and drags A$_b$ off its init.', fontsize=9.3)
a.legend(fontsize=7.0, frameon=False, loc='lower right')

# ---------------- (c) neither arm is converged ----------------
a = ax[0, 2]
xs = np.arange(3)
spread = [np.median([np.ptp(sm(D[t][s]['Ab'])[-300:]) for s in S]) for t, _, _ in ARMS]
a.bar(xs, spread, color=[c for _, c, _ in ARMS], edgecolor='white', width=.6)
for i, v in enumerate(spread):
    a.text(i, v * 1.15, f'{v:.3f}', ha='center', fontsize=8.6)
a.set_yscale('log'); a.set_ylim(5e-4, 1.2)
a.set_xticks(xs); a.set_xticklabels([t for t, _, _ in ARMS], fontsize=9)
a.set_ylabel('A$_b$ end-of-run spread (% error, peak-to-peak)')
a.set_title('(c) …AND NEITHER IS CONVERGED. Calib LR at readout:\n'
            '0.059 of peak (30 epochs) vs BT2\'s 8.0e−5 — a factor 700.', fontsize=9.3)

# ---------------- (d) all parameters ----------------
a = ax[1, 0]
xs = np.arange(len(R.P)); w = .26
for i, (tag, col, lab) in enumerate(ARMS):
    v = [np.median([abs(fin(tag, s, p)) for s in S]) for p in R.P]
    a.bar(xs + (i - 1) * w, v, width=w, color=col, edgecolor='white', label=lab)
a.set_yscale('log')
a.set_xticks(xs); a.set_xticklabels([PL[p] for p in R.P], fontsize=8, rotation=18, ha='right')
a.set_ylabel('|error| (%)')
a.set_title('(d) no arm wins across the board, and every difference\n'
            'is inside the oscillation amplitude of panel (c).', fontsize=9.3)
a.legend(fontsize=6.8, frameon=False)

# ---------------- (e) the objective ----------------
a = ax[1, 1]
lo = [np.median([fin(t, s, 'loss') for s in S]) for t, _, _ in ARMS]
a.bar(np.arange(3), lo, color=[c for _, c, _ in ARMS], edgecolor='white', width=.6)
for i, v in enumerate(lo):
    a.text(i, v + 6, f'{v:.0f}', ha='center', fontsize=8.6)
a.set_ylim(min(lo) - 120, max(lo) + 90)
a.set_xticks(np.arange(3)); a.set_xticklabels([t for t, _, _ in ARMS], fontsize=9)
a.set_ylabel('loss (PPP NLL)')
a.set_title(f'(e) THE CORROBORATION: all three sit within '
            f'{100*(max(lo)/min(lo)-1):.1f}% on the\nobjective actually minimised. Nothing '
            'descended.', fontsize=9.3)

# ---------------- (f) per-seed A_b, the honest picture ----------------
a = ax[1, 2]
for i, (tag, col, lab) in enumerate(ARMS):
    for s in S:
        v = fin(tag, s, 'Ab')
        sp = np.ptp(sm(D[tag][s]['Ab'])[-300:]) / 2
        a.errorbar(i + (s - .5) * .22, v, yerr=sp, fmt='o', ms=8, color=col,
                   alpha=.9 if s == 0 else .5, capsize=4,
                   label=f'seed {s}' if i == 0 else None)
a.axhline(0, color=C['red'], lw=1.3, ls='--')
a.set_xticks(np.arange(3)); a.set_xticklabels([t for t, _, _ in ARMS], fontsize=9)
a.set_ylabel('A$_b$ % error  (bars = end-of-run spread)')
a.set_title('(f) per seed, with the oscillation shown. ALT\'s "0.008" on\n'
            'seed 1 carries a ±0.065 wander — it is a sample, not a value.', fontsize=9.3)
a.legend(fontsize=7.2, frameon=False)

fig.suptitle('Fig 62 — the warm-start control RETRACTS fig 59. The warm start is erased in ~200 '
             'iterations (the dE/dx block is not warm-started and\ndrags A$_b$ with it), and '
             'neither frozen-geometry arm is converged — their calibration LR at readout is 700× '
             'BT2\'s, and A$_b$ still wanders\nby ±0.13 against BT2\'s ±0.001. Fig 59\'s '
             '"A$_b$ 0.340 → 0.147, ×2.3" compared a frozen endpoint against a mid-oscillation '
             'sample.', fontsize=10.0)
fig.tight_layout(rect=[0, 0, 1, .90])
os.makedirs(OUT, exist_ok=True)
fig.savefig(f'{OUT}/fig62_altwarm.png'); plt.close(fig)
print('wrote', f'{OUT}/fig62_altwarm.png')
for p in R.P:
    print(f'  {p:11s} ' + '  '.join(
        f'{t} {np.median([abs(fin(t,s,p)) for s in S]):7.3f}' for t, _, _ in ARMS))
print('\nAb per seed (final, end-spread):')
for t, _, _ in ARMS:
    for s in S:
        print(f'  {t:5s} seed {s}: {fin(t,s,"Ab"):+.3f}  spread {np.ptp(sm(D[t][s]["Ab"])[-300:]):.3f}')
