"""Fig 66 — CORRECTED. Is the lifetime error the optimiser's or the objective's? Mostly the
optimiser's: the objective PREFERS truth, and the fit has not got there.

WHAT THIS FIGURE ORIGINALLY CLAIMED, AND WHY IT WAS WRONG. The first version scanned each
calibration parameter with geometry and dEdx pinned at their FITTED values, found the fit sitting on
that scan's minimum every time, and concluded the objective was displaced from truth. That is a
CONDITIONAL minimum -- a 1-D slice through a coupled problem -- and a conditional minimum sitting at
the fit says nothing about where the JOINT minimum is. The degeneracy-matrix script already carried
this caveat for the CURVATURE and it was not applied to the LOCATION.

THE SAME FILES CONTAIN THE TEST. Every scan also evaluates constructed points: P0_fit (everything as
fitted), P1_truth (calibration AND dEdx AND geometry all at truth), and P2/P3/P4 (one block at truth,
the rest fitted). Panel (a) is the comparison that settles it.

  P1_truth beats P0_fit on 7 of 7 arm-seeds, by 5.6% of the objective on average.

The objective PREFERS truth. The fit is therefore NOT at the joint minimum, and the calibration error
is a reachability problem, not an objective bias -- which is exactly what study 2 (section 6s)
concluded and this figure briefly contradicted.

WHAT THE CONDITIONAL SCAN DOES LEGITIMATELY SHOW (panel d) is COUPLING: given nuisances that are
wrong, the objective's preferred lifetime is dragged to wherever the fit is. That is why moving a
single block to truth can make the loss WORSE -- P2_calib_true is +0.18% and P3_dedx_true +2.79% --
while moving everything to truth makes it much better. Blocks have to move together.

AND THE GAP IS CLOSING (panel b): truth beats the fit by 9.4% at BT2, 6.6% at BM0 and only 2.5% at
G20K. G20K genuinely reached a better region; it did not merely trade one error for another.

ONE GENUINE OBJECTIVE BIAS SURVIVES, and it is not lifetime's: P3_dedx_true is WORSE than the fit on
7 of 7, by 2.8%. Putting the per-segment dEdx at TRUTH makes the objective worse -- the dEdx prior
really does displace its own block's minimum, the standing section-6s result.

THE RESIDUAL, stated honestly (panel e). With geometry and dEdx BOTH at truth the lifetime slice
minimum is still 2.26% from truth, down from 2.83%. So the nuisance state explains only about a
fifth of it. But that scan still holds the OTHER FOUR calibration parameters at their fitted values,
so it remains a conditional number and cannot be read as a joint bias either. Settling that needs a
profile over the full calibration block, which no run has done.
"""
import glob, os, pickle, io, contextlib
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
M = [0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2]
ARMS = [('BT2', 'DEGEN', [0, 1], C['ink']), ('BM0', 'LSCANBM0', [0, 1], C['red']),
        ('G20K', 'LSCANG20K', [0, 1, 2], C['blue'])]
PL = {'Ab': 'A$_b$', 'eField': 'E field', 'lifetime': 'lifetime',
      'tran_diff': 'tran. diff.', 'long_diff': 'long. diff.'}
PMK = {'Ab': 'o', 'eField': 's', 'tran_diff': '^', 'long_diff': 'D', 'lifetime': '*'}
COMBOS = [('gfit', 'dfit', 'nuisances\nat FIT'), ('gtru', 'dfit', 'geometry\nat truth'),
          ('gfit', 'dtru', 'dE/dx\nat truth'), ('gtru', 'dtru', 'BOTH\nat truth')]


def scan(tag, s):
    fs = glob.glob(f'fit_result/sci_full_{tag}/history_iter*seed{s}.pkl')
    return pickle.load(open(fs[0], 'rb')).get('truthpoint', []) if fs else None


def agg(Rs, key):
    t = 0.0
    for r in Rs:
        if key not in r['points']:
            return None
        t += r['points'][key]['loss']
    return t


def par(ys):
    ys = np.asarray(ys, float)
    i = int(np.argmin(ys)); edge = (i == 0 or i == len(ys) - 1)
    i = min(max(i, 1), len(ys) - 2)
    c = np.polyfit(M[i - 1:i + 2], ys[i - 1:i + 2], 2)
    return float(-c[1] / (2 * c[0])), edge


def srcvals(arm, s):
    pat = {'G20K': 'fit_result/sci_full_G20K/history_iter20000_*seed%d.pkl',
           'BM0': 'fit_result/sci_full_BM0/history_iter3000_*seed%d.pkl',
           'BT2': 'fit_result/sci_full_BT2/history_iter10000_*seed%d.pkl'}[arm]
    h = pickle.load(open(glob.glob(pat % s)[0], 'rb'))
    return ({p: float(np.ravel(h[p + '_iter'])[-1]) for p in R.P},
            {p: float(np.ravel(h[p + '_target'])[0]) for p in R.P})


BASE, D, CND = {}, {}, {}
for arm, tag, seeds, _ in ARMS:
    for s in seeds:
        Rs = scan(tag, s)
        if not Rs:
            continue
        fit, tgt = srcvals(arm, s)
        tot = {k: sum(r['points'][k]['loss'] for r in Rs if k in r['points'])
               for k in ['P0_fit', 'P1_truth', 'P2_calib_true', 'P3_dedx_true', 'P4_geom_true']}
        BASE[(arm, s)] = {k: 100 * (v / tot['P0_fit'] - 1) for k, v in tot.items()}
        for p in R.P:
            for g, d, lab in COMBOS:
                ys = [agg(Rs, f'C{p}_{g}_{d}_{m:g}') for m in M]
                if any(y is None for y in ys):
                    continue
                m, _ = par(ys)
                CND[(arm, s, p, lab)] = 100 * (m * fit[p] / tgt[p] - 1)
            if (arm, s, p, COMBOS[0][2]) in CND:
                D[(arm, s, p)] = (100 * (fit[p] / tgt[p] - 1), CND[(arm, s, p, COMBOS[0][2])])

fig, ax = plt.subplots(2, 3, figsize=(16.8, 9.4))

# ---------------- (a) THE TEST THAT SETTLES IT ----------------
a = ax[0, 0]
i = 0
for arm, _, seeds, col in ARMS:
    for s in seeds:
        v = BASE[(arm, s)]['P1_truth']
        a.barh(i, v, color=col, edgecolor='white', height=.7)
        a.text(v - .25, i, f'{v:+.2f}%', va='center', ha='right', fontsize=7.8, color='white')
        i += 1
a.axvline(0, color=C['ink'], lw=2)
a.annotate(' the FIT', (0, i - .4), fontsize=8, fontweight='bold')
a.set_yticks(range(i))
a.set_yticklabels([f'{arm} s{s}' for arm, _, seeds, _ in ARMS for s in seeds], fontsize=8)
a.invert_yaxis()
a.set_xlabel('loss at FULL TRUTH − loss at the fit (%)')
a.set_title('(a) THE TEST THAT SETTLES IT: everything at truth beats the fit\n'
            'on 7/7, by 5.6%. THE OBJECTIVE PREFERS TRUTH.', fontsize=9.2)

# ---------------- (b) the gap is closing ----------------
a = ax[0, 1]
for arm, _, seeds, col in ARMS:
    v = [-BASE[(arm, s)]['P1_truth'] for s in seeds]
    a.bar(arm, np.mean(v), color=col, edgecolor='white', width=.6)
    for y in v:
        a.plot(arm, y, 'o', ms=7, color='white', mew=0)
a.set_ylabel('how much better truth is than the fit (%)')
a.set_title('(b) AND THE GAP IS CLOSING. 9.4% → 6.6% → 2.5%: G20K really\n'
            'reached a better region, it did not just trade errors.', fontsize=9.2)

# ---------------- (c) blocks must move together ----------------
a = ax[0, 2]
KEYS = [('P4_geom_true', 'geometry\nalone'), ('P2_calib_true', 'calibration\nalone'),
        ('P3_dedx_true', 'dE/dx\nalone'), ('P1_truth', 'ALL THREE\ntogether')]
for j, (k, lab) in enumerate(KEYS):
    v = [BASE[key][k] for key in BASE]
    col = C['green'] if np.mean(v) < 0 else C['red']
    a.bar(j, np.mean(v), color=col, edgecolor='white', width=.62)
    for y in v:
        a.plot(j, y, 'o', ms=5, color='white', mew=0)
    a.text(j, np.mean(v) + (.25 if np.mean(v) > 0 else -.55), f'{np.mean(v):+.2f}%',
           ha='center', fontsize=8.4)
a.axhline(0, color=C['ink'], lw=1.8)
a.set_xticks(range(4)); a.set_xticklabels([l for _, l in KEYS], fontsize=8)
a.set_ylabel('Δ loss vs the fit (%)   negative = better')
a.set_title('(c) BLOCKS MUST MOVE TOGETHER. Moving calibration or dE/dx to\n'
            'truth ALONE makes the loss WORSE. Only geometry helps alone.', fontsize=9.2)

# ---------------- (d) what the conditional scan really shows ----------------
a = ax[1, 0]
for arm, _, seeds, col in ARMS:
    for s in seeds:
        for p in R.P:
            if (arm, s, p) in D:
                fe, oe = D[(arm, s, p)]
                a.plot(fe, oe, PMK[p], ms=11 if p == 'lifetime' else 7, color=col, mew=0)
lim = 11
a.plot([-lim, lim], [-lim, lim], color=C['grey'], lw=2, ls='--', zorder=0)
a.axhline(0, color=C['green'], lw=1.4)
a.set_xlim(-lim, lim); a.set_ylim(-lim, lim)
a.set_xlabel('fit error (%)')
a.set_ylabel('CONDITIONAL optimum error (%)')
h1 = [plt.Line2D([], [], marker=PMK[p], ls='', color=C['grey'], label=PL[p]) for p in R.P]
a.legend(handles=h1, fontsize=6.6, frameon=False, ncol=2, loc='lower right')
a.set_title('(d) WHAT THE 1-D SCAN ACTUALLY SHOWS — COUPLING, not bias.\n'
            'Wrong nuisances drag the preferred value onto the fit.', fontsize=9.2)

# ---------------- (e) the residual, honestly ----------------
a = ax[1, 1]
for j, (_, _, lab) in enumerate(COMBOS):
    v = [abs(CND[(arm, s, 'lifetime', lab)]) for arm, _, seeds, _ in ARMS for s in seeds
         if (arm, s, 'lifetime', lab) in CND]
    a.bar(j, np.mean(v), color=C['purple'], edgecolor='white', width=.62)
    for y in v:
        a.plot(j, y, 'o', ms=5, color='white', mew=0)
    a.text(j, np.mean(v) + .12, f'{np.mean(v):.2f}%', ha='center', fontsize=8.6)
a.set_xticks(range(4)); a.set_xticklabels([l for _, _, l in COMBOS], fontsize=7.8)
a.set_ylabel('|lifetime slice minimum − truth| (%)')
a.set_title('(e) THE RESIDUAL, honestly. Both nuisances at truth removes only\n'
            '⅕ of it — but this still fixes the other 4 calib params.', fontsize=9.2)

# ---------------- (f) the killed hypothesis ----------------
a = ax[1, 2]
X, Y = [], []
for arm, _, seeds, col in ARMS:
    for s in seeds:
        if (arm, s, 'lifetime') in D and (arm, s, 'long_diff') in D:
            X.append(D[(arm, s, 'long_diff')][1]); Y.append(D[(arm, s, 'lifetime')][1])
            a.plot(X[-1], Y[-1], 'o', ms=11, color=col)
            a.annotate(f'  {arm} s{s}', (X[-1], Y[-1]), fontsize=6.8)
a.axhline(0, color=C['grey'], lw=1.2); a.axvline(0, color=C['grey'], lw=1.2)
a.set_xlabel('longitudinal diffusion — conditional optimum (%)')
a.set_ylabel('lifetime — conditional optimum (%)')
a.set_title(f'(f) A HYPOTHESIS KILLED, and this part stands. Both act along\n'
            f'drift, so a trade was obvious. Correlation {np.corrcoef(X, Y)[0,1]:+.2f} (n=7). None.',
            fontsize=9.2)

fig.suptitle('Fig 66 (CORRECTED) — the objective PREFERS truth: with calibration, dE/dx and geometry '
             'all at truth the loss beats the fit on 7/7 arm-seeds\nby 5.6%, so the error is '
             'reachability, not objective bias. The earlier reading came from a 1-D scan with the '
             'nuisances pinned at their\nFITTED values — a conditional minimum, which sits on the '
             'fit by construction and says nothing about where the joint minimum is.',
             fontsize=10.0)
fig.tight_layout(rect=[0, 0, 1, .90])
os.makedirs(OUT, exist_ok=True)
fig.savefig(f'{OUT}/fig66_objective_bias.png'); plt.close(fig)
print('wrote', f'{OUT}/fig66_objective_bias.png')
for k, lab in KEYS:
    v = [BASE[key][k] for key in BASE]
    print(f'  {lab:22s} {np.mean(v):+7.2f}%   better on {sum(1 for x in v if x<0)}/{len(v)}')
