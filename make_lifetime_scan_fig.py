"""Fig 66 — is the lifetime error the OPTIMISER's fault or the OBJECTIVE's? It is the objective's.

THE QUESTION. Every arm that improved the loss made lifetime worse (fig 65f). Two readings: either
the optimiser walks to a point the objective does not actually prefer, or the objective's own
minimum is not at truth. These demand completely different work, so they had to be separated.

THE TEST, evaluation only, no fitting. Re-run exactly the fig 58 calibration scan -- each parameter
over +-20% of its fitted value -- but at BM0's and G20K's converged checkpoints instead of BT2's.
For each, compare:
    fit_err = 100*(fitted/target - 1)          where the FIT put the parameter
    opt_err = 100*(m* * fitted/target - 1)     where the OBJECTIVE's minimum sits
5 scans x 100 batches x 145 points; BT2's arm is the existing fig 58 scan, unchanged.

THE ANSWER IS UNAMBIGUOUS: the two agree everywhere. |fit_err - opt_err| is 0.14-0.26 points for
A_b, E field and transverse diffusion, 0.65 for lifetime and 1.54 for longitudinal diffusion --
in every case far below the error itself. THE OPTIMISER IS DOING ITS JOB. It walks to the bottom of
the objective it is given, and the bottom is not at truth.

SO THE BIAS IS IN THE OBJECTIVE, and it is parameter-specific: for A_b and E field the objective's
own minimum sits within 0.15-0.47% of truth, which is why those two are the parameters we recover
well. For the diffusions and lifetime it is displaced by 2-7%.

A NOTE ON E FIELD, so panel (b) is not over-read: both of its bars sit at ~0.15%, which is the
parabola fit's own resolution on a grid spaced 5% apart -- not a measurement. The fit is 0.02% from
truth, finer than this scan can resolve, so E field simply has no error left to apportion.

AND THE DISPLACEMENT IS NOT A FIXED OFFSET -- IT MOVES WITH THE NUISANCE STATE. That is the part
with teeth. BM0's objective wants lifetime at +4.94% (+4.72 and +5.16 on the two seeds, tight),
while BT2's wants -1.09% and G20K's -0.23%: a 6-point swing in the location of the minimum, caused
by nothing but re-annealing geometry and dEdx. BT2 and G20K straddle zero seed-to-seed by +-2-3%,
so at those points the honest statement is a seed-dependent wander of a couple of percent rather
than a systematic offset.

A HYPOTHESIS THIS KILLS. The obvious mechanism was a lifetime<->longitudinal-diffusion trade: both
act along the drift axis, and BM0 improved long. diffusion by 5.7 points while degrading lifetime by
3.6. Across all seven arm-seeds the correlation between their two optima is +0.03. There is no such
trade. Whatever moves the lifetime minimum, it is not longitudinal diffusion.
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
M = [0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2]
ARMS = [('BT2', 'DEGEN', [0, 1], C['ink']), ('BM0', 'LSCANBM0', [0, 1], C['red']),
        ('G20K', 'LSCANG20K', [0, 1, 2], C['blue'])]
PL = {'Ab': 'A$_b$', 'eField': 'E field', 'lifetime': 'lifetime',
      'tran_diff': 'tran. diff.', 'long_diff': 'long. diff.'}
PMK = {'Ab': 'o', 'eField': 's', 'tran_diff': '^', 'long_diff': 'D', 'lifetime': '*'}


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


D, SH = {}, {}
for arm, tag, seeds, _ in ARMS:
    for s in seeds:
        Rs = scan(tag, s)
        if not Rs:
            continue
        fit, tgt = srcvals(arm, s)
        for p in R.P:
            ys = [agg(Rs, f'C{p}_gfit_dfit_{m:g}') for m in M]
            if any(y is None for y in ys):
                continue
            m, _ = par(ys)
            D[(arm, s, p)] = (100 * (fit[p] / tgt[p] - 1), 100 * (m * fit[p] / tgt[p] - 1))
            ys2 = [agg(Rs, f'C{p}_gtru_dfit_{m2:g}') for m2 in M]
            if not any(y is None for y in ys2):
                m2, e2 = par(ys2)
                SH[(arm, s, p)] = (100 * (m2 - m), e2)

fig, ax = plt.subplots(2, 3, figsize=(16.8, 9.3))

# ---------------- (a) the money panel ----------------
a = ax[0, 0]
for arm, _, seeds, col in ARMS:
    for s in seeds:
        for p in R.P:
            if (arm, s, p) not in D:
                continue
            fe, oe = D[(arm, s, p)]
            a.plot(fe, oe, PMK[p], ms=11 if p == 'lifetime' else 7, color=col,
                   alpha=.9, mew=0)
lim = 11
a.plot([-lim, lim], [-lim, lim], color=C['grey'], lw=2, ls='--', zorder=0)
a.annotate('the fit IS the objective’s optimum', (-10, -8.4), fontsize=7.8, color=C['grey'],
           rotation=39)
a.axhline(0, color=C['green'], lw=1.6)
a.annotate(' objective wants TRUTH', (-10.5, .35), fontsize=7.6, color=C['green'])
a.set_xlim(-lim, lim); a.set_ylim(-lim, lim)
a.set_xlabel('fit error (%)  — where the FIT put it')
a.set_ylabel('optimum error (%)  — where the OBJECTIVE wants it')
h1 = [plt.Line2D([], [], marker=PMK[p], ls='', color=C['grey'], label=PL[p]) for p in R.P]
h2 = [plt.Line2D([], [], marker='o', ls='', color=c, label=a_) for a_, _, _, c in ARMS]
a.legend(handles=h1 + h2, fontsize=6.6, frameon=False, ncol=2, loc='lower right')
a.set_title('(a) EVERY POINT IS ON THE DIAGONAL, not on the green line.\n'
            'The optimiser goes where the objective points — and it is wrong.', fontsize=9.2)

# ---------------- (b) optimiser is fine ----------------
a = ax[1, 0]
xs = np.arange(len(R.P))
v = [np.mean([abs(D[k][0] - D[k][1]) for k in D if k[2] == p]) for p in R.P]
w = [np.mean([abs(D[k][1]) for k in D if k[2] == p]) for p in R.P]
a.bar(xs - .19, v, width=.38, color=C['green'], edgecolor='white',
      label='|fit − objective optimum|  (the optimiser’s share)')
a.bar(xs + .19, w, width=.38, color=C['red'], edgecolor='white',
      label='|objective optimum − truth|  (the OBJECTIVE’s share)')
a.set_xticks(xs); a.set_xticklabels([PL[p] for p in R.P], fontsize=8, rotation=18, ha='right')
a.set_ylabel('% of the target')
a.set_title('(b) THE ERROR BUDGET. Red dominates wherever there is real error\n'
            'to explain: for lifetime 4.3×, long. diff. 2.7×, tran. diff. 5.3×.', fontsize=9.2)
a.annotate('E field: both bars ≈0.15% = the scan’s own\nresolution on a 5% grid, not physics\n'
           '(the fit itself is 0.02% from truth)', (1.0, 3.4), fontsize=6.6, color=C['grey'],
           ha='center')
a.legend(fontsize=6.9, frameon=False)

# ---------------- (c) the objective's own bias per parameter ----------------
a = ax[0, 1]
wid = .26
for i, (arm, _, seeds, col) in enumerate(ARMS):
    for pi, p in enumerate(R.P):
        vv = [abs(D[(arm, s, p)][1]) for s in seeds if (arm, s, p) in D]
        x = pi + (i - 1) * wid
        a.bar(x, np.mean(vv), width=wid, color=col, edgecolor='white',
              label=arm if pi == 0 else None)
        for y in vv:
            a.plot(x, y, 'o', ms=4, color='white', mew=0)
a.set_yscale('log')
a.set_xticks(np.arange(len(R.P)))
a.set_xticklabels([PL[p] for p in R.P], fontsize=8, rotation=18, ha='right')
a.set_ylabel('|objective optimum − truth| (%)')
a.set_title('(c) THE BIAS IS PARAMETER-SPECIFIC. A$_b$ and E field: the\n'
            'objective wants truth to ≲0.5%. The rest: 2–7% off.', fontsize=9.2)
a.legend(fontsize=7.2, frameon=False)

# ---------------- (d) lifetime, per seed ----------------
a = ax[1, 1]
for i, (arm, _, seeds, col) in enumerate(ARMS):
    vv = [D[(arm, s, 'lifetime')][1] for s in seeds if (arm, s, 'lifetime') in D]
    ff = [D[(arm, s, 'lifetime')][0] for s in seeds if (arm, s, 'lifetime') in D]
    for j, (y, f_) in enumerate(zip(vv, ff)):
        a.plot(i - .12, f_, 'o', ms=9, color=col, alpha=.35)
        a.plot(i + .12, y, 'o', ms=11, color=col)
        a.plot([i - .12, i + .12], [f_, y], color=col, lw=1, alpha=.4)
a.axhline(0, color=C['green'], lw=2)
a.annotate(' truth', (2.3, .2), fontsize=7.8, color=C['green'])
a.set_xticks(range(3)); a.set_xticklabels([f'{a_}\nfaint = fit, solid = optimum'
                                           for a_, _, _, _ in ARMS], fontsize=7.6)
a.set_ylabel('lifetime error (%)')
a.set_title('(d) LIFETIME: BM0’s objective wants +4.9% on BOTH seeds.\n'
            'BT2/G20K straddle truth by ±2–3% — a wander, not an offset.', fontsize=9.2)

# ---------------- (e) the falsified hypothesis ----------------
a = ax[0, 2]
X, Y = [], []
for arm, _, seeds, col in ARMS:
    for s in seeds:
        if (arm, s, 'lifetime') in D and (arm, s, 'long_diff') in D:
            X.append(D[(arm, s, 'long_diff')][1]); Y.append(D[(arm, s, 'lifetime')][1])
            a.plot(X[-1], Y[-1], 'o', ms=11, color=col)
            a.annotate(f'  {arm} s{s}', (X[-1], Y[-1]), fontsize=6.8)
a.axhline(0, color=C['grey'], lw=1.2); a.axvline(0, color=C['grey'], lw=1.2)
a.set_xlabel('longitudinal diffusion — objective optimum (%)')
a.set_ylabel('lifetime — objective optimum (%)')
a.set_title(f'(e) A HYPOTHESIS KILLED. Both act along drift, so a trade was\n'
            f'the obvious mechanism. Correlation = {np.corrcoef(X, Y)[0,1]:+.2f} (n=7). No trade.',
            fontsize=9.2)

# ---------------- (f) does better geometry shrink the coupling? ----------------
a = ax[1, 2]
for i, p in enumerate(['tran_diff', 'long_diff', 'lifetime', 'Ab']):
    vals, cols = [], []
    for arm, _, seeds, col in ARMS:
        vv = [abs(SH[(arm, s, p)][0]) for s in seeds if (arm, s, p) in SH]
        vals.append(np.mean(vv) if vv else np.nan); cols.append(col)
    for j, (v, col) in enumerate(zip(vals, cols)):
        a.bar(i + (j - 1) * .26, v, width=.26, color=col, edgecolor='white')
a.set_yscale('log')
a.set_xticks(range(4))
a.set_xticklabels(['tran. diff.', 'long. diff.', 'lifetime', 'A$_b$'], fontsize=8)
a.set_ylabel('|shift of the optimum| when geometry → truth (%)')
a.set_title('(f) BETTER GEOMETRY SHRINKS THE COUPLING for the diffusions\n'
            'and A$_b$ — but NOT for lifetime, which is flat after BT2.', fontsize=9.2)
a.annotate('BT2 → BM0 → G20K within each group.\nMinima run off the ±20% grid, so these are\n'
           'extrapolations: read the ORDER, not the values.', (1.5, 45), fontsize=6.6,
           color=C['grey'], ha='center')

fig.suptitle('Fig 66 — the lifetime error is the OBJECTIVE’s, not the optimiser’s. At every arm and '
             'every seed the fit sits on the objective’s own minimum\n(panel a, all points on the '
             'diagonal); that minimum is within 0.5% of truth for A$_b$ and E field and 2–7% away '
             'for the diffusions and lifetime.\nAnd it MOVES with the nuisance state: re-annealing '
             'alone swings the lifetime minimum by 6 points. No schedule or iteration count fixes '
             'this.', fontsize=10.0)
fig.tight_layout(rect=[0, 0, 1, .90])
os.makedirs(OUT, exist_ok=True)
fig.savefig(f'{OUT}/fig66_objective_bias.png'); plt.close(fig)
print('wrote', f'{OUT}/fig66_objective_bias.png')
