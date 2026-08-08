"""Fig 53 (study 3): deliberately mis-setting the dE/dx mean-constraint TARGET.

The loss pins the length-weighted <dE/dx> to `dedx_mean_constraint_target` (1.887) with weight
`dedx_mean_constraint_weight`. If A_b or lifetime are absorbing a mismatch there, mis-setting the
target on purpose must move them in a measurable, monotone way -- and the slope is the lever arm.

Crossed with the WEIGHT on purpose: the target only matters as much as the stiffness lets it. At
1e5 the fit cannot ignore a wrong target; at 3000 it can. The cross gives d(param)/d(target) AS A
FUNCTION of stiffness, which is what says whether the target is a lever or a nuisance.

NOT the same as sci_full_FITDM0/FITDMP2/FITDMM2, which varied the SIM INPUT FILE's dE/dx mean with
the target left at 1.887 in all three. This varies the PENALTY TARGET.

All comparisons are PAIRED: every arm shares the same seeds and --seed draws the target, so the
slope is fitted per seed and then averaged. `S3T099` (target 0.99x) produced no completed runs -- it
was cancelled mid-flight as a redundant 5th scan point -- and `S3T102` has n=5.
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
NOMTGT, TRUEMEAN, T_DRIFT = 1.887, 1.88656, 0.087 * 2200.0
STIFF = [('S3T098', 1.84926), ('S3T100', 1.887), ('S3T101', 1.90587), ('S3T102', 1.92474)]
SOFT = [('S3W098', 1.84926), ('S3W100', 1.887), ('S3W102', 1.92474)]
SETS = [(1e5, STIFF, C['red'], 'weight 1e5  (production)'),
        (3e3, SOFT, C['blue'], 'weight 3000  (≈ the derived 2520)')]
PL = {'Ab': 'A$_b$', 'eField': 'E field', 'lifetime': 'lifetime',
      'tran_diff': 'tran. diff.', 'long_diff': 'long. diff.'}


def load(tag):
    o = {}
    for f in sorted(glob.glob(f'fit_result/sci_ceiling_{tag}/history_iter10000_*.pkl')):
        s = int(re.search(r'seed(\d+)', f).group(1))
        h = pickle.load(open(f, 'rb'))
        d = {}
        for p in R.P:
            v = np.ravel(h[p + '_iter']); t = float(np.ravel(h[p + '_target'])[0])
            d[p] = 100 * (np.median(v[int(len(v) * .8):]) / t - 1)
        md = np.asarray([x for x in h.get('mean_dedx_iter', []) if x == x], float)
        d['md'] = float(np.median(md[int(len(md) * .8):]))
        d['tau'] = float(np.ravel(h['lifetime_target'])[0])
        d['swing'] = 100 * (1 - np.exp(-T_DRIFT / d['tau']))
        o[s] = d
    return o


D = {t: load(t) for t, _ in STIFF + SOFT}
print({t: sorted(D[t]) for t in D})


def slopes(arms):
    tags = [t for t, _ in arms]
    ss = sorted(set.intersection(*[set(D[t]) for t in tags]))
    xp = np.array([100 * (g / NOMTGT - 1) for _, g in arms])
    out = {}
    for p in R.P + ['md']:
        sl = np.array([np.polyfit(xp, [D[t][s][p] for t in tags], 1)[0] for s in ss])
        out[p] = (sl.mean(), sl.std(ddof=1) / np.sqrt(len(sl)))
    return out, ss, xp


SL = {w: slopes(arms)[0] for w, arms, _, _ in SETS}
for w, arms, _, lab in SETS:
    s, ss, xp = slopes(arms)
    print(f'w={w:.0f} seeds {ss}: ' + '  '.join(f'{p}={s[p][0]:+.4f}±{s[p][1]:.4f}' for p in R.P))

fig, ax = plt.subplots(2, 3, figsize=(16.4, 8.8))

# ---------------- (a) A_b: the headline ----------------
a = ax[0, 0]
for w, arms, col, lab in SETS:
    xp = np.array([100 * (g / NOMTGT - 1) for _, g in arms])
    m = [np.mean([D[t][s]['Ab'] for s in D[t]]) for t, _ in arms]
    e = [np.std([D[t][s]['Ab'] for s in D[t]], ddof=1) / np.sqrt(len(D[t])) for t, _ in arms]
    a.errorbar(xp, m, yerr=e, lw=2.0, elinewidth=2.2, marker='o', ms=8, capsize=4, color=col,
               label=f'{lab}\nslope {SL[w]["Ab"][0]:+.3f} ± {SL[w]["Ab"][1]:.3f} pts/%')
a.axvline(0, color=C['grey'], lw=1.4); a.axhline(0, color=C['ink'], lw=1.4)
a.set_xlabel('dE/dx mean TARGET offset from 1.887 (%)')
a.set_ylabel('A$_b$ error vs truth (%)')
a.set_title('(a) A$_b$ ABSORBS the mis-set mean, linearly.\n'
            'Stiffer constraint → 3.2× steeper response.', fontsize=9.3)
a.legend(fontsize=7.0, frameon=False)

# ---------------- (b) all five slopes ----------------
a = ax[0, 1]
xs = np.arange(len(R.P))
for k, (w, arms, col, lab) in enumerate(SETS):
    m = [SL[w][p][0] for p in R.P]
    e = [SL[w][p][1] for p in R.P]
    a.errorbar(xs + (k - .5) * .2, m, yerr=e, lw=0, elinewidth=2.4, marker='o', ms=7, capsize=4,
               color=col, label=lab)
a.axhline(0, color=C['ink'], lw=1.5)
a.set_xticks(xs); a.set_xticklabels([PL[p] for p in R.P], fontsize=8, rotation=18, ha='right')
a.set_ylabel('d(parameter) / d(target %)   [points per %]')
a.set_title('(b) ONLY A$_b$ responds. Lifetime is consistent with\n'
            'zero at both weights — the target is not a lifetime lever.', fontsize=9.3)
a.legend(fontsize=7.2, frameon=False)

# ---------------- (c) does the constraint dictate the mean? ----------------
a = ax[0, 2]
gs = np.linspace(1.84, 1.935, 10)
a.plot(gs, gs, color=C['ink'], lw=1.8, ls='--', label='the fit obeys the target exactly')
for w, arms, col, lab in SETS:
    x = [g for _, g in arms]
    y = [np.mean([D[t][s]['md'] for s in D[t]]) for t, _ in arms]
    frac = SL[w]['md'][0] / (NOMTGT / 100)
    a.plot(x, y, marker='o', ms=9, lw=2.2, color=col, label=f'{lab} — follows {100*frac:.0f}%')
a.axhline(TRUEMEAN, color=C['green'], lw=2.0, label=f'TRUE mean {TRUEMEAN:.4f}')
a.set_xlabel('mean-constraint TARGET'); a.set_ylabel('FITTED length-weighted ⟨dE/dx⟩')
a.set_title('(c) the mechanism: at 1e5 the constraint DICTATES the\n'
            'mean; at 3000 the rest of the objective overrules it.', fontsize=9.3)
a.legend(fontsize=6.9, frameon=False)

# ---------------- (d) lifetime: the null ----------------
a = ax[1, 0]
for w, arms, col, lab in SETS:
    xp = np.array([100 * (g / NOMTGT - 1) for _, g in arms])
    for k, (t, _) in enumerate(arms):
        a.scatter([xp[k]] * len(D[t]), [D[t][s]['lifetime'] for s in D[t]], s=18, color=col, alpha=.45)
    m = [np.mean([D[t][s]['lifetime'] for s in D[t]]) for t, _ in arms]
    e = [np.std([D[t][s]['lifetime'] for s in D[t]], ddof=1) / np.sqrt(len(D[t])) for t, _ in arms]
    a.errorbar(xp, m, yerr=e, lw=2.0, elinewidth=2.4, marker='o', ms=9, capsize=4, color=col,
               label=f'{lab}\nslope {SL[w]["lifetime"][0]:+.2f} ± {SL[w]["lifetime"][1]:.2f}')
a.axvline(0, color=C['grey'], lw=1.4); a.axhline(0, color=C['red'], lw=1.4, ls='--')
a.set_xlabel('target offset (%)'); a.set_ylabel('lifetime error (%)')
a.set_title('(d) lifetime does NOT respond. The hoped-for new handle\n'
            'on our weakest parameter is not there.', fontsize=9.3)
a.legend(fontsize=6.9, frameon=False)

# ---------------- (e) lifetime stratified by lever arm ----------------
a = ax[1, 1]
xs2 = np.arange(2)
for k, (w, arms, col, lab) in enumerate(SETS):
    hi = [abs(D[t][s]['lifetime']) for t, _ in arms for s in D[t] if D[t][s]['swing'] > 6.5]
    lo = [abs(D[t][s]['lifetime']) for t, _ in arms for s in D[t] if D[t][s]['swing'] <= 6.5]
    m = [np.mean(hi), np.mean(lo)]
    e = [np.std(hi, ddof=1) / np.sqrt(len(hi)), np.std(lo, ddof=1) / np.sqrt(len(lo))]
    a.bar(xs2 + (k - .5) * .34, m, yerr=e, width=.32, color=col, edgecolor='white', capsize=4,
          label=f'{lab}  (n={len(hi)}/{len(lo)})')
a.set_xticks(xs2); a.set_xticklabels(['lever arm > 6.5%', 'lever arm < 6.5%'], fontsize=8.4)
a.set_ylabel('mean |lifetime error| (%)')
a.set_title('(e) and the lifetime scatter is the LEVER ARM again,\n'
            'not the weight — the same story as study 1.', fontsize=9.3)
a.legend(fontsize=7.2, frameon=False)

# ---------------- (f) a correction to fig 45(e) ----------------
a = ax[1, 2]
def lt(tag):
    o = load(tag); return o
A, B = lt('CEILP05'), lt('CEILW3KP05')
common = sorted(set(A) & set(B))
groups = [('CEILP05\nw = 1e5\nseeds 0–2', np.std([A[s]['lifetime'] for s in sorted(A)], ddof=1), C['red']),
          ('CEILW3KP05\nw = 3000\nseeds 0–5', np.std([B[s]['lifetime'] for s in sorted(B)], ddof=1), C['blue']),
          ('CEILW3KP05\nw = 3000\nseeds 0–2 ONLY', np.std([B[s]['lifetime'] for s in common], ddof=1), C['green'])]
a.bar(range(3), [g[1] for g in groups], color=[g[2] for g in groups], edgecolor='white')
for k, g in enumerate(groups):
    a.text(k, g[1] + .12, f'{g[1]:.2f}', ha='center', fontsize=8.4)
a.set_xticks(range(3)); a.set_xticklabels([g[0] for g in groups], fontsize=7.2)
a.set_ylabel('lifetime between-seed s.d. (points)')
a.set_title('(f) CORRECTION to fig 45(e): the "7.5× scatter cost of\n'
            'weight 3000" was mostly a different SEED SET.', fontsize=9.3)
a.text(.5, .88, 'the n=3 arm never saw the two low-lever-arm seeds',
       transform=a.transAxes, ha='center', va='top', fontsize=7.2, color=C['grey'])
a.set_ylim(0, 6.6)

fig.suptitle('Fig 53 — study 3: mis-setting the dE/dx mean target. A$_b$ ABSORBS IT LINEARLY '
             '(−0.84 points per % at weight 1e5, 47σ),\nand LIFETIME DOES NOT RESPOND AT ALL at '
             'either weight. The stiffness sets how much: at 1e5 the fitted mean follows 93% of any\n'
             'target error, at 3000 only 30%. The target is an A$_b$ systematic to control, not a '
             'lifetime lever to exploit.', fontsize=10.1)
fig.tight_layout(rect=[0, 0, 1, .90])
os.makedirs(OUT, exist_ok=True)
fig.savefig(f'{OUT}/fig53_study3_meantarget.png'); plt.close(fig)
print('wrote', f'{OUT}/fig53_study3_meantarget.png')
for w, arms, _, lab in SETS:
    f = SL[w]['md'][0] / (NOMTGT / 100)
    print(f'  w={w:.0f}: fitted mean follows {100*f:.0f}% of the target;  '
          f'dAb/dtarget% = {SL[w]["Ab"][0]:+.4f} ± {SL[w]["Ab"][1]:.4f};  '
          f'dlifetime = {SL[w]["lifetime"][0]:+.3f} ± {SL[w]["lifetime"][1]:.3f}')
err = 100 * (NOMTGT / TRUEMEAN - 1)
print(f'  the target is off by {err:+.3f}% -> A_b bias {err*SL[1e5]["Ab"][0]:+.3f} points at w=1e5')
print(f'  to keep |A_b bias| < 0.1 pts the target must be right to {0.1/abs(SL[1e5]["Ab"][0]):.2f}%')
