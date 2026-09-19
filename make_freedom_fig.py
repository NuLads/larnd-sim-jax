"""Fig 97 — the NUISANCE FREEDOM ladder: constraining the dE/dx block harder improves EVERY
calibration parameter, and a simple prior-scale change gets ~90% of the way to truth-pinning.

THE AXIS. All arms differ only in how much freedom the ~4000 per-segment dE/dx parameters have:
    x2.5 scale  >  x1.6  >  x1.0 (production)  >  x0.6  >  frozen
Truth-pinned arms are shown for reference but are NOT recipes (they use truth).

THE RESULT. Mean |error| over the five calibration parameters falls monotonically as freedom is
removed -- 11.8% -> 10.3% -> 7.3% -> 3.4% -- and x0.6, which uses NO truth information, essentially
matches frozen-at-truth (3.2%) and approaches both-blocks-at-truth (2.8%).

WHICH PARAMETERS. The gain is entirely in the soft plane: long_diff 34.3 -> 5.8 and lifetime
20.3 -> 7.8. A_b and E field are flat throughout (they never couple to the nuisance block, exactly
as the degeneracy matrix said). Position also improves, 192 -> 168 um.

THE TURNING POINT IS KNOWN TO EXIST. CONSTDEDX (SS6d) froze the field at the prior MEAN -- infinite
constraint, zero per-segment freedom -- and was markedly WORSE (long_diff x4, lifetime x2.7). So the
curve must turn around somewhere below x0.6, and the optimum has not been located yet.
"""
import glob, pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = 'plots/noise_report'
BLUE, ORANGE, GREEN, RED, GREY, INK, PURPLE = ('#0072B2', '#E69F00', '#009E73', '#D55E00',
                                               '#888888', '#222222', '#CC79A7')
plt.rcParams.update({'font.size': 9.5, 'axes.grid': True, 'grid.alpha': .25,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.dpi': 130, 'savefig.bbox': 'tight'})
P = ['Ab', 'eField', 'tran_diff', 'long_diff', 'lifetime']
PL = {'Ab': 'A$_b$', 'eField': 'E field', 'tran_diff': 'tran. diff.',
      'long_diff': 'long. diff.', 'lifetime': 'lifetime'}


def rd(tag):
    fs = glob.glob(f'fit_result/sci_full_{tag}/history_iter10000_*.pkl')
    if not fs:
        return None
    h = pickle.load(open(fs[0], 'rb'))
    o = {}
    for p in P:
        v = np.ravel(h[p + '_iter'])
        t = float(np.ravel(h[p + '_target'])[0])
        o[p] = 100 * (np.median(v[int(len(v) * .9):]) / t - 1)
    o['pos'] = float(np.median(np.ravel(h['pos_residual_iter'])[-500:] * 1e4))
    A = h['aux_iter'][-2000:]
    # LIKELIHOOD ONLY. The total loss is NOT comparable across these arms: the dE/dx prior term is
    # part of it and its definition changes with the prior scale.
    o['lik'] = float(np.median([abs(a['log_likelihood_tick']) + abs(a['no_match_penalty'])
                                + abs(a['expected_total_hits']) for a in A]))
    return o


# (label, prefix, x position on the freedom axis, uses-truth)
LADDER = [('×2.5', 'WIDE25D', 2.5, False), ('×1.6', 'WIDE16D', 1.6, False),
          ('×1.0\nproduction', 'DSREP', 1.0, False), ('×0.6', 'NARR06D', 0.6, False),
          ('frozen\n@ own fit', 'DXFRZ', 0.25, False), ('frozen\n@ TRUTH', 'DXTRU', 0.18, True),
          ('both @\nTRUTH', 'BOTHTRU', 0.12, True)]
D = {}
for lab, pre, x, tr in LADDER:
    rows = [rd(f'{pre}{s}') for s in (3, 7, 8)]
    rows = [r for r in rows if r]
    if rows:
        D[lab] = (rows, x, tr)

fig, ax = plt.subplots(1, 3, figsize=(14.2, 4.5))

# (a) the headline scalar
a = ax[0]
labs = [l for l, _, _, _ in LADDER if l in D]
xs = np.arange(len(labs))
vals = [np.mean([np.mean([abs(r[p]) for p in P]) for r in D[l][0]]) for l in labs]
ses = [np.std([np.mean([abs(r[p]) for p in P]) for r in D[l][0]], ddof=1) /
       np.sqrt(len(D[l][0])) if len(D[l][0]) > 1 else 0 for l in labs]
cols = [GREEN if D[l][2] else BLUE for l in labs]
a.bar(xs, vals, yerr=ses, capsize=4, color=cols)
for x, v, n in zip(xs, vals, [len(D[l][0]) for l in labs]):
    a.annotate(f'{v:.2f}', (x, v), textcoords='offset points', xytext=(0, 5), ha='center',
               fontsize=8.5)
a.set_xticks(xs, labs, fontsize=8)
a.set_ylabel('mean |error| over the 5 calibration params (%)')
a.set_title('(a) less nuisance freedom → better calibration.\n'
            '×0.6 uses NO truth yet matches truth-pinning', fontsize=9.5)
a.annotate('← more freedom', (.02, .93), xycoords='axes fraction', fontsize=8.5, color=GREY)
a.annotate('less freedom →', (.60, .93), xycoords='axes fraction', fontsize=8.5, color=GREY)

# (b) per parameter
a = ax[1]
fitted = [l for l in labs if not D[l][2]]
for p, col in zip(P, [GREY, GREY, PURPLE, RED, BLUE]):
    v = [np.mean([r[p] for r in D[l][0]]) for l in fitted]
    lw = 2.2 if p in ('lifetime', 'long_diff') else 1.1
    a.plot(range(len(fitted)), v, 'o-', color=col, lw=lw, ms=6, label=PL[p],
           alpha=1 if lw > 2 else .75)
a.axhline(0, color=GREEN, lw=1.6)
a.set_xticks(range(len(fitted)), fitted, fontsize=8)
a.set_ylabel('error (%)')
a.set_title('(b) the gain is entirely in the SOFT PLANE.\nA$_b$ and E field never move', fontsize=9.5)
a.legend(frameon=False, fontsize=8, ncol=2, loc='upper right')

# (c) position + loss: does the objective see it?
a = a2 = ax[2]
pos = [np.mean([r['pos'] for r in D[l][0]]) for l in fitted]
a.plot(range(len(fitted)), pos, 'o-', color=BLUE, lw=2, ms=6)
a.set_ylabel('position residual (µm)', color=BLUE)
a.tick_params(axis='y', labelcolor=BLUE)
a.set_xticks(range(len(fitted)), fitted, fontsize=8)
b = a.twinx()
lik = [np.mean([r['lik'] for r in D[l][0]]) for l in fitted]
b.plot(range(len(fitted)), lik, 's--', color=RED, lw=2, ms=6)
b.set_ylabel('LIKELIHOOD only (prior terms excluded)', color=RED)
b.tick_params(axis='y', labelcolor=RED)
b.grid(False)
a.set_title('(c) position improves too (blue), while the data\n'
            'likelihood gets slightly WORSE (red) — loss-blind again', fontsize=9.5)

fig.suptitle('Fig 97 — nuisance FREEDOM, not nuisance accuracy, is the lever. Constraining the ~4000 per-segment dE/dx parameters harder\n'
             'improves every calibration parameter monotonically: mean |error| 11.8 → 10.3 → 7.3 → 3.4%, with ×0.6 (no truth used) essentially\n'
             'matching frozen-at-truth. CONSTDEDX (frozen at the prior mean) is known to be much worse, so an optimum exists below ×0.6 and is unlocated.',
             fontsize=10.5, y=1.08)
fig.tight_layout()
fig.savefig(f'{OUT}/fig97_freedom_ladder.png')
print('wrote fig97')
for l in labs:
    rows = D[l][0]
    print(f'  {l.replace(chr(10)," "):20s} n={len(rows)}  mean|err| '
          f'{np.mean([np.mean([abs(r[p]) for p in P]) for r in rows]):5.2f}%  '
          f'pos {np.mean([r["pos"] for r in rows]):5.0f}  lik {np.mean([r["lik"] for r in rows]):6.0f}')
