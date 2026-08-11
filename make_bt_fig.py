"""Fig 54 — the BT arms at the full n = 6.

Supersedes the n = 4 preliminary read, and CORRECTS it: that version reported a longitudinal
diffusion "regression" (BT0 -0.60% -> BT2 +7.32%). That was a SIGNED comparison of a parameter
whose per-seed errors change sign. In |error| the paired difference is -0.50 +- 3.65 points, i.e.
nothing at all. The regression is withdrawn.

Signed means are quoted only where the sign is stable (A_b, E field). For longitudinal diffusion,
transverse diffusion and lifetime the per-seed errors mix sign, so |error| is the honest statistic
and is what panels (b), (e) and (g) use.

Three PRODUCTION arms (geometry fitted), 10 000 iterations, differing only in the dE/dx prior:

  BT0   current shape, w = 5,       mean weight 1e5     (control, on this tree)
  BT1   current shape, w = 1/dx,    mean weight 2500    (correct WEIGHT, original SHAPE)
  BT2   REFIT shape,   w = 1/dx,    mean weight 2500    (the headline)

All comparisons PAIRED: the three arms share seeds and --seed draws the target.

WHY THE SPLIT INTO BT1 AND BT2 EARNED ITS SIX JOBS. Correcting the weight ALONE is not merely a
smaller improvement -- it is actively harmful. BT1 recovers the dE/dx spread almost perfectly (0.93
of truth, against BT0's 0.31) and simultaneously makes the MEAN worse (-1.02% against BT0's
-0.45%), because at beta = 1 the ORIGINAL shape is wider than truth on this grid and the softened
mean constraint can no longer hold the scale. The refit fixes the mean (-0.21%) at a more modest
spread (0.68). Since the mean is the moment that couples to A_b and lifetime, that difference is
the whole ballgame.

The signed mean of a noisy parameter is a trap here: BT0's longitudinal diffusion averages to
-0.60% only because its per-seed errors cancel (-0.39, +10.79, -6.36, -6.45). Its typical |error|
is 6.0%. Panels (b) and (e) therefore show |error| for that parameter.
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
SEEDS = sorted(int(re.search(r'seed(\d+)', f).group(1))
               for f in glob.glob('fit_result/sci_full_BT0/history_iter10000_*.pkl'))
ARMS = [('BT0', 'BT0  current, w=5, mean 1e5', C['ink']),
        ('BT1', 'BT1  current, w=1/dx, mean 2500', C['orange']),
        ('BT2', 'BT2  REFIT, w=1/dx, mean 2500', C['green'])]
PL = {'Ab': 'A$_b$', 'eField': 'E field', 'lifetime': 'lifetime',
      'tran_diff': 'tran. diff.', 'long_diff': 'long. diff.'}


def rw(v):
    lo, hi = np.percentile(v, [15.865, 84.135])
    return (hi - lo) / 2


V = {}
for tag, _, _ in ARMS:
    V[tag] = {}
    for s in SEEDS:
        g = glob.glob(f'fit_result/sci_full_{tag}/history_iter10000_*seed{s}.pkl')
        if not g:
            continue
        f = g[0]
        h = pickle.load(open(f, 'rb'))
        d = {}
        for p in R.P:
            v = np.ravel(h[p + '_iter']); t = float(np.ravel(h[p + '_target'])[0])
            d[p] = 100 * (np.median(v[int(len(v) * .8):]) / t - 1)
        pr = np.ravel(h['pos_residual_iter']) * 1e4
        d['pos'] = float(np.median(pr[int(len(pr) * .8):]))
        a = np.asarray([x for x in h.get('dedx_mae_truth_iter', []) if x == x], float)
        d['mae'] = float(a[-50:].mean())
        dc, tm, sd = h['dedx_cache'], h['true_dedx_matched_cache'], h['seg_dx_cache']
        F, T, D = [], [], []
        for b in sorted(dc):
            fv = np.exp(np.asarray(dc[b]['log_dedx'], float))
            tv = np.asarray(tm[b], float); dv = np.asarray(sd[b], float)
            m = min(len(fv), len(tv), len(dv)); fv, tv, dv = fv[:m], tv[:m], dv[:m]
            ok = np.isfinite(tv) & (tv > 0)
            F.append(fv[ok]); T.append(tv[ok]); D.append(dv[ok])
        F, T, D = np.concatenate(F), np.concatenate(T), np.concatenate(D)
        d['mean_err'] = 100 * ((np.sum(F * D) / np.sum(D)) / (np.sum(T * D) / np.sum(D)) - 1)
        d['wratio'] = rw(F) / rw(T)
        d['tau'] = float(np.ravel(h['lifetime_target'])[0])
        d['swing'] = 100 * (1 - np.exp(-0.087 * 2200.0 / d['tau']))
        V[tag][s] = d

fig, ax = plt.subplots(2, 3, figsize=(16.4, 8.8))
xs = np.arange(len(R.P))

# ---------------- (a) absolute calibration ----------------
a = ax[0, 0]
for k, (t, lab, col) in enumerate(ARMS):
    m = [np.mean([V[t][s][p] for s in SEEDS]) for p in R.P]
    e = [np.std([V[t][s][p] for s in SEEDS], ddof=1) / np.sqrt(len(SEEDS)) for p in R.P]
    a.errorbar(xs + (k - 1) * .19, m, yerr=e, lw=0, elinewidth=2.2, marker='o', ms=7, capsize=4,
               color=col, label=lab)
a.axhline(0, color=C['grey'], lw=1.4)
a.axhspan(-1, 1, color=C['green'], alpha=.10)
a.set_xticks(xs); a.set_xticklabels([PL[p] for p in R.P], fontsize=8, rotation=18, ha='right')
a.set_ylabel('error vs truth (%)')
a.set_title('(a) absolute, signed. A$_b$ improves with the REFIT.\n'
            'Mixed-sign parameters are read in |error| in (b).', fontsize=9.3)
a.legend(fontsize=6.9, frameon=False)

# ---------------- (b) paired differences ----------------
a = ax[0, 1]
PAIRS = [('BT0', 'BT1', C['orange'], 'weight only  (BT0→BT1)'),
         ('BT1', 'BT2', C['green'], 'refit on top  (BT1→BT2)'),
         ('BT0', 'BT2', C['blue'], 'the full change  (BT0→BT2)')]
KEYS = ['Ab', 'tran_diff', 'long_diff', 'lifetime']
xs2 = np.arange(len(KEYS))
for k, (u, w, col, lab) in enumerate(PAIRS):
    m, e = [], []
    for p in KEYS:
        f = abs if p in ('long_diff', 'tran_diff', 'lifetime') else (lambda x: x)
        d = np.array([f(V[w][s][p]) - f(V[u][s][p]) for s in SEEDS])
        m.append(d.mean()); e.append(d.std(ddof=1) / np.sqrt(len(d)))
    a.errorbar(xs2 + (k - 1) * .19, m, yerr=e, lw=0, elinewidth=2.4, marker='o', ms=7, capsize=4,
               color=col, label=lab)
a.axhline(0, color=C['ink'], lw=1.5)
a.set_xticks(xs2)
a.set_xticklabels([PL[p] + ('\n(|error|)' if p in ('long_diff', 'tran_diff', 'lifetime')
                                    else '') for p in KEYS], fontsize=7.8, rotation=12, ha='right')
a.set_ylabel('PAIRED difference (points).  negative = better')
a.set_title('(b) BT2 improves or matches EVERY parameter;\n'
            'BT1 degrades long. diffusion by +11.1 ± 2.0 (0/6 seeds)', fontsize=9.3)
a.legend(fontsize=7.0, frameon=False)

# ---------------- (c) A_b per seed: the clean result ----------------
a = ax[0, 2]
for k, (t, lab, col) in enumerate(ARMS):
    a.plot(SEEDS, [V[t][s]['Ab'] for s in SEEDS], marker='o', ms=8, lw=1.8, color=col, label=lab)
a.axhline(0, color=C['red'], lw=1.4, ls='--')
a.set_xticks(SEEDS); a.set_xlabel('seed')
a.set_ylabel('A$_b$ error vs truth (%)')
a.set_title('(c) A$_b$: BT2 is best on ALL SIX seeds.\n'
            'Paired −0.50 ± 0.09 vs BT0 (5.6σ).', fontsize=9.3)
a.legend(fontsize=7.0, frameon=False)

# ---------------- (d) the dE/dx field: the mechanism check ----------------
a = ax[1, 0]
xs3 = np.arange(len(ARMS))
me = [np.mean([V[t][s]['mean_err'] for s in SEEDS]) for t, _, _ in ARMS]
ms = [np.std([V[t][s]['mean_err'] for s in SEEDS], ddof=1) for t, _, _ in ARMS]
wr = [np.mean([V[t][s]['wratio'] for s in SEEDS]) for t, _, _ in ARMS]
ws = [np.std([V[t][s]['wratio'] for s in SEEDS], ddof=1) for t, _, _ in ARMS]
a.bar(xs3 - .19, [abs(x) for x in me], yerr=ms, width=.36, color=C['orange'], edgecolor='white',
      capsize=4, label='|mean error| vs truth (%)')
a2 = a.twinx()
a2.bar(xs3 + .19, wr, yerr=ws, width=.36, color=C['green'], edgecolor='white', capsize=4,
       label='spread recovered / true')
a2.axhline(1.0, color=C['ink'], lw=1.5, ls='--'); a2.set_ylim(0, 1.15); a2.grid(False)
a2.set_ylabel('fraction of the true spread recovered')
for k in range(len(ARMS)):
    a.annotate(f'{abs(me[k]):.2f}%', (k - .19, abs(me[k])), textcoords='offset points',
               xytext=(0, 4), ha='center', fontsize=7.4)
    a2.annotate(f'{wr[k]:.2f}', (k + .19, wr[k]), textcoords='offset points', xytext=(0, 4),
                ha='center', fontsize=7.4)
a.set_xticks(xs3); a.set_xticklabels([t for t, _, _ in ARMS], fontsize=9)
a.set_ylabel('|dE/dx mean error| (%)')
a.set_title('(d) THE MECHANISM. BT1 fixes the SPREAD and breaks\n'
            'the MEAN; only the refit fixes both.', fontsize=9.3)
h1, l1 = a.get_legend_handles_labels(); h2, l2 = a2.get_legend_handles_labels()
a.legend(h1 + h2, l1 + l2, fontsize=7.0, frameon=False, loc='upper center')

# ---------------- (e) lifetime, stratified by lever arm ----------------
# Pooling lifetime across seeds mixes problems with 4.2% and 15.2% of signal (sec 6t), so it is
# quoted stratified. This is the panel n = 4 could not produce.
a = ax[1, 1]
HI = [s for s in SEEDS if V['BT0'][s]['swing'] > 6.5]
LO = [s for s in SEEDS if V['BT0'][s]['swing'] <= 6.5]
xs4 = np.arange(2)
for k, (t, lab, col) in enumerate(ARMS):
    m = [np.mean([abs(V[t][s]['lifetime']) for s in HI]),
         np.mean([abs(V[t][s]['lifetime']) for s in LO])]
    e = [np.std([abs(V[t][s]['lifetime']) for s in HI], ddof=1) / np.sqrt(len(HI)),
         np.std([abs(V[t][s]['lifetime']) for s in LO], ddof=1) / np.sqrt(len(LO))]
    a.bar(xs4 + (k - 1) * .26, m, yerr=e, width=.24, color=col, edgecolor='white', capsize=4,
          label=t)
    for j, y in enumerate(m):
        a.text(j + (k - 1) * .26, y + .25, f'{y:.2f}', ha='center', fontsize=7.4)
a.set_xticks(xs4)
a.set_xticklabels([f'lever arm > 6.5%\n(seeds {HI})', f'lever arm < 6.5%\n(seeds {LO})'],
                  fontsize=7.6)
a.set_ylabel('mean |lifetime error| (%)')
a.set_title('(e) LIFETIME: 5.97 → 1.62% overall, and BT2 is better in\n'
            'BOTH lever-arm strata — the panel n = 4 could not produce.', fontsize=9.3)
a.legend(fontsize=7.4, frameon=False)

# ---------------- (f) geometry and the dE/dx metric ----------------
a = ax[1, 2]
pos = [np.mean([V[t][s]['pos'] for s in SEEDS]) for t, _, _ in ARMS]
pose = [np.std([V[t][s]['pos'] for s in SEEDS], ddof=1) / np.sqrt(len(SEEDS)) for t, _, _ in ARMS]
mae = [np.mean([V[t][s]['mae'] for s in SEEDS]) for t, _, _ in ARMS]
maee = [np.std([V[t][s]['mae'] for s in SEEDS], ddof=1) / np.sqrt(len(SEEDS)) for t, _, _ in ARMS]
a.bar(xs3 - .19, pos, yerr=pose, width=.36, color=C['purple'], edgecolor='white', capsize=4,
      label='position residual (µm)')
a3 = a.twinx()
a3.bar(xs3 + .19, mae, yerr=maee, width=.36, color=C['blue'], edgecolor='white', capsize=4,
       label='dE/dx MAE vs TRUTH')
a3.set_ylabel('per-segment dE/dx MAE vs truth'); a3.grid(False)
for k in range(len(ARMS)):
    a.annotate(f'{pos[k]:.0f}', (k - .19, pos[k]), textcoords='offset points', xytext=(0, 4),
               ha='center', fontsize=7.4)
    a3.annotate(f'{mae[k]:.4f}', (k + .19, mae[k]), textcoords='offset points', xytext=(0, 4),
                ha='center', fontsize=7.4)
a.set_xticks(xs3); a.set_xticklabels([t for t, _, _ in ARMS], fontsize=9)
a.set_ylabel('position residual (µm)'); a.set_ylim(0, max(pos) * 1.35)
a.set_title('(f) geometry costs ~13 µm; the dE/dx MAE against TRUTH\n'
            'is best for BT2 and worst for BT1.', fontsize=9.3)
h1, l1 = a.get_legend_handles_labels(); h3, l3 = a3.get_legend_handles_labels()
a.legend(h1 + h3, l1 + l3, fontsize=7.0, frameon=False, loc='lower center')

fig.suptitle('Fig 54 — the BT arms at n = 6. The refit prior delivers what fig 48 predicted: '
             'A$_b$ 0.77 → 0.27% (6/6 seeds, 5.6σ),\nlifetime |error| 5.97 → 1.62%, and the dE/dx '
             'mean −0.45 → −0.20%. Correcting the WEIGHT ALONE remains actively harmful (d).\n'
             'The longitudinal-diffusion "regression" of the n = 4 read was a signed-mean artefact '
             'and is WITHDRAWN.', fontsize=10.0)
fig.tight_layout(rect=[0, 0, 1, .90])
os.makedirs(OUT, exist_ok=True)
fig.savefig(f'{OUT}/fig54_bt_arms.png'); plt.close(fig)
print('wrote', f'{OUT}/fig54_bt_arms.png')
print(f'\nn={len(SEEDS)} seeds {SEEDS}')
print(f'{"arm":5s}' + ''.join(f'{p:>11s}' for p in R.P) + f'{"pos":>8s}{"MAEtru":>9s}{"dEdx mean":>11s}{"spread":>8s}')
for t, _, _ in ARMS:
    print(f'{t:5s}' + ''.join(f'{np.mean([V[t][s][p] for s in SEEDS]):+11.2f}' for p in R.P)
          + f'{np.mean([V[t][s]["pos"] for s in SEEDS]):8.0f}'
          + f'{np.mean([V[t][s]["mae"] for s in SEEDS]):9.4f}'
          + f'{np.mean([V[t][s]["mean_err"] for s in SEEDS]):+11.3f}'
          + f'{np.mean([V[t][s]["wratio"] for s in SEEDS]):8.3f}')
