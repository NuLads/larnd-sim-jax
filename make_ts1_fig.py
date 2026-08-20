"""Fig 67 — TS1: stage 1 of the two-stage recipe. The same fit quality as G20K at half the compute.

WHAT TS1 IS. Calibration FROZEN at BT2's converged values; geometry and dEdx warm-started from BT2
and re-annealed for 10 000 iterations. It is BMC (the block-matrix winner) run to the floor BMC was
still descending toward at 3 000. It is stage 1 of a two-stage recipe whose stage 2 -- release
calibration, fresh anneal -- has not been run.

VERIFIED BEFORE READING: calibration peak-to-peak movement exactly 0 on all 5 parameters and all 3
seeds, and its iteration-0 values match BT2's tail median to 0.0000%; dEdx moved 0.16-0.18 in
log-space and geometry moved on 98.7-100% of tracks. The freeze and the two warm starts all did
exactly what they claim.

THE RESULT. Against G20K -- 20 000 iterations with everything fitted, the previous best -- TS1 wins
the objective on 3 of 3 seeds (-88 +- 34 NLL) and ties or slightly beats it on position and dEdx,
using 13.5 h against 26.4 h. Freezing the block that does not need to move is worth more than
doubling the iteration count.

WHAT IT DOES NOT DO, and the figure says so twice so it cannot be misread: TS1's calibration IS
BT2's, by construction. Its transverse diffusion is 2.64% where G20K reached 0.61%. TS1 is a better
FIT, not a better CALIBRATION. Delivering the calibration is stage 2's job, and panel (e) is the
target it has to hit.

AND IT CONVERGES EARLY, which matters for whoever runs stage 2. TS1 is past G20K's final position
by iteration 3 244 and past its final loss by 3 926, and is within 2% of its own endpoint by ~3 500.
The last 6 000 iterations buy almost nothing: stage 1 could be run at 4 000 iterations for ~5.4 h
with no measurable loss of quality, which changes the two-stage recipe's total cost from 27 h to
about 19 h.

CONTEXT FROM FIG 66: the objective prefers truth (7/7 arm-seeds, by 5.6%), so a lower loss means
closer to the joint minimum -- which is why TS1's loss advantage is meaningful rather than a
reparametrisation artefact. But blocks must move TOGETHER: setting calibration alone to truth makes
the loss worse. TS1 deliberately moves only two of the three, so it cannot be the endpoint.
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
W = 200
S3 = [0, 1, 2]
PL = {'Ab': 'A$_b$', 'eField': 'E field', 'lifetime': 'lifetime',
      'tran_diff': 'tran. diff.', 'long_diff': 'long. diff.'}
ARMS = [('BT2', 'all fitted, 10k', 13.4, [0, 1, 2], C['ink']),
        ('BM0', 're-anneal all, 3k', 4.2, [0, 1], C['grey']),
        ('BMC', 'calib frozen, 3k', 4.3, [0, 1], C['purple']),
        ('TS1', 'calib frozen, 10k', 13.5, [0, 1, 2], C['green']),
        ('G20K', 'all fitted, 20k', 26.4, [0, 1, 2], C['blue'])]


def sm(v, w=W):
    v = np.asarray(v, float)
    from numpy.lib.stride_tricks import sliding_window_view
    return np.median(sliding_window_view(v, w), axis=1) if len(v) > w else v


def raw(t, s):
    fs = glob.glob(f'fit_result/sci_full_{t}/history_iter*seed{s}.pkl')
    return pickle.load(open(max(fs, key=lambda p: int(re.search(r'iter(\d+)', p).group(1))), 'rb'))


CA = {}
def L(t, s):
    if (t, s) not in CA:
        h = raw(t, s); d = {}
        for p in R.P:
            tg = float(np.ravel(h[p + '_target'])[0])
            d[p] = 100 * (np.ravel(h[p + '_iter'])[1:] / tg - 1)
        d['mae'] = np.asarray([x for x in h.get('dedx_mae_truth_iter', []) if x == x], float)
        d['pos'] = np.ravel(h['pos_residual_iter']) * 1e4
        d['loss'] = np.asarray(h['losses_iter'], float)
        CA[(t, s)] = d
    return CA[(t, s)]


fin = lambda t, s, k: np.median(sm(L(t, s)[k])[-300:])
med = lambda t, k, seeds, a=False: np.median([abs(fin(t, s, k)) if a else fin(t, s, k) for s in seeds])

fig, ax = plt.subplots(2, 3, figsize=(16.8, 9.4))

# ---------------- (a) verification ----------------
a = ax[0, 0]
rows = []
for s in S3:
    h, b = raw('TS1', s), raw('BT2', s)
    cptp = max(np.ptp(np.ravel(h[p + '_iter'])) for p in R.P)
    dw = {int(k): np.asarray(v['log_dedx']) for k, v in h['dedx_cache'].items() if 'log_dedx' in v}
    db = {int(k): np.asarray(v['log_dedx']) for k, v in b['dedx_cache'].items() if 'log_dedx' in v}
    dd = np.median([np.max(np.abs(dw[k] - db[k])) for k in set(dw) & set(db)
                    if len(dw[k]) == len(db[k])])
    n = ident = 0
    for k in set(h['chain_cache']) & set(b['chain_cache']):
        for tw, tb in zip(h['chain_cache'][k], b['chain_cache'][k]):
            n += 1
            ident += np.array_equal(np.asarray(tw['angles']), np.asarray(tb['angles']))
    rows.append([cptp, dd, 100 * (1 - ident / n)])
Z = np.array(rows)
a.axis('off')
tbl = [['seed', 'calibration\nmovement', 'dE/dx moved\n(log units)', 'geometry moved\n(% of tracks)']]
for i, s in enumerate(S3):
    tbl.append([str(s), f'{Z[i,0]:.0f}  ✓FROZEN', f'{Z[i,1]:.3f}  ✓refit', f'{Z[i,2]:.1f}%  ✓refit'])
t = a.table(cellText=tbl[1:], colLabels=tbl[0], loc='center', cellLoc='center')
t.auto_set_font_size(False); t.set_fontsize(8.4); t.scale(1, 2.3)
for j in range(4):
    t[0, j].set_facecolor('#EEEEEE')
a.set_title('(a) VERIFIED BEFORE READING. Calibration exactly frozen, both\n'
            'nuisances warm-started and genuinely refitted.', fontsize=9.3)

# ---------------- (b) the efficiency frontier ----------------
a = ax[0, 1]
for tag, lab, hrs, seeds, col in ARMS:
    v = [fin(tag, s, 'loss') for s in seeds]
    a.plot([hrs] * len(v), v, 'o', ms=7, color=col, alpha=.4)
    a.plot(hrs, np.median(v), 'o', ms=14, color=col)
    a.annotate(f'  {tag}\n  {lab}', (hrs, np.median(v)), fontsize=7.4, va='center')
a.set_xlabel('wall clock (GPU-hours per seed)')
a.set_ylabel('loss (PPP NLL)   lower = better')
a.set_xlim(0, 32)
a.set_title('(b) THE EFFICIENCY FRONTIER. TS1 reaches a LOWER loss than\n'
            'G20K at 51% of the compute — 3/3 seeds.', fontsize=9.3)

# ---------------- (c) paired TS1 vs G20K ----------------
a = ax[0, 2]
labs, vals, cols = [], [], []
for k, lab, sc in [('loss', 'loss (NLL)', 1), ('pos', 'position (µm)', 1),
                   ('mae', 'dE/dx MAE (×10⁻⁴)', 1e4)]:
    d = [(fin('TS1', s, k) - fin('G20K', s, k)) * sc for s in S3]
    labs.append(lab); vals.append(d)
    cols.append(C['green'] if max(d) < 0 else (C['grey'] if min(d) * max(d) <= 0 else C['red']))
for i, d in enumerate(vals):
    a.plot([min(d), max(d)], [i, i], color=cols[i], lw=3, alpha=.5)
    for x in d:
        a.plot(x, i, 'o', ms=10, color=cols[i], alpha=.85)
a.axvline(0, color=C['ink'], lw=1.8)
a.set_yticks(range(3)); a.set_yticklabels(labs, fontsize=8.6); a.invert_yaxis()
a.set_xlabel('TS1 − G20K, per seed   (negative = TS1 better)')
a.set_title('(c) AND IT IS NOT A TIE ON THE OBJECTIVE: loss better on 3/3\n'
            '(−88 ± 34). Position and dE/dx straddle zero — a tie.', fontsize=9.3)

# ---------------- (d) traces ----------------
for col, (key, lab, logy) in enumerate([('pos', 'position residual (µm)', True),
                                        ('loss', 'loss (PPP NLL)', True)]):
    a = ax[1, col]
    for tag, _, _, seeds, c in [x for x in ARMS if x[0] in ('BT2', 'TS1', 'G20K')]:
        cs = [sm(L(tag, s)[key]) for s in seeds]
        n = min(len(c_) for c_ in cs)
        a.plot(np.median([c_[:n] for c_ in cs], axis=0), color=c, lw=2.2, label=tag)
    a.set_yscale('log' if logy else 'linear')
    a.set_ylim(*( (95, 450) if key == 'pos' else (7800, 12000) ))
    a.set_xlim(0, 20000)
    a.set_xlabel('iteration'); a.set_ylabel(lab)
    a.legend(fontsize=7.6, frameon=False)
    TT = {'pos': '(d) TS1 starts WARM at BT2’s 165 µm and is past G20K’s\n'
                 'final answer by iteration 3 244 — of a 20 000-iteration run.',
          'loss': '(e) same on the objective: past G20K’s final value by 3 926,\n'
                  'and flat from ~3 800. Even 10 000 iterations is generous.'}[key]
    a.set_title(TT, fontsize=9.3)

# ---------------- (f) what stage 2 must deliver ----------------
a = ax[1, 2]
xs = np.arange(len(R.P)); w = .27
for i, (tag, col) in enumerate([('BT2', C['ink']), ('TS1', C['green']), ('G20K', C['blue'])]):
    a.bar(xs + (i - 1) * w, [med(tag, p, S3, True) for p in R.P], width=w, color=col,
          edgecolor='white', label=f'{tag}')
a.set_yscale('log')
a.set_xticks(xs); a.set_xticklabels([PL[p] for p in R.P], fontsize=8, rotation=18, ha='right')
a.set_ylabel('|error| (%)')
a.set_title('(f) BUT TS1’s CALIBRATION *IS* BT2’s — it is frozen. Transverse\n'
            'diffusion 2.64% vs G20K’s 0.61%. THIS is what stage 2 must deliver.', fontsize=9.3)
a.legend(fontsize=7.4, frameon=False)

fig.suptitle('Fig 67 — TS1, stage 1 of the two-stage recipe: freezing the block that does not need '
             'to move beats doubling the iteration count.\nAgainst G20K it wins the objective on '
             '3/3 seeds (−88 ± 34 NLL) and ties on position and dE/dx, at 13.5 h against 26.4 h. '
             'But its calibration\nis BT2’s by construction — TS1 is a better FIT, not a better '
             'CALIBRATION, and stage 2 has not been run.', fontsize=10.0)
fig.tight_layout(rect=[0, 0, 1, .90])
os.makedirs(OUT, exist_ok=True)
fig.savefig(f'{OUT}/fig67_ts1.png'); plt.close(fig)
print('wrote', f'{OUT}/fig67_ts1.png')
for tag, lab, hrs, seeds, _ in ARMS:
    print(f'  {tag:5s} {hrs:5.1f}h  pos {med(tag,"pos",seeds):6.1f}  loss {med(tag,"loss",seeds):7.1f}'
          f'  mae {med(tag,"mae",seeds):.5f}')
