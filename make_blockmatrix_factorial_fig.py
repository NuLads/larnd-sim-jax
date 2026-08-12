"""Fig 65 — the block matrix read as the FACTORIAL DESIGN it actually is.

Seven arms is six comparisons too many to hold in the head. But the design is not seven arbitrary
configurations: for each block being FITTED, the other two are each either frozen or fitted, which
is a complete 2x2. So every outcome decomposes into two MAIN EFFECTS and one INTERACTION, and that
is three numbers instead of four.

  outcome            2x2 over                arms
  ---------------    --------------------    ------------------------------
  position (G)       freeze C x freeze D     BM0, BMC, BMD, BMCD
  dE/dx MAE (D)      freeze C x freeze G     BM0, BMC, BMG, BMCG
  calibration (C)    freeze D x freeze G     BM0, BMD, BMG, BMDG

MAIN EFFECT of freezing X = mean(arms with X frozen) - mean(arms with X fitted). It answers "what
does pinning block X do to block Y", averaged over whether the third block is pinned -- which is
exactly the question the degeneracy matrix (fig 58) asks of the OBJECTIVE, now asked of the FIT.

INTERACTION = (both frozen) - (only X frozen) - (only Y frozen) + (neither), halved. Near zero means
the two effects simply add and can be reasoned about independently; large means they do not.

EVERY POINT IS PLOTTED PER SEED, and the effects are computed per seed and then averaged, so the
two seeds are visible as independent replicates rather than hidden inside a median of two. With
n = 2 an effect is credible when both seeds agree in SIGN; the bars show the seed spread, not a
standard error, which would be meaningless here.
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
S = [0, 1]
W = 200
PL = {'Ab': 'A$_b$', 'eField': 'E field', 'lifetime': 'lifetime',
      'tran_diff': 'tran. diff.', 'long_diff': 'long. diff.'}


def sm(v, w=W):
    v = np.asarray(v, float)
    from numpy.lib.stride_tricks import sliding_window_view
    return np.median(sliding_window_view(v, w), axis=1) if len(v) > w else v


def load(tag, s):
    fs = glob.glob(f'fit_result/sci_full_{tag}/history_iter*seed{s}.pkl')
    h = pickle.load(open(max(fs, key=lambda p: int(re.search(r'iter(\d+)', p).group(1))), 'rb'))
    d = {}
    for p in R.P:
        t = float(np.ravel(h[p + '_target'])[0])
        d[p] = 100 * (np.ravel(h[p + '_iter'])[1:] / t - 1)
    d['mae'] = np.asarray([x for x in h.get('dedx_mae_truth_iter', []) if x == x], float)
    d['pos'] = np.ravel(h['pos_residual_iter']) * 1e4
    d['loss'] = np.asarray(h['losses_iter'], float)
    return d


TAGS = ['BM0', 'BMC', 'BMD', 'BMG', 'BMCD', 'BMCG', 'BMDG', 'BT2']
D = {t: {s: load(t, s) for s in S} for t in TAGS}
def fin(t, s, k, a=False):
    v = np.median(sm(D[t][s][k])[-300:])
    return abs(v) if a else v


# (outcome key, absolute?, the 2x2 as [neither, X frozen, Y frozen, both], X name, Y name, label)
CELLS = {
    'pos':  (['BM0', 'BMC', 'BMD', 'BMCD'], 'calibration', 'dE/dx', 'position residual (µm)'),
    'mae':  (['BM0', 'BMC', 'BMG', 'BMCG'], 'calibration', 'geometry', 'dE/dx MAE vs truth'),
    'cal':  (['BM0', 'BMD', 'BMG', 'BMDG'], 'dE/dx', 'geometry', 'calibration'),
}


def grid(key, absv=False, param=None):
    """returns arms, and a per-seed 2x2 array [seed][Xfrozen][Yfrozen]"""
    arms = CELLS[key if key in CELLS else 'cal'][0]
    k = param if param else key
    Z = np.zeros((len(S), 2, 2))
    for si, s in enumerate(S):
        Z[si, 0, 0] = fin(arms[0], s, k, absv)
        Z[si, 1, 0] = fin(arms[1], s, k, absv)
        Z[si, 0, 1] = fin(arms[2], s, k, absv)
        Z[si, 1, 1] = fin(arms[3], s, k, absv)
    return arms, Z


def effects(Z):
    """main effect of X, main effect of Y, interaction — per seed"""
    mx = (Z[:, 1, 0] + Z[:, 1, 1]) / 2 - (Z[:, 0, 0] + Z[:, 0, 1]) / 2
    my = (Z[:, 0, 1] + Z[:, 1, 1]) / 2 - (Z[:, 0, 0] + Z[:, 1, 0]) / 2
    it = (Z[:, 1, 1] - Z[:, 1, 0] - Z[:, 0, 1] + Z[:, 0, 0]) / 2
    return mx, my, it


fig, ax = plt.subplots(2, 3, figsize=(16.8, 9.4))

# ---------------- row 1: the three 2x2 tables ----------------
for col, (key, absv, param, fmt, ttl) in enumerate([
        ('pos', False, None, '{:.1f}', 'position residual (µm) — the GEOMETRY block'),
        ('mae', False, None, '{:.4f}', 'dE/dx MAE vs truth — the dE/dx block'),
        ('cal', True, 'lifetime', '{:.2f}', '|lifetime error| (%) — the CALIBRATION block')]):
    a = ax[0, col]
    arms, Z = grid(key, absv, param)
    M = Z.mean(axis=0)
    xn, yn = CELLS[key][1], CELLS[key][2]
    lo, hi = M.min(), M.max()
    a.imshow(M, cmap='RdYlGn_r', vmin=lo - .15 * (hi - lo), vmax=hi + .15 * (hi - lo), aspect='auto')
    names = [[arms[0], arms[2]], [arms[1], arms[3]]]
    for i in range(2):
        for j in range(2):
            a.text(j, i - .16, names[i][j], ha='center', fontsize=8.6, fontweight='bold')
            a.text(j, i + .10, fmt.format(M[i, j]), ha='center', fontsize=10.5)
            a.text(j, i + .30, '  '.join(fmt.format(Z[s, i, j]) for s in S), ha='center',
                   fontsize=6.8, color='#444444')
    a.set_xticks([0, 1]); a.set_xticklabels([f'{yn}\nFITTED', f'{yn}\nFROZEN'], fontsize=8.2)
    a.set_yticks([0, 1]); a.set_yticklabels([f'{xn}\nFITTED', f'{xn}\nFROZEN'], fontsize=8.2)
    a.grid(False)
    a.set_title(f'({"abc"[col]}) {ttl}\nlower is better; small grey = the two seeds', fontsize=9.2)

# ---------------- (d) main effects on the two nuisance blocks ----------------
a = ax[1, 0]
rows, vals, cols_ = [], [], []
for key, absv, param, scale, lab in [('pos', False, None, 1, 'position (µm)'),
                                     ('mae', False, None, 1e4, 'dE/dx MAE (×10⁻⁴)')]:
    arms, Z = grid(key, absv, param)
    mx, my, it = effects(Z)
    for nm, e in [(f'freeze {CELLS[key][1]}', mx), (f'freeze {CELLS[key][2]}', my),
                  ('interaction', it)]:
        rows.append(f'{lab}\n{nm}'); vals.append(e * scale)
        cols_.append(C['grey'] if min(e) * max(e) <= 0 else
                     (C['green'] if np.mean(e) < 0 else C['red']))
ys = np.arange(len(rows))
for i, e in enumerate(vals):
    a.plot([min(e), max(e)], [i, i], color=cols_[i], lw=3, alpha=.55)
    for s in S:
        a.plot(e[s], i, 'o', ms=9, color=cols_[i], alpha=.9 if s == 0 else .5)
a.axvline(0, color=C['ink'], lw=1.6)
a.set_yticks(ys); a.set_yticklabels(rows, fontsize=7.4); a.invert_yaxis()
a.set_xlabel('main effect (negative = freezing it HELPS)')
a.set_title('(d) NUISANCE BLOCKS: freezing CALIBRATION is the big lever and\n'
            'helps both, on both seeds. The other effects are ≲⅓ the size.', fontsize=9.2)

# ---------------- (e) main effects on calibration ----------------
a = ax[1, 1]
xs = np.arange(len(R.P)); w = .38
for i, (which, col, lab) in enumerate([(0, C['blue'], 'freeze dE/dx'),
                                       (1, C['orange'], 'freeze geometry')]):
    for pi, p in enumerate(R.P):
        arms, Z = grid('cal', True, p)
        e = effects(Z)[which]
        x = pi + (i - .5) * w
        a.plot([x, x], [min(e), max(e)], color=col, lw=3, alpha=.5)
        for s in S:
            a.plot(x, e[s], 'o', ms=8, color=col, alpha=.9 if s == 0 else .5,
                   label=lab if (pi == 0 and s == 0) else None)
a.axhline(0, color=C['ink'], lw=1.6)
a.set_xticks(xs); a.set_xticklabels([PL[p] for p in R.P], fontsize=8, rotation=18, ha='right')
a.set_ylabel('main effect on |error| (%)  — negative = helps')
a.set_title('(e) CALIBRATION: freezing geometry is the only lever, and it\n'
            'trades transverse diffusion against longitudinal.', fontsize=9.2)
a.legend(fontsize=7.4, frameon=False)

# ---------------- (f) the loss, and what it costs in truth ----------------
# PAIRED WITHIN SEED. The two seeds sit ~200 NLL apart in absolute loss, so plotting raw loss
# against raw error puts most of the spread on the seed axis and manufactures a trend that is
# really a seed offset. Differencing against the SAME seed's BT2 removes it and puts BT2 at the
# origin, where the quadrant is the statement.
a = ax[1, 2]
a.axhspan(0, 9, xmin=0, xmax=.5, color=C['red'], alpha=.06)
a.annotate('better FIT, worse PHYSICS', (.03, .93), xycoords='axes fraction', fontsize=8.2,
           color=C['red'], fontweight='bold')
for t in ['BM0', 'BMD', 'BMG', 'BMDG']:          # arms that FIT calibration
    dx = [fin(t, s, 'loss') - fin('BT2', s, 'loss') for s in S]
    dy = [fin(t, s, 'lifetime', True) - fin('BT2', s, 'lifetime', True) for s in S]
    a.plot(dx, dy, color=C['red'], lw=1.6, alpha=.35)
    for s in S:
        a.plot(dx[s], dy[s], 'o', ms=11, color=C['red'], alpha=.85 if s == 0 else .45)
    a.annotate(f'  {t}', (np.mean(dx), np.mean(dy)), fontsize=8.2)
a.plot(0, 0, '*', ms=22, color=C['ink'], zorder=5)
a.annotate('  BT2', (0, 0), fontsize=8.6, fontweight='bold')
a.axvline(0, color=C['ink'], lw=1.4)
a.axhline(0, color=C['ink'], lw=1.4)
a.set_xlabel('Δ loss vs BT2, same seed   (negative = better FIT)')
a.set_ylabel('Δ |lifetime error| vs BT2, same seed\n(positive = worse PHYSICS)')
a.set_title('(f) AND THEY POINT OPPOSITE WAYS. 7 of 8 points (4 arms × 2\n'
            'seeds) sit ABOVE zero; the 3 biggest loss gains are the 3 worst.', fontsize=9.2)

fig.suptitle('Fig 65 — the same seven arms read as the FACTORIAL they are: for each fitted block '
             'the other two form a complete 2×2, so every outcome\nreduces to two main effects and '
             'one interaction. Freezing CALIBRATION helps both nuisance blocks on both seeds; '
             'freezing a nuisance does\nnothing for the other; and the loss and the lifetime point '
             'in OPPOSITE directions.', fontsize=10.0)
fig.tight_layout(rect=[0, 0, 1, .905])
os.makedirs(OUT, exist_ok=True)
fig.savefig(f'{OUT}/fig65_block_factorial.png'); plt.close(fig)
print('wrote', f'{OUT}/fig65_block_factorial.png')

print('\n=== MAIN EFFECTS (per seed, then mean) ===')
for key, absv, param, lab in [('pos', False, None, 'position (um)'),
                              ('mae', False, None, 'dEdx MAE')]:
    arms, Z = grid(key, absv, param)
    mx, my, it = effects(Z)
    print(f'{lab}:  freeze {CELLS[key][1]} {mx.mean():+.4g} [{mx[0]:+.4g},{mx[1]:+.4g}]   '
          f'freeze {CELLS[key][2]} {my.mean():+.4g} [{my[0]:+.4g},{my[1]:+.4g}]   '
          f'interaction {it.mean():+.4g}')
print('calibration |error|:')
for p in R.P:
    arms, Z = grid('cal', True, p)
    mx, my, it = effects(Z)
    print(f'  {p:11s} freeze dEdx {mx.mean():+7.3f} [{mx[0]:+.2f},{mx[1]:+.2f}]   '
          f'freeze geom {my.mean():+7.3f} [{my[0]:+.2f},{my[1]:+.2f}]   inter {it.mean():+7.3f}')
