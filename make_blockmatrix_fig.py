"""Fig 63 — the block matrix: warm-start every block from BT2, then freeze subsets and re-anneal.

THE SETUP. The fit has three blocks -- calibration C (5 physics params), per-segment dEdx D (~4000
nuisances) and per-track spline geometry G. They are normally optimised together, and the degeneracy
matrix (fig 58) says they are coupled: each block's optimum MOVES as the others move. So a block may
sit where it does not because the data wants it there but because it co-adapted to the others while
they were wrong. Seven arms, all three blocks warm-started from BT2 (nothing restarts from scratch),
then subsets frozen.

THIS IS THE FIRST STUDY BUILT WITH TRAP 1b DESIGNED IN. Fig 62 killed the earlier alternation result
because a 3 000-iteration arm was compared against a converged 10 000-iteration one at equal
ITERATION COUNT, which is not equal SCHEDULE PHASE. Here both schedules are COMPRESSED so 3 000
iterations land on BT2's terminal LR (SCILRDECAY 0.91 -> 0.73, SCICHDECAY 0.9997 -> 0.999). Panel
(a) is the other half of that discipline: every freeze verified against the checkpoint before any
number was read.

BM0 -- everything warm-started, nothing frozen -- IS THE CONTROL, and it is not optional: a
re-anneal alone moves position 165 -> 127 um. Any arm must be read against BM0, not against BT2, or
the re-anneal is credited to the freeze.

WHAT IT FINDS.
  * Freezing CALIBRATION is the clear win on the objective: BMC (fit D+G) reaches the best loss of
    any arm, -3.25% against BT2 and -2.06% against the BM0 control, with the best position (115.7 um)
    and the best dEdx MAE. The nuisance blocks were being held back by co-adaptation with
    calibration.
  * Freezing GEOMETRY erases essentially the whole descent -- BMG/BMCG/BMDG all sit within 0.1% of
    BT2's loss. Geometry is where the objective lives, exactly as the alpha-scan measured (5.1x the
    dEdx question), and it is why the fig 59 frozen-geometry arms could not have been descending.
  * Panel (f) is the uncomfortable one. BM0 finds a LOWER loss than BT2 while moving lifetime
    FURTHER from truth, +3.59 points, consistently on both seeds. Better fit, worse physics: for
    lifetime the objective's optimum is displaced from truth, which is this campaign's oldest open
    problem and here it is in a single controlled pair.

n = 2 SEEDS: a screen sized for a large effect. Every per-seed value is printed rather than a median
over two, because fig 62 is the standing reminder that a median of 2 is the mean of 2.
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
ARMS = ['BM0', 'BMC', 'BMD', 'BMG', 'BMCD', 'BMCG', 'BMDG']
FIT = {'BM0': 'CDG', 'BMC': 'DG', 'BMD': 'CG', 'BMG': 'CD',
       'BMCD': 'G', 'BMCG': 'D', 'BMDG': 'C'}
S = [0, 1]
W = 200
PL = {'Ab': 'A$_b$', 'eField': 'E field', 'lifetime': 'lifetime',
      'tran_diff': 'tran. diff.', 'long_diff': 'long. diff.'}


def sm(v, w=W):
    v = np.asarray(v, float)
    from numpy.lib.stride_tricks import sliding_window_view
    return np.median(sliding_window_view(v, w), axis=1) if len(v) > w else v


def raw(tag, s):
    fs = glob.glob(f'fit_result/sci_full_{tag}/history_iter*seed{s}.pkl')
    return pickle.load(open(max(fs, key=lambda p: int(re.search(r'iter(\d+)', p).group(1))), 'rb'))


def load(tag, s):
    h = raw(tag, s)
    d = {}
    for p in R.P:
        t = float(np.ravel(h[p + '_target'])[0])
        d[p] = 100 * (np.ravel(h[p + '_iter'])[1:] / t - 1)
    d['mae'] = np.asarray([x for x in h.get('dedx_mae_truth_iter', []) if x == x], float)
    d['pos'] = np.ravel(h['pos_residual_iter']) * 1e4
    d['loss'] = np.asarray(h['losses_iter'], float)
    return d


D = {t: {s: load(t, s) for s in S} for t in ARMS + ['BT2']}
fin = lambda t, s, k: np.median(sm(D[t][s][k])[-300:])
med = lambda t, k, a=False: np.median([abs(fin(t, s, k)) if a else fin(t, s, k) for s in S])

fig, ax = plt.subplots(2, 3, figsize=(16.6, 9.2))

# ---------------- (a) verification ----------------
a = ax[0, 0]
Z = np.zeros((len(ARMS), 3))
for i, t in enumerate(ARMS):
    for s in S:
        h, b = raw(t, s), raw('BT2', s)
        Z[i, 0] += (max(np.ptp(np.ravel(h[p + '_iter'])) for p in R.P) == 0) / len(S)
        dw = {int(k): np.asarray(v['log_dedx']) for k, v in h['dedx_cache'].items() if 'log_dedx' in v}
        db = {int(k): np.asarray(v['log_dedx']) for k, v in b['dedx_cache'].items() if 'log_dedx' in v}
        dd = max((np.max(np.abs(dw[k] - db[k])) for k in set(dw) & set(db)
                  if len(dw[k]) == len(db[k])), default=1)
        Z[i, 1] += (dd == 0) / len(S)
        n = ident = 0
        for k in set(h['chain_cache']) & set(b['chain_cache']):
            for tw, tb in zip(h['chain_cache'][k], b['chain_cache'][k]):
                n += 1
                ident += np.array_equal(np.asarray(tw['angles']), np.asarray(tb['angles']))
        Z[i, 2] += (ident / n > .9) / len(S)
a.imshow(Z, cmap='Greens', vmin=0, vmax=1.6, aspect='auto')
for i, t in enumerate(ARMS):
    for j, blk in enumerate('CDG'):
        want = blk in ''.join(set('CDG') - set(FIT[t]))
        a.text(j, i, 'FROZEN' if Z[i, j] > .5 else 'fitted', ha='center', va='center',
               fontsize=7.6, color=C['ink'] if Z[i, j] > .5 else C['grey'],
               fontweight='bold' if Z[i, j] > .5 else 'normal')
        assert (Z[i, j] > .5) == want, f'{t} {blk} freeze does not match its declaration'
a.set_xticks([0, 1, 2]); a.set_xticklabels(['calibration', 'dE/dx', 'geometry'], fontsize=8.6)
a.set_yticks(range(len(ARMS))); a.set_yticklabels(ARMS, fontsize=8.6)
a.grid(False)
a.set_title('(a) VERIFIED, not assumed: every freeze checked against\n'
            'the checkpoint. All 7 arms match their declaration.', fontsize=9.3)

# ---------------- (b) the loss ----------------
a = ax[0, 1]
order = sorted(ARMS, key=lambda t: med(t, 'loss'))
v = [med(t, 'loss') for t in order]
# GEOMETRY-frozen takes colour priority: it is the dominant effect, and BMCG is BOTH
# calibration- and geometry-frozen -- coluring it green would file it with the winners when it
# is pinned at BT2's loss like every other geometry-frozen arm.
cols = [C['red'] if 'G' not in FIT[t] else (C['green'] if 'C' not in FIT[t] else C['blue'])
        for t in order]
a.barh(np.arange(len(order)), v, color=cols, edgecolor='white', height=.62)
a.axvline(med('BT2', 'loss'), color=C['ink'], lw=1.8, ls='--')
a.annotate(' BT2', (med('BT2', 'loss'), len(order) - .4), fontsize=7.6, color=C['ink'])
a.axvline(med('BM0', 'loss'), color=C['grey'], lw=1.8, ls=':')
a.annotate('BM0 control ', (med('BM0', 'loss'), 6.45), fontsize=7.2, color=C['grey'], ha='right')
for i, t in enumerate(order):
    a.text(v[i] - 12, i, f'fits {FIT[t]}   {100*(v[i]/med("BT2","loss")-1):+.2f}%', va='center',
           ha='right', fontsize=7.4, color='white')
a.set_yticks(np.arange(len(order))); a.set_yticklabels(order, fontsize=8.6)
a.set_xlim(8700, 9200); a.invert_yaxis()
a.set_xlabel('loss (PPP NLL)')
a.set_title('(b) THE OBJECTIVE. Red = geometry frozen → pinned at BT2.\n'
            'Green = geometry fitted AND calibration frozen → the winners.', fontsize=9.3)

# ---------------- (c) position ----------------
a = ax[0, 2]
G = ['BM0', 'BMC', 'BMD', 'BMCD']
for i, t in enumerate(G):
    for s in S:
        a.plot(i, fin(t, s, 'pos'), 'o', ms=9, color=C['blue'], alpha=.85 if s == 0 else .45)
    a.plot(i, med(t, 'pos'), '_', ms=26, color=C['ink'], mew=2.2)
a.axhline(med('BT2', 'pos'), color=C['ink'], lw=1.8, ls='--')
a.annotate(f' BT2 {med("BT2","pos"):.0f} µm', (-.4, med('BT2', 'pos')), fontsize=7.6, va='bottom')
a.axhline(med('BM0', 'pos'), color=C['grey'], lw=1.6, ls=':')
a.annotate(f' BM0 control {med("BM0","pos"):.0f}', (1.5, med('BM0', 'pos')), fontsize=7.2,
           va='bottom', color=C['grey'])
a.set_xticks(range(len(G)))
a.set_xticklabels([f'{t}\nfits {FIT[t]}' for t in G], fontsize=7.8)
a.set_ylabel('position residual (µm)')
a.set_title('(c) POSITION — only arms that FIT geometry. A re-anneal alone\n'
            'gives 165→127; freezing calibration adds a further −11 µm.', fontsize=9.3)

# ---------------- (d) dE/dx ----------------
a = ax[1, 0]
DD = ['BM0', 'BMC', 'BMG', 'BMCG']
for i, t in enumerate(DD):
    for s in S:
        a.plot(i, fin(t, s, 'mae'), 'o', ms=9, color=C['orange'], alpha=.85 if s == 0 else .45)
    a.plot(i, med(t, 'mae'), '_', ms=26, color=C['ink'], mew=2.2)
a.axhline(med('BT2', 'mae'), color=C['ink'], lw=1.8, ls='--')
a.annotate(' BT2', (-.4, med('BT2', 'mae')), fontsize=7.6, va='bottom')
a.set_xticks(range(len(DD)))
a.set_xticklabels([f'{t}\nfits {FIT[t]}' for t in DD], fontsize=7.8)
a.set_ylabel('per-segment dE/dx MAE vs truth')
a.set_title('(d) dE/dx — only arms that FIT it. Same ordering as the loss:\n'
            'the two calibration-frozen arms are the two best.', fontsize=9.3)

# ---------------- (e) calibration ----------------
a = ax[1, 1]
CA = ['BT2', 'BM0', 'BMD', 'BMG', 'BMDG']
cols = [C['ink'], C['grey'], C['blue'], C['orange'], C['purple']]
xs = np.arange(len(R.P)); w = .16
for i, t in enumerate(CA):
    a.bar(xs + (i - 2) * w, [med(t, p, True) for p in R.P], width=w, color=cols[i],
          edgecolor='white', label=f'{t} (fits {FIT.get(t,"CDG")})')
a.set_yscale('log')
a.set_xticks(xs); a.set_xticklabels([PL[p] for p in R.P], fontsize=8, rotation=18, ha='right')
a.set_ylabel('|error| (%)')
a.set_title('(e) CALIBRATION — only arms that FIT it. Every arm improves\n'
            'long. diffusion and every arm degrades LIFETIME.', fontsize=9.3)
a.legend(fontsize=6.4, frameon=False, ncol=2)

# ---------------- (f) the displacement ----------------
a = ax[1, 2]
lab, val, col = [], [], []
for p in R.P:
    d = [abs(fin('BM0', s, p)) - abs(fin('BT2', s, p)) for s in S]
    lab.append(PL[p]); val.append(d)
    # grey whenever the two seeds straddle zero: with n = 2 that is a null, not a direction
    col.append(C['grey'] if min(d) * max(d) <= 0 else (C['red'] if np.mean(d) > 0 else C['green']))
for i, d in enumerate(val):
    for s in S:
        a.plot(d[s], i, 'o', ms=9, color=col[i], alpha=.85 if s == 0 else .45)
    a.plot([min(d), max(d)], [i, i], color=col[i], lw=2.4, alpha=.5)
a.axvline(0, color=C['ink'], lw=1.6)
a.set_yticks(range(len(lab))); a.set_yticklabels(lab, fontsize=8.6); a.invert_yaxis()
a.set_xlabel('Δ|error| vs BT2, per seed  (negative = better; grey = seeds straddle 0 → null)')
dl = 100 * (med('BM0', 'loss') / med('BT2', 'loss') - 1)
a.set_title(f'(f) THE UNCOMFORTABLE ONE. BM0 finds a LOWER loss ({dl:+.2f}%)\n'
            'yet moves LIFETIME +3.6 further from truth, on both seeds.', fontsize=9.3)

fig.suptitle('Fig 63 — the block matrix. Freezing CALIBRATION lets the nuisances descend: BMC '
             'reaches the best loss (−3.25% vs BT2, −2.06% vs the\nBM0 control), the best position '
             '(115.7 µm) and the best dE/dx. Freezing GEOMETRY erases the descent entirely — those '
             'arms sit within 0.1%\nof BT2. And a re-anneal buys a lower loss while pushing '
             'LIFETIME further from truth: better fit, worse physics.', fontsize=10.0)
fig.tight_layout(rect=[0, 0, 1, .90])
os.makedirs(OUT, exist_ok=True)
fig.savefig(f'{OUT}/fig63_block_matrix.png'); plt.close(fig)
print('wrote', f'{OUT}/fig63_block_matrix.png')
print('\nloss:', ', '.join(f'{t} {med(t,"loss"):.0f} ({100*(med(t,"loss")/med("BT2","loss")-1):+.2f}%)'
                           for t in order))
print('position:', ', '.join(f'{t} {med(t,"pos"):.1f}' for t in G), f'| BT2 {med("BT2","pos"):.1f}')
print('mae:', ', '.join(f'{t} {med(t,"mae"):.5f}' for t in DD), f'| BT2 {med("BT2","mae"):.5f}')
