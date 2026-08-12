"""Fig 64 — block-matrix convergence traces: what actually HAPPENS during each arm.

Fig 63 reports endpoints. This is the same seven arms as trajectories, which is where the mechanism
is visible rather than inferred.

HOW TO READ IT. Line style encodes whether the plotted quantity's OWN block is frozen:
  solid  = that block is being fitted in this arm
  dotted = that block is FROZEN, so the trace is flat at BT2's value by construction -- not a
           result, just the baseline drawn on the same axes for reference.
Colour encodes the arm. BT2's own trace (10 000 iterations, thin grey) is underneath everything as
the point all seven arms start from: every arm here begins where BT2 finished.

WHAT THE TRACES SHOW THAT THE BAR CHARTS CANNOT.
  * The re-anneal KICKS every fitted block away from BT2 first and then re-descends. That transient
    is not noise -- it is the whole reason a warm start does not simply stay put, and it is what
    erased ALTW's warm start (fig 62). Here it is deliberate and every arm pays it.
  * POSITION descends monotonically in every geometry-fitting arm and is still falling at 3 000, so
    none of these arms is at its floor -- the position numbers are upper bounds.
  * LIFETIME is the opposite: it is kicked away and does NOT come back within 3 000 iterations in
    any calibration-fitting arm. That is the +3.6 point degradation of fig 63(f), seen as a
    trajectory: the re-anneal moves lifetime to a new place and the loss is happy there.
  * The LOSS panel separates the arms into exactly two families -- geometry fitted (descending) and
    geometry frozen (flat from iteration ~0) -- which is the 5.1x information asymmetry made visual.

200-iteration rolling median throughout; thin = the two seeds, thick = their median.
"""
import glob, os, pickle, re, io, contextlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
with contextlib.redirect_stdout(io.StringIO()):
    import robust_convergence as R

OUT = 'plots/noise_report'
C = dict(blue='#0072B2', orange='#E69F00', green='#009E73', red='#D55E00',
         purple='#CC79A7', cyan='#56B4E9', yellow='#F0E442', grey='#999999', ink='#222222')
plt.rcParams.update({'font.size': 9, 'axes.grid': True, 'grid.alpha': .25,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.dpi': 130, 'savefig.bbox': 'tight'})
# arm -> (blocks FITTED, colour)
ARMS = [('BM0', 'CDG', C['ink']), ('BMC', 'DG', C['green']), ('BMD', 'CG', C['blue']),
        ('BMG', 'CD', C['red']), ('BMCD', 'G', C['cyan']), ('BMCG', 'D', C['purple']),
        ('BMDG', 'C', C['orange'])]
S = [0, 1]
W = 200
TRUE_MEAN = 1.88693
PL = {'Ab': 'A$_b$ (recombination)', 'eField': 'E field', 'lifetime': 'lifetime',
      'tran_diff': 'transverse diffusion', 'long_diff': 'longitudinal diffusion'}
# which block does each panel belong to?
BLK = {'Ab': 'C', 'eField': 'C', 'tran_diff': 'C', 'long_diff': 'C', 'lifetime': 'C',
       'mae': 'D', 'meandedx': 'D', 'pos': 'G', 'loss': None}


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
    d['meandedx'] = np.asarray([x for x in h.get('mean_dedx_iter', []) if x == x], float)
    d['pos'] = np.ravel(h['pos_residual_iter']) * 1e4
    d['loss'] = np.asarray(h['losses_iter'], float)
    return d


D = {t: {s: load(t, s) for s in S} for t, _, _ in ARMS}
BT = {s: load('BT2', s) for s in S}

PANELS = [('Ab', PL['Ab'], '% error'), ('eField', PL['eField'], '% error'),
          ('tran_diff', PL['tran_diff'], '% error'),
          ('long_diff', PL['long_diff'], '% error'),
          ('lifetime', PL['lifetime'] + '   ← every arm leaves BT2, none returns', '% error'),
          ('mae', 'per-segment dE/dx MAE vs TRUTH', 'MAE'),
          ('meandedx', 'length-weighted ⟨dE/dx⟩', 'MeV/cm'),
          ('pos', 'position residual   ← still falling at 3 000 (y-axis clipped)', 'µm'),
          ('loss', 'loss (PPP NLL)   ← splits into exactly two families', 'NLL')]

fig, axes = plt.subplots(3, 3, figsize=(16.8, 11.4), sharex=True)
for k, (key, title, ylab) in enumerate(PANELS):
    a = axes.ravel()[k]
    # BT2's own trace, as the common starting point
    cs = [sm(BT[s][key]) for s in S]
    n = min(len(c) for c in cs)
    a.plot(np.linspace(-3000, 0, n), np.median([c[:n] for c in cs], axis=0),
           color=C['grey'], lw=1.4, alpha=.8)
    for tag, fitb, col in ARMS:
        frozen = BLK[key] is not None and BLK[key] not in fitb
        cs = [sm(D[tag][s][key]) for s in S]
        n = min(len(c) for c in cs)
        x = np.arange(n)
        if not frozen:
            for c in cs:
                a.plot(x, c[:n], color=col, lw=.55, alpha=.30)
        a.plot(x, np.median([c[:n] for c in cs], axis=0), color=col,
               lw=1.4 if frozen else 2.2, ls=':' if frozen else '-',
               alpha=.55 if frozen else .95)
    a.axvline(0, color=C['ink'], lw=1.0, ls='-', alpha=.5)
    if ylab == '% error':
        a.axhline(0, color=C['red'], lw=1.2, ls='--')
    if key == 'meandedx':
        a.axhline(TRUE_MEAN, color=C['red'], lw=1.4, ls='--')
        a.set_ylim(1.855, 1.895)
    if key in ('pos', 'loss'):
        a.set_yscale('log')
        # clip to the region where the ARMS live: BT2's early descent (700 um / 18 000 NLL)
        # otherwise sets the range and squashes every difference this figure exists to show.
        a.set_ylim(*( (100, 420) if key == 'pos' else (8600, 11500) ))
    if key == 'long_diff':
        a.set_ylim(-15, 25)
    if key == 'lifetime':
        a.set_ylim(-8, 12)
    if key == 'Ab':
        a.set_ylim(-1.2, 1.2)
    a.set_xlim(-3000, 3000)
    a.set_title(title, fontsize=9.4)
    a.set_ylabel(ylab, fontsize=8.6)
    if k >= 6:
        a.set_xlabel('iteration   (negative = BT2, the run every arm starts from)')
    if k == 0:
        a.annotate('BT2', (-2600, 0), fontsize=7.6, color=C['grey'])

handles = [Line2D([], [], color=c, lw=2.2, label=f'{t}  fits {f}') for t, f, c in ARMS]
handles += [Line2D([], [], color=C['grey'], lw=1.4, label='BT2 (before iteration 0)'),
            Line2D([], [], color=C['ink'], lw=1.4, ls=':', label='dotted = that block is FROZEN')]
axes.ravel()[1].legend(handles=handles, fontsize=6.9, frameon=True, framealpha=.92, ncol=2,
                       loc='upper right')

fig.suptitle('Fig 64 — block-matrix TRAJECTORIES (fig 63 shows only their endpoints). Every arm '
             'starts where BT2 finished, at iteration 0.\nDotted = that block is frozen, so the '
             'line is flat at BT2 by construction. The re-anneal kicks every fitted block away '
             'first: position\nre-descends past BT2 and is still falling at 3 000, while LIFETIME '
             'is kicked away and never comes back — and the loss is happy there.', fontsize=10.0)
fig.tight_layout(rect=[0, 0, 1, .925])
os.makedirs(OUT, exist_ok=True)
fig.savefig(f'{OUT}/fig64_block_traces.png'); plt.close(fig)
print('wrote', f'{OUT}/fig64_block_traces.png')
