"""Fig 68 — TS2, the calibration release. The two-stage recipe does NOT dominate.

WHAT TS2 IS. Stage 2 of the two-stage recipe: everything warm-started from TS1, calibration
RELEASED, fresh 10 000-iteration anneal. TS1 had given the best FIT of the campaign but its
calibration was BT2's, frozen; this is the run that was supposed to convert that into a better
CALIBRATION. 3 seeds, all complete.

VERIFIED FIRST: calibration genuinely moved (peak-to-peak 755-1434 in physical units) from an
initialisation matching TS1's to 0.0000%; dEdx moved 0.088-0.107 log units; geometry moved on
98.7-100% of tracks. The release did what it says.

EVERY NUMBER HERE IS A PAIRED PER-SEED DIFFERENCE, and that is not a stylistic choice. The three
arms sit within ~3 um and ~130 NLL of each other in absolute terms, which is smaller than the
seed-to-seed spread -- so the MEAN and the MEDIAN of the absolute levels disagree about whether TS1
or TS2 has the lower loss. Paired differences are immune to that; absolute levels are not, and are
not quoted.

DID THE RELEASE HELP? Partly, and it cost something.
  transverse diffusion  -0.93, better on 3/3   <- the one real gain
  position              +2.7 um, worse on 3/3
  E field               +0.023, worse on 3/3   (0.022 -> 0.045: real in ratio, negligible absolutely)
  loss, A_b, long. diff., lifetime -- all straddle zero across seeds, i.e. nulls

IS THE RECIPE BETTER THAN G20K AT EQUAL COMPUTE? (27.0 h against 26.4 h.) No -- it trades.
  loss                  -131, TS2 better on 3/3
  transverse diffusion  +0.49, G20K better on 3/3
  everything else       tie

SO THE HONEST VERDICT IS A NULL WITH A TWIST. Neither arm dominates: the two-stage recipe wins the
OBJECTIVE and loses TRANSVERSE DIFFUSION against a single long run of the same cost. That pattern --
lower loss, worse physics -- is the same one figs 63 and 65 found, and fig 66 explained: the
objective prefers truth only when all three blocks move together, and stage 2 moves calibration
against nuisances that are themselves no longer free to follow.

AND THE PRACTICAL READING: TS1 alone, at 13.5 h, is the best value in the campaign. Stage 2 doubles
the cost to buy one parameter and give back position. Anyone continuing this should treat the second
stage as optional.
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
S = [0, 1, 2]
W = 200
PL = {'Ab': 'A$_b$', 'eField': 'E field', 'lifetime': 'lifetime',
      'tran_diff': 'tran. diff.', 'long_diff': 'long. diff.'}


def sm(v, w=W):
    v = np.asarray(v, float)
    from numpy.lib.stride_tricks import sliding_window_view
    return np.median(sliding_window_view(v, w), axis=1) if len(v) > w else v


CA = {}
def L(t, s):
    if (t, s) not in CA:
        fs = glob.glob(f'fit_result/sci_full_{t}/history_iter*seed{s}.pkl')
        h = pickle.load(open(max(fs, key=lambda p: int(re.search(r'iter(\d+)', p).group(1))), 'rb'))
        d = {}
        for p in R.P:
            tg = float(np.ravel(h[p + '_target'])[0])
            d[p] = 100 * (np.ravel(h[p + '_iter'])[1:] / tg - 1)
        d['mae'] = np.asarray([x for x in h.get('dedx_mae_truth_iter', []) if x == x], float)
        d['pos'] = np.ravel(h['pos_residual_iter']) * 1e4
        d['loss'] = np.asarray(h['losses_iter'], float)
        CA[(t, s)] = d
    return CA[(t, s)]


fin = lambda t, s, k: np.median(sm(L(t, s)[k])[-300:])
ROWS = [('loss', 'loss (NLL)', 0, 1), ('pos', 'position (µm)', 0, 1),
        ('mae', 'dE/dx MAE (×10⁻⁴)', 0, 1e4)] + [(p, PL[p] + ' |err|', 1, 1) for p in R.P]


def paired(a, b, k, ab, sc):
    f = (lambda x: abs(x)) if ab else (lambda x: x)
    return [(f(fin(a, s, k)) - f(fin(b, s, k))) * sc for s in S]


fig, ax = plt.subplots(1, 3, figsize=(16.8, 5.6))

for j, (a_, b_, title) in enumerate([
        ('TS2', 'TS1', '(a) DID THE RELEASE HELP?  TS2 − TS1'),
        ('TS2', 'G20K', '(b) THE PRODUCTION CHOICE  TS2 − G20K\n(27.0 h vs 26.4 h — equal compute)'),
        ('TS2', 'BT2', '(c) AND VERSUS THE BASELINE  TS2 − BT2')]):
    a = ax[j]
    for i, (k, lab, ab, sc) in enumerate(ROWS):
        d = paired(a_, b_, k, ab, sc)
        null = min(d) * max(d) <= 0
        col = C['grey'] if null else (C['green'] if np.mean(d) < 0 else C['red'])
        a.plot([min(d), max(d)], [i, i], color=col, lw=3, alpha=.5)
        for x in d:
            a.plot(x, i, 'o', ms=9, color=col, alpha=.85)
        n = sum(1 for x in d if x < 0)
        a.text(1.02, i, f'{n}/3', transform=a.get_yaxis_transform(), va='center',
               fontsize=7.4, color=col)
    a.axvline(0, color=C['ink'], lw=1.8)
    a.set_yticks(range(len(ROWS)))
    a.set_yticklabels([r[1] for r in ROWS], fontsize=8.2)
    a.invert_yaxis()
    a.set_xscale('symlog', linthresh=1)
    a.set_xlabel(f'{a_} − {b_}, per seed   (negative = {a_} better)')
    a.set_title(title, fontsize=9.3)
    if j == 0:
        a.annotate('grey = seeds straddle 0 → null', (.03, .03), xycoords='axes fraction',
                   fontsize=7.2, color=C['grey'])

fig.suptitle('Fig 68 — TS2, the calibration release: the two-stage recipe does NOT dominate. '
             'Releasing calibration buys TRANSVERSE DIFFUSION (−0.93, 3/3)\nand gives back position '
             '(+2.7 µm, 0/3); everything else is a null. Against G20K at equal compute it wins the '
             'OBJECTIVE (−131, 3/3) and loses\ntransverse diffusion (+0.49, 0/3) — lower loss, worse '
             'physics, the same pattern figs 63/65 found and fig 66 explained.', fontsize=10.0)
fig.tight_layout(rect=[0, 0, 1, .84])
os.makedirs(OUT, exist_ok=True)
fig.savefig(f'{OUT}/fig68_ts2.png'); plt.close(fig)
print('wrote', f'{OUT}/fig68_ts2.png')
for a_, b_ in [('TS2', 'TS1'), ('TS2', 'G20K'), ('TS2', 'BT2')]:
    print(f'\n{a_} vs {b_}:')
    for k, lab, ab, sc in ROWS:
        d = paired(a_, b_, k, ab, sc)
        tag = 'null' if min(d) * max(d) <= 0 else ('BETTER' if np.mean(d) < 0 else 'worse')
        print(f'  {lab:22s} {np.mean(d):+9.3f}  {sum(1 for x in d if x<0)}/3  {tag}')
