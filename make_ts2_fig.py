"""Fig 68 — TS2, the calibration release, as a slopegraph. Follow one seed's line across the recipe.

WHY THIS FORM. The three arms differ by less than the seed-to-seed spread, so a plot of averages
would be actively misleading -- the mean and the median of the absolute loss disagree about whether
TS1 or TS2 is lower. The honest comparison is PAIRED, within a seed. A slopegraph does that while
still showing the real values: each thin line is one seed followed across the arms, so "all three
seeds moved the same way" is something you see rather than a statistic you are asked to trust.

ONE THING NOT TO MISREAD: in the five calibration panels the BT2 -> TS1 segment is flat BY
CONSTRUCTION, because stage 1 freezes calibration. It is a definition, not a result. Those panels
only start carrying information at TS1 -> TS2, which is why their verdict strips count that step.

HOW TO READ IT. Left to right within each panel: BT2 (the baseline) -> TS1 (stage 1: calibration
frozen, nuisances refitted) -> TS2 (stage 2: calibration released). G20K sits after the gap as the
alternative that costs the same as the full recipe. DOWN IS BETTER everywhere. The label under each
arm is its wall-clock cost. The verdict strip above each panel counts how many of the three seeds
improved on the step that panel is about.

WHAT IT SHOWS.
  * The whole recipe's gain is in STAGE 1. Loss, position and dEdx all fall steeply from BT2 to TS1
    and then go flat -- or back up -- at TS2.
  * Stage 2 buys exactly one thing, TRANSVERSE DIFFUSION, on 3 of 3 seeds, and gives back position.
  * G20K, at twice stage 1's cost, ends up beside TS2: better on transverse diffusion, worse on the
    objective. Neither dominates.
  * LIFETIME never improves anywhere in the recipe. Its lines fan out rather than descend, which is
    the visual signature of the seed-dependent behaviour section 6w documents.
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
         grey='#888888', ink='#222222')
SEEDCOL = {0: '#0072B2', 1: '#E69F00', 2: '#009E73'}
plt.rcParams.update({'font.size': 9, 'axes.grid': True, 'grid.alpha': .22,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.dpi': 130, 'savefig.bbox': 'tight'})
S = [0, 1, 2]
W = 200
ARMS = [('BT2', 'BT2\nbaseline\n13.4 h'), ('TS1', 'TS1\nstage 1\n13.5 h'),
        ('TS2', 'TS2\nstage 2\n27.0 h'), ('G20K', 'G20K\nalternative\n26.4 h')]
XP = [0, 1, 2, 3.5]          # gap before G20K: it is not part of the recipe


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


def val(t, s, k, ab):
    v = np.median(sm(L(t, s)[k])[-300:])
    return abs(v) if ab else v


# key, label, absolute?, scale, the step the verdict counts, verdict text
PAN = [
    ('loss', 'loss  (PPP NLL)', 0, 1, ('TS1', 'BT2'), 'stage 1 does the work'),
    ('pos', 'position residual  (µm)', 0, 1, ('TS1', 'BT2'), 'stage 1 does the work'),
    ('mae', 'per-segment dE/dx MAE', 0, 1, ('TS1', 'BT2'), 'stage 1 does the work'),
    ('tran_diff', 'transverse diffusion  |error| %', 1, 1, ('TS2', 'TS1'),
     'THE ONE THING stage 2 buys'),
    ('Ab', 'A$_b$  |error| %', 1, 1, ('TS2', 'TS1'), 'stage 2: no change'),
    ('eField', 'E field  |error| %', 1, 1, ('TS2', 'TS1'), 'stage 2: no change'),
    ('long_diff', 'longitudinal diffusion  |error| %', 1, 1, ('TS2', 'TS1'), 'stage 2: no change'),
    ('lifetime', 'lifetime  |error| %', 1, 1, ('TS2', 'TS1'), 'never improves, anywhere'),
]

fig, axes = plt.subplots(2, 4, figsize=(17.4, 9.2))
for i, (k, lab, ab, sc, (aa, bb), verdict) in enumerate(PAN):
    a = axes.ravel()[i]
    for s in S:
        y = [val(t, s, k, ab) * sc for t, _ in ARMS]
        a.plot(XP[:3], y[:3], '-o', color=SEEDCOL[s], lw=1.8, ms=7, alpha=.9,
               label=f'seed {s}' if i == 0 else None)
        a.plot(XP[3], y[3], 'D', color=SEEDCOL[s], ms=8, alpha=.9)
        a.plot([XP[2], XP[3]], [y[2], y[3]], ':', color=SEEDCOL[s], lw=1.0, alpha=.45)
    n = sum(1 for s in S if val(aa, s, k, ab) < val(bb, s, k, ab))
    col = C['green'] if n == 3 else (C['red'] if n == 0 else C['grey'])
    a.set_title(f'{lab}\n{verdict}  —  {aa} better than {bb} on {n}/3 seeds',
                fontsize=9.0, color=col if n in (0, 3) else C['ink'])
    a.set_xticks(XP)
    a.set_xticklabels([l for _, l in ARMS], fontsize=7.4)
    a.set_xlim(-.35, 3.9)
    a.axvline(2.75, color=C['grey'], lw=1, ls='--', alpha=.6)
    if k in ('loss', 'pos'):
        a.set_yscale('log')
    if i == 0:
        a.legend(fontsize=7.4, frameon=False, loc='upper right')
    if ab:   # a calibration panel: the first segment is flat BY DEFINITION, not by result
        a.annotate('flat here by construction —\ncalibration is FROZEN in stage 1',
                   (.02, .90), xycoords='axes fraction', fontsize=6.9, color=C['grey'],
                   style='italic', va='top')
axes.ravel()[0].annotate('down = better', (.03, .06), xycoords='axes fraction',
                         fontsize=7.6, color=C['grey'], style='italic')

fig.suptitle('Fig 68 — the two-stage recipe, one line per seed. Follow a single line left to right: '
             'BT2 → stage 1 → stage 2, with G20K after the gap as the\nsame-cost alternative. '
             'Down is better. Almost the whole gain is in STAGE 1; stage 2 buys transverse diffusion '
             'on 3/3 seeds and gives back\nposition; and lifetime improves nowhere. Seed-paired by '
             'construction, because the arms differ by less than the spread between seeds.',
             fontsize=10.2)
fig.tight_layout(rect=[0, 0, 1, .90])
os.makedirs(OUT, exist_ok=True)
fig.savefig(f'{OUT}/fig68_ts2.png'); plt.close(fig)
print('wrote', f'{OUT}/fig68_ts2.png')
for k, lab, ab, sc, (aa, bb), _ in PAN:
    n = sum(1 for s in S if val(aa, s, k, ab) < val(bb, s, k, ab))
    print(f'  {lab:36s} {aa} better than {bb} on {n}/3')
