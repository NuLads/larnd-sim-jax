"""Fig 93 — every measured lever on the lifetime bias, on one scale, and the new leader.

`DXTRU` (dE/dx pinned at its arc-length-matched truth for the whole fit, verified MAE = 0.00000)
was launched expecting ~2 points from the §6f spread ladder. It delivers -10.4 +- 3.3 (3/3), which
makes the dE/dx block the joint-largest lever measured -- level with initialising geometry at truth.
Seed 7 is the striking one: +14.63 -> -1.80, i.e. essentially AT truth.

WHY THE LADDER UNDER-PREDICTED, and it matters for what to do next: the ladder varied only the
SPREAD of the dE/dx field, preserving each segment's rank. The fitted field is wrong in two ways:
it is shrunk (71% of true width) AND it has per-segment noise (MAE 0.055, comparable to the true
spread 0.084 itself). Pinning at truth removes both; the ladder only ever measured the first.

AND THE LOSS GOES UP IN ALL THREE ARMS (+99, +158, +187) while lifetime, long_diff and position
all improve sharply -- the same loss-blindness as figs 80-83, now inside the nuisance block. This
is textbook nuisance-parameter bias: the likelihood's optimum for ~4000 per-segment parameters is
not the optimum for the calibration parameter we care about.
"""
import glob, pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = 'plots/noise_report'
BLUE, ORANGE, GREEN, RED, GREY, INK = ('#0072B2', '#E69F00', '#009E73', '#D55E00',
                                       '#888888', '#222222')
plt.rcParams.update({'font.size': 9.5, 'axes.grid': True, 'grid.alpha': .25,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.dpi': 130, 'savefig.bbox': 'tight'})


def g(tag, p='lifetime'):
    fs = glob.glob(f'fit_result/sci_full_{tag}/history_iter10000_*.pkl')
    if not fs:
        return None
    h = pickle.load(open(fs[0], 'rb'))
    v = np.ravel(h[p + '_iter'])
    t = float(np.ravel(h[p + '_target'])[0])
    return 100 * (np.median(v[int(len(v) * .9):]) / t - 1)


# measured levers: (label, delta in points, note, uses-truth?)
LEVERS = [
    ('dE/dx pinned at truth\n(DXTRU, n=3)', np.mean([g(f'DXTRU{s}') - g(f'DSREP{s}')
                                                     for s in (3, 7, 8)]), True),
    ('geometry init at truth\n(gtru8, n=1)', g('GTRU8') - g('DSREP8'), True),
    ('gentle vs violent restart\n(RLX, n=3)', -7.0, False),
    ('calibration init at truth\n(CINITT, n=6)', -2.38, False),
    ('drift penalty w=1e3\n(dw8, n=1)', -6.0, False),
    ('slower calib anneal\n(hot8, n=1)', -1.8, False),
    ('multi-pass bootstrap\n(ensemble, n=4)', +2.0, False),
    ('calibration gate\n(late8, n=1)', +39.0, False),
]
LEVERS = [(l, d, u) for l, d, u in LEVERS if d is not None]
LEVERS.sort(key=lambda x: x[1])

fig, ax = plt.subplots(1, 2, figsize=(13.2, 4.6),
                       gridspec_kw={'width_ratios': [1.35, 1]})

a = ax[0]
y = np.arange(len(LEVERS))
cols = [GREEN if u else (BLUE if d < 0 else RED) for _, d, u in LEVERS]
a.barh(y, [d for _, d, _ in LEVERS], color=cols)
a.set_yticks(y, [l for l, _, _ in LEVERS], fontsize=8.2)
a.axvline(0, color=INK, lw=1.2)
for i, (_, d, _) in enumerate(LEVERS):
    a.annotate(f'{d:+.1f}', (d, i), textcoords='offset points',
               xytext=(6 if d > 0 else -6, -3), ha='left' if d > 0 else 'right', fontsize=8.5)
a.set_xlabel('change in lifetime bias (percentage points)   ← better')
a.set_title('(a) every measured lever, one scale.\nGreen = uses truth (an upper bound, not a recipe)',
            fontsize=9.5)
a.set_xlim(-13, 45)

a = ax[1]
arms = [('seed 3', 'DSREP3', 'DXTRU3', ORANGE), ('seed 7', 'DSREP7', 'DXTRU7', BLUE),
        ('seed 8', 'DSREP8', 'DXTRU8', RED)]
for i, (lab, b, t, col) in enumerate(arms):
    a.plot([0, 1], [g(b), g(t)], 'o-', color=col, lw=2, ms=8, label=lab)
    a.annotate(f'{g(b):+.1f}', (0, g(b)), textcoords='offset points', xytext=(-8, -3),
               ha='right', fontsize=9, color=col)
    a.annotate(f'{g(t):+.1f}', (1, g(t)), textcoords='offset points', xytext=(8, -3),
               fontsize=9, color=col)
a.axhline(0, color=GREEN, lw=1.8)
a.annotate('truth', (.5, 0), textcoords='offset points', xytext=(0, 5), ha='center',
           color=GREEN, fontweight='bold', fontsize=9)
a.axhspan(0, 2.0, color=GREY, alpha=.16, lw=0)
a.annotate('what the §6f spread ladder predicted\nfrom our 71% recovered width: ~2 pts',
           (.06, 2.4), fontsize=8.2, color='#555555')
a.set_xticks([0, 1], ['dE/dx FITTED\n(MAE 0.055)', 'dE/dx PINNED AT TRUTH\n(MAE 0.00000)'],
             fontsize=8.5)
a.set_xlim(-.35, 1.5)
a.set_ylabel('lifetime error (%)')
a.set_title('(b) the surprise: −10.4 ± 3.3 points, not 2 (3/3) —\nand the LOSS gets WORSE in every arm', fontsize=9.5)
a.legend(frameon=False, fontsize=9, loc='center right')

fig.suptitle('Fig 93 — the dE/dx block is the joint-largest lever on the lifetime bias, not a second-order one.\n'
             'The spread ladder under-predicted because it varied only the field’s WIDTH; the fitted field is also wrong segment-by-segment\n'
             '(MAE 0.055 against a true spread of 0.084). Removing both error modes buys 10.4 ± 3.3 points, 3/3 — at a WORSE loss.',
             fontsize=10.5, y=1.06)
fig.tight_layout()
fig.savefig(f'{OUT}/fig93_levers.png')
print('wrote fig93')
for l, d, u in LEVERS:
    print(f'  {l.splitlines()[0]:38s} {d:+6.1f}')
