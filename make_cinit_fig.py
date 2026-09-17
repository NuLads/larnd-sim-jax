"""Fig 84 — the calibration-initialization test: my pre-registered prediction was WRONG.

PREDICTED (registered 15 Sep, before results): landings insensitive to the calibration start,
because the hot phase drives calibration to the nuisance-conditional optimum regardless.

MEASURED: the start matters, and monotonically on rep 3 — the further BELOW truth calibration
starts, the HIGHER it lands (overshoot grows with the distance travelled). Starting AT truth is
best on both replicas AND gives the lowest loss of the family. So part of the +8.6% ensemble bias
is calibration-init memory, not only nuisance-init memory.

Panel (c) is the degeneracy's most dramatic single demonstration: CINITA8 (all five shifted) ends
at A_b +7.8% / lifetime -59.3% with a loss of 9026 -- BETTER than the baseline's 9167 at A_b
+0.01% / lifetime +16.4%. A catastrophically wrong fit that the objective prefers.
"""
import glob, pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = 'plots/noise_report'
plt.rcParams.update({'font.size': 10, 'axes.grid': True, 'grid.alpha': .25,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.dpi': 130, 'savefig.bbox': 'tight'})
BLUE, ORANGE, GREEN, RED, GREY = '#0072B2', '#E69F00', '#009E73', '#D55E00', '#888888'


def ep(tag):
    fs = glob.glob(f'fit_result/sci_full_{tag}/history_iter10000_*.pkl')
    if not fs:
        return None
    h = pickle.load(open(fs[0], 'rb'))
    o = {}
    for p in ['lifetime', 'Ab', 'long_diff']:
        v = np.ravel(h[p + '_iter'])
        t = float(np.ravel(h[p + '_target'])[0])
        o[p] = 100 * (np.median(v[int(len(v) * .9):]) / t - 1)
        o['i_' + p] = 100 * (v[0] / t - 1)
    o['loss'] = float(np.median(np.asarray(h['losses_iter'], float)[-500:]))
    return o


ARMS = [('CINITL', 'low\n−31%'), ('DSREP', 'nominal\n−8.6%'), ('CINITT', 'at truth\n0%'),
        ('CINITH', 'high\n+14%')]
SEEDS_LADDER = [3, 8]          # the original pair (all four arms)
SEEDS_REPL = [1, 3, 7, 8, 12, 13]   # low/nominal/at-truth on six replicas
fig, ax = plt.subplots(1, 3, figsize=(13.4, 4.2))

# (a) landing vs init, ordered by init
for s, col in ((3, ORANGE), (8, RED)):
    xs, ys, ls = [], [], []
    for pre, lab in ARMS:
        e = ep(f'{pre}{s}')
        if e:
            xs.append(e['i_lifetime']); ys.append(e['lifetime']); ls.append(e['loss'])
    ax[0].plot(xs, ys, 'o-', color=col, lw=1.8, ms=7, label=f'replica {s}')
    for x, y in zip(xs, ys):
        ax[0].annotate(f'{y:+.1f}', (x, y), textcoords='offset points', xytext=(6, 5),
                       fontsize=8, color=col)
ax[0].axhline(0, color=GREEN, lw=1.6)
ax[0].axvline(0, color=GREY, lw=.8, ls=':')
ax[0].annotate('truth', (.98, 0), xycoords=('axes fraction', 'data'), ha='right',
               textcoords='offset points', xytext=(0, 5), color=GREEN, fontweight='bold')
ax[0].set_xlabel('calibration START: lifetime error (%)')
ax[0].set_ylabel('LANDING: lifetime error (%)')
ax[0].set_title('(a) the start matters — and on rep 3 monotonically:\nstart lower → land higher')
ax[0].legend(frameon=False, fontsize=9)

# (b) loss of each arm
w = .35
for i, (s, col) in enumerate(((3, ORANGE), (8, RED))):
    labs, L = [], []
    for pre, lab in ARMS:
        e = ep(f'{pre}{s}')
        if e:
            labs.append(lab.replace('\n', ' ')); L.append(e['loss'])
    base = [x for (pre, _), x in zip(ARMS, L) if pre == 'DSREP'][0]
    ax[1].bar(np.arange(len(L)) + (i - .5) * w, np.array(L) - base, w, color=col,
              label=f'replica {s}')
ax[1].axhline(0, color=GREY, lw=1)
ax[1].set_xticks(range(len(ARMS)), [l for _, l in ARMS], fontsize=8.5)
ax[1].set_ylabel('loss − nominal-init loss')
ax[1].set_title('(b) …and starting at truth also gives the BEST loss:\nthe nominal init is not even loss-optimal')
ax[1].legend(frameon=False, fontsize=9)

# (c) the degeneracy demo
pts = [('DSREP8', 'nominal init\n(baseline)', BLUE), ('CINITT8', 'init at truth', GREEN),
       ('CINITA8', 'all-shifted init', RED)]
for tag, lab, col in pts:
    e = ep(tag)
    ax[2].scatter(e['Ab'], e['lifetime'], s=110, color=col, zorder=3, edgecolors='white')
    ax[2].annotate(f"{lab}\nloss {e['loss']:.0f}", (e['Ab'], e['lifetime']),
                   textcoords='offset points', xytext=(9, -4), fontsize=8.5, color=col)
ax[2].axhline(0, color=GREEN, lw=1.4)
ax[2].axvline(0, color=GREEN, lw=1.4)
ax[2].set_xlabel('A_b error (%)')
ax[2].set_ylabel('lifetime error (%)')
ax[2].set_xlim(-2, 12)
ax[2].set_ylim(-70, 28)
ax[2].set_title('(c) the degeneracy, in one picture: A_b +7.8 / lifetime −59\nhas a BETTER loss than the baseline')

fig.suptitle('Fig 84 — the calibration-initialization test (replicas 3 and 8, everything else identical). '
             'The pre-registered prediction\n("the start will not matter") is FALSIFIED: where calibration starts '
             'changes where it lands by up to 18 points.',
             fontsize=11, y=1.06)
fig.tight_layout()
fig.savefig(f'{OUT}/fig84_calib_init.png')
print('wrote fig84')
