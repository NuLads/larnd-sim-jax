"""Fig 78 — the relaxation arms: schedule engineering is exhausted.

Every restart of the optimiser — violent (full re-anneal, pass 4) or gentle (SCILR=3e-3,
33x below standard peak, all blocks warm-started including calibration) — moves lifetime
toward a schedule-specific attractor and NOT toward truth:

  * gentle relaxation from +9.1 / +16.4 / +0.65 all land at +5..+7% -- an attractor near +6%,
    better than the re-anneal's ~+14% but still biased;
  * the near-truth point (+0.65%, loss 8293) is UNSTABLE under BOTH restart types: full
    re-anneal -> +16.5% (loss 8324), gentle -> +7.2% (loss 8354). Both end at WORSE loss than
    the point they started from: the transient kicks the state off a better optimum and the
    anneal freezes it before it can return;
  * the upward direction is systematic (10/10 restarts drift up, never down).

CONCLUSION: the +8.6% ensemble landing bias is not a schedule property — the optimiser-side
levers are exhausted. The pull is the objective's lifetime<->dE/dx coupling during nuisance
transients; next test = the (scale,shape) dE/dx reparametrisation on the same replicas
(SSR3/7/8, running), which removes the rank-1 coupling term analytically.
"""
import glob, pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = 'plots/noise_report'
C = dict(blue='#0072B2', orange='#E69F00', green='#009E73', red='#D55E00',
         purple='#CC79A7', grey='#888888', ink='#222222')
plt.rcParams.update({'font.size': 10, 'axes.grid': True, 'grid.alpha': .25,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.dpi': 130, 'savefig.bbox': 'tight'})


def endpoint(tag):
    f = glob.glob(f'fit_result/sci_full_{tag}/history_iter10000_*.pkl')[0]
    h = pickle.load(open(f, 'rb'))
    v = np.ravel(h['lifetime_iter'])
    t = float(np.ravel(h['lifetime_target'])[0])
    L = np.asarray(h['losses_iter'], float)
    return 100 * (np.median(v[int(len(v) * .9):]) / t - 1), float(np.median(L[int(len(L) * .95):]))


# (label, start tag, end tag, color, style)
ARMS = [
    ('rep 3: gentle',            'DSREP3',  'RLXR3',  C['orange'], '-'),
    ('rep 7: gentle',            'DSREP7',  'RLXR7',  C['purple'], '-'),
    ('rep 8: gentle',            'DSREP8',  'RLXR8',  C['red'],    '-'),
    ('rep 8 near-truth: gentle', 'PASS3R8', 'RLXP38', C['blue'],   '-'),
    ('rep 8 near-truth: re-anneal', 'PASS3R8', 'PASS4R8', C['blue'], '--'),
]

fig, ax = plt.subplots(1, 2, figsize=(9.6, 4.3))

for lab, t0, t1, col, ls in ARMS:
    (e0, L0), (e1, L1) = endpoint(t0), endpoint(t1)
    ax[0].plot([0, 1], [e0, e1], ls, color=col, marker='o', ms=6, lw=1.8, label=lab)
    ax[0].annotate(f'{e1:+.1f}', (1, e1), textcoords='offset points', xytext=(8, -3),
                   fontsize=8, color=col)
    ax[1].plot([0, 1], [0, L1 - L0], ls, color=col, marker='o', ms=6, lw=1.8)
    ax[1].annotate(f'{L1 - L0:+.0f}', (1, L1 - L0), textcoords='offset points',
                   xytext=(8, -3), fontsize=8, color=col)

ax[0].axhline(0, color=C['green'], lw=1.6)
ax[0].annotate('truth', (.02, 0), xycoords=('axes fraction', 'data'),
               textcoords='offset points', xytext=(0, 4), color=C['green'],
               fontweight='bold', fontsize=9)
ax[0].axhspan(5, 7.3, color=C['grey'], alpha=.15, lw=0)
ax[0].annotate('the gentle attractor (+5…+7%)', (.03, 7.6), xycoords=('axes fraction', 'data'),
               fontsize=8.5, color=C['grey'])
ax[0].set_xticks([0, 1], ['start (warm-start point)', 'endpoint'])
ax[0].set_xlim(-.15, 1.35)
ax[0].set_ylabel('lifetime error (%)')
ax[0].set_title('(a) every restart drifts toward its schedule’s attractor,\nnever toward truth — including FROM the near-truth point')
ax[0].legend(frameon=False, fontsize=8, loc='lower center', bbox_to_anchor=(.55, .02))

ax[1].axhline(0, color=C['grey'], lw=1, ls=':')
ax[1].set_xticks([0, 1], ['start', 'endpoint'])
ax[1].set_xlim(-.15, 1.35)
ax[1].set_ylabel('final loss − loss of the starting point')
ax[1].set_title('(b) restarts from the near-truth point end at WORSE loss\n(blue): the transient kicks the state off a better optimum')

fig.suptitle('Fig 78 — relaxation arms (calibration LR 33× reduced, all blocks warm-started): schedule\n'
             'engineering is exhausted; the systematic upward pull is the objective’s, not the schedule’s.',
             fontsize=10.5, y=1.07)
fig.tight_layout()
fig.savefig(f'{OUT}/fig78_relax_arms.png')
print('wrote', f'{OUT}/fig78_relax_arms.png')
