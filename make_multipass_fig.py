"""Fig 77 — the multi-pass bootstrap ACROSS THE ENSEMBLE: the rep-8 ladder does not generalise,
and its fixed point is not fixed.

WHAT WAS HOPED (from rep 8 alone): pass N re-anneals calibration from nominal against pass N-1's
converged nuisances; rep 8 collapsed 16.4 -> 11.7 -> 0.65% and the recipe looked like THE fix.

WHAT THE ENSEMBLE SAYS (this figure):
  * pass 2 makes lifetime WORSE on 2 of 3 new replicas (rep3 +9.1 -> +19.4, rep7 +14.6 -> +17.1)
    and barely helps rep1 (+6.5 -> +5.9);
  * pass 4 on rep 8 -- re-annealing FROM the +0.65% point -- lands at +16.5% with a final loss
    ABOVE pass 3's (8324 vs 8293). The re-anneal destroyed a better optimum it started from.
  * panel (b): the loss DOES improve pass-over-pass (until it doesn't), while lifetime scatters
    by +-8% around no clear trend: each full re-anneal re-runs the hot-phase overshoot and lands
    on a fresh draw from a wide landing distribution. Rep 8's 0.65% was a lucky draw.

MECHANISM CONCLUSION: the destructive element is the RE-ANNEAL ITSELF (calibration restarted at
nominal at full LR against imperfect nuisances -> overshoot -> truncated relaxation, every time).
The follow-up on the cluster now: warm-start ALL blocks at the previous endpoint with a gentle
calibration schedule (peak LR 1e-3, decay 0.955) -- pure relaxation, no overshoot to recover
from (tags RLXR3/RLXR7/RLXR8, and RLXP38 to test survival of the good point).
"""
import glob, pickle, re
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
    fs = glob.glob(f'fit_result/sci_full_{tag}/history_iter10000_*.pkl')
    if not fs:
        return None
    h = pickle.load(open(fs[0], 'rb'))
    v = np.ravel(h['lifetime_iter'])
    t = float(np.ravel(h['lifetime_target'])[0])
    L = np.asarray(h['losses_iter'], float)
    return (100 * (np.median(v[int(len(v) * .9):]) / t - 1),
            float(np.median(L[int(len(L) * .95):])))


LADDERS = {  # replica -> ordered pass tags
    'rep 1': ['DSREP1', 'PASS2R1', 'PASS3R1'],
    'rep 3': ['DSREP3', 'PASS2R3', 'PASS3R3'],
    'rep 7': ['DSREP7', 'PASS2R7', 'PASS3R7'],
    'rep 8': ['DSREP8', 'PASS2R8', 'PASS3R8', 'PASS4R8'],
}
COLS = {'rep 1': C['blue'], 'rep 3': C['orange'], 'rep 7': C['purple'], 'rep 8': C['red']}

fig, ax = plt.subplots(1, 2, figsize=(9.6, 4.1))

for name, tags in LADDERS.items():
    pts = [(i + 1, endpoint(t)) for i, t in enumerate(tags)]
    pts = [(n, e) for n, e in pts if e is not None]
    if not pts:
        continue
    ns = [n for n, _ in pts]
    lt = [e[0] for _, e in pts]
    dl = [e[1] - pts[0][1][1] for _, e in pts]
    ax[0].plot(ns, lt, 'o-', color=COLS[name], label=name, lw=1.8, ms=6)
    ax[1].plot(ns, dl, 'o-', color=COLS[name], lw=1.8, ms=6)
    for n, y in zip(ns, lt):
        dy = -12 if (name, n) == ('rep 7', 1) else 4
        ax[0].annotate(f'{y:+.1f}', (n, y), textcoords='offset points',
                       xytext=(6, dy), fontsize=8, color=COLS[name])

ax[0].axhline(0, color=C['green'], lw=1.6)
ax[0].annotate('truth', (.98, 0), xycoords=('axes fraction', 'data'),
               textcoords='offset points', xytext=(0, 4), ha='right',
               color=C['green'], fontweight='bold', fontsize=9)
ax[0].set_xlabel('pass number (each = full 10 000-it re-anneal,\ncalibration restarted at nominal)')
ax[0].set_ylabel('lifetime error at endpoint (%)')
ax[0].set_title('(a) lifetime: no ensemble collapse —\npass 2 worsens 2/3 replicas; pass 4 destroys rep 8')
ax[0].set_xticks([1, 2, 3, 4])
ax[0].legend(frameon=False, fontsize=9, loc='lower left')

ax[1].axhline(0, color=C['grey'], lw=1, ls=':')
ax[1].set_xlabel('pass number')
ax[1].set_ylabel('final loss − pass-1 final loss')
ax[1].set_title('(b) loss improves pass-over-pass (until pass 4)\nwhile lifetime scatters: a soft manifold, re-sampled')
ax[1].set_xticks([1, 2, 3, 4])

fig.suptitle('Fig 77 — the multi-pass bootstrap across the ensemble: rep 8’s 0.65% was a lucky draw,\n'
             'and re-annealing from it lands back at +16.5% (higher loss). The re-anneal itself is the destructive step.',
             fontsize=10.5, y=1.06)
fig.tight_layout()
fig.savefig(f'{OUT}/fig77_multipass_ensemble.png')
print('wrote', f'{OUT}/fig77_multipass_ensemble.png')
