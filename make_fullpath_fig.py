"""Fig 69 — the fit -> truth path: no barrier. The optimiser stops short, it is not blocked.

THE QUESTION. The objective prefers truth (P1 beats P0 on 7/7 arm-seeds by 5.6%, fig 66) and the
optimiser does not get there. Two very different reasons, needing very different work:
  a BARRIER or SHELF between the fit and truth -> no local method reaches it; spend the effort on
                                                  the objective instead
  a MONOTONE DOWNHILL path                     -> nothing structural blocks us; the gap is
                                                  conditioning / step size, and a better
                                                  preconditioned optimiser should close it

One-block scans cannot answer this: moving any SINGLE block to truth alone can make the loss WORSE
(calibration alone +0.18%, dEdx alone +2.79%). The question is only well posed along the JOINT line,
with calibration, dEdx and geometry all moving together.

THE ANSWER IS UNAMBIGUOUS: the path is MONOTONE DOWNHILL from the fit to truth in every arm and
every seed. There is no barrier and no shelf. Nothing structural stands between where the optimiser
stops and where truth is.

AND THE MINIMUM SITS AT alpha = 1, NOT BEYOND IT. Past truth the loss climbs steeply -- by alpha=1.2
it is above the value at the fit in the converged arms. So along this direction the objective's
optimum IS truth. That is a stronger statement than any conditional scan in this campaign supports,
and it is the cleanest evidence yet that the remaining error is not an objective bias.

WHY THEN DOES THE FIT STOP? Because it is not at a stationary point. The schedules anneal the
learning rate to ~8e-5 of peak, so a run ends when it runs out of step size, not when the gradient
vanishes -- and a gradient pointing toward truth is exactly what the monotone descent shows is still
there. Combined with the shape (shallow near the fit, steepening toward truth) this is the classic
signature of an ILL-CONDITIONED valley: steepest descent is nearly orthogonal to the direction that
actually reduces the loss, so progress along it is slow even though the path is open.

THE ACTIONABLE CONCLUSION: build a preconditioner. The gap is reachable, and more of the same
first-order optimisation is not the way to reach it -- G20K already showed that doubling the
iterations plateaus.

A SECONDARY RESULT WORTH KEEPING: TS1 is the CLOSEST arm to truth (1.34-1.36% of loss remaining
against G20K's 2.09% and BT2's 9.1-9.6%), which is an independent confirmation, on a measure none of
them was tuned against, that the cheap two-block recipe beats the long single-block run.
"""
import glob, os, pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = 'plots/noise_report'
C = dict(blue='#0072B2', orange='#E69F00', green='#009E73', red='#D55E00',
         grey='#888888', ink='#222222')
plt.rcParams.update({'font.size': 9, 'axes.grid': True, 'grid.alpha': .25,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.dpi': 130, 'savefig.bbox': 'tight'})
ALPHA = [0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.05, 1.1, 1.2]
ARMS = [('FPBT2', 'BT2  (baseline, 13.4 h)', C['ink'], [0, 1]),
        ('FPTS1', 'TS1  (stage 1, 13.5 h)', C['green'], [0, 1]),
        ('FPG20K', 'G20K (20 000 it, 26.4 h)', C['blue'], [0, 1])]


def curve(tag, s):
    fs = glob.glob(f'fit_result/sci_full_{tag}/history_iter*seed{s}.pkl')
    if not fs:
        return None
    Rs = pickle.load(open(fs[0], 'rb')).get('truthpoint', [])
    if not Rs:
        return None
    out = []
    for a in ALPHA:
        k = f'F{a:g}'
        if any(k not in r['points'] for r in Rs):
            return None
        out.append(sum(r['points'][k]['loss'] for r in Rs))
    return np.array(out)


D = {}
for tag, _, _, seeds in ARMS:
    for s in seeds:
        c = curve(tag, s)
        if c is not None:
            D[(tag, s)] = c
print(f'{len(D)} arm-seeds loaded')

fig, ax = plt.subplots(1, 3, figsize=(16.6, 5.4))
A = np.array(ALPHA)

# ---------------- (a) the paths ----------------
a = ax[0]
for tag, lab, col, seeds in ARMS:
    for s in seeds:
        if (tag, s) not in D:
            continue
        c = D[(tag, s)]
        a.plot(A, 100 * (c / c[0] - 1), '-o', color=col, ms=4, lw=1.8,
               alpha=.95 if s == 0 else .5, label=lab if s == 0 else None)
a.axvline(1, color=C['red'], lw=1.8, ls='--')
a.annotate(' TRUTH', (1, 1), fontsize=8, color=C['red'])
a.axhline(0, color=C['grey'], lw=1.2)
a.annotate('the fit', (0, .3), fontsize=7.6, color=C['grey'])
a.set_xlabel('α   (0 = the fit,  1 = truth,  all three blocks moving together)')
a.set_ylabel('loss, % relative to the fit')
a.set_title('(a) NO BARRIER. Monotone downhill from the fit to truth in\n'
            'every arm and seed — and the minimum is AT truth, not beyond.', fontsize=9.3)
a.legend(fontsize=7.4, frameon=False, loc='lower left')

# ---------------- (b) zoom on the converged arms ----------------
a = ax[1]
for tag, lab, col, seeds in ARMS:
    if tag == 'FPBT2':
        continue
    for s in seeds:
        if (tag, s) not in D:
            continue
        c = D[(tag, s)]
        a.plot(A, 100 * (c / c[0] - 1), '-o', color=col, ms=5, lw=2,
               alpha=.95 if s == 0 else .5, label=lab if s == 0 else None)
a.axvline(1, color=C['red'], lw=1.8, ls='--')
a.axhline(0, color=C['grey'], lw=1.2)
a.set_xlabel('α')
a.set_ylabel('loss, % relative to the fit')
a.set_title('(b) THE CONVERGED ARMS, zoomed. TS1 has the least left to\n'
            'gain — it is the closest arm to truth of any we have run.', fontsize=9.3)
a.legend(fontsize=7.4, frameon=False, loc='lower left')

# ---------------- (c) fraction of the baseline gap closed ----------------
# F1 is IDENTICAL across arms for a given seed (verified: spread 1e-8 relative), because truth does
# not depend on which fit you started from. That makes it a fixed reference per seed, so the arms
# can be compared on absolute distance to it -- a far more interpretable number than each arm's
# own-relative percentage.
a = ax[2]
labs, closed, cols = [], [], []
i1 = ALPHA.index(1)
for s in [0, 1]:
    if ('FPBT2', s) not in D:
        continue
    base = D[('FPBT2', s)][0] - D[('FPBT2', s)][i1]      # BT2's gap to truth, in loss units
    for tag, lab, col, seeds in ARMS:
        if (tag, s) not in D:
            continue
        gap = D[(tag, s)][0] - D[(tag, s)][i1]
        labs.append(f'{lab.split()[0]} s{s}')
        closed.append(100 * (1 - gap / base))
        cols.append(col)
y = np.arange(len(labs))
a.barh(y, closed, color=cols, edgecolor='white', height=.62)
for i, v in enumerate(closed):
    a.text(v + 1.2, i, f'{v:.1f}%', va='center', fontsize=8.2)
a.set_yticks(y); a.set_yticklabels(labs, fontsize=8.2); a.invert_yaxis()
a.set_xlim(0, 108)
a.set_xlabel('% of BT2’s gap to truth that this arm has closed')
a.set_title('(c) HOW FAR EACH ARM GOT. TS1 closes 86–87% of the baseline’s\n'
            'gap to truth; G20K 77–79%, for twice the compute.', fontsize=9.3)

fig.suptitle('Fig 69 — the fit→truth path, all three blocks moving together. It is MONOTONE '
             'DOWNHILL everywhere and its minimum sits AT truth, not past it:\nthere is no barrier '
             'and no shelf, so the remaining gap is reachable. The fits stop because the schedule '
             'anneals the learning rate to ~8e−5 of peak,\nnot because the gradient vanishes — the '
             'signature of an ill-conditioned valley. The actionable conclusion is to build a '
             'preconditioner.', fontsize=10.0)
fig.tight_layout(rect=[0, 0, 1, .84])
os.makedirs(OUT, exist_ok=True)
fig.savefig(f'{OUT}/fig69_fullpath.png'); plt.close(fig)
print('wrote', f'{OUT}/fig69_fullpath.png')
for (tag, s), c in sorted(D.items()):
    i1 = ALPHA.index(1)
    print(f'  {tag[2:]:5s} s{s}: gap {-100*(c[i1]/c[0]-1):5.2f}%   min at α={ALPHA[int(np.argmin(c))]:g}   '
          f'monotone to α=1: {bool(np.all(np.diff(c[:i1+1])<0))}   α=1.2 vs fit: {100*(c[-1]/c[0]-1):+.2f}%')
