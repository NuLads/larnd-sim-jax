"""Fig 57 — the geometry-schedule screen at 3 000 iterations.

Three variants against BT2's OWN trace, which is the matched control at no cost: at equal iteration
count every arm indexes its schedules identically (calibration per epoch, geometry per iteration,
100 batches), and the seeds and targets are the same.

PARTIAL: GQ2 (10x learning rate) is still running, but it is already decided -- at iteration 1 000
it sits at 1 833 um against the baseline's 581.

The screen separates two things that "more geometry learning rate" conflates. Raising the INITIAL
rate (GQ1, GQ2) is worse, monotonically so with the size of the raise. Sustaining the rate LONGER
by decaying it more slowly (GQ3) is better, and the gap WIDENS with iteration. Geometry appears to
need small steps early, when the fit is far away and large steps are unstable, and sustained steps
late, when it is refining.

CAVEAT: 3 000 iterations of a 10 000-iteration problem. A schedule that descends faster here can
still settle worse; the calibration parameters in particular are nowhere near converged at this
point (BT2's own long. diffusion is +13.6% here against +7.3% at 10 000), so nothing is claimed
about them.
"""
import glob, os, pickle, re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = 'plots/noise_report'
C = dict(blue='#0072B2', orange='#E69F00', green='#009E73', red='#D55E00', grey='#666666',
         ink='#222222')
plt.rcParams.update({'font.size': 9, 'axes.grid': True, 'grid.alpha': .25,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.dpi': 130, 'savefig.bbox': 'tight'})
W = 200
ARMS = [('BT2', 'baseline   lr 1e-2, decay 0.9997', C['ink']),
        ('GQ1', 'lr 3e-2  (3× initial)', C['orange']),
        ('GQ2', 'lr 1e-1  (10× initial)  PARTIAL', C['red']),
        ('GQ3', 'decay 0.99993  (rate sustained)', C['green'])]


def sm(v, w=W):
    v = np.asarray(v, float)
    if len(v) <= w:
        return v
    from numpy.lib.stride_tricks import sliding_window_view
    return np.median(sliding_window_view(v, w), axis=1)


def load(tag, s):
    g = glob.glob(f'fit_result/sci_full_{tag}/history_iter*seed{s}.pkl')
    if not g:
        return None
    f = max(g, key=lambda p: int(re.search(r'iter(\d+)', p).group(1)))
    return np.ravel(pickle.load(open(f, 'rb'))['pos_residual_iter']) * 1e4


fig, ax = plt.subplots(1, 3, figsize=(16.2, 4.7))

a = ax[0]
for tag, lab, col in ARMS:
    cs = [sm(load(tag, s)) for s in [0, 1] if load(tag, s) is not None]
    cs = [c for c in cs if len(c) > W]
    if not cs:
        continue
    n = min(len(c) for c in cs)
    a.plot(np.median([c[:n] for c in cs], axis=0), color=col, lw=2.2, label=lab)
a.set_yscale('log'); a.set_xlim(0, 3000)
a.set_xlabel('iteration'); a.set_ylabel('position residual (µm)')
a.set_title('(a) raising the initial rate is WORSE; sustaining it\nlonger is BETTER', fontsize=9.4)
a.legend(fontsize=7.4, frameon=False)

a = ax[1]
ITS = [1000, 1500, 2000, 2500, 2900]
base = {}
for tag, lab, col in ARMS:
    v = []
    for it in ITS:
        vals = []
        for s in [0, 1]:
            p = load(tag, s)
            if p is None:
                continue
            ps = sm(p)
            if len(ps) > it - W:
                vals.append(ps[min(it - W // 2, len(ps) - 1)])
        v.append(np.mean(vals) if vals else np.nan)
    if tag == 'BT2':
        base = dict(zip(ITS, v))
    a.plot(ITS, [100 * (x / base[i] - 1) for i, x in zip(ITS, v)], color=col, lw=2.2,
           marker='o', ms=7, label=lab)
a.axhline(0, color=C['ink'], lw=1.5)
a.set_xlabel('iteration'); a.set_ylabel('position residual vs baseline (%)')
a.set_ylim(-15, 60)
a.set_title('(b) and the sustained-rate gain WIDENS: −3.7% at 1 500,\n−8.1% at 2 900', fontsize=9.4)
a.legend(fontsize=7.2, frameon=False)

a = ax[2]
xs = np.arange(2)
for k, (tag, lab, col) in enumerate([x for x in ARMS if x[0] != 'GQ2']):
    v = []
    for s in [0, 1]:
        p = load(tag, s)
        ps = sm(p)
        v.append(ps[min(2900 - W // 2, len(ps) - 1)])
    a.bar(xs + (k - 1) * .26, v, width=.24, color=col, edgecolor='white', label=lab)
    for j, y in enumerate(v):
        a.text(j + (k - 1) * .26, y + 4, f'{y:.0f}', ha='center', fontsize=7.4)
a.set_xticks(xs); a.set_xticklabels(['seed 0', 'seed 1'], fontsize=9)
a.set_ylabel('position residual at iteration 2 900 (µm)')
a.set_ylim(0, 470)
a.set_title('(c) the sustained-rate arm wins on BOTH seeds;\nthe 3× arm loses on one and wins on one',
            fontsize=9.4)
a.legend(fontsize=7.2, frameon=False)

fig.suptitle('Fig 57 — geometry-schedule screen (3 000 of 10 000 iterations, 2 seeds, BT2 as the '
             'matched control). "More geometry learning rate"\nsplits in two: a bigger INITIAL step '
             'is worse (10× diverges outright), a rate SUSTAINED longer is better and pulling away.',
             fontsize=10.1)
fig.tight_layout(rect=[0, 0, 1, .86])
os.makedirs(OUT, exist_ok=True)
fig.savefig(f'{OUT}/fig57_geom_screen.png'); plt.close(fig)
print('wrote', f'{OUT}/fig57_geom_screen.png')
