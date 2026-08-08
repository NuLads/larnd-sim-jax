"""Fig 50 (study 1, per seed): lifetime convergence for all five arms, one panel per seed.

Fig 49(e) plotted the across-seed MEDIAN, which is what hid the effect: the six seeds are six
DIFFERENT problems, not six samples of one. Each draws its own target lifetime while the detector
depth is fixed, so the anode-to-cathode charge swing -- the entire lifetime signal available to
that fit -- runs from 4.15% to 15.21%.

Panels are ordered by that lever arm, easiest problem first. Read left-to-right and the pattern in
fig 49(h) is visible directly: on the four high-lever-arm seeds the fast-annealing arms
(dE/dx -> 0.01x, calib 0.80) converge to sub-percent while the baseline settles several percent
low; on the two low-lever-arm seeds they diverge instead.

Single runs, so each curve is one realisation with no across-seed averaging -- rolling median over
200 iterations, and the residual wiggle is real per-batch scatter, not noise to be smoothed away.
"""
import glob, os, pickle, re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = 'plots/noise_report'
C = dict(blue='#0072B2', orange='#E69F00', green='#009E73', red='#D55E00',
         purple='#CC79A7', grey='#666666', ink='#222222')
plt.rcParams.update({'font.size': 9, 'axes.grid': True, 'grid.alpha': .25,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.dpi': 130, 'savefig.bbox': 'tight'})
ARMS = [('S1A0', 'baseline  0.91 / const', C['ink']),
        ('S1A1', 'calib 0.80  (less travel)', C['blue']),
        ('S1A2', 'calib 0.97  (more travel)', C['green']),
        ('S1A3', 'dE/dx → 0.1×', C['orange']),
        ('S1A4', 'dE/dx → 0.01×', C['red'])]
T_DRIFT = 0.087 * 2200.0        # detector depth in us (sec 6k)
W = 200


def smooth(v, w=W):
    v = np.asarray(v, float)
    if len(v) <= w:
        return v
    from numpy.lib.stride_tricks import sliding_window_view
    return np.median(sliding_window_view(v, w), axis=1)


D = {}
for tag, _, _ in ARMS:
    for f in sorted(glob.glob(f'fit_result/sci_ceiling_{tag}/history_iter10000_*.pkl')):
        s = int(re.search(r'seed(\d+)', f).group(1))
        h = pickle.load(open(f, 'rb'))
        t = float(np.ravel(h['lifetime_target'])[0])
        D[(tag, s)] = (100 * (np.ravel(h['lifetime_iter'])[1:] / t - 1), t)
SEEDS = sorted({s for _, s in D})
SWING = {s: 100 * (1 - np.exp(-T_DRIFT / D[('S1A0', s)][1])) for s in SEEDS}
ORDER = sorted(SEEDS, key=lambda s: -SWING[s])          # easiest problem first
print(f'{len(SEEDS)} seeds, ordered by lever arm: '
      + ', '.join(f'seed {s} ({SWING[s]:.1f}%)' for s in ORDER))

fig, axes = plt.subplots(2, 3, figsize=(16.4, 8.6), sharex=True)
for k, s in enumerate(ORDER):
    a = axes.ravel()[k]
    tau = D[('S1A0', s)][1]
    a.axhspan(-1, 1, color=C['green'], alpha=.10)
    a.axhline(0, color=C['red'], lw=1.6, ls='--', zorder=1)
    fin = {}
    for tag, lab, col in ARMS:
        v = D[(tag, s)][0]
        a.plot(smooth(v), color=col, lw=1.7, label=lab if k == 0 else None)
        fin[tag] = float(np.median(v[int(len(v) * .8):]))
    best = min(fin, key=lambda t_: abs(fin[t_]))
    a.set_title(f'seed {s}   τ = {tau:.0f} µs   swing {SWING[s]:.1f}%'
                + ('   ← nominal-ish' if 6.5 < SWING[s] < 9 else ''), fontsize=9.4)
    txt = '\n'.join(f'{t_}: {fin[t_]:+6.2f}%' + ('  ←best' if t_ == best else '')
                    for t_, _, _ in ARMS)
    a.text(.985, .035, txt, transform=a.transAxes, ha='right', va='bottom', fontsize=6.9,
           family='monospace', bbox=dict(fc='white', ec=C['grey'], alpha=.85, lw=.7))
    if k % 3 == 0:
        a.set_ylabel('lifetime error (%)')
    if k >= 3:
        a.set_xlabel('iteration')
    lo, hi = min(fin.values()), max(fin.values())
    a.set_ylim(min(lo, -2) - 3, max(hi, 2) + 5)
    if SWING[s] <= 6.5:
        for sp in a.spines.values():
            sp.set_edgecolor(C['red']); sp.set_linewidth(1.8)
        a.text(.02, .97, 'LOW LEVER ARM', transform=a.transAxes, va='top', fontsize=7.4,
               color=C['red'], fontweight='bold')

axes.ravel()[0].legend(fontsize=7.2, frameon=True, framealpha=.9, loc='upper right')
fig.suptitle('Fig 50 — study 1 per seed: the same five schedules on six DIFFERENT problems, '
             'ordered by lever arm (easiest first).\nOn the four seeds with signal, dE/dx→0.01× '
             '(red) converges to sub-percent while the baseline (black) settles several percent low. '
             'On the two\nred-framed low-lever-arm seeds it diverges instead — which is why the '
             'across-seed mean in fig 49(a) showed nothing.', fontsize=10.2)
fig.tight_layout(rect=[0, 0, 1, .90])
os.makedirs(OUT, exist_ok=True)
fig.savefig(f'{OUT}/fig50_study1_per_seed.png'); plt.close(fig)
print('wrote', f'{OUT}/fig50_study1_per_seed.png')
print(f'\n{"seed":>5s} {"tau":>6s} {"swing":>6s} ' + ''.join(f'{t:>9s}' for t, _, _ in ARMS))
for s in ORDER:
    row = ''.join(f'{np.median(D[(t, s)][0][int(len(D[(t, s)][0]) * .8):]):+9.2f}'
                  for t, _, _ in ARMS)
    print(f'{s:5d} {D[("S1A0", s)][1]:6.0f} {SWING[s]:5.1f}% ' + row)
