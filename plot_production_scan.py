#!/usr/bin/env python3
"""Production-scale validation of the dEdx recommendations — and why it is inconclusive.

A: per-seed lifetime for the 4 configurations -- the seed scatter swamps everything
B: mean |calibration error| with seed error bars -- no configuration is resolved
C: the effect of lowering the mean-constraint weight, TOY vs PRODUCTION -- it does not transfer
"""
import pickle

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
BLUE, ORANGE, RED, AQUA, VIOLET = "#2a78d6", "#eb6834", "#e34948", "#1baf7a", "#4a3aa7"
plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 9,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": GRID, "axes.titlecolor": INK,
})

TRUTH = {'Ab': 0.83488, 'eField': 0.52152, 'tran_diff': 1.0028e-05,
         'long_diff': 5.8142e-06, 'lifetime': 2406.4}
BASE = 'fit_result/sci_full/history_iter2500_sci_full_b100_len50'
CFG = [('perseg\n$w=10^5$', '%s_mdx0.15_seed%%d.pkl' % BASE, BLUE),
       ('perseg\n$w=3000$', '%s_mw3000.0_mdx0.15_seed%%d.pkl' % BASE, AQUA),
       ('scale_shape\n$w=10^5$', '%s_scale_shape_mdx0.15_seed%%d.pkl' % BASE, ORANGE),
       ('scale_shape\n$w=3000$', '%s_mw3000.0_scale_shape_mdx0.15_seed%%d.pkl' % BASE, RED)]


def robust(h, key):
    a = np.asarray(h[key + '_iter'], float)
    a = a[np.isfinite(a)]
    return np.median(a[-int(len(a) * 0.2):])


life, allerr = [], []
for _, pat, _c in CFG:
    lv, ae = [], []
    for s in range(3):
        h = pickle.load(open(pat % s, 'rb'))
        lv.append(100 * (robust(h, 'lifetime') - TRUTH['lifetime']) / TRUTH['lifetime'])
        ae.append(np.mean([abs(100 * (robust(h, k) - v) / v) for k, v in TRUTH.items()]))
    life.append(lv); allerr.append(ae)
life, allerr = np.array(life), np.array(allerr)

fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(12.6, 4.0))

# ---- A: per-seed lifetime ----
for i, (lab, _p, col) in enumerate(CFG):
    axA.scatter([i] * 3, life[i], s=70, color=col, alpha=0.9, linewidths=0, zorder=3)
    axA.plot([i - 0.22, i + 0.22], [life[i].mean()] * 2, color=INK, lw=2, zorder=4)
axA.axhline(0, color=MUTED, lw=1.2, ls=(0, (4, 3)), zorder=2)
axA.set_xticks(range(4)); axA.set_xticklabels([c[0] for c in CFG], fontsize=8)
axA.set_ylabel('lifetime error  (%)')
axA.set_title('Every config has one seed at $-$45 %\nand two at +27…+92 %', fontsize=9.5, pad=6)
axA.grid(axis='y', color=GRID, lw=0.6); axA.set_axisbelow(True)
for sp in ('top', 'right'):
    axA.spines[sp].set_visible(False)
axA.text(0.02, 0.97, 'black bar = mean of 3 seeds', transform=axA.transAxes,
         fontsize=7.5, color=INK2, va='top')

# ---- B: mean |err| with seed error bars ----
m, e = allerr.mean(1), allerr.std(1)
axB.bar(range(4), m, 0.6, yerr=e, capsize=4, color=[c[2] for c in CFG], zorder=3,
        error_kw=dict(ecolor=INK2, lw=1.2))
axB.set_xticks(range(4)); axB.set_xticklabels([c[0] for c in CFG], fontsize=8)
axB.set_ylabel('mean |error| over the 5 calib params  (%)')
axB.set_title('No configuration is resolved\n(error bars = seed sd, n=3)', fontsize=9.5, pad=6)
axB.grid(axis='y', color=GRID, lw=0.6); axB.set_axisbelow(True)
for sp in ('top', 'right'):
    axB.spines[sp].set_visible(False)
for i, v in enumerate(m):
    axB.text(i, v + e[i] + 0.4, '%.1f' % v, ha='center', fontsize=8, color=INK)

# ---- C: does lowering w transfer? ----
# toy numbers are the measured values recorded in sections 6.4 and 9.2 of the report
toy = [25.8, 6.2]          # 1270 cm toy, calibration at truth: w=1e5 -> w=3000
prod = [m[0], m[1]]        # production, perseg: w=1e5 -> w=3000
x = np.arange(2)
axC.plot(x, toy, marker='o', ms=8, lw=2.4, color=VIOLET, zorder=3,
         label='TOY (1270 cm, true geometry,\ncalibration started at truth)')
axC.plot(x, prod, marker='s', ms=8, lw=2.4, color=RED, zorder=3,
         label='PRODUCTION (SCIMODE=full,\nfitted positions from linear guess)')
for xi, yi in zip(x, toy):
    axC.text(xi, yi + 1.2, '%.1f' % yi, ha='center', fontsize=8, color=VIOLET)
for xi, yi in zip(x, prod):
    axC.text(xi, yi - 2.0, '%.1f' % yi, ha='center', fontsize=8, color=RED)
axC.set_xticks(x); axC.set_xticklabels(['$w=10^5$\n(current)', '$w=3000$'])
axC.set_xlim(-0.35, 1.35)
axC.set_ylabel('mean |calibration error|  (%)')
axC.set_title('The toy gain does NOT transfer', fontsize=9.5, pad=6)
axC.grid(axis='y', color=GRID, lw=0.6); axC.set_axisbelow(True)
for sp in ('top', 'right'):
    axC.spines[sp].set_visible(False)
leg = axC.legend(frameon=False, fontsize=7.4, loc='lower left', bbox_to_anchor=(-0.02, -0.03))
for t in leg.get_texts():
    t.set_color(INK2)
axC.set_ylim(0, 31)
axC.text(1.12, toy[1], '$%+.1f$' % (toy[1] - toy[0]), fontsize=9, color=VIOLET,
         va='center', ha='left', fontweight='bold')
axC.text(1.12, prod[1], '$%+.1f$' % (prod[1] - prod[0]), fontsize=9, color=RED,
         va='center', ha='left', fontweight='bold')
axC.set_xlim(-0.35, 1.6)

fig.suptitle('Production-scale validation: the toy ranking does not reproduce, and the seed '
             'scatter makes it unresolvable', fontsize=10.5, y=1.02, color=INK)
fig.tight_layout()
fig.savefig('plots/fig_production_scan.png', dpi=200, bbox_inches='tight')
print('wrote plots/fig_production_scan.png')
for i, (lab, _p, _c) in enumerate(CFG):
    print('  %-24s lifetime %s  mean|err| %.2f +- %.2f'
          % (lab.replace('\n', ' '), np.array2string(life[i], precision=1), m[i], e[i]))
