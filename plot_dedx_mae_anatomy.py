#!/usr/bin/env python3
"""Anatomy of the dEdx MAE degradation, and the minimum-length cut that fixes it.

A: MAE vs iteration, split by batch -- one batch improves, the other degrades 2.7x
B: |error| vs segment length -- the degradation is short segments, corr = -0.75
C: the minimum-length cut, before/after
"""
import glob
import pickle
import re

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

fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(12.6, 3.9))

# ---------------- A: per-batch MAE trajectory ----------------
h = pickle.load(open('fit_result/dedx_gn/history_iter610_dedxgn_adam_mw3000.0_r1_s0.pkl', 'rb'))
M = np.asarray(h['dedx_mae_iter'], float)
for b, col, lbl in [(0, AQUA, 'batch 0  (shortest segment 0.69 cm)'),
                    (1, RED, 'batch 1  (shortest segment 0.0355 cm)')]:
    sub = M[b::2]
    idx = np.arange(len(sub)) * 2 + b
    k = 15
    sm = np.convolve(sub, np.ones(k) / k, mode='valid')
    axA.plot(idx[k - 1:], sm, lw=2.0, color=col, label=lbl, zorder=3)
axA.set_xlabel('iteration')
axA.set_ylabel('dE/dx MAE vs truth  (MeV/cm)')
axA.set_title('The degradation is ONE batch\n(not global overfitting)', fontsize=9.5, pad=6)
axA.grid(color=GRID, lw=0.6); axA.set_axisbelow(True)
for sp in ('top', 'right'):
    axA.spines[sp].set_visible(False)
leg = axA.legend(frameon=False, fontsize=7.6, loc='upper left')
for t in leg.get_texts():
    t.set_color(INK2)
axA.annotate('improves', xy=(500, 0.033), xytext=(330, 0.062), fontsize=8, color=AQUA,
             arrowprops=dict(arrowstyle='->', color=AQUA, lw=1.2))
axA.annotate('degrades 2.7$\\times$', xy=(520, 0.150), xytext=(250, 0.120), fontsize=8, color=RED,
             arrowprops=dict(arrowstyle='->', color=RED, lw=1.2))

# ---------------- B: |error| vs segment length ----------------
sdc = h['seg_dx_cache']
fit = np.concatenate([np.exp(np.asarray(e['log_dedx'])) for _, e in sorted(h['dedx_cache'].items())])
tru = np.concatenate([np.asarray(h['true_dedx_cache'][b]) for b in sorted(h['true_dedx_cache'])])
dx = np.concatenate([np.asarray(sdc[b]) for b in sorted(sdc)])
n = min(len(fit), len(tru), len(dx))
fit, tru, dx = fit[:n], tru[:n], dx[:n]
ok = (dx > 0) & (tru > 0) & np.isfinite(fit)
fit, tru, dx = fit[ok], tru[ok], dx[ok]
err = np.abs(fit - tru)
r = np.corrcoef(np.log(dx), err)[0, 1]
axB.scatter(dx, err, s=34, color=BLUE, alpha=0.8, linewidths=0, zorder=3)
w = np.argmax(err)
axB.scatter([dx[w]], [err[w]], s=130, facecolors='none', edgecolors=RED, linewidths=1.8, zorder=4)
axB.annotate('0.0355 cm segment\n1044$\\times$ less curvature than the median\n$\\to$ an unidentifiable free parameter',
             xy=(dx[w], err[w]), xytext=(0.55, 0.30), textcoords='axes fraction',
             fontsize=7.6, color=INK2, ha='left',
             arrowprops=dict(arrowstyle='->', color=RED, lw=1.1))
axB.axvline(0.15, color=VIOLET, lw=1.6, ls=(0, (4, 3)), zorder=2)
axB.text(0.17, 2.2e-3, 'min-length\ncut 0.15 cm', fontsize=7.6, color=VIOLET, va='bottom')
axB.set_xscale('log'); axB.set_yscale('log')
axB.set_xlabel('segment length dx  (cm)')
axB.set_ylabel('|fitted $-$ true|  (MeV/cm)')
axB.set_title('Short segments carry the error\ncorr(log dx, |err|) = %+.2f' % r, fontsize=9.5, pad=6)
axB.grid(color=GRID, lw=0.6, which='both'); axB.set_axisbelow(True)
for sp in ('top', 'right'):
    axB.spines[sp].set_visible(False)

# ---------------- C: the cut ----------------
D = {}
for f in glob.glob('fit_result/dedx_rp/history_iter610_*_mdx*.pkl'):
    md = float(re.search(r'_mdx([\d.]+)_', f).group(1))
    hh = pickle.load(open(f, 'rb'))
    s = hh['seg_dx_cache']
    ae = []
    worst = 0
    for b, e in hh['dedx_cache'].items():
        ff = np.exp(np.asarray(e['log_dedx'])); tt = np.asarray(hh['true_dedx_cache'][b])
        dd = np.asarray(s[b])
        m = min(len(ff), len(tt), len(dd)); ff, tt, dd = ff[:m], tt[:m], dd[:m]
        kk = (dd > 0) & (tt > 0)
        ae.append(np.abs(ff[kk] - tt[kk]))
        worst = max(worst, float(np.abs(ff[kk] / tt[kk] - 1).max() * 100))
    D.setdefault(md, []).append([np.concatenate(ae).mean(), worst])
labels = ['cut OFF', 'cut at 0.15 cm']
mae = [np.array(D[0.0])[:, 0].mean(), np.array(D[0.15])[:, 0].mean()]
wst = [np.array(D[0.0])[:, 1].mean(), np.array(D[0.15])[:, 1].mean()]
x = np.arange(2)
b1 = axC.bar(x - 0.19, mae, 0.36, color=BLUE, label='mean MAE (MeV/cm)', zorder=3)
ax2 = axC.twinx()
b2 = ax2.bar(x + 0.19, wst, 0.36, color=ORANGE, label='worst segment error (%)', zorder=3)
axC.set_xticks(x); axC.set_xticklabels(labels)
axC.set_ylabel('mean MAE  (MeV/cm)', color=BLUE)
ax2.set_ylabel('worst segment error  (%)', color=ORANGE)
axC.tick_params(axis='y', colors=BLUE); ax2.tick_params(axis='y', colors=ORANGE)
ax2.spines['top'].set_visible(False); axC.spines['top'].set_visible(False)
for i, v in enumerate(mae):
    axC.text(i - 0.19, v * 1.03, '%.4f' % v, ha='center', fontsize=8, color=INK)
for i, v in enumerate(wst):
    ax2.text(i + 0.19, v * 1.03, '%.0f%%' % v, ha='center', fontsize=8, color=INK)
axC.set_title('Freezing sub-0.15 cm segments\nMAE $-$62 %, runaway gone', fontsize=9.5, pad=6)
axC.grid(axis='y', color=GRID, lw=0.6); axC.set_axisbelow(True)
axC.set_ylim(0, max(mae) * 1.25); ax2.set_ylim(0, max(wst) * 1.25)

fig.suptitle('The dE/dx MAE puzzle: one information-free segment, and the cut that fixes it',
             fontsize=10.5, y=1.02, color=INK)
fig.tight_layout()
fig.savefig('plots/fig_dedx_mae_anatomy.png', dpi=200, bbox_inches='tight')
print('wrote plots/fig_dedx_mae_anatomy.png')
print('  corr(log dx, |err|) = %+.3f ; worst %.4f at dx=%.4f cm' % (r, err.max(), dx[np.argmax(err)]))
print('  MAE off/cut = %.4f / %.4f ; worst%% off/cut = %.0f / %.0f' % (mae[0], mae[1], wst[0], wst[1]))
