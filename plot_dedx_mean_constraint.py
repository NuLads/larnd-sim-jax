#!/usr/bin/env python3
"""Figures for the dEdx mean-constraint investigation.

Panel A: true batch-to-batch scatter of the length-weighted mean dEdx vs batch length,
         against the sigma the production weight actually imposes.   [truth data, no fit]
Panel B: mean |calibration error| vs mean-constraint weight.          [18-arm toy]
Panel C: fitted dEdx mean bias vs mean-constraint weight.             [18-arm toy]
"""
import glob
import re
import pickle
from collections import defaultdict

import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
BLUE, ORANGE, RED, AQUA = "#2a78d6", "#eb6834", "#e34948", "#1baf7a"
plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 9,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": GRID, "axes.titlecolor": INK,
})

W_PROD = 1e5
TRUTH = {'Ab': 0.83488, 'eField': 0.52152, 'tran_diff': 1.0028e-05,
         'long_diff': 5.8142e-06, 'lifetime': 2406.4}
IN_H5 = '/sdf/data/neutrino/cyifan/diffsim_input/true_through_muon_edep_10cm_vol1cm.h5'

# ---------------------------------------------------------------- panel A data
with h5py.File(IN_H5, 'r') as f:
    s = f['segments']
    dedx, dx = np.asarray(s['dEdx']), np.asarray(s['dx'])
ok = (dx > 0) & np.isfinite(dedx) & (dedx > 0)
dedx, dx = dedx[ok], dx[ok]
GLOBAL = (dedx * dx).sum() / dx.sum()
cum = np.cumsum(dx)
LENS = [50, 100, 200, 400, 1000]
spread = []
for L in LENS:
    idx = np.searchsorted(cum, np.arange(L, cum[-1], L))
    parts = np.split(np.arange(len(dx)), idx)
    ms = np.array([(dedx[p] * dx[p]).sum() / dx[p].sum()
                   for p in parts if len(p) > 1 and dx[p].sum() > 0.5 * L])
    spread.append(100 * ms.std() / ms.mean())
spread = np.array(spread)
imposed = 100 / np.sqrt(2 * W_PROD) / GLOBAL

# ---------------------------------------------------------------- panel B/C data
D = defaultdict(list)
for f in sorted(glob.glob('fit_result/dedx_bias/history_iter8000_dedxbias_*.pkl')):
    m = re.match(r'loc([\d.]+)_jac(\d)_mw([\d.]+)_off([\d.]+)_s(\d+)',
                 re.search(r'dedxbias_(.+)\.pkl', f).group(1))
    loc, mw = float(m.group(1)), float(m.group(3))
    h = pickle.load(open(f, 'rb'))
    sdc = h.get('seg_dx_cache') or {}
    nf = nt = den = 0.0
    for b, e in h['dedx_cache'].items():
        fit = np.exp(np.asarray(e['log_dedx']))
        tru = np.asarray(h['true_dedx_cache'][b])
        sdx = np.asarray(sdc[b])
        n = min(len(fit), len(tru), len(sdx))
        fit, tru, sdx = fit[:n], tru[:n], sdx[:n]
        m2 = (sdx > 0) & (tru > 0) & np.isfinite(fit)
        nf += (fit[m2] * sdx[m2]).sum(); nt += (tru[m2] * sdx[m2]).sum(); den += sdx[m2].sum()
    cur = h['current_params_state']
    cal = np.array([100 * (float(np.asarray(cur[k]).ravel()[0]) - TRUTH[k]) / TRUTH[k]
                    for k in TRUTH])
    D[(loc, mw)].append((100 * (nf / nt - 1), np.abs(cal).mean()))

MWS = [1e5, 3000.0, 0.0]
XL = ['$10^5$\n(current)', '3000\n(measured)', '0\n(none)']
LOCS = [(1.874, BLUE, 'o', 'loc = 1.874  (prior mode, current)'),
        (1.8847, ORANGE, 's', 'loc = 1.8847  (prior mean)')]

fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(12.4, 3.9))

# ---- A -----------------------------------------------------------------
axA.plot(LENS, spread, marker='o', ms=5.5, lw=1.8, color=AQUA, zorder=3,
         label='TRUE batch-to-batch $\\sigma$ of the mean')
axA.axhline(imposed, color=RED, lw=1.8, ls=(0, (4, 3)), zorder=3,
            label='$\\sigma$ IMPOSED by $w=10^5$')
axA.annotate('', xy=(400, spread[3]), xytext=(400, imposed),
             arrowprops=dict(arrowstyle='<->', color=INK2, lw=1.1))
axA.text(430, np.sqrt(spread[3] * imposed), '6$\\times$\ntoo stiff', fontsize=8.5,
         color=INK, va='center')
axA.set_xscale('log'); axA.set_yscale('log')
axA.set_xticks(LENS); axA.set_xticklabels([str(l) for l in LENS])
axA.set_xlabel('batch length (cm)')
axA.set_ylabel('$\\sigma$ on the batch mean dE/dx  (%)')
axA.set_title('The constraint is stiffer than reality\n(truth data, 3.0M segments, no fit)',
              fontsize=9.5, pad=6)
axA.set_ylim(0.075, 2.6)
axA.grid(color=GRID, lw=0.6, which='both'); axA.set_axisbelow(True)
leg = axA.legend(frameon=False, fontsize=7.6, loc='upper right')
for t in leg.get_texts(): t.set_color(INK2)

# ---- B and C -----------------------------------------------------------
for ax, idx, ttl, ylab in [
        (axB, 1, 'Stiff constraint WRECKS calibration', 'mean |calibration error|  (%)'),
        (axC, 0, 'Fitted dE/dx mean bias', 'bias vs this data\'s true mean  (%)')]:
    for loc, col, mk, lbl in LOCS:
        y = [np.array(D[(loc, w)])[:, idx].mean() for w in MWS]
        e = [np.array(D[(loc, w)])[:, idx].std() for w in MWS]
        ax.errorbar(range(3), y, yerr=e, marker=mk, ms=6, lw=1.8, color=col,
                    capsize=3, zorder=3, label=lbl)
    ax.set_xticks(range(3)); ax.set_xticklabels(XL)
    ax.set_xlim(-0.35, 2.35)
    ax.set_xlabel('dedx_mean_constraint_weight')
    ax.set_ylabel(ylab)
    ax.set_title(ttl, fontsize=9.5, pad=6)
    ax.grid(axis='y', color=GRID, lw=0.6); ax.set_axisbelow(True)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)

axC.axhline(0, color=MUTED, lw=1, ls=(0, (4, 3)), zorder=2)
axC.text(0.03, 0.06, 'negative = mode-shrinkage bias\n(prior pulls the mean to the MPV)',
         transform=axC.transAxes, fontsize=7.5, color=INK2, ha='left', va='bottom')
leg = axB.legend(frameon=False, fontsize=7.8, loc='upper right')
for t in leg.get_texts(): t.set_color(INK2)
axB.text(0.03, 0.55, 'calibration STARTED\nAT TRUTH — all of this\nis injected bias',
         transform=axB.transAxes, fontsize=7.5, color=INK2, va='top')

fig.suptitle('The dE/dx mean constraint: right target, wrong stiffness  '
             '(17 batches, 376 segments, 1270 cm, 8000 iters, 3 seeds/arm)',
             fontsize=10.5, y=1.03, color=INK)
fig.tight_layout()
fig.savefig('plots/fig_dedx_mean_constraint.png', dpi=200, bbox_inches='tight')
print('wrote plots/fig_dedx_mean_constraint.png')
print('global length-weighted mean = %.4f ; imposed sigma = %.3f%% ; spread = %s'
      % (GLOBAL, imposed, np.array2string(spread, precision=2)))
