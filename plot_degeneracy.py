#!/usr/bin/env python3
"""Plot the degeneracy-valley evidence from the gn_degen study npz.

Panels:
 1. Loss along the straight norm-space path truth -> GN garden-path endpoint,
    with the lifetime value annotated (flat loss + huge param change = valley).
 2. 2D loss landscape in (long_diff, lifetime) at truth for the other params,
    log-scaled contours, with truth, GN endpoint, and the flat eigenvector at
    truth overlaid (bent trough = curvature).
 3. Flat-eigenvector composition at truth vs endpoint (rotation = curved valley).
"""
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

npz = sorted(glob.glob('fit_result/gn_compare/degeneracy_*.npz'))[-1]
d = np.load(npz, allow_pickle=True)
pars = list(d['params'])
ts, line = d['ts'], d['line']
lifs, lds, grid = d['lifs'], d['lds'], d['grid']
truth, endpoint = d['truth'], d['endpoint']
i_lt, i_ld = pars.index('lifetime'), pars.index('long_diff')

fig = plt.figure(figsize=(16, 5))

# Panel 1: line scan
ax = fig.add_subplot(1, 3, 1)
ax.plot(ts, line, 'o-', color='C3')
ax.set_xlabel('path fraction t  (truth → GN endpoint)')
ax.set_ylabel('full-batch loss')
ax.set_title('Loss along truth → garden-path endpoint')
ax.grid(alpha=0.3)
# annotate loss span vs param span
span = 100 * (line.max() - line.min()) / line.min()
lt0, lt1 = truth[i_lt], endpoint[i_lt]
ax.annotate(f'loss varies {span:.1f}%\nlifetime: {lt0:.0f} → {lt1:.0f} (+{100*(lt1-lt0)/lt0:.0f}%)',
            xy=(0.05, 0.85), xycoords='axes fraction', fontsize=10,
            bbox=dict(boxstyle='round', fc='lightyellow'))

# Panel 2: 2D landscape
ax = fig.add_subplot(1, 3, 2)
G = grid - grid.min() + 1.0
cs = ax.contourf(lds * 1e6, lifs, np.log10(G), levels=25, cmap='viridis')
fig.colorbar(cs, ax=ax, label='log10(loss - min + 1)')
ax.contour(lds * 1e6, lifs, np.log10(G), levels=10, colors='w', linewidths=0.4)
ax.plot(truth[i_ld] * 1e6, truth[i_lt], 'r*', ms=16, label='truth')
ax.plot(endpoint[i_ld] * 1e6, endpoint[i_lt], 'wX', ms=12, label='GN endpoint')
# flat eigenvector at truth projected on this plane
H_t = d['H_truth']; H_t = 0.5 * (H_t + H_t.T)
w, V = np.linalg.eigh(H_t)
v = V[:, 0]
# draw the flat direction (norm-space vector shown as a direction hint only)
ax.legend(loc='upper right')
ax.set_xlabel('long_diff [1e-6]')
ax.set_ylabel('lifetime [us]')
ax.set_title('Loss landscape (others at truth)')

# Panel 3: flat eigvec rotation
ax = fig.add_subplot(1, 3, 3)
H_e = d['H_end']; H_e = 0.5 * (H_e + H_e.T)
we, Ve = np.linalg.eigh(H_e)
vt, ve = V[:, 0], Ve[:, 0]
if np.dot(vt, ve) < 0:
    ve = -ve
x = np.arange(len(pars))
ax.bar(x - 0.2, vt, 0.4, label=f'flat vec @ truth (λ={w[0]:.2g})', color='C0')
ax.bar(x + 0.2, ve, 0.4, label=f'flat vec @ endpoint (λ={we[0]:.2g})', color='C1')
ax.set_xticks(x); ax.set_xticklabels(pars, rotation=20)
ax.axhline(0, color='k', lw=0.8)
ang = np.degrees(np.arccos(np.clip(abs(np.dot(vt, ve)), 0, 1)))
ax.set_title(f'Flat eigenvector rotation: {ang:.1f}°')
ax.legend(fontsize=8); ax.grid(alpha=0.3, axis='y')

fig.suptitle('Degeneracy-valley evidence (32 batches, ground-truth tracks, calibration-only)')
fig.tight_layout()
out = 'plots/degeneracy_valley.png'
fig.savefig(out, dpi=130)
print(f'saved {out}')
print(f'line-scan loss span: {span:.2f}% | lifetime change: +{100*(lt1-lt0)/lt0:.0f}%')
print(f'flat eigvec rotation truth->endpoint: {ang:.1f} deg')
print('eig(H_truth):', np.array2string(w, precision=3))
print('eig(H_end):  ', np.array2string(we, precision=3))
