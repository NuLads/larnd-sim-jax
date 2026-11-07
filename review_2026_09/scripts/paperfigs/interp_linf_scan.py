"""Reproduce the old 'L_inf error vs long diff std' plot with the current code: max over time of |interpolated -
exactly convolved| template (central pixel bin, response_44.npy, absolute template units) vs sigma_L in ticks,
for the quadratic interpolation used in the code and for linear interpolation (9 Oct 2026)."""
import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.stats import norm
R = np.load('/sdf/group/neutrino/pgranger/larnd-sim-jax/src/larndsim/detector_properties/response_44.npy')
grid = np.linspace(0.001, 10, 100); ext = 20
kern = lambda s: (lambda g: g / g.sum())(norm.pdf(np.arange(-ext, ext + 1), scale=s))
r = R[0, 0]; T = np.stack([np.convolve(r, kern(s), mode='same') if k else r for k, s in enumerate(grid)])
def quad(s):
    i = int(np.clip(np.searchsorted(grid, s), 1, len(grid) - 2)); x0, x1, x2 = grid[i - 1], grid[i], grid[i + 1]
    return ((s-x1)*(s-x2)/((x0-x1)*(x0-x2)))*T[i-1] + ((s-x0)*(s-x2)/((x1-x0)*(x1-x2)))*T[i] + ((s-x0)*(s-x1)/((x2-x0)*(x2-x1)))*T[i+1]
def lin(s):
    i = int(np.clip(np.searchsorted(grid, s), 1, len(grid) - 1)); f = (s - grid[i-1]) / (grid[i] - grid[i-1]); return (1-f)*T[i-1] + f*T[i]
S = np.linspace(0.5, 4.9, 2000)
eq = np.array([np.abs(quad(s) - np.convolve(r, kern(s), mode='same')).max() for s in S])
el = np.array([np.abs(lin(s) - np.convolve(r, kern(s), mode='same')).max() for s in S])
print(f'peak of template {np.abs(r).max():.3f}; quadratic L_inf max {eq.max():.2e} at {S[eq.argmax()]:.2f} ticks, median {np.median(eq):.1e}; '
      f'linear L_inf max {el.max():.2e} at {S[el.argmax()]:.2f}, median {np.median(el):.1e}')
fig, ax = plt.subplots(figsize=(5, 3.4))
ax.scatter(S, el, s=4, color='0.6', label='linear'); ax.scatter(S, eq, s=4, color='C0', label='quadratic (code)')
ax.set_xlabel('Long diff std (nb ticks)'); ax.set_ylabel(r'$L_\infty$ error'); ax.legend(frameon=False)
fig.savefig('/sdf/scratch/neutrino/pgranger/review_2026_09/SAMPLING_STEP/interp_linf_scan.png', dpi=150, bbox_inches='tight')
