"""Accuracy of the longitudinal-diffusion template interpolation (paper Sec. II, 9 Oct 2026).
Exact = induced-current response (response_44.npy) convolved with the normalised discrete Gaussian of width sigma
(as consts_jax.load_lut does for the grid nodes); compared with linear and with quadratic (3-node Lagrange, as
sim_jax.simulate_signals) interpolation between the grid templates (100 nodes, 0.001-10 ticks), for sigma in
0.5-3 ticks (the drift range of the fits) on six transverse bins. Metrics: max |interp - exact| / peak over time,
and the same for d/dsigma (central differences, h = 1e-3 tick).
Result: template error median 1.9e-4 (linear) vs 1.3e-5 (quadratic) of the peak, max 4.0e-3 vs 3.1e-3;
derivative error median 0.7% vs 0.08%, max 14% vs 2.5%."""
import numpy as np
from scipy.stats import norm
R = np.load('/sdf/group/neutrino/pgranger/larnd-sim-jax/src/larndsim/detector_properties/response_44.npy')
grid = np.linspace(0.001, 10, 100); ext = 20
kern = lambda s: (lambda g: g / g.sum())(norm.pdf(np.arange(-ext, ext + 1), scale=s))
exact = lambda r, s: np.convolve(r, kern(s), mode='same')
def quad(s, T):
    i = int(np.clip(np.searchsorted(grid, s), 1, len(grid) - 2)); x0, x1, x2 = grid[i - 1], grid[i], grid[i + 1]
    a = (s - x1) * (s - x2) / ((x0 - x1) * (x0 - x2)); b = (s - x0) * (s - x2) / ((x1 - x0) * (x1 - x2)); c = (s - x0) * (s - x1) / ((x2 - x0) * (x2 - x1))
    return a * T[i - 1] + b * T[i] + c * T[i + 1]
def lin(s, T):
    i = int(np.clip(np.searchsorted(grid, s), 1, len(grid) - 1)); f = (s - grid[i - 1]) / (grid[i] - grid[i - 1])
    return (1 - f) * T[i - 1] + f * T[i]
rng = np.random.default_rng(1)
sig = np.concatenate([0.5 * (grid[1:] + grid[:-1])[(grid[1:] > 0.5) & (grid[1:] < 3.2)], rng.uniform(0.5, 3.0, 60)])
out = {k: ([], []) for k in ('linear', 'quadratic')}
for ix, iy in [(0, 0), (2, 3), (4, 4), (5, 0), (9, 9), (12, 3)]:
    r = R[ix, iy]
    if not np.abs(r).max(): continue
    T = np.stack([exact(r, s) if k else r for k, s in enumerate(grid)])
    for s in sig:
        e = exact(r, s); de = (exact(r, s + 1e-3) - exact(r, s - 1e-3)) / 2e-3
        for k, f in (('linear', lin), ('quadratic', quad)):
            out[k][0].append(np.abs(f(s, T) - e).max() / np.abs(e).max())
            out[k][1].append(np.abs((f(s + 1e-3, T) - f(s - 1e-3, T)) / 2e-3 - de).max() / np.abs(de).max())
for k, (a, d) in out.items():
    print(f'{k:9s}: template median {np.median(a):.2e} max {np.max(a):.2e} | derivative median {np.median(d):.2e} max {np.max(d):.2e} (n={len(a)})')
