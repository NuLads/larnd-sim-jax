"""Paper figure for subsec:templates ("Induced-current templates and longitudinal diffusion").

main.tex (search `subsec:templates`) explains that the induced current on a pixel is read from a
lookup table indexed by [longitudinal-diffusion template, transverse position, time], that the
longitudinal-diffusion axis is handled by TABULATION rather than convolution (templates
precomputed on a grid of sigma_L values expressed in time ticks), and that the response at an
arbitrary sigma_L is obtained by QUADRATIC Lagrange interpolation through the three nearest
nodes, T(sigma_L) = a*T_{k-1} + b*T_k + c*T_{k+1}, chosen over linear interpolation specifically
because a linear interpolant's derivative wrt sigma_L is piecewise constant (jumps at every node)
-- "precisely the quantity the fit uses to constrain the longitudinal diffusion" -- whereas the
quadratic one is smooth. An author comment there asks for "a figure to show the initial signal
and the diffused one". This script builds that figure, plus (per the task's optional panel (c))
the transverse sub-pixel projection weights of Eq. tran_diff_weights (subsec:projection).

Everything is computed from the real, loaded code/tables -- nothing is hand-drawn:
  - `paper_common.params()` -> `consts_jax.load_detector_properties` + `consts_jax.load_lut`,
    i.e. the SAME `response_template` array (shape (100, 45, 45, 1891): 100 pre-tabulated
    sigma_L templates x 45x45 folded transverse sub-pixel bins x 1891 time ticks) and the same
    `params.long_diff_template` sigma_L grid (default `jnp.linspace(0.001, 10, 100)` ticks --
    grepped the whole repo: nothing overrides this, so it IS the production grid) used by
    `sim_jax.simulate_wfs` for every fit in this paper.
  - The quadratic interpolation (panels a-c) is `T_QUAD` below, a line-for-line port of
    `sim_jax.simulate_signals` lines ~205-211/229-233 (same variable names idx/x0/x1/x2/a/b/c),
    just applied directly to `response_template` instead of to a batch of electrons -- so it can
    be swept continuously in sigma_L and differentiated with `jax.grad` (the literal autodiff
    path the fit's backward pass uses). Verified below against two EXACT algebraic properties of
    Lagrange interpolation (reduces to the tabulated value at a node; a+b+c=1 identically) before
    it is used for anything in the figure.
  - The "linear interpolation" foil (the alternative the paper argues against) is the standard
    piecewise-linear spline through the SAME nodes, `jnp.interp` -- not a hand-drawn comparison.
  - Panel (d) calls the actual `detsim_jax.density_2d` (== Eq. tran_diff_weights, confirmed by
    inspection: two 1D Gaussian-CDF differences via `gaussian_1d_integral_new`, combined as an
    outer product == the 1/4*erf*erf product of the paper's Eq. up to the trivial algebraic
    factor of 2 absorbed into each 1D integral already summing to 1).
  - The transverse-position choice for panels (a)-(c), currents_idx=(0,0), is not a guess: in
    `sim_jax.get_bin_shifts` + `simulate_drift_new`, the absolute sub-pixel bin index b (from the
    TPC edge) is folded as `|(b % n_sub) - n_sub/2 + 1/2|`; a charge exactly at a pixel's own
    centre has (offset from the pixel's lower edge) = pixel_pitch/2 = n_sub/2 bins, i.e.
    `b % n_sub = n_sub/2 = 5`, which folds to `|5-5+0.5|=0.5 -> astype(int) -> 0`. So (0,0) is
    exactly "a point charge on the collecting pixel, centred on it", as the task asks for.

Regenerate with:
    JAX_PLATFORMS=cpu python fig_diffusion_templates.py
Writes:
    /sdf/group/neutrino/pgranger/NuLads_draft/figures/diffusion_templates.pdf
    /sdf/group/neutrino/pgranger/NuLads_draft/figures/diffusion_templates.png
and prints the grid/physics numbers used, and the two Lagrange self-consistency checks.

CPU only; creates only the two output figure files; does not edit any tracked file; no git, no
SLURM.
"""
import os
os.environ.setdefault('JAX_PLATFORMS', 'cpu')  # must precede `import jax` anywhere (incl. via paper_common)
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib import cm

import matplotlib.patheffects as pe
import jax
import jax.numpy as jnp

sys.path.insert(0, '/sdf/scratch/neutrino/pgranger/review_2026_09/PAPERJOBS')
import paper_common as pc  # noqa: E402  (does os.chdir to the repo on import)

from larndsim.consts_jax import get_vdrift  # noqa: E402
from larndsim.detsim_jax import density_2d  # noqa: E402  (the real Eq. tran_diff_weights)

OUT_DIR = '/sdf/group/neutrino/pgranger/NuLads_draft/figures'

# ----------------------------------------------------------------------
# Style (shared paper style guide, matching review_2026_09/scripts/paperfigs/fig_mc_sampling.py)
# ----------------------------------------------------------------------
C_BLUE, C_VERM, C_GREEN, C_ORANGE, C_SKY, C_PURP, C_BLACK = (
    '#0072B2', '#D55E00', '#009E73', '#E69F00', '#56B4E9', '#CC79A7', '#000000')

plt.rcParams.update({
    'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8,
    'legend.fontsize': 7, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
    'axes.linewidth': 0.6, 'lines.linewidth': 1.2,
    'font.family': 'sans-serif',
})


LEGEND_KW = dict(frameon=True, edgecolor='none', facecolor='white', framealpha=0.82)  # "frameless" = no border


def panel_label(ax, text, loc='upper left'):
    xy = {'upper left': (0.04, 0.95), 'upper right': (0.96, 0.95)}[loc]
    ha = 'left' if 'left' in loc else 'right'
    ax.text(xy[0], xy[1], text, transform=ax.transAxes, fontsize=7, va='top', ha=ha,
            path_effects=[pe.withStroke(linewidth=2.2, foreground='white')], zorder=10)


# ----------------------------------------------------------------------
# Load the production detector parameters and response LUT (the real code/tables)
# ----------------------------------------------------------------------
print('Loading detector parameters and response LUT (paper_common.params()) ...')
truth, response = pc.params()
response = np.asarray(response)  # (Ntemplates=100, Nx=45, Ny=45, Nt=1891); keep a numpy copy for slicing

template_vals_jax = truth.long_diff_template          # (100,) sigma_L grid, in TIME TICKS
template_vals = np.asarray(template_vals_jax)
vdrift = float(get_vdrift(truth))                      # cm/us
t_sampling = float(truth.t_sampling)                   # us / tick
D_L = float(truth.long_diff)                            # cm^2/us
D_T = float(truth.tran_diff)                            # cm^2/us
drift_length = float(truth.drift_length)                # cm
pixel_pitch = float(truth.pixel_pitch)                   # cm
nb_sub = int(truth.nb_sampling_bins_per_pixel)           # n_sub = 10
nb_T = int(truth.nb_tran_diff_bins)                      # n_T = 5
t_drift_max = drift_length / vdrift                      # us, full-drift time

Ntemplates, Nx, Ny, Nt = response.shape
node_spacing = float(template_vals[1] - template_vals[0])

print(f'  response_template shape = {response.shape}  (Ntemplates, Nx, Ny, Nt)')
print(f'  long_diff_template grid: {Ntemplates} nodes, sigma_L in [{template_vals[0]:.4f}, '
      f'{template_vals[-1]:.3f}] ticks, spacing {node_spacing:.5f} ticks (uniform)')
print(f'  vdrift={vdrift:.5f} cm/us, t_sampling={t_sampling} us/tick, drift_length={drift_length:.3f} cm, '
      f'full-drift time={t_drift_max:.2f} us')
print(f'  D_L={D_L:.4e} cm^2/us, D_T={D_T:.4e} cm^2/us, pixel_pitch={pixel_pitch} cm, '
      f'n_sub={nb_sub}, n_T={nb_T}')


# ----------------------------------------------------------------------
# T_QUAD: exact port of sim_jax.simulate_signals' quadratic Lagrange interpolation in sigma_L
# (lines ~205-211 and ~229-233 of src/larndsim/sim_jax.py). `curve_table`'s leading axis must be
# the template axis (size Ntemplates); any trailing shape is carried through unchanged, so the
# SAME function serves panel (a) (scalar sigma, curve_table = response[:, 0, 0, :], full time
# axis carried through) and panels (b)/(c) (curve_table = response[:, 0, 0, tick] in a jax.vmap
# over an array of sigma).
# ----------------------------------------------------------------------
def T_quad(sigma, curve_table, template_vals=template_vals_jax):
    idx = jnp.clip(jnp.searchsorted(template_vals, sigma), 1, template_vals.shape[0] - 2)
    x0, x1, x2 = template_vals[idx - 1], template_vals[idx], template_vals[idx + 1]
    a = (sigma - x1) * (sigma - x2) / ((x0 - x1) * (x0 - x2))
    b = (sigma - x0) * (sigma - x2) / ((x1 - x0) * (x1 - x2))
    c = (sigma - x0) * (sigma - x1) / ((x2 - x0) * (x2 - x1))
    T0, T1, T2 = curve_table[idx - 1], curve_table[idx], curve_table[idx + 1]
    return a * T0 + b * T1 + c * T2


def T_lin(sigma, curve_table, template_vals=template_vals_jax):
    """The alternative the paper argues against: a standard piecewise-LINEAR spline through the
    same tabulated nodes (not production code -- there is no linear path in sim_jax.py -- but a
    standard, non-hand-drawn numerical method, used here only as the foil in panel (c))."""
    return jnp.interp(sigma, template_vals, curve_table)


# --- self-consistency checks of T_quad against the two exact algebraic properties of a quadratic
# Lagrange interpolant, BEFORE it is used for anything in the figure ---
print('\nValidating T_quad against sim_jax.simulate_signals Lagrange formula ...')
rng_check = np.random.default_rng(0)
curve_check = rng_check.normal(size=Ntemplates).astype(np.float64)  # arbitrary synthetic "template"
max_node_err, max_unity_err = 0.0, 0.0
for k in range(1, Ntemplates - 1):
    v_node = float(T_quad(jnp.asarray(template_vals[k]), jnp.asarray(curve_check)))
    max_node_err = max(max_node_err, abs(v_node - curve_check[k]))
sigma_probe = jnp.linspace(template_vals[1], template_vals[-2], 4001)
idx_p = jnp.clip(jnp.searchsorted(template_vals_jax, sigma_probe), 1, Ntemplates - 2)
x0p, x1p, x2p = template_vals_jax[idx_p - 1], template_vals_jax[idx_p], template_vals_jax[idx_p + 1]
a_p = (sigma_probe - x1p) * (sigma_probe - x2p) / ((x0p - x1p) * (x0p - x2p))
b_p = (sigma_probe - x0p) * (sigma_probe - x2p) / ((x1p - x0p) * (x1p - x2p))
c_p = (sigma_probe - x0p) * (sigma_probe - x1p) / ((x2p - x0p) * (x2p - x1p))
max_unity_err = float(jnp.max(jnp.abs(a_p + b_p + c_p - 1.0)))
print(f'  max |T_quad(node_k) - table[k]| over all interior nodes: {max_node_err:.3e}  (expect ~0, float32)')
print(f'  max |a+b+c - 1| over {sigma_probe.shape[0]} probe sigma values: {max_unity_err:.3e}  (expect ~0, float32)')
# threshold is float32-precision-limited (~1e-7), not exact-zero, since JAX defaults to float32
assert max_node_err < 1e-5 and max_unity_err < 1e-5, 'T_quad does not reproduce the Lagrange identities -- bug'
print('  PASSED: T_quad exactly reproduces the tabulated nodes and is an exact partition of unity (to float32 precision).')


# ----------------------------------------------------------------------
# Panel (a) physics: undiffused template + 3 physical sigma_L values spanning the drift volume
# ----------------------------------------------------------------------
IX0, IY0 = 0, 0  # "a point charge on the collecting pixel, centred on it" -- see module docstring
DRIFT_FRACS = [0.1, 0.5, 1.0]   # "short", "mid", "full" drift, as fractions of the full-drift time
DRIFT_LABELS = ['short', 'mid', 'full']

t_drift_vals = [f * t_drift_max for f in DRIFT_FRACS]
sigma_L_cm_vals = [float(np.sqrt(2 * D_L * t)) for t in t_drift_vals]
sigma_L_ticks_vals = [s / vdrift / t_sampling for s in sigma_L_cm_vals]

print('\nPanel (a): drift times / sigma_L used (D_L={:.4e} cm^2/us):'.format(D_L))
for lab, td, sl_cm, sl_tk in zip(DRIFT_LABELS, t_drift_vals, sigma_L_cm_vals, sigma_L_ticks_vals):
    print(f'  {lab:5s}: t_drift={td:7.2f} us  sigma_L={sl_cm*1e4:6.1f} um = {sl_cm:.5f} cm '
          f'= {sl_tk:.4f} ticks')

curve_table_time = jnp.asarray(response[:, IX0, IY0, :])       # (Ntemplates, Nt)
curve_undiffused = response[0, IX0, IY0, :]                    # raw LUT entry, NO interpolation
curves_diffused = [np.asarray(T_quad(jnp.asarray(sl), curve_table_time)) for sl in sigma_L_ticks_vals]

peak_tick_undiff = int(np.argmax(curve_undiffused))
for lab, c in zip(DRIFT_LABELS, curves_diffused):
    print(f'  {lab:5s}: peak template value = {c.max():.4f} (undiffused peak = {curve_undiffused.max():.4f}, '
          f'{100*(1-c.max()/curve_undiffused.max()):+.1f}% change)')

# time axis: ticks relative to the undiffused pulse's cutoff, converted to microseconds
T0_TICK = peak_tick_undiff + 1
TICK_LO, TICK_HI = peak_tick_undiff - 32, Nt  # ~3.2 us before cutoff to the array edge
tick_axis = np.arange(TICK_LO, TICK_HI)
time_axis_us = (tick_axis - T0_TICK) * t_sampling

viridis_pts = [0.15, 0.55, 0.88]
DIFF_COLORS = [cm.viridis(p) for p in viridis_pts]


# ----------------------------------------------------------------------
# Panels (b)/(c) physics: pick 3 fixed ticks programmatically (not by hand) -- one on the rising
# edge, one at the undiffused template's cutoff (its peak), one just past it (the "shoulder" that
# only exists once there is diffusion) -- so that T(sigma_L) at fixed time has visibly different
# behaviour (decreasing / increasing) across the panel, which is exactly the signal the fit uses.
# ----------------------------------------------------------------------
rise_tick = int(np.searchsorted(curve_undiffused[:peak_tick_undiff], 0.2 * curve_undiffused.max()))
FIXED_TICKS = [rise_tick, peak_tick_undiff, peak_tick_undiff + 3]
FIXED_LABELS = ['rising edge', 'cutoff', 'shoulder']
FIXED_COLORS = [C_BLUE, C_VERM, C_GREEN]

SIG_MAX = 1.15 * sigma_L_ticks_vals[-1]  # a bit beyond the full-drift sigma_L
sigma_grid = jnp.linspace(1e-6, SIG_MAX, 2000)

print(f'\nPanels (b)/(c): fixed ticks = {FIXED_TICKS} '
      f'(rising edge / undiffused cutoff / post-cutoff shoulder), sigma_L grid up to {SIG_MAX:.3f} ticks')

value_curves, deriv_quad_curves, deriv_lin_curves = [], [], []
for tck in FIXED_TICKS:
    ctab = jnp.asarray(response[:, IX0, IY0, tck])
    val = jax.vmap(lambda s: T_quad(s, ctab))(sigma_grid)
    dq = jax.vmap(jax.grad(lambda s: T_quad(s, ctab)))(sigma_grid)
    dl = jax.vmap(jax.grad(lambda s: T_lin(s, ctab)))(sigma_grid)
    value_curves.append(np.asarray(val))
    deriv_quad_curves.append(np.asarray(dq))
    deriv_lin_curves.append(np.asarray(dl))

# tabulated nodes within the plotted sigma range, for the "nodes marked" request
node_mask = (template_vals >= float(sigma_grid[0])) & (template_vals <= float(sigma_grid[-1]))
node_sigma = template_vals[node_mask]


# ----------------------------------------------------------------------
# Panel (d) physics: transverse sub-pixel weights, Eq. tran_diff_weights, via the real density_2d
# ----------------------------------------------------------------------
width = pixel_pitch / nb_sub                      # sub-pixel bin width (cm), n_sub bins per pixel
nb_sym = (nb_T - 1) // 2                           # = 2 for n_T = 5
bin_edges = jnp.linspace(-nb_sym * width, (nb_sym + 1) * width, nb_T + 1)  # exactly as in simulate_drift_new

t_drift_heatmap = 0.5 * t_drift_max
sigma_T_heatmap = float(np.sqrt(2 * D_T * t_drift_heatmap))
X0_OFFSET, Y0_OFFSET = 0.30 * width, 0.65 * width   # an off-centre offset within the home bin, (x0,y0) in [0,width)

weights = density_2d(bin_edges, jnp.asarray([X0_OFFSET]), jnp.asarray([Y0_OFFSET]), jnp.asarray([sigma_T_heatmap]))
weights = np.asarray(weights[0])  # (n_T, n_T)

print(f'\nPanel (d): sigma_T at mid-drift ({t_drift_heatmap:.1f} us) = {sigma_T_heatmap*1e4:.1f} um '
      f'= {sigma_T_heatmap:.5f} cm (bin width={width*1e4:.1f} um); offset (x0,y0)=({X0_OFFSET*1e4:.1f},'
      f' {Y0_OFFSET*1e4:.1f}) um; sum(weights)={weights.sum():.6f} (expect 1 exactly)')
assert abs(weights.sum() - 1.0) < 1e-6


# ======================================================================
# Figure: (a) full width on top, (b)(c)(d) below
# ======================================================================
fig = plt.figure(figsize=(7.0, 4.7))
gs = GridSpec(2, 3, height_ratios=[1.0, 1.05], hspace=0.40, wspace=0.42, figure=fig)

# --- (a) induced-current template vs time, undiffused + 3 diffusion widths ---
ax = fig.add_subplot(gs[0, :])
ax.plot(time_axis_us, curve_undiffused[TICK_LO:TICK_HI], color=C_BLACK, linestyle='--',
        linewidth=0.7, marker='o', markersize=2.2, markeredgewidth=0, zorder=5, label=r'no diffusion ($\sigma_L=0$)')
for lab, td, sl_cm, curve, color in zip(DRIFT_LABELS, t_drift_vals, sigma_L_cm_vals, curves_diffused, DIFF_COLORS):
    ax.plot(time_axis_us, curve[TICK_LO:TICK_HI], color=color, linewidth=0.7, marker='o', markersize=2.2,
            markeredgewidth=0, zorder=4,
            label=fr'{lab} drift ($t_{{\rm drift}}={td:.0f}\,\mu$s, $\sigma_L={sl_cm*1e4:.0f}\,\mu$m)')
ax.set_xlabel(r'time relative to the undiffused pulse cutoff ($\mu$s); points at the $0.1~\mu$s tick')
ax.set_ylabel('induced current (arb. units)')
ax.set_xlim(time_axis_us[0], time_axis_us[-1])
ax.set_ylim(top=curve_undiffused.max() * 1.22)
ax.axhline(0, color='0.6', linewidth=0.5, zorder=1)
ax.legend(loc='upper left', bbox_to_anchor=(0.0, 0.90), handletextpad=0.5, labelspacing=0.3, borderaxespad=0.4, **LEGEND_KW)
ax.grid(alpha=0.25, linewidth=0.5)
panel_label(ax, '(a)')

# --- (b) template value at fixed ticks vs sigma_L, nodes marked ---
ax = fig.add_subplot(gs[1, 0])
for tck, val, color, lab in zip(FIXED_TICKS, value_curves, FIXED_COLORS, FIXED_LABELS):
    ax.plot(np.asarray(sigma_grid), val, color=color, linewidth=1.3,
            label=lab)
    node_vals = response[:, IX0, IY0, tck][node_mask]
    ax.plot(node_sigma, node_vals, 'o', color=color, markersize=2.4, markeredgewidth=0, zorder=5)
for sl, lab in zip(sigma_L_ticks_vals, DRIFT_LABELS):
    ax.axvline(sl, color='0.75', linewidth=0.6, linestyle=':', zorder=1)
ax.set_xlabel(r'$\sigma_L$ (ticks)')
ax.set_ylabel('template value (arb. units)')
ax.set_xlim(0, float(sigma_grid[-1]))
ax.legend(loc='center left', bbox_to_anchor=(0.0, 0.43), handletextpad=0.4, labelspacing=0.25, borderaxespad=0.3, **LEGEND_KW)
ax.grid(alpha=0.25, linewidth=0.5)
panel_label(ax, '(b)')

# --- (c) derivative wrt sigma_L: quadratic (smooth) vs linear (piecewise-constant) ---
ax = fig.add_subplot(gs[1, 1])
for tck, dq, dl, color, lab in zip(FIXED_TICKS, deriv_quad_curves, deriv_lin_curves, FIXED_COLORS, FIXED_LABELS):
    ax.plot(np.asarray(sigma_grid), dq, color=color, linewidth=1.3, linestyle='-')
    ax.plot(np.asarray(sigma_grid), dl, color=color, linewidth=0.9, linestyle=':', alpha=0.9)
ax.plot([], [], color='0.3', linestyle='-', linewidth=1.3, label='quadratic')
ax.plot([], [], color='0.3', linestyle=':', linewidth=0.9, label='linear')
for sl in sigma_L_ticks_vals:
    ax.axvline(sl, color='0.75', linewidth=0.6, linestyle=':', zorder=1)
ax.axhline(0, color='0.6', linewidth=0.5, zorder=1)
ax.set_xlabel(r'$\sigma_L$ (ticks)')
ax.set_ylabel(r'$\partial({\rm template})/\partial\sigma_L$ (arb. units/tick)')
ax.set_xlim(0, float(sigma_grid[-1]))
ax.set_ylim(bottom=-0.225)
ax.legend(loc='lower right', handlelength=1.4, handletextpad=0.3, labelspacing=0.2, borderaxespad=0.2, ncol=2, columnspacing=0.6, **LEGEND_KW)
ax.grid(alpha=0.25, linewidth=0.5)
panel_label(ax, '(c)')

# --- (d) transverse sub-pixel projection weights, Eq. tran_diff_weights ---
ax = fig.add_subplot(gs[1, 2])
extent_mm = [float(bin_edges[0]) * 10, float(bin_edges[-1]) * 10] * 2  # cm -> mm
im = ax.imshow(weights, origin='lower', cmap='viridis', extent=extent_mm, aspect='equal')
for i in range(nb_T):
    for j in range(nb_T):
        val = weights[i, j]
        txt_color = 'white' if val < 0.5 * weights.max() else 'black'
        x_c = 10 * (0.5 * (bin_edges[j] + bin_edges[j + 1]))
        y_c = 10 * (0.5 * (bin_edges[i] + bin_edges[i + 1]))
        ax.text(x_c, y_c, f'{val:.2f}', ha='center', va='center', fontsize=5.0, color=txt_color)
ax.plot(10 * X0_OFFSET, 10 * Y0_OFFSET, marker='+', color=C_VERM, markersize=6, markeredgewidth=1.0)
ax.set_xlabel('x (mm)')
ax.set_ylabel('y (mm)')
cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.06)
cbar.ax.set_ylabel(r'weight $w_{mn}$', fontsize=7)
cbar.ax.tick_params(labelsize=6)
panel_label(ax, '(d)')

os.makedirs(OUT_DIR, exist_ok=True)
fig.savefig(f'{OUT_DIR}/diffusion_templates.pdf', bbox_inches='tight', pad_inches=0.04)
fig.savefig(f'{OUT_DIR}/diffusion_templates.png', dpi=300, bbox_inches='tight', pad_inches=0.04)
print(f'\nsaved {OUT_DIR}/diffusion_templates.pdf and .png')
