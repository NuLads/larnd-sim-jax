"""Paper figure for subsec:projection: why Monte-Carlo sampling of drifting electrons (the
standard larnd-sim path, retained here as `mc_diff` for cross-checks only -- main.tex says
"A Monte Carlo sampling path is retained in the code for cross-checks against the reference
simulation, but it is not used for calibration") is unaffordable/noisy for a calibration fit,
compared with the deterministic projected-charge method (Eq. tran_diff_weights) this paper uses.

Two things are done:
  1. Analytic packet-vs-electron statistics (sigma_T, sigma_L, electrons/packet, the
     sqrt(electrons-per-packet) excess-noise factor) for a real through-going-muon segment at
     mid and at full drift, read from the production truth parameters and a real loaded track
     (paper_common.params(), paper_common.load()) -- nothing hand-copied.
  2. A real demonstration using the library's own `mc_diff` switch: `sim_jax.simulate_drift`
     is run with `params.mc_diff=True` (Monte-Carlo packet, `detsim_jax.generate_electrons`)
     and `params.mc_diff=False` (deterministic projection, `detsim_jax.apply_tran_diff`) on the
     SAME real chopped track pieces, through the SAME current model (`detsim_jax.current_mc`),
     for ~40 random seeds, at the production electron_sampling_resolution=0.01 cm. This isolates
     exactly the "packet sampling" step the paper's claim is about, without also changing the
     current model or the pixel-neighbour handling.

IMPORTANT CODE FINDINGS (see the printed report at the end -- also in the hand-off message):
  - `sim_jax.simulate_wfs` (the function that generates the production/target waveforms) does
    NOT take an rngkey and does NOT consult `params.mc_diff` at all: it always calls
    `simulate_drift_new`, a separate, always-deterministic code path. The `mc_diff` MC/projection
    switch only exists inside `sim_jax.simulate_drift` + `simulate_signals_parametrized`
    (bundled as `sim_jax.simulate_parametrized`), which is the "cross-check" path quoted above.
    This script therefore uses `simulate_drift` + the `current_mc` current model for BOTH sides
    of the comparison, which is the only code path where the two can be toggled apples-to-apples.
  - `detsim_jax.apply_tran_diff` (the deterministic branch of that cross-check path) requires
    `params.tran_diff_bin_edges`, which is a dataclass field defaulting to None and is NOT set
    anywhere else in the repository (grepped). It must be supplied by hand; this script supplies
    5 pixel-wide bins (bin_width = pixel_pitch), matching n_T=5 of Eq. tran_diff_weights.
  - `detsim_jax.apply_tran_diff`'s Gaussian-integral weights are computed with
    `gaussian_1d_integral(bin_edges, sigma)`, which has NO x0 (sub-bin offset) argument --
    unlike the production sub-pixel path (`gaussian_1d_integral_new`/`density_2d`, used by
    `simulate_drift_new`) and unlike Eq. tran_diff_weights itself, which is explicitly centred on
    "a cloud centred at offset (x0, y0)". `apply_tran_diff` therefore implicitly assumes x0=0
    (cloud exactly centred on its home bin) for every piece. Since sigma_T is always sub-pixel in
    this detector (confirmed below -- a few 1e-2 cm against a 0.4434 cm pitch) while the real sub-
    pixel offset x0 is essentially uniform over +/-pixel_pitch/2, this is wrong whenever a piece
    isn't within ~sigma_T of its pixel centre -- i.e. almost always -- and in practice makes
    `apply_tran_diff` dump ~100% of the charge into the home pixel regardless of sigma_T. Verified
    directly: on a real piece, the shipped `apply_tran_diff` disagrees with the Monte-Carlo mean
    (n=200 seeds) by 33 sigma (SE of the mean) on a pixel's total charge; a corrected version
    using `gaussian_1d_integral_new` with the piece's real (x0, y0) agrees with the MC mean to
    0.94 sigma. This script therefore defines and uses `apply_tran_diff_x0` (a local, corrected
    reimplementation calling the existing public `gaussian_1d_integral_new`, not an edit to any
    tracked file) for the deterministic side of the demonstration, and reports the bug in
    `apply_tran_diff` as a finding rather than silently working around it.

Regenerate with:
    JAX_PLATFORMS=cpu python fig_mc_sampling.py
Writes:
    /sdf/group/neutrino/pgranger/NuLads_draft/figures/mc_sampling.pdf
    /sdf/group/neutrino/pgranger/NuLads_draft/figures/mc_sampling.png
and prints the Section-1 numbers table and the Section-2 measured spreads to stdout.

CPU only; does not submit SLURM jobs; does not edit any tracked file.
"""
import os
os.environ.setdefault('JAX_PLATFORMS', 'cpu')  # must precede `import jax` anywhere (incl. via paper_common)
import sys
import time

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp

sys.path.insert(0, '/sdf/scratch/neutrino/pgranger/review_2026_09/PAPERJOBS')
import paper_common as pc  # noqa: E402  (does os.chdir to the repo on import)

from larndsim.consts_jax import get_vdrift  # noqa: E402
from larndsim.sim_jax import simulate_drift, shift_tracks  # noqa: E402
from larndsim.quenching_jax import quench  # noqa: E402
from larndsim.drifting_jax import drift as drift_fn  # noqa: E402
from larndsim.detsim_jax import (get_pixels, id2pixel, get_pixel_coordinates, current_mc,  # noqa: E402
                                  accumulate_signals_parametrized, gaussian_1d_integral_new,
                                  apply_tran_diff as apply_tran_diff_orig)
from larndsim.fee_jax import get_adc_values  # noqa: E402
from optimize.dataio import chop_tracks  # noqa: E402

OUT_DIR = '/sdf/group/neutrino/pgranger/NuLads_draft/figures'

# ----------------------------------------------------------------------
# Style (shared paper style guide, matching review_2026_09/scripts/paperfigs/fig_performance.py etc.)
# ----------------------------------------------------------------------
C_PROJ, C_MC, C_GREEN, C_ORANGE, C_SKY, C_PURP, C_BLACK = (
    '#0072B2', '#D55E00', '#009E73', '#E69F00', '#56B4E9', '#CC79A7', '#000000')

plt.rcParams.update({
    'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8,
    'legend.fontsize': 7, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
    'axes.linewidth': 0.6, 'lines.linewidth': 1.2,
    'font.family': 'sans-serif',
})

N_SEEDS = 40          # MC packet-placement seeds (task asks for 20-50)
N_NOISE_SEEDS = 200   # pure-electronics-noise throws (on the fixed deterministic waveform)
RESOLUTIONS = (0.01, 0.001)  # cm; production fits vs. the reference larnd-sim step
# (DUNE/larnd-sim detsim.py: MIN_STEP_SIZE = 0.001 cm, MC_SAMPLE_MULTIPLIER = 1, i.e. one Gaussian draw per 10 um step;
#  JINST 18 (2023) P04034 Sec. 3.3.2: '10 um sampling interval ... roughly 50 electrons' per sample point)
RES_FINE = 0.001


# ----------------------------------------------------------------------
# Data and truth parameters
# ----------------------------------------------------------------------
print('Loading real through-going-muon tracks and truth parameters ...')
ds_sim, ds_tgt, fields = pc.load(300., 1, data_seed=1)
tracks_full = np.asarray(ds_sim[0])
parent_ids = ds_sim.get_parent_segment_ids(0)
truth, response = pc.params()

vdrift = float(get_vdrift(truth))
pixel_pitch = float(truth.pixel_pitch)
t_sampling = float(truth.t_sampling)
drift_length = float(truth.drift_length)
t_drift_max = drift_length / vdrift
D_T, D_L = float(truth.tran_diff), float(truth.long_diff)
lifetime = float(truth.lifetime)

zidx = fields.index('z')
valid = parent_ids >= 0
uniq_pid = np.unique(parent_ids[valid])
zmean = {p: float(tracks_full[parent_ids == p, zidx].mean()) for p in uniq_pid}

# Pick one segment near mid-drift and one near full drift from the real, loaded sample.
# NOTE: this detector's two anode planes sit at z = +/-drift_length (cathode near z=0), so
# drift distance = drift_length - |z| (approximately): |z| ~ drift_length/2 is mid-drift, and
# |z| ~ 0 (near the cathode) -- NOT |z| ~ drift_length -- is full drift. Verified directly
# against drift()'s own drift_distance for this geometry before writing this selection.
# The two anode planes' valid z-ranges leave a ~0.3 cm dead gap straddling the cathode at z=0
# (tpc_borders cathode-side edge is at |z|=0.159 cm); a segment whose pieces cross that gap gets
# some of them masked to n_electrons=0 by drift()'s TPC-bounds check (verified directly: the
# naive abs(z)-closest-to-0 pick had 70/100 pieces masked out). CATHODE_MARGIN keeps the chosen
# "full drift" segment entirely clear of that gap so none of its pieces are spuriously masked.
CATHODE_MARGIN = 0.3  # cm
pid_mid = min(uniq_pid, key=lambda p: abs(abs(zmean[p]) - 0.5 * drift_length))


def _clears_cathode_gap(p):
    zv = tracks_full[parent_ids == p, zidx]
    return zv.min() * zv.max() > 0 and np.abs(zv).min() > CATHODE_MARGIN


full_candidates = [p for p in uniq_pid if _clears_cathode_gap(p)]
pid_full = min(full_candidates, key=lambda p: abs(zmean[p]))
CASES = {'mid': tracks_full[parent_ids == pid_mid], 'full': tracks_full[parent_ids == pid_full]}

print(f'  vdrift={vdrift:.5f} cm/us, drift_length={drift_length:.3f} cm, '
      f'max drift time={t_drift_max:.1f} us, lifetime={lifetime:.1f} us')
print(f'  pixel_pitch={pixel_pitch} cm, t_sampling={t_sampling} us, '
      f'DISCRIMINATION_THRESHOLD={float(truth.DISCRIMINATION_THRESHOLD):.0f} e-, '
      f'RESET_NOISE_CHARGE={float(truth.RESET_NOISE_CHARGE):.0f} e-')
for tag, seg in CASES.items():
    z = seg[:, zidx].mean()
    print(f'  case "{tag}": segment z={z:+.2f} cm (exact t_drift in Section-1 table below), '
          f'{seg.shape[0]} chopped pieces, dx_tot={seg[:, fields.index("dx")].sum():.3f} cm')


# ----------------------------------------------------------------------
# A corrected, x0-aware deterministic projection (see module docstring: the shipped
# apply_tran_diff ignores the sub-pixel offset x0 of each piece). This reuses the existing
# public `gaussian_1d_integral_new` (the same function the production sub-pixel path,
# detsim_jax.density_2d, uses) -- no tracked file is edited.
# ----------------------------------------------------------------------
def apply_tran_diff_x0(params, electrons, fields):
    borders = params.tpc_borders[electrons[:, fields.index('pixel_plane')].astype(int)]
    bin_width = params.tran_diff_bin_edges[1] - params.tran_diff_bin_edges[0]
    # offset of the piece from the CENTRE of its home pixel, matching the symmetric,
    # zero-centred bin_edges grid constructed below (±2.5, ±1.5, ±0.5 pixel widths).
    x0 = (electrons[:, fields.index('x')] - borders[:, 0, 0]) % bin_width - bin_width / 2
    y0 = (electrons[:, fields.index('y')] - borders[:, 1, 0]) % bin_width - bin_width / 2
    std = electrons[:, fields.index('tran_diff')]
    gx = gaussian_1d_integral_new(params.tran_diff_bin_edges[None, :], x0[:, None], std[:, None])
    gy = gaussian_1d_integral_new(params.tran_diff_bin_edges[None, :], y0[:, None], std[:, None])
    gaussian_integral_2d = gx[:, :, None] * gy[:, None, :]

    nbins_sym = params.tran_diff_bin_edges.shape[0] / 2 - 1
    X, Y = jnp.mgrid[-nbins_sym:nbins_sym + 1, -nbins_sym:nbins_sym + 1]
    shifts = jnp.vstack([X.ravel(), Y.ravel()]).T
    new_electrons = jnp.repeat(electrons[:, None, :], shifts.shape[0], axis=1)
    new_electrons = new_electrons.at[:, :, fields.index('x')].add(shifts[:, 0] * bin_width)
    new_electrons = new_electrons.at[:, :, fields.index('y')].add(shifts[:, 1] * bin_width)
    new_electrons = new_electrons.at[:, :, fields.index('n_electrons')].multiply(
        gaussian_integral_2d.reshape(-1, shifts.shape[0]))
    return new_electrons.reshape(-1, electrons.shape[1])


TRAN_DIFF_BIN_EDGES = pixel_pitch * jnp.arange(-2.5, 3.0, 1.0)  # 6 edges -> 5 pixel-wide bins (n_T=5)


def make_params(mc_diff, diffusion_in_current_sim, resolution, noise=True):
    p = truth.replace(mc_diff=mc_diff, diffusion_in_current_sim=diffusion_in_current_sim,
                       number_pix_neighbors=0, electron_sampling_resolution=resolution,
                       tran_diff_bin_edges=TRAN_DIFF_BIN_EDGES)
    if not noise:
        p = p.replace(RESET_NOISE_CHARGE=0., UNCORRELATED_NOISE_CHARGE=0.)
    return p


def drifted_pieces(params, tracks_jax):
    """shift -> quench -> drift, i.e. simulate_drift up to (but excluding) the diffusion step."""
    st = shift_tracks(params, tracks_jax, fields)
    st = quench(params, st, fields)
    st = drift_fn(params, st, fields)
    return st


def electrons_for(params, tracks_jax, rngkey):
    """The diffusion-generation step of simulate_drift, with the deterministic branch replaced
    by the x0-corrected projection (see module docstring)."""
    st = drifted_pieces(params, tracks_jax)
    if params.mc_diff:
        from larndsim.detsim_jax import generate_electrons
        return generate_electrons(st, fields, rngkey, not params.diffusion_in_current_sim)
    return apply_tran_diff_x0(params, st, fields)


def build_wfs(params, electrons, unique_pixels, nticks_wf):
    """Reimplements the `wfs`-building half of sim_jax.simulate_signals_parametrized (current_mc
    + accumulate_signals_parametrized), stopping short of the noise/threshold step, so the clean
    pre-noise current waveform is available for the figure. Calls only existing public functions."""
    pIDs = get_pixels(params, electrons, fields).ravel()
    xpitch, ypitch, plane, eid = id2pixel(params, pIDs)
    pixels_coord = get_pixel_coordinates(params, xpitch, ypitch, plane)
    t0, signals = current_mc(params, electrons, pixels_coord, fields)
    pix_renumbering = jnp.searchsorted(unique_pixels, pIDs)
    in_range = jnp.clip(pix_renumbering, 0, unique_pixels.shape[0] - 1)
    valid_mask = (pix_renumbering < unique_pixels.shape[0]) & (unique_pixels[in_range] == pIDs)
    pix_renumbering = jnp.where(valid_mask, pix_renumbering, 0)
    start_ticks = t0 - signals.shape[1]
    wfs = jnp.zeros((unique_pixels.shape[0], nticks_wf))
    signals_masked = jnp.where(valid_mask[:, None], signals, 0.)
    return accumulate_signals_parametrized(wfs, signals_masked, pix_renumbering, start_ticks)


NTICKS_WF = int(truth.time_interval[1] / t_sampling) + 1

results = {}  # per-case collected numbers, used both for the printed report and the figure

for tag, seg in CASES.items():
    tracks_jax = jnp.array(seg)
    z = float(seg[:, zidx].mean())

    params_proj = make_params(mc_diff=False, diffusion_in_current_sim=True, resolution=0.01)
    params_mc = make_params(mc_diff=True, diffusion_in_current_sim=False, resolution=0.01)

    # --- physical per-piece quantities (real, post quench+drift) ---
    st = drifted_pieces(params_proj, tracks_jax)
    n_masked = int(np.sum(np.asarray(st[:, fields.index('n_electrons')]) < 1.0))
    if n_masked > 0:
        print(f'  WARNING case "{tag}": {n_masked}/{st.shape[0]} pieces masked to n_electrons=0 by '
              f'drift()\'s TPC-bounds check (likely a cathode-gap or other boundary artifact) -- '
              f'electrons_per_cm below will be biased low; pick a different segment.')
    sigma_T = float(jnp.mean(st[:, fields.index('tran_diff')]))
    sigma_L = float(jnp.mean(st[:, fields.index('long_diff')]))
    n_electrons_tot = float(jnp.sum(st[:, fields.index('n_electrons')]))
    dx_tot = float(jnp.sum(st[:, fields.index('dx')]))
    electrons_per_cm_attenuated = n_electrons_tot / dx_tot
    # Real drift time from the post-drift geometry (NOT abs(z)/vdrift: this detector's two anode
    # planes sit at z=+/-drift_length with the cathode near z=0, so drift_distance = |z - z_anode|,
    # z_anode being +drift_length or -drift_length depending which plane the piece was assigned
    # to by drift() -- picking the plane up front avoids re-deriving that assignment by hand).
    plane_mean = int(round(float(jnp.mean(st[:, fields.index('pixel_plane')]))))
    z_anode = float(truth.tpc_borders[plane_mean, 2, 0])
    t_drift = abs(z - z_anode) / vdrift
    electrons_per_cm_raw = electrons_per_cm_attenuated / float(jnp.exp(-t_drift / lifetime))

    # --- deterministic projection (corrected, x0-aware) and the shipped (buggy) apply_tran_diff ---
    electrons_proj = apply_tran_diff_x0(params_proj, st, fields)
    electrons_buggy = apply_tran_diff_orig(params_proj, st, fields)

    # union of pixels touched by the projection and by a handful of MC seeds, so every seed's
    # waveform can be written into the SAME pixel array for an apples-to-apples comparison.
    pix_lists = [jnp.unique(get_pixels(params_proj, electrons_proj, fields).ravel())]
    for s in range(8):
        e_mc, _ = simulate_drift(params_mc, tracks_jax, fields, jax.random.key(900 + s))
        pix_lists.append(jnp.unique(get_pixels(params_mc, e_mc, fields).ravel()))
    unique_pixels = jnp.unique(jnp.concatenate(pix_lists))

    wfs_proj = build_wfs(params_proj, electrons_proj, unique_pixels, NTICKS_WF)
    wfs_buggy = build_wfs(params_proj, electrons_buggy, unique_pixels, NTICKS_WF)
    tot_proj = np.asarray(wfs_proj).sum(axis=1) * t_sampling
    tot_buggy = np.asarray(wfs_buggy).sum(axis=1) * t_sampling
    dominant = int(np.argmax(tot_proj))
    pix_id = int(unique_pixels[dominant])

    q_pixel_total = float(tot_proj[dominant])                                  # "electrons per pixel"
    q_pixel_peak_tick = float(np.asarray(wfs_proj)[dominant].max()) * t_sampling  # "electrons per (pixel,tick) cell"

    # --- MC seeds: waveform curves + per-seed total charge and threshold-crossing tick ---
    params_mc_noiseless = params_mc.replace(RESET_NOISE_CHARGE=0., UNCORRELATED_NOISE_CHARGE=0.)
    params_proj_noiseless = params_proj.replace(RESET_NOISE_CHARGE=0., UNCORRELATED_NOISE_CHARGE=0.)

    wfs_mc_list = []
    totals_mc, crossing_mc_noiseless = [], []
    for s in range(N_SEEDS):
        e_mc, _ = simulate_drift(params_mc, tracks_jax, fields, jax.random.key(1000 + s))
        wfs_mc = build_wfs(params_mc, e_mc, unique_pixels, NTICKS_WF)
        wfs_mc_list.append(np.asarray(wfs_mc)[dominant])
        totals_mc.append(float(np.asarray(wfs_mc).sum(axis=1)[dominant] * t_sampling))
        integral_mc, ticks_mc = get_adc_values(params_mc_noiseless, wfs_mc[:, 1:], jax.random.key(0))
        crossing_mc_noiseless.append(float(np.asarray(ticks_mc)[dominant, 0]))
    totals_mc = np.array(totals_mc)
    crossing_mc_noiseless = np.array(crossing_mc_noiseless)

    integral_proj, ticks_proj = get_adc_values(params_proj_noiseless, wfs_proj[:, 1:], jax.random.key(0))
    crossing_proj = float(np.asarray(ticks_proj)[dominant, 0])

    # --- the same MC at the reference larnd-sim step (10 um): re-chop the SAME pieces 10x finer ---
    seg_fine, _ = chop_tracks(np.asarray(seg).copy(), fields, precision=RES_FINE)
    tracks_fine = jnp.array(seg_fine)
    params_mc_fine = make_params(mc_diff=True, diffusion_in_current_sim=False, resolution=RES_FINE)
    params_mc_fine_noiseless = params_mc_fine.replace(RESET_NOISE_CHARGE=0., UNCORRELATED_NOISE_CHARGE=0.)
    totals_mc_fine, crossing_mc_fine = [], []
    for s in range(N_SEEDS):
        e_mc, _ = simulate_drift(params_mc_fine, tracks_fine, fields, jax.random.key(3000 + s))
        wfs_mc = build_wfs(params_mc_fine, e_mc, unique_pixels, NTICKS_WF)
        totals_mc_fine.append(float(np.asarray(wfs_mc).sum(axis=1)[dominant] * t_sampling))
        integral_mc, ticks_mc = get_adc_values(params_mc_fine_noiseless, wfs_mc[:, 1:], jax.random.key(0))
        crossing_mc_fine.append(float(np.asarray(ticks_mc)[dominant, 0]))
    totals_mc_fine = np.array(totals_mc_fine)
    crossing_mc_fine = np.array(crossing_mc_fine)
    print(f'  case "{tag}": {seg_fine.shape[0]} pieces at {RES_FINE} cm (vs {seg.shape[0]} at 0.01 cm); '
          f'MC mean total {totals_mc_fine.mean():.1f} e- vs projection {float(tot_proj[dominant]):.1f} e-', flush=True)

    # --- analytic projection at the reference step: how much does OUR output change from 100 um to 10 um? (9 Oct) ---
    params_proj_fine = make_params(mc_diff=False, diffusion_in_current_sim=True, resolution=RES_FINE)
    electrons_proj_fine = apply_tran_diff_x0(params_proj_fine, drifted_pieces(params_proj_fine, tracks_fine), fields)
    wfs_proj_fine = build_wfs(params_proj_fine, electrons_proj_fine, unique_pixels, NTICKS_WF)
    tot_proj_fine = float(np.asarray(wfs_proj_fine).sum(axis=1)[dominant] * t_sampling)
    _, ticks_pf = get_adc_values(params_proj_noiseless, wfs_proj_fine[:, 1:], jax.random.key(0))
    crossing_proj_fine = float(np.asarray(ticks_pf)[dominant, 0])
    dcum_proj = float(np.abs(np.cumsum(np.asarray(wfs_proj_fine)[dominant]) - np.cumsum(np.asarray(wfs_proj)[dominant])).max() * t_sampling)
    dq_proj = abs(tot_proj_fine - float(tot_proj[dominant])); dt_proj = abs(crossing_proj_fine - crossing_proj)
    print(f'  case "{tag}": analytic projection 0.01 -> {RES_FINE} cm: |dQ| {dq_proj:.1f} e-, max |d cumulative| {dcum_proj:.1f} e-, '
          f'|d crossing| {dt_proj:.0f} ticks', flush=True)

    # --- pure electronics (readout) noise alone: same deterministic waveform, noise-only seeds ---
    crossing_noise_only = []
    totals_noise_only = []
    for s in range(N_NOISE_SEEDS):
        integral_n, ticks_n = get_adc_values(params_proj, wfs_proj[:, 1:], jax.random.key(2000 + s))
        crossing_noise_only.append(float(np.asarray(ticks_n)[dominant, 0]))
        totals_noise_only.append(float(np.asarray(integral_n)[dominant, 0]))
    crossing_noise_only = np.array(crossing_noise_only)
    totals_noise_only = np.array(totals_noise_only)

    results[tag] = dict(
        z=z, t_drift=t_drift, sigma_T=sigma_T, sigma_L=sigma_L,
        electrons_per_cm_raw=electrons_per_cm_raw, electrons_per_cm_attenuated=electrons_per_cm_attenuated,
        q_pixel_total=q_pixel_total, q_pixel_peak_tick=q_pixel_peak_tick,
        pix_id=pix_id, dominant=dominant, unique_pixels=unique_pixels,
        wfs_proj_dominant=np.asarray(wfs_proj)[dominant], wfs_mc_list=np.array(wfs_mc_list),
        tot_proj=float(tot_proj[dominant]), tot_buggy=float(tot_buggy[dominant]),
        totals_mc=totals_mc, crossing_mc_noiseless=crossing_mc_noiseless, crossing_proj=crossing_proj,
        crossing_noise_only=crossing_noise_only, totals_noise_only=totals_noise_only,
        totals_mc_fine=totals_mc_fine, crossing_mc_fine=crossing_mc_fine,
        dq_proj=dq_proj, dt_proj=dt_proj, dcum_proj=dcum_proj,
    )

    print(f'\n--- case "{tag}" (pixel id {pix_id}) ---')
    se_mc = totals_mc.std(ddof=1) / np.sqrt(N_SEEDS)
    print(f'  corrected (x0-aware) projection, dominant-pixel total charge: {tot_proj[dominant]:.1f} e-')
    print(f'  shipped apply_tran_diff (buggy, x0 ignored):                 {tot_buggy[dominant]:.1f} e-'
          f'  ({(tot_buggy[dominant]-totals_mc.mean())/se_mc:+.1f} sigma vs MC mean)')
    print(f'  MC (n={N_SEEDS}) mean={totals_mc.mean():.1f} e-  std={totals_mc.std(ddof=1):.1f} e-  '
          f'SE={se_mc:.1f} e-  ({(totals_mc.mean()-tot_proj[dominant])/se_mc:+.2f} sigma vs corrected projection)')


# ----------------------------------------------------------------------
# Section 1: analytic packet-vs-electron numbers (tied to the measurements above)
# ----------------------------------------------------------------------
print('\n' + '=' * 78)
print('SECTION 1: analytic packet-vs-electron statistics')
print('=' * 78)
table_rows = []
for tag in ('mid', 'full'):
    r = results[tag]
    row = dict(case=tag, z=r['z'], t_drift=r['t_drift'], sigma_T=r['sigma_T'], sigma_L=r['sigma_L'],
               electrons_per_cm=r['electrons_per_cm_attenuated'], electrons_per_cm_raw=r['electrons_per_cm_raw'],
               q_pixel_total=r['q_pixel_total'], q_pixel_peak_tick=r['q_pixel_peak_tick'])
    for res in RESOLUTIONS:
        electrons_per_packet = r['electrons_per_cm_attenuated'] * res
        excess_noise_factor = np.sqrt(electrons_per_packet)
        packets_per_pixel = r['q_pixel_total'] / electrons_per_packet
        packets_per_cell = r['q_pixel_peak_tick'] / electrons_per_packet
        row[f'electrons_per_packet_{res}'] = electrons_per_packet
        row[f'excess_noise_factor_{res}'] = excess_noise_factor
        row[f'packets_per_pixel_{res}'] = packets_per_pixel
        row[f'packets_per_cell_{res}'] = packets_per_cell
    table_rows.append(row)

hdr = (f"{'case':5s} {'z[cm]':>7s} {'t_d[us]':>8s} {'sig_T[cm]':>10s} {'sig_L[cm]':>10s} "
       f"{'e-/cm(raw)':>10s} {'e-/cm(att)':>10s} {'e-/pix':>9s} {'e-/cell(pk)':>12s}")
print(hdr)
for row in table_rows:
    print(f"{row['case']:5s} {row['z']:7.2f} {row['t_drift']:8.1f} {row['sigma_T']:10.5f} "
          f"{row['sigma_L']:10.5f} {row['electrons_per_cm_raw']:10.0f} {row['electrons_per_cm']:10.0f} "
          f"{row['q_pixel_total']:9.0f} {row['q_pixel_peak_tick']:12.1f}")
print('  (e-/cm(raw) = before electron-lifetime attenuation over the drift to the anode; '
      'e-/cm(att) = after -- the value actually used below)')
print()
hdr2 = (f"{'case':5s} {'res[cm]':>8s} {'e-/packet':>10s} {'excess':>8s} "
        f"{'packets/pix':>12s} {'packets/cell(pk)':>17s}")
print(hdr2)
for row in table_rows:
    for res in RESOLUTIONS:
        print(f"{row['case']:5s} {res:8.3f} {row[f'electrons_per_packet_{res}']:10.2f} "
              f"{row[f'excess_noise_factor_{res}']:8.2f} {row[f'packets_per_pixel_{res}']:12.4f} "
              f"{row[f'packets_per_cell_{res}']:17.6f}")

# packets needed for electron-level fidelity, relative to the production resolution (0.01 cm)
print()
for row in table_rows:
    ratio = row['electrons_per_packet_0.01']
    print(f'  case "{row["case"]}": electron-level fidelity (1 electron/packet) needs '
          f'{ratio:.0f}x more packets (hence ~{ratio:.0f}x the memory/time) than the production '
          f'0.01 cm resolution already uses; {row["electrons_per_packet_0.001"]:.0f}x more than the '
          f'library-default 0.001 cm resolution.')


# ----------------------------------------------------------------------
# Timing scaling check: wall-clock of the MC packet-generation step vs. number of packets
# (array row count), on CPU. Demonstrates the claimed linear scaling directly.
# ----------------------------------------------------------------------
print('\n' + '=' * 78)
print('Timing-scaling check (CPU, mc_diff=True branch, simulate_drift)')
print('=' * 78)
base_seg = CASES['mid']
params_mc_time = make_params(mc_diff=True, diffusion_in_current_sim=False, resolution=0.01)
n_repeats_list = [1, 4, 16, 64]
timing_rows = []
for nrep in n_repeats_list:
    tracks_rep = jnp.array(np.tile(base_seg, (nrep, 1)))
    rk = jax.random.key(0)
    simulate_drift(params_mc_time, tracks_rep, fields, rk)  # warm-up / compile, excluded from timing
    t0 = time.perf_counter()
    for _ in range(5):
        e, p = simulate_drift(params_mc_time, tracks_rep, fields, rk)
        jax.block_until_ready(e)
    dt = (time.perf_counter() - t0) / 5
    n_packets = tracks_rep.shape[0]
    timing_rows.append((n_packets, dt))
    print(f'  n_packets={n_packets:7d}  wall-clock={dt*1e3:8.3f} ms  ({dt/n_packets*1e6:.4f} us/packet)')
n_arr = np.array([t[0] for t in timing_rows], dtype=float)
t_arr = np.array([t[1] for t in timing_rows], dtype=float)
slope, intercept = np.polyfit(n_arr, t_arr, 1)
pred = slope * n_arr + intercept
r2 = 1 - np.sum((t_arr - pred) ** 2) / np.sum((t_arr - t_arr.mean()) ** 2)
print(f'  linear fit: time = {slope*1e6:.4f} us/packet * n_packets + {intercept*1e3:.3f} ms   (R2={r2:.4f})')


# ----------------------------------------------------------------------
# Section 2 summary numbers (already measured above per-case)
# ----------------------------------------------------------------------
print('\n' + '=' * 78)
print('SECTION 2: measured seed-to-seed spreads (dominant pixel of each case)')
print('=' * 78)
reset_noise = float(truth.RESET_NOISE_CHARGE)
for tag in ('mid', 'full'):
    r = results[tag]
    std_q_mc = r['totals_mc'].std(ddof=1)
    std_t_mc = r['crossing_mc_noiseless'].std(ddof=1) * t_sampling  # us
    std_q_noise = r['totals_noise_only'].std(ddof=1)
    std_t_noise = r['crossing_noise_only'].std(ddof=1) * t_sampling
    print(f'  case "{tag}" (pixel {r["pix_id"]}, projection total charge {r["tot_proj"]:.0f} e-):')
    print(f'    MC packet-sampling (noiseless electronics, n={N_SEEDS}):'
          f'  charge std={std_q_mc:.1f} e- ({100*std_q_mc/r["tot_proj"]:.2f}% of total),'
          f'  crossing-time std={std_t_mc:.4f} us ({std_t_mc/t_sampling:.2f} ticks)')
    print(f'    electronics readout noise alone (n={N_NOISE_SEEDS}):'
          f'  charge std={std_q_noise:.1f} e-,  crossing-time std={std_t_noise:.4f} us '
          f'({std_t_noise/t_sampling:.2f} ticks)')
    print(f'    reference scales: RESET_NOISE_CHARGE={reset_noise:.0f} e-,  1 tick={t_sampling} us')
    std_q_fine = r['totals_mc_fine'].std(ddof=1)
    std_t_fine = r['crossing_mc_fine'].std(ddof=1) * t_sampling
    print(f'    MC packet-sampling at the reference {RES_FINE} cm step (noiseless electronics, n={N_SEEDS}):'
          f'  charge std={std_q_fine:.1f} e- ({100*std_q_fine/r["tot_proj"]:.2f}% of total, {std_q_fine/reset_noise:.2f} x reset noise),'
          f'  crossing-time std={std_t_fine:.4f} us ({std_t_fine/t_sampling:.2f} ticks)')
    print(f'    ratio 0.01 cm / {RES_FINE} cm spread: charge {std_q_mc/std_q_fine:.2f}, time {std_t_mc/max(std_t_fine,1e-12):.2f} '
          f'(sqrt(10) = 3.16 expected for charge)')
    results[tag]['std_q_fine'] = std_q_fine
    results[tag]['std_t_fine_us'] = std_t_fine
    results[tag]['std_q_mc'] = std_q_mc
    results[tag]['std_t_mc_us'] = std_t_mc
    results[tag]['std_q_noise'] = std_q_noise
    results[tag]['std_t_noise_us'] = std_t_noise


# ======================================================================
# Figure: 2 panels
# ======================================================================
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.75))

# --- (a) cumulative charge vs tick: projection vs MC seeds, dominant pixel, "full" case ---
ax = axes[0]
PLOT_CASE = 'full'
r = results[PLOT_CASE]
cum_proj = np.cumsum(r['wfs_proj_dominant']) * t_sampling
# window centred on the rise, wide enough to show pre- and post-threshold behaviour
rise_tick = int(np.searchsorted(cum_proj, 0.5 * cum_proj.max()))
lo, hi = max(0, rise_tick - 30), rise_tick + 30
ticks_axis = (np.arange(lo, hi) - rise_tick) * t_sampling  # us, centred on the rise

for s in range(min(25, N_SEEDS)):
    cum_mc = np.cumsum(r['wfs_mc_list'][s]) * t_sampling
    ax.plot(ticks_axis, cum_mc[lo:hi], color=C_MC, alpha=0.35, linewidth=0.7, zorder=2)
ax.plot([], [], color=C_MC, alpha=0.8, linewidth=0.7, label=f'MC packets ({N_SEEDS} seeds)')
ax.plot(ticks_axis, cum_proj[lo:hi], color=C_PROJ, linewidth=1.6, zorder=4, label='projection')
ax.axhline(float(truth.DISCRIMINATION_THRESHOLD), color=C_BLACK, linestyle=':', linewidth=0.8, zorder=1)
ax.text(ticks_axis[0], float(truth.DISCRIMINATION_THRESHOLD), 'threshold ', va='bottom', ha='left',
        fontsize=6.5, color=C_BLACK)
ax.set_xlabel(r'time relative to threshold crossing ($\mu$s)')
ax.set_ylabel('cumulative pixel charge (e$^-$)')
ax.set_xlim(ticks_axis[0], ticks_axis[-1])
ax.legend(frameon=False, loc='lower right', handletextpad=0.5, labelspacing=0.3, borderaxespad=0.3)
ax.text(0.04, 0.97, '(a)', transform=ax.transAxes, fontsize=7, va='top', ha='left')
ax.grid(alpha=0.25, linewidth=0.5)

# --- (b) seed-to-seed spread vs. physical reference scales, both cases ---
ax = axes[1]
labels = ['charge\n(mid drift)', 'charge\n(full drift)', 'time\n(mid drift)', 'time\n(full drift)']
values = [results['mid']['std_q_mc'] / reset_noise, results['full']['std_q_mc'] / reset_noise,
          results['mid']['std_t_mc_us'] / t_sampling, results['full']['std_t_mc_us'] / t_sampling]
values_fine = [results['mid']['std_q_fine'] / reset_noise, results['full']['std_q_fine'] / reset_noise,
               results['mid']['std_t_fine_us'] / t_sampling, results['full']['std_t_fine_us'] / t_sampling]
values_proj = [results['mid']['dq_proj'] / reset_noise, results['full']['dq_proj'] / reset_noise,
               results['mid']['dt_proj'], results['full']['dt_proj']]
xpos = np.arange(len(labels))
bw = 0.26
ax.bar(xpos - bw, values, color=C_MC, width=bw, zorder=3, label=r'MC packets, $100~\mu$m step')
ax.bar(xpos, values_fine, color=C_ORANGE, width=bw, zorder=3, label=r'MC packets, $10~\mu$m step (larnd-sim)')
ax.bar(xpos + bw, values_proj, color=C_PROJ, width=bw, zorder=3, label=r'analytic projection: change from $100$ to $10~\mu$m')
for xp, v in zip(xpos + bw, values_proj):
    ax.text(xp, v + 0.03, f'{v:.2f}', ha='center', va='bottom', fontsize=5.5, color=C_PROJ)
ax.legend(frameon=False, loc='upper right', handletextpad=0.5, labelspacing=0.3, borderaxespad=0.3, fontsize=6.3)
ax.axhline(1.0, color=C_BLACK, linestyle='--', linewidth=0.9, zorder=2)
ax.text(0.03, 1.0, 'reset noise / 1 tick ', va='bottom', ha='left', fontsize=6.5, color=C_BLACK,
        transform=ax.get_yaxis_transform())
ax.set_xticks(xpos)
ax.set_xticklabels(labels, fontsize=7)
ax.set_ylabel('spread or change\n(multiples of the physical reference)')
ax.set_ylim(0, max(values) * 1.6)
ax.grid(alpha=0.25, linewidth=0.5, axis='y')
ax.text(0.04, 0.97, '(b)', transform=ax.transAxes, fontsize=7, va='top', ha='left')

fig.tight_layout(w_pad=1.6)
os.makedirs(OUT_DIR, exist_ok=True)
fig.savefig(f'{OUT_DIR}/mc_sampling.pdf', bbox_inches='tight')
fig.savefig(f'{OUT_DIR}/mc_sampling.png', dpi=300, bbox_inches='tight')
print(f'\nsaved {OUT_DIR}/mc_sampling.pdf and .png')
