#!/usr/bin/env python3
"""
Linear Track Approximation Study
=================================
Comprehensive analysis of the systematic differences between the straight-line
approximation of muon tracks and the true simulated trajectories.

Checks performed:
  1. Geometric: perpendicular distance, oriented deviations (dx/dy/dz)
  2. Extent: does the line span the full true track start/end points?
  3. Direction: opening angle between true and guess directions
  4. dE/dx profile: true vs. assumed-constant along track
  5. Pixel-level: which pixels are activated by true vs. guess; overlap fraction
  6. Charge/ADC: total charge and per-pixel ADC response comparison
  7. Loss landscape: NLL as a function of the key parameters for true vs. guess track
"""

import sys, os
PROJECT = '/sdf/group/neutrino/pgranger/larnd-sim-jax'
sys.path.insert(0, PROJECT)
sys.path.insert(0, f'{PROJECT}/src')
os.chdir(PROJECT)

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import jax
import jax.numpy as jnp

from larndsim.consts_jax import build_params_class, load_detector_properties, load_lut
from optimize.strategies import LUTSimulation, LUTProbabilisticSimulation, ProbabilisticLossStrategy
from optimize.dataio import TracksDataset, np_from_structured, chop_tracks
from optimize.ranges import ranges

print('JAX devices:', jax.devices())

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
INPUT_FILE   = '/sdf/data/neutrino/cyifan/diffsim_input/true_through_muon_edep_10cm_vol1cm.h5'
LUT_FILE     = 'src/larndsim/detector_properties/response_44_v2a_full_tick.npz'
DET_PROPS    = 'src/larndsim/detector_properties/module0.yaml'
PIX_LAYOUT   = 'src/larndsim/pixel_layouts/multi_tile_layout-2.4.16_v4.yaml'
OUT_DIR      = 'scripts/linear_approx_study_results'
N_TRAJ       = 200       # trajectories used for geometric/pixel studies
N_TRAJ_SIM   = 20        # trajectories used for full simulation (slower)
SEGMENT_LEN  = 10.0      # cm, segment length for straight-line guess
CHOP_PREC    = 0.1       # cm, chopping precision

RELEVANT_PARAMS = ['Ab', 'kb', 'lifetime', 'tran_diff', 'long_diff',
                   'eField', 'shift_x', 'shift_y', 'shift_z']

os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Load dataset and detector
# ---------------------------------------------------------------------------
dataset = TracksDataset(filename=INPUT_FILE, nevents=-1,
                        electron_sampling_resolution=CHOP_PREC,
                        chopped=False, pad=False)
fields  = dataset.get_track_fields()
print(f'Loaded {len(dataset)} trajectories. Fields: {fields}')

ParamsClass  = build_params_class(RELEVANT_PARAMS)
ref_params   = load_detector_properties(ParamsClass, DET_PROPS, PIX_LAYOUT)
ref_params   = ref_params.replace(
    electron_sampling_resolution=CHOP_PREC,
    number_pix_neighbors=4,
    signal_length=150,
    time_window=150,
    excess_offset=0.0,
    smooth_sigma_scale=1.0,
)
response, ref_params = load_lut(LUT_FILE, ref_params)
params = ref_params.replace(**{p: ranges[p]['nom'] for p in RELEVANT_PARAMS})

sim_strategy = LUTSimulation(response)
sim_strategy.padded_small_nb = 256
sim_strategy.padded_large_nb = 256

# ---------------------------------------------------------------------------
# Helper: build straight-line guess
# ---------------------------------------------------------------------------
def make_straight_line_guess(true_track, fields, segment_length=SEGMENT_LEN):
    x_start = true_track[0, fields.index('x_start')]
    y_start = true_track[0, fields.index('y_start')]
    z_start = true_track[0, fields.index('z_start')]
    x_end   = true_track[-1, fields.index('x_end')]
    y_end   = true_track[-1, fields.index('y_end')]
    z_end   = true_track[-1, fields.index('z_end')]

    start_pos = np.array([x_start, y_start, z_start])
    end_pos   = np.array([x_end,   y_end,   z_end])
    direction = end_pos - start_pos
    total_len = np.linalg.norm(direction)
    if total_len < 1e-6:
        return None
    unit = direction / total_len

    n_segs    = max(1, int(np.ceil(total_len / segment_length)))
    step      = total_len / n_segs
    total_de  = true_track[:, fields.index('dE')].sum()
    de_per_seg = total_de / n_segs

    guess = np.zeros((n_segs, len(fields)), dtype=np.float32)
    for col in ['eventID', 'trackID', 'pdgID', 't0']:
        if col in fields:
            guess[:, fields.index(col)] = true_track[0, fields.index(col)]

    for i in range(n_segs):
        s = start_pos + i       * step * unit
        e = start_pos + (i + 1) * step * unit
        guess[i, fields.index('x_start')] = s[0]
        guess[i, fields.index('y_start')] = s[1]
        guess[i, fields.index('z_start')] = s[2]
        guess[i, fields.index('x_end')]   = e[0]
        guess[i, fields.index('y_end')]   = e[1]
        guess[i, fields.index('z_end')]   = e[2]
        guess[i, fields.index('x')]  = 0.5*(s[0]+e[0])
        guess[i, fields.index('y')]  = 0.5*(s[1]+e[1])
        guess[i, fields.index('z')]  = 0.5*(s[2]+e[2])
        guess[i, fields.index('dE')] = de_per_seg
        guess[i, fields.index('dx')] = step
        if 'dEdx' in fields:
            guess[i, fields.index('dEdx')] = de_per_seg / step
    return guess


def track_endpoints(track):
    """Return the two most distant points in a track's point cloud."""
    starts = np.stack([track[:, fields.index('x_start')],
                       track[:, fields.index('y_start')],
                       track[:, fields.index('z_start')]], axis=1)
    ends   = np.stack([track[:, fields.index('x_end')],
                       track[:, fields.index('y_end')],
                       track[:, fields.index('z_end')]], axis=1)
    pts = np.vstack([starts, ends])
    dists = np.linalg.norm(pts[:, None] - pts[None, :], axis=2)
    i, j = np.unravel_index(np.argmax(dists), dists.shape)
    return pts[i], pts[j], dists[i, j]


def perp_dist_pts_to_line(pts, line_start, line_end):
    """Perpendicular distance from each row of pts to the infinite line."""
    v   = line_end - line_start
    vn  = v / (np.linalg.norm(v) + 1e-12)
    d   = pts - line_start
    proj = (d @ vn)[:, None] * vn
    perp = d - proj
    return np.linalg.norm(perp, axis=1), perp  # scalar dist, oriented vector


# ===========================================================================
# PART 1 — GEOMETRIC ANALYSIS  (N_TRAJ trajectories, CPU only)
# ===========================================================================
print(f'\n=== Part 1: Geometric analysis over {N_TRAJ} trajectories ===')

geo = dict(
    perp_mean=[], perp_max=[], perp_rms=[],
    dx_all=[], dy_all=[], dz_all=[],
    angle_deg=[],
    true_len=[], guess_len=[],
    # Extent: distance from true endpoints to the guess line
    start_miss=[], end_miss=[],
    # dE/dx ratio: guess mean vs. true mean
    dedx_true_mean=[], dedx_true_std=[],
    # Number of true segments vs. guess segments
    n_seg_true=[], n_seg_guess=[],
)

for t in range(min(N_TRAJ, len(dataset.trajectory_row_indices))):
    rows = dataset.trajectory_row_indices[t]
    sel  = dataset.tracks_struct[rows]
    true_track = np_from_structured(sel)
    guess_track = make_straight_line_guess(true_track, fields)
    if guess_track is None:
        continue

    # True track points
    true_starts = np.stack([true_track[:, fields.index('x_start')],
                            true_track[:, fields.index('y_start')],
                            true_track[:, fields.index('z_start')]], axis=1)
    true_ends   = np.stack([true_track[:, fields.index('x_end')],
                            true_track[:, fields.index('y_end')],
                            true_track[:, fields.index('z_end')]], axis=1)
    true_pts    = np.vstack([true_starts, true_ends])

    # Guess line start/end
    gs = np.array([guess_track[0, fields.index('x_start')],
                   guess_track[0, fields.index('y_start')],
                   guess_track[0, fields.index('z_start')]])
    ge = np.array([guess_track[-1, fields.index('x_end')],
                   guess_track[-1, fields.index('y_end')],
                   guess_track[-1, fields.index('z_end')]])

    # 1. Perpendicular distance from true points to guess line
    dists, perp_vecs = perp_dist_pts_to_line(true_pts, gs, ge)
    geo['perp_mean'].append(dists.mean())
    geo['perp_max'].append(dists.max())
    geo['perp_rms'].append(np.sqrt((dists**2).mean()))
    geo['dx_all'].extend(perp_vecs[:, 0].tolist())
    geo['dy_all'].extend(perp_vecs[:, 1].tolist())
    geo['dz_all'].extend(perp_vecs[:, 2].tolist())

    # 2. True track extent and guess extent
    tp0, tp1, tlen = track_endpoints(true_track)
    gp0, gp1, glen = track_endpoints(guess_track)
    geo['true_len'].append(tlen)
    geo['guess_len'].append(glen)

    # 3. Direction angle
    true_dir  = (tp1 - tp0) / (np.linalg.norm(tp1 - tp0) + 1e-12)
    guess_dir = (ge - gs)   / (np.linalg.norm(ge - gs)   + 1e-12)
    cos_a = np.clip(np.dot(true_dir, guess_dir), -1, 1)
    geo['angle_deg'].append(np.degrees(np.arccos(np.abs(cos_a))))

    # 4. Extent coverage: does guess span the true endpoints?
    # Project true endpoints onto guess line; check they fall within [0, glen]
    v    = ge - gs
    vlen = np.linalg.norm(v) + 1e-12
    proj_p0 = np.dot(tp0 - gs, v) / vlen  # should be ~0
    proj_p1 = np.dot(tp1 - gs, v) / vlen  # should be ~glen
    # Miss = how far each true endpoint is from the guess line endpoints
    geo['start_miss'].append(abs(proj_p0))
    geo['end_miss'].append(abs(proj_p1 - vlen))

    # 5. True dE/dx variation
    dx_vals  = true_track[:, fields.index('dx')]
    de_vals  = true_track[:, fields.index('dE')]
    dedx_seg = de_vals / (dx_vals + 1e-12)
    geo['dedx_true_mean'].append(dedx_seg.mean())
    geo['dedx_true_std'].append(dedx_seg.std())

    # 6. Segment counts
    geo['n_seg_true'].append(len(true_track))
    geo['n_seg_guess'].append(len(guess_track))

# Convert to arrays
for k in ['perp_mean', 'perp_max', 'perp_rms', 'angle_deg',
          'true_len', 'guess_len', 'start_miss', 'end_miss',
          'dedx_true_mean', 'dedx_true_std', 'n_seg_true', 'n_seg_guess']:
    geo[k] = np.array(geo[k])
for k in ['dx_all', 'dy_all', 'dz_all']:
    geo[k] = np.array(geo[k])

# Print summary
N = len(geo['perp_mean'])
print(f'Analyzed {N} trajectories.')
print(f'\n--- Perpendicular distance (true pts -> guess line) ---')
print(f'  Mean of means : {geo["perp_mean"].mean()*10:.3f} mm')
print(f'  Mean of maxes : {geo["perp_max"].mean()*10:.3f} mm')
print(f'  Mean of RMSEs : {geo["perp_rms"].mean()*10:.3f} mm')
print(f'\n--- Opening angle (true direction vs. guess line) ---')
print(f'  Mean: {geo["angle_deg"].mean():.4f} deg')
print(f'  Max : {geo["angle_deg"].max():.4f} deg')
print(f'  >0.1 deg: {(geo["angle_deg"]>0.1).sum()} / {N}')
print(f'\n--- Length comparison ---')
print(f'  True len mean : {geo["true_len"].mean():.2f} cm')
print(f'  Guess len mean: {geo["guess_len"].mean():.2f} cm')
print(f'  Len ratio (guess/true) mean: {(geo["guess_len"]/geo["true_len"]).mean():.4f}')
print(f'\n--- Extent coverage (endpoint miss distance) ---')
print(f'  Start miss mean: {geo["start_miss"].mean()*10:.3f} mm')
print(f'  End   miss mean: {geo["end_miss"].mean()*10:.3f} mm')
print(f'\n--- dE/dx variation in true tracks ---')
print(f'  Mean dE/dx: {geo["dedx_true_mean"].mean():.4f} MeV/cm')
print(f'  Mean std(dE/dx) per track: {geo["dedx_true_std"].mean():.4f} MeV/cm')
print(f'  Std/Mean ratio (intra-track): {(geo["dedx_true_std"]/geo["dedx_true_mean"]).mean():.3f}')

# --- PLOT 1: Geometric deviations ---
fig = plt.figure(figsize=(18, 14))
gs_fig = GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.35)

# Row 0: perpendicular distance distributions
ax = fig.add_subplot(gs_fig[0, 0])
ax.hist(geo['perp_mean']*10, bins=50, color='steelblue', edgecolor='k', alpha=0.7)
ax.axvline(geo['perp_mean'].mean()*10, color='red', ls='--',
           label=f'Mean={geo["perp_mean"].mean()*10:.2f} mm')
ax.set_xlabel('Mean perp. distance per track [mm]')
ax.set_ylabel('Count')
ax.set_title('Mean perp. dist. (true pts → guess line)')
ax.legend(); ax.grid(alpha=0.3)

ax = fig.add_subplot(gs_fig[0, 1])
ax.hist(geo['perp_max']*10, bins=50, color='tomato', edgecolor='k', alpha=0.7)
ax.axvline(geo['perp_max'].mean()*10, color='red', ls='--',
           label=f'Mean={geo["perp_max"].mean()*10:.2f} mm')
ax.set_xlabel('Max perp. distance per track [mm]')
ax.set_title('Max perp. dist. (true pts → guess line)')
ax.legend(); ax.grid(alpha=0.3)

ax = fig.add_subplot(gs_fig[0, 2])
ax.hist(geo['angle_deg'], bins=50, color='mediumseagreen', edgecolor='k', alpha=0.7)
ax.axvline(geo['angle_deg'].mean(), color='red', ls='--',
           label=f'Mean={geo["angle_deg"].mean():.3f}°')
ax.set_xlabel('Opening angle [degrees]')
ax.set_title('True vs. guess direction angle')
ax.legend(); ax.grid(alpha=0.3)

# Row 1: oriented deviations
for col_i, (arr, label, color) in enumerate(zip(
        [geo['dx_all'], geo['dy_all'], geo['dz_all']],
        ['dx [mm]', 'dy [mm]', 'dz [mm]'],
        ['lightskyblue', 'lightcoral', 'palegreen'])):
    ax = fig.add_subplot(gs_fig[1, col_i])
    d = arr * 10  # cm -> mm
    mu, sig = d.mean(), d.std()
    ax.hist(d, bins=100, density=True, color=color, edgecolor='k', alpha=0.7)
    x = np.linspace(d.min(), d.max(), 300)
    ax.plot(x, np.exp(-0.5*((x-mu)/sig)**2)/(sig*np.sqrt(2*np.pi)),
            'r-', lw=2, label=f'μ={mu:.3f} mm\nσ={sig:.3f} mm')
    ax.set_xlabel(f'Oriented deviation {label}')
    ax.set_ylabel('Density')
    ax.set_title(f'Oriented deviation {label}')
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

# Row 2: length comparison, extent coverage, dE/dx spread
ax = fig.add_subplot(gs_fig[2, 0])
ax.scatter(geo['true_len'], geo['guess_len'], s=5, alpha=0.5, color='navy')
lmin, lmax = geo['true_len'].min(), geo['true_len'].max()
ax.plot([lmin, lmax], [lmin, lmax], 'r--', label='y=x')
ax.set_xlabel('True track length [cm]')
ax.set_ylabel('Guess (straight-line) length [cm]')
ax.set_title('Track length: true vs. guess')
ax.legend(); ax.grid(alpha=0.3)

ax = fig.add_subplot(gs_fig[2, 1])
miss = np.concatenate([geo['start_miss'], geo['end_miss']]) * 10
ax.hist(miss, bins=60, color='orange', edgecolor='k', alpha=0.8)
ax.axvline(miss.mean(), color='red', ls='--', label=f'Mean={miss.mean():.3f} mm')
ax.set_xlabel('Endpoint miss distance [mm]')
ax.set_title('True endpoint distance from guess line ends')
ax.legend(); ax.grid(alpha=0.3)

ax = fig.add_subplot(gs_fig[2, 2])
rel_std = geo['dedx_true_std'] / (geo['dedx_true_mean'] + 1e-12)
ax.hist(rel_std, bins=50, color='plum', edgecolor='k', alpha=0.8)
ax.axvline(rel_std.mean(), color='red', ls='--', label=f'Mean={rel_std.mean():.3f}')
ax.set_xlabel('std(dE/dx) / mean(dE/dx) per track')
ax.set_title('Intra-track dE/dx variation (relative)')
ax.legend(); ax.grid(alpha=0.3)

fig.suptitle(f'Geometric Analysis of Straight-Line Approximation ({N} trajectories)',
             fontsize=14, fontweight='bold')
fig.savefig(f'{OUT_DIR}/01_geometric_analysis.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'Saved 01_geometric_analysis.png')


# ===========================================================================
# PART 2 — PIXEL & CHARGE COMPARISON  (N_TRAJ_SIM trajectories, needs GPU)
# ===========================================================================
print(f'\n=== Part 2: Pixel & charge comparison over {N_TRAJ_SIM} trajectories ===')

pixel_overlap_frac = []    # |intersection| / |union| of active pixels
charge_ratio       = []    # total charge guess / total charge true
adc_rms_diff       = []    # per-pixel ADC RMS difference (on common pixels)
n_pix_true         = []
n_pix_guess        = []
n_pix_common       = []

for t in range(min(N_TRAJ_SIM, len(dataset.trajectory_row_indices))):
    rows = dataset.trajectory_row_indices[t]
    sel  = dataset.tracks_struct[rows]
    true_track  = np_from_structured(sel)
    guess_track = make_straight_line_guess(true_track, fields)
    if guess_track is None:
        continue

    # Fix mean dE/dx for guess
    guess_mean = guess_track.copy()
    total_de   = true_track[:, fields.index('dE')].sum()
    total_dx   = guess_track[:, fields.index('dx')].sum()
    mean_dedx  = total_de / total_dx
    guess_mean[:, fields.index('dE')] = mean_dedx * guess_mean[:, fields.index('dx')]
    if 'dEdx' in fields:
        guess_mean[:, fields.index('dEdx')] = mean_dedx

    # Chop and simulate
    tc, _ = chop_tracks(true_track,  fields, precision=CHOP_PREC)
    gc, _ = chop_tracks(guess_mean,  fields, precision=CHOP_PREC)

    try:
        pred_t = sim_strategy.predict(params, jax.device_put(tc), fields, rngkey=t)
        pred_g = sim_strategy.predict(params, jax.device_put(gc), fields, rngkey=t)
    except Exception as e:
        print(f'  Traj {t}: simulation failed: {e}')
        continue

    pix_t = set(int(x) for x in np.array(pred_t['hit_pixels']) if x >= 0)
    pix_g = set(int(x) for x in np.array(pred_g['hit_pixels']) if x >= 0)
    union     = pix_t | pix_g
    intersect = pix_t & pix_g

    n_pix_true.append(len(pix_t))
    n_pix_guess.append(len(pix_g))
    n_pix_common.append(len(intersect))
    pixel_overlap_frac.append(len(intersect) / len(union) if union else 0.0)

    # Total charge (sum of ADCs as proxy)
    adcs_t = np.array(pred_t['adcs']).sum() if len(pred_t['adcs']) > 0 else 0.0
    adcs_g = np.array(pred_g['adcs']).sum() if len(pred_g['adcs']) > 0 else 0.0
    charge_ratio.append(adcs_g / (adcs_t + 1e-6))

    print(f'  Traj {t:3d}: pix_true={len(pix_t):4d}, pix_guess={len(pix_g):4d}, '
          f'overlap={len(intersect)/len(union)*100:.1f}%, '
          f'charge_ratio={adcs_g/(adcs_t+1e-6):.3f}')

pixel_overlap_frac = np.array(pixel_overlap_frac)
charge_ratio       = np.array(charge_ratio)
n_pix_true         = np.array(n_pix_true)
n_pix_guess        = np.array(n_pix_guess)
n_pix_common       = np.array(n_pix_common)

print(f'\n--- Pixel summary ({len(pixel_overlap_frac)} tracks) ---')
print(f'  Mean active pixels (true) : {n_pix_true.mean():.1f}')
print(f'  Mean active pixels (guess): {n_pix_guess.mean():.1f}')
print(f'  Mean common pixels        : {n_pix_common.mean():.1f}')
print(f'  Mean Jaccard overlap      : {pixel_overlap_frac.mean():.3f}  ({pixel_overlap_frac.mean()*100:.1f}%)')
print(f'  Mean charge ratio (g/t)   : {charge_ratio.mean():.4f}')
print(f'  Std  charge ratio (g/t)   : {charge_ratio.std():.4f}')

# --- PLOT 2: Pixel & charge ---
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

ax = axes[0, 0]
ax.hist(pixel_overlap_frac*100, bins=20, color='steelblue', edgecolor='k', alpha=0.8)
ax.axvline(pixel_overlap_frac.mean()*100, color='red', ls='--',
           label=f'Mean={pixel_overlap_frac.mean()*100:.1f}%')
ax.set_xlabel('Pixel Jaccard overlap [%]')
ax.set_title('Pixel set overlap (true ∩ guess) / (true ∪ guess)')
ax.legend(); ax.grid(alpha=0.3)

ax = axes[0, 1]
ax.scatter(n_pix_true, n_pix_guess, s=15, alpha=0.7, color='navy')
lim = max(n_pix_true.max(), n_pix_guess.max())
ax.plot([0, lim], [0, lim], 'r--', label='y=x')
ax.set_xlabel('Active pixels (true track)')
ax.set_ylabel('Active pixels (guess track)')
ax.set_title('Number of active pixels: true vs. guess')
ax.legend(); ax.grid(alpha=0.3)

ax = axes[0, 2]
ax.hist(charge_ratio, bins=20, color='tomato', edgecolor='k', alpha=0.8)
ax.axvline(1.0, color='black', ls=':', label='Ratio=1 (no bias)')
ax.axvline(charge_ratio.mean(), color='red', ls='--',
           label=f'Mean={charge_ratio.mean():.3f}')
ax.set_xlabel('Total ADC ratio (guess / true)')
ax.set_title('Total charge: guess vs. true')
ax.legend(); ax.grid(alpha=0.3)

ax = axes[1, 0]
ax.scatter(n_pix_true, pixel_overlap_frac*100, s=10, alpha=0.6, color='seagreen')
ax.set_xlabel('Active pixels (true track)')
ax.set_ylabel('Jaccard overlap [%]')
ax.set_title('Overlap vs. number of true pixels')
ax.grid(alpha=0.3)

ax = axes[1, 1]
ax.scatter(geo['true_len'][:len(pixel_overlap_frac)],
           pixel_overlap_frac*100, s=10, alpha=0.6, color='purple')
ax.set_xlabel('True track length [cm]')
ax.set_ylabel('Jaccard overlap [%]')
ax.set_title('Overlap vs. track length')
ax.grid(alpha=0.3)

ax = axes[1, 2]
ax.scatter(geo['perp_mean'][:len(pixel_overlap_frac)]*10,
           pixel_overlap_frac*100, s=10, alpha=0.6, color='darkorange')
ax.set_xlabel('Mean perp. distance [mm]')
ax.set_ylabel('Jaccard overlap [%]')
ax.set_title('Pixel overlap vs. geometric deviation')
ax.grid(alpha=0.3)

fig.suptitle(f'Pixel & Charge Comparison: True vs. Guess Track ({len(pixel_overlap_frac)} trajectories)',
             fontsize=13, fontweight='bold')
fig.tight_layout()
fig.savefig(f'{OUT_DIR}/02_pixel_charge_comparison.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'Saved 02_pixel_charge_comparison.png')


# ===========================================================================
# PART 3 — EXAMPLE TRACK: detailed visual comparison (first 3 trajectories)
# ===========================================================================
print(f'\n=== Part 3: Detailed single-track comparisons ===')

for t in range(min(3, len(dataset.trajectory_row_indices))):
    rows = dataset.trajectory_row_indices[t]
    sel  = dataset.tracks_struct[rows]
    true_track  = np_from_structured(sel)
    guess_track = make_straight_line_guess(true_track, fields)
    if guess_track is None:
        continue

    guess_mean = guess_track.copy()
    total_de   = true_track[:, fields.index('dE')].sum()
    total_dx_g = guess_track[:, fields.index('dx')].sum()
    mean_dedx  = total_de / total_dx_g
    guess_mean[:, fields.index('dE')] = mean_dedx * guess_mean[:, fields.index('dx')]
    if 'dEdx' in fields:
        guess_mean[:, fields.index('dEdx')] = mean_dedx

    tc, _ = chop_tracks(true_track,  fields, precision=CHOP_PREC)
    gc, _ = chop_tracks(guess_mean,  fields, precision=CHOP_PREC)

    # True segment dE/dx along arc length
    dx_arr   = true_track[:, fields.index('dx')]
    de_arr   = true_track[:, fields.index('dE')]
    arc_len  = np.concatenate([[0], np.cumsum(dx_arr)])
    arc_mid  = 0.5 * (arc_len[:-1] + arc_len[1:])
    dedx_seg = de_arr / (dx_arr + 1e-12)

    # 3D coordinates
    xs = np.concatenate([tc[:, fields.index('x_start')], [tc[-1, fields.index('x_end')]]])
    ys = np.concatenate([tc[:, fields.index('y_start')], [tc[-1, fields.index('y_end')]]])
    zs = np.concatenate([tc[:, fields.index('z_start')], [tc[-1, fields.index('z_end')]]])
    xg = np.concatenate([gc[:, fields.index('x_start')], [gc[-1, fields.index('x_end')]]])
    yg = np.concatenate([gc[:, fields.index('y_start')], [gc[-1, fields.index('y_end')]]])
    zg = np.concatenate([gc[:, fields.index('z_start')], [gc[-1, fields.index('z_end')]]])

    fig = plt.figure(figsize=(18, 10))
    gs2 = GridSpec(2, 4, figure=fig, hspace=0.4, wspace=0.35)

    # XY view
    ax = fig.add_subplot(gs2[0, 0])
    ax.plot(xs, ys, 'b-', lw=1, label='True')
    ax.plot(xg, yg, 'r--', lw=2, label='Guess (straight)')
    ax.set_xlabel('X [cm]'); ax.set_ylabel('Y [cm]')
    ax.set_title(f'Traj {t}: XY view'); ax.legend(); ax.grid(alpha=0.3)

    # XZ view
    ax = fig.add_subplot(gs2[0, 1])
    ax.plot(xs, zs, 'b-', lw=1, label='True')
    ax.plot(xg, zg, 'r--', lw=2, label='Guess')
    ax.set_xlabel('X [cm]'); ax.set_ylabel('Z [cm]')
    ax.set_title(f'Traj {t}: XZ view'); ax.legend(); ax.grid(alpha=0.3)

    # YZ view
    ax = fig.add_subplot(gs2[0, 2])
    ax.plot(ys, zs, 'b-', lw=1, label='True')
    ax.plot(yg, zg, 'r--', lw=2, label='Guess')
    ax.set_xlabel('Y [cm]'); ax.set_ylabel('Z [cm]')
    ax.set_title(f'Traj {t}: YZ view'); ax.legend(); ax.grid(alpha=0.3)

    # dE/dx along arc length
    ax = fig.add_subplot(gs2[0, 3])
    ax.step(arc_mid, dedx_seg, where='mid', color='blue', lw=1.5, label='True dE/dx')
    ax.axhline(mean_dedx, color='red', ls='--', lw=2, label=f'Guess (const={mean_dedx:.3f})')
    ax.set_xlabel('Arc length [cm]'); ax.set_ylabel('dE/dx [MeV/cm]')
    ax.set_title(f'Traj {t}: dE/dx profile'); ax.legend(); ax.grid(alpha=0.3)

    # Simulate and compare waveforms per pixel
    try:
        pred_t = sim_strategy.predict(params, jax.device_put(tc), fields, rngkey=0)
        pred_g = sim_strategy.predict(params, jax.device_put(gc), fields, rngkey=0)

        pix_t_ids = set(int(x) for x in np.array(pred_t['hit_pixels']) if x >= 0)
        pix_g_ids = set(int(x) for x in np.array(pred_g['hit_pixels']) if x >= 0)
        common = sorted(pix_t_ids & pix_g_ids)

        # Pixel ADC comparison on common pixels
        adcs_t_all = np.array(pred_t['adcs']) if len(pred_t['adcs']) > 0 else np.array([])
        adcs_g_all = np.array(pred_g['adcs']) if len(pred_g['adcs']) > 0 else np.array([])

        ax = fig.add_subplot(gs2[1, 0])
        all_pix = sorted(pix_t_ids | pix_g_ids)
        in_t = np.array([p in pix_t_ids for p in all_pix])
        in_g = np.array([p in pix_g_ids for p in all_pix])
        x_pos = np.arange(len(all_pix))
        ax.bar(x_pos[in_t & in_g],  np.ones((in_t & in_g).sum()),  color='purple', alpha=0.7, label='Both')
        ax.bar(x_pos[in_t & ~in_g], np.ones((in_t & ~in_g).sum()), color='blue',   alpha=0.7, label='True only')
        ax.bar(x_pos[~in_t & in_g], np.ones((~in_t & in_g).sum()), color='red',    alpha=0.7, label='Guess only')
        ax.set_xlabel('Pixel index (sorted)'); ax.set_ylabel('Presence')
        ax.set_title(f'Traj {t}: pixel activation pattern\n'
                     f'T={len(pix_t_ids)}, G={len(pix_g_ids)}, '
                     f'Common={len(common)} ({len(common)/len(pix_t_ids|pix_g_ids)*100:.0f}%)')
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

        # Total ADC per hit for true and guess (sorted by tick)
        ax = fig.add_subplot(gs2[1, 1])
        if len(adcs_t_all) > 0 and len(adcs_g_all) > 0:
            ax.hist(adcs_t_all, bins=40, alpha=0.6, color='blue', label='True', density=True)
            ax.hist(adcs_g_all, bins=40, alpha=0.6, color='red',  label='Guess', density=True)
        ax.set_xlabel('ADC count per hit')
        ax.set_title(f'Traj {t}: per-hit ADC distribution')
        ax.legend(); ax.grid(alpha=0.3)

        # Cumulative charge along ticks (time)
        ticks_t = np.array(pred_t['ticks']) if len(pred_t['ticks']) > 0 else np.array([])
        ticks_g = np.array(pred_g['ticks']) if len(pred_g['ticks']) > 0 else np.array([])
        ax = fig.add_subplot(gs2[1, 2])
        if len(ticks_t) > 0:
            t_sort = np.argsort(ticks_t)
            ax.step(ticks_t[t_sort], np.cumsum(adcs_t_all[t_sort]),
                    color='blue', lw=1.5, label=f'True (tot={adcs_t_all.sum():.0f})')
        if len(ticks_g) > 0:
            g_sort = np.argsort(ticks_g)
            ax.step(ticks_g[g_sort], np.cumsum(adcs_g_all[g_sort]),
                    color='red', lw=1.5, ls='--', label=f'Guess (tot={adcs_g_all.sum():.0f})')
        ax.set_xlabel('Tick'); ax.set_ylabel('Cumulative ADC')
        ax.set_title(f'Traj {t}: cumulative charge vs. time')
        ax.legend(); ax.grid(alpha=0.3)

        # Pixel position comparison (x,y)
        px_t = np.array(pred_t['pixel_x'])
        py_t = np.array(pred_t['pixel_y'])
        px_g = np.array(pred_g['pixel_x'])
        py_g = np.array(pred_g['pixel_y'])
        ax = fig.add_subplot(gs2[1, 3])
        ax.scatter(px_t, py_t, s=30, color='blue', alpha=0.7, label='True', marker='o')
        ax.scatter(px_g, py_g, s=30, color='red',  alpha=0.7, label='Guess', marker='x')
        ax.set_xlabel('Pixel X'); ax.set_ylabel('Pixel Y')
        ax.set_title(f'Traj {t}: active pixel positions')
        ax.legend(); ax.grid(alpha=0.3)

    except Exception as e:
        print(f'  Traj {t} simulation error: {e}')

    fig.suptitle(f'Detailed Track Comparison — Trajectory {t}', fontsize=13, fontweight='bold')
    fig.savefig(f'{OUT_DIR}/03_track_{t}_detail.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved 03_track_{t}_detail.png')


# ===========================================================================
# PART 4 — NLL LOSS SCAN: true track vs. guess track
# ===========================================================================
print(f'\n=== Part 4: NLL loss scans — true vs. guess track ===')

prob_strategy = LUTProbabilisticSimulation(response)
prob_strategy.padded_small_nb = 256
prob_strategy.padded_large_nb = 256
loss_strategy = ProbabilisticLossStrategy(
    sigma_charge=params.RESET_NOISE_CHARGE, eps=1e-8, time_window=5
)

ab_vals = np.linspace(0.70, 0.90, 12)
N_SCAN  = 5
loss_true  = np.zeros((N_SCAN, len(ab_vals)))
loss_guess = np.zeros((N_SCAN, len(ab_vals)))

for t in range(N_SCAN):
    rows = dataset.trajectory_row_indices[t]
    sel  = dataset.tracks_struct[rows]
    true_track  = np_from_structured(sel)
    guess_track = make_straight_line_guess(true_track, fields)
    if guess_track is None:
        continue

    guess_mean = guess_track.copy()
    total_de   = true_track[:, fields.index('dE')].sum()
    total_dx_g = guess_track[:, fields.index('dx')].sum()
    mean_dedx  = total_de / total_dx_g
    guess_mean[:, fields.index('dE')] = mean_dedx * guess_mean[:, fields.index('dx')]
    if 'dEdx' in fields:
        guess_mean[:, fields.index('dEdx')] = mean_dedx

    tc, _ = chop_tracks(true_track,  fields, precision=CHOP_PREC)
    gc, _ = chop_tracks(guess_mean,  fields, precision=CHOP_PREC)
    tc_jax = jax.device_put(tc)
    gc_jax = jax.device_put(gc)

    # Generate target from the true track at nominal Ab
    target_pred = sim_strategy.predict(params, tc_jax, fields, rngkey=42)
    target = {
        'pixel_id': target_pred['hit_pixels'],
        'ticks':    target_pred['ticks'],
        'adcs':     target_pred['adcs'],
    }

    for j, ab in enumerate(ab_vals):
        p_eval = params.replace(Ab=ab)
        try:
            pred_t = prob_strategy.predict(p_eval, tc_jax, fields, rngkey=42)
            lv, _  = loss_strategy.compute(p_eval, pred_t, target)
            loss_true[t, j] = float(lv)

            pred_g = prob_strategy.predict(p_eval, gc_jax, fields, rngkey=42)
            lv, _  = loss_strategy.compute(p_eval, pred_g, target)
            loss_guess[t, j] = float(lv)
        except Exception as e:
            loss_true[t, j]  = np.nan
            loss_guess[t, j] = np.nan
    print(f'  Traj {t}: true min Ab={ab_vals[np.nanargmin(loss_true[t])]:.3f}, '
          f'guess min Ab={ab_vals[np.nanargmin(loss_guess[t])]:.3f}')

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
ax = axes[0]
for t in range(N_SCAN):
    ax.plot(ab_vals, loss_true[t]  - np.nanmin(loss_true[t]),
            color='blue', alpha=0.5, lw=1)
    ax.plot(ab_vals, loss_guess[t] - np.nanmin(loss_guess[t]),
            color='red',  alpha=0.5, lw=1, ls='--')
ax.plot([], [], color='blue', label='True track (relative NLL)')
ax.plot([], [], color='red',  ls='--', label='Guess track (relative NLL)')
ax.axvline(params.Ab, color='black', ls=':', label=f'True Ab={params.Ab}')
ax.set_xlabel('Ab'); ax.set_ylabel('Relative NLL')
ax.set_title('Ab NLL scan: true vs. guess track\n(each normalized to its own minimum)')
ax.legend(); ax.grid(alpha=0.3)

ax = axes[1]
best_ab_true  = ab_vals[np.nanargmin(loss_true,  axis=1)]
best_ab_guess = ab_vals[np.nanargmin(loss_guess, axis=1)]
ax.scatter(range(N_SCAN), best_ab_true,  color='blue', s=80, label='True track', zorder=3)
ax.scatter(range(N_SCAN), best_ab_guess, color='red',  s=80, marker='x', label='Guess track', zorder=3)
ax.axhline(params.Ab, color='black', ls='--', label=f'True Ab={params.Ab}')
ax.set_xlabel('Trajectory index'); ax.set_ylabel('Best-fit Ab')
ax.set_title('Best-fit Ab: true vs. guess track')
ax.legend(); ax.grid(alpha=0.3)

fig.suptitle('NLL Scan: Does the Guess Track Bias the Fitted Ab?', fontsize=13, fontweight='bold')
fig.tight_layout()
fig.savefig(f'{OUT_DIR}/04_nll_scan_ab.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'Saved 04_nll_scan_ab.png')

# ===========================================================================
# Summary
# ===========================================================================
print('\n' + '='*60)
print('SUMMARY')
print('='*60)
print(f'Geometric ({N} tracks):')
print(f'  Mean perp. dist. (true->guess): {geo["perp_mean"].mean()*10:.2f} mm')
print(f'  Mean max   dist. (true->guess): {geo["perp_max"].mean()*10:.2f} mm')
print(f'  Mean opening angle:             {geo["angle_deg"].mean():.4f} deg')
print(f'  Length bias (guess/true - 1):   {((geo["guess_len"]/geo["true_len"]).mean()-1)*100:.3f}%')
print(f'  Mean dE/dx intra-track std/mean:{(geo["dedx_true_std"]/geo["dedx_true_mean"]).mean():.3f}')
print(f'Pixel ({len(pixel_overlap_frac)} tracks):')
print(f'  Mean Jaccard overlap:    {pixel_overlap_frac.mean()*100:.1f}%')
print(f'  Mean charge ratio (g/t): {charge_ratio.mean():.4f} ± {charge_ratio.std():.4f}')
print(f'NLL scan Ab bias ({N_SCAN} tracks):')
print(f'  True track  best Ab: {best_ab_true.mean():.3f} ± {best_ab_true.std():.3f}')
print(f'  Guess track best Ab: {best_ab_guess.mean():.3f} ± {best_ab_guess.std():.3f}')
print(f'\nAll plots saved to {OUT_DIR}/')
print('Done.')
