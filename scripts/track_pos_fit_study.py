#!/usr/bin/env python3
"""
Track Position Fitting & Profiling Toy Study
=============================================
Compares three strategies for jointly fitting a calibration parameter (Ab)
and the track geometry (6-DOF: translation + rotation):

  Scenario A: Ab only  — geometry frozen at the straight-line guess (zeros)
  Scenario B: Joint    — Ab + 6D geometry fitted together via Adam (JAX/Optax)
  Scenario C: Alternating — Ab via Adam; geometry via L-BFGS-B profiling each step

Run via SLURM: sbatch scripts/track_pos_fit_study.slurm
"""
import sys, os
PROJECT = '/sdf/group/neutrino/pgranger/larnd-sim-jax'
sys.path.insert(0, PROJECT)
sys.path.insert(0, f'{PROJECT}/src')
os.chdir(PROJECT)

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import optax
from scipy.optimize import minimize

from larndsim.consts_jax import build_params_class, load_detector_properties, load_lut
from larndsim.sim_jax import simulate_wfs, simulate_probabilistic, pad_to_closest_multiple
from optimize.strategies import LUTSimulation, LUTProbabilisticSimulation, ProbabilisticLossStrategy
from optimize.dataio import TracksDataset, np_from_structured, chop_tracks
from optimize.ranges import ranges
from optimize.fit_params import apply_transformation, simulate_wfs_static

print('JAX devices:', jax.devices())

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
INPUT_FILE = '/sdf/data/neutrino/cyifan/diffsim_input/true_through_muon_edep_10cm_vol1cm.h5'
dataset = TracksDataset(
    filename=INPUT_FILE, nevents=10,
    electron_sampling_resolution=0.1, chopped=False, pad=False
)
print(f'Loaded {len(dataset)} trajectories.')

TRAJ_IDX = 0
traj_indices = dataset.batch_traj_indices[TRAJ_IDX]
rows = np.concatenate([dataset.trajectory_row_indices[t] for t in traj_indices])
selected_struct = dataset.tracks_struct[rows]
true_track_np = np_from_structured(selected_struct)
fields = dataset.get_track_fields()
print(f'Selected Trajectory {TRAJ_IDX} with {true_track_np.shape[0]} segments.')

# ---------------------------------------------------------------------------
# 2. Build straight-line guess track
# ---------------------------------------------------------------------------
def make_straight_line_guess(true_track, fields, segment_length=10.0):
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
    direction_unit = direction / (total_len + 1e-10)

    n_segs    = int(np.ceil(total_len / segment_length))
    step_size = total_len / n_segs
    total_de  = true_track[:, fields.index('dE')].sum()
    de_per_seg = total_de / n_segs

    guess_track = np.zeros((n_segs, len(fields)), dtype=np.float32)
    for col_name in ['eventID', 'trackID', 'pdgID', 't0']:
        if col_name in fields:
            guess_track[:, fields.index(col_name)] = true_track[0, fields.index(col_name)]

    for i in range(n_segs):
        seg_start = start_pos + i       * step_size * direction_unit
        seg_end   = start_pos + (i + 1) * step_size * direction_unit
        guess_track[i, fields.index('x_start')] = seg_start[0]
        guess_track[i, fields.index('y_start')] = seg_start[1]
        guess_track[i, fields.index('z_start')] = seg_start[2]
        guess_track[i, fields.index('x_end')]   = seg_end[0]
        guess_track[i, fields.index('y_end')]   = seg_end[1]
        guess_track[i, fields.index('z_end')]   = seg_end[2]
        guess_track[i, fields.index('x')]  = 0.5 * (seg_start[0] + seg_end[0])
        guess_track[i, fields.index('y')]  = 0.5 * (seg_start[1] + seg_end[1])
        guess_track[i, fields.index('z')]  = 0.5 * (seg_start[2] + seg_end[2])
        guess_track[i, fields.index('dE')] = de_per_seg
        guess_track[i, fields.index('dx')] = step_size
        if 'dEdx' in fields:
            guess_track[i, fields.index('dEdx')] = de_per_seg / step_size
    return guess_track

guess_track_np = make_straight_line_guess(true_track_np, fields, segment_length=10.0)
print(f'True track: {true_track_np.shape[0]} segments, guess: {guess_track_np.shape[0]} segments.')

# Chop both to fine precision
true_chopped,  _          = chop_tracks(true_track_np,  fields, precision=0.1)
guess_chopped, parent_ids = chop_tracks(guess_track_np, fields, precision=0.1)
print(f'True chopped: {true_chopped.shape[0]} pts, Guess chopped: {guess_chopped.shape[0]} pts.')

# ---------------------------------------------------------------------------
# 3. Detector setup
# ---------------------------------------------------------------------------
LUT_FILE  = 'src/larndsim/detector_properties/response_44_v2a_full_tick.npz'
DET_PROPS = 'src/larndsim/detector_properties/module0.yaml'
PIX_LAYOUT = 'src/larndsim/pixel_layouts/multi_tile_layout-2.4.16_v4.yaml'
RELEVANT_PARAMS = ['Ab', 'kb', 'lifetime', 'tran_diff', 'long_diff', 'eField',
                   'shift_x', 'shift_y', 'shift_z']

ParamsClass = build_params_class(RELEVANT_PARAMS)
ref_params  = load_detector_properties(ParamsClass, DET_PROPS, PIX_LAYOUT)
ref_params  = ref_params.replace(
    electron_sampling_resolution=0.1,
    number_pix_neighbors=4,
    signal_length=50,   # reduced from 150 to cut Nticks and backward-pass memory
    time_window=50,
    excess_offset=0.0,
    smooth_sigma_scale=1.0,
)
response, ref_params = load_lut(LUT_FILE, ref_params)
current_params = ref_params.replace(**{p: ranges[p]['nom'] for p in RELEVANT_PARAMS})

sim_strategy  = LUTSimulation(response)
prob_strategy = LUTProbabilisticSimulation(response)
sim_strategy.padded_small_nb  = 256
sim_strategy.padded_large_nb  = 256
prob_strategy.padded_small_nb = 256
prob_strategy.padded_large_nb = 256

# ---------------------------------------------------------------------------
# 4. Generate stochastic target
# ---------------------------------------------------------------------------
true_chopped_jax = jax.device_put(true_chopped)
target_pred = sim_strategy.predict(current_params, true_chopped_jax, fields, rngkey=42)
target = {
    'pixel_id': target_pred['hit_pixels'],
    'ticks':    target_pred['ticks'],
    'adcs':     target_pred['adcs'],
}
print(f'Target generated with {len(target["pixel_id"])} hits.')

loss_strategy = ProbabilisticLossStrategy(
    sigma_charge=current_params.RESET_NOISE_CHARGE, eps=1e-8, time_window=5
)

# ---------------------------------------------------------------------------
# 5. Setup static pixel list and JIT'd loss functions
# ---------------------------------------------------------------------------
init_Ab  = 0.70
true_Ab  = 0.80
steps    = 100

# Mean dEdx guess track (fixed at 1.887 MeV/cm for the toy)
guess_mean_dedx_chopped = guess_chopped.copy()
guess_mean_dedx_chopped[:, fields.index('dE')] = 1.887 * guess_mean_dedx_chopped[:, fields.index('dx')]
if 'dEdx' in fields:
    guess_mean_dedx_chopped[:, fields.index('dEdx')] = 1.887
guess_mean_dedx_chopped_jax = jax.device_put(guess_mean_dedx_chopped)

# Precompute static pixel list (union of true + guess pixels)
print('Precomputing static pixel lists...')
wfs_true,  true_pixels  = simulate_wfs(current_params, response, true_chopped_jax, fields)
wfs_guess, guess_pixels = simulate_wfs(current_params, response, guess_mean_dedx_chopped_jax, fields)

combined_pixels = np.unique(np.concatenate([
    np.array(true_pixels[true_pixels >= 0]),
    np.array(guess_pixels[guess_pixels >= 0]),
]))
combined_pixels = np.append(combined_pixels, -1)
padded_len = int(((len(combined_pixels) + 127) // 128) * 128)
fixed_pixels_np = pad_to_closest_multiple(combined_pixels, multiple=padded_len, pad_value=-1, pad_front=False)
fixed_pixels_np = np.sort(fixed_pixels_np)
fixed_pixels = jnp.array(fixed_pixels_np)
print(f'Static pixel list size: {padded_len}')

@jax.checkpoint
def _simulate_checkpointed(params_eval, tracks_eval):
    """The expensive simulation wrapped in checkpoint so the backward pass
    recomputes activations instead of storing them (trades FLOP for memory)."""
    wfs = simulate_wfs_static(params_eval, response, tracks_eval, fields, fixed_pixels)
    wfs_padded = pad_to_closest_multiple(wfs, dims_to_pad=(0,), multiple=128, pad_value=0.0, pad_front=True)
    adcs_distrib, pixel_x, pixel_y, ticks_prob, event = simulate_probabilistic(
        params_eval, wfs_padded, fixed_pixels
    )
    return adcs_distrib, ticks_prob, pixel_x, pixel_y, event

def predict_static(params_eval, tracks_eval):
    """JIT-compatible predict using precomputed fixed pixel list."""
    adcs_distrib, ticks_prob, pixel_x, pixel_y, event = _simulate_checkpointed(
        params_eval, tracks_eval
    )
    return {
        'adcs_distrib': adcs_distrib,
        'hit_prob':     ticks_prob,
        'pixel_x':      pixel_x,
        'pixel_y':      pixel_y,
        'unique_pixels': fixed_pixels,
        'event':        event,
    }

def _core_loss(ab_val, geom_vars):
    """Single differentiable loss: takes Ab (scalar) and 6D geom (JAX arrays)."""
    translation = geom_vars[0:3]
    rotation    = geom_vars[3:6] * 0.1
    transformed_track = apply_transformation(
        guess_mean_dedx_chopped_jax, fields, translation, rotation
    )
    p_eval   = current_params.replace(Ab=ab_val)
    pred     = predict_static(p_eval, transformed_track)
    loss_val, _ = loss_strategy.compute(p_eval, pred, target)
    return loss_val

# Compile all gradient functions once
_core_val_and_grad     = jax.jit(jax.value_and_grad(_core_loss, argnums=(0, 1)))
init_geom_vars         = jnp.zeros(6, dtype=jnp.float32)

def loss_fn_A(ab_val):
    loss_val, _ = _core_val_and_grad(ab_val, init_geom_vars)
    return loss_val

_grad_fn_A             = jax.jit(jax.value_and_grad(loss_fn_A))
_local_geom_val_and_grad = jax.jit(
    jax.value_and_grad(lambda gv, ab: _core_loss(ab, gv), argnums=0)
)
_ab_val_and_grad       = jax.jit(
    jax.value_and_grad(lambda ab, gv: _core_loss(ab, gv), argnums=0)
)

print('JIT-compiled gradient functions ready.')

# Warm up JIT (avoids timing contamination in first step)
print('Warming up JIT...')
_ = _grad_fn_A(jnp.array(init_Ab))
_ = _core_val_and_grad(jnp.array(init_Ab), init_geom_vars)
_ = _local_geom_val_and_grad(init_geom_vars, jnp.array(init_Ab))
_ = _ab_val_and_grad(jnp.array(init_Ab), init_geom_vars)
print('JIT warmup done.')

# ---------------------------------------------------------------------------
# 6. Scenario A: Ab only, geometry frozen at zero
# ---------------------------------------------------------------------------
def run_fit_A(steps=100, lr=0.005):
    ab_val    = jnp.array(init_Ab)
    optimizer = optax.adam(learning_rate=lr)
    opt_state = optimizer.init(ab_val)
    history_ab   = [float(ab_val)]
    history_loss = []

    print('\n--- Scenario A: Ab only (geometry frozen at zero) ---')
    for step in range(steps):
        loss_val, grad = _grad_fn_A(ab_val)
        updates, opt_state = optimizer.update(grad, opt_state)
        ab_val = optax.apply_updates(ab_val, updates)
        history_ab.append(float(ab_val))
        history_loss.append(float(loss_val))
        if step % 10 == 0 or step == steps - 1:
            print(f'  Step {step:3d}: Ab = {float(ab_val):.4f} (True: {true_Ab:.2f}), '
                  f'Loss = {float(loss_val):.4f}')
    return history_ab, history_loss

# ---------------------------------------------------------------------------
# 7. Scenario B: Joint fit (Ab + geometry via Adam)
# ---------------------------------------------------------------------------
def run_fit_B(steps=100, lr_ab=0.005, lr_geom=0.01):
    ab_val    = jnp.array(init_Ab)
    geom_vars = jnp.array(init_geom_vars)

    opt_ab    = optax.adam(learning_rate=lr_ab)
    state_ab  = opt_ab.init(ab_val)
    opt_geom  = optax.adam(learning_rate=lr_geom)
    state_geom = opt_geom.init(geom_vars)

    history_ab   = [float(ab_val)]
    history_geom = [np.array(geom_vars)]
    history_loss = []

    print('\n--- Scenario B: Joint fit (Ab + 6D geometry via Adam) ---')
    for step in range(steps):
        loss_val, (grad_ab, grad_geom) = _core_val_and_grad(ab_val, geom_vars)

        updates_ab, state_ab = opt_ab.update(grad_ab, state_ab)
        ab_val = optax.apply_updates(ab_val, updates_ab)

        updates_geom, state_geom = opt_geom.update(grad_geom, state_geom)
        geom_vars = optax.apply_updates(geom_vars, updates_geom)

        history_ab.append(float(ab_val))
        history_geom.append(np.array(geom_vars))
        history_loss.append(float(loss_val))
        if step % 10 == 0 or step == steps - 1:
            print(f'  Step {step:3d}: Ab = {float(ab_val):.4f} (True: {true_Ab:.2f}), '
                  f'Trans = {np.array(geom_vars[0:3])}, Loss = {float(loss_val):.4f}')
    return history_ab, history_geom, history_loss

# ---------------------------------------------------------------------------
# 8. Scenario C: Alternating (Ab via Adam, geometry via L-BFGS-B)
# ---------------------------------------------------------------------------
def run_fit_C(steps=100, lr_ab=0.005, track_max_iter=10):
    ab_val       = jnp.array(init_Ab)
    geom_vars_np = np.zeros(6, dtype=np.float64)

    opt_ab   = optax.adam(learning_rate=lr_ab)
    state_ab = opt_ab.init(ab_val)

    history_ab   = [float(ab_val)]
    history_geom = [geom_vars_np.copy()]
    history_loss = []

    print('\n--- Scenario C: Alternating (Adam on Ab + L-BFGS-B on geometry) ---')
    for step in range(steps):
        # Inner: minimize geometry for current Ab
        ab_val_fixed = ab_val  # JAX array — no retrace on value change
        def scipy_loss(x):
            gv = jnp.array(x, dtype=jnp.float32)
            l, g = _local_geom_val_and_grad(gv, ab_val_fixed)
            return float(l), np.array(g, dtype=np.float64)

        res = minimize(
            scipy_loss, geom_vars_np, jac=True, method='L-BFGS-B',
            bounds=[(-2.0, 2.0)] * 3 + [(-31.4, 31.4)] * 3,
            options={'maxiter': track_max_iter, 'ftol': 1e-10, 'gtol': 1e-6},
        )
        geom_vars_np = res.x

        # Outer: update Ab at aligned geometry
        geom_vars_jax = jnp.array(geom_vars_np, dtype=jnp.float32)
        loss_val, grad_ab = _ab_val_and_grad(ab_val, geom_vars_jax)
        updates_ab, state_ab = opt_ab.update(grad_ab, state_ab)
        ab_val = optax.apply_updates(ab_val, updates_ab)

        history_ab.append(float(ab_val))
        history_geom.append(geom_vars_np.copy())
        history_loss.append(float(loss_val))
        if step % 10 == 0 or step == steps - 1:
            print(f'  Step {step:3d}: Ab = {float(ab_val):.4f} (True: {true_Ab:.2f}), '
                  f'Trans = {geom_vars_np[0:3]}, Loss = {float(loss_val):.4f}')
    return history_ab, history_geom, history_loss

# ---------------------------------------------------------------------------
# 9. Run all scenarios
# ---------------------------------------------------------------------------
history_ab_A, history_loss_A = run_fit_A(steps=steps)
history_ab_B, history_geom_B, history_loss_B = run_fit_B(steps=steps)
history_ab_C, history_geom_C, history_loss_C = run_fit_C(steps=steps)

# ---------------------------------------------------------------------------
# 10. Save results to numpy
# ---------------------------------------------------------------------------
OUT_DIR = 'scripts/track_pos_fit_results'
os.makedirs(OUT_DIR, exist_ok=True)

np.savez(
    f'{OUT_DIR}/results.npz',
    history_ab_A   = np.array(history_ab_A),
    history_loss_A = np.array(history_loss_A),
    history_ab_B   = np.array(history_ab_B),
    history_loss_B = np.array(history_loss_B),
    history_geom_B = np.array(history_geom_B),
    history_ab_C   = np.array(history_ab_C),
    history_loss_C = np.array(history_loss_C),
    history_geom_C = np.array(history_geom_C),
    true_Ab        = np.array(true_Ab),
    init_Ab        = np.array(init_Ab),
)
print(f'\nResults saved to {OUT_DIR}/results.npz')

# ---------------------------------------------------------------------------
# 11. Plots
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# --- Ab convergence ---
ax = axes[0, 0]
ax.plot(history_ab_A, label='A: Ab only (frozen geometry)', color='tab:red',   marker='o', markevery=10)
ax.plot(history_ab_B, label='B: Joint (Ab + geometry)',     color='tab:blue',  marker='s', markevery=10)
ax.plot(history_ab_C, label='C: Alternating profiling',     color='tab:green', marker='^', markevery=10)
ax.axhline(true_Ab, color='black', linestyle='--', label=f'True Ab ({true_Ab:.2f})')
ax.set_xlabel('Iteration')
ax.set_ylabel('Fitted Ab')
ax.set_title('Recombination Parameter Ab Convergence')
ax.legend()
ax.grid(True, alpha=0.3)

# --- Loss convergence ---
ax = axes[0, 1]
ax.plot(history_loss_A, label='A: Ab only',        color='tab:red',   marker='o', markevery=10)
ax.plot(history_loss_B, label='B: Joint',          color='tab:blue',  marker='s', markevery=10)
ax.plot(history_loss_C, label='C: Alternating',    color='tab:green', marker='^', markevery=10)
ax.set_xlabel('Iteration')
ax.set_ylabel('NLL Loss')
ax.set_title('Loss Convergence')
ax.legend()
ax.grid(True, alpha=0.3)

# --- Translation convergence (B and C) ---
ax = axes[1, 0]
geom_B = np.array(history_geom_B)
geom_C = np.array(history_geom_C)
for k, label in enumerate(['tx', 'ty', 'tz']):
    ax.plot(geom_B[:, k], linestyle='-',  label=f'B {label}')
    ax.plot(geom_C[:, k], linestyle='--', label=f'C {label}')
ax.axhline(0, color='black', linestyle=':', label='True (zero)')
ax.set_xlabel('Iteration')
ax.set_ylabel('Translation [cm]')
ax.set_title('Translation Components (B vs C)')
ax.legend(ncol=2, fontsize=8)
ax.grid(True, alpha=0.3)

# --- Rotation convergence (B and C) ---
ax = axes[1, 1]
for k, label in enumerate(['rx', 'ry', 'rz']):
    ax.plot(geom_B[:, 3 + k] * 0.1, linestyle='-',  label=f'B {label}')
    ax.plot(geom_C[:, 3 + k] * 0.1, linestyle='--', label=f'C {label}')
ax.axhline(0, color='black', linestyle=':', label='True (zero)')
ax.set_xlabel('Iteration')
ax.set_ylabel('Rotation angle [rad]')
ax.set_title('Rotation Components (B vs C)')
ax.legend(ncol=2, fontsize=8)
ax.grid(True, alpha=0.3)

plt.suptitle(
    'Track Position Fitting Toy Study\n'
    'A: Ab only  |  B: Joint JAX/Optax  |  C: Alternating L-BFGS-B',
    fontsize=13,
)
plt.tight_layout()
plot_path = f'{OUT_DIR}/convergence.png'
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
print(f'Plot saved to {plot_path}')
plt.close()

print('\nDone.')
