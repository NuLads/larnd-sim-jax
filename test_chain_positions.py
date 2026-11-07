"""
test_chain_positions.py
=======================
Smoke-test + demonstration of the new fit_chain_positions=True mode in
GradientDescentFitter. Runs on a single GPU (or CPU if --cpu) for a small
number of iterations and prints a summary showing that:

  1. Chain contexts are built (one per track in each batch).
  2. Chain angles update from their initial values.
  3. The MCS prior appears in the loss auxiliary output.
  4. Ab moves toward the true value over the run.

Usage (inside Singularity):
    python3 test_chain_positions.py [--iterations 30] [--cpu]
"""
import sys, os, argparse, time
import numpy as np

sys.path.insert(0, '/sdf/group/neutrino/pgranger/larnd-sim-jax')
sys.path.insert(0, '/sdf/group/neutrino/pgranger/larnd-sim-jax/src')
os.chdir('/sdf/group/neutrino/pgranger/larnd-sim-jax')

parser = argparse.ArgumentParser()
parser.add_argument('--iterations', type=int, default=30)
parser.add_argument('--max_nbatch', type=int, default=2)
parser.add_argument('--cpu', action='store_true')
args = parser.parse_args()

if args.cpu:
    os.environ['JAX_PLATFORMS'] = 'cpu'

import jax
import jax.numpy as jnp

print('JAX devices:', jax.devices())

from larndsim.consts_jax import build_params_class, load_detector_properties, load_lut
from optimize.dataio import TracksDataset, TgtTracksDataset, DataLoader
from optimize.fit_params import GradientDescentFitter

# ── Paths ─────────────────────────────────────────────────────────────────────
INPUT_SIM = '/sdf/group/neutrino/pgranger/lads-data/linear_guess_segments.h5'
INPUT_TGT = '/sdf/data/neutrino/cyifan/diffsim_input/true_through_muon_edep_10cm_vol1cm.h5'
DETECTOR  = 'src/larndsim/detector_properties/module0.yaml'
PIXELS    = 'src/larndsim/pixel_layouts/multi_tile_layout-2.4.16_v4.yaml'
LUT_FILE  = 'src/larndsim/detector_properties/response_44.npy'
AB_TRUE   = 0.8348813503927325

# ── Data loading ─────────────────────────────────────────────────────────────
print('\n=== Loading data ===')
dataset_sim = TracksDataset(
    filename=INPUT_SIM, nevents=-1, max_nbatch=args.max_nbatch,
    track_len_sel=2., max_abs_costheta_sel=0.966, min_abs_segz_sel=15.,
    track_z_bound=28., max_batch_len=200.,
    electron_sampling_resolution=0.01, chopped=True, pad=True,
)
dataset_tgt = TgtTracksDataset(
    filename=INPUT_TGT, dataset_sim=dataset_sim,
    electron_sampling_resolution=0.01, chopped=True, pad=True,
)
dl_sim = DataLoader(dataset_sim, batch_size=1)
dl_tgt = DataLoader(dataset_tgt, batch_size=1)
print(f'Batches: {len(dl_sim)},  sim fields: {dataset_sim.get_track_fields()[:6]} ...')

# Report number of fine steps per batch
for bi in range(len(dl_sim)):
    trk = np.asarray(dl_sim[bi]).reshape(-1, len(dataset_sim.get_track_fields()))
    ev_f = dataset_sim.get_track_fields().index('eventID')
    tr_f = dataset_sim.get_track_fields().index('trackID')
    pairs = set(zip(trk[:, ev_f].tolist(), trk[:, tr_f].tolist()))
    print(f'  batch {bi}: {trk.shape[0]} fine steps, {len(pairs)} tracks')

# ── Fitter ────────────────────────────────────────────────────────────────────
print('\n=== Creating fitter ===')
Params = build_params_class(['Ab'])
ref_params = load_detector_properties(
    Params, DETECTOR, PIXELS)
ref_params, _ = load_lut(LUT_FILE, ref_params)

sim_fields = dataset_sim.get_track_fields()
tgt_fields = dataset_tgt.get_track_fields()

fitter = GradientDescentFitter(
    relevant_params={'Ab': 1e-1},
    set_init_params=[],
    sim_track_fields=sim_fields,
    tgt_track_fields=tgt_fields,
    detector_props=DETECTOR,
    pixel_layouts=PIXELS,
    lr=1e-1,
    out_label='chain_pos_test',
    test_name='chain_pos_test',
    readout_noise_target=False,
    readout_noise_guess=False,
    # dEdx fitting (required for chain geometry mode)
    fit_dedx=True,
    dedx_prior_weight=0.5,
    dedx_lr=1e-2,
    dedx_start_iter=0,
    dedx_use_split_t=True,
    dedx_student_nu_l=4.785,
    dedx_student_nu_r=2.073,
    dedx_student_scale_l=0.1204,
    dedx_student_scale_r=0.1058,
    dedx_soft_barrier_threshold=8.5,
    dedx_soft_barrier_weight=1.0,
    dedx_mean_constraint_weight=0.0,
    # Chain geometry fitting (new!)
    fit_chain_positions=True,
    chain_lr=3e-5,
    chain_start_iter=0,
    mcs_prior_weight=1.0,
    chain_step_len=2.0,
    chain_momentum_GeV=3.0,
    # Simulation settings
    probabilistic_sim=True,
    normalization_scheme='sigmoid',
    config={},
)

# Manually wire the ref_params so the fitter knows about the detector
fitter.ref_params = ref_params
fitter.response = ref_params  # will be overwritten by load_lut

print(f'fit_chain_positions = {fitter.fit_chain_positions}')
print(f'fit_dedx            = {fitter.fit_dedx}')

# ── Manual training loop to inspect internals ────────────────────────────────
print('\n=== Preparing fit (precomputes static pixels + chain contexts) ===')
fitter.prepare_fit()
fitter.precompute_static_pixels(dl_sim, dl_tgt)

print('\nChain contexts built:')
for bi, ctxs in fitter._batch_chain_contexts.items():
    print(f'  batch {bi}: {len(ctxs)} tracks')
    for ctx in ctxs:
        print(f'    track {ctx.track_id}  n_chain={ctx.n_chain}  '
              f'step_len={ctx.step_len:.2f} cm  L={ctx.total_len:.1f} cm  '
              f'sigma_mcs={ctx.sigma_mcs*1e3:.2f} mrad')

# Run the actual fit using the high-level method (captures loss/aux per iter)
print(f'\n=== Running {args.iterations} iterations ===')
t0 = time.time()

# Monkey-patch train loop to capture per-step loss and aux
ab_history  = []
mcs_history = []
loss_history = []

_orig_compute = fitter.compute_loss
_step_count = [0]

def _patched_compute(tracks, i, *a, **kw):
    result = _orig_compute(tracks, i, *a, **kw)
    loss_val = result[0]
    aux      = result[4]
    if aux is not None:
        loss_history.append(float(loss_val))
        mcs_history.append(float(aux.get('mcs_prior', 0.)))
    return result

fitter.compute_loss = _patched_compute

try:
    fitter.fit(dl_sim, dl_tgt, iterations=args.iterations, save_freq=9999)
except Exception as e:
    print(f'\nFitter raised exception: {e}')
    import traceback; traceback.print_exc()

elapsed = time.time() - t0
print(f'\nElapsed: {elapsed:.1f}s  ({elapsed/max(1,args.iterations):.2f}s/iter)')

# ── Report Ab ────────────────────────────────────────────────────────────────
ab_hist = fitter.training_history.get('Ab_iter', [])
print(f'\nAb trajectory (first 5, last 5):')
n = len(ab_hist)
for idx in list(range(min(5, n))) + list(range(max(0, n-5), n)):
    phys = float(jnp.array(ab_hist[idx]))
    print(f'  iter {idx:3d}: Ab = {phys:.6f}  (true={AB_TRUE:.6f}  bias={phys-AB_TRUE:+.6f})')

# ── Report chain angle updates ────────────────────────────────────────────────
print('\nChain angle states after fit:')
for bi, states in fitter._chain_cache.items():
    print(f'  batch {bi}:')
    for si, s in enumerate(states):
        ang = np.array(s['angles'])
        n_c = fitter._batch_chain_contexts[bi][si].n_chain
        th0, ph0 = ang[0], ang[1]
        dth = ang[2:n_c+1]
        dph = ang[n_c+1:]
        rms_dth = float(np.sqrt(np.mean(dth**2))) if len(dth) else 0.
        rms_dph = float(np.sqrt(np.mean(dph**2))) if len(dph) else 0.
        ctx = fitter._batch_chain_contexts[bi][si]
        print(f'    track {ctx.track_id}  th0={th0:.4f} ph0={ph0:.4f}  '
              f'rms(dtheta)={rms_dth*1e3:.2f}mrad  rms(dphi)={rms_dph*1e3:.2f}mrad  '
              f'(sigma_mcs={ctx.sigma_mcs*1e3:.2f}mrad)')

# ── Report loss / MCS prior ───────────────────────────────────────────────────
if loss_history:
    print(f'\nLoss: first={loss_history[0]:.4e}  last={loss_history[-1]:.4e}  '
          f'ratio={loss_history[-1]/loss_history[0]:.3f}')
if mcs_history:
    print(f'MCS prior: first={mcs_history[0]:.4e}  last={mcs_history[-1]:.4e}')

print('\n=== Test complete ===')
