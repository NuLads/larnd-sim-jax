"""Generates debug_simulation_loss.ipynb using plain json (no nbformat dependency)."""
import json, uuid

cells = []

def _id(): return uuid.uuid4().hex[:8]

def code(src):
    return {"cell_type":"code","execution_count":None,"id":_id(),
            "metadata":{},"outputs":[],
            "source": src if isinstance(src, list) else src.splitlines(keepends=True)}

def md(src):
    return {"cell_type":"markdown","id":_id(),"metadata":{},
            "source": src if isinstance(src, list) else src.splitlines(keepends=True)}

# ── Section 0: Setup ─────────────────────────────────────────────────────────
cells.append(md("# Simulation / Loss Debugging\n\n**Sections:** 1 Waveforms · 2 Stochastic vs Probabilistic · 3 Loss scan"))

cells.append(code("""\
%reload_ext autoreload
%autoreload 2
import sys, os
import jax, jax.numpy as jnp, numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import stats

project_root = '/sdf/group/neutrino/pgranger/larnd-sim-jax/src'
if project_root not in sys.path:
    sys.path.insert(0, project_root)
%cd /sdf/group/neutrino/pgranger/larnd-sim-jax

from larndsim.consts_jax import build_params_class, load_detector_properties, load_lut
from larndsim.losses_jax import adc2charge
from larndsim.fee_jax import get_adc_values, digitize
from larndsim.sim_jax import simulate_wfs
from optimize.strategies import (LUTSimulation, LUTProbabilisticSimulation,
                                  ProbabilisticLossStrategy)
from optimize.dataio import TracksDataset, DataLoader
from optimize.ranges import ranges
import logging, optimize.strategies
logging.getLogger(optimize.strategies.__name__).setLevel(logging.WARNING)
"""))

cells.append(code("""\
# ── Configuration ─────────────────────────────────────────────────────────────
INPUT_FILE   = '/sdf/data/neutrino/cyifan/diffsim_input/true_through_muon_edep_10cm_vol1cm.h5'
LUT_FILE     = 'src/larndsim/detector_properties/response_44_v2a_full_tick.npz'
DET_PROPS    = 'src/larndsim/detector_properties/module0.yaml'
PIXEL_LAYOUTS= 'src/larndsim/pixel_layouts/multi_tile_layout-2.4.16_v4.yaml'

RELEVANT_PARAMS = ['Ab','kb','lifetime','tran_diff','long_diff','eField','shift_x','shift_y','shift_z']
ELECTRON_SAMPLING_RESOLUTION = 0.01
NUMBER_PIX_NEIGHBORS = 4
SIGNAL_LENGTH = 150

ParamsClass = build_params_class(RELEVANT_PARAMS)
ref_params  = load_detector_properties(ParamsClass, DET_PROPS, PIXEL_LAYOUTS)
ref_params  = ref_params.replace(
    electron_sampling_resolution=ELECTRON_SAMPLING_RESOLUTION,
    number_pix_neighbors=NUMBER_PIX_NEIGHBORS,
    signal_length=SIGNAL_LENGTH,
    time_window=SIGNAL_LENGTH,
)
try:
    response, ref_params = load_lut(LUT_FILE, ref_params)
    print("LUT loaded.")
except Exception:
    response, ref_params = load_lut('src/larndsim/detector_properties/response_44.npy', ref_params)
    print("Fallback LUT loaded.")

initial_values  = {p: ranges[p]['nom'] for p in RELEVANT_PARAMS}
current_params  = ref_params.replace(**initial_values)
print("Parameters initialised.")
"""))

cells.append(code("""\
# ── Data ──────────────────────────────────────────────────────────────────────
dataset    = TracksDataset(filename=INPUT_FILE, nevents=-1,
                           electron_sampling_resolution=ELECTRON_SAMPLING_RESOLUTION)
dataloader = DataLoader(dataset, batch_size=-1)
tracks_batch = dataloader[0]
track_fields = dataset.get_track_fields()
tracks = jax.device_put(tracks_batch.reshape(-1, len(track_fields)))
print(f"Loaded {tracks.shape[0]} track segments.")

stoch_strategy = LUTSimulation(response)
prob_strategy  = LUTProbabilisticSimulation(response)
"""))

# ── Section 1: Waveforms ──────────────────────────────────────────────────────
cells.append(md("---\n## Section 1 — Waveform Visualisation"))

cells.append(code("""\
# Compute waveforms (no hit-finding yet)
wfs, unique_pixels = simulate_wfs(current_params, response, tracks, track_fields)
valid_mask = unique_pixels >= 0
valid_pixels = unique_pixels[valid_mask]
valid_wfs    = wfs[valid_mask]
print(f"Waveforms: {wfs.shape}  |  Valid pixels: {valid_mask.sum()}")
"""))

cells.append(code("""\
# ── Plot 1: Single pixel waveform ─────────────────────────────────────────────
PIXEL_IDX = 2          # change to inspect different pixels
Nticks = valid_wfs.shape[1]
ticks  = np.arange(Nticks) * float(current_params.t_sampling) * 1e6   # µs

fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(ticks, np.array(valid_wfs[PIXEL_IDX]), lw=1.5, label=f'Pixel {int(valid_pixels[PIXEL_IDX])}')
ax.axhline(float(current_params.DISCRIMINATION_THRESHOLD),
           ls='--', color='red', lw=1, label='Threshold')
ax.set_xlabel('Time [µs]'); ax.set_ylabel('Induced charge [e⁻/tick]')
ax.set_title('Induced-charge waveform — single pixel')
ax.legend(); plt.tight_layout(); plt.show()

# ── Plot 1b: Zoom around peak ────────────────────────────────────────────────
peak_tick = np.argmax(np.array(valid_wfs[PIXEL_IDX]))
zoom_width = 10 / (float(current_params.t_sampling) * 1e6) # ~10µs window
t_start = max(0, int(peak_tick - zoom_width/2))
t_end   = min(Nticks, int(peak_tick + zoom_width/2))

fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(ticks[t_start:t_end], np.array(valid_wfs[PIXEL_IDX][t_start:t_end]), marker='.', lw=1.5)
ax.axhline(float(current_params.DISCRIMINATION_THRESHOLD), ls='--', color='red', lw=1)
ax.set_xlabel('Time [µs]'); ax.set_ylabel('Induced charge [e⁻/tick]')
ax.set_title(f'Zoom around peak — Pixel {int(valid_pixels[PIXEL_IDX])}')
plt.tight_layout(); plt.show()
"""))

cells.append(code("""\
# ── Plot 2: Waveform gallery (N most-active pixels) ───────────────────────────
N_GALLERY = 16
peak_charge = np.array(valid_wfs).max(axis=1)
top_idx     = np.argsort(peak_charge)[::-1][:N_GALLERY]

ncols, nrows = 4, int(np.ceil(N_GALLERY / 4))
fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3*nrows), sharex=True)
axes = axes.flat
for ax_i, pix_i in enumerate(top_idx):
    axes[ax_i].plot(ticks, np.array(valid_wfs[pix_i]), lw=1)
    axes[ax_i].axhline(float(current_params.DISCRIMINATION_THRESHOLD),
                       ls='--', color='red', lw=0.8)
    axes[ax_i].set_title(f'pix {int(valid_pixels[pix_i])}', fontsize=8)
for ax_i in range(len(top_idx), len(axes)): axes[ax_i].set_visible(False)
fig.supxlabel('Time [µs]'); fig.supylabel('Induced charge [e⁻/tick]')
fig.suptitle('Top-N pixel waveforms'); plt.tight_layout(); plt.show()
"""))

cells.append(code("""\
# ── Plot 3: Summed waveform ───────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(ticks, np.array(valid_wfs).sum(axis=0), lw=1.5, color='steelblue')
ax.set_xlabel('Time [µs]'); ax.set_ylabel('Summed induced charge [e⁻/tick]')
ax.set_title('Sum of all pixel waveforms')
plt.tight_layout(); plt.show()
"""))

# ── Section 2: Stochastic vs Probabilistic ────────────────────────────────────
cells.append(md("---\n## Section 2 — Stochastic vs Probabilistic Validation"))

cells.append(code("""\
# Run probabilistic simulation
print("Running probabilistic simulation...")
prob_pred = prob_strategy.predict(current_params, tracks, track_fields, 42)
prob_unique_pixels = prob_pred['unique_pixels']
prob_wfs           = prob_pred['wfs']           # (Npix, Nticks)
hit_prob_log       = prob_pred['hit_prob']      # (Npix, Nhits, Nticks) – log-probs
adcs_distrib       = prob_pred['adcs_distrib']  # (Npix, Nhits, Nticks)
Npix_prob, Nhits, Nticks_prob = hit_prob_log.shape
print(f"Probabilistic: {Npix_prob} pixels, {Nhits} hit slots, {Nticks_prob} ticks")
"""))

cells.append(code("""\
# Run stochastic simulation (all hits, vmapped chunks)
N_STOCH   = 10_000
CHUNK_SZ  = 200
print(f"Running {N_STOCH} stochastic draws in chunks of {CHUNK_SZ}...")

@jax.jit
def run_chunk_all_hits(keys_chunk):
    vmapped = jax.vmap(get_adc_values, in_axes=(None, None, 0))
    integrals, ticks = vmapped(current_params, prob_wfs, keys_chunk)
    adcs = digitize(current_params, integrals)
    return ticks, adcs  # (CHUNK_SZ, Npix, MAX_ADC_VALUES)

keys = jax.random.split(jax.random.PRNGKey(0), N_STOCH)
all_ticks_list, all_adcs_list = [], []
for i in tqdm(range(0, N_STOCH, CHUNK_SZ)):
    chunk_keys = keys[i:i+CHUNK_SZ]
    t, a = run_chunk_all_hits(chunk_keys)
    all_ticks_list.append(t)
    all_adcs_list.append(a)

stoch_ticks = jnp.concatenate(all_ticks_list, axis=0)  # (N_STOCH, Npix, MAX_ADC_VALUES)
stoch_adcs  = jnp.concatenate(all_adcs_list,  axis=0)
print(f"Stochastic draws shape: ticks={stoch_ticks.shape}, adcs={stoch_adcs.shape}")
"""))

cells.append(code("""\
# ── Per-pixel, per-hit comparison ─────────────────────────────────────────────
TARGET_PIX_IDX = 2   # index into prob_unique_pixels; change to inspect others
VALID_TICK_BOUND = Nticks_prob - 3

fig, axes = plt.subplots(Nhits, 2, figsize=(14, 4*Nhits))
if Nhits == 1: axes = axes[np.newaxis, :]

for hit_i in range(Nhits):
    # Stochastic ticks & adcs for this hit slot
    s_ticks = np.array(stoch_ticks[:, TARGET_PIX_IDX, hit_i])
    s_adcs  = np.array(stoch_adcs[:,  TARGET_PIX_IDX, hit_i])
    valid   = s_ticks < VALID_TICK_BOUND
    s_ticks_v = s_ticks[valid]
    s_adcs_v  = s_adcs[valid]

    # Probabilistic distributions
    p_log  = np.array(hit_prob_log[TARGET_PIX_IDX, hit_i, :])
    p_prob = np.exp(p_log)
    p_adc  = np.array(adcs_distrib[TARGET_PIX_IDX, hit_i, :])

    # ── Tick distribution ──────────────────────────────────────────────────
    ax = axes[hit_i, 0]
    if len(s_ticks_v) > 0:
        ax.hist(s_ticks_v, bins=50, density=True, alpha=0.5, label='Stochastic')
    ax.plot(np.arange(Nticks_prob), p_prob / max(p_prob.sum(), 1e-30),
            'r-', lw=1.5, label='Probabilistic')
    hit_frac = valid.mean()
    lam      = p_prob.sum()
    ax.set_title(f'Hit {hit_i} — tick dist | frac={hit_frac:.3f} λ={lam:.3f}')
    ax.set_xlabel('Tick'); ax.legend()

    # ── Charge distribution ────────────────────────────────────────────────
    ax = axes[hit_i, 1]
    s_charge = np.array(adc2charge(s_adcs_v, current_params)) if len(s_adcs_v) else np.array([])
    p_charge = np.array(adc2charge(p_adc, current_params))
    if len(s_charge) > 0:
        ax.hist(s_charge, bins=30, density=True, alpha=0.5, label='Stochastic')
    ax.plot(p_charge, p_prob / max(p_prob.sum(), 1e-30),
            'r-', lw=1.5, label='Probabilistic')
    ax.set_title(f'Hit {hit_i} — charge dist')
    ax.set_xlabel('Charge [ke⁻]'); ax.legend()

plt.suptitle(f'Pixel index {TARGET_PIX_IDX} (id={int(prob_unique_pixels[TARGET_PIX_IDX])})',
             fontsize=13)
plt.tight_layout(); plt.show()
"""))

cells.append(code("""\
# ── Aggregate validation metrics ──────────────────────────────────────────────
print(f"{'Pix':>5} {'Hit':>4} {'KS-tick':>9} {'|Δλ|':>8} {'λ_prob':>8} {'λ_stoch':>8}")
rows = []
for pix_i in range(min(Npix_prob, 20)):          # cap at 20 pixels for display
    pid = int(prob_unique_pixels[pix_i])
    if pid < 0: continue
    for hit_i in range(Nhits):
        s_t   = np.array(stoch_ticks[:, pix_i, hit_i])
        valid = s_t < (Nticks_prob - 3)
        lam_s = valid.mean()
        p_log_slice = np.array(hit_prob_log[pix_i, hit_i, :])
        p_prob_slice = np.exp(p_log_slice)
        lam_p = p_prob_slice.sum()
        ks    = np.nan
        if valid.sum() > 20:
            hist_s, _ = np.histogram(s_t[valid], bins=Nticks_prob, range=(0, Nticks_prob), density=True)
            ks = stats.ks_2samp(hist_s, p_prob_slice / max(lam_p, 1e-30)).statistic
        rows.append((pid, hit_i, ks, abs(lam_p - lam_s), lam_p, lam_s))
        print(f"{pid:>5} {hit_i:>4} {ks:>9.4f} {abs(lam_p-lam_s):>8.4f} {lam_p:>8.4f} {lam_s:>8.4f}")
"""))

# ── Section 3: Loss Scan ──────────────────────────────────────────────────────
cells.append(md("---\n## Section 3 — Loss Scanning\n\nFour loss modes averaged over 500 stochastic draws."))

cells.append(code("""\
from larndsim.detsim_jax import id2pixel, get_hit_z
from optimize.strategies import compute_occurrence_indices

def build_target_from_draw(ticks_draw, adcs_draw, prob_unique_pixels, current_params, Nticks):
    \"\"\"
    Convert a single stochastic draw (Npix, MAX_ADC_VALUES) into a flat target dict
    compatible with ProbabilisticLossStrategy.compute().
    \"\"\"
    Npix, Nhits_max = ticks_draw.shape
    _, _, pixel_plane, _ = id2pixel(current_params, prob_unique_pixels)

    hit_pixels_list, ticks_list, adcs_list, z_list = [], [], [], []
    for hit_i in range(Nhits_max):
        t  = ticks_draw[:, hit_i]
        a  = adcs_draw[:, hit_i]
        valid = (t < (Nticks - 3)) & (prob_unique_pixels >= 0)
        if valid.sum() == 0: continue
        pids = prob_unique_pixels[valid]
        planes = pixel_plane[valid]
        ti   = t[valid].astype(jnp.int32)
        z    = get_hit_z(current_params, ti, planes)
        hit_pixels_list.append(pids)
        ticks_list.append(ti)
        adcs_list.append(a[valid])
        z_list.append(z)

    if not hit_pixels_list:
        dummy = jnp.zeros(1, dtype=jnp.int32)
        return dict(hit_pixels=dummy-1, ticks=dummy, adcs=jnp.zeros(1),
                    pixel_z=jnp.zeros(1), pixel_x=dummy, pixel_y=dummy,
                    event=dummy, hit_prob=jnp.ones(1))

    hit_pixels = jnp.concatenate(hit_pixels_list)
    ticks_all  = jnp.concatenate(ticks_list)
    adcs_all   = jnp.concatenate(adcs_list)
    z_all      = jnp.concatenate(z_list)

    # Sort by pixel id for compute_occurrence_indices
    sort_idx    = jnp.argsort(hit_pixels)
    hit_pixels  = hit_pixels[sort_idx]
    ticks_all   = ticks_all[sort_idx]
    adcs_all    = adcs_all[sort_idx]
    z_all       = z_all[sort_idx]

    return dict(
        hit_pixels=hit_pixels,
        pixel_id=hit_pixels,
        ticks=ticks_all,
        adcs=adcs_all,
        pixel_z=z_all,
        pixel_x=jnp.zeros_like(ticks_all),   # not used by ProbabilisticLossStrategy
        pixel_y=jnp.zeros_like(ticks_all),
        event=jnp.zeros_like(ticks_all),
        hit_prob=jnp.ones_like(ticks_all, dtype=jnp.float32),
    )
"""))

cells.append(code("""\
# Define the four loss strategies
DEADTIME_TICKS = round(
    (3 * float(current_params.CLOCK_CYCLE) + float(current_params.ADC_HOLD_DELAY) * float(current_params.CLOCK_CYCLE))
    / float(current_params.t_sampling)
)
print(f"Deadtime ticks: {DEADTIME_TICKS}")

loss_strategies = {
    '1st hit only':          ProbabilisticLossStrategy(first_hit_only=True),
    'All hits':              ProbabilisticLossStrategy(first_hit_only=False),
    'All hits + deadtime':   ProbabilisticLossStrategy(first_hit_only=False,
                                                        apply_deadtime=True,
                                                        deadtime_ticks=DEADTIME_TICKS),
    'Smooth (Gaussian win)': ProbabilisticLossStrategy(first_hit_only=False,
                                                        time_window=2, sigma_time=1.0),
}
"""))

cells.append(code("""\
# Collect 500 stochastic draws (vmapped, chunked) and compute losses
N_LOSS_DRAWS = 500
LOSS_CHUNK   = 50
loss_keys    = jax.random.split(jax.random.PRNGKey(99), N_LOSS_DRAWS)

@jax.jit
def run_chunk_loss_draws(keys_chunk):
    vmapped = jax.vmap(get_adc_values, in_axes=(None, None, 0))
    integrals, ticks = vmapped(current_params, prob_wfs, keys_chunk)
    adcs = digitize(current_params, integrals)
    return ticks, adcs   # (LOSS_CHUNK, Npix, MAX_ADC_VALUES)

loss_records = {name: [] for name in loss_strategies}
aux_records  = {name: [] for name in loss_strategies}

print("Scanning losses over stochastic draws...")
for chunk_start in tqdm(range(0, N_LOSS_DRAWS, LOSS_CHUNK)):
    chunk_keys = loss_keys[chunk_start:chunk_start+LOSS_CHUNK]
    t_chunk, a_chunk = run_chunk_loss_draws(chunk_keys)
    # Python loop only at chunk granularity
    for draw_i in range(t_chunk.shape[0]):
        target = build_target_from_draw(
            t_chunk[draw_i], a_chunk[draw_i],
            prob_unique_pixels, current_params, Nticks_prob
        )
        for name, strategy in loss_strategies.items():
            nll, aux = strategy.compute(current_params, prob_pred, target)
            loss_records[name].append(float(nll))
            aux_records[name].append({k: float(v) for k, v in aux.items()})

print("Done.")
for name, vals in loss_records.items():
    arr = np.array(vals)
    print(f"  {name:30s}  mean={arr.mean():.4f}  std={arr.std():.4f}")
"""))

cells.append(code("""\
# ── Plot: NLL distribution per loss mode ─────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

names = list(loss_records.keys())
data  = [np.array(loss_records[n]) for n in names]

# Violin plot
vp = axes[0].violinplot(data, showmedians=True, showextrema=True)
axes[0].set_xticks(range(1, len(names)+1))
axes[0].set_xticklabels(names, rotation=20, ha='right', fontsize=9)
axes[0].set_ylabel('NLL'); axes[0].set_title('NLL distribution across 500 stochastic draws')

# Loss vs draw index
for name, arr in zip(names, data):
    running = np.cumsum(arr) / (np.arange(len(arr)) + 1)
    axes[1].plot(running, label=name, lw=1.5)
axes[1].set_xlabel('Draw index'); axes[1].set_ylabel('Cumulative mean NLL')
axes[1].set_title('Convergence of mean NLL'); axes[1].legend(fontsize=8)
plt.tight_layout(); plt.show()
"""))

cells.append(code("""\
# ── Plot: Aux term breakdown ──────────────────────────────────────────────────
aux_keys = list(aux_records[names[0]][0].keys())
x = np.arange(len(names))
width = 0.18
fig, ax = plt.subplots(figsize=(13, 5))
for k_i, key in enumerate(aux_keys):
    means = [np.mean([r[key] for r in aux_records[n]]) for n in names]
    ax.bar(x + k_i*width, means, width, label=key)
ax.set_xticks(x + width*(len(aux_keys)-1)/2)
ax.set_xticklabels(names, rotation=15, ha='right', fontsize=9)
ax.set_ylabel('Mean aux value'); ax.set_title('NLL term breakdown by loss mode')
ax.legend(fontsize=8); plt.tight_layout(); plt.show()
"""))

cells.append(code("""\
# ── Optional: gradient w.r.t. lifetime ────────────────────────────────────────
# Pick the first valid target draw for gradient check
target_for_grad = build_target_from_draw(
    stoch_ticks[0], stoch_adcs[0], prob_unique_pixels, current_params, Nticks_prob
)

def nll_vs_lifetime(lifetime_val, strategy):
    p = current_params.replace(lifetime=lifetime_val)
    # Re-run probabilistic simulation
    from larndsim.fee_jax import get_adc_values_average_noise_vmap
    log_prob, adc_d = get_adc_values_average_noise_vmap(p, prob_wfs)
    pred_local = dict(prob_pred, hit_prob=log_prob, adcs_distrib=adc_d)
    nll, _ = strategy.compute(p, pred_local, target_for_grad)
    return nll

print("Computing lifetime gradients (may compile ~30s)...")
nom_lt = float(current_params.lifetime)
for name, strategy in loss_strategies.items():
    grad_fn = jax.grad(nll_vs_lifetime)
    g = float(grad_fn(nom_lt, strategy))
    print(f"  {name:30s}  dNLL/d(lifetime) = {g:.6f}")
"""))

# ── Write notebook ────────────────────────────────────────────────────────────
nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.12"}
    },
    "cells": cells
}
out = '/sdf/group/neutrino/pgranger/larnd-sim-jax/debug_simulation_loss.ipynb'
with open(out, 'w') as f:
    json.dump(nb, f, indent=1)
print(f"Notebook written → {out}")
