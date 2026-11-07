"""
plot_dedx_distribution.py
Compares true (raw h5), linear-guess input, and fitted dEdx distributions.
"""
import os, sys, pickle
os.environ['JAX_PLATFORMS'] = 'cpu'
sys.path.insert(0, '/sdf/group/neutrino/pgranger/larnd-sim-jax')
sys.path.insert(0, '/sdf/group/neutrino/pgranger/larnd-sim-jax/src')
os.chdir('/sdf/group/neutrino/pgranger/larnd-sim-jax')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import h5py
from optimize.dataio import TracksDataset, TgtTracksDataset, DataLoader

CKPT = 'fit_result/chain_geom_test/history_iter100_chain_geom_test_100iter_b50.pkl'
SIM  = '/sdf/group/neutrino/pgranger/lads-data/linear_guess_segments.h5'
TGT  = '/sdf/data/neutrino/cyifan/diffsim_input/true_through_muon_edep_10cm_vol1cm.h5'
os.makedirs('plots', exist_ok=True)

with open(CKPT, 'rb') as f:
    h = pickle.load(f)
cfg = h['config']

# ── 1. Raw dEdx from h5 files (all muon segments, no batch selection) ─────────
def load_raw_dedx(path, max_n=500_000):
    with h5py.File(path, 'r') as f:
        segs = f['segments'][:]
    mask = (np.abs(segs['pdg_id']) == 13) & (segs['dx'] > 0) \
         & (segs['dEdx'] > 0.2) & (segs['dEdx'] < 20.)
    d = segs['dEdx'][mask]
    rng = np.random.default_rng(42)
    if len(d) > max_n:
        d = rng.choice(d, max_n, replace=False)
    return d

raw_sim = load_raw_dedx(SIM)
raw_tgt = load_raw_dedx(TGT)

# ── 2. Fitted dEdx from the checkpoint (per original segment, both batches) ──
dedx_cache = h['dedx_cache']
fitted_all = np.concatenate([np.exp(np.array(v['log_dedx'])) for v in dedx_cache.values()])

# ── 3. Batch-level sim dEdx (same batches as the fit) ────────────────────────
ds_sim = TracksDataset(
    filename=cfg.input_file_sim, nevents=-1, max_nbatch=cfg.max_nbatch,
    track_len_sel=cfg.track_len_sel, max_abs_costheta_sel=cfg.max_abs_costheta_sel,
    min_abs_segz_sel=cfg.min_abs_segz_sel, track_z_bound=cfg.track_z_bound,
    max_batch_len=cfg.max_batch_len,
    electron_sampling_resolution=cfg.electron_sampling_resolution,
    chopped=True, pad=True)
ds_tgt = TgtTracksDataset(
    filename=cfg.input_file_tgt, dataset_sim=ds_sim,
    electron_sampling_resolution=cfg.electron_sampling_resolution,
    chopped=True, pad=True)
dl_sim = DataLoader(ds_sim, batch_size=1)
dl_tgt = DataLoader(ds_tgt, batch_size=1)
sf = ds_sim.get_track_fields()
tf = ds_tgt.get_track_fields()
di_s, dx_s = sf.index('dEdx'), sf.index('dx')
di_t, dx_t = tf.index('dEdx'), tf.index('dx')

batch_sim_dedx, batch_tgt_dedx = [], []
for bi in range(len(dl_sim)):
    s = np.asarray(dl_sim[bi]).reshape(-1, len(sf))
    valid = s[:, dx_s] > 0
    batch_sim_dedx.append(s[valid, di_s])
    t = np.asarray(dl_tgt[bi]).reshape(-1, len(tf))
    valid_t = t[:, dx_t] > 0
    batch_tgt_dedx.append(t[valid_t, di_t])
batch_sim_dedx = np.concatenate(batch_sim_dedx)
batch_tgt_dedx = np.concatenate(batch_tgt_dedx)

print(f"raw_tgt:       n={len(raw_tgt):6d}  mean={raw_tgt.mean():.4f}  std={raw_tgt.std():.4f}")
print(f"raw_sim:       n={len(raw_sim):6d}  mean={raw_sim.mean():.4f}  std={raw_sim.std():.4f}")
print(f"batch_tgt:     n={len(batch_tgt_dedx):6d}  mean={batch_tgt_dedx.mean():.4f}  std={batch_tgt_dedx.std():.4f}")
print(f"batch_sim:     n={len(batch_sim_dedx):6d}  mean={batch_sim_dedx.mean():.4f}  std={batch_sim_dedx.std():.4f}")
print(f"fitted_dedx:   n={len(fitted_all):6d}  mean={fitted_all.mean():.4f}  std={fitted_all.std():.4f}")

# ── Plotting ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Per-segment dEdx distributions — chain geometry fit demo', fontsize=12, fontweight='bold')

bins_raw   = np.linspace(0.5, 6, 80)
bins_batch = np.linspace(1.5, 2.5, 60)

# Left: full raw h5 distributions
ax = axes[0]
ax.hist(raw_tgt, bins=bins_raw, density=True, alpha=0.6, color='crimson',
        label=f'True target (raw h5)\nn={len(raw_tgt):,}  μ={raw_tgt.mean():.3f}  σ={raw_tgt.std():.3f}')
ax.hist(raw_sim, bins=bins_raw, density=True, alpha=0.6, color='steelblue',
        label=f'Linear-guess input (raw h5)\nn={len(raw_sim):,}  μ={raw_sim.mean():.3f}  σ={raw_sim.std():.3f}')
ax.axvline(raw_tgt.mean(), color='crimson',   lw=2, ls='--')
ax.axvline(raw_sim.mean(), color='steelblue', lw=2, ls='--')
ax.set_xlabel('dEdx  [MeV/cm]')
ax.set_ylabel('Density')
ax.set_title('Full dataset (all muon segments)')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_xlim(0.5, 6)

# Right: batch-level (what the fitter actually sees) + fitted result
ax = axes[1]
ax.hist(batch_tgt_dedx, bins=bins_batch, density=True, alpha=0.55, color='crimson',
        label=f'True target (batch fine-steps)\nn={len(batch_tgt_dedx):,}  μ={batch_tgt_dedx.mean():.3f}  σ={batch_tgt_dedx.std():.3f}')
ax.hist(batch_sim_dedx, bins=bins_batch, density=True, alpha=0.55, color='steelblue',
        label=f'Sim input (batch fine-steps)\nn={len(batch_sim_dedx):,}  μ={batch_sim_dedx.mean():.3f}  σ={batch_sim_dedx.std():.3f}')
ax.hist(fitted_all, bins=bins_batch, density=True, alpha=0.55, color='darkorange', histtype='step', lw=2,
        label=f'Fitted dEdx (per original segment)\nn={len(fitted_all):,}  μ={fitted_all.mean():.3f}  σ={fitted_all.std():.3f}')
ax.axvline(batch_tgt_dedx.mean(), color='crimson',   lw=1.5, ls='--')
ax.axvline(batch_sim_dedx.mean(), color='steelblue', lw=1.5, ls='--')
ax.axvline(fitted_all.mean(),     color='darkorange', lw=1.5, ls='--')
ax.set_xlabel('dEdx  [MeV/cm]')
ax.set_ylabel('Density')
ax.set_title('Batches used in fit (fine-steps vs fitted segments)')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
out = 'plots/dedx_distribution_comparison.png'
fig.savefig(out, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'\nSaved: {out}')
