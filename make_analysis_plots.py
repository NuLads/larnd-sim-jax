"""
make_analysis_plots.py
======================
Produces analysis figures comparing joint vs no_pos chain fitting.

Figures produced:
  plots/fig1_ab_overview.png      — Ab calibration overview (3 panels)
  plots/fig2_timing.png           — Timing comparison
  plots/fig3_active_tracks.png    — Convergence + Ab for active tracks
  plots/fig4_geometry_*.png       — Chain geometry for selected tracks
  plots/fig5_residuals.png        — Residual distributions
"""
import os, sys, pickle, types, dataclasses
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

os.chdir('/sdf/group/neutrino/pgranger/larnd-sim-jax')
sys.path.insert(0, 'src')
os.makedirs('plots', exist_ok=True)

AB_TRUE  = 0.8348813503927325
DEDX_NOM = 1.887

# ── Pickle helpers ────────────────────────────────────────────────────────────
@dataclasses.dataclass
class TrackContext:
    x0_fixed: object; step_len: float; total_len: float
    sigma_mcs: float; n_chain: int; theta0_i: float; phi0_i: float
    dtheta_i: object; dphi_i: object; x0_chain: object
    bounds_chain: object; n_chain_params: int; target_pixels: object
    target_adc: object; fixed_pixels: object; psn: int; pln: int

@dataclasses.dataclass
class OptResult:
    x: object; fun: float

def _register(name):
    m = types.ModuleType(name)
    m.TrackContext = TrackContext; m.OptResult = OptResult
    sys.modules[name] = m

_register('run_chain_fit'); _register('__main__')

def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

# ── Load data ─────────────────────────────────────────────────────────────────
print('Loading CSVs...')
df_j  = pd.read_csv('chain_fit_summary_joint_b6.csv')
df_np = pd.read_csv('chain_fit_summary_no_pos_b6.csv')

for df in (df_j, df_np):
    for col in ['Ab','dEdx_S2','Ri','Rs3','L','t_s']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

merged = pd.merge(df_j.rename(columns={c:c+'_j' for c in df_j.columns if c not in ('b','ev','tr')}),
                  df_np.rename(columns={c:c+'_np' for c in df_np.columns if c not in ('b','ev','tr')}),
                  on=['b','ev','tr'])

merged['Ab_bias_j']  = merged['Ab_j']  - AB_TRUE
merged['Ab_bias_np'] = merged['Ab_np'] - AB_TRUE
merged['delta_resid'] = merged['Ri_j'] - merged['Rs3_j']   # pos improvement (joint)
merged['label'] = [f"b{int(r.b)}e{int(r.ev)}t{int(r.tr)}" for _,r in merged.iterrows()]
active = merged[np.abs(merged['Ab_j'] - AB_TRUE) > 0.001].copy()

print(f'  {len(merged)} matched tracks, {len(active)} with Ab deviation')

print('Loading pickles...')
ckpt_j  = load_pkl('chain_fit_joint_b6.pkl')
ckpt_np = load_pkl('chain_fit_no_pos_b6.pkl')

# ── Figure 1: Ab overview ─────────────────────────────────────────────────────
print('Figure 1: Ab overview...')
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Ab calibration: joint vs no-pos fitting  (30 tracks)', fontsize=13, fontweight='bold')

# Panel 1a: scatter joint vs no_pos Ab
ax = axes[0]
sc = ax.scatter(merged['Ab_np'], merged['Ab_j'],
                c=merged['delta_resid'], cmap='RdYlGn', vmin=-0.1, vmax=0.1,
                s=80, edgecolors='k', linewidths=0.5, zorder=3)
lim = [min(merged['Ab_j'].min(), merged['Ab_np'].min()) - 0.02,
       max(merged['Ab_j'].max(), merged['Ab_np'].max()) + 0.02]
ax.plot(lim, lim, 'k--', lw=1, label='no change')
ax.axhline(AB_TRUE, color='blue', lw=1, ls=':', alpha=0.7, label=f'Ab_true={AB_TRUE:.4f}')
ax.axvline(AB_TRUE, color='blue', lw=1, ls=':', alpha=0.7)
plt.colorbar(sc, ax=ax, label='Δresid (joint, cm)', shrink=0.8)
for _, row in active.iterrows():
    ax.annotate(row['label'], (row['Ab_np'], row['Ab_j']),
                fontsize=7, xytext=(4, 3), textcoords='offset points')
ax.set_xlabel('Ab  [no_pos]'); ax.set_ylabel('Ab  [joint]')
ax.set_title('Ab fitted: joint vs no-pos\n(color = position improvement)'); ax.legend(fontsize=8)
ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect('equal'); ax.grid(True, alpha=0.3)

# Panel 1b: Ab bias histogram
ax = axes[1]
bins = np.linspace(-0.20, 0.20, 40)
ax.hist(merged['Ab_bias_j'],  bins=bins, alpha=0.6, color='steelblue',  label=f'joint  (σ={merged["Ab_bias_j"].std():.4f})')
ax.hist(merged['Ab_bias_np'], bins=bins, alpha=0.6, color='darkorange', label=f'no_pos (σ={merged["Ab_bias_np"].std():.4f})')
ax.axvline(0, color='black', lw=1.5, ls='--', label='unbiased')
ax.set_xlabel('Ab − Ab_true'); ax.set_ylabel('Tracks')
ax.set_title('Ab bias distribution\n(all 30 tracks)')
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

# Panel 1c: Ab bias vs initial residual Ri
ax = axes[2]
for row in merged.itertuples():
    kw = dict(s=max(30, row.L_j*0.8), edgecolors='k', linewidths=0.4, zorder=3)
    ax.scatter(row.Ri_j, abs(row.Ab_bias_j),  marker='o', color='steelblue',  **kw)
    ax.scatter(row.Ri_j, abs(row.Ab_bias_np), marker='^', color='darkorange', **kw)
    if abs(row.Ab_bias_j) > 0.001 or abs(row.Ab_bias_np) > 0.001:
        ax.annotate(row.label, (row.Ri_j, max(abs(row.Ab_bias_j), abs(row.Ab_bias_np))),
                    fontsize=7, xytext=(3, 3), textcoords='offset points')
legend_elems = [Line2D([0],[0], marker='o', color='w', markerfacecolor='steelblue',
                        markeredgecolor='k', markersize=8, label='joint'),
                Line2D([0],[0], marker='^', color='w', markerfacecolor='darkorange',
                        markeredgecolor='k', markersize=8, label='no_pos')]
ax.set_xlabel('Initial residual Ri [cm]'); ax.set_ylabel('|Ab − Ab_true|')
ax.set_title('Ab bias vs initial position residual\n(marker size ∝ track length)')
ax.legend(handles=legend_elems, fontsize=9); ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig('plots/fig1_ab_overview.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print('  Saved plots/fig1_ab_overview.png')

# ── Figure 2: Timing ──────────────────────────────────────────────────────────
print('Figure 2: Timing...')
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Fitting time: joint vs no-pos', fontsize=13, fontweight='bold')

ax = axes[0]
is_active = np.abs(merged['Ab_j'] - AB_TRUE) > 0.001
colors_j  = np.where(is_active, 'steelblue', 'lightsteelblue')
colors_np = np.where(is_active, 'darkorange', 'moccasin')
x = np.arange(len(merged))
w = 0.38
bars_j  = ax.bar(x - w/2, merged['t_s_j'],  w, color=colors_j,  label='joint', edgecolor='k', linewidth=0.3)
bars_np = ax.bar(x + w/2, merged['t_s_np'], w, color=colors_np, label='no_pos', edgecolor='k', linewidth=0.3)
ax.set_xticks(x[::3]); ax.set_xticklabels(merged['label'].iloc[::3], rotation=45, ha='right', fontsize=7)
ax.set_ylabel('Wall time [s]'); ax.set_title('Time per track')
legend_elems = [Line2D([0],[0], color='steelblue',     lw=8, label='joint (active)'),
                Line2D([0],[0], color='lightsteelblue', lw=8, label='joint (stuck)'),
                Line2D([0],[0], color='darkorange',     lw=8, label='no_pos (active)'),
                Line2D([0],[0], color='moccasin',       lw=8, label='no_pos (stuck)')]
ax.legend(handles=legend_elems, fontsize=8); ax.grid(True, alpha=0.3, axis='y')

ax = axes[1]
ax.scatter(merged['L_j'], merged['t_s_j'],  s=60, color='steelblue',  label=f'joint  (total={merged["t_s_j"].sum()/3600:.1f}h)',  edgecolors='k', linewidths=0.5)
ax.scatter(merged['L_j'], merged['t_s_np'], s=60, color='darkorange', label=f'no_pos (total={merged["t_s_np"].sum()/3600:.1f}h)', edgecolors='k', linewidths=0.5, marker='^')
for _, row in active.iterrows():
    ax.annotate(row['label'], (row['L_j'], max(row['t_s_j'], row['t_s_np'])),
                fontsize=7, xytext=(3, 3), textcoords='offset points')
ax.set_xlabel('Track length [cm]'); ax.set_ylabel('Wall time [s]')
ax.set_title('Time vs track length'); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig('plots/fig2_timing.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print('  Saved plots/fig2_timing.png')

# ── Figure 3: Active tracks convergence ───────────────────────────────────────
print('Figure 3: Active tracks convergence...')
active_keys = [(int(r.b), int(r.ev), int(r.tr)) for _, r in active.iterrows()]
n_act = len(active_keys)
ncols = 3
nrows = int(np.ceil(n_act / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*5, nrows*4))
axes = np.array(axes).flatten()
fig.suptitle('Convergence history — active tracks (|Ab − Ab_true| > 0.001)', fontsize=13, fontweight='bold')

for i, key in enumerate(active_keys):
    ax = axes[i]
    label = f"b{key[0]}e{key[1]}t{key[2]}"
    row = merged[merged['label']==label].iloc[0]

    # Joint convergence
    if key in ckpt_j:
        h3 = ckpt_j[key]['hist_s3']
        losses = [h['loss'] for h in h3]
        gnorms = [h['grad_norm'] for h in h3]
        steps  = np.arange(len(losses))
        ax.plot(steps, losses, color='steelblue', lw=1.5, label=f'joint loss (final={losses[-1]:.3e})')
        ax2 = ax.twinx()
        ax2.plot(steps, gnorms, color='steelblue', lw=0.8, ls='--', alpha=0.5)
        ax2.set_yscale('log'); ax2.set_ylabel('‖∇‖', color='gray', fontsize=8)
        try: ax.set_yscale('log')
        except: pass

    # No-pos convergence (S2+S3 combined)
    if key in ckpt_np:
        h2 = ckpt_np[key]['hist_s2']; h3 = ckpt_np[key]['hist_s3']
        losses2 = [h['loss'] for h in h2 if np.isfinite(h['loss'])]
        losses3 = [h['loss'] for h in h3]
        combined = losses2 + losses3
        if combined:
            ax.plot(np.arange(len(combined)), combined, color='darkorange', lw=1.5,
                    label=f'no_pos loss (final={combined[-1]:.3e})')

    ax.axhline(0, color='gray', lw=0.5, ls=':')
    ax.set_title(f'{label}  L={row["L_j"]:.0f}cm  Ri={row["Ri_j"]:.3f}cm\n'
                 f'Ab_j={row["Ab_j"]:.4f}  Ab_np={row["Ab_np"]:.4f}  Ab_true={AB_TRUE:.4f}')
    ax.set_xlabel('Step'); ax.set_ylabel('Loss')
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

for j in range(i+1, len(axes)):
    axes[j].set_visible(False)

plt.tight_layout()
fig.savefig('plots/fig3_active_convergence.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print('  Saved plots/fig3_active_convergence.png')

# ── Figure 4: Chain geometry for top active tracks ────────────────────────────
print('Figure 4: Chain geometry...')
# Pick tracks with largest position improvement or largest Ab deviation
top_keys = sorted(active_keys, key=lambda k: -abs(
    merged[merged['label']==f"b{k[0]}e{k[1]}t{k[2]}"]['Ab_bias_j'].values[0]))[:6]

for key in top_keys:
    label = f"b{key[0]}e{key[1]}t{key[2]}"
    if key not in ckpt_j:
        continue
    res_j  = ckpt_j[key]
    ctx    = res_j['ctx']
    cn_j   = res_j['chain_nodes']   # dict: init, s1, s2, s3
    row    = merged[merged['label']==label].iloc[0]

    # Try to get no_pos chain nodes too
    cn_np = ckpt_np[key]['chain_nodes'] if key in ckpt_np else None

    # Get target segment positions from the pkl
    tgt_adc = np.array(ctx.target_adc)   # (n_tgt_px, n_ticks)
    # We don't have 3D target points here directly, use residuals from result
    # Instead, show chain geometry in XZ and YZ
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ab_np_str = f'{row["Ab_np"]:.4f}' if cn_np is not None else 'n/a'
    fig.suptitle(f'Chain geometry — {label}  L={ctx.total_len:.1f}cm  '
                 f'Ab_j={row["Ab_j"]:.4f}  Ab_np={ab_np_str}  Ab_true={AB_TRUE:.4f}',
                 fontsize=11, fontweight='bold')

    for ax, (da, db, xl, yl, ttl) in zip(axes, [
        (2, 0, 'z [cm]', 'x [cm]', 'ZX view'),
        (2, 1, 'z [cm]', 'y [cm]', 'ZY view'),
    ]):
        init = np.array(cn_j['init'])
        fit_j = np.array(cn_j['s3'])
        ax.plot(init[:,da], init[:,db], 'gray', lw=1, ls='--', label='Initial (straight)', zorder=2)
        ax.plot(fit_j[:,da], fit_j[:,db], 'steelblue', lw=2, marker='o', ms=3, label='Joint fitted', zorder=4)
        if cn_np is not None:
            fit_np = np.array(cn_np['s3'])
            ax.plot(fit_np[:,da], fit_np[:,db], 'darkorange', lw=2, ls='-.', marker='^', ms=3,
                    label='No-pos (unchanged)', zorder=3)
        ax.scatter([float(ctx.x0_fixed[da])], [float(ctx.x0_fixed[db])],
                   marker='*', s=200, c='gold', zorder=6, label='Entry')
        ax.set_xlabel(xl); ax.set_ylabel(yl); ax.set_title(ttl)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3); ax.set_aspect('equal')

    plt.tight_layout()
    fname = f'plots/fig4_geometry_{label}.png'
    fig.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {fname}')

# ── Figure 5: Residual distributions ─────────────────────────────────────────
print('Figure 5: Residuals...')
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Track-to-true residuals (all 30 tracks)', fontsize=13, fontweight='bold')

ax = axes[0]
ax.scatter(merged['Ri_j'], merged['Rs3_j'],  s=60, color='steelblue',  marker='o',
           edgecolors='k', linewidths=0.5, label='joint',  zorder=3)
ax.scatter(merged['Ri_j'], merged['Rs3_np'], s=60, color='darkorange', marker='^',
           edgecolors='k', linewidths=0.5, label='no_pos', zorder=3)
lim2 = [0, max(merged['Ri_j'].max(), merged['Rs3_j'].max(), merged['Rs3_np'].max()) + 0.02]
ax.plot(lim2, lim2, 'k--', lw=1, label='no improvement')
ax.set_xlabel('Initial residual Ri [cm]'); ax.set_ylabel('Final residual Rs3 [cm]')
ax.set_title('Initial vs final residual'); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

ax = axes[1]
bins = np.linspace(0, 0.25, 40)
ax.hist(merged['Rs3_j'],  bins=bins, alpha=0.6, color='steelblue',
        label=f'joint   mean={merged["Rs3_j"].mean():.4f} cm')
ax.hist(merged['Rs3_np'], bins=bins, alpha=0.6, color='darkorange',
        label=f'no_pos  mean={merged["Rs3_np"].mean():.4f} cm')
ax.hist(merged['Ri_j'],   bins=bins, alpha=0.4, color='gray',
        label=f'initial mean={merged["Ri_j"].mean():.4f} cm', histtype='step', lw=2, ls='--')
ax.set_xlabel('Residual [cm]'); ax.set_ylabel('Tracks')
ax.set_title('Residual distribution'); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig('plots/fig5_residuals.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print('  Saved plots/fig5_residuals.png')

# ── Figure 6: dEdx profiles for active tracks ─────────────────────────────────
print('Figure 6: dEdx profiles...')
n_act = len(active_keys)
ncols = 3
nrows = int(np.ceil(n_act / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*5, nrows*4))
axes = np.array(axes).flatten()
fig.suptitle('Fitted dEdx profiles — active tracks', fontsize=13, fontweight='bold')

for i, key in enumerate(active_keys):
    ax = axes[i]
    label = f"b{key[0]}e{key[1]}t{key[2]}"
    row = merged[merged['label']==label].iloc[0]

    for mode, ckpt, col, mk in [('joint', ckpt_j, 'steelblue', 'o'), ('no_pos', ckpt_np, 'darkorange', '^')]:
        if key not in ckpt: continue
        r = ckpt[key]
        ctx = r['ctx']; nc = ctx.n_chain
        r3  = r['res_s3']
        if len(r3.x) > ctx.n_chain_params + nc:
            ld = r3.x[ctx.n_chain_params:ctx.n_chain_params+nc]
        elif len(r3.x) > ctx.n_chain_params:
            ld = r3.x[ctx.n_chain_params:ctx.n_chain_params+nc]
        else:
            continue
        arc = np.arange(nc)*ctx.step_len + ctx.step_len/2
        ax.plot(arc, np.exp(ld), color=col, lw=1.8, marker=mk, ms=4,
                label=f'{mode} (Ab={row["Ab_j" if mode=="joint" else "Ab_np"]:.4f})')

    ax.axhline(DEDX_NOM, color='gray', ls='--', lw=1.2, label=f'nominal {DEDX_NOM}')
    ax.axhline(AB_TRUE,  color='black', ls=':', lw=1, alpha=0.5)
    ax.set_xlabel('Arc length [cm]'); ax.set_ylabel('dE/dx [MeV/cm]')
    ax.set_title(f'{label}  L={row["L_j"]:.0f}cm')
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

for j in range(i+1, len(axes)):
    axes[j].set_visible(False)

plt.tight_layout()
fig.savefig('plots/fig6_dedx_profiles.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print('  Saved plots/fig6_dedx_profiles.png')

print('\nAll figures done.')
