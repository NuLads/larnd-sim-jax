"""
plot_joint_demo.py  —  post-process the joint Ab+dEdx+position fit checkpoint.
"""
import os, sys, pickle
os.environ['JAX_PLATFORMS'] = 'cpu'
sys.path.insert(0, '/sdf/group/neutrino/pgranger/larnd-sim-jax')
sys.path.insert(0, '/sdf/group/neutrino/pgranger/larnd-sim-jax/src')
os.chdir('/sdf/group/neutrino/pgranger/larnd-sim-jax')

import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import jax.numpy as jnp
from optimize.fit_params import _absolute_chain_forward

CKPT = 'fit_result/joint_demo/history_iter500_joint_500iter.pkl'
os.makedirs('plots', exist_ok=True)

with open(CKPT, 'rb') as f:
    h = pickle.load(f)

Ab_true    = float(np.array(h['Ab_target']))
Ab_init    = float(np.array(h['Ab_init']))
Ab_iter    = np.array(h['Ab_iter'])
losses     = np.array(h['losses_iter'])
pr_iter    = np.array([x for x in h.get('pos_residual_iter', []) if x is not None])
md_iter    = np.array(h.get('mean_dedx_iter', []))
chain_cache  = h['chain_cache']
chain_ctxs   = h['chain_contexts']
true_positions = h['true_positions']
true_dedx_cache = h.get('true_dedx_cache', {})
dedx_cache   = h.get('dedx_cache', {})

all_true_dedx = np.concatenate([np.array(v).ravel() for v in true_dedx_cache.values()]) \
    if true_dedx_cache else np.array([])


def interp_chain(ctx, angles, query_alphas):
    n_c = ctx['n_chain']
    nodes, _ = _absolute_chain_forward(
        jnp.array(ctx['x0_fixed']),
        jnp.array(angles[:n_c]), jnp.array(angles[n_c:]),
        ctx['step_len'])
    nodes = np.array(nodes)
    k = np.clip((np.asarray(query_alphas) * n_c).astype(int), 0, n_c - 1)
    f = np.asarray(query_alphas) * n_c - k
    return nodes[k] + f[:, None] * (nodes[k + 1] - nodes[k])


# ── collect per-track results ────────────────────────────────────────────────
results = []
for bi in sorted(chain_ctxs.keys()):
    tp_bi = true_positions.get(int(bi), {})
    dc_bi = dedx_cache.get(int(bi), {})
    for j, (ctx, state) in enumerate(zip(chain_ctxs[bi], chain_cache[bi])):
        true_pts = tp_bi.get(str(ctx['track_id']))
        if true_pts is None:
            continue
        true_pts = np.asarray(true_pts)
        qa = np.asarray(ctx['alpha_mid'])
        angles_fit  = np.array(state['angles'])
        n_c = ctx['n_chain']
        angles_init = np.empty_like(angles_fit)
        angles_init[:n_c] = ctx['theta0_i']
        angles_init[n_c:] = ctx['phi0_i']
        pos_fit  = interp_chain(ctx, angles_fit,  qa)
        pos_init = interp_chain(ctx, angles_init, qa)
        ri = np.linalg.norm(pos_init - true_pts, axis=1)
        rf = np.linalg.norm(pos_fit  - true_pts, axis=1)
        # dEdx is stored per-batch (not per-track); attach all batch dEdx to first track
        fitted_dedx = np.exp(np.array(dc_bi['log_dedx'])) if j == 0 and isinstance(dc_bi, dict) and 'log_dedx' in dc_bi else np.array([])
        results.append(dict(batch=bi, track=ctx['track_id'],
            n_chain=n_c, total_len=ctx['total_len'],
            res_init=ri, res_fit=rf,
            res_init_mean=float(ri.mean()), res_fit_mean=float(rf.mean()),
            query_alphas=qa, pos_init=pos_init, pos_fit=pos_fit, pos_true=true_pts,
            fitted_dedx=fitted_dedx,
            dtheta_rms=float(np.sqrt(np.mean(np.diff(angles_fit[:n_c])**2)))*1e3,
        ))
        print(f"B{bi} {ctx['track_id']}: init={ri.mean():.4f}  fit={rf.mean():.4f} cm  "
              f"imp={ri.mean()-rf.mean():.4f}  ({(ri.mean()-rf.mean())/max(ri.mean(),1e-6)*100:.0f}%)")


# ── Figure 1: 5-panel summary ─────────────────────────────────────────────────
fig, axes = plt.subplots(1, 5, figsize=(26, 5))
fig.suptitle(f'Joint Ab + dEdx + position fit  ({int(h["total_iter"])} iterations on A100)',
             fontsize=12, fontweight='bold')

# Loss
ax = axes[0]
ax.plot(losses, color='steelblue', lw=0.8, alpha=0.7)
ax.set_xlabel('Iteration'); ax.set_ylabel('Total loss')
ax.set_title('Loss'); ax.grid(True, alpha=0.3)

# Ab convergence
ax = axes[1]
ax.plot(Ab_iter, color='darkorange', lw=1.2)
ax.axhline(Ab_true, color='crimson', lw=1.5, ls='--', label=f'True={Ab_true:.4f}')
ax.axhline(Ab_init, color='gray',    lw=1.0, ls=':',  label=f'Init={Ab_init:.4f}')
ax.set_xlabel('Iteration'); ax.set_ylabel('Ab')
ax.set_title(f'Ab  (final={Ab_iter[-1]:.4f},  {(Ab_iter[-1]-Ab_init)/(Ab_true-Ab_init)*100:.0f}% closed)')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# Position residual
ax = axes[2]
ax.plot(pr_iter, color='purple', lw=0.8, alpha=0.8)
ax.set_xlabel('Training step'); ax.set_ylabel('Mean 3D residual [cm]')
ax.set_title(f'Position residual\n({pr_iter[:30].mean():.4f} → {pr_iter[-30:].mean():.4f} cm)')
ax.grid(True, alpha=0.3)

# mean dEdx
ax = axes[3]
if len(md_iter) > 0:
    ax.plot(md_iter, color='seagreen', lw=0.8, alpha=0.8)
    if len(all_true_dedx) > 0:
        ax.axhline(all_true_dedx.mean(), color='crimson', lw=1.5, ls='--',
                   label=f'True mean={all_true_dedx.mean():.4f}')
    ax.set_xlabel('Iteration'); ax.set_ylabel('Mean dEdx [MeV/cm]')
    ax.set_title(f'Mean dEdx  ({md_iter[:30].mean():.4f} → {md_iter[-30:].mean():.4f})')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# Per-track position bar
ax = axes[4]
n_t = len(results)
x = np.arange(n_t); w = 0.35
labels = [f"B{r['batch']}\n({r['track'][0]:.0f},{r['track'][1]:.0f})\nL={r['total_len']:.0f}cm"
          for r in results]
ax.bar(x - w/2, [r['res_init_mean'] for r in results], w, color='crimson',   alpha=0.75, label='Init')
ax.bar(x + w/2, [r['res_fit_mean']  for r in results], w, color='steelblue', alpha=0.75, label='Fitted')
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7)
ax.set_ylabel('Mean 3D pos residual [cm]')
ax.set_title('Per-track position residual')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis='y')
for xi, r in enumerate(results):
    iv, fv = r['res_init_mean'], r['res_fit_mean']
    imp = (iv - fv) / max(iv, 1e-6) * 100
    ax.text(xi, max(iv, fv)*1.05, f'{imp:+.0f}%', ha='center', fontsize=7,
            color='green' if imp > 0 else 'red')

fig.tight_layout()
fig.savefig('plots/joint_demo_summary.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print('Saved: plots/joint_demo_summary.png')


# ── Figure 2: per-track residual profile ─────────────────────────────────────
for ri_idx, r in enumerate(results):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Track {r['track']},  L={r['total_len']:.1f}cm,  n_chain={r['n_chain']},  "
                 f"dθ_rms={r['dtheta_rms']:.2f}mrad", fontsize=11, fontweight='bold')

    ax = axes[0]
    ax.plot(r['query_alphas'], r['res_init'], color='crimson',   lw=0.8, label=f"Init  ({r['res_init_mean']:.4f} cm)")
    ax.plot(r['query_alphas'], r['res_fit'],  color='steelblue', lw=0.8, label=f"Fit   ({r['res_fit_mean']:.4f} cm)")
    ax.set_xlabel('Arc-length α'); ax.set_ylabel('3D residual [cm]')
    ax.set_title('Position residual profile'); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    ax = axes[1]
    if len(r['fitted_dedx']) > 0:
        seg_alphas = np.linspace(0, 1, len(r['fitted_dedx']))
        ax.plot(seg_alphas, r['fitted_dedx'], color='seagreen', lw=0.8, label='Fitted dEdx (batch)')
        if len(all_true_dedx) > 0:
            ax.axhline(all_true_dedx.mean(), color='crimson', lw=1.5, ls='--',
                       label=f'True mean={all_true_dedx.mean():.3f}')
        ax.axhline(r['fitted_dedx'].mean(), color='seagreen', lw=1.5, ls=':',
                   label=f'Fit mean={r["fitted_dedx"].mean():.3f}')
        ax.set_xlabel('Segment index (α)'); ax.set_ylabel('dEdx [MeV/cm]')
        ax.set_title('Batch dEdx profile'); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'dEdx\n(other tracks in batch)', ha='center', va='center', transform=ax.transAxes)

    ax = axes[2]
    ax = fig.add_subplot(1, 3, 3, projection='3d')
    sl = slice(None, None, max(1, len(r['query_alphas']) // 200))
    ax.plot(*r['pos_true'][sl].T, color='crimson',   lw=1.5, label='True',  alpha=0.8)
    ax.plot(*r['pos_init'][sl].T, color='gray',      lw=1.5, label='Init',  alpha=0.6, ls='--')
    ax.plot(*r['pos_fit'][sl].T,  color='steelblue', lw=1.5, label='Fitted',alpha=0.8)
    ax.set_xlabel('x [cm]'); ax.set_ylabel('y [cm]'); ax.set_zlabel('z [cm]')
    ax.set_title('3D track'); ax.legend(fontsize=8)

    fig.tight_layout()
    out = f'plots/joint_demo_track_{ri_idx}.png'
    fig.savefig(out, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out}')

# ── Figure 3: dEdx distribution ───────────────────────────────────────────────
all_fitted_dedx = np.concatenate([r['fitted_dedx'] for r in results if len(r['fitted_dedx']) > 0])
if len(all_fitted_dedx) > 0 and len(all_true_dedx) > 0:
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(0, 8, 80)
    ax.hist(all_true_dedx,   bins=bins, density=True, alpha=0.5, color='crimson',   label=f'True  (mean={all_true_dedx.mean():.3f})')
    ax.hist(all_fitted_dedx, bins=bins, density=True, alpha=0.5, color='steelblue', label=f'Fitted (mean={all_fitted_dedx.mean():.3f})')
    ax.set_xlabel('dEdx [MeV/cm]'); ax.set_ylabel('Density')
    ax.set_title('dEdx distribution: true vs fitted'); ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig('plots/joint_demo_dedx_dist.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('Saved: plots/joint_demo_dedx_dist.png')
