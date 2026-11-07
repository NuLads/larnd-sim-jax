"""
plot_pos_residual.py
====================
Post-process the position-only demo checkpoint.

For each chain-fitted track, compares:
  - Initial positions  : straight-line (linear_guess entry + uniform direction)
  - Fitted positions   : after N Adam steps on chain angles
  - True positions     : from checkpoint's 'true_positions' (arc-length interpolated
                         from the target batch at alpha_mid fractions — same matching
                         used during training, so event/track IDs are always consistent)

Position residual = mean(‖fitted_pos_j − true_pos_j‖₂) over fine steps.
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
import jax.numpy as jnp
from optimize.fit_params import _absolute_chain_forward

CKPT = 'fit_result/pos_demo/history_iter300_pos_demo_300iter.pkl'
os.makedirs('plots', exist_ok=True)

# ── Load checkpoint ────────────────────────────────────────────────────────────
with open(CKPT, 'rb') as f:
    h = pickle.load(f)

chain_cache    = h['chain_cache']     # {bi: [{'angles': np.array}]}
chain_ctxs     = h['chain_contexts']  # {bi: [dict]}
true_positions = h['true_positions']  # {bi: {str(track_id): (N, 3) float32}}
losses         = np.array(h['losses_iter'])
pr_iter        = np.array([x for x in h.get('pos_residual_iter', []) if x is not None])


def chain_positions_at_alphas(ctx, angles, query_alphas):
    """Evaluate chain positions at query_alphas ∈ [0,1] using absolute-direction layout."""
    n_c    = ctx['n_chain']
    thetas = jnp.array(angles[:n_c])
    phis   = jnp.array(angles[n_c:])
    pos_nodes, _ = _absolute_chain_forward(
        jnp.array(ctx['x0_fixed']), thetas, phis, ctx['step_len'])
    pos_nodes = np.array(pos_nodes)
    nc = n_c
    result = []
    for alpha in query_alphas:
        k    = int(np.clip(int(np.floor(alpha * nc)), 0, nc - 1))
        frac = float(alpha * nc - k)
        pos  = pos_nodes[k] + frac * (pos_nodes[k + 1] - pos_nodes[k])
        result.append(pos)
    return np.array(result)


# ── Process each track ─────────────────────────────────────────────────────────
results = []

for bi in sorted(chain_ctxs.keys()):
    ctxs   = chain_ctxs[bi]
    states = chain_cache[bi]
    tp_bi  = true_positions.get(int(bi), {})

    for ctx, state in zip(ctxs, states):
        tid_key = str(ctx['track_id'])
        true_pts = tp_bi.get(tid_key)
        if true_pts is None:
            print(f"  Batch {bi}, track {ctx['track_id']}: no true positions in checkpoint, skipping")
            continue
        true_pts = np.asarray(true_pts)

        query_alphas = np.asarray(ctx['alpha_mid'])
        angles_fit   = np.array(state['angles'])
        pos_fitted   = chain_positions_at_alphas(ctx, angles_fit, query_alphas)

        # Initial: all segments pointing in the initial track direction
        n_c = ctx['n_chain']
        angles_init        = np.empty_like(angles_fit)
        angles_init[:n_c]  = ctx['theta0_i']
        angles_init[n_c:]  = ctx['phi0_i']
        pos_init = chain_positions_at_alphas(ctx, angles_init, query_alphas)

        res_init   = np.linalg.norm(pos_init   - true_pts, axis=1)
        res_fitted = np.linalg.norm(pos_fitted - true_pts, axis=1)

        # RMS of consecutive direction changes (= effective deflections)
        thetas_fit = angles_fit[:n_c]
        phis_fit   = angles_fit[n_c:]
        dtheta_rms = float(np.sqrt(np.mean(np.diff(thetas_fit) ** 2))) * 1e3
        dphi_rms   = float(np.sqrt(np.mean(np.diff(phis_fit)   ** 2))) * 1e3

        results.append({
            'batch': bi, 'track': ctx['track_id'],
            'n_chain': n_c, 'total_len': ctx['total_len'],
            'res_init_mean':  float(res_init.mean()),
            'res_fit_mean':   float(res_fitted.mean()),
            'res_init':  res_init, 'res_fitted': res_fitted,
            'query_alphas': query_alphas,
            'pos_init': pos_init, 'pos_fitted': pos_fitted, 'pos_true': true_pts,
            'dtheta_rms_mrad': dtheta_rms, 'dphi_rms_mrad': dphi_rms,
        })
        print(f"Batch {bi}, track {ctx['track_id']}:  L={ctx['total_len']:.1f}cm  "
              f"init={res_init.mean():.4f}cm  fitted={res_fitted.mean():.4f}cm  "
              f"improvement={res_init.mean()-res_fitted.mean():.4f}cm  "
              f"({(res_init.mean()-res_fitted.mean())/max(res_init.mean(),1e-6)*100:.1f}%)  "
              f"dθ_rms={dtheta_rms:.2f}mrad")

if not results:
    print("No results — check checkpoint has 'true_positions' key")
    sys.exit(1)

# ── Figure 1: loss + pos_residual_iter + per-track residual bars ───────────────
n_tracks = len(results)
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Position-only chain fitting  (uniform dEdx, Ab frozen)\n'
             f'{int(h["total_iter"])} iterations on A100', fontsize=12, fontweight='bold')

# Loss curve
ax = axes[0]
ax.plot(losses, color='steelblue', lw=1.0, alpha=0.7)
ax.set_xlabel('Iteration'); ax.set_ylabel('Total loss')
ax.set_title('Loss vs iteration'); ax.grid(True, alpha=0.3)

# In-training position residual
ax = axes[1]
if len(pr_iter) > 0:
    ax.plot(pr_iter, color='darkorange', lw=1.0, alpha=0.8)
    ax.set_xlabel('Training step (batch)'); ax.set_ylabel('Mean 3D pos residual [cm]')
    ax.set_title(f'In-training residual\n(init≈{pr_iter[:30].mean():.4f}  final≈{pr_iter[-30:].mean():.4f} cm)')
    ax.grid(True, alpha=0.3)
else:
    ax.text(0.5, 0.5, 'no pos_residual_iter', ha='center', va='center', transform=ax.transAxes)

# Per-track bar chart
ax = axes[2]
x = np.arange(n_tracks)
w = 0.35
labels = [f"B{r['batch']}\n({r['track'][0]:.0f},{r['track'][1]:.0f})\nL={r['total_len']:.0f}cm"
          for r in results]
init_vals   = [r['res_init_mean']  for r in results]
fitted_vals = [r['res_fit_mean']   for r in results]
ax.bar(x - w/2, init_vals,   w, color='crimson',   alpha=0.75, label='Initial (straight)')
ax.bar(x + w/2, fitted_vals, w, color='steelblue', alpha=0.75, label='After fitting')
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel('Mean 3D pos residual vs true [cm]')
ax.set_title('Per-track residual: before vs after')
ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis='y')
for xi, (iv, fv) in enumerate(zip(init_vals, fitted_vals)):
    imp = (iv - fv) / max(iv, 1e-6) * 100
    ax.text(xi, max(iv, fv) + max(iv, fv) * 0.05, f'{imp:+.0f}%',
            ha='center', fontsize=8, color='green' if imp > 0 else 'red')

fig.tight_layout()
fig.savefig('plots/pos_demo_residuals.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print('\nSaved: plots/pos_demo_residuals.png')

# ── Figure 2: per-track residual profile + 3D overlay ─────────────────────────
for ri, r in enumerate(results):
    fig = plt.figure(figsize=(14, 5))
    fig.suptitle(f"Track {r['track']},  L={r['total_len']:.1f}cm,  n_chain={r['n_chain']}\n"
                 f"dθ_rms={r['dtheta_rms_mrad']:.2f}mrad  dφ_rms={r['dphi_rms_mrad']:.2f}mrad",
                 fontsize=11, fontweight='bold')

    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(r['query_alphas'], r['res_init'],   color='crimson',   lw=0.8, alpha=0.8,
             label=f"Initial  ({r['res_init_mean']:.4f} cm)")
    ax1.plot(r['query_alphas'], r['res_fitted'], color='steelblue', lw=0.8, alpha=0.8,
             label=f"Fitted   ({r['res_fit_mean']:.4f} cm)")
    ax1.set_xlabel('Arc-length α'); ax1.set_ylabel('3D residual vs true [cm]')
    ax1.set_title('Residual profile'); ax1.legend(fontsize=9); ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    step = max(1, len(r['query_alphas']) // 200)
    sl = slice(None, None, step)
    ax2.plot(*r['pos_true'][sl].T,   color='crimson',   lw=1.5, label='True',    alpha=0.8)
    ax2.plot(*r['pos_init'][sl].T,   color='gray',      lw=1.5, label='Initial', alpha=0.6, ls='--')
    ax2.plot(*r['pos_fitted'][sl].T, color='steelblue', lw=1.5, label='Fitted',  alpha=0.8)
    ax2.set_xlabel('x [cm]'); ax2.set_ylabel('y [cm]'); ax2.set_zlabel('z [cm]')
    ax2.set_title('3D track comparison'); ax2.legend(fontsize=8)

    fig.tight_layout()
    out = f'plots/pos_demo_track_{ri}.png'
    fig.savefig(out, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out}')

print('\n=== Summary ===')
mean_init = np.mean([r['res_init_mean']  for r in results])
mean_fit  = np.mean([r['res_fit_mean']   for r in results])
print(f'Mean initial residual : {mean_init:.4f} cm')
print(f'Mean final residual   : {mean_fit:.4f} cm')
print(f'Mean improvement      : {mean_init-mean_fit:.4f} cm  ({(mean_init-mean_fit)/max(mean_init,1e-6)*100:.1f}%)')
if len(pr_iter) > 0:
    print(f'In-training residual  : {pr_iter[:30].mean():.4f} → {pr_iter[-30:].mean():.4f} cm')
