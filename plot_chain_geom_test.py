"""
plot_chain_geom_test.py  —  visualise results from chain geometry fitting demo
"""
import os, sys, pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

os.environ['JAX_PLATFORMS'] = 'cpu'
sys.path.insert(0, '/sdf/group/neutrino/pgranger/larnd-sim-jax')
sys.path.insert(0, '/sdf/group/neutrino/pgranger/larnd-sim-jax/src')
os.chdir('/sdf/group/neutrino/pgranger/larnd-sim-jax')

CKPT = 'fit_result/chain_geom_test/history_iter100_chain_geom_test_100iter_b50.pkl'
AB_TRUE = 0.8348813503927325
AB_INIT = 0.8
os.makedirs('plots', exist_ok=True)

with open(CKPT, 'rb') as f:
    h = pickle.load(f)

iters = np.arange(len(h['Ab_iter']))       # 0..100 (101 points)
ab    = np.array(h['Ab_iter'])             # Ab at each update boundary
ab_grad  = np.array(h['Ab_grad'])          # gradient per step
loss_all = np.array(h['losses_iter'])
aux      = h['aux_iter']                   # list of dicts, length 100

# Pull per-iter aux quantities
def _aux(key, default=0.):
    return np.array([a.get(key, default) for a in aux])

mcs_prior  = _aux('mcs_prior')
dedx_prior = _aux('dedx_prior')
llhd_tick  = _aux('log_likelihood_tick')
no_match   = _aux('no_match_penalty')
mean_dedx  = np.array(h['mean_dedx_iter'])
dedx_mae   = np.array(h['dedx_mae_iter'])
step_time  = np.array(h['step_time'])

# dEdx fitted vs true
dedx_cache = h['dedx_cache']      # batch_idx → {log_dedx, opt_state}
true_cache = h['true_dedx_cache'] # batch_idx → array of true dEdx

fitted_dedx_all = []
true_dedx_all   = []
for bi in sorted(dedx_cache.keys()):
    ld  = np.array(dedx_cache[bi]['log_dedx'])
    fitted = np.exp(ld)
    fitted_dedx_all.append(fitted)
    if bi in true_cache:
        true_dedx_all.append(np.array(true_cache[bi]))

fitted_dedx = np.concatenate(fitted_dedx_all)  if fitted_dedx_all else np.array([])
true_dedx   = np.concatenate(true_dedx_all)    if true_dedx_all   else np.array([])

# ── Figure 1: Ab + loss panels ────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 12))
gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.42, wspace=0.30)
ax1 = fig.add_subplot(gs[0, :])   # wide Ab panel
ax2 = fig.add_subplot(gs[1, 0])   # loss components
ax3 = fig.add_subplot(gs[1, 1])   # MCS prior
ax4 = fig.add_subplot(gs[2, 0])   # mean dEdx
ax5 = fig.add_subplot(gs[2, 1])   # dEdx scatter

# ── Panel 1: Ab convergence ───────────────────────────────────────────────────
ax1.plot(iters, ab, 'o-', color='steelblue', ms=3, lw=1.5, label='Ab fitted')
ax1.axhline(AB_TRUE, color='crimson',  lw=2,   ls='--', label=f'Ab true = {AB_TRUE:.4f}')
ax1.axhline(AB_INIT, color='gray',     lw=1.2, ls=':',  label=f'Ab init = {AB_INIT:.3f}')

# shade bias region
ax1.fill_between(iters, AB_TRUE*0.99, AB_TRUE*1.01, alpha=0.12, color='crimson',
                 label='±1% of true')

final_ab  = float(ab[-1])
final_bias = final_ab - AB_TRUE
ax1.annotate(f'Final: Ab={final_ab:.4f}  (bias={final_bias:+.4f},  {final_bias/AB_TRUE*100:+.2f}%)',
             xy=(iters[-1], final_ab), xytext=(60, final_ab + 0.003),
             fontsize=10, color='steelblue',
             arrowprops=dict(arrowstyle='->', color='steelblue', lw=1.2))

ax1.set_xlabel('Iteration')
ax1.set_ylabel('Ab')
ax1.set_title('Ab convergence  (fit_chain_positions=True, fit_dedx=True)', fontsize=12)
ax1.legend(fontsize=9, loc='lower right')
ax1.grid(True, alpha=0.3)

# ── Panel 2: loss components ──────────────────────────────────────────────────
ax2_twin = ax2.twinx()
ax2.plot(loss_all,  color='black',      lw=1.5, label='Total loss')
ax2.plot(llhd_tick, color='royalblue',  lw=1.2, label='LLHD (tick)')
ax2.plot(no_match,  color='darkorange', lw=1.0, ls='--', label='No-match penalty')
ax2_twin.plot(-dedx_prior, color='green', lw=1.0, ls='-.', label='−dEdx NLL (rhs)')
ax2.set_xlabel('Iteration')
ax2.set_ylabel('Loss value')
ax2_twin.set_ylabel('dEdx prior (neg NLL)', color='green')
ax2_twin.tick_params(axis='y', colors='green')
lines1, labels1 = ax2.get_legend_handles_labels()
lines2, labels2 = ax2_twin.get_legend_handles_labels()
ax2.legend(lines1+lines2, labels1+labels2, fontsize=8)
ax2.set_title('Loss components')
ax2.grid(True, alpha=0.3)

# ── Panel 3: MCS prior ────────────────────────────────────────────────────────
ax3.plot(mcs_prior, color='purple', lw=1.5)
ax3.set_xlabel('Iteration')
ax3.set_ylabel('MCS prior (normalised)')
ax3.set_title('MCS prior on chain deflection angles\n(grows as geometry moves from straight-line)')
ax3.grid(True, alpha=0.3)

# Also show |Ab gradient|
ax3_twin = ax3.twinx()
ax3_twin.semilogy(np.abs(ab_grad), color='steelblue', lw=0.8, alpha=0.6)
ax3_twin.set_ylabel('|∂L/∂Ab|  (log)', color='steelblue')
ax3_twin.tick_params(axis='y', colors='steelblue')
ax3_twin.legend([Line2D([0],[0],color='steelblue',lw=1)], ['|Ab grad|'],
                loc='upper right', fontsize=8)

# ── Panel 4: mean dEdx ───────────────────────────────────────────────────────
n_md = len(mean_dedx)
ax4.plot(np.arange(n_md), mean_dedx,  color='darkorange', lw=1.5, label='Mean fitted dEdx')
ax4.plot(np.arange(n_md), dedx_mae,   color='green',      lw=1.2, ls='--', label='dEdx MAE vs true')
ax4.axhline(1.8740, color='gray', lw=1, ls=':', label='Prior centre (1.874 MeV/cm)')
ax4.set_xlabel('Iteration')
ax4.set_ylabel('dEdx  [MeV/cm]')
ax4.set_title('Per-segment dEdx evolution')
ax4.legend(fontsize=8)
ax4.grid(True, alpha=0.3)

# ── Panel 5: fitted vs true dEdx scatter ─────────────────────────────────────
if len(fitted_dedx) > 0 and len(true_dedx) > 0:
    n_min = min(len(fitted_dedx), len(true_dedx))
    fd, td = fitted_dedx[:n_min], true_dedx[:n_min]
    lim = max(td.max(), fd.max()) * 1.05
    ax5.scatter(td, fd, s=4, alpha=0.5, color='steelblue', rasterized=True)
    ax5.plot([0, lim], [0, lim], 'k--', lw=1, label='ideal (y=x)')
    ax5.set_xlabel('True dEdx  [MeV/cm]')
    ax5.set_ylabel('Fitted dEdx  [MeV/cm]')
    ax5.set_title(f'Per-segment dEdx: fitted vs true\n(N={n_min} segments)')
    ax5.set_xlim(0, lim); ax5.set_ylim(0, lim)
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)
    # correlation
    corr = float(np.corrcoef(td, fd)[0,1])
    ax5.text(0.05, 0.92, f'r = {corr:.3f}', transform=ax5.transAxes, fontsize=10)
else:
    ax5.text(0.5, 0.5, 'no dEdx cache data', ha='center', va='center',
             transform=ax5.transAxes)

fig.suptitle('Chain geometry fitting demonstration  (fit_chain_positions=True)\n'
             f'100 iterations, 3 batches (1–2 tracks each), RTX 2080 Ti, ~8 min',
             fontsize=12, fontweight='bold')

out1 = 'plots/chain_geom_test_results.png'
fig.savefig(out1, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'Saved: {out1}')

# ── Figure 2: per-step timing ─────────────────────────────────────────────────
fig2, ax = plt.subplots(figsize=(10, 4))
ax.plot(step_time, color='teal', lw=1.2)
ax.axvline(3, color='gray', ls=':', lw=1.2, label='JIT warm-up ends')
ax.set_xlabel('Iteration')
ax.set_ylabel('Step time  [s]')
ax.set_title('Per-iteration wall time  (JIT recompilation visible in first 2 steps)')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_ylim(bottom=0)
out2 = 'plots/chain_geom_test_timing.png'
fig2.savefig(out2, dpi=150, bbox_inches='tight')
plt.close(fig2)
print(f'Saved: {out2}')

# ── Figure 3: dEdx histogram ─────────────────────────────────────────────────
if len(fitted_dedx) > 0 and len(true_dedx) > 0:
    n_min = min(len(fitted_dedx), len(true_dedx))
    fd, td = fitted_dedx[:n_min], true_dedx[:n_min]
    fig3, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0, min(10, max(fd.max(), td.max())*1.1), 60)
    ax.hist(td, bins=bins, alpha=0.55, color='crimson',   label='True dEdx',   density=True)
    ax.hist(fd, bins=bins, alpha=0.55, color='steelblue', label='Fitted dEdx', density=True)
    ax.axvline(float(td.mean()), color='crimson',   lw=2, ls='--',
               label=f'True mean = {td.mean():.3f} MeV/cm')
    ax.axvline(float(fd.mean()), color='steelblue', lw=2, ls='--',
               label=f'Fitted mean = {fd.mean():.3f} MeV/cm')
    ax.set_xlabel('dEdx  [MeV/cm]')
    ax.set_ylabel('Density')
    ax.set_title('Distribution of per-segment dEdx: fitted vs true (after 100 iters)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    out3 = 'plots/chain_geom_test_dedx_dist.png'
    fig3.savefig(out3, dpi=150, bbox_inches='tight')
    plt.close(fig3)
    print(f'Saved: {out3}')

print('\n=== Summary ===')
print(f'Ab: init={AB_INIT:.4f}  final={ab[-1]:.4f}  true={AB_TRUE:.4f}  '
      f'bias={ab[-1]-AB_TRUE:+.4f} ({(ab[-1]-AB_TRUE)/AB_TRUE*100:+.2f}%)')
print(f'Loss: iter0={loss_all[0]:.2f}  iter99={loss_all[-1]:.2f}  '
      f'ratio={loss_all[-1]/loss_all[0]:.3f}')
print(f'MCS prior: max={mcs_prior.max():.4f}  at iter {mcs_prior.argmax()}')
if len(fitted_dedx) > 0 and len(true_dedx) > 0:
    n_min = min(len(fitted_dedx), len(true_dedx))
    corr = float(np.corrcoef(fitted_dedx[:n_min], true_dedx[:n_min])[0,1])
    print(f'dEdx correlation (fitted vs true): r={corr:.3f}')
print(f'Timing: first iter={step_time[0]:.1f}s  median={np.median(step_time[3:]):.1f}s')
