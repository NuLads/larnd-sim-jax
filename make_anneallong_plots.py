"""Convergence diagnostics for ANNEALLONG (3 seeds x 10000 iterations)."""
import pickle, glob, re, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import optax

OUT = 'plots/anneallong'
os.makedirs(OUT, exist_ok=True)
P = ['Ab', 'eField', 'tran_diff', 'long_diff', 'lifetime']
PL = {'Ab': 'A$_b$ (recomb.)', 'eField': 'E field', 'tran_diff': 'tran. diffusion',
      'long_diff': 'long. diffusion', 'lifetime': 'lifetime'}
SC = ['#0072B2', '#E69F00', '#009E73']          # seed colours (Okabe-Ito)
GREY, RED = '#666666', '#D55E00'
plt.rcParams.update({'font.size': 9, 'axes.grid': True, 'grid.alpha': .25,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.dpi': 130, 'savefig.bbox': 'tight'})

runs = {}
for f in sorted(glob.glob('fit_result/sci_full_ANNEALLONG/history_iter*.pkl')):
    runs[int(re.search(r'seed(\d+)', f).group(1))] = pickle.load(open(f, 'rb'))
seeds = sorted(runs)
N = len(np.ravel(np.array(runs[seeds[0]]['lifetime_iter'])))


def err(h, p):
    v = np.ravel(np.array(h[p + '_iter'])); t = np.ravel(h[p + '_target'])[0]
    return (v / t - 1) * 100


def roll(a, k, f=np.median):
    a = np.asarray(a, float); n = len(a)
    return np.array([f(a[max(0, i - k):i + 1]) for i in range(n)])


# LR schedule actually used (decay 0.91 per 100-iteration epoch, warmup 500)
lr = np.array([float(optax.warmup_exponential_decay_schedule(
    init_value=0., peak_value=1e-1, warmup_steps=500, transition_steps=100,
    decay_rate=0.91, staircase=True)(t)) for t in range(N)])

# ── Fig A ── parameter traces + the LR that drives them ─────────────────────
fig, axes = plt.subplots(2, 3, figsize=(14.2, 6.4))
for ax, p in zip(axes.ravel(), P):
    for j, s in enumerate(seeds):
        e = err(runs[s], p)
        ax.plot(e, lw=.8, color=SC[j], alpha=.75)
        ax.plot(roll(e, 201), lw=1.7, color=SC[j], label=f'seed {s}')
    ax.axhline(0, color='k', lw=.9)
    ax.axhspan(-5, 5, color=GREY, alpha=.16, zorder=0)
    ax.axvline(6000, ls=':', color=RED, lw=1.2)
    lim = max(12, np.percentile([abs(err(runs[s], p)).max() for s in seeds], 100) * .45)
    ax.set_ylim(-lim, lim)
    ax.set_title(PL[p], loc='left', fontsize=9.5)
    ax.set_xlabel('iteration'); ax.set_ylabel('error (%)')
    ax.legend(fontsize=7, frameon=False, ncol=3)
ax = axes.ravel()[5]
ax.plot(lr / 1e-1 * 100, lw=2, color=RED)
ax.axvline(6000, ls=':', color=RED, lw=1.2)
ax.text(6150, 30, 'LR < 0.1% of peak\n→ parameters frozen\n   beyond here', fontsize=7.5, color=RED)
ax.set_yscale('log'); ax.set_xlabel('iteration'); ax.set_ylabel('calibration LR (% of peak)')
ax.set_title('the schedule driving all five', loc='left', fontsize=9.5)
fig.suptitle('ANNEALLONG — parameter traces, 3 seeds x 10000 iterations. Thin = raw, thick = rolling median.\n'
             'Shaded ±5%. Dotted line = where the calibration LR falls below 0.1% of peak.',
             x=.02, ha='left', y=1.02)
fig.tight_layout(); fig.savefig(f'{OUT}/A_traces.png'); plt.close(fig)

# ── Fig B ── the decision-relevant view: what would you have reported? ──────
fig, axes = plt.subplots(1, 3, figsize=(14.2, 3.9))
ax = axes[0]
for j, s in enumerate(seeds):
    for p, ls in [('lifetime', '-'), ('long_diff', '--')]:
        e = err(runs[s], p)
        run_est = np.array([np.median(e[max(0, i - int(0.2 * i)):i + 1]) if i > 400 else np.nan
                            for i in range(N)])
        ax.plot(run_est, ls, lw=1.4, color=SC[j], alpha=.9,
                label=f'seed {s} {"τ" if p=="lifetime" else "long"}')
ax.axhline(0, color='k', lw=.9); ax.axhspan(-5, 5, color=GREY, alpha=.16, zorder=0)
ax.set_xlim(400, N); ax.set_ylim(-30, 30)
ax.set_xlabel('iteration at which you STOP'); ax.set_ylabel('value you would report (%)')
ax.legend(fontsize=6.5, frameon=False, ncol=3, loc='lower right')
ax.set_title('(a) Stopping-point sensitivity\n(trailing-20% median = our robust estimator)', loc='left', fontsize=9)

ax = axes[1]
for j, s in enumerate(seeds):
    for p, ls in [('lifetime', '-'), ('long_diff', '--'), ('Ab', ':')]:
        e = err(runs[s], p)
        w = 500
        d = np.array([np.median(e[max(0, i - w):i + 1]) - np.median(e[max(0, i - 2 * w):max(1, i - w)])
                      if i > 2 * w else np.nan for i in range(N)])
        # smooth |drift| so zero-crossings don't spike a log axis into noise
        ad = np.abs(d); m = np.isfinite(ad)
        sm = np.full_like(ad, np.nan); sm[m] = roll(ad[m], 301, np.mean)
        ax.plot(sm, ls, lw=1.4, color=SC[j], alpha=.9)
ax.axhline(1.25, color=RED, lw=1.4)
ax.text(1700, 1.55, 'run-to-run noise floor (1.25 pts)', color=RED, fontsize=7.5)
ax.set_yscale('log'); ax.set_xlim(1500, N); ax.set_ylim(1e-3, 60)
ax.set_xlabel('iteration'); ax.set_ylabel('|drift| over a 500-iter window (pts, smoothed)')
ax.set_title('(b) Residual drift — solid τ, dashed long, dotted A$_b$\nbelow the red line = unresolvable',
             loc='left', fontsize=9)

ax = axes[2]
for j, s in enumerate(seeds):
    for p, ls in [('lifetime', '-'), ('long_diff', '--')]:
        e = err(runs[s], p)
        ax.plot(roll(np.abs(np.diff(e, prepend=e[0])), 201, np.mean), ls, lw=1.3, color=SC[j], alpha=.85)
ax.set_yscale('log'); ax.set_xlabel('iteration'); ax.set_ylabel('mean |step| per iteration (pts)')
ax.axvline(6000, ls=':', color=RED, lw=1.2)
ax.set_title('(c) Step size — shows the quench directly', loc='left', fontsize=9)
fig.suptitle('ANNEALLONG — convergence diagnostics. (a) is the one to read: it shows how the reported '
             'number\ndepends on where you stop.', x=.02, ha='left', y=1.04)
fig.tight_layout(); fig.savefig(f'{OUT}/B_convergence.png'); plt.close(fig)

# ── Fig C ── the things that were still moving at 5000 ──────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14.2, 3.7))
for j, s in enumerate(seeds):
    h = runs[s]
    pr = np.ravel(h['pos_residual_iter']) * 1e4
    axes[0].plot(pr, lw=.3, color=SC[j], alpha=.18)
    axes[0].plot(roll(pr, 301), lw=1.7, color=SC[j], label=f'seed {s}')
    axes[1].plot(roll(np.ravel(h['dedx_mae_iter']), 201, np.mean), lw=1.5, color=SC[j], label=f'seed {s}')
    axes[2].plot(roll(np.ravel(h['losses_iter']), 201, np.mean), lw=1.5, color=SC[j], label=f'seed {s}')
axes[0].set_yscale('log'); axes[0].set_ylabel('position residual (µm)')
axes[0].axvline(5000, ls=':', color=RED); axes[0].text(5150, 900, '5000\n(old stop)', fontsize=7, color=RED)
axes[0].set_title('Position — kept improving to 10k\n(255 µm → 168 µm)', loc='left', fontsize=9)
axes[1].set_ylabel('per-segment dE/dx MAE (MeV/cm)')
axes[1].axvline(5000, ls=':', color=RED)
axes[1].set_title('dE/dx nuisance accuracy', loc='left', fontsize=9)
axes[2].set_yscale('log'); axes[2].set_ylabel('loss')
axes[2].axvline(5000, ls=':', color=RED)
axes[2].set_title('Total loss', loc='left', fontsize=9)
for ax in axes:
    ax.set_xlabel('iteration'); ax.legend(fontsize=7.5, frameon=False)
fig.suptitle('ANNEALLONG — quantities that had NOT converged at 5000 iterations', x=.02, ha='left', y=1.04)
fig.tight_layout(); fig.savefig(f'{OUT}/C_position_dedx_loss.png'); plt.close(fig)

# numeric companion
print('ANNEALLONG convergence summary (robust: trailing-20% median)')
print(f"{'param':11s} " + ' '.join(f'{"seed "+str(s):>12s}' for s in seeds) + f"{'mean±sd':>16s}{'|drift|':>9s}")
for p in P:
    cs, ds = [], []
    for s in seeds:
        e = err(runs[s], p)
        c = np.median(e[int(N * .8):]); cs.append(c)
        ds.append(abs(np.median(e[int(N * .9):]) - np.median(e[int(N * .8):int(N * .9)])))
    print(f'{p:11s} ' + ' '.join(f'{c:+12.2f}' for c in cs)
          + f'{np.mean(cs):+9.2f}±{np.std(cs):4.2f}{np.mean(ds):9.2f}')
for lab, key, sc in [('position(µm)', 'pos_residual_iter', 1e4), ('dEdx MAE', 'dedx_mae_iter', 1)]:
    vs = []
    for s in seeds:
        a = np.ravel(runs[s][key]) * sc
        vs.append(np.median(a[int(len(a) * .8):]))
    print(f'{lab:11s} ' + ' '.join(f'{v:12.3f}' for v in vs) + f'{np.mean(vs):9.3f}±{np.std(vs):4.3f}')
print('\nwrote', OUT)
