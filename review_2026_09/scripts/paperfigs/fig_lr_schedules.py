"""Fig. lr_schedules (paper version, 10 Oct): the learning rates of the three parameter blocks over a production fit,
computed with the same optax schedules and the production settings (sci_full_STRFRZ50*, config + env of the
checkpoints): calibration warmup_exponential_decay (peak 0.1, warm-up 500, x0.91 per epoch of 100 iterations,
staircase); per-segment dE/dx constant 1e-2 (LARND_DEDX_LR_FINAL_FRAC unset); trajectory coefficients 1e-2 x 0.9997
per iteration from 0. The alternative schedule of the path-invariance test (sci_full_SCHED*): transition_begin 1200,
x0.89768 per epoch. The optimiser step count in the production checkpoints is 10000 (checked), so the calibration LR
ends at 1.29e-5 = 0.013% of its peak.
(a) learning rate relative to each block's peak; (b) cumulative calibration travel sum(eta)."""
import numpy as np, optax
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
OUT = '/sdf/group/neutrino/pgranger/NuLads_draft/figures/lr_schedules'
N = 10000; it = np.arange(N + 1)
prod = optax.warmup_exponential_decay_schedule(init_value=0., peak_value=0.1, warmup_steps=500, transition_steps=100,
                                               decay_rate=0.91, staircase=True)
alt = optax.warmup_exponential_decay_schedule(init_value=0., peak_value=0.1, warmup_steps=500, transition_steps=100,
                                              decay_rate=0.89768, transition_begin=1200, staircase=True)
cal = np.array([float(prod(t)) for t in it]); cal2 = np.array([float(alt(t)) for t in it])
chain = 0.9997 ** it; dedx = np.ones_like(it, dtype=float)
print(f'calibration at {N}: production {cal[-1]:.3e} ({100*cal[-1]/0.1:.4f}% of peak), alternative {cal2[-1]:.3e}; '
      f'travel {cal.sum():.1f} vs {cal2.sum():.1f} (x{cal2.sum()/cal.sum():.3f}); trajectories at {N}: {100*chain[-1]:.1f}%')

plt.rcParams.update({'font.size': 8, 'axes.labelsize': 8, 'legend.fontsize': 6.8, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
                     'axes.linewidth': 0.6})
VERM, BLUE, GREEN, GREY = '#D55E00', '#0072B2', '#009E73', '#666666'
fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.5), gridspec_kw=dict(width_ratios=[1.35, 1]))
a.plot(it, 100 * cal / 0.1, color=VERM, lw=1.3, label=r'calibration $\theta$ ($\times 0.91$/epoch)')
a.plot(it, 100 * cal2 / 0.1, color=VERM, lw=1.0, ls='--', label=r'calibration, test schedule')
a.plot(it, 100 * dedx, color=BLUE, lw=1.3, label=r'segment $dE/dx$ (constant)')
a.plot(it, 100 * chain, color=GREEN, lw=1.3, label=r'trajectories $\beta$ ($\times 0.9997$/it.)')
a.set_yscale('log'); a.set_ylim(5e-3, 3e3); a.set_xlim(0, N)
a.set_xlabel('iteration'); a.set_ylabel("% of the block's peak learning rate")
a.grid(alpha=0.25, lw=0.5); a.legend(loc='lower left', frameon=False, handlelength=2.2, borderaxespad=0.3)
a.text(0.02, 0.95, '(a)', transform=a.transAxes, va='top')
a.annotate(f'{100*cal[-1]/0.1:.3f}%', xy=(N, 100 * cal[-1] / 0.1), xytext=(7600, 0.8), fontsize=6.5, color=VERM,
           arrowprops=dict(arrowstyle='->', color=VERM, lw=0.6))
b.plot(it, np.cumsum(cal), color=VERM, lw=1.3, label='production')
b.plot(it, np.cumsum(cal2), color=VERM, lw=1.0, ls='--', label=f'path-invariance test ($\\times{cal2.sum()/cal.sum():.2f}$)')
b.set_xlim(0, N); b.set_ylim(0, None); b.set_xlabel('iteration'); b.set_ylabel(r'cumulative calibration travel $\sum \eta$')
b.grid(alpha=0.25, lw=0.5); b.legend(loc='lower right', frameon=False)
b.text(0.04, 0.95, '(b)', transform=b.transAxes, va='top')
fig.tight_layout(w_pad=1.5)
fig.savefig(OUT + '.pdf', bbox_inches='tight'); fig.savefig(OUT + '.png', dpi=300, bbox_inches='tight')
print('saved', OUT + '.pdf/.png')
