"""Fig. noise_effect, version 7 (19 Oct 2026; v6 with panel (c) = charge of the first hit): "if the noise averaged out, these would all be zero".
Per pixel, noise-averaged readout minus noiseless readout, for the observables of the fit. Existing outputs only:
  - SAMPLING_STEP/prob_vs_stoch_cache.npz: 1e4 noise throws of the production stochastic electronics on the 926 pixels of
    Fig. prob_vs_stoch (tot, cnt, st = summed hit counts / first-second hit counts / summed hit ticks) and the
    probabilistic model (lam_all, Et);
  - SAMPLING_STEP/noise_effect_v3_cache.npz: the noiseless readout of the same pixels (hit0, t0) and the threshold scan
    (hits without noise against the mean over 1e3 throws, 3-10 ke).
(a) number of hits per pixel; (b) time of the first hit; (c) total number of hits of the batch against the threshold.
usage: python fig_noise_effect_v6.py [out_stem]"""
import sys, json
import numpy as np
D = '/sdf/scratch/neutrino/pgranger/review_2026_09/'
OUT = sys.argv[1] if len(sys.argv) > 1 else D + 'FIG4/noise_effect_v7'
p = np.load(D + 'SAMPLING_STEP/prob_vs_stoch_cache.npz'); v = np.load(D + 'SAMPLING_STEP/noise_effect_v3_cache.npz')
N = int(p['N']); ts = float(p['t_sampling']); h0 = v['hit0']; t0 = v['t0']
n0 = h0.sum(1)
dn_mc = p['tot'] / N - n0                      # noise average minus noiseless, hits per pixel
dn_mod = p['lam_all'].sum(1) - n0              # same from the probabilistic model
sel = h0[:, 0] & (p['cnt'][:, 0] >= 100)       # pixels with a first hit without noise (and in >= 1% of the throws)
dt_mc = (p['st'][:, 0] / np.maximum(p['cnt'][:, 0], 1) - t0[:, 0])[sel] * ts      # us
dt_mod = (p['Et'][:, 0] - t0[:, 0])[sel] * ts
# ADC = ALPHA * Q + BETA (fee_jax.digitize, continuous; module0.yaml: GAIN 4e-3 mV/e, V_PEDESTAL 580, V_CM 284.277, V_REF 1282.715 mV, 256 counts)
ALPHA = 4e-3 * 256 / (1282.71484375 - 284.27734375); BETA = (580.0 - 284.27734375) * 256 / (1282.71484375 - 284.27734375)
Q0 = (v['a0'][:, 0] - BETA) / ALPHA
dq_mc = 100 * (((p['sa'][sel, 0] / p['cnt'][sel, 0] - BETA) / ALPHA) / Q0[sel] - 1)      # %
dq_mod = 100 * (((p['Eadc'][sel, 0] - BETA) / ALPHA) / Q0[sel] - 1)
num = dict(n_pixels=int(len(n0)), dn_abs_gt_0p1=int((np.abs(dn_mc) > 0.1).sum()), dn_abs_gt_0p3=int((np.abs(dn_mc) > 0.3).sum()),
           hits_noiseless=int(n0.sum()), hits_mean=float(p['tot'].sum() / N),
           n_dt=int(sel.sum()), dt_median_us=float(np.median(dt_mc)), dt_mean_us=float(dt_mc.mean()),
           dt_frac_earlier=float((dt_mc < 0).mean()), dt_model_median_us=float(np.median(dt_mod)),
           q0_median_ke=float(np.median(Q0[sel]) / 1e3), dq_median_pct=float(np.median(dq_mc)), dq_mean_pct=float(dq_mc.mean()),
           dq_frac_higher=float((dq_mc > 0).mean()), dq_model_median_pct=float(np.median(dq_mod)))
print(json.dumps(num, indent=1))

import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 8, 'axes.labelsize': 8, 'legend.fontsize': 6.3, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
                     'axes.linewidth': 0.6, 'font.family': 'sans-serif'})
C0, C1, CG = '#D55E00', '#0072B2', '#7F7F7F'
fig, (a, b, c) = plt.subplots(1, 3, figsize=(7.0, 2.7), gridspec_kw=dict(width_ratios=[1, 1, 1.05]))

def zero_line(ax):
    ax.axvline(0, color=C0, lw=1.4, label='if the noise averaged out')

bins = np.arange(-0.775, 0.65, 0.05)
a.hist(np.clip(dn_mc, bins[0], bins[-1] - 1e-9), bins=bins, color=C1, alpha=0.6, label='average over $10^4$ noise throws')
a.hist(np.clip(dn_mod, bins[0], bins[-1] - 1e-9), bins=bins, histtype='step', color='k', lw=0.8, label='probabilistic model')
a.set_yscale('log'); a.set_ylim(0.7, 3e3)
zero_line(a)
a.set_xlabel('hits per pixel:\naverage $-$ without noise'); a.set_ylabel('pixels')
a.text(0.97, 0.86, f"{num['dn_abs_gt_0p1']} of {num['n_pixels']}\npixels differ\nby more than 0.1", transform=a.transAxes, fontsize=6.3, va='top', ha='right')
a.set_xlim(-0.75, 0.95)

bins = np.arange(-0.6, 0.12, 0.02)
b.hist(np.clip(dt_mc, bins[0], bins[-1] - 1e-9), bins=bins, color=C1, alpha=0.6)
b.hist(np.clip(dt_mod, bins[0], bins[-1] - 1e-9), bins=bins, histtype='step', color='k', lw=0.8)
zero_line(b)
b.set_xlabel('first-hit time ($\\mu$s):\naverage $-$ without noise'); b.set_ylabel('pixels')
b.text(0.03, 0.86, f"earlier on {100 * num['dt_frac_earlier']:.0f}% of pixels\nmedian {num['dt_median_us']:+.2f} $\\mu$s,\n$\\simeq 170\\,\\mu$m of drift",
       transform=b.transAxes, fontsize=6, va='top')
b.set_xlim(bins[0], bins[-1])

bins = np.arange(-11, 12.01, 0.5)
c.hist(np.clip(dq_mc, bins[0], bins[-1] - 1e-9), bins=bins, color=C1, alpha=0.6)
c.hist(np.clip(dq_mod, bins[0], bins[-1] - 1e-9), bins=bins, histtype='step', color='k', lw=0.8)
zero_line(c)
c.set_xlabel('first-hit charge (%):\naverage / without noise $-$ 1'); c.set_ylabel('pixels'); c.set_xlim(bins[0], bins[-1])
c.text(0.03, 0.86, f"higher on\n{100 * num['dq_frac_higher']:.0f}% of pixels,\nmedian {num['dq_median_pct']:+.1f}%", transform=c.transAxes,
       fontsize=6.3, va='top', ha='left')
for ax, tag in ((a, '(a)'), (b, '(b)'), (c, '(c)')):
    ax.text(0.03, 0.97, tag, transform=ax.transAxes, ha='left', va='top', fontsize=8)
h, l = a.get_legend_handles_labels()
order = [l.index('if the noise averaged out'), l.index('average over $10^4$ noise throws'), l.index('probabilistic model')]
fig.legend([h[i] for i in order], [l[i] for i in order], loc='upper center', ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.0),
           handlelength=1.4, columnspacing=1.6)
fig.tight_layout(w_pad=1.6, rect=(0, 0, 1, 0.92))
fig.savefig(OUT + '.pdf'); fig.savefig(OUT + '.png', dpi=200)
json.dump(num, open(OUT + '_numbers.json', 'w'), indent=1)
print('wrote', OUT)
