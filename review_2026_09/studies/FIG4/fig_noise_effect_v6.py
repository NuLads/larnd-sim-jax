"""Fig. noise_effect, version 6 (19 Oct 2026): "if the noise averaged out, these would all be zero".
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
OUT = sys.argv[1] if len(sys.argv) > 1 else D + 'FIG4/noise_effect_v6'
p = np.load(D + 'SAMPLING_STEP/prob_vs_stoch_cache.npz'); v = np.load(D + 'SAMPLING_STEP/noise_effect_v3_cache.npz')
N = int(p['N']); ts = float(p['t_sampling']); h0 = v['hit0']; t0 = v['t0']
n0 = h0.sum(1)
dn_mc = p['tot'] / N - n0                      # noise average minus noiseless, hits per pixel
dn_mod = p['lam_all'].sum(1) - n0              # same from the probabilistic model
sel = h0[:, 0] & (p['cnt'][:, 0] >= 100)       # pixels with a first hit without noise (and in >= 1% of the throws)
dt_mc = (p['st'][:, 0] / np.maximum(p['cnt'][:, 0], 1) - t0[:, 0])[sel] * ts      # us
dt_mod = (p['Et'][:, 0] - t0[:, 0])[sel] * ts
sc = v['scan']; T = sc[:, 0] / 1e3; rel = 100 * (sc[:, 1] / sc[:, 3] - 1)
num = dict(n_pixels=int(len(n0)), dn_abs_gt_0p1=int((np.abs(dn_mc) > 0.1).sum()), dn_abs_gt_0p3=int((np.abs(dn_mc) > 0.3).sum()),
           hits_noiseless=int(n0.sum()), hits_mean=float(p['tot'].sum() / N),
           n_dt=int(sel.sum()), dt_median_us=float(np.median(dt_mc)), dt_mean_us=float(dt_mc.mean()),
           dt_frac_earlier=float((dt_mc < 0).mean()), dt_model_median_us=float(np.median(dt_mod)),
           scan_rel_pct={f'{t:.1f}': round(float(r), 2) for t, r in zip(T, rel)})
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
a.text(0.03, 0.86, f"{num['dn_abs_gt_0p1']} of {num['n_pixels']} pixels\ndiffer by more than 0.1", transform=a.transAxes, fontsize=6.3, va='top')
a.set_xlim(-0.75, 0.65)

bins = np.arange(-0.6, 0.12, 0.02)
b.hist(np.clip(dt_mc, bins[0], bins[-1] - 1e-9), bins=bins, color=C1, alpha=0.6)
b.hist(np.clip(dt_mod, bins[0], bins[-1] - 1e-9), bins=bins, histtype='step', color='k', lw=0.8)
zero_line(b)
b.set_xlabel('first-hit time ($\\mu$s):\naverage $-$ without noise'); b.set_ylabel('pixels')
b.text(0.03, 0.86, f"earlier on {100 * num['dt_frac_earlier']:.0f}% of pixels\nmedian {num['dt_median_us']:+.2f} $\\mu$s,\n$\\simeq 170\\,\\mu$m of drift",
       transform=b.transAxes, fontsize=6, va='top')
b.set_xlim(bins[0], bins[-1])

c.axhline(0, color=C0, lw=1.3)
c.plot(T, rel, color=C1, marker='o', ms=3, lw=1.1)
i5 = int(np.argmin(np.abs(T - 5.0)))
c.plot(T[i5], rel[i5], 'o', ms=6, mfc='none', mec='k', mew=0.7)
c.text(T[i5] - 0.15, rel[i5] + 0.9, f'{rel[i5]:+.1f}%', ha='right', va='bottom', fontsize=6.3)
c.axvline(5.0, color=CG, ls=':', lw=0.8)
c.text(5.0 - 0.12, -9.6, 'nominal threshold', rotation=90, ha='right', va='bottom', fontsize=6, color=CG)
c.set_xlabel('discrimination threshold\n(ke$^-$)'); c.set_ylabel('hits in the batch: without noise\n/ average $-$ 1 (%)')
c.set_xlim(2.7, 10.3); c.set_ylim(-10, 7); c.set_yticks(range(-10, 7, 2)); c.grid(alpha=0.25, lw=0.4)
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
