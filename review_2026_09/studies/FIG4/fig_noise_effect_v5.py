"""Fig. noise_effect, version 5 (19 Oct 2026, option A): "the readout without noise is not the average readout".
Reuses existing outputs only (no new noise throws):
  - SAMPLING_STEP/prob_vs_stoch_cache.npz: 1e4 noise throws of the production stochastic electronics on the 926 pixels of
    Fig. prob_vs_stoch (cnt = number of throws with a first / second hit) and the probabilistic model (lam);
  - FIG4/noiseless_quantities.npz (noise_throws.py, deterministic part): hits without noise n0, peak cumulative charge
    Q_max, peak charge collected after the first noiseless reset R_max, for the same pixels (order checked: n0 equals
    hit0 of the v3 cache);
  - SAMPLING_STEP/noise_effect_v3_cache.npz: threshold scan of the same batch, hits without noise against the mean over
    1e3 noise throws, 3-10 ke.
(a) probability of a first hit against (Q_max - V_th)/sigma_r; (b) probability of a second hit against
(R_max - V_th)/sigma_r, pixels with a first hit without noise; (c) hits without noise relative to the noise average,
against the threshold.  usage: python fig_noise_effect_v5.py [out_stem]"""
import sys, json
import numpy as np
D = '/sdf/scratch/neutrino/pgranger/review_2026_09/'
OUT = sys.argv[1] if len(sys.argv) > 1 else D + 'FIG4/noise_effect_v5'
p = np.load(D + 'SAMPLING_STEP/prob_vs_stoch_cache.npz'); b = np.load(D + 'FIG4/noiseless_quantities.npz')
v = np.load(D + 'SAMPLING_STEP/noise_effect_v3_cache.npz')
N = int(p['N']); f1, f2 = p['cnt'][:, 0] / N, p['cnt'][:, 1] / N; l1, l2 = p['lam'][:, 0], p['lam'][:, 1]
n0 = b['n0']; thr = float(b['thr']); sr = float(b['sig_r'])
assert np.array_equal(n0, v['hit0'].sum(1))
x1 = (b['Qmax'] - thr) / sr
m2 = n0 >= 1; x2 = (b['Rmax'] - thr) / sr
nexp = p['tot'] / N
num = dict(N_throws=N, n_pixels=int(len(n0)),
           n_pix_hits_differ_0p1=int((np.abs(nexp - n0) > 0.1).sum()),
           first_within_2sig=int((np.abs(x1) < 2).sum()), first_differ_0p1=int((np.abs(f1 - (n0 >= 1)) > 0.1).sum()),
           first_above_4sig=int((x1 > 4).sum()),
           second_pixels=int(m2.sum()), second_within_2sig=int((m2 & (np.abs(x2) < 2)).sum()),
           second_differ_0p1=int((m2 & (np.abs(f2 - (n0 >= 2)) > 0.1)).sum()),
           single_noiseless_refire_0p1=int(((n0 == 1) & (f2 > 0.1)).sum()), single_noiseless=int((n0 == 1).sum()),
           single_below_thr_refire_0p1=int(((n0 == 1) & (b['Rmax'] < thr) & (f2 > 0.1)).sum()),
           total_hits_noiseless=int(n0.sum()), total_hits_mean=float(nexp.sum()))
sc = v['scan']; T = sc[:, 0] / 1e3; rel = 100 * (sc[:, 1] / sc[:, 3] - 1); err = 100 * sc[:, 1] / sc[:, 3] ** 2 * sc[:, 4] / np.sqrt(1000)
num['scan_rel_pct'] = {f'{t:.1f}': round(float(r), 2) for t, r in zip(T, rel)}
print(json.dumps(num, indent=1))

import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 8, 'axes.labelsize': 8, 'legend.fontsize': 6.3, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
                     'axes.linewidth': 0.6, 'font.family': 'sans-serif'})
C0, C1, CG = '#D55E00', '#0072B2', '#7F7F7F'
fig, (a, bb, c) = plt.subplots(1, 3, figsize=(7.0, 2.65), gridspec_kw=dict(width_ratios=[1, 1, 1.05]))
XL = (-4, 4)

def prob_panel(ax, x, f, lam, noiseless, sel, xlabel, ylabel, tag):
    s = sel & (x > XL[0]) & (x < XL[1])
    ax.axvspan(-2, 2, color=C1, alpha=0.07, lw=0)
    ax.scatter(x[s], noiseless[s].astype(float), s=26, marker='_', color=C0, lw=1.6, label='without noise', zorder=5)
    ax.scatter(x[s], f[s], s=7, color=C1, alpha=0.75, lw=0, label=f'average over {N:.0e} noise throws'.replace('1e+04', '$10^4$'), zorder=3)
    ax.scatter(x[s], lam[s], s=9, facecolors='none', edgecolors='k', lw=0.4, label='probabilistic model', zorder=4)
    ax.set_xlim(*XL); ax.set_ylim(-0.04, 1.04); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.text(0.03, 0.96, tag, transform=ax.transAxes, ha='left', va='top', fontsize=8)
    ax.grid(alpha=0.25, lw=0.4)
    return s

prob_panel(a, x1, f1, l1, n0 >= 1, np.ones_like(m2), r'(peak charge $-$ threshold) / $\sigma_r$', 'probability of a first hit', '(a)')
a.text(2.1, 0.30, f"{num['first_differ_0p1']} pixels differ\nby more than 0.1\n({num['first_above_4sig']} pixels\nabove $4\\sigma_r$ not shown)",
       fontsize=6, color=CG, ha='left', va='center')
prob_panel(bb, x2, f2, l2, n0 >= 2, m2, r'(peak charge after reset $-$ threshold) / $\sigma_r$', 'probability of a second hit', '(b)')
bb.text(2.1, 0.30, f"{num['second_differ_0p1']} pixels differ\nby more than 0.1", fontsize=6, color=CG, ha='left', va='center')

c.axhline(0, color='k', lw=0.6)
c.errorbar(T, rel, yerr=err, color=C0, marker='o', ms=3, lw=1.1, capsize=0)
c.axvline(thr / 1e3, color=CG, ls=':', lw=0.8)
i5 = int(np.argmin(np.abs(T - thr / 1e3)))
c.plot(T[i5], rel[i5], 'o', ms=6, mfc='none', mec='k', mew=0.7)
c.text(T[i5] - 0.15, rel[i5] + 0.9, f'{rel[i5]:+.1f}%', ha='right', va='bottom', fontsize=6.3)
c.text(thr / 1e3 - 0.12, -9.6, 'nominal threshold', ha='right', rotation=90, va='bottom', fontsize=6, color=CG)
c.text(0.97, 0.96, 'more hits without noise', transform=c.transAxes, ha='right', va='top', fontsize=6, color=CG)
c.text(0.97, 0.04, 'fewer hits without noise', transform=c.transAxes, ha='right', va='bottom', fontsize=6, color=CG,
       bbox=dict(fc='white', ec='none', pad=0.5))
c.set_xlabel(r'threshold (ke$^-$)'); c.set_ylabel('hits without noise / average $-$ 1 (%)')
c.set_xlim(2.7, 10.3); c.set_ylim(-10, 7); c.grid(alpha=0.25, lw=0.4)
c.text(0.03, 0.96, '(c)', transform=c.transAxes, ha='left', va='top', fontsize=8)
h, l = a.get_legend_handles_labels()
fig.legend(h, l, loc='upper center', ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.0), handletextpad=0.3, columnspacing=1.5, markerscale=1.4)
fig.tight_layout(w_pad=1.2, rect=(0, 0, 1, 0.93))
fig.savefig(OUT + '.pdf'); fig.savefig(OUT + '.png', dpi=200)
json.dump(num, open(OUT + '_numbers.json', 'w'), indent=1)
print('wrote', OUT)
