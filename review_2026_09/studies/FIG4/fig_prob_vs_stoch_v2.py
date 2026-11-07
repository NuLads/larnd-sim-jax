"""Fig. prob_vs_stoch, version 2 (19 Oct 2026, Pierre: overlay the differences due to noise seeds).
Per pixel and hit (first, second): the probabilistic model minus the mean of 1e4 stochastic throws (as in v1,
fig_prob_vs_stoch.py, cache SAMPLING_STEP/prob_vs_stoch_cache.npz), overlaid with the difference between two single
stochastic throws with different noise seeds (seed_pairs.py, 50 pairs, FIG4/seed_pairs_50.npz): the scale on which two
equally valid noisy readouts of the same waveforms differ.
(a) hit probability: model lambda minus sampled frequency / hit present in seed A minus seed B (-1, 0, +1);
(b) hit time (ticks) and (c) hit charge (ADC counts): model expectation minus sampled mean / seed A minus seed B, for
pixels with >= 200 sampled hits (and, for the seeds, the hit present in both).
usage: python fig_prob_vs_stoch_v2.py [out_stem]"""
import sys, json
import numpy as np
D = '/sdf/scratch/neutrino/pgranger/review_2026_09/'
OUT = sys.argv[1] if len(sys.argv) > 1 else D + 'FIG4/probabilistic_vs_stochastic_v2'
z = np.load(D + 'SAMPLING_STEP/prob_vs_stoch_cache.npz'); N = float(z['N'])
s = np.load(D + 'FIG4/seed_pairs_50.npz'); M = int(s['M']); A, B = slice(0, M), slice(M, 2 * M)
f = z['cnt'] / N
mt = z['st'] / np.maximum(z['cnt'], 1); ma = z['sa'] / np.maximum(z['cnt'], 1)
dl = z['lam'] - f; dt = z['Et'] - mt; da = z['Eadc'] - ma
MINC = 200
num = {}
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D; from matplotlib.patches import Patch
plt.rcParams.update({'font.size': 8, 'axes.labelsize': 8, 'legend.fontsize': 6.5, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
                     'axes.linewidth': 0.6, 'font.family': 'sans-serif'})
C = ['#0072B2', '#D55E00']
fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.55));
spec = [('lambda', dl, np.linspace(-1.1, 1.1, 89), r'$\Delta$ hit probability'),
        ('time', dt, np.arange(-30.5, 30.6, 1.0), r'$\Delta$ hit time (ticks)'),
        ('adc', da, np.arange(-12.25, 12.3, 0.5), r'$\Delta$ hit charge (ADC counts)')]
for j, (name, res, bins, xl) in enumerate(spec):
    ax = axs[j]; lim = bins[-1]
    for n in (0, 1):
        m_mod = (z['cnt'][:, n] >= (1 if j == 0 else MINC)) | ((j == 0) & (z['lam'][:, n] > 1e-3))
        r_mod = res[m_mod, n]
        if j == 0:
            r_seed = (s['hit'][A, :, n].astype(float) - s['hit'][B, :, n].astype(float))[:, m_mod].ravel()
        else:
            both = (z['cnt'][:, n] >= MINC)[None, :] & s['hit'][A, :, n] & s['hit'][B, :, n]
            key = 'tick' if name == 'time' else 'adc'
            r_seed = (s[key][A, :, n] - s[key][B, :, n]).astype(float)[both]
        ax.hist(np.clip(r_seed, -lim, lim - 1e-9), bins=bins, color=C[n], alpha=0.22, weights=np.full(len(r_seed), 1 / M), lw=0)
        ax.hist(np.clip(r_mod, -lim, lim - 1e-9), bins=bins, histtype='step', color=C[n], lw=1.1)
        q16, q84 = np.percentile(r_mod, [16, 84]); s16, s84 = np.percentile(r_seed, [16, 84])
        num[f'{name}_{n}'] = dict(model_half68=float(0.5 * (q84 - q16)), model_median=float(np.median(r_mod)),
                                  seeds_half68=float(0.5 * (s84 - s16)), seeds_frac_nonzero=float(np.mean(r_seed != 0)),
                                  n_model=int(len(r_mod)), n_seed_per_pair=float(len(r_seed) / M))
    ax.set_yscale('log'); ax.set_ylabel('pixels'); ax.set_xlabel(xl); ax.set_xlim(bins[0], bins[-1])
    ax.set_ylim(0.1, 4e3); ax.grid(alpha=0.25, lw=0.5)
    ax.set_title(['(a) hit probability', '(b) hit time', '(c) hit charge'][j], fontsize=8)
a0, a1, a2 = axs
a0.text(0.5, 0.97, f"two seeds disagree on\n{100 * num['lambda_0']['seeds_frac_nonzero']:.0f}% (first) and "
        f"{100 * num['lambda_1']['seeds_frac_nonzero']:.0f}% (second hit)\nof the pixels", transform=a0.transAxes,
        ha='center', va='top', fontsize=5.8, color='0.3')
a1.text(0.03, 0.97, f"two seeds: 68% within\n$\\pm${num['time_0']['seeds_half68']:.0f} ticks (first hit)", transform=a1.transAxes,
        ha='left', va='top', fontsize=5.8, color='0.3')
a1.text(0.97, 0.97, f"model: within\n$\\pm${num['time_0']['model_half68']:.2f} ticks", transform=a1.transAxes, ha='right', va='top', fontsize=5.8, color='0.3')
a2.text(0.03, 0.97, f"two seeds: 68% within\n$\\pm${num['adc_0']['seeds_half68']:.1f} counts (first hit)", transform=a2.transAxes,
        ha='left', va='top', fontsize=5.8, color='0.3')
a2.text(0.97, 0.97, f"model: within\n$\\pm${num['adc_0']['model_half68']:.2f} counts", transform=a2.transAxes, ha='right', va='top', fontsize=5.8, color='0.3')
handles = [Line2D([], [], color=C[0], lw=1.1, label='first hit'), Line2D([], [], color=C[1], lw=1.1, label='second hit'),
           Line2D([], [], color='k', lw=1.1, label=r'model $-$ mean of $10^4$ throws'),
           Patch(color='k', alpha=0.22, lw=0, label='seed A $-$ seed B (single throws)')]
fig.legend(handles=handles, loc='upper center', ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.0), handlelength=1.5, columnspacing=1.4)
fig.tight_layout(rect=(0, 0, 1, 0.92), w_pad=1.0)
fig.savefig(OUT + '.pdf'); fig.savefig(OUT + '.png', dpi=200)
json.dump(num, open(OUT + '_numbers.json', 'w'), indent=1); print(json.dumps(num, indent=1)); print('wrote', OUT)
