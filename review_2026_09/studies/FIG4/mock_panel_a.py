import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
D = '/sdf/scratch/neutrino/pgranger/review_2026_09/'
z = np.load(D + 'SAMPLING_STEP/prob_vs_stoch_cache.npz'); N = float(z['N']); s = np.load(D + 'FIG4/seed_pairs_50.npz'); M = int(s['M'])
f = z['cnt'] / N; dl = z['lam'] - f; lam = z['lam']
C = ['#0072B2', '#D55E00']
plt.rcParams.update({'font.size': 8, 'axes.labelsize': 8, 'legend.fontsize': 6.3, 'xtick.labelsize': 7, 'ytick.labelsize': 7, 'axes.linewidth': 0.6})
fig, (a1, a2) = plt.subplots(1, 2, figsize=(5.0, 2.3))
# option 1: per pixel, difference against the hit probability, with the seed-to-seed band
x = np.linspace(0, 1, 400); band = np.sqrt(2 * x * (1 - x))
a1.fill_between(x, -band, band, color='0.6', alpha=0.3, lw=0, label='seed A $-$ seed B ($\\pm1\\sigma$)')
for n in (0, 1):
    m = (z['cnt'][:, n] >= 1) | (lam[:, n] > 1e-3)
    a1.plot(lam[m, n], dl[m, n], 'o', ms=1.8, color=C[n], alpha=0.8, mew=0, label=['first hit', 'second hit'][n] + r': model $-$ mean')
a1.set_xlim(0, 1); a1.set_ylim(-0.8, 0.8); a1.set_xlabel('hit probability'); a1.set_ylabel(r'$\Delta$ hit probability')
a1.set_title('option 1: per pixel', fontsize=8); a1.legend(loc='lower center', frameon=False, fontsize=5.6); a1.grid(alpha=0.25, lw=0.5)
# option 2: averaged over the pixels, model error vs seed disagreement
A, B = slice(0, M), slice(M, 2 * M); w = 0.35
for n in (0, 1):
    m = (z['cnt'][:, n] >= 1) | (lam[:, n] > 1e-3)
    seed = float((s['hit'][A, :, n] != s['hit'][B, :, n])[:, m].mean()); mod = float(np.abs(dl[m, n]).mean())
    a2.bar(n - w / 2, seed, w, color=C[n], alpha=0.3); a2.bar(n + w / 2, mod, w, fill=False, ec=C[n], lw=1.1)
    a2.text(n - w / 2, seed * 1.15, f'{100 * seed:.1f}%', ha='center', fontsize=6); a2.text(n + w / 2, mod * 1.15, f'{100 * mod:.2f}%', ha='center', fontsize=6)
a2.set_yscale('log'); a2.set_ylim(1e-4, 0.3); a2.set_xticks([0, 1]); a2.set_xticklabels(['first hit', 'second hit'])
a2.set_ylabel('per pixel, averaged'); a2.set_title('option 2: averaged', fontsize=8)
a2.text(0.5, 0.93, 'shaded: two seeds disagree\noutline: model $-$ mean, |$\\Delta$ prob.|', transform=a2.transAxes, ha='center', va='top', fontsize=5.6)
fig.tight_layout(); fig.savefig(D + 'FIG4/mock_panel_a.png', dpi=200)
