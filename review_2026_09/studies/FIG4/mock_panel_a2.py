import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
D = '/sdf/scratch/neutrino/pgranger/review_2026_09/'
z = np.load(D + 'SAMPLING_STEP/prob_vs_stoch_cache.npz'); N = float(z['N']); s = np.load(D + 'FIG4/seed_pairs_50.npz'); M = int(s['M'])
f = z['cnt'] / N; dl = z['lam'] - f
C = ['#0072B2', '#D55E00']
plt.rcParams.update({'font.size': 8, 'axes.labelsize': 8, 'legend.fontsize': 6.0, 'xtick.labelsize': 7, 'ytick.labelsize': 7, 'axes.linewidth': 0.6})
fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(7.0, 2.4))
A, B = slice(0, M), slice(M, 2 * M)
for n in (0, 1):
    m = (z['cnt'][:, n] >= 1) | (z['lam'][:, n] > 1e-3)
    pdis = 2 * f[m, n] * (1 - f[m, n]); err = np.abs(dl[m, n])
    emp = (s['hit'][A, :, n] != s['hit'][B, :, n])[:, m].mean(0)
    if n == 0: print('check 2f(1-f) vs 50-pair rate: mean', pdis.mean(), emp.mean())
    k = pdis > 0
    # (ii) per-pixel log-log scatter
    a1.loglog(np.clip(pdis[k], 1e-5, 1), np.clip(err[k], 1e-7, 1), 'o', ms=1.8, mew=0, color=C[n], alpha=0.7)
    # (iii) log-x histograms
    bins = np.logspace(-7, 0, 36)
    a2.hist(np.clip(err, 1e-7, 1), bins=bins, histtype='step', color=C[n], lw=1.1)
    a2.hist(pdis[k], bins=bins, color=C[n], alpha=0.25)
    # (i) ratio to the seed spread
    sig = np.sqrt(pdis[k]); a3.hist(np.clip(err[k] / sig, 1e-6, 10), bins=np.logspace(-6, 1, 36), histtype='step', color=C[n], lw=1.1)
x = np.logspace(-5, 0, 10); a1.plot(x, x, 'k--', lw=0.7); a1.text(2e-2, 4e-2, 'error = noise', rotation=33, fontsize=6)
a1.set_xlabel('P(two seeds disagree)'); a1.set_ylabel(r'|model $-$ mean|'); a1.set_title('(ii) per pixel', fontsize=8)
a2.set_xscale('log'); a2.set_yscale('log'); a2.set_xlabel('probability'); a2.set_ylabel('pixels')
a2.set_title('(iii) filled: seeds disagree\nline: |model $-$ mean|', fontsize=7)
a3.set_xscale('log'); a3.axvline(1, color='k', ls='--', lw=0.7); a3.set_xlabel(r'|model $-$ mean| / seed spread'); a3.set_ylabel('pixels')
a3.set_title('(i) error in units of the noise', fontsize=8)
for a in (a1, a2, a3): a.grid(alpha=0.25, lw=0.5)
fig.tight_layout(); fig.savefig(D + 'FIG4/mock_panel_a2.png', dpi=200)
