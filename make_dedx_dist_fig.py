"""Fig 90 — true vs fitted per-segment dE/dx, the fig-48 diagnostics redone on the REFIT prior.

Same four views that convicted the old w=5 prior (fig 48 panels a, c, f, d), now on the 16 cold
replicas that run the BT2 refit prior. Direct before/after, all from checkpoints on disk.

  (a) the distributions: fitted vs matched truth, with the prior's own density overlaid
  (b) regression to the prior centre: fitted vs true per segment. The old prior gave slope 0.25;
      a perfect field gives 1.0.
  (c) Q-Q: is the shrinkage uniform across the range, or a tail artefact?
  (d) per-batch shrinkage, all 16 replicas: the old prior under-dispersed 100% of 900 batches at
      a median ratio 0.33.
"""
import glob, pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = 'plots/noise_report'
BLUE, ORANGE, GREEN, RED, GREY, INK = ('#0072B2', '#E69F00', '#009E73', '#D55E00',
                                       '#888888', '#222222')
plt.rcParams.update({'font.size': 9.5, 'axes.grid': True, 'grid.alpha': .25,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.dpi': 130, 'savefig.bbox': 'tight'})

# refit (production) prior parameters
NU_L, NU_R, LOC, S_L, S_R = 21.760, 4.720, 1.8786, 0.0782, 0.0801


def split_t(x):
    """Unnormalised split-student-t density with the production parameters."""
    z = (x - LOC) / np.where(x < LOC, S_L, S_R)
    nu = np.where(x < LOC, NU_L, NU_R)
    s = np.where(x < LOC, S_L, S_R)
    return (1 + z ** 2 / nu) ** (-(nu + 1) / 2) / s


F_all, T_all, ratios = [], [], []
per_seed = {}
for s in range(1, 17):
    fs = glob.glob(f'fit_result/sci_full_DSREP{s}/history_iter10000_*.pkl')
    if not fs:
        continue
    h = pickle.load(open(fs[0], 'rb'))
    dc, tc = h['dedx_cache'], h['true_dedx_matched_cache']
    F, T = [], []
    for k in dc:
        if k not in tc:
            continue
        f = np.exp(np.ravel(dc[k]['log_dedx']))
        t = np.ravel(tc[k])
        n = min(len(f), len(t))
        f, t = f[:n], t[:n]
        F.append(f); T.append(t)
        hwf = (np.percentile(f, 84) - np.percentile(f, 16)) / 2
        hwt = (np.percentile(t, 84) - np.percentile(t, 16)) / 2
        ratios.append(hwf / hwt)
    F, T = np.concatenate(F), np.concatenate(T)
    per_seed[s] = (F, T)
    F_all.append(F); T_all.append(T)
F_all, T_all = np.concatenate(F_all), np.concatenate(T_all)
ratios = np.array(ratios)


def hw(v):
    return (np.percentile(v, 84) - np.percentile(v, 16)) / 2


fig, ax = plt.subplots(1, 4, figsize=(17, 4.2))

# (a) distributions
a = ax[0]
bins = np.linspace(1.55, 2.25, 90)
a.hist(T_all, bins=bins, density=True, color=GREEN, alpha=.55,
       label=f'TRUE (matched)   width {hw(T_all):.4f}')
a.hist(F_all, bins=bins, density=True, histtype='step', lw=1.8, color=BLUE,
       label=f'FITTED   width {hw(F_all):.4f}')
xx = np.linspace(1.3, 2.5, 20001)
p = split_t(xx); p /= np.trapezoid(p, xx)
cdf = np.cumsum(p) * (xx[1] - xx[0])
q16, q84 = np.interp([0.16, 0.84], cdf, xx)
prior_hw = (q84 - q16) / 2
a.plot(xx, p, color=ORANGE, lw=1.8, ls='--',
       label=f'the PRIOR (refit, β=1)   width {prior_hw:.4f}')
a.set_xlabel('per-segment dE/dx (MeV/cm)')
a.set_ylabel('density')
a.set_title(f'(a) the PRIOR now matches TRUTH ({prior_hw:.4f} vs {hw(T_all):.4f});\n'
            f'the fitted field is narrower ({hw(F_all)/hw(T_all):.0%}) — that is the LIKELIHOOD shrinking it', fontsize=9.5)
a.legend(frameon=False, fontsize=7.6, loc='upper right')

# (b) regression to the prior centre
a = ax[1]
rng = np.random.default_rng(1)
idx = rng.choice(len(F_all), 6000, replace=False)
a.scatter(T_all[idx], F_all[idx], s=3, alpha=.25, color=BLUE, lw=0)
lo, hi = 1.6, 2.2
a.plot([lo, hi], [lo, hi], '--', color=INK, lw=1.4, label='no shrinkage (identity)')
b1, b0 = np.polyfit(T_all, F_all, 1)
a.plot([lo, hi], [b1 * lo + b0, b1 * hi + b0], color=RED, lw=2,
       label=f'fit: slope {b1:.2f}   (old prior: 0.25)')
a.axhline(LOC, color=ORANGE, lw=1.2, ls=':', label=f'prior centre {LOC}')
a.set_xlim(lo, hi); a.set_ylim(lo, hi)
a.set_xlabel('TRUE per-segment dE/dx')
a.set_ylabel('FITTED per-segment dE/dx')
a.set_title(f'(b) regression toward the prior centre:\nslope {b1:.2f} vs 0.25 with the old prior', fontsize=9.5)
a.legend(frameon=False, fontsize=7.6, loc='upper left')

# (c) Q-Q
a = ax[2]
q = np.linspace(1, 99, 99)
qt, qf = np.percentile(T_all, q), np.percentile(F_all, q)
a.plot(qt, qf, 'o-', color=BLUE, ms=3, lw=1.2, label='quantile–quantile')
a.plot([lo, hi], [lo, hi], '--', color=INK, lw=1.4, label='identity')
sl, ic = np.polyfit(qt, qf, 1)
a.plot([lo, hi], [sl * lo + ic, sl * hi + ic], color=RED, lw=1.6,
       label=f'slope {sl:.2f}   (old prior: 0.36)')
a.set_xlim(lo, hi); a.set_ylim(lo, hi)
a.set_xlabel('TRUE quantiles')
a.set_ylabel('FITTED quantiles')
a.set_title(f'(c) the compression is uniform, not a tail effect:\nQ-Q slope {sl:.2f}', fontsize=9.5)
a.legend(frameon=False, fontsize=7.6, loc='upper left')

# (d) per-batch shrinkage
a = ax[3]
a.hist(ratios, bins=40, color=BLUE, alpha=.8)
a.axvline(1.0, color=INK, lw=1.8, label='no shrinkage')
a.axvline(np.median(ratios), color=RED, lw=1.8, ls='--',
          label=f'median {np.median(ratios):.2f}   (old prior: 0.33)')
a.axvline(0.66, color=GREEN, lw=1.6, ls=':', label="data's optimum 0.66")
a.set_xlabel('width(fitted) / width(true), per batch')
a.set_ylabel('batches')
a.set_title(f'(d) per-batch, all {len(per_seed)} replicas ({len(ratios):,} batches):\n'
            f'{100*np.mean(ratios<1):.0f}% under-dispersed (old prior: 100%)', fontsize=9.5)
a.legend(frameon=False, fontsize=7.6, loc='upper left')

fig.suptitle('Fig 90 — true vs fitted per-segment dE/dx with the REFIT prior (16 cold replicas), the fig-48 diagnostics redone.\n'
             f'Every measure that convicted the old w=5 prior improved: width {hw(F_all)/hw(T_all):.0%} of truth (was 32%), '
             f'regression slope {b1:.2f} (was 0.25), Q-Q slope {sl:.2f} (was 0.36), {100*np.mean(ratios<1):.0f}% under-dispersed batches (was 100%).\n'
             'AND THE DIAGNOSIS HAS CHANGED: the prior density now sits on top of truth (a), so the residual shrinkage is no longer the prior being\n'
             'mis-calibrated — it is the ordinary MAP shrinkage of a finite-information likelihood, landing at 71% against the 66% the likelihood alone wants.',
             fontsize=10.5, y=1.16)
fig.tight_layout()
fig.savefig(f'{OUT}/fig90_dedx_distributions.png')
print(f'wrote fig90 | n={len(F_all):,} segments, {len(ratios)} batches')
print(f'  width fitted {hw(F_all):.4f} / true {hw(T_all):.4f} = {hw(F_all)/hw(T_all):.4f}')
print(f'  mean fitted {F_all.mean():.4f} / true {T_all.mean():.4f} = {100*(F_all.mean()/T_all.mean()-1):+.3f}%')
print(f'  regression slope {b1:.4f} | Q-Q slope {sl:.4f} | per-batch median ratio {np.median(ratios):.4f}')
print(f'  under-dispersed batches: {100*np.mean(ratios<1):.1f}%')
