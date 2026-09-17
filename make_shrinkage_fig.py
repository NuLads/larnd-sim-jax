"""Fig 94 — where "the data wants 66% of the true spread" comes from, and why it is the wrong
target for CALIBRATION.

THE CLAIM (report §6s, fig 52): the likelihood-only optimum of the fit->truth alpha-scan sits at
alpha* = 0.50, a field whose width is 66% of truth. Stated justification: "with finite hits per
segment the maximum-likelihood field is itself a shrunk version of truth."

THAT JUSTIFICATION IS BACKWARDS, and this figure measures it. A noisy per-segment estimate is
OVER-dispersed, not shrunk: Var(estimate) = Var(truth) + Var(noise). Decomposing the fitted field
(602k core segments, 16 replicas) gives per-segment noise sd 0.102 against a true spread of 0.078
-- SNR 0.77, i.e. each segment's dE/dx is individually measured worse than it varies -- so the
unshrunk ML field would sit at ~165% of the true width.

What is shrunk is the MAP/posterior-mean estimate, and the fit is doing it near-optimally
(measured slope 0.46 vs the Bayes-optimal 0.37 for this SNR). That is correct for minimising
per-segment MSE. It is NOT correct for calibration: DXTRU (fig 93) pins the field at 100% and buys
-10.4 +- 3.3 points of lifetime bias while the loss gets WORSE. The shrinkage bias is coherent --
every segment pulled toward the same centre in proportion to its deviation -- and a coherent,
drift-correlated distortion of the charge field is exactly what lifetime absorbs.

The likelihood prefers <100% for a second reason the scan cannot separate: the fitted field
absorbed THIS data realisation's noise, so partially retaining it scores well IN SAMPLE. The
report's own caution ("pushing the field there would be fitting the noise") was right; the
"some shrinkage is correct" sentence beside it was not.
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

F, T = [], []
for s in range(1, 17):
    fs = glob.glob(f'fit_result/sci_full_DSREP{s}/history_iter10000_*.pkl')
    if not fs:
        continue
    h = pickle.load(open(fs[0], 'rb'))
    dc, tc = h['dedx_cache'], h['true_dedx_matched_cache']
    for k in dc:
        if k not in tc:
            continue
        f = np.exp(np.ravel(dc[k]['log_dedx']))
        t = np.ravel(tc[k])
        n = min(len(f), len(t))
        F.append(f[:n]); T.append(t[:n])
F, T = np.concatenate(F), np.concatenate(T)
m = ((F > np.percentile(F, 2)) & (F < np.percentile(F, 98)) &
     (T > np.percentile(T, 2)) & (T < np.percentile(T, 98)))
Fc, Tc = F[m], T[m]
lam = np.cov(Fc, Tc)[0, 1] / np.var(Tc)
vt = np.var(Tc); vn = np.var(Fc) / lam ** 2 - vt
snr = np.sqrt(vt / vn)
ml_frac = np.sqrt(1 + vn / vt)
lam_opt = vt / (vt + vn)


def hw(v):
    return (np.percentile(v, 84) - np.percentile(v, 16)) / 2


fit_frac = hw(F) / hw(T)

fig, ax = plt.subplots(1, 2, figsize=(12.6, 4.4))

# (a) the ladder of widths
a = ax[0]
items = [('unshrunk ML\n(what the data alone gives)', 100 * ml_frac, RED),
         ('TRUTH', 100.0, GREEN),
         ('our fitted field\n(MAP shrinkage)', 100 * fit_frac, BLUE),
         ("the α-scan's\nlikelihood optimum", 66.0, ORANGE)]
items.sort(key=lambda x: x[1])
y = np.arange(len(items))
a.barh(y, [v for _, v, _ in items], color=[c for _, _, c in items])
a.set_yticks(y, [l for l, _, _ in items], fontsize=8.5)
a.axvline(100, color=GREEN, lw=2)
for i, (_, v, c) in enumerate(items):
    a.annotate(f'{v:.0f}%', (v, i), textcoords='offset points', xytext=(6, -3),
               fontsize=9, color=c)
a.set_xlabel('width of the per-segment dE/dx field (% of truth)')
a.set_title('(a) the ML field is OVER-dispersed, not shrunk —\n'
            'so "ML is itself shrunk" cannot justify aiming below 100%', fontsize=9.5)
a.set_xlim(0, 200)

# (b) why shrinkage is right for MSE and wrong for calibration
a = ax[1]
lams = np.linspace(0.05, 1.3, 300)
mse = (lams - 1) ** 2 * vt + lams ** 2 * vn        # per-segment MSE vs shrinkage
mse /= mse.min()
a.plot(lams, mse, color=BLUE, lw=2, label='per-segment dE/dx MSE (arb.)')
a.axvline(lam_opt, color=BLUE, lw=1.4, ls='--')
a.annotate(f'MSE-optimal\nλ = {lam_opt:.2f}', (lam_opt, 2.6), color=BLUE, fontsize=8.5,
           ha='center')
a.axvline(lam, color=INK, lw=1.6)
a.annotate(f'our fit\nλ = {lam:.2f}', (lam, 3.6), color=INK, fontsize=8.5, ha='center')
a.axvline(1.0, color=GREEN, lw=2)
a.annotate('no shrinkage\n(λ = 1) — what DXTRU\npins, −10.4 pts of\nlifetime bias',
           (1.02, 1.6), color=GREEN, fontsize=8.5)
a.set_xlabel('shrinkage factor λ applied to the per-segment field')
a.set_ylabel('per-segment MSE (normalised)')
a.set_ylim(0.9, 4.5)
a.set_title('(b) the fit is near the MSE optimum — and that is the\n'
            'WRONG objective: calibration wants λ = 1', fontsize=9.5)
a.legend(frameon=False, fontsize=8.5, loc='upper right')

fig.suptitle(f'Fig 94 — why "the data wants 66%" is not a target. Per-segment SNR is only {snr:.2f} '
             f'(noise sd {np.sqrt(vn):.3f} vs true spread {np.sqrt(vt):.3f}),\n'
             f'so the unshrunk ML field sits at {100*ml_frac:.0f}% of the true width and MAP shrinkage pulls it to '
             f'{100*fit_frac:.0f}%. That minimises per-segment MSE,\n'
             'but shrinkage bias is COHERENT across segments — and a coherent charge distortion is exactly what lifetime absorbs.',
             fontsize=10.5, y=1.08)
fig.tight_layout()
fig.savefig(f'{OUT}/fig94_shrinkage_logic.png')
print(f'wrote fig94 | SNR {snr:.2f} | ML width {100*ml_frac:.0f}% | fitted {100*fit_frac:.0f}% '
      f'| lam {lam:.3f} vs optimal {lam_opt:.3f}')
