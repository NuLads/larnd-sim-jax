"""Fig 52 (study 2, α-scan): where along fit→truth does each term want the dE/dx field?

The per-segment field is walked along the straight line from the FITTED values to the TRUE ones,
in LOG space (the coordinate the block is fitted in), with calibration and geometry held at the fit:

    log_dedx(α) = log_fit + α (log_true − log_fit)

α = 0 is the fit, α = 1 is truth, α > 1 over-shoots. 2 seeds × 100 batches × 9 values of α, no
fitting. Because `dedx_prior` enters the total linearly in its weight, one scan re-scores offline at
ANY prior weight -- panel (c) is a weight sweep with no extra GPU passes.

THIS IS THE MEASUREMENT figs 46-48 inferred. Those said the prior over-shrinks the field; this asks
the objective directly where it wants the dispersion, and gets a number.

The answer is not the one the earlier figures implied. The DATA's own optimum is α ≈ 0.5 -- not 1.
That is expected once stated: with finite hits per segment the maximum-likelihood field is itself a
shrunk version of truth, because the per-segment values are individually noisy. Shrinkage is
correct; the question is how much, and in which direction the prior moves it.
"""
import glob, os, pickle, re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = 'plots/noise_report'
RUN = 'fit_result/sci_full_TPA'
C = dict(blue='#0072B2', orange='#E69F00', green='#009E73', red='#D55E00',
         purple='#CC79A7', grey='#666666', ink='#222222')
plt.rcParams.update({'font.size': 9, 'axes.grid': True, 'grid.alpha': .25,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.dpi': 130, 'savefig.bbox': 'tight'})
LIK = ['log_likelihood_tick', 'no_match_penalty', 'expected_total_hits']
PRI = ['dedx_prior', 'dedx_mean_penalty', 'dedx_barrier', 'mcs_prior',
       'chain_drift_penalty', 'dedx_drift_penalty', 'spatial_moment']
AL = np.array([0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0])
KEY = lambda a: f'A{a:g}'
DX = 0.9913                       # median per-segment dx, so beta = w * dx
W_PROD, W_BETA1, W_DEF = 5.0, 1.0088, 0.5

tot, lik, pri, dpr, seed = [], [], [], [], []
for f in sorted(glob.glob(RUN + '/history_iter*.pkl')):
    s = int(re.search(r'seed(\d+)', f).group(1))
    for rec in pickle.load(open(f, 'rb'))['truthpoint']:
        if not all(KEY(a) in rec['points'] for a in AL):
            continue
        P = rec['points']
        tot.append([P[KEY(a)]['loss'] for a in AL])
        lik.append([sum(float(P[KEY(a)]['aux'].get(k, 0.)) for k in LIK) for a in AL])
        pri.append([sum(float(P[KEY(a)]['aux'].get(k, 0.)) for k in PRI) for a in AL])
        dpr.append([float(P[KEY(a)]['aux'].get('dedx_prior', 0.)) for a in AL])
        seed.append(s)
tot, lik, pri, dpr = (np.array(x) for x in (tot, lik, pri, dpr))
seed = np.array(seed)
N = len(tot)
print(f'{N} batch scans, seeds {sorted(set(seed))}, alpha grid {list(AL)}')


def astar(y, lo=0., hi=2.):
    """Parabola through the three grid points bracketing the discrete minimum."""
    i = int(np.argmin(y)); i = min(max(i, 1), len(y) - 2)
    c = np.polyfit(AL[i - 1:i + 2], y[i - 1:i + 2], 2)
    return float(np.clip(-c[1] / (2 * c[0]), lo, hi))


L, T, D, OTH = lik.sum(0), tot.sum(0), dpr.sum(0), (pri - dpr).sum(0)
A_LIK, A_TOT = astar(L), astar(T)
print(f'likelihood-only optimum  alpha* = {A_LIK:.3f}')
print(f'total (w=5)     optimum  alpha* = {A_TOT:.3f}')

fig, ax = plt.subplots(2, 3, figsize=(16.4, 8.8))

# ---------------- (a) the three curves ----------------
a = ax[0, 0]
for y, col, lab in [(T, C['ink'], 'TOTAL objective (w = 5)'),
                    (L, C['blue'], 'LIKELIHOOD only'),
                    (pri.sum(0), C['orange'], 'PRIORS only')]:
    a.plot(AL, y - y.min(), color=col, lw=2.4, marker='o', ms=5, label=lab)
a.axvline(0, color=C['grey'], lw=1.2); a.axvline(1, color=C['red'], lw=1.6, ls='--')
a.annotate('fit', (0, 0), textcoords='offset points', xytext=(4, 6), fontsize=7.6, color=C['grey'])
a.annotate('truth', (1, 0), textcoords='offset points', xytext=(4, 6), fontsize=7.6, color=C['red'])
a.set_yscale('symlog', linthresh=100)
a.set_xlabel('α   (0 = fit, 1 = truth)'); a.set_ylabel('Δ from each curve’s own minimum')
a.set_title('(a) the priors fall monotonically toward the fit;\nthe likelihood has a minimum in between',
            fontsize=9.3)
a.legend(fontsize=7.4, frameon=False)

# ---------------- (b) the likelihood alone ----------------
a = ax[0, 1]
a.plot(AL, L - L.min(), color=C['blue'], lw=2.6, marker='o', ms=6)
xf = np.linspace(0, 1.2, 200)
i = int(np.argmin(L)); i = min(max(i, 1), len(L) - 2)
c = np.polyfit(AL[i - 1:i + 2], L[i - 1:i + 2], 2)
a.plot(xf, np.polyval(c, xf) - L.min(), color=C['grey'], lw=1.4, ls=':', label='local parabola')
a.axvline(A_LIK, color=C['green'], lw=2.4, label=f'DATA’s optimum  α* = {A_LIK:.2f}')
a.axvline(0, color=C['ink'], lw=1.8, label=f'the fit sits here (ΔNLL {L[0]-L.min():.0f})')
a.axvline(1, color=C['red'], lw=1.8, ls='--', label=f'truth (ΔNLL {L[5]-L.min():.0f})')
a.set_xlim(-.05, 1.3)
a.set_ylim(-300, max(L[0], L[5]) - L.min() + 2500)
a.set_xlabel('α'); a.set_ylabel('Δ likelihood NLL (summed over 200 batches)')
a.set_title('(b) the data wants HALF WAY. The fit (α=0) and truth\n'
            '(α=1) are almost equally disfavoured by the likelihood.', fontsize=9.3)
a.legend(fontsize=7.0, frameon=False)

# ---------------- (c) the weight sweep — the decisive panel ----------------
a = ax[0, 2]
WS = np.concatenate([[0.0], np.logspace(-1.6, 0.9, 60)])
AS = np.array([astar(L + OTH + (w / W_PROD) * D) for w in WS])
a.plot(np.maximum(WS, 1e-2), AS, color=C['ink'], lw=2.6)
a.axhline(A_LIK, color=C['green'], lw=2.2, label=f'the DATA’s optimum  α* = {A_LIK:.2f}')
for w, col, lab in [(W_PROD, C['red'], 'production w = 5'),
                    (W_BETA1, C['blue'], 'β = 1  (w = 1/dx)'),
                    (W_DEF, C['purple'], 'script default 0.5')]:
    av = astar(L + OTH + (w / W_PROD) * D)
    a.axvline(w, color=col, lw=1.8, ls=':' if w != W_PROD else '-')
    a.plot([w], [av], marker='o', ms=9, color=col, zorder=5)
    a.annotate(f'  {lab}\n  α* = {av:.2f}', (w, av), fontsize=7.0, color=col, va='top')
a.set_xscale('log')
a.set_xlabel('dE/dx prior weight (re-scored, not re-fitted)')
a.set_ylabel('α the OBJECTIVE prefers')
a.set_ylim(-0.05, 0.62)
a.set_title('(c) no weight of THIS prior reaches the data’s optimum —\n'
            'it is narrower than the data wants, so it only pulls DOWN', fontsize=9.3)
a.legend(fontsize=7.4, frameon=False, loc='lower left')

# ---------------- (d) per-batch spread of the data's optimum ----------------
a = ax[1, 0]
pb = np.array([astar(lik[k]) for k in range(N)])
a.hist(pb, bins=28, color=C['blue'], edgecolor='white', linewidth=.5)
a.axvline(np.median(pb), color=C['green'], lw=2.4, label=f'median {np.median(pb):.2f}')
a.axvline(0, color=C['ink'], lw=1.8, label='the fit')
a.axvline(1, color=C['red'], lw=1.8, ls='--', label='truth')
a.set_xlabel('per-batch likelihood-optimal α'); a.set_ylabel('batches')
a.set_title(f'(d) and it is not an average: {100*(pb>0.05).mean():.0f}% of batches want MORE\n'
            f'dispersion than the fit, {100*(pb<0.95).mean():.0f}% want less than truth', fontsize=9.3)
a.legend(fontsize=7.4, frameon=False)

# ---------------- (e) term by term along alpha ----------------
a = ax[1, 1]
for k, col, lab in [('log_likelihood_tick', C['blue'], 'joint hit log-intensity'),
                    ('no_match_penalty', C['red'], 'unmatched-target penalty'),
                    ('expected_total_hits', C['green'], 'expected hit count'),
                    ('dedx_prior', C['orange'], 'dE/dx student-t prior'),
                    ('dedx_mean_penalty', C['grey'], 'dE/dx mean constraint')]:
    y = np.zeros(len(AL))
    for f in sorted(glob.glob(RUN + '/history_iter*.pkl')):
        for rec in pickle.load(open(f, 'rb'))['truthpoint']:
            if all(KEY(x) in rec['points'] for x in AL):
                y += np.array([float(rec['points'][KEY(x)]['aux'].get(k, 0.)) for x in AL])
    a.plot(AL, y - y[0], color=col, lw=2.0, marker='o', ms=4, label=lab)
a.axhline(0, color=C['ink'], lw=1.3)
a.axvline(1, color=C['red'], lw=1.4, ls='--')
a.set_yscale('symlog', linthresh=100)
a.set_xlabel('α'); a.set_ylabel('Δ from its value at the fit (α = 0)')
a.set_title('(e) only the student-t prior rises steeply with α.\n'
            'The mean constraint barely moves.', fontsize=9.3)
a.legend(fontsize=6.9, frameon=False)

# ---------------- (f) what it means for the dispersion ----------------
a = ax[1, 2]
FITW, TRUW = 0.0268, 0.0843          # 16-84% half-widths, production grid (fig 48)
PRIORW = 0.0465                      # the prior AS APPLIED (beta = 4.96), fig 48(a)
al = np.linspace(0, 1.2, 200)
# in log space the interpolation is linear, so the width interpolates to first order
wid = FITW + al * (TRUW - FITW)
a.plot(al, wid, color=C['ink'], lw=2.4)
for x, col, lab in [(0.0, C['blue'], 'the fit'), (A_LIK, C['green'], 'the data’s optimum'),
                    (1.0, C['red'], 'truth')]:
    y = FITW + x * (TRUW - FITW)
    a.plot([x], [y], marker='o', ms=10, color=col, zorder=5)
    a.annotate(f'  {lab}\n  width {y:.3f} ({100*y/TRUW:.0f}% of truth)', (x, y), fontsize=7.2,
               color=col, va='bottom' if x < .9 else 'top',
               ha='left' if x < .9 else 'right')
a.axhline(PRIORW, color=C['orange'], lw=1.8, ls=':')
a.annotate(f'the prior AS APPLIED: {PRIORW:.3f}\nNARROWER than the data wants,\n'
           f'so it can only pull α down', (0.72, PRIORW), fontsize=7.2, color=C['orange'],
           va='top', ha='left')
a.set_xlabel('α'); a.set_ylabel('16–84% half-width of the dE/dx field')
a.set_title('(f) in physical terms: the data supports 66% of the true\n'
            'spread; the fit delivers 32%; our prior wants 55%.', fontsize=9.3)

fig.suptitle('Fig 52 — study 2, the α-scan: the objective asked directly where it wants the dE/dx '
             'dispersion, and answered α = 0.50.\nNot truth — with finite hits per segment the '
             'maximum-likelihood field is itself shrunk, so some shrinkage is CORRECT. The fit sits '
             'at α = 0.\nOur prior is narrower than the data wants, so every weight of it pulls α '
             'DOWN; a correctly-calibrated one would pull α UP toward truth.', fontsize=10.1)
fig.tight_layout(rect=[0, 0, 1, .90])
os.makedirs(OUT, exist_ok=True)
fig.savefig(f'{OUT}/fig52_alpha_scan.png'); plt.close(fig)
print('wrote', f'{OUT}/fig52_alpha_scan.png')
print(f'  per-batch alpha*: median {np.median(pb):.3f}  IQR {np.percentile(pb,25):.3f}-{np.percentile(pb,75):.3f}')
for w in [5.0, 3.0, W_BETA1, 0.5, 0.0]:
    print(f'  w = {w:6.3f} (beta {w*DX:5.3f}) -> objective prefers alpha* = {astar(L + OTH + (w/W_PROD)*D):.3f}')
