"""Fig 46 (study 2): is the loss minimum misplaced, or merely unreachable?

6 seeds x 100 batches. For every batch the objective is evaluated at the fit point, at the truth
point (each block at the best its parameterisation can represent), and at the three one-block
hybrids -- no fitting, so nothing here depends on an optimiser.

The headline is that the question had a compound answer: the two blocks fail for OPPOSITE reasons.
Geometry is unreachable (the likelihood wants truth and we do not get there); dE/dx is displaced
(the likelihood is indifferent and a prior pushes us off truth). Panel (b) is the one that
separates them, and it is why the total-objective verdict in panel (a) is misleading on its own.
"""
import glob, pickle, re, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = 'plots/noise_report'
RUN = 'fit_result/sci_full_TP'
C = dict(blue='#0072B2', orange='#E69F00', green='#009E73', red='#D55E00',
         purple='#CC79A7', grey='#666666', ink='#222222')
plt.rcParams.update({'font.size': 9, 'axes.grid': True, 'grid.alpha': .25,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.dpi': 130, 'savefig.bbox': 'tight'})
POINTS = ['P0_fit', 'P1_truth', 'P2_calib_true', 'P3_dedx_true', 'P4_geom_true']
LIK = ['log_likelihood_tick', 'no_match_penalty', 'expected_total_hits']
PRI = ['dedx_prior', 'dedx_mean_penalty', 'dedx_barrier', 'mcs_prior',
       'chain_drift_penalty', 'dedx_drift_penalty', 'spatial_moment']
ACHIEVED_UM = 159.0            # production position residual, 8 distinct seeds (fig 42)

rows = []
for f in sorted(glob.glob(RUN + '/history_iter*.pkl')):
    seed = int(re.search(r'seed(\d+)', f).group(1))
    for rec in pickle.load(open(f, 'rb'))['truthpoint']:
        if not all(p in rec['points'] for p in POINTS):
            continue
        d = {'seed': seed, 'floor': rec['repr_floor_um']}
        for p in POINTS:
            a = rec['points'][p]['aux']
            d[p] = rec['points'][p]['loss']
            d[p + '_lik'] = sum(float(a.get(k, 0.0)) for k in LIK)
            d[p + '_pri'] = sum(float(a.get(k, 0.0)) for k in PRI)
            d[p + '_dedxprior'] = float(a.get('dedx_prior', 0.0))
            for k in ('dedx_prior', 'dedx_mean_penalty', 'mcs_prior', 'log_likelihood_tick',
                      'expected_total_hits', 'no_match_penalty'):
                d[f'{p}_{k}'] = float(a.get(k, 0.0))
        rows.append(d)
g = lambda k: np.array([r[k] for r in rows], float)
n = len(rows)
p0 = g('P0_fit')
print(f'{n} batch evaluations, {len(set(r["seed"] for r in rows))} seeds')

fig, ax = plt.subplots(2, 3, figsize=(16.2, 8.8))

# ---------------- (a) total objective ----------------
a = ax[0, 0]
rel = 100 * (g('P1_truth') - p0) / np.abs(p0)
a.hist(rel, bins=44, color=C['grey'], edgecolor='white', linewidth=.5)
a.axvline(0, color=C['ink'], lw=1.6)
a.axvline(np.median(rel), color=C['red'], lw=2.2, ls='--',
          label=f'median {np.median(rel):+.2f}%')
a.set_xlabel('loss(truth) − loss(fit), % of the fit loss')
a.set_ylabel('batches')
a.set_title(f'(a) TOTAL objective: truth wins in only\n'
            f'{(rel<0).sum()}/{n} batches ({100*(rel<0).mean():.0f}%) — looks like a displaced minimum',
            fontsize=9.3)
a.legend(fontsize=7.6, frameon=False)

# ---------------- (b) the split ----------------
a = ax[0, 1]
dl = g('P1_truth_lik') - g('P0_fit_lik')
dp = g('P1_truth_pri') - g('P0_fit_pri')
bins = np.linspace(min(dl.min(), -1200), max(dp.max(), 1200), 60)
a.hist(dl, bins=bins, color=C['blue'], alpha=.78, edgecolor='white', linewidth=.4,
       label=f'LIKELIHOOD: truth wins {100*(dl<0).mean():.0f}%  (median {np.median(dl):+.0f})')
a.hist(dp, bins=bins, color=C['orange'], alpha=.78, edgecolor='white', linewidth=.4,
       label=f'PRIORS: truth wins {100*(dp<0).mean():.0f}%  (median {np.median(dp):+.0f})')
a.axvline(0, color=C['ink'], lw=1.6)
a.set_xscale('symlog', linthresh=100)
a.set_xlabel('term(truth) − term(fit)   [symlog, linear within ±100]')
a.set_ylabel('batches')
a.set_title('(b) THE ANSWER. The likelihood prefers TRUTH in 86%\n'
            'of batches; the priors prefer the fit in 100% of them', fontsize=9.3)
a.legend(fontsize=7.2, frameon=False, loc='upper left')

# ---------------- (c) block attribution ----------------
a = ax[0, 2]
BL = [('P2_calib_true', 'calibration'), ('P3_dedx_true', 'dE/dx'), ('P4_geom_true', 'geometry')]
xs = np.arange(len(BL))
mlik = [np.median(g(p + '_lik') - g('P0_fit_lik')) for p, _ in BL]
mpri = [np.median(g(p + '_pri') - g('P0_fit_pri')) for p, _ in BL]
a.bar(xs - .19, mlik, width=.36, color=C['blue'], edgecolor='white', label='likelihood')
a.bar(xs + .19, mpri, width=.36, color=C['orange'], edgecolor='white', label='priors')
for k in range(len(BL)):
    for off, v in [(-.19, mlik[k]), (.19, mpri[k])]:
        a.annotate(f'{v:+.0f}', (k + off, v), textcoords='offset points',
                   xytext=(0, 5 if v >= 0 else -12), ha='center', fontsize=7.4)
a.axhline(0, color=C['ink'], lw=1.5)
a.set_xticks(xs); a.set_xticklabels([b[1] for b in BL], fontsize=8.6)
a.set_ylabel('median Δ when THIS block alone moves to truth')
a.set_title('(c) opposite failures. Geometry: the likelihood wants\ntruth. dE/dx: only the prior objects.',
            fontsize=9.3)
a.legend(fontsize=7.6, frameon=False)

# ---------------- (d) which prior term ----------------
a = ax[1, 0]
TERMS = [('dedx_prior', 'dE/dx student-t prior', C['orange']),
         ('log_likelihood_tick', 'joint hit log-intensity', C['blue']),
         ('dedx_mean_penalty', 'dE/dx mean constraint', C['grey']),
         ('expected_total_hits', 'expected hit count', C['green']),
         ('no_match_penalty', 'unmatched-target penalty', C['red']),
         ('mcs_prior', 'MCS prior', C['purple'])]
vals = [np.median(g(f'P3_dedx_true_{t}') - g(f'P0_fit_{t}')) for t, _, _ in TERMS]
ys = np.arange(len(TERMS))
a.barh(ys, vals, color=[c for _, _, c in TERMS], edgecolor='white', height=.62)
for k, v in enumerate(vals):
    a.text(v + 14, k, f'{v:+.1f}', va='center', ha='left', fontsize=7.6)
a.axvline(0, color=C['ink'], lw=1.5)
a.set_xlim(-60, max(vals) * 1.18)
a.set_yticks(ys); a.set_yticklabels([l for _, l, _ in TERMS], fontsize=7.6)
a.invert_yaxis()
a.set_xlabel('median Δ when dE/dx alone moves to truth')
a.set_title('(d) ONE term does all of it: the student-t prior\ncharges +562 where the data charges −10',
            fontsize=9.3)

# ---------------- (e) representation floor ----------------
a = ax[1, 1]
fl = g('floor')
a.hist(fl, bins=40, color=C['green'], edgecolor='white', linewidth=.5)
a.axvline(np.median(fl), color=C['ink'], lw=2.2, ls='--',
          label=f'basis floor, median {np.median(fl):.0f} µm')
a.axvline(ACHIEVED_UM, color=C['red'], lw=2.2,
          label=f'production achieves {ACHIEVED_UM:.0f} µm')
a.set_xlabel('position residual of the truth PROJECTION (µm)')
a.set_ylabel('batches')
a.set_title(f'(e) the spline basis is not the limit:\n'
            f'{ACHIEVED_UM/np.median(fl):.2f}× headroom remains in geometry', fontsize=9.3)
a.legend(fontsize=7.6, frameon=False)

# ---------------- (f) what-if on the prior weight ----------------
a = ax[1, 2]
ws = np.logspace(-2, 0.30, 60)
frac, med = [], []
for w in ws:
    t0 = p0 - (1 - w) * g('P0_fit_dedxprior')
    t1 = g('P1_truth') - (1 - w) * g('P1_truth_dedxprior')
    dd = t1 - t0
    frac.append(100 * (dd < 0).mean())
    med.append(np.median(100 * dd / np.abs(t0)))
frac, med = np.array(frac), np.array(med)
a.plot(ws * 5.0, frac, color=C['blue'], lw=2.4)
a.axhline(50, color=C['grey'], ls=':', lw=1.4)
a.axvline(5.0, color=C['red'], lw=2.0, label='production prior weight 5')
a.axvline(0.5, color=C['green'], lw=2.0, label='script default 0.5')
_x = ws * 5.0
_cross = _x[np.argmin(np.abs(frac - 50))]
a.plot([_cross], [50], marker='o', ms=9, color=C['ink'], zorder=5)
a.annotate(f'  crossover ≈ {_cross:.2f}', (_cross, 50), fontsize=7.8, va='bottom')
a.set_xscale('log')
a.set_xlabel('dE/dx prior weight (the term re-scored, NOT re-fitted)')
a.set_ylabel('% of batches where truth beats the fit')
a.set_title('(f) at the default weight the objective would\nalready prefer truth in 63% of batches',
            fontsize=9.3)
a.legend(fontsize=7.4, frameon=False, loc='lower left')

fig.suptitle('Fig 46 — study 2: the minimum is not misplaced, the PRIOR is. The likelihood prefers '
             'truth in 86% of batches;\nthe dE/dx student-t prior at weight 5 overrides it. Geometry '
             'is a separate, genuinely unreached direction.', fontsize=10.4)
fig.tight_layout(rect=[0, 0, 1, .92])
os.makedirs(OUT, exist_ok=True)
fig.savefig(f'{OUT}/fig46_truthpoint.png'); plt.close(fig)
print('wrote', f'{OUT}/fig46_truthpoint.png')
print(f'  crossover prior weight for a 50% truth-win rate: {_cross:.3f}')
print(f'  at weight 5.0  : {frac[np.argmin(np.abs(_x-5.0))]:.1f}% of batches, '
      f'median {med[np.argmin(np.abs(_x-5.0))]:+.2f}%')
print(f'  at weight 0.5  : {frac[np.argmin(np.abs(_x-0.5))]:.1f}% of batches, '
      f'median {med[np.argmin(np.abs(_x-0.5))]:+.2f}%')
