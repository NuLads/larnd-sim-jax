"""Fig 70 — the profile likelihood by re-fitting DOES NOT WORK, and the noise budget says why.

WHAT WAS TRIED. A genuine profile over the calibration block: pin lifetime at each of 7 multipliers
of its fitted value and re-optimise everything else -- the other four calibration parameters AND
both nuisance blocks -- from TS1. 3 000 iterations per point, schedules compressed so every point
reads out at the same schedule phase. The pin was validated first and held to float32 round-off at
every point (relative drift 8e-9 to 1.4e-7 against free parameters moving 1e-3 to 1e-2).

WHAT CAME BACK IS NOT A PROFILE. The curve is jagged rather than convex: the minimum sits at x0.975
while x1.1, at the far end of the grid, is almost as low. A likelihood profile cannot look like
this. Putting an error bar on each point -- from the scatter of independent one-epoch windows within
the same run -- shows why.

THE NOISE BUDGET, all in nats per batch:
    expected lifetime signal across the whole +-10% grid     7.5     (and this is an UPPER bound:
                                                                     it is the CONDITIONAL curvature,
                                                                     and profiling only flattens it)
    scatter between the 7 independent re-fits               41.4
    error bar on a single point                             10.5
The signal is smaller than the noise on a single point, and five times smaller than the scatter
between points. The measurement is swamped.

AND IT CANNOT BE FIXED BY RUNNING MORE. A 1-sigma interval needs Delta = 0.5 in the SUMMED loss,
i.e. 0.005 nats per batch. The achieved per-point noise is 1050 nats summed -- about 2 100 times
too large. Averaging it down would take ~4 million times more sampling. This is not a resolution
problem, it is the wrong instrument.

WHY: each point is an INDEPENDENT stochastic re-optimisation of ~9 000 nuisance parameters, and
where that optimisation lands varies far more, in loss, than the lifetime dependence being measured.
The noise is not in the data -- every point already averages over full epochs -- it is in the
optimiser's landing point.

THE REDIRECTION. Uncertainties have to come from the CURVATURE at the fitted point, computed
analytically -- the profile covariance is the Schur complement of the full Hessian, marginalising
the nuisance blocks -- not from re-fitting. That is deterministic and has no landing-point noise.
And it is the SAME machinery study 1 asks for: fig 69 showed the path to truth is open but
ill-conditioned and wants a preconditioner. Both studies converge on one build: second-order
information. The GGN/Newton code already written for the dEdx block is the starting point.
"""
import glob, os, pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = 'plots/noise_report'
C = dict(blue='#0072B2', orange='#E69F00', green='#009E73', red='#D55E00',
         grey='#888888', ink='#222222')
plt.rcParams.update({'font.size': 9, 'axes.grid': True, 'grid.alpha': .25,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.dpi': 130, 'savefig.bbox': 'tight'})
LIK = ['log_likelihood_tick', 'log_likelihood_charge', 'no_match_penalty', 'expected_total_hits']
MULTS = [('090', 0.90), ('095', 0.95), ('0975', 0.975), ('100', 1.00),
         ('1025', 1.025), ('105', 1.05), ('110', 1.10)]

pts = []
for tag, m in MULTS:
    f = glob.glob(f'fit_result/sci_full_PROFlifetime{tag}/history_iter*seed0.pkl')[0]
    h = pickle.load(open(f, 'rb'))
    a = h['aux_iter']
    lik = np.array([sum(float(x.get(k, 0.)) for k in LIK) for x in a])
    w = lik[-1000:].reshape(-1, 100).mean(axis=1)       # independent 1-epoch means
    lt = np.ravel(h['lifetime_iter'])
    tgt = float(np.ravel(h['lifetime_target'])[0])
    pts.append(dict(m=m, mu=w.mean(), sem=w.std(ddof=1) / np.sqrt(len(w)),
                    err=100 * (np.median(lt) / tgt - 1),
                    drift=np.ptp(lt) / abs(np.median(lt))))

mu = np.array([p['mu'] for p in pts]); sem = np.array([p['sem'] for p in pts])
xs = np.array([p['err'] for p in pts])
SIG = 0.5 * (10 / 0.258) ** 2 / 100          # conditional signal across the grid, nats/batch
NOISE_SUM = np.mean(sem) * 100               # per-point noise, summed over 100 batches

fig, ax = plt.subplots(1, 3, figsize=(16.4, 5.2))

# ---------------- (a) the "profile" ----------------
a = ax[0]
a.errorbar(xs, mu - mu.min(), yerr=sem, fmt='o-', color=C['blue'], ms=8, lw=1.8, capsize=4)
a.axvline(0, color=C['red'], lw=1.8, ls='--')
a.annotate(' TRUTH', (0, (mu - mu.min()).max() * .9), fontsize=8, color=C['red'])
a.set_xlabel('lifetime error at the pinned value (%)')
a.set_ylabel('profiled likelihood − min  (nats per batch)')
a.set_title('(a) THIS IS NOT A PROFILE. Jagged, not convex: the minimum is\n'
            'at −5.3% while the far end of the grid is nearly as low.', fontsize=9.3)

# ---------------- (b) the noise budget ----------------
a = ax[1]
names = ['expected signal\nacross the WHOLE grid\n(upper bound)',
         'error bar on\na single point', 'scatter between\nthe 7 re-fits']
vals = [SIG, np.mean(sem), np.ptp(mu)]
cols = [C['green'], C['orange'], C['red']]
a.bar(range(3), vals, color=cols, edgecolor='white', width=.62)
for i, v in enumerate(vals):
    a.text(i, v * 1.06, f'{v:.1f}', ha='center', fontsize=9.5)
a.set_xticks(range(3)); a.set_xticklabels(names, fontsize=7.6)
a.set_ylabel('nats per batch')
a.set_title('(b) THE SIGNAL IS SMALLER THAN THE NOISE ON ONE POINT,\n'
            'and 5× smaller than the scatter between points.', fontsize=9.3)

# ---------------- (c) how far off we are ----------------
a = ax[2]
need = 0.5 / 100          # Delta = 0.5 summed -> nats per batch
have = np.mean(sem)
a.bar([0, 1], [need, have], color=[C['green'], C['red']], edgecolor='white', width=.55)
a.set_yscale('log')
a.set_xticks([0, 1])
a.set_xticklabels(['needed for a 1σ interval\n(Δ = 0.5 summed)', 'achieved per-point noise'],
                  fontsize=8)
a.set_ylabel('nats per batch')
for i, v in enumerate([need, have]):
    a.text(i, v * 1.35, f'{v:.4g}', ha='center', fontsize=9.5)
a.set_title(f'(c) AND IT CANNOT BE AVERAGED AWAY: off by {have/need:.0f}×,\n'
            f'which would need ~{(have/need)**2/1e6:.1f} million times more sampling.', fontsize=9.3)

fig.suptitle('Fig 70 — profiling by RE-FITTING does not work. Each point is an independent '
             'stochastic re-optimisation of ~9 000 nuisance parameters,\nand where that lands '
             'varies far more in loss than the lifetime dependence being measured. Uncertainties '
             'must come from the CURVATURE\nat the fitted point (the Schur complement of the '
             'Hessian) — the same second-order machinery fig 69 asks for.', fontsize=10.0)
fig.tight_layout(rect=[0, 0, 1, .85])
os.makedirs(OUT, exist_ok=True)
fig.savefig(f'{OUT}/fig70_profile.png'); plt.close(fig)
print('wrote', f'{OUT}/fig70_profile.png')
print(f'\n  signal (upper bound) {SIG:.1f}   per-point noise {np.mean(sem):.1f}   '
      f'between-point scatter {np.ptp(mu):.1f}  [nats/batch]')
print(f'  need {need:.4g}, have {have:.4g}  ->  {have/need:.0f}x too noisy')
print(f'  pin drift range: {min(p["drift"] for p in pts):.1e} to {max(p["drift"] for p in pts):.1e} '
      f'(float32 eps 1.19e-07)')
