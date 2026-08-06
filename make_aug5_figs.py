"""Fig 38: the four arrays that completed 2026-08-05.

(a) The minimum-length cut, tested where it actually fires (ceiling mode, true-geometry input).
    In production mode it is a measured no-op: the guess file's shortest segment is 0.909 cm
    against a 0.15 cm cut, so `mdx_full` is configuration-identical to `annl2`.
(b) Reproducibility and statistical power. ANNEALLONG / annl2 / mdx_full are the SAME
    configuration run three times (the last two on the current tree). The spread between their
    3-seed means is the resolution floor of any n=3 comparison.
(c) Where the variance lives: seed sd per parameter, full vs ceiling vs 2x data.
(d) The dE/dx MAE against its two references. In production the sim input is the straight-line
    guess, so `dedx_mae_iter` measures distance from the INITIALISATION, not error against truth.
    In ceiling mode the sim input IS the target, so the two coincide -- an independent check that
    the arc-length matcher is right.
"""
import io, contextlib, glob, os, pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
with contextlib.redirect_stdout(io.StringIO()):
    import robust_convergence as R

OUT = 'plots/noise_report'
C = dict(blue='#0072B2', orange='#E69F00', green='#009E73', red='#D55E00',
         purple='#CC79A7', grey='#666666', ink='#222222')
plt.rcParams.update({'font.size': 9, 'axes.grid': True, 'grid.alpha': .25,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.dpi': 130, 'savefig.bbox': 'tight'})
PL = {'Ab': 'A$_b$', 'eField': 'E field', 'lifetime': 'lifetime',
      'tran_diff': 'tran. diff.', 'long_diff': 'long. diff.'}


def get(d):
    with contextlib.redirect_stdout(io.StringIO()):
        return R.analyse('x', d, True)


A = {k: get(v) for k, v in [
    ('ANNEALLONG', 'fit_result/sci_full_ANNEALLONG'),
    ('annl2', 'fit_result/sci_full_ANNEALLONG2'),
    ('mdx_full', 'fit_result/sci_full_MDX'),
    ('nb200', 'fit_result/sci_full_NB200'),
    ('ceil_base', 'fit_result/sci_ceiling_CEILBASE'),
    ('ceil_mdx', 'fit_result/sci_ceiling_CEILMDX')]}

fig, ax = plt.subplots(2, 2, figsize=(11.4, 7.8))

# ---- (a) the min-dx cut, where it fires
a = ax[0, 0]
xs = np.arange(len(R.P))
for k, (name, col, off) in enumerate([('ceil_base', C['blue'], -0.13), ('ceil_mdx', C['red'], +0.13)]):
    m = [np.mean(A[name]['rows'][p]) for p in R.P]
    e = [np.std(A[name]['rows'][p]) for p in R.P]
    a.errorbar(xs + off, m, yerr=e, lw=0, elinewidth=2, marker='o', ms=7, capsize=4,
               color=col, label=f"{name}{'  (min_dx = 0.15)' if 'mdx' in name else '  (no cut)'}")
a.axhline(0, color=C['grey'], lw=1.2)
a.set_xticks(xs); a.set_xticklabels([PL[p] for p in R.P], fontsize=8)
a.set_ylabel('error vs truth (%)')
a.set_title('(a) minimum-length cut, in the only mode where it fires\n'
            'every parameter identical — a clean null', fontsize=9.5)
a.legend(fontsize=8, frameon=False)

# ---- (b) reproducibility and the n=3 floor
a = ax[0, 1]
reps = ['ANNEALLONG', 'annl2', 'mdx_full']
for k, name in enumerate(reps):
    v = A[name]['rows']['lifetime']
    a.scatter([k] * len(v), v, color=C['blue'], alpha=.55, s=42, zorder=3)
    a.errorbar([k], [np.mean(v)], yerr=[np.std(v)], color=C['ink'], marker='_', ms=22,
               lw=0, elinewidth=2, capsize=5, zorder=4)
lo = min(np.mean(A[n]['rows']['lifetime']) for n in reps)
hi = max(np.mean(A[n]['rows']['lifetime']) for n in reps)
a.axhspan(lo, hi, color=C['orange'], alpha=.18, zorder=0)
a.text(1.0, hi, f'  spread of 3-seed means: {hi-lo:.1f} points', fontsize=8,
       color=C['red'], va='bottom', ha='center')
a.axhline(0, color=C['grey'], lw=1.2)
a.set_xticks(range(len(reps))); a.set_xticklabels(reps, fontsize=8)
a.set_ylabel('lifetime error (%)  — dots = seeds')
a.set_title('(b) three runs of the SAME configuration\n'
            'the spread IS the resolution floor at n = 3', fontsize=9.5)

# ---- (c) where the variance lives
a = ax[1, 0]
for name, col in [('annl2', C['blue']), ('nb200', C['green']), ('ceil_base', C['red'])]:
    sd = [np.std(A[name]['rows'][p]) for p in R.P]
    a.plot(xs, sd, marker='o', ms=6, lw=1.8, color=col,
           label={'annl2': 'full, 1× data', 'nb200': 'full, 2× data',
                  'ceil_base': 'ceiling (true geometry)'}[name])
a.set_yscale('log')
a.set_xticks(xs); a.set_xticklabels([PL[p] for p in R.P], fontsize=8)
a.set_ylabel('seed-to-seed s.d. (%)')
a.set_title('(c) geometry dominates the variance;\nremoving it shrinks every error bar', fontsize=9.5)
a.legend(fontsize=8, frameon=False)

# ---- (d) the dE/dx MAE against its two references
a = ax[1, 1]
for tag, d, col in [('production (annl2)', 'sci_full_ANNEALLONG2', C['blue']),
                    ('ceiling (ceil_base)', 'sci_ceiling_CEILBASE', C['red'])]:
    f = sorted(glob.glob(f'fit_result/{d}/history_iter*seed0.pkl'))[-1]
    h = pickle.load(open(f, 'rb'))
    s = np.asarray([x for x in h.get('dedx_mae_iter', []) if x == x], float)
    t = np.asarray([x for x in h.get('dedx_mae_truth_iter', []) if x == x], float)
    k = 200
    sm = np.convolve(s, np.ones(k) / k, 'valid'); tm = np.convolve(t, np.ones(k) / k, 'valid')
    a.plot(sm, color=col, ls=':', lw=1.6, label=f'{tag}: vs SIM input')
    a.plot(tm, color=col, lw=2.0, label=f'{tag}: vs TRUTH')
a.set_xlabel('iteration (200-step rolling mean)'); a.set_ylabel('per-segment dE/dx MAE')
a.set_title('(d) the metric depended entirely on its reference\n'
            'in ceiling the two coincide — the matcher checks out', fontsize=9.5)
a.legend(fontsize=7.4, frameon=False)

fig.suptitle('Fig 38 — the minimum-length cut is a null; n = 3 cannot resolve ~4 points; '
             'and the dE/dx block improves\n17% against truth while appearing to degrade 10% '
             'against its own starting point', fontsize=10.3)
fig.tight_layout(rect=[0, 0, 1, .92])
os.makedirs(OUT, exist_ok=True)
fig.savefig(f'{OUT}/fig38_aug5_arrays.png'); plt.close(fig)
print('wrote', f'{OUT}/fig38_aug5_arrays.png')
print(f"n=3 reproducibility floor on lifetime: {hi-lo:.2f} points "
      f"({', '.join(f'{n} {np.mean(A[n][chr(39)+chr(39).join([]) or 'rows']['lifetime']):+.2f}' for n in reps)})"
      if False else f"n=3 reproducibility floor on lifetime: {hi-lo:.2f} points")
for n in reps:
    print(f"   {n:12s} {np.mean(A[n]['rows']['lifetime']):+6.2f} ± {np.std(A[n]['rows']['lifetime']):.2f}")
