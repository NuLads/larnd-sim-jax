"""Fig 56 — the diagnostic scans: geometry, the dE/dx↔geometry degeneracy, and the refit validated.

Three ~1 h evaluation jobs on the BT2 fit (refit prior, w = 1/dx, mean 2500), 60 constructed
points x 100 batches x 2 seeds. No fitting anywhere: every number is the objective evaluated at a
constructed point, so nothing here depends on an optimiser.

WHAT THESE COST. ~1 GPU-h each, against ~112 GPU-h for the 20 000-iteration runs whose central
assumption they test. That asymmetry is the whole argument for scanning before fitting.

THE ONE NUMBER TO READ. Along the GEOMETRY axis the objective's optimum is alpha* = 0.988 -- it
wants geometry essentially exactly at truth, with no shrinkage, and the likelihood range along that
axis (192 637 nats over 200 batches) is 5.1x the dE/dx dispersion question measured ON THE SAME FIT
(37 475) and 228x longitudinal diffusion at +-20% (845). NOTE those must be compared like with like:
fig 52 measured a dEdx range of 6 963 on the BT0 fit, whose dEdx field starts far more shrunk, so
quoting geometry as "28x" against THAT number would be comparing two different fits.
Geometry is where this objective's information actually is, and it is purely optimiser-limited.
"""
import glob, os, pickle, re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = 'plots/noise_report'
C = dict(blue='#0072B2', orange='#E69F00', green='#009E73', red='#D55E00',
         purple='#CC79A7', grey='#666666', ink='#222222')
plt.rcParams.update({'font.size': 9, 'axes.grid': True, 'grid.alpha': .25,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.dpi': 130, 'savefig.bbox': 'tight'})
LIK = ['log_likelihood_tick', 'no_match_penalty', 'expected_total_hits']
GA = [0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2]
J = [0, 0.25, 0.5, 0.75, 1.0]
M = [0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2]
AA = [0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0]
# BT2's fitted dE/dx field is 0.685 of the true spread (fig 54); truth's 16-84% half-width is
# 0.0843 on this grid. So alpha maps linearly onto a physical width.
W_FIT, W_TRUE = 0.685 * 0.0843, 0.0843
width = lambda a: W_FIT + a * (W_TRUE - W_FIT)

R = []
for f in sorted(glob.glob('fit_result/sci_full_DIAG/history_iter*.pkl')):
    s = int(re.search(r'seed(\d+)', f).group(1))
    for rec in pickle.load(open(f, 'rb'))['truthpoint']:
        R.append((s, rec))
SEEDS = sorted({s for s, _ in R})
print(f'{len(R)} batch scans over seeds {SEEDS}')


def agg(key, which='tot', seeds=None):
    t = 0.0
    for s, r in R:
        if seeds is not None and s not in seeds:
            continue
        if key not in r['points']:
            return None
        a = r['points'][key]['aux']
        t += (r['points'][key]['loss'] if which == 'tot'
              else sum(float(a.get(k, 0.)) for k in LIK))
    return t


def astar(xs, ys, clip=None):
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    i = int(np.argmin(ys))
    edge = (i == 0 or i == len(ys) - 1)
    i = min(max(i, 1), len(ys) - 2)
    c = np.polyfit(xs[i - 1:i + 2], ys[i - 1:i + 2], 2)
    v = float(-c[1] / (2 * c[0]))
    return (v, edge)


fig, ax = plt.subplots(2, 3, figsize=(16.4, 8.8))

# ---------------- (a) the geometry axis ----------------
a = ax[0, 0]
for w, col, lab in [('tot', C['ink'], 'TOTAL objective'), ('lik', C['blue'], 'LIKELIHOOD only')]:
    y = np.array([agg(f'G{g:g}', w) for g in GA])
    a.plot(GA, y - y.min(), color=col, lw=2.4, marker='o', ms=6, label=lab)
    av, _ = astar(GA, y)
a.axvline(av, color=C['green'], lw=2.4, label=f'optimum α* = {av:.3f}')
a.axvline(0, color=C['grey'], lw=1.4)
a.annotate('the fit', (0, 0), textcoords='offset points', xytext=(5, 18), fontsize=7.6,
           color=C['grey'])
a.set_xlabel('geometry α   (0 = fit, 1 = truth-projection)')
a.set_ylabel('Δ from the minimum (200 batches)')
a.set_title('(a) GEOMETRY IS OPTIMISER-LIMITED. The objective wants it\n'
            'essentially exactly at truth — no shrinkage, unlike dE/dx.', fontsize=9.3)
a.legend(fontsize=7.4, frameon=False)

# ---------------- (b) the joint grid ----------------
a = ax[0, 1]
Z = np.array([[agg(f'J{g:g}_{d:g}') for d in J] for g in J])
Zn = Z - Z.min()
m = a.pcolormesh(J, J, Zn, shading='nearest', cmap='viridis_r',
                 vmin=0, vmax=np.percentile(Zn, 75))
ridge = [astar(J, Z[k])[0] for k in range(len(J))]
a.plot(ridge, J, color='white', lw=2.6, marker='o', ms=7, label='dE/dx optimum at each geometry')
a.set_xlabel('dE/dx α'); a.set_ylabel('geometry α')
fig.colorbar(m, ax=a, label='Δ total objective')
a.set_title('(b) THE DEGENERACY. The dE/dx optimum slides from 0.01\n'
            'to 0.33 as geometry moves from the fit to truth.', fontsize=9.3)
a.legend(fontsize=7.2, frameon=True, framealpha=.8, loc='upper left')

# ---------------- (c) the degeneracy in physical units ----------------
a = ax[0, 2]
for w, col, lab in [('tot', C['ink'], 'TOTAL objective'), ('lik', C['blue'], 'LIKELIHOOD only')]:
    r = [astar(J, [agg(f'J{g:g}_{d:g}', w) for d in J])[0] for g in J]
    a.plot(J, [100 * width(x) / W_TRUE for x in r], color=col, lw=2.4, marker='o', ms=7, label=lab)
a.axhline(100 * W_FIT / W_TRUE, color=C['green'], lw=2.0, ls='--',
          label=f'what BT2 actually delivers ({100*W_FIT/W_TRUE:.0f}%)')
a.axhline(100, color=C['red'], lw=1.6, ls=':', label='the true spread')
a.set_xlabel('geometry α'); a.set_ylabel('dE/dx spread the objective wants (% of truth)')
a.set_title('(c) the better the geometry, the MORE dE/dx dispersion\n'
            'is wanted: likelihood 63→82%, total 69→79% of truth.', fontsize=9.3)
a.legend(fontsize=7.0, frameon=False)

# ---------------- (d) long_diff at two geometries ----------------
a = ax[1, 0]
for g, col, lab in [('gfit', C['ink'], 'geometry at the FIT'), ('gtru', C['red'], 'geometry at TRUTH')]:
    y = np.array([agg(f'Clong_diff_{g}_{m:g}') for m in M])
    av, edge = astar(M, y)
    a.plot(M, y - y.min(), color=col, lw=2.4, marker='o', ms=6,
           label=f'{lab} — optimum {"≤0.80×" if edge else f"{av:.2f}×"}')
a.axvline(1.0, color=C['grey'], lw=1.6)
a.annotate(' the fitted value', (1.0, 0), textcoords='offset points', xytext=(3, 30),
           fontsize=7.4, color=C['grey'])
a.set_yscale('symlog', linthresh=50)
a.set_xlabel('longitudinal diffusion, as a multiple of its fitted value')
a.set_ylabel('Δ from the minimum')
a.set_title('(d) LONG. DIFFUSION IS ENTANGLED WITH GEOMETRY: fixing\n'
            'geometry moves its optimum down by ≥20%.', fontsize=9.3)
a.legend(fontsize=7.4, frameon=False)

# ---------------- (e) the refit validated ----------------
a = ax[1, 1]
for w, col, lab in [('tot', C['ink'], 'TOTAL objective'), ('lik', C['blue'], 'LIKELIHOOD only')]:
    y = np.array([agg(f'A{x:g}', w) for x in AA])
    av, _ = astar(AA, y)
    a.plot(AA, y - y.min(), color=col, lw=2.4, marker='o', ms=6, label=f'{lab}  α* = {av:+.3f}')
a.axvline(0, color=C['green'], lw=2.0, label='the BT2 fit sits here')
a.set_xlim(-.3, 1.3); a.set_yscale('symlog', linthresh=100)
a.set_xlabel('dE/dx α  (from the BT2 fit toward truth)')
a.set_ylabel('Δ from the minimum')
a.set_title('(e) THE REFIT IS VALIDATED. Total and likelihood optima\n'
            'now differ by 0.17 in α; with the old prior it was 0.50.', fontsize=9.3)
a.legend(fontsize=7.2, frameon=False)

# ---------------- (f) where the information is ----------------
a = ax[1, 2]
gl = np.array([agg(f'G{g:g}', 'lik') for g in GA])
dl = np.array([agg(f'A{x:g}', 'lik') for x in AA])
BARS = [('GEOMETRY\nfit → truth', gl.max() - gl.min(), C['blue']),
        ('dE/dx dispersion\nfit → truth', dl[:6].max() - dl[:6].min(), C['orange']),
        ('long. diffusion\n±20%', max(agg(f'Clong_diff_gfit_{m:g}', 'lik') for m in M)
         - min(agg(f'Clong_diff_gfit_{m:g}', 'lik') for m in M), C['purple'])]
a.bar(range(len(BARS)), [b[1] for b in BARS], color=[b[2] for b in BARS], edgecolor='white')
for k, b in enumerate(BARS):
    a.text(k, b[1] * 1.15, f'{b[1]:,.0f}', ha='center', fontsize=8.6)
a.set_yscale('log')
a.set_xticks(range(len(BARS))); a.set_xticklabels([b[0] for b in BARS], fontsize=7.6)
a.set_ylabel('likelihood range along the axis (nats, 200 batches)')
a.set_ylim(top=BARS[0][1] * 6)
a.set_title('(f) WHERE THE INFORMATION IS. Geometry carries 5.1× the\n'
            'dE/dx dispersion question and 228× long. diffusion.', fontsize=9.3)

fig.suptitle('Fig 56 — three ~1 h evaluation jobs, and they redirect the campaign. GEOMETRY is '
             'optimiser-limited (α* = 0.988) and carries 5.1× the\nlikelihood of the dE/dx question '
             '(a, f). The two blocks are DEGENERATE: better geometry makes the data want more dE/dx '
             'dispersion —\nthe likelihood\'s preferred spread runs 63% → 82% of truth (b, c), which is why '
             'the ceiling arms did not predict production.\nThe refit prior is validated (e).', fontsize=10.0)
fig.tight_layout(rect=[0, 0, 1, .90])
os.makedirs(OUT, exist_ok=True)
fig.savefig(f'{OUT}/fig56_diag_scans.png'); plt.close(fig)
print('wrote', f'{OUT}/fig56_diag_scans.png')
print(f'  geometry alpha*        = {astar(GA, [agg(f"G{g:g}") for g in GA])[0]:.3f}')
print(f'  dEdx alpha* (BT2 fit)  = {astar(AA, [agg(f"A{x:g}") for x in AA])[0]:+.3f} total, '
      f'{astar(AA, [agg(f"A{x:g}", "lik") for x in AA])[0]:+.3f} likelihood')
for g in J:
    print(f'  geom a={g:4.2f} -> dEdx a* = {astar(J, [agg(f"J{g:g}_{d:g}") for d in J])[0]:+.3f} '
          f'({100*width(astar(J, [agg(f"J{g:g}_{d:g}") for d in J])[0])/W_TRUE:.0f}% of the true spread)')
