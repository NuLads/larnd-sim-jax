"""Fig 41: three controlled ceiling comparisons — dE/dx prior weight, mean-constraint weight, noise.

All four arms are ceiling (true geometry, frozen), 10 000 iterations, 100 batches of 400 cm, and
differ from their control in exactly ONE knob. Ceiling is the right venue because it removes the
geometry variance that dominates production and shrinks the seed s.d. by 11-70x, which is what
makes single-knob differences resolvable at all (three runs of an identical production config span
3.6 points on lifetime).

  (a) dE/dx PRIOR weight 5 -> 0.5. The production ANNEALLONG value is 5; the script default is 0.5.
  (b) dE/dx MEAN-CONSTRAINT weight 1e5 -> 3000, at the corrected prior. The previous attempt at this
      test ran at prior 5, which entangled the two dE/dx weights.
  (c) readout noise ON -> OFF, self-consistent (target and guess both).
  (d) what each costs the dE/dx block itself, as the fraction of the prior->truth gap closed.
"""
import glob, os, pickle, io, contextlib
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

ARMS = {
    'base':   ('fit_result/sci_ceiling_CEILBASE',   'prior 5  (production value)'),
    'p05':    ('fit_result/sci_ceiling_CEILP05',    'prior 0.5  (script default)'),
    'w3k':    ('fit_result/sci_ceiling_CEILW3KP05', 'prior 0.5, mean w = 3000'),
    'nonoise':('fit_result/sci_ceiling_CEILP05NN',  'prior 0.5, noise OFF'),
}


def load(d):
    with contextlib.redirect_stdout(io.StringIO()):
        r = R.analyse('x', d, True)
    mae = []
    for f in sorted(glob.glob(d + '/history_iter*.pkl')):
        h = pickle.load(open(f, 'rb'))
        a = np.asarray([x for x in h.get('dedx_mae_iter', []) if x == x], float)
        if len(a) > 200:
            mae.append((a[:50].mean(), a[-50:].mean()))
    m = np.array(mae)
    gap = 100 * (1 - m[:, 1].mean() / m[:, 0].mean()) if len(m) else np.nan
    return r, gap


D = {k: load(v[0]) for k, v in ARMS.items()}
xs = np.arange(len(R.P))

fig, axes = plt.subplots(1, 4, figsize=(15.2, 4.3))
PAIRS = [('base', 'p05', '(a) dE/dx PRIOR weight 5 → 0.5', C['red'], C['blue']),
         ('p05', 'w3k', '(b) mean-constraint weight 1e5 → 3000', C['blue'], C['orange']),
         ('p05', 'nonoise', '(c) readout noise ON → OFF', C['blue'], C['green'])]

for ax, (ka, kb, title, ca, cb) in zip(axes[:3], PAIRS):
    for k, (key, col, off) in enumerate([(ka, ca, -0.14), (kb, cb, +0.14)]):
        r = D[key][0]
        m = [np.mean(r['rows'][p]) for p in R.P]
        e = [np.std(r['rows'][p]) for p in R.P]
        ax.errorbar(xs + off, m, yerr=e, lw=0, elinewidth=2, marker='o', ms=6.5, capsize=4,
                    color=col, label=f'{ARMS[key][1]}  (n={r["n"]})')
    ax.axhline(0, color=C['grey'], lw=1.2)
    ax.set_xticks(xs); ax.set_xticklabels([PL[p] for p in R.P], fontsize=7.6, rotation=20, ha='right')
    ax.set_title(title, fontsize=9.3)
    ax.legend(fontsize=7.2, frameon=False, loc='lower left')
axes[0].set_ylabel('error vs truth (%)')

ax = axes[3]
order = ['base', 'p05', 'w3k', 'nonoise']
cols = [C['red'], C['blue'], C['orange'], C['green']]
gaps = [D[k][1] for k in order]
ax.bar(range(len(order)), gaps, color=cols, edgecolor='white', linewidth=.8)
for k, g in enumerate(gaps):
    ax.text(k, g + 1, f'{g:.1f}%', ha='center', fontsize=8)
ax.axhspan(40, 49, color=C['grey'], alpha=.18)
ax.text(0.05, 44.5, ' historical noise-ON band', fontsize=7.2, color=C['grey'])
ax.set_xticks(range(len(order)))
ax.set_xticklabels(['prior 5', 'prior 0.5', 'w=3000', 'noise OFF'], fontsize=7.6, rotation=20, ha='right')
ax.set_ylabel('dE/dx: fraction of prior→truth gap closed (%)')
ax.set_title('(d) what each knob costs the dE/dx block', fontsize=9.3)

fig.suptitle('Fig 41 — three single-knob ceiling comparisons. The production dE/dx prior (5) is '
             'badly mis-set: dropping it to the\ndefault 0.5 moves long. diffusion −9.95% → +1.43% '
             'and 2.5× the dE/dx recovery. Lowering the mean weight does NOT help.', fontsize=10.0)
fig.tight_layout(rect=[0, 0, 1, .90])
os.makedirs(OUT, exist_ok=True)
fig.savefig(f'{OUT}/fig41_ceiling_knobs.png'); plt.close(fig)
print('wrote', f'{OUT}/fig41_ceiling_knobs.png')
for k in order:
    r, g = D[k]
    print(f'  {ARMS[k][1]:32s} n={r["n"]} gap={g:5.1f}%  ' +
          '  '.join(f'{p}={np.mean(r["rows"][p]):+.2f}±{np.std(r["rows"][p]):.2f}' for p in R.P))
