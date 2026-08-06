"""Fig 43: convergence traces of the production configuration — everything the fit is moving.

Eight panels on one iteration axis: the five calibration parameters (as % error against each
seed's own target, since targets are drawn per seed and span lifetime 984-4901 us), the
per-segment dE/dx MAE against BOTH of its references, the position residual, and the loss.

WHY THE dE/dx PANEL HAS TWO CURVES. `dedx_mae_iter` is computed against `--input_file_sim`, which
in every S4 run is the straight-line guess -- so it measures distance from the INITIALISATION, not
error against truth, and it RISES while the fit is in fact improving. `dedx_mae_truth_iter` is the
arc-length-matched truth reference. The divergence between them is the whole of the long-standing
"the MAE gets worse" puzzle.

Traces are per-seed (thin) with the across-seed median (thick). All per-batch quantities are
rolling-median smoothed over 200 iterations, because a single iteration sees one batch and batches
differ enormously (the position residual alone spans 62-3051 um between batches).
"""
import glob, os, re, pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = 'plots/noise_report'
RUN = 'fit_result/sci_full_ANNEALLONG2'          # production config, current tree, has truth MAE
W = 200
C = dict(blue='#0072B2', orange='#E69F00', green='#009E73', red='#D55E00',
         purple='#CC79A7', grey='#666666', ink='#222222')
plt.rcParams.update({'font.size': 9, 'axes.grid': True, 'grid.alpha': .25,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.dpi': 130, 'savefig.bbox': 'tight'})
P = ['Ab', 'eField', 'tran_diff', 'long_diff', 'lifetime']
PL = {'Ab': 'A$_b$ (recombination)', 'eField': 'E field', 'lifetime': 'lifetime',
      'tran_diff': 'transverse diffusion', 'long_diff': 'longitudinal diffusion'}


def smooth(v, w=W):
    v = np.asarray(v, float)
    if len(v) < w:
        return v
    from numpy.lib.stride_tricks import sliding_window_view
    return np.median(sliding_window_view(v, w), axis=1)


runs = []
for f in sorted(glob.glob(RUN + '/history_iter*.pkl')):
    if int(re.search(r'iter(\d+)', f).group(1)) != 10000:
        continue
    h = pickle.load(open(f, 'rb'))
    d = {'seed': int(re.search(r'seed(\d+)', f).group(1))}
    for p in P:
        t = float(np.ravel(h[p + '_target'])[0])
        d[p] = 100 * (np.ravel(h[p + '_iter'])[1:] / t - 1)
    d['pos'] = np.ravel(h['pos_residual_iter']) * 1e4
    d['loss'] = np.asarray(h['losses_iter'], float)
    d['mae_sim'] = np.asarray([x for x in h.get('dedx_mae_iter', []) if x == x], float)
    d['mae_tru'] = np.asarray([x for x in h.get('dedx_mae_truth_iter', []) if x == x], float)
    runs.append(d)
print(f'{len(runs)} completed runs from {RUN}')

fig, axes = plt.subplots(2, 4, figsize=(16.0, 7.4), sharex=True)
ax = axes.ravel()

for k, p in enumerate(P):
    a = ax[k]
    curves = []
    for r in runs:
        y = smooth(r[p]); a.plot(y, lw=.7, color=C['blue'], alpha=.35)
        curves.append(y)
    n = min(len(c) for c in curves)
    a.plot(np.median([c[:n] for c in curves], axis=0), lw=2.2, color=C['ink'])
    a.axhline(0, color=C['red'], lw=1.3, ls='--')
    a.set_title(PL[p], fontsize=9.4)
    a.set_ylabel('error vs truth (%)' if k in (0, 4) else '')

a = ax[5]
for r in runs:
    if len(r['mae_sim']): a.plot(smooth(r['mae_sim']), lw=.7, color=C['orange'], alpha=.4)
    if len(r['mae_tru']): a.plot(smooth(r['mae_tru']), lw=.7, color=C['green'], alpha=.4)
for key, col, lab in [('mae_sim', C['orange'], 'vs SIM input (the guess file)'),
                      ('mae_tru', C['green'], 'vs TRUTH (arc-length matched)')]:
    cs = [smooth(r[key]) for r in runs if len(r[key])]
    if cs:
        n = min(len(c) for c in cs)
        a.plot(np.median([c[:n] for c in cs], axis=0), lw=2.2, color=col, label=lab)
a.set_title('per-segment dE/dx MAE', fontsize=9.4); a.set_ylabel('MAE')
a.legend(fontsize=7.0, frameon=False)

a = ax[6]
cs = [smooth(r['pos']) for r in runs]
for c in cs: a.plot(c, lw=.7, color=C['purple'], alpha=.35)
n = min(len(c) for c in cs)
a.plot(np.median([c[:n] for c in cs], axis=0), lw=2.2, color=C['ink'])
a.set_yscale('log'); a.set_title('position residual', fontsize=9.4); a.set_ylabel('µm')

a = ax[7]
cs = [smooth(r['loss']) for r in runs]
for c in cs: a.plot(c, lw=.7, color=C['green'], alpha=.35)
n = min(len(c) for c in cs)
a.plot(np.median([c[:n] for c in cs], axis=0), lw=2.2, color=C['ink'])
a.set_yscale('log'); a.set_title('loss (PPP negative log-likelihood)', fontsize=9.4); a.set_ylabel('NLL')

for a in ax[4:]:
    a.set_xlabel('iteration')

fig.suptitle('Fig 43 — production convergence traces (thin = seeds, thick = median, 200-step rolling '
             'median).\nA$_b$ and E field settle early; the two diffusions and lifetime keep '
             'drifting. The dE/dx MAE RISES against its input while FALLING against truth.',
             fontsize=10.2)
fig.tight_layout(rect=[0, 0, 1, .91])
os.makedirs(OUT, exist_ok=True)
fig.savefig(f'{OUT}/fig43_convergence_traces.png'); plt.close(fig)
print('wrote', f'{OUT}/fig43_convergence_traces.png')
for p in P:
    e = [np.median(r[p][int(len(r[p]) * .8):]) for r in runs]
    print(f'  {p:11s} tail median {np.mean(e):+7.2f}%')
for key, lab in [('mae_sim', 'MAE vs SIM  '), ('mae_tru', 'MAE vs TRUTH')]:
    v = [r[key] for r in runs if len(r[key])]
    if v:
        print(f'  {lab} {np.mean([x[:50].mean() for x in v]):.4f} -> {np.mean([x[-50:].mean() for x in v]):.4f}')
