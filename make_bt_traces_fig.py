"""Fig 55 — convergence traces for the three BT arms, all 6 seeds.

Nine quantities on one iteration axis: the five calibration parameters, the per-segment dE/dx MAE
against TRUTH, the length-weighted mean dE/dx, the position residual and the loss. Thin = seeds,
thick = across-seed median, 200-iteration rolling median throughout (a single iteration sees one
batch, and batches differ enormously).

THE PANEL TO READ FIRST IS <dE/dx>. It shows the mechanism behind fig 54(d) directly: BT1 (correct
weight, ORIGINAL shape) slides steadily away from the true mean all run, because at beta = 1 that
shape is wider than truth on this grid and the softened mean constraint can no longer hold the
scale. BT2 (refit) holds it. The A_b panel tracks that slide almost exactly, which is the coupling
section 6u measured -- A_b absorbs a charge-normalisation error.

Calibration parameters are plotted as % error against each seed's OWN target, since --seed draws
the target and they span lifetime 1160-4518 us.
"""
import glob, os, pickle, re, io, contextlib
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
SEEDS = sorted(int(re.search(r'seed(\d+)', f).group(1))
               for f in glob.glob('fit_result/sci_full_BT0/history_iter10000_*.pkl'))
ARMS = [('BT0', 'BT0  current, w=5, mean 1e5', C['ink']),
        ('BT1', 'BT1  current, w=1/dx, mean 2500', C['orange']),
        ('BT2', 'BT2  REFIT, w=1/dx, mean 2500', C['green'])]
PL = {'Ab': 'A$_b$ (recombination)', 'eField': 'E field', 'lifetime': 'lifetime',
      'tran_diff': 'transverse diffusion', 'long_diff': 'longitudinal diffusion'}
TRUE_MEAN = 1.88693
W = 200


def smooth(v, w=W):
    v = np.asarray(v, float)
    if len(v) <= w:
        return v
    from numpy.lib.stride_tricks import sliding_window_view
    return np.median(sliding_window_view(v, w), axis=1)


D = {}
for tag, _, _ in ARMS:
    D[tag] = {}
    for s in SEEDS:
        g = glob.glob(f'fit_result/sci_full_{tag}/history_iter10000_*seed{s}.pkl')
        if not g:
            continue
        f = g[0]
        h = pickle.load(open(f, 'rb'))
        d = {}
        for p in R.P:
            t = float(np.ravel(h[p + '_target'])[0])
            d[p] = 100 * (np.ravel(h[p + '_iter'])[1:] / t - 1)
        d['mae'] = np.asarray([x for x in h.get('dedx_mae_truth_iter', []) if x == x], float)
        d['meandedx'] = np.asarray([x for x in h.get('mean_dedx_iter', []) if x == x], float)
        d['pos'] = np.ravel(h['pos_residual_iter']) * 1e4
        d['loss'] = np.asarray(h['losses_iter'], float)
        D[tag][s] = d

PANELS = [('Ab', PL['Ab'], '% error', None),
          ('eField', PL['eField'], '% error', None),
          ('tran_diff', PL['tran_diff'], '% error', None),
          ('long_diff', PL['long_diff'], '% error', None),
          ('lifetime', PL['lifetime'], '% error', None),
          ('mae', 'per-segment dE/dx MAE vs TRUTH', 'MAE', None),
          ('meandedx', 'length-weighted ⟨dE/dx⟩', 'MeV/cm', TRUE_MEAN),
          ('pos', 'position residual', 'µm', None),
          ('loss', 'loss (PPP negative log-likelihood)', 'NLL', None)]

fig, axes = plt.subplots(3, 3, figsize=(16.6, 11.2), sharex=True)
for k, (key, title, ylab, hline) in enumerate(PANELS):
    a = axes.ravel()[k]
    for tag, lab, col in ARMS:
        cs = [smooth(D[tag][s][key]) for s in SEEDS if len(D[tag][s][key]) > W]
        if not cs:
            continue
        for c in cs:
            a.plot(c, color=col, lw=.6, alpha=.30)
        n = min(len(c) for c in cs)
        a.plot(np.median([c[:n] for c in cs], axis=0), color=col, lw=2.3,
               label=lab if k == 0 else None)
    if hline is not None:
        a.axhline(hline, color=C['red'], lw=1.6, ls='--')
        a.annotate(f' TRUE {hline:.4f}', (0, hline), fontsize=7.4, color=C['red'], va='bottom')
    elif ylab == '% error':
        a.axhline(0, color=C['red'], lw=1.4, ls='--')
    a.set_title(title, fontsize=9.5)
    a.set_ylabel(ylab, fontsize=8.6)
    if key in ('pos', 'loss'):
        a.set_yscale('log')
    if key == 'long_diff':
        a.set_ylim(-25, 35)
    if key == 'lifetime':
        a.set_ylim(-12, 12)
    if key == 'meandedx':
        a.set_ylim(1.855, 1.895)
    if k >= 6:
        a.set_xlabel('iteration')
axes.ravel()[0].legend(fontsize=7.6, frameon=True, framealpha=.9, loc='upper right')

fig.suptitle('Fig 55 — BT convergence traces, all 6 seeds. Thin = seeds, thick = median, '
             '200-iteration rolling median.\nRead ⟨dE/dx⟩ first: BT1 (orange) slides away from the '
             'true mean all run while BT2 (green) holds it — and the A$_b$ panel tracks that '
             'slide,\nwhich is the A$_b$↔charge-normalisation coupling §6u measured. Longitudinal '
             'diffusion is the only parameter unsettled in every arm.', fontsize=10.1)
fig.tight_layout(rect=[0, 0, 1, .93])
os.makedirs(OUT, exist_ok=True)
fig.savefig(f'{OUT}/fig55_bt_traces.png'); plt.close(fig)
print('wrote', f'{OUT}/fig55_bt_traces.png')

print(f'\nn={len(SEEDS)} seeds {SEEDS}')
print('end-of-run values (median over seeds), and residual drift over the last 20%')
for key, title, _, _ in PANELS:
    row = ''
    for tag, _, _ in ARMS:
        v = [D[tag][s][key] for s in SEEDS]
        fin = np.median([np.median(x[int(len(x) * .8):]) for x in v])
        dr = np.median([np.median(x[int(len(x) * .9):]) - np.median(x[int(len(x) * .8):int(len(x) * .9)])
                        for x in v])
        row += f'  {tag} {fin:10.4f} (drift {dr:+7.4f})'
    print(f'  {title:36s}{row}')
