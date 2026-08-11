"""Fig 59 — does alternating the blocks help? A second pass with geometry FROZEN.

`BT2` fits geometry, dE/dx and calibration simultaneously for 10 000 iterations. `ALT` takes BT2's
converged geometry, FREEZES it (LARND_CHAIN_INIT_FROM + LARND_CHAIN_FREEZE_ITER=1), and re-fits
calibration and dE/dx from scratch for 3 000 iterations. Fig 56 is the motivation: the calibration
optima MOVE as geometry improves, so during a joint fit the blocks chase each other.

THE FREEZE IS VERIFIED, and the check is worth recording because the naive one FAILS. ALT's
position residual reads 1104 um against BT2's 170 um median, which looks like the geometry loaded
wrong. It did not: with geometry frozen the residual is computed ONCE, for batch 0 only, and BT2's
own batch-0 value is 1027 um -- the per-batch spread is enormous and the median over 100 batches is
not the number to compare against. Directly on the coefficients, 860 of 864 tracks match BT2
exactly and none are zero.

2 seeds. Small, and the section says so -- this is a screen for whether the effect exists, not a
measurement of its size.
"""
import glob, os, pickle, re, io, contextlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
with contextlib.redirect_stdout(io.StringIO()):
    import robust_convergence as R

OUT = 'plots/noise_report'
C = dict(blue='#0072B2', orange='#E69F00', green='#009E73', red='#D55E00', grey='#666666',
         ink='#222222')
plt.rcParams.update({'font.size': 9, 'axes.grid': True, 'grid.alpha': .25,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.dpi': 130, 'savefig.bbox': 'tight'})
SEEDS = [0, 1]
PL = {'Ab': 'A$_b$', 'eField': 'E field', 'lifetime': 'lifetime',
      'tran_diff': 'tran. diff.', 'long_diff': 'long. diff.'}
W = 200


def sm(v, w=W):
    v = np.asarray(v, float)
    if len(v) <= w:
        return v
    from numpy.lib.stride_tricks import sliding_window_view
    return np.median(sliding_window_view(v, w), axis=1)


def get(tag, s):
    f = max(glob.glob(f'fit_result/sci_full_{tag}/history_iter*seed{s}.pkl'),
            key=lambda p: int(re.search(r'iter(\d+)', p).group(1)))
    h = pickle.load(open(f, 'rb'))
    d = {}
    for p in R.P:
        t = float(np.ravel(h[p + '_target'])[0])
        d[p] = 100 * (np.ravel(h[p + '_iter'])[1:] / t - 1)
    d['mae'] = np.asarray([x for x in h.get('dedx_mae_truth_iter', []) if x == x], float)
    d['st'] = np.asarray(h['step_time'], float)
    return d


fig, ax = plt.subplots(1, 3, figsize=(16.2, 4.8))
xs = np.arange(len(R.P))

# ---------------- (a) final quality ----------------
a = ax[0]
b = [np.mean([abs(np.median(get('BT2', s)[p][8000:])) for s in SEEDS]) for p in R.P]
v = [np.mean([abs(np.median(get('ALT', s)[p][2400:])) for s in SEEDS]) for p in R.P]
a.bar(xs - .19, b, width=.36, color=C['ink'], edgecolor='white', label='BT2 — joint, 10 000 iters')
a.bar(xs + .19, v, width=.36, color=C['green'], edgecolor='white',
      label='ALT — geometry FROZEN, 3 000 iters')
for i in range(len(R.P)):
    a.text(i + .19, v[i] * 1.3, f'×{b[i]/v[i]:.1f}' if v[i] > 1e-9 else '', ha='center',
           fontsize=8.2, color=C['green'])
a.set_yscale('log')
a.set_xticks(xs); a.set_xticklabels([PL[p] for p in R.P], fontsize=8, rotation=18, ha='right')
a.set_ylabel('|error| vs truth (%)')
a.set_title('(a) a frozen-geometry second pass MATCHES or beats the\n'
            'joint fit — A$_b$ 2.3×, tran. diff. 2.2× — in 30% of the iterations', fontsize=9.3)
a.legend(fontsize=7.4, frameon=False)

# ---------------- (b) per seed, the two that move ----------------
a = ax[1]
xs2 = np.arange(4)
lab, bb, vv = [], [], []
for p in ['Ab', 'tran_diff']:
    for s in SEEDS:
        lab.append(f'{PL[p]}\nseed {s}')
        bb.append(abs(np.median(get('BT2', s)[p][8000:])))
        vv.append(abs(np.median(get('ALT', s)[p][2400:])))
a.bar(xs2 - .19, bb, width=.36, color=C['ink'], edgecolor='white', label='BT2')
a.bar(xs2 + .19, vv, width=.36, color=C['green'], edgecolor='white', label='ALT')
for i in range(4):
    a.text(i - .19, bb[i] + .05, f'{bb[i]:.2f}', ha='center', fontsize=7.4)
    a.text(i + .19, vv[i] + .05, f'{vv[i]:.2f}', ha='center', fontsize=7.4)
a.set_xticks(xs2); a.set_xticklabels(lab, fontsize=7.6)
a.set_ylabel('|error| (%)')
a.set_title('(b) A$_b$ improves on BOTH seeds; transverse diffusion on\n'
            'one strongly and one mildly worse', fontsize=9.3)
a.legend(fontsize=7.6, frameon=False)

# ---------------- (c) A_b traces ----------------
a = ax[2]
for tag, col, lab_, off in [('BT2', C['ink'], 'BT2 — joint', 0), ('ALT', C['green'], 'ALT — frozen geometry', 0)]:
    cs = [sm(get(tag, s)['Ab']) for s in SEEDS]
    n = min(len(c) for c in cs)
    a.plot(np.median([c[:n] for c in cs], axis=0), color=col, lw=2.3, label=lab_)
a.axhline(0, color=C['red'], lw=1.4, ls='--')
a.set_xlabel('iteration'); a.set_ylabel('A$_b$ error (%)')
a.set_ylim(-1.2, 1.6)
a.set_title('(c) and it gets there FASTER: ALT is at its final A$_b$\n'
            'by ~1 500 iterations', fontsize=9.3)
a.legend(fontsize=7.6, frameon=False)

st_b = np.mean([np.median(get('BT2', s)['st']) for s in SEEDS])
st_a = np.mean([np.median(get('ALT', s)['st']) for s in SEEDS])
fig.suptitle('Fig 59 — alternation works, on the parameters fig 58 says are DECOUPLED from geometry. '
             'A second pass with geometry frozen reaches\nBT2 quality in 3 000 iterations and beats '
             f'it on A$_b$ (2.3×) and transverse diffusion (2.2×), at {st_a*1e3:.0f} ms/iter against '
             f'{st_b*1e3:.0f}.\nlong. diffusion and lifetime are unchanged — exactly the parameters '
             'that need geometry to MOVE, which a frozen pass cannot give them.', fontsize=10.0)
fig.tight_layout(rect=[0, 0, 1, .84])
os.makedirs(OUT, exist_ok=True)
fig.savefig(f'{OUT}/fig59_alternation.png'); plt.close(fig)
print('wrote', f'{OUT}/fig59_alternation.png')
for i, p in enumerate(R.P):
    print(f'  {p:11s} BT2 {b[i]:7.3f}   ALT {v[i]:7.3f}   ratio x{b[i]/v[i]:.2f}')
print(f'  step time: BT2 {st_b*1e3:.0f} ms, ALT {st_a*1e3:.0f} ms ({100*(1-st_a/st_b):.0f}% cheaper)')
