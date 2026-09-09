"""Fig 75 — Test 1 illustrated: on the BIASED replicas the objective still prefers truth, the
gradient at the fit is nonzero, and the lifetime direction is revealed as nearly FREE in loss.

(a) The joint fit->truth line on each biased replica (and the fixed sample for context): monotone
    downhill from the very first step, minimum AT truth, climbing past it. Nothing traps the fit.
(b) The panel that names the mechanism: the truth-fit loss gap is the SAME ~9-10% whether the
    lifetime bias is +6.5% or +16.4%. Moving lifetime by ten points costs almost nothing in loss
    -- the soft direction of the -0.83 degeneracy, made visible. The optimiser lands wherever the
    hot phase carried it along this near-free direction, and the anneal freezes it there.
(c) The one-block moves reproduce the fixed-sample pattern on every replica: geometry alone helps,
    calibration or dEdx alone HURT -- blocks must move together, on typical samples too.
"""
import glob, os, pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = 'plots/noise_report'
C = dict(blue='#0072B2', orange='#E69F00', green='#009E73', red='#D55E00', grey='#888888',
         ink='#222222')
plt.rcParams.update({'font.size': 10, 'axes.grid': True, 'grid.alpha': .25,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.dpi': 130, 'savefig.bbox': 'tight'})
ALPHA = [0, 0.25, 0.5, 0.75, 1, 1.1]
BIAS = {1: 6.53, 7: 14.61, 8: 16.40}
COL = {1: '#6BAED6', 7: '#EF6548', 8: '#99000D'}


def pts(path, alphas):
    Rs = pickle.load(open(path, 'rb'))['truthpoint']
    out = {}
    for k in (['P0_fit', 'P1_truth', 'P2_calib_true', 'P3_dedx_true', 'P4_geom_true']
              + [f'F{a:g}' for a in alphas]):
        v = [r['points'][k]['loss'] for r in Rs if k in r['points']]
        out[k] = sum(v) if len(v) == len(Rs) else None
    return out


REP = {ds: pts(glob.glob(f'fit_result/sci_full_TPREP{ds}/history_iter*.pkl')[0], ALPHA)
       for ds in (1, 7, 8)}
AL_FP = [0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.05, 1.1, 1.2]
FIX = pts(glob.glob('fit_result/sci_full_FPBT2/history_iter*seed0.pkl')[0], AL_FP)

fig, ax = plt.subplots(1, 3, figsize=(16.6, 5.6))

# ---------------- (a) the paths ----------------
a = ax[0]
for ds in (1, 7, 8):
    c = np.array([REP[ds][f'F{x:g}'] for x in ALPHA])
    a.plot(ALPHA, 100 * (c / c[0] - 1), '-o', color=COL[ds], lw=2.4, ms=7,
           label=f'replica {ds}  (lifetime bias +{BIAS[ds]:.1f}%)')
cf = np.array([FIX[f'F{x:g}'] for x in AL_FP])
a.plot(AL_FP, 100 * (cf / cf[0] - 1), '--', color=C['grey'], lw=1.8,
       label='the fixed sample (fig 69), context')
a.axvline(1, color=C['green'], lw=2.2)
a.annotate(' TRUTH', (1.005, .5), fontsize=9, color=C['green'], fontweight='bold')
a.axhline(0, color=C['ink'], lw=1.2)
a.annotate('the fit', (0.01, .35), fontsize=8.5, color=C['ink'])
a.annotate('already downhill\nat the first step →', (.06, -2.1), fontsize=8.5, color=C['ink'])
a.set_xlabel('α along the joint fit → truth line (all three blocks together)')
a.set_ylabel('loss, % relative to the fit')
a.set_title('(a) NO TRAP, even on the worst replicas: monotone downhill\n'
            'from the fit, minimum AT truth, climbing past it.', fontsize=10)
a.legend(fontsize=8, frameon=False, loc='lower left')

# ---------------- (b) the mechanism panel ----------------
a = ax[1]
for ds in (1, 7, 8):
    gap = -100 * (REP[ds]['P1_truth'] / REP[ds]['P0_fit'] - 1)
    a.plot(BIAS[ds], gap, 'o', ms=15, color=COL[ds], zorder=4)
    a.annotate(f'  rep {ds}', (BIAS[ds], gap), fontsize=9)
a.plot(3.04, -100 * (FIX['P1_truth'] / FIX['P0_fit'] - 1), '*', ms=22, color=C['grey'], zorder=4)
a.annotate('  fixed sample\n  (|bias| 3.0%)', (3.04, 9.4), fontsize=8.5, color=C['grey'])
a.set_xlim(0, 19); a.set_ylim(0, 12)
a.axhspan(8.5, 10.5, color=C['green'], alpha=.10)
a.annotate('the gap to truth is the SAME ~9–10% everywhere', (9.5, 11.1), fontsize=9.5,
           ha='center', color=C['ink'], fontweight='bold')
a.annotate('→ moving lifetime by TEN POINTS costs almost no loss.\n'
           '   The lifetime direction is nearly FREE — the soft axis of\n'
           '   the −0.83 degeneracy. The optimiser lands wherever the\n'
           '   hot phase carried it, and the anneal freezes it there.',
           (.7, 2.6), fontsize=9.2, color=C['ink'])
a.set_xlabel('|lifetime bias| of the fit (%)')
a.set_ylabel('loss still available between fit and truth (%)')
a.set_title('(b) THE MECHANISM: the loss gap does NOT grow with the\n'
            'lifetime bias — lifetime is a near-free direction.', fontsize=10)

# ---------------- (c) blocks must move together ----------------
a = ax[2]
moves = [('P4_geom_true', 'geometry\nalone'), ('P2_calib_true', 'calibration\nalone'),
         ('P3_dedx_true', 'dE/dx\nalone'), ('P1_truth', 'ALL THREE\ntogether')]
w = .26
for i, ds in enumerate((1, 7, 8)):
    v = [100 * (REP[ds][k] / REP[ds]['P0_fit'] - 1) for k, _ in moves]
    a.bar(np.arange(4) + (i - 1) * w, v, width=w, color=COL[ds], edgecolor='white',
          label=f'rep {ds}')
a.axhline(0, color=C['ink'], lw=1.6)
a.set_xticks(range(4)); a.set_xticklabels([l for _, l in moves], fontsize=9)
a.set_ylabel('Δ loss vs the fit (%)   negative = better')
a.set_title('(c) same anatomy as the fixed sample: single blocks moved\n'
            'to truth alone HURT; only together (or geometry) do they help.', fontsize=10)
a.legend(fontsize=8.4, frameon=False, loc='lower right')

fig.suptitle('Fig 75 — Test 1: the truth-point evaluation ON the biased replicas (all previous '
             '“objective prefers truth” evidence came from the lucky fixed sample).\nThe objective '
             'is sound on typical data too — truth wins by 9–10% with a nonzero gradient at every '
             'fit — and panel (b) names the mechanism: the lifetime\ndirection is nearly free in '
             'loss, so the anneal freezes each fit wherever its early dynamics landed along it.',
             fontsize=10.6)
fig.tight_layout(rect=[0, 0, 1, .86])
os.makedirs(OUT, exist_ok=True)
fig.savefig(f'{OUT}/fig75_tprep.png'); plt.close(fig)
print('wrote', f'{OUT}/fig75_tprep.png')
